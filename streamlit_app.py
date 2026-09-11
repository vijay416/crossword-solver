import json
import os
import re
from collections import Counter

import streamlit as st

# Optional word frequency package
try:
    from wordfreq import zipf_frequency

    WORDFREQ_AVAILABLE = True
except ImportError:
    WORDFREQ_AVAILABLE = False

# Optional OpenAI
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

WORD_FILE = "words.txt"
DICTIONARY_FILE = "dictionary.json"

MAX_WORD_LENGTH = 9
MAX_RESULTS = 20


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Crossword Solver",
    page_icon="🧩",
    layout="wide",
)


# ============================================================
# LOAD WORD LIST
# ============================================================

@st.cache_data
def load_words():
    """
    Load the authoritative crossword word list.

    The file may contain uppercase or lowercase words.
    Internally everything is normalized to uppercase.
    """

    if not os.path.exists(WORD_FILE):
        return []

    words = []
    seen = set()

    with open(WORD_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            word = line.strip()

            if not word:
                continue

            # Normalize to uppercase
            word = word.upper()

            # Current crossword list is assumed to contain
            # alphabetic entries.
            #
            # If your list later contains hyphens/apostrophes,
            # this can be expanded.
            if not word.isalpha():
                continue

            if len(word) < 3 or len(word) > MAX_WORD_LENGTH:
                continue

            if word not in seen:
                seen.add(word)
                words.append(word)

    return words


# ============================================================
# LOAD LOCAL DICTIONARY
# ============================================================

@st.cache_resource
def load_dictionary():
    """
    Load dictionary.json into memory.

    Expected format:

    {
        "word": "definition",
        "anotherword": "definition"
    }

    Keys are normalized to lowercase so dictionary lookups
    work regardless of capitalization in words.txt.
    """

    if not os.path.exists(DICTIONARY_FILE):
        return {}

    try:
        with open(
            DICTIONARY_FILE,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            raw_dictionary = json.load(f)

        dictionary = {}

        for key, value in raw_dictionary.items():

            normalized_key = str(key).strip().lower()

            if isinstance(value, str):
                definition = value.strip()

            elif isinstance(value, dict):
                # Support a slightly richer dictionary format
                definitions = value.get("definitions")

                if isinstance(definitions, list):
                    definition = " ".join(
                        str(x).strip()
                        for x in definitions
                        if str(x).strip()
                    )
                else:
                    definition = str(
                        value.get("definition", "")
                    ).strip()

            elif isinstance(value, list):
                definition = " ".join(
                    str(x).strip()
                    for x in value
                    if str(x).strip()
                )

            else:
                definition = str(value).strip()

            if normalized_key and definition:
                dictionary[normalized_key] = definition

        return dictionary

    except Exception as e:
        st.error(f"Could not load dictionary.json: {e}")
        return {}


WORDS = load_words()
DICTIONARY = load_dictionary()


# ============================================================
# WORD FREQUENCY / RANKING
# ============================================================

COMMON_LETTERS = set("ETAOINSHRDLU")
COMMON_SHORT_WORDS = {
    "AND",
    "THE",
    "FOR",
    "ARE",
    "BUT",
    "NOT",
    "YOU",
    "ALL",
    "CAN",
    "HER",
    "WAS",
    "ONE",
    "OUT",
    "HAS",
    "HAD",
    "HIS",
    "WHO",
    "HOW",
    "NEW",
    "NOW",
    "OLD",
    "ANY",
    "TWO",
    "WAY",
    "DAY",
    "MAN",
    "MEN",
    "WAS",
    "DID",
    "GET",
    "LET",
    "SAY",
    "SHE",
    "OUR",
    "USE",
    "OWN",
    "SEE",
    "TOO",
    "WHY",
    "ITS",
}


def word_frequency_score(word):
    """
    Return a word-frequency score.

    If wordfreq is installed, Zipf frequency is used.

    Otherwise a lightweight heuristic is used.
    """

    lower_word = word.lower()

    if WORDFREQ_AVAILABLE:
        return zipf_frequency(lower_word, "en")

    # Fallback heuristic
    score = 0.0

    # Dictionary presence is a strong signal
    if lower_word in DICTIONARY:
        score += 5.0

    # Common letters
    score += sum(
        0.15 for char in word if char in COMMON_LETTERS
    )

    # Prefer reasonable vowel balance
    vowels = sum(
        1 for char in word if char in "AEIOU"
    )

    if len(word) >= 4:
        vowel_ratio = vowels / len(word)

        if 0.20 <= vowel_ratio <= 0.60:
            score += 1.0

    # Common short words
    if word in COMMON_SHORT_WORDS:
        score += 2.0

    return score


def crossword_score(word):
    """
    Score a word for crossword usefulness.

    This intentionally does NOT aggressively remove uncommon words.

    Rare words can still be perfectly valid crossword answers.
    Instead, common / dictionary-backed words are ranked higher.
    """

    score = word_frequency_score(word)

    lower_word = word.lower()

    # --------------------------------------------------------
    # Dictionary availability
    # --------------------------------------------------------

    if lower_word in DICTIONARY:
        score += 2.0

    # --------------------------------------------------------
    # Length preference
    # --------------------------------------------------------

    length = len(word)

    # Slight preference for common crossword lengths
    length_bonus = {
        3: 0.20,
        4: 0.45,
        5: 0.65,
        6: 0.55,
        7: 0.45,
        8: 0.30,
        9: 0.20,
    }

    score += length_bonus.get(length, 0)

    # --------------------------------------------------------
    # Vowel balance
    # --------------------------------------------------------

    vowels = sum(
        1 for char in word if char in "AEIOU"
    )

    if length >= 4:

        vowel_ratio = vowels / length

        if 0.20 <= vowel_ratio <= 0.60:
            score += 0.60

        elif vowel_ratio < 0.10:
            score -= 0.80

    # --------------------------------------------------------
    # Common letters
    # --------------------------------------------------------

    common_letter_count = sum(
        1 for char in word if char in COMMON_LETTERS
    )

    score += common_letter_count * 0.08

    # --------------------------------------------------------
    # Excessive consonant clusters
    # --------------------------------------------------------

    if re.search(r"[BCDFGHJKLMNPQRSTVWXYZ]{4,}", word):
        score -= 1.0

    if re.search(r"[BCDFGHJKLMNPQRSTVWXYZ]{5,}", word):
        score -= 1.5

    # --------------------------------------------------------
    # Repeated letters
    # --------------------------------------------------------

    repeated_count = len(word) - len(set(word))

    if repeated_count >= 3:
        score -= 0.30

    # --------------------------------------------------------
    # Obvious inflection penalty
    #
    # Only a slight penalty. We don't want to remove
    # legitimate crossword answers.
    # --------------------------------------------------------

    if len(word) >= 6:

        if word.endswith("ING"):
            score -= 0.10

        elif word.endswith("ED"):
            score -= 0.08

        elif word.endswith("ES"):
            score -= 0.06

    return score


def rank_matches(matches):
    """
    Rank candidate words from most likely to least likely.
    """

    scored = []

    for word in matches:
        score = crossword_score(word)

        scored.append(
            (
                score,
                word,
            )
        )

    scored.sort(
        key=lambda x: (-x[0], x[1])
    )

    return [word for score, word in scored]


# ============================================================
# PATTERN MATCHING
# ============================================================

def pattern_to_regex(pattern):
    """
    Convert crossword pattern to regex.

    Supported:
        ? = unknown letter
        _ = unknown letter
        A-Z = fixed letter
        - = literal hyphen
        ' = literal apostrophe

    Examples:

        C?? = CAT, CAR, CAN, etc.

        ?A? = CAT, MAN, MAP, etc.

        C_T = CAT, COT, CUT, etc.
    """

    pattern = pattern.strip()

    regex_parts = []

    for ch in pattern:

        if ch in ["?", "_"]:
            regex_parts.append(".")

        elif ch.isalnum():
            regex_parts.append(re.escape(ch.upper()))

        elif ch in ["-", "'"]:
            regex_parts.append(re.escape(ch))

        else:
            regex_parts.append(re.escape(ch))

    return "^" + "".join(regex_parts) + "$"


def find_pattern_matches(pattern):
    """
    Find all words matching a crossword pattern.
    """

    if not pattern:
        return []

    regex_string = pattern_to_regex(pattern)

    try:
        regex = re.compile(regex_string)

    except re.error:
        return []

    matches = []

    for word in WORDS:

        if regex.fullmatch(word):
            matches.append(word)

    return rank_matches(matches)


# ============================================================
# LOCAL MEANING
# ============================================================

def get_meaning(word):
    """
    Look up meaning from local dictionary.json.
    """

    return DICTIONARY.get(
        word.strip().lower()
    )


def shorten_definition(definition, max_chars=280):
    """
    Make long dictionary definitions easier to display.
    """

    if not definition:
        return "Definition unavailable."

    definition = re.sub(
        r"\s+",
        " ",
        definition,
    ).strip()

    if len(definition) <= max_chars:
        return definition

    # Try to stop at a sentence
    sentences = re.split(
        r"(?<=[.!?])\s+",
        definition,
    )

    if sentences:

        first = sentences[0].strip()

        if len(first) <= max_chars:
            return first

    shortened = definition[:max_chars]

    if " " in shortened:
        shortened = shortened.rsplit(
            " ",
            1,
        )[0]

    return shortened + "…"


# ============================================================
# SESSION STATE
# ============================================================

if "pattern_matches" not in st.session_state:
    st.session_state.pattern_matches = []

if "last_pattern" not in st.session_state:
    st.session_state.last_pattern = ""

if "selected_meaning_word" not in st.session_state:
    st.session_state.selected_meaning_word = None


# ============================================================
# OPENAI
# ============================================================

def get_openai_client():

    if not OPENAI_AVAILABLE:
        return None

    api_key = None

    try:
        api_key = st.secrets.get(
            "OPENAI_API_KEY"
        )
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

    if not api_key:
        return None

    return OpenAI(
        api_key=api_key
    )


def get_openai_model():

    try:
        model = st.secrets.get(
            "OPENAI_MODEL"
        )

        if model:
            return model

    except Exception:
        pass

    return os.getenv(
        "OPENAI_MODEL",
        "gpt-4o-mini",
    )


# ============================================================
# AI CLUE SOLVER
# ============================================================

def ai_solve_clue(
    clue,
    pattern=None,
    candidates=None,
):
    """
    Ask AI to rank or identify crossword answers.

    IMPORTANT:
    Any returned answer is subsequently validated against
    the local words.txt list.
    """

    client = get_openai_client()

    if client is None:
        return None, (
            "OpenAI API is not configured. "
            "Pattern Search and local meanings "
            "are still available."
        )

    if not clue.strip():
        return None, "Please enter a clue."

    # --------------------------------------------------------
    # Candidate handling
    # --------------------------------------------------------

    if candidates:
        candidate_text = ", ".join(
            candidates[:100]
        )

        candidate_instruction = f"""
You may ONLY choose an answer from this candidate list:

{candidate_text}

Do not invent a word outside this list.
"""

    else:
        candidate_instruction = """
Suggest the most likely crossword answer.
The final answer must exist in the supplied word list.
"""

    pattern_instruction = ""

    if pattern:
        pattern_instruction = f"""
The crossword pattern is:

{pattern}

The answer must match this pattern exactly.
"""

    prompt = f"""
You are a crossword solving assistant.

Clue:
{clue}

{pattern_instruction}

{candidate_instruction}

Return ONLY the single best answer in uppercase.
Do not provide explanations.
"""

    try:

        response = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert crossword solver. "
                        "Always obey the supplied candidate list "
                        "and pattern."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=20,
        )

        answer = (
            response.choices[0]
            .message
            .content
            .strip()
            .upper()
        )

        # Remove accidental punctuation
        answer = re.sub(
            r"[^A-Z'-]",
            "",
            answer,
        )

        # ----------------------------------------------------
        # Validate answer against word list
        # ----------------------------------------------------

        if answer not in set(WORDS):

            return None, (
                "AI suggested an answer that is not "
                "present in words.txt."
            )

        # Validate pattern too
        if pattern:

            regex = pattern_to_regex(
                pattern
            )

            if not re.fullmatch(
                regex,
                answer,
            ):
                return None, (
                    "AI answer did not match "
                    "the supplied pattern."
                )

        return answer, None

    except Exception as e:

        error_text = str(e)

        # Handle common OpenAI quota issue
        if (
            "insufficient_quota" in error_text
            or "credit_balance_exhausted" in error_text
            or "429" in error_text
        ):
            return None, (
                "AI is currently unavailable because "
                "the OpenAI API quota has been exhausted. "
                "Pattern Search and local meanings "
                "are still available."
            )

        return None, (
            f"AI error: {error_text}"
        )


# ============================================================
# HEADER
# ============================================================

st.title("🧩 Crossword Solver")

st.caption(
    "Pattern matching + local dictionary meanings + optional AI clue solving"
)


# ============================================================
# STATUS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Words",
        f"{len(WORDS):,}",
    )

with col2:
    st.metric(
        "Meanings",
        f"{len(DICTIONARY):,}",
    )

with col3:

    if WORDFREQ_AVAILABLE:
        st.metric(
            "Ranking",
            "Word frequency",
        )
    else:
        st.metric(
            "Ranking",
            "Crossword heuristic",
        )


# ============================================================
# WARNINGS
# ============================================================

if not WORDS:

    st.error(
        f"Could not find `{WORD_FILE}`. "
        "Place it in the same folder as streamlit_app.py."
    )

if not DICTIONARY:

    st.warning(
        f"`{DICTIONARY_FILE}` was not loaded. "
        "Pattern solving will work, but local meanings "
        "will not be available."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "🔎 Pattern Search",
        "🤖 Clue Solver",
    ]
)


# ============================================================
# PATTERN SEARCH TAB
# ============================================================

with tab1:

    st.subheader(
        "Find words by pattern"
    )

    st.markdown(
        """
**Pattern rules**

- `?` = one unknown letter
- `_` = one unknown letter
- Letters = fixed letters
- `-` = literal hyphen
- `'` = literal apostrophe

Examples:

`C??` → 3-letter words beginning with C

`?A?` → 3-letter words with A in the middle

`C??E` → 4-letter words beginning with C and ending with E
"""
    )

    pattern = st.text_input(
        "Enter crossword pattern",
        placeholder="Example: C??E",
        key="pattern_input",
    )

    search_col, clear_col = st.columns(
        [1, 1]
    )

    with search_col:

        search_clicked = st.button(
            "🔎 Search",
            type="primary",
            use_container_width=True,
        )

    with clear_col:

        clear_clicked = st.button(
            "Clear",
            use_container_width=True,
        )

    if clear_clicked:

        st.session_state.pattern_matches = []
        st.session_state.last_pattern = ""
        st.session_state.selected_meaning_word = None

        st.rerun()

    if search_clicked:

        if not pattern.strip():

            st.warning(
                "Please enter a pattern."
            )

        else:

            matches = find_pattern_matches(
                pattern
            )

            st.session_state.pattern_matches = matches
            st.session_state.last_pattern = pattern.upper()
            st.session_state.selected_meaning_word = None

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    matches = st.session_state.pattern_matches

    if matches:

        st.success(
            f"Found {len(matches):,} matching words. "
            f"Showing the top {min(MAX_RESULTS, len(matches))}."
        )

        # Show selected meaning above results
        selected_word = (
            st.session_state.selected_meaning_word
        )

        if selected_word:

            definition = get_meaning(
                selected_word
            )

            st.markdown("---")

            st.subheader(
                f"📖 {selected_word}"
            )

            if definition:

                st.write(
                    shorten_definition(
                        definition
                    )
                )

            else:

                st.info(
                    "No local definition found "
                    "for this word."
                )

        st.markdown("---")

        # ----------------------------------------------------
        # Result table
        # ----------------------------------------------------

        top_matches = matches[:MAX_RESULTS]

        for index, word in enumerate(
            top_matches,
            start=1,
        ):

            meaning_available = (
                word.lower()
                in DICTIONARY
            )

            col_num, col_word, col_meaning = st.columns(
                [0.7, 2.5, 1.3]
            )

            with col_num:

                st.write(
                    f"**{index}**"
                )

            with col_word:

                # Display uppercase because words.txt
                # is treated as uppercase
                st.write(
                    f"### {word}"
                )

                if meaning_available:
                    st.caption(
                        "📖 Local definition available"
                    )
                else:
                    st.caption(
                        "No local definition"
                    )

            with col_meaning:

                if meaning_available:

                    if st.button(
                        "Meaning",
                        key=(
                            f"meaning_"
                            f"{index}_"
                            f"{word}"
                        ),
                        use_container_width=True,
                    ):

                        st.session_state.selected_meaning_word = word

                        st.rerun()

                else:

                    st.button(
                        "No meaning",
                        key=(
                            f"no_meaning_"
                            f"{index}_"
                            f"{word}"
                        ),
                        disabled=True,
                        use_container_width=True,
                    )

            st.divider()

    elif (
        st.session_state.last_pattern
    ):

        st.info(
            "No words match this pattern."
        )


# ============================================================
# CLUE SOLVER TAB
# ============================================================

with tab2:

    st.subheader(
        "🤖 Crossword Clue Solver"
    )

    st.write(
        "Enter a clue and optionally provide a pattern. "
        "The AI answer is validated against your local "
        "`words.txt`."
    )

    clue = st.text_area(
        "Clue",
        placeholder=(
            "Example: "
            "A place where books are kept"
        ),
        height=100,
    )

    clue_pattern = st.text_input(
        "Pattern (optional)",
        placeholder="Example: L?BR?RY",
    )

    solve_button = st.button(
        "🤖 Solve Clue",
        type="primary",
        use_container_width=True,
    )

    if solve_button:

        if not clue.strip():

            st.warning(
                "Please enter a clue."
            )

        else:

            # ------------------------------------------------
            # Generate candidates from pattern
            # ------------------------------------------------

            candidates = []

            if clue_pattern.strip():

                candidates = find_pattern_matches(
                    clue_pattern
                )

                if not candidates:

                    st.warning(
                        "No words in words.txt "
                        "match that pattern."
                    )

                else:

                    st.info(
                        f"{len(candidates):,} "
                        "pattern candidates found. "
                        "AI will rank them."
                    )

            # ------------------------------------------------
            # AI solving
            # ------------------------------------------------

            with st.spinner(
                "Solving clue..."
            ):

                answer, error = ai_solve_clue(
                    clue=clue,
                    pattern=clue_pattern,
                    candidates=candidates,
                )

            if error:

                st.warning(error)

            elif answer:

                st.success(
                    f"Best answer: **{answer}**"
                )

                definition = get_meaning(
                    answer
                )

                if definition:

                    st.markdown(
                        "**Meaning:**"
                    )

                    st.write(
                        shorten_definition(
                            definition
                        )
                    )

                else:

                    st.info(
                        "The answer exists in "
                        "words.txt, but no local "
                        "definition was found."
                    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Dictionary & Word List"
    )

    st.write(
        f"**Word list:** `{WORD_FILE}`"
    )

    st.write(
        f"**Dictionary:** `{DICTIONARY_FILE}`"
    )

    st.write(
        f"**Words loaded:** {len(WORDS):,}"
    )

    st.write(
        f"**Definitions loaded:** "
        f"{len(DICTIONARY):,}"
    )

    st.divider()

    st.subheader(
        "Ranking"
    )

    if WORDFREQ_AVAILABLE:

        st.success(
            "wordfreq is enabled"
        )

        st.caption(
            "Candidate ranking uses English "
            "word-frequency information plus "
            "crossword-specific signals."
        )

    else:

        st.warning(
            "wordfreq is not installed"
        )

        st.caption(
            "Using the built-in crossword "
            "ranking heuristic."
        )

    st.divider()

    st.subheader(
        "AI"
    )

    if get_openai_client():

        st.success(
            "OpenAI API configured"
        )

        st.caption(
            f"Model: `{get_openai_model()}`"
        )

    else:

        st.info(
            "OpenAI API not configured"
        )

        st.caption(
            "Pattern search and local "
            "dictionary still work."
        )

    st.divider()

    st.caption(
        "Local dictionary lookup does not "
        "require an internet connection."
    )
