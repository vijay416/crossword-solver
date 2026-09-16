import streamlit as st
import json
import re
import math
from collections import Counter

# wordfreq is optional but recommended
try:
    from wordfreq import zipf_frequency
    WORDFREQ_AVAILABLE = True
except ImportError:
    WORDFREQ_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

WORDS_FILE = "words.txt"
DICTIONARY_FILE = "dictionary.json"

MAX_PATTERN_RESULTS = 20
MAX_CLUE_RESULTS = 20


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Crossword Solver",
    page_icon="🧩",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .crossword-word {
        font-size: 17px;
        font-weight: 600;
        line-height: 1.25;
    }

    .crossword-rank {
        font-size: 12px;
        color: #777;
    }

    .crossword-definition {
        font-size: 13px;
        line-height: 1.3;
        padding-top: 3px;
    }

    .crossword-pos {
        font-size: 11px;
        color: #777;
        font-style: italic;
    }

    .clue-result-word {
        font-size: 17px;
        font-weight: 600;
    }

    .clue-result-definition {
        font-size: 13px;
        line-height: 1.35;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WORD LIST LOADER
# ============================================================

@st.cache_data
def load_words():
    """
    Load the COMPLETE words.txt.

    Supports:
      ABC
      AARD-VARK
      AARON'SROD

    The original capitalization is normalized internally.
    Display remains uppercase.
    """

    words = []
    seen = set()

    try:
        with open(WORDS_FILE, "r", encoding="utf-8", errors="ignore") as f:

            for line in f:
                word = line.strip()

                if not word:
                    continue

                # Normalize to uppercase
                word = word.upper()

                # Valid crossword characters:
                # A-Z, apostrophe and hyphen.
                #
                # This intentionally DOES NOT use .isalpha()
                # because entries such as AARD-VARK and
                # AARON'SROD are valid in the supplied list.
                if not re.fullmatch(r"[A-Z]+(?:[-'][A-Z]+)*", word):
                    continue

                if word not in seen:
                    seen.add(word)
                    words.append(word)

    except FileNotFoundError:
        st.error(
            f"Could not find {WORDS_FILE}. "
            "Make sure it is in the same repository as streamlit_app.py."
        )
        return []

    return words


# ============================================================
# DICTIONARY LOADER
# ============================================================

@st.cache_data
def load_dictionary():
    """
    Load local dictionary.json.

    Supports either:

    "word": "definition"

    OR

    "word": {
        "definition": "...",
        "part_of_speech": "noun"
    }
    """

    try:
        with open(DICTIONARY_FILE, "r", encoding="utf-8", errors="ignore") as f:
            raw = json.load(f)

    except FileNotFoundError:
        st.error(
            f"Could not find {DICTIONARY_FILE}. "
            "Make sure it is in the same repository as streamlit_app.py."
        )
        return {}

    except json.JSONDecodeError as e:
        st.error(f"Could not read {DICTIONARY_FILE}: {e}")
        return {}

    dictionary = {}

    for key, value in raw.items():

        word = str(key).strip().lower()

        if not word:
            continue

        if isinstance(value, str):

            dictionary[word] = {
                "definition": value,
                "part_of_speech": ""
            }

        elif isinstance(value, dict):

            definition = value.get("definition", "")

            # Also support definitions as a list
            if isinstance(definition, list):
                definition = " ".join(
                    str(x) for x in definition if x
                )

            dictionary[word] = {
                "definition": str(definition),
                "part_of_speech": str(
                    value.get("part_of_speech", "")
                )
            }

        else:
            dictionary[word] = {
                "definition": str(value),
                "part_of_speech": ""
            }

    return dictionary


# ============================================================
# WORD NORMALIZATION
# ============================================================

def normalized_word(word):
    """
    Normalize a crossword word for matching/scoring.

    AARD-VARK -> aard-vark
    AARON'SROD -> aaron'srod
    """

    return word.lower()


def letters_only(word):
    """
    Remove punctuation for frequency scoring.

    AARD-VARK -> aardvark
    AARON'SROD -> aaronsrod
    """

    return re.sub(r"[^a-z]", "", word.lower())


# ============================================================
# PATTERN -> REGEX
# ============================================================

def pattern_to_regex(pattern):
    """
    Pattern rules:
    ? or _ = exactly one unknown character
    *      = zero or more unknown characters
    A-Z    = fixed letter
    -      = literal hyphen
    '      = literal apostrophe
    """

    regex_parts = []

    for ch in pattern.upper():
        if ch in ["?", "_"]:
            regex_parts.append(".")
        elif ch == "*":
            regex_parts.append(".*")
        elif ch.isalnum():
            regex_parts.append(re.escape(ch))
        elif ch in ["-", "'"]:
            regex_parts.append(re.escape(ch))
        else:
            regex_parts.append(re.escape(ch))

    return "^" + "".join(regex_parts) + "$"

# ============================================================
# WORD FREQUENCY
# ============================================================

def frequency_score(word):
    """
    Return a commonness score.

    wordfreq:
      Zipf scale is approximately 1-8.

    For crossword ranking, common words get a higher score,
    but obscure words are NOT removed.
    """

    clean = letters_only(word)

    if not clean:
        return 0.0

    if WORDFREQ_AVAILABLE:

        try:
            return float(
                zipf_frequency(clean, "en")
            )
        except Exception:
            pass

    # --------------------------------------------------------
    # Fallback heuristic
    # --------------------------------------------------------

    common_words = {
        "THE", "AND", "FOR", "ARE", "BUT", "NOT",
        "YOU", "ALL", "CAN", "HER", "WAS", "ONE",
        "OUR", "OUT", "DAY", "GET", "HAS", "HAD",
        "HIS", "HOW", "MAN", "NEW", "NOW", "OLD",
        "SEE", "TWO", "WAY", "WHO", "BOY", "DID",
        "ITS", "LET", "PUT", "SAY", "SHE", "TOO",
        "USE", "YES"
    }

    if word.upper() in common_words:
        return 7.0

    length = len(clean)

    if length <= 3:
        return 5.5

    if length <= 5:
        return 5.0

    if length <= 7:
        return 4.5

    if length <= 9:
        return 4.0

    return 3.5


# ============================================================
# CROSSWORD RANKING
# ============================================================

COMMON_LETTERS = set(
    "ETAOINSHRDLU"
)


def crossword_score(word, dictionary):
    """
    Rank a word for crossword usefulness.

    Important:
      This function ranks words.
      It does NOT eliminate words.
    """

    score = 0.0

    # --------------------------------------------------------
    # Commonness
    # --------------------------------------------------------

    freq = frequency_score(word)

    score += freq * 10.0

    # --------------------------------------------------------
    # Dictionary presence
    # --------------------------------------------------------

    if normalized_word(word) in dictionary:
        score += 18.0

    # --------------------------------------------------------
    # Short-word bonus
    #
    # Short words are particularly useful in crosswords.
    # --------------------------------------------------------

    clean = letters_only(word)
    length = len(clean)

    if length == 2:
        score += 16

    elif length == 3:
        score += 12

    elif length == 4:
        score += 8

    elif length == 5:
        score += 5

    elif length == 6:
        score += 3

    # --------------------------------------------------------
    # Common-letter bonus
    # --------------------------------------------------------

    if clean:

        common_count = sum(
            1 for c in clean.upper()
            if c in COMMON_LETTERS
        )

        score += common_count * 0.5

    # --------------------------------------------------------
    # Vowel balance
    # --------------------------------------------------------

    if clean:

        vowels = sum(
            1 for c in clean
            if c in "aeiou"
        )

        ratio = vowels / len(clean)

        # Reward reasonable vowel balance
        if 0.20 <= ratio <= 0.55:
            score += 2.5

        elif ratio < 0.10:
            score -= 2.0

    # --------------------------------------------------------
    # Punctuation penalty
    #
    # Keep these words, but slightly lower them.
    # --------------------------------------------------------

    if "-" in word:
        score -= 3.0

    if "'" in word:
        score -= 2.0

    return score


# ============================================================
# PATTERN SEARCH
# ============================================================

def find_pattern_matches(pattern, words, dictionary, selected_length="Any"):

    regex_string = pattern_to_regex(pattern)

    try:
        regex = re.compile(regex_string)
    except re.error:
        return []

    matches = []

    for word in words:

        if not regex.fullmatch(word):
            continue

        # Optional exact letter-length filter
        if selected_length != "Any":
            letter_count = sum(
                ch.isalpha()
                for ch in word
            )

            if letter_count != selected_length:
                continue

        score = crossword_score(
            word,
            dictionary
        )

        matches.append(
            (word, score)
        )

    matches.sort(
        key=lambda x: (-x[1], x[0])
    )

    return matches[:MAX_PATTERN_RESULTS]

# ============================================================
# TOKENIZATION
# ============================================================

STOPWORDS = {
    "a", "an", "the", "of", "to", "in",
    "on", "for", "with", "and", "or",
    "is", "are", "was", "were", "be",
    "by", "from", "as", "at", "into",
    "that", "this", "it", "its",
    "one", "used", "use"
}


def tokenize(text):

    words = re.findall(
        r"[a-z]+",
        text.lower()
    )

    return [
        w for w in words
        if w not in STOPWORDS and len(w) > 1
    ]


# ============================================================
# DICTIONARY CLUE SEARCH
# ============================================================

@st.cache_data
def build_dictionary_search_index(dictionary):

    """
    Build a lightweight reverse index:

       definition word -> dictionary words

    This avoids repeatedly scanning the entire 22 MB
    dictionary for every clue.
    """

    index = {}

    for dictionary_word, info in dictionary.items():

        definition = info.get(
            "definition",
            ""
        )

        tokens = set(
            tokenize(definition)
        )

        for token in tokens:

            if token not in index:
                index[token] = set()

            index[token].add(
                dictionary_word
            )

    return index


def search_dictionary_for_clue(
    clue,
    pattern,
    words,
    dictionary,
    dictionary_index
):

    clue_tokens = tokenize(clue)

    if not clue_tokens:
        return []

    # --------------------------------------------------------
    # Candidate dictionary words
    # --------------------------------------------------------

    candidate_words = set()

    for token in clue_tokens:

        if token in dictionary_index:

            candidate_words.update(
                dictionary_index[token]
            )

    # --------------------------------------------------------
    # Restrict to actual crossword words
    # --------------------------------------------------------

    word_set = {
        normalized_word(w)
        for w in words
    }

    candidate_words &= word_set

    # --------------------------------------------------------
    # Apply pattern if supplied
    # --------------------------------------------------------

    if pattern.strip():

        regex_string = pattern_to_regex(
            pattern
        )

        try:
            regex = re.compile(
                regex_string,
                re.IGNORECASE
            )

            candidate_words = {
                w for w in candidate_words
                if regex.fullmatch(w)
            }

        except re.error:
            return []

    # --------------------------------------------------------
    # Score candidates
    # --------------------------------------------------------

    results = []

    for candidate in candidate_words:

        info = dictionary.get(
            candidate,
            {}
        )

        definition = info.get(
            "definition",
            ""
        )

        if not definition:
            continue

        definition_lower = definition.lower()

        score = 0.0

        # ----------------------------------------------------
        # Exact phrase bonus
        # ----------------------------------------------------

        clue_lower = clue.lower().strip()

        if clue_lower in definition_lower:
            score += 30

        # ----------------------------------------------------
        # Token overlap
        # ----------------------------------------------------

        definition_tokens = tokenize(
            definition
        )

        definition_counter = Counter(
            definition_tokens
        )

        for token in clue_tokens:

            if token in definition_counter:

                score += 10

                # Multiple occurrences
                score += min(
                    definition_counter[token] - 1,
                    3
                ) * 2

        # ----------------------------------------------------
        # Length adjustment
        # ----------------------------------------------------

        if len(candidate) <= 5:
            score += 2

        # ----------------------------------------------------
        # Word frequency
        # ----------------------------------------------------

        score += frequency_score(candidate) * 1.5

        results.append(
            (
                candidate.upper(),
                score,
                definition,
                info.get("part_of_speech", "")
            )
        )

    results.sort(
        key=lambda x: (-x[1], x[0])
    )

    return results[:MAX_CLUE_RESULTS]


# ============================================================
# LOAD DATA
# ============================================================

words = load_words()
dictionary = load_dictionary()

dictionary_index = build_dictionary_search_index(
    dictionary
)


# ============================================================
# SESSION STATE
# ============================================================

if "selected_meaning_word" not in st.session_state:
    st.session_state.selected_meaning_word = None

if "pattern_results" not in st.session_state:
    st.session_state.pattern_results = []

if "last_pattern" not in st.session_state:
    st.session_state.last_pattern = ""


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧩 Crossword Solver")

st.sidebar.markdown(
    f"**Words loaded:** {len(words):,}"
)

st.sidebar.markdown(
    f"**Dictionary entries:** {len(dictionary):,}"
)

st.sidebar.markdown(
    "### Search rules"
)

st.sidebar.markdown(
    """
    `?` or `_` → unknown letter

    `A-Z` → fixed letter

    `-` → literal hyphen

    `'` → literal apostrophe
    """
)

if WORDFREQ_AVAILABLE:
    st.sidebar.success(
        "Word-frequency ranking enabled"
    )
else:
    st.sidebar.info(
        "Using built-in crossword ranking. "
        "Install `wordfreq` for better commonness ranking."
    )


# ============================================================
# TITLE
# ============================================================

st.title("🧩 Crossword Solver")

st.caption(
    "Pattern search + local dictionary clue search"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "🔎 Pattern Search",
        "📖 Dictionary Clue Solver"
    ]
)


# ============================================================
# TAB 1 — PATTERN SEARCH
# ============================================================

with tab1:

    st.subheader("Pattern Search")

    pattern = st.text_input(
        "Enter crossword pattern",
        placeholder="Example: A??E or ?A? or AARD-V_RK",
        key="pattern_input"
    )

    length_options = ["Any"] + list(range(3, 10))

    length_options = ["Any"] + list(range(3, 10))

    selected_length = st.selectbox(
        "Exact word length",
        options=length_options,
        index=0,
        help="Choose Any for all lengths, or select 3–9 for an exact letter count."
    )

    search_clicked = st.button(
        "🔍 Search",
        type="primary",
        key="pattern_search_button"
    )

    if search_clicked:

        if not pattern.strip():

            st.warning(
                "Please enter a pattern."
            )

            st.session_state.pattern_results = []
            st.session_state.last_pattern = ""

        else:

            with st.spinner("Searching word list..."):

                results = find_pattern_matches(
                pattern,
                words,
                dictionary,
                selected_length
            )

            st.session_state.pattern_results = results
            st.session_state.last_pattern = pattern.upper()

            # Reset selected meaning for a new search
            st.session_state.selected_meaning_word = None

    # --------------------------------------------------------
    # Display previous/current results
    # --------------------------------------------------------

    results = st.session_state.pattern_results

    if results:

        st.markdown(
            f"**Top {len(results)} possibilities for "
            f"`{st.session_state.last_pattern}`**"
        )

        for rank, (word, score) in enumerate(
            results,
            start=1
        ):

            col1, col2, col3 = st.columns(
                [0.10, 0.28, 0.62]
            )

            with col1:

                st.markdown(
                    f"<div class='crossword-rank'>"
                    f"#{rank}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    f"<div class='crossword-word'>"
                    f"{word}"
                    f"</div>",
                    unsafe_allow_html=True
                )

                st.caption(
                    f"Score: {score:.1f}"
                )

            with col3:

                # ------------------------------------------------
                # Meaning button
                # ------------------------------------------------

                if st.button(
                    "Meaning",
                    key=f"meaning_{rank}_{word}"
                ):

                    st.session_state.selected_meaning_word = (
                        word
                    )

                # ------------------------------------------------
                # Meaning shown beside the selected word
                # ------------------------------------------------

                if (
                    st.session_state.selected_meaning_word
                    == word
                ):

                    dictionary_key = (
                        normalized_word(word)
                    )

                    info = dictionary.get(
                        dictionary_key
                    )

                    if info:

                        definition = info.get(
                            "definition",
                            ""
                        )

                        part_of_speech = info.get(
                            "part_of_speech",
                            ""
                        )

                        if part_of_speech:

                            st.markdown(
                                f"<div class='crossword-pos'>"
                                f"{part_of_speech}"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                        if definition:

                            st.markdown(
                                f"<div class='crossword-definition'>"
                                f"{definition}"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                        else:

                            st.caption(
                                "Definition not available."
                            )

                    else:

                        st.caption(
                            "Meaning not found in local dictionary."
                        )

    elif st.session_state.last_pattern:

        st.info(
            "No matching words found."
        )


# ============================================================
# TAB 2 — DICTIONARY CLUE SOLVER
# ============================================================

with tab2:

    st.subheader(
        "Dictionary Clue Solver"
    )

    st.caption(
        "Searches your local dictionary only. "
        "No AI ranking or external API is used."
    )

    clue = st.text_input(
        "Enter clue",
        placeholder="Example: Small domesticated feline",
        key="clue_input"
    )

    clue_pattern = st.text_input(
        "Optional pattern",
        placeholder="Example: C??",
        key="clue_pattern_input"
    )

    clue_search_clicked = st.button(
        "📖 Search Dictionary",
        type="primary",
        key="clue_search_button"
    )

    if clue_search_clicked:

        if not clue.strip():

            st.warning(
                "Please enter a clue."
            )

        else:

            with st.spinner(
                "Searching local dictionary..."
            ):

                clue_results = (
                    search_dictionary_for_clue(
                        clue,
                        clue_pattern,
                        words,
                        dictionary,
                        dictionary_index
                    )
                )

            if not clue_results:

                st.info(
                    "No strong dictionary matches found."
                )

            else:

                st.markdown(
                    f"**Top {len(clue_results)} "
                    f"dictionary matches**"
                )

                for (
                    rank,
                    (
                        word,
                        score,
                        definition,
                        part_of_speech
                    )
                ) in enumerate(
                    clue_results,
                    start=1
                ):

                    st.markdown(
                        f"""
                        <div class="clue-result-word">
                            #{rank} &nbsp; {word}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if part_of_speech:

                        st.markdown(
                            f"""
                            <div class="crossword-pos">
                                {part_of_speech}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        f"""
                        <div class="clue-result-definition">
                            {definition}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.caption(
                        f"Match score: {score:.1f}"
                    )

                    st.divider()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Word list: words.txt • "
    "Meanings: dictionary.json • "
    "Clue search: dictionary only"
)
