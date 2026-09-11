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


# ============================================================
# CONFIGURATION
# ============================================================

WORD_FILE = "words.txt"
DICTIONARY_FILE = "dictionary.json"

MAX_WORD_LENGTH = 9
MAX_PATTERN_RESULTS = 20
MAX_CLUE_RESULTS = 20


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Crossword Solver",
    page_icon="🧩",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Smaller pattern result text */
    .crossword-word {
        font-size: 18px;
        font-weight: 600;
        margin: 0;
        padding-top: 3px;
    }

    .crossword-rank {
        font-size: 14px;
        color: #777;
        padding-top: 5px;
    }

    .crossword-definition {
        font-size: 14px;
        line-height: 1.35;
        padding-top: 4px;
    }

    .crossword-no-definition {
        font-size: 13px;
        color: #888;
        padding-top: 5px;
    }

    /* Reduce spacing between result rows */
    div[data-testid="stHorizontalBlock"] {
        margin-bottom: 2px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD WORD LIST
# ============================================================

@st.cache_data
def load_words():

    if not os.path.exists(WORD_FILE):
        return []

    words = []
    seen = set()

    with open(
        WORD_FILE,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:

            word = line.strip().upper()

            if not word:
                continue

            # Current words.txt is expected to contain
            # alphabetic crossword entries.
            if not word.isalpha():
                continue

            if len(word) < 3 or len(word) > MAX_WORD_LENGTH:
                continue

            if word not in seen:

                seen.add(word)
                words.append(word)

    return words


WORDS = load_words()

# Set gives much faster validation for clue results
WORD_SET = set(WORDS)


# ============================================================
# LOAD LOCAL DICTIONARY
# ============================================================

@st.cache_resource
def load_dictionary():

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

            word = str(key).strip().lower()

            if not word:
                continue

            # Standard format:
            #
            # "word": "definition"

            if isinstance(value, str):

                definition = value.strip()

            # Also support:
            #
            # "word": {
            #     "definition": "...",
            #     "part_of_speech": "noun"
            # }

            elif isinstance(value, dict):

                if "definition" in value:

                    definition = str(
                        value["definition"]
                    ).strip()

                elif "definitions" in value:

                    definitions = value["definitions"]

                    if isinstance(
                        definitions,
                        list,
                    ):

                        definition = " ".join(
                            str(x).strip()
                            for x in definitions
                            if str(x).strip()
                        )

                    else:

                        definition = str(
                            definitions
                        ).strip()

                else:

                    definition = ""

            elif isinstance(value, list):

                definition = " ".join(
                    str(x).strip()
                    for x in value
                    if str(x).strip()
                )

            else:

                definition = str(value).strip()

            if definition:

                dictionary[word] = definition

        return dictionary

    except Exception as e:

        st.error(
            f"Could not load dictionary.json: {e}"
        )

        return {}


DICTIONARY = load_dictionary()


# ============================================================
# COMMON WORD / RANKING DATA
# ============================================================

COMMON_LETTERS = set(
    "ETAOINSHRDLU"
)

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
}


# ============================================================
# CROSSWORD RANKING
# ============================================================

def word_frequency_score(word):

    lower_word = word.lower()

    if WORDFREQ_AVAILABLE:

        return zipf_frequency(
            lower_word,
            "en",
        )

    # Fallback if wordfreq is unavailable

    score = 0.0

    if lower_word in DICTIONARY:
        score += 5.0

    common_count = sum(
        1
        for char in word
        if char in COMMON_LETTERS
    )

    score += common_count * 0.15

    vowels = sum(
        1
        for char in word
        if char in "AEIOU"
    )

    if len(word) >= 4:

        ratio = vowels / len(word)

        if 0.20 <= ratio <= 0.60:
            score += 1.0

    if word in COMMON_SHORT_WORDS:
        score += 2.0

    return score


def crossword_score(word):

    score = word_frequency_score(word)

    lower_word = word.lower()

    # Dictionary presence
    if lower_word in DICTIONARY:
        score += 2.0

    # Length
    length_bonus = {
        3: 0.20,
        4: 0.45,
        5: 0.65,
        6: 0.55,
        7: 0.45,
        8: 0.30,
        9: 0.20,
    }

    score += length_bonus.get(
        len(word),
        0,
    )

    # Vowel balance
    vowels = sum(
        1
        for char in word
        if char in "AEIOU"
    )

    if len(word) >= 4:

        vowel_ratio = vowels / len(word)

        if 0.20 <= vowel_ratio <= 0.60:
            score += 0.60

        elif vowel_ratio < 0.10:
            score -= 0.80

    # Common letters
    common_count = sum(
        1
        for char in word
        if char in COMMON_LETTERS
    )

    score += common_count * 0.08

    # Excessive consonant clusters
    if re.search(
        r"[BCDFGHJKLMNPQRSTVWXYZ]{4,}",
        word,
    ):
        score -= 1.0

    if re.search(
        r"[BCDFGHJKLMNPQRSTVWXYZ]{5,}",
        word,
    ):
        score -= 1.5

    # Repeated letters
    repeated = (
        len(word)
        - len(set(word))
    )

    if repeated >= 3:
        score -= 0.30

    # Slight inflection penalty
    if len(word) >= 6:

        if word.endswith("ING"):
            score -= 0.10

        elif word.endswith("ED"):
            score -= 0.08

        elif word.endswith("ES"):
            score -= 0.06

    return score


def rank_matches(matches):

    scored = []

    for word in matches:

        scored.append(
            (
                crossword_score(word),
                word,
            )
        )

    scored.sort(
        key=lambda x: (
            -x[0],
            x[1],
        )
    )

    return [
        word
        for score, word in scored
    ]


# ============================================================
# PATTERN MATCHING
# ============================================================

def pattern_to_regex(pattern):

    parts = []

    for ch in pattern.strip():

        if ch in ["?", "_"]:

            # One unknown character
            parts.append(".")

        elif ch.isalnum():

            parts.append(
                re.escape(
                    ch.upper()
                )
            )

        elif ch in ["-", "'"]:

            parts.append(
                re.escape(ch)
            )

        else:

            parts.append(
                re.escape(ch)
            )

    return "^" + "".join(parts) + "$"


def find_pattern_matches(pattern):

    if not pattern.strip():
        return []

    regex_string = pattern_to_regex(
        pattern
    )

    try:

        regex = re.compile(
            regex_string
        )

    except re.error:

        return []

    matches = [
        word
        for word in WORDS
        if regex.fullmatch(word)
    ]

    return rank_matches(
        matches
    )


# ============================================================
# LOCAL MEANING
# ============================================================

def get_meaning(word):

    return DICTIONARY.get(
        word.strip().lower()
    )


def shorten_definition(
    definition,
    max_chars=300,
):

    if not definition:
        return "Definition unavailable."

    definition = re.sub(
        r"\s+",
        " ",
        definition,
    ).strip()

    if len(definition) <= max_chars:
        return definition

    sentences = re.split(
        r"(?<=[.!?])\s+",
        definition,
    )

    if sentences:

        first = sentences[0].strip()

        if len(first) <= max_chars:
            return first

    shortened = definition[
        :max_chars
    ]

    if " " in shortened:

        shortened = shortened.rsplit(
            " ",
            1,
        )[0]

    return shortened + "…"


# ============================================================
# DICTIONARY-ONLY CLUE SEARCH
# ============================================================

# Common English words that don't add much meaning
# when comparing a clue with dictionary definitions.

STOP_WORDS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "as",
    "this",
    "that",
    "these",
    "those",
    "its",
    "it",
    "their",
    "his",
    "her",
    "his",
    "used",
    "use",
    "one",
    "someone",
    "something",
    "person",
}


def tokenize(text):

    words = re.findall(
        r"[a-z]+",
        text.lower(),
    )

    return [
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) >= 2
    ]


def clue_score(
    clue_tokens,
    definition,
    word,
):

    definition_lower = (
        definition.lower()
    )

    definition_tokens = set(
        tokenize(definition)
    )

    if not clue_tokens:
        return 0.0

    # --------------------------------------------------------
    # Direct token overlap
    # --------------------------------------------------------

    overlap = sum(
        1
        for token in clue_tokens
        if token in definition_tokens
    )

    score = overlap * 10.0

    # --------------------------------------------------------
    # Exact phrase bonus
    # --------------------------------------------------------

    clue_phrase = " ".join(
        clue_tokens
    )

    if clue_phrase in definition_lower:

        score += 20.0

    # --------------------------------------------------------
    # Individual clue word occurrence
    # --------------------------------------------------------

    for token in clue_tokens:

        if token in definition_lower:

            score += 2.0

    # --------------------------------------------------------
    # Definition length penalty
    #
    # This helps avoid very long definitions that happen
    # to contain many common words.
    # --------------------------------------------------------

    if len(definition) > 500:

        score -= 0.5

    if len(definition) > 1000:

        score -= 0.5

    # --------------------------------------------------------
    # Dictionary-backed answer bonus
    # --------------------------------------------------------

    if word.lower() in DICTIONARY:

        score += 1.0

    return score


def search_dictionary_for_clue(
    clue,
    pattern=None,
):

    clue_tokens = tokenize(
        clue
    )

    if not clue_tokens:
        return []

    # --------------------------------------------------------
    # If pattern supplied, restrict dictionary search to
    # matching words first.
    # --------------------------------------------------------

    if pattern and pattern.strip():

        candidate_words = (
            find_pattern_matches(
                pattern
            )
        )

    else:

        # Only search words that are both:
        # 1. in words.txt
        # 2. in dictionary.json
        #
        # This prevents dictionary-only words that aren't
        # valid crossword candidates from appearing.

        candidate_words = [
            word
            for word in WORDS
            if word.lower()
            in DICTIONARY
        ]

    scored = []

    for word in candidate_words:

        definition = DICTIONARY.get(
            word.lower()
        )

        if not definition:
            continue

        score = clue_score(
            clue_tokens,
            definition,
            word,
        )

        # Only keep entries with some evidence of relevance
        if score > 0:

            scored.append(
                (
                    score,
                    word,
                    definition,
                )
            )

    # --------------------------------------------------------
    # Sort by clue relevance first.
    #
    # For equal relevance, use crossword score.
    # --------------------------------------------------------

    scored.sort(
        key=lambda x: (
            -x[0],
            -crossword_score(x[1]),
            x[1],
        )
    )

    return scored[
        :MAX_CLUE_RESULTS
    ]


# ============================================================
# SESSION STATE
# ============================================================

if (
    "pattern_matches"
    not in st.session_state
):

    st.session_state.pattern_matches = []


if (
    "last_pattern"
    not in st.session_state
):

    st.session_state.last_pattern = ""


if (
    "selected_meaning_word"
    not in st.session_state
):

    st.session_state.selected_meaning_word = None


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧩 Crossword Solver"
)

st.caption(
    "Pattern matching + local dictionary meanings + dictionary-based clue solving"
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
        "words.txt could not be loaded. "
        "Make sure it is in the same folder "
        "as streamlit_app.py."
    )


if not DICTIONARY:

    st.warning(
        "dictionary.json could not be loaded. "
        "Pattern search will still work, "
        "but meanings and clue search will not."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "🔎 Pattern Search",
        "📖 Dictionary Clue Solver",
    ]
)


# ============================================================
# PATTERN SEARCH
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

Examples: `C??`, `?A?`, `C??E`
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

            st.session_state.pattern_matches = (
                matches
            )

            st.session_state.last_pattern = (
                pattern.upper()
            )

            st.session_state.selected_meaning_word = (
                None
            )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    matches = (
        st.session_state.pattern_matches
    )

    if matches:

        st.success(
            f"Found {len(matches):,} matching words. "
            f"Showing the top "
            f"{min(MAX_PATTERN_RESULTS, len(matches))}."
        )

        st.markdown("---")

        for index, word in enumerate(
            matches[
                :MAX_PATTERN_RESULTS
            ],
            start=1,
        ):

            # Three columns:
            #
            # rank | word | meaning
            #
            col_rank, col_word, col_meaning = st.columns(
                [0.5, 2.0, 5.0]
            )

            with col_rank:

                st.markdown(
                    f'<div class="crossword-rank">'
                    f'{index}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col_word:

                st.markdown(
                    f'<div class="crossword-word">'
                    f'{word}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col_meaning:

                # If this word has been selected,
                # show the meaning HERE.
                #
                # This means the user doesn't have
                # to scroll back to the top.

                if (
                    st.session_state.selected_meaning_word
                    == word
                ):

                    definition = get_meaning(
                        word
                    )

                    if definition:

                        st.markdown(
                            '<div class="crossword-definition">'
                            '<b>Meaning:</b> '
                            + shorten_definition(
                                definition
                            )
                            + '</div>',
                            unsafe_allow_html=True,
                        )

                    else:

                        st.markdown(
                            '<div class="crossword-no-definition">'
                            'No local definition available.'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                else:

                    if word.lower() in DICTIONARY:

                        if st.button(
                            "Meaning",
                            key=(
                                f"meaning_"
                                f"{index}_"
                                f"{word}"
                            ),
                        ):

                            st.session_state.selected_meaning_word = (
                                word
                            )

                            st.rerun()

                    else:

                        st.caption(
                            "No definition"
                        )

            st.divider()

    elif st.session_state.last_pattern:

        st.info(
            "No words match this pattern."
        )


# ============================================================
# DICTIONARY CLUE SOLVER
# ============================================================

with tab2:

    st.subheader(
        "📖 Solve Clue from Dictionary"
    )

    st.write(
        "This solver searches your local "
        "`dictionary.json` definitions. "
        "It does not use AI."
    )

    clue = st.text_area(
        "Enter clue",
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
        "📖 Search Dictionary",
        type="primary",
        use_container_width=True,
    )

    if solve_button:

        if not clue.strip():

            st.warning(
                "Please enter a clue."
            )

        else:

            with st.spinner(
                "Searching dictionary..."
            ):

                clue_results = (
                    search_dictionary_for_clue(
                        clue=clue,
                        pattern=clue_pattern,
                    )
                )

            if not clue_results:

                st.info(
                    "No strong dictionary matches "
                    "were found."
                )

            else:

                st.success(
                    f"Found {len(clue_results)} "
                    "possible answers."
                )

                st.markdown("---")

                for index, (
                    score,
                    word,
                    definition,
                ) in enumerate(
                    clue_results,
                    start=1,
                ):

                    col_rank, col_answer, col_definition = (
                        st.columns(
                            [0.5, 1.8, 6]
                        )
                    )

                    with col_rank:

                        st.markdown(
                            f'<div class="crossword-rank">'
                            f'{index}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    with col_answer:

                        st.markdown(
                            f'<div class="crossword-word">'
                            f'{word}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    with col_definition:

                        st.markdown(
                            f'<div class="crossword-definition">'
                            f'{shorten_definition(definition)}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Solver Information"
    )

    st.write(
        f"**Word list:** `{WORD_FILE}`"
    )

    st.write(
        f"**Dictionary:** `{DICTIONARY_FILE}`"
    )

    st.write(
        f"**Words loaded:** "
        f"{len(WORDS):,}"
    )

    st.write(
        f"**Definitions loaded:** "
        f"{len(DICTIONARY):,}"
    )

    st.divider()

    st.subheader(
        "Pattern Search"
    )

    st.caption(
        "Uses words.txt as the authoritative "
        "crossword word list."
    )

    st.caption(
        "Results are ranked using word frequency "
        "and crossword-oriented signals."
    )

    st.divider()

    st.subheader(
        "Clue Solver"
    )

    st.caption(
        "Uses dictionary.json only."
    )

    st.caption(
        "No OpenAI API calls are made."
    )

    st.divider()

    st.success(
        "Local dictionary enabled"
    )

    st.caption(
        "Meanings and clue searches work "
        "without an external dictionary API."
    )
