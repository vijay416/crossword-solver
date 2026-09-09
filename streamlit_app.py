import os
import re
import json
import requests
import streamlit as st

from openai import OpenAI


# =========================================================
# CONFIG
# =========================================================

PATTERN_RESULT_LIMIT = 20
AI_CANDIDATE_LIMIT = 500
AI_RESULT_LIMIT = 5
DICTIONARY_TIMEOUT = 5

# How many words to consider when ranking pattern matches
RANKING_CANDIDATE_LIMIT = 1000


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Crossword Solver",
    page_icon="🧩",
    layout="centered"
)


# =========================================================
# SESSION STATE
# =========================================================

if "pattern_matches" not in st.session_state:
    st.session_state["pattern_matches"] = []

if "searched_pattern" not in st.session_state:
    st.session_state["searched_pattern"] = ""

if "meaning_cache" not in st.session_state:
    st.session_state["meaning_cache"] = {}


# =========================================================
# OPENAI CONFIGURATION
# =========================================================

def get_openai_key():

    try:
        return st.secrets["OPENAI_API_KEY"]

    except Exception:
        return os.environ.get("OPENAI_API_KEY")


def get_openai_model():

    try:
        return st.secrets.get(
            "OPENAI_MODEL",
            "gpt-4o-mini"
        )

    except Exception:
        return os.environ.get(
            "OPENAI_MODEL",
            "gpt-4o-mini"
        )


OPENAI_API_KEY = get_openai_key()

if OPENAI_API_KEY:

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

else:

    client = None


# =========================================================
# WORD LIST
# =========================================================

@st.cache_data
def load_words():

    try:

        with open(
            "words.txt",
            "r",
            encoding="utf-8"
        ) as f:

            words = []

            for line in f:

                word = line.strip().lower()

                if not word:
                    continue

                # Maximum 9 characters
                if len(word) > 9:
                    continue

                # Ignore words containing spaces
                if " " in word:
                    continue

                # Keep only alphabetic words plus
                # apostrophe/hyphenated words
                if not all(
                    ch.isalpha() or ch in ["-", "'"]
                    for ch in word
                ):
                    continue

                words.append(word)

            return sorted(set(words))

    except FileNotFoundError:

        return []


WORD_LIST = load_words()


# =========================================================
# WORD COMMONNESS / RANKING
# =========================================================

@st.cache_resource
def load_word_frequency():

    """
    Try to use the wordfreq package.

    If it is not installed, the application still works
    using the fallback scoring system.
    """

    try:

        from wordfreq import zipf_frequency

        return zipf_frequency

    except Exception:

        return None


ZIPF_FREQUENCY = load_word_frequency()


def word_score(word):

    """
    Estimate how likely a word is to be a useful/common
    crossword answer.

    Higher score = more common/useful.

    If wordfreq is available, use its English frequency.

    Otherwise use a lightweight heuristic.
    """

    word = word.lower()

    # -----------------------------------------------------
    # Best option: wordfreq
    # -----------------------------------------------------

    if ZIPF_FREQUENCY:

        score = ZIPF_FREQUENCY(
            word,
            "en"
        )

        # Small crossword-friendly adjustments

        # Penalize very long words slightly
        if len(word) >= 8:
            score -= 0.15

        # Penalize obvious plural
        if (
            len(word) > 3
            and word.endswith("s")
        ):
            score -= 0.20

        return score

    # -----------------------------------------------------
    # Fallback scoring
    # -----------------------------------------------------

    score = 0.0

    length = len(word)

    # Prefer normal crossword lengths
    if 3 <= length <= 7:
        score += 2.0

    elif length == 8:
        score += 1.0

    elif length == 9:
        score += 0.5

    # Slight preference for shorter answers
    score += max(
        0,
        8 - length
    ) * 0.1

    # Penalize obvious plurals
    if (
        length > 3
        and word.endswith("s")
    ):
        score -= 0.5

    # Penalize unusual letter combinations
    unusual_patterns = [
        "qz",
        "qx",
        "jq",
        "zx",
        "jz"
    ]

    for pattern in unusual_patterns:

        if pattern in word:
            score -= 1.0

    return score


def rank_words(words):

    """
    Sort words by estimated English commonness.

    Most likely/common words first.
    """

    return sorted(
        words,
        key=lambda word: (
            -word_score(word),
            len(word),
            word
        )
    )


# =========================================================
# PATTERN → REGEX
# =========================================================

def pattern_to_regex(pattern):

    """
    Crossword pattern rules:

        ? = unknown letter
        _ = unknown letter

        letters = fixed letters
        numbers = fixed numbers

        - = literal hyphen
        ' = literal apostrophe

    Examples:

        C_T
        C?T
        ??A?
        S???
    """

    regex_parts = []

    for ch in pattern.lower():

        if ch in ["?", "_"]:

            # One unknown character
            regex_parts.append(".")

        elif ch.isalnum():

            # Fixed character
            regex_parts.append(ch)

        elif ch in ["-", "'"]:

            # Literal punctuation
            regex_parts.append(
                re.escape(ch)
            )

        else:

            # Escape anything else
            regex_parts.append(
                re.escape(ch)
            )

    return "^" + "".join(regex_parts) + "$"


# =========================================================
# FIND PATTERN MATCHES
# =========================================================

def find_candidates(
    pattern,
    limit=PATTERN_RESULT_LIMIT
):

    if not pattern:

        return []

    pattern = pattern.strip().lower()

    try:

        regex = re.compile(
            pattern_to_regex(pattern)
        )

    except re.error:

        return []

    matches = []

    for word in WORD_LIST:

        if regex.fullmatch(word):

            matches.append(word)

    # Rank by commonness
    matches = rank_words(matches)

    return matches[:limit]


# =========================================================
# DICTIONARY LOOKUP
# =========================================================

@st.cache_data(ttl=86400)
def get_meaning(word):

    """
    Fetch definition only when requested.

    Result is cached for 24 hours.
    """

    word = word.lower().strip()

    try:

        url = (
            "https://api.dictionaryapi.dev/"
            f"api/v2/entries/en/{word}"
        )

        response = requests.get(
            url,
            timeout=DICTIONARY_TIMEOUT
        )

        if response.status_code != 200:

            return "Definition unavailable."

        data = response.json()

        if not isinstance(data, list):
            return "Definition unavailable."

        if not data:
            return "Definition unavailable."

        # Search through all returned meanings
        for entry in data:

            meanings = entry.get(
                "meanings",
                []
            )

            for meaning in meanings:

                definitions = meaning.get(
                    "definitions",
                    []
                )

                for definition in definitions:

                    text = definition.get(
                        "definition"
                    )

                    if text:

                        return text

        return "Definition unavailable."

    except requests.exceptions.Timeout:

        return "Definition lookup timed out."

    except Exception:

        return "Definition unavailable."


# =========================================================
# AI CLUE SOLVER
# =========================================================

def rank_candidates(
    clue,
    pattern,
    candidates
):

    if not client:

        return []

    if not candidates:

        return []

    candidate_text = ", ".join(
        candidates
    )

    prompt = f"""
You are an expert crossword puzzle solver.

Clue:
"{clue}"

Pattern:
"{pattern}"

These are the ONLY possible answers:

{candidate_text}

Rank the best answers for the clue.

Rules:

1. ONLY use words from the supplied list.
2. Never invent an answer.
3. Every answer must match the supplied pattern.
4. Consider synonyms, alternate meanings,
   crossword conventions and wordplay.
5. Return at most 5 answers.
6. Most likely answer first.
7. Return ONLY valid JSON.

Format:

[
  {{
    "word": "answer",
    "reason": "short explanation"
  }}
]
"""

    try:

        response = client.chat.completions.create(

            model=get_openai_model(),

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0
        )

        text = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        # -------------------------------------------------
        # Handle ```json ... ``` if model returns it
        # -------------------------------------------------

        if text.startswith("```"):

            text = re.sub(
                r"^```(?:json)?\s*",
                "",
                text
            )

            text = re.sub(
                r"\s*```$",
                "",
                text
            )

        results = json.loads(text)

        if not isinstance(
            results,
            list
        ):

            return []

        candidate_set = set(
            candidates
        )

        final_results = []

        for item in results:

            if not isinstance(
                item,
                dict
            ):

                continue

            word = str(
                item.get(
                    "word",
                    ""
                )
            ).strip().lower()

            # -------------------------------------------------
            # SECURITY / VALIDATION
            # -------------------------------------------------

            if word not in candidate_set:
                continue

            final_results.append({

                "word": word,

                "reason": str(
                    item.get(
                        "reason",
                        ""
                    )
                )

            })

            if len(final_results) >= AI_RESULT_LIMIT:
                break

        return final_results

    except Exception as e:

        error_text = str(e)

        if (
            "429" in error_text
            or "insufficient_quota" in error_text
            or "credit_balance_exhausted" in error_text
        ):

            st.warning(
                "AI is currently unavailable because "
                "the OpenAI API quota has been exhausted. "
                "Pattern Search is still available."
            )

        else:

            st.warning(
                "AI could not solve this clue."
            )

        return []


# =========================================================
# HEADER
# =========================================================

st.title("🧩 Crossword Solver")

if WORD_LIST:

    st.caption(
        f"Word database: {len(WORD_LIST):,} words"
    )

else:

    st.error(
        "words.txt was not found. "
        "Please add words.txt to the repository."
    )


# =========================================================
# TABS
# =========================================================

pattern_tab, clue_tab = st.tabs([
    "🔎 Pattern Search",
    "🤖 AI Clue Solver"
])


# =========================================================
# PATTERN SEARCH
# =========================================================

with pattern_tab:

    st.subheader(
        "Find words from a pattern"
    )

    pattern = st.text_input(
        "Pattern",
        placeholder="Example: C?T",
        key="pattern"
    )

    st.caption(
        "Use ? or _ for an unknown letter. "
        "Letters are fixed. Example: C?T or C_T"
    )

    if st.button(
        "Find Words",
        type="primary",
        key="pattern_button"
    ):

        if not pattern.strip():

            st.warning(
                "Please enter a pattern."
            )

        else:

            matches = find_candidates(
                pattern.strip(),
                PATTERN_RESULT_LIMIT
            )

            # Save results in session state
            st.session_state[
                "pattern_matches"
            ] = matches

            st.session_state[
                "searched_pattern"
            ] = pattern.strip().lower()

    # -----------------------------------------------------
    # DISPLAY SAVED RESULTS
    # -----------------------------------------------------

    matches = st.session_state.get(
        "pattern_matches",
        []
    )

    searched_pattern = st.session_state.get(
        "searched_pattern",
        ""
    )

    if matches:

        st.success(
            f"Showing top {len(matches)} "
            f"matches for **{searched_pattern.upper()}**"
        )

        st.caption(
            "Results are ranked by estimated "
            "English word commonness."
        )

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------

        for index, word in enumerate(
            matches,
            1
        ):

            col1, col2 = st.columns(
                [4, 1]
            )

            with col1:

                st.markdown(
                    f"**{index}. {word.upper()}**"
                )

            with col2:

                # Use an expander-style button
                # but preserve results in session state

                show_key = (
                    f"meaning_visible_{word}"
                )

                if st.button(
                    "Meaning",
                    key=f"meaning_button_{word}_{index}"
                ):

                    st.session_state[
                        show_key
                    ] = not st.session_state.get(
                        show_key,
                        False
                    )

            # -------------------------------------------------
            # MEANING
            # -------------------------------------------------

            if st.session_state.get(
                show_key,
                False
            ):

                meaning = get_meaning(
                    word
                )

                st.caption(
                    f"📖 {meaning}"
                )

            st.divider()

    elif searched_pattern:

        st.info(
            f"No words found for "
            f"**{searched_pattern.upper()}**."
        )


# =========================================================
# AI CLUE SOLVER
# =========================================================

with clue_tab:

    st.subheader(
        "Solve a crossword clue"
    )

    clue = st.text_input(
        "Clue",
        placeholder="Example: Feline pet",
        key="clue"
    )

    ai_pattern = st.text_input(
        "Pattern",
        placeholder="Example: C?T",
        key="ai_pattern"
    )

    st.caption(
        "Use ? or _ for unknown letters."
    )

    if st.button(
        "Solve Clue",
        type="primary",
        key="ai_button"
    ):

        if not clue.strip():

            st.warning(
                "Please enter a clue."
            )

        elif not ai_pattern.strip():

            st.warning(
                "Please enter the answer pattern."
            )

        elif not client:

            st.warning(
                "OpenAI API is not configured. "
                "You can still use Pattern Search."
            )

        else:

            # -------------------------------------------------
            # FIND CANDIDATES
            # -------------------------------------------------

            with st.spinner(
                "Finding possible answers..."
            ):

                candidates = find_candidates(
                    ai_pattern.strip(),
                    AI_CANDIDATE_LIMIT
                )

            if not candidates:

                st.error(
                    "No words in the wordlist "
                    "match this pattern."
                )

            else:

                st.caption(
                    f"Found {len(candidates)} "
                    "possible candidates."
                )

                # -------------------------------------------------
                # AI RANKING
                # -------------------------------------------------

                with st.spinner(
                    "AI is ranking the candidates..."
                ):

                    results = rank_candidates(
                        clue,
                        ai_pattern,
                        candidates
                    )

                if results:

                    st.success(
                        "Best matches:"
                    )

                    for index, result in enumerate(
                        results,
                        1
                    ):

                        word = result["word"]

                        st.markdown(
                            f"### {index}. "
                            f"{word.upper()}"
                        )

                        if result.get(
                            "reason"
                        ):

                            st.write(
                                f"💡 {result['reason']}"
                            )

                        # Meaning is still cached
                        meaning = get_meaning(
                            word
                        )

                        st.caption(
                            f"📖 {meaning}"
                        )

                        st.divider()

                else:

                    st.info(
                        "AI could not rank the "
                        "candidate words."
                    )


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "🧩 Pattern search works without AI. "
    "🤖 AI is used only for clue interpretation."
)
