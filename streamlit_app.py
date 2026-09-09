import os
import re
import json
import requests
import streamlit as st
from openai import OpenAI


# =========================================================
# CONFIG
# =========================================================

PATTERN_RESULT_LIMIT = 100
AI_CANDIDATE_LIMIT = 500
AI_RESULT_LIMIT = 5
DICTIONARY_TIMEOUT = 5


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Crossword Solver",
    page_icon="🧩",
    layout="centered"
)


# =========================================================
# OPENAI
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
    client = OpenAI(api_key=OPENAI_API_KEY)
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

                # Maximum 9 letters
                if len(word) > 9:
                    continue

                # Ignore spaces
                if " " in word:
                    continue

                words.append(word)

            return sorted(set(words))

    except FileNotFoundError:

        return []


WORD_LIST = load_words()
# =========================================================
# SESSION STATE
# =========================================================

if "pattern_matches" not in st.session_state:
    st.session_state["pattern_matches"] = []

if "searched_pattern" not in st.session_state:
    st.session_state["searched_pattern"] = ""

if "show_meanings" not in st.session_state:
    st.session_state["show_meanings"] = {}


# =========================================================
# PATTERN SEARCH
# =========================================================

def pattern_to_regex(pattern):
    """
    Convert crossword pattern to regex.

    Rules:
        ? = unknown letter
        _ = unknown letter
        letters/numbers = fixed characters
        - = literal hyphen
        ' = literal apostrophe

    Examples:
        C_T  -> CAT, COT, CUT
        C?T  -> CAT, COT, CUT
        ??A? -> four-letter words with A in position 3
    """

    regex_parts = []

    for ch in pattern.lower():

        if ch in ["?", "_"]:
            # One unknown character
            regex_parts.append(".")

        elif ch.isalnum():
            # Known letter/number
            regex_parts.append(ch)

        elif ch in ["-", "'"]:
            # Literal punctuation
            regex_parts.append(re.escape(ch))

        else:
            # Escape any other character
            regex_parts.append(re.escape(ch))

    return "^" + "".join(regex_parts) + "$"

def find_candidates(
    pattern,
    limit=PATTERN_RESULT_LIMIT
):

    if not pattern:
        return []

    try:

        regex = re.compile(
            pattern_to_regex(pattern)
        )

    except re.error:

        return []

    results = []

    for word in WORD_LIST:

        if regex.match(word):

            results.append(word)

            if len(results) >= limit:
                break

    return results


# =========================================================
# DICTIONARY LOOKUP
# =========================================================

@st.cache_data(ttl=86400)
def get_meaning(word):

    """
    Fetch definition only when the user asks for it.

    Cached for 24 hours.
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

        meanings = data[0].get(
            "meanings",
            []
        )

        if not meanings:
            return "Definition unavailable."

        # Look through meanings for a definition
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

The following are the ONLY possible answers:

{candidate_text}

Rank the best answers for the clue.

Rules:

1. ONLY use words from the supplied list.
2. Never invent an answer.
3. Every answer must match the pattern.
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

        results = json.loads(text)

        if not isinstance(results, list):
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

        if "429" in error_text:

            st.warning(
                "AI is currently unavailable "
                "because the OpenAI API quota "
                "has been exhausted."
            )

        else:

            st.warning(
                f"AI could not solve this clue: "
                f"{error_text}"
            )

        return []


# =========================================================
# RESULT DISPLAY
# =========================================================

def display_candidate(
    word,
    index,
    show_meaning=False
):

    st.markdown(
        f"### {index}. {word.upper()}"
    )

    if show_meaning:

        meaning = get_meaning(word)

        st.caption(
            f"📖 {meaning}"
        )


# =========================================================
# HEADER
# =========================================================

st.title("🧩 Crossword Solver")

st.caption(
    f"Word database: {len(WORD_LIST):,} words"
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

# =========================================================
# PATTERN SEARCH
# =========================================================

with pattern_tab:

    st.subheader("Find words from a pattern")

    pattern = st.text_input(
        "Pattern",
        placeholder="Example: C_T or C?T",
        key="pattern"
    )

    st.caption(
        "Use ? or _ for an unknown letter. "
        "Example: C?T or C_T"
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
                pattern.strip()
            )

            # Save results so they survive Streamlit reruns
            st.session_state["pattern_matches"] = matches

            # Remember which pattern was searched
            st.session_state["searched_pattern"] = pattern.strip()

            # Reset meanings when doing a new search
            st.session_state["show_meanings"] = {}

    # -----------------------------------------------------
    # DISPLAY SAVED RESULTS
    # -----------------------------------------------------

    matches = st.session_state.get(
        "pattern_matches",
        []
    )

    if matches:

        st.success(
            f"Found {len(matches)} matching words."
        )

        st.write(
            "Click **Meaning** for any word you want to investigate."
        )

        for index, word in enumerate(matches, 1):

            col1, col2 = st.columns([3, 1])

            with col1:

                st.markdown(
                    f"**{word.upper()}**"
                )

            with col2:

                if st.button(
                    "Meaning",
                    key=f"meaning_{word}_{index}"
                ):

                    st.session_state[
                        "show_meanings"
                    ][word] = True

            # -------------------------------------------------
            # SHOW MEANING IF REQUESTED
            # -------------------------------------------------

            if st.session_state.get(
                "show_meanings",
                {}
            ).get(word, False):

                meaning = get_meaning(word)

                st.caption(
                    f"📖 {meaning}"
                )

            st.divider()

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
        "Providing the pattern gives "
        "much better results."
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
                    "possible answers."
                )

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

                        if result.get("reason"):

                            st.write(
                                f"💡 {result['reason']}"
                            )

                        # Meaning is retrieved
                        # only for AI-selected words

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
    "AI is used only for clue interpretation."
)
