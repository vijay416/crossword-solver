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
AI_RESULT_LIMIT = 5

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
        return st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    except Exception:
        return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


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
        with open("words.txt", "r", encoding="utf-8") as f:

            words = []

            for line in f:

                word = line.strip().lower()

                if not word:
                    continue

                # Keep crossword-friendly words
                if len(word) > 9:
                    continue

                # Remove anything containing spaces
                if " " in word:
                    continue

                words.append(word)

            # Remove duplicates
            words = sorted(set(words))

            return words

    except FileNotFoundError:
        return []


WORD_LIST = load_words()


# =========================================================
# PATTERN HANDLING
# =========================================================

def pattern_to_regex(pattern):

    """
    Crossword pattern:

    _ or ? = unknown letter

    Example:

    C_T
    C?T

    both match:

    CAT
    COT
    CUT
    etc.
    """

    regex_parts = []

    for ch in pattern.lower():

        if ch in ["_", "?"]:
            regex_parts.append(".")

        elif ch.isalnum():
            regex_parts.append(ch)

        elif ch in ["-", "'"]:
            regex_parts.append(re.escape(ch))

        else:
            regex_parts.append(re.escape(ch))

    return "^" + "".join(regex_parts) + "$"


def find_candidates(pattern, limit=500):

    if not pattern:
        return []

    regex = re.compile(pattern_to_regex(pattern))

    results = []

    for word in WORD_LIST:

        if regex.match(word):

            results.append(word)

            if len(results) >= limit:
                break

    return results


# =========================================================
# DICTIONARY
# =========================================================

@st.cache_data(ttl=86400)
def get_meaning(word):

    try:

        url = (
            "https://api.dictionaryapi.dev/"
            f"api/v2/entries/en/{word}"
        )

        response = requests.get(
            url,
            timeout=5
        )

        if response.status_code != 200:
            return "Meaning not found."

        data = response.json()

        if not isinstance(data, list):
            return "Meaning not found."

        meanings = data[0].get("meanings", [])

        if not meanings:
            return "Meaning not found."

        definitions = meanings[0].get("definitions", [])

        if not definitions:
            return "Meaning not found."

        return definitions[0].get(
            "definition",
            "Meaning not found."
        )

    except Exception:
        return "Meaning not found."


# =========================================================
# AI RANKING
# =========================================================

def rank_candidates(clue, pattern, candidates):

    if not client:
        return []

    if not candidates:
        return []

    candidate_text = ", ".join(candidates)

    prompt = f"""
You are an expert crossword puzzle solver.

A crossword clue is:

"{clue}"

The answer pattern is:

"{pattern}"

Here are the ONLY possible candidate answers:

{candidate_text}

Your task is to rank the best answers for the clue.

Rules:

1. ONLY use words from the supplied candidate list.
2. Do not invent new answers.
3. Respect the pattern.
4. Consider common crossword meanings, synonyms,
   abbreviations, wordplay and alternate meanings.
5. Return at most 5 answers.
6. Put the most likely answer first.
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

        text = response.choices[0].message.content.strip()

        results = json.loads(text)

        if not isinstance(results, list):
            return []

        # Safety check:
        # Only allow words that actually came from our word list.

        candidate_set = set(candidates)

        filtered = []

        for item in results:

            if not isinstance(item, dict):
                continue

            word = str(
                item.get("word", "")
            ).strip().lower()

            if word not in candidate_set:
                continue

            filtered.append({
                "word": word,
                "reason": item.get(
                    "reason",
                    ""
                )
            })

            if len(filtered) >= AI_RESULT_LIMIT:
                break

        return filtered

    except Exception as e:

        st.error(f"AI error: {e}")

        return []


# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_results(results):

    if not results:

        st.info("No matches found.")

        return

    for index, result in enumerate(results, 1):

        word = result["word"]

        meaning = get_meaning(word)

        st.markdown(
            f"### {index}. {word.upper()}"
        )

        if result.get("reason"):

            st.write(
                f"**Why:** {result['reason']}"
            )

        st.write(
            f"**Meaning:** {meaning}"
        )

        st.divider()


# =========================================================
# UI
# =========================================================

st.title("🧩 Crossword Solver")

st.caption(
    f"Word database: {len(WORD_LIST):,} words"
)


tab1, tab2 = st.tabs([
    "🔎 Pattern Search",
    "🤖 AI Clue Solver"
])


# =========================================================
# PATTERN SEARCH
# =========================================================

with tab1:

    st.subheader("Find words from a pattern")

    pattern = st.text_input(
        "Pattern",
        placeholder="Example: C_T or C?T"
    )

    st.caption(
        "_ or ? = unknown letter"
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

            if not matches:

                st.info(
                    "No matching words found."
                )

            else:

                st.success(
                    f"Found {len(matches)} matches"
                )

                for word in matches:

                    st.write(
                        f"**{word.upper()}** — "
                        f"{get_meaning(word)}"
                    )


# =========================================================
# AI CLUE SOLVER
# =========================================================

with tab2:

    st.subheader("Solve a crossword clue")

    clue = st.text_input(
        "Clue",
        placeholder='Example: Feline pet'
    )

    ai_pattern = st.text_input(
        "Pattern",
        placeholder="Example: C_T"
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
                "For best results, enter the pattern too."
            )

        else:

            with st.spinner(
                "Finding and ranking candidates..."
            ):

                candidates = find_candidates(
                    ai_pattern.strip(),
                    limit=500
                )

                if not candidates:

                    st.error(
                        "No words in the word list "
                        "match this pattern."
                    )

                else:

                    st.caption(
                        f"AI is ranking "
                        f"{len(candidates)} candidates..."
                    )

                    results = rank_candidates(
                        clue,
                        ai_pattern,
                        candidates
                    )

                    display_results(results)
