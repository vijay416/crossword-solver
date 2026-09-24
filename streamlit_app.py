import math
import json
import re
from collections import Counter
import streamlit as st

# Wordfreq is optional but recommended
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
COMMON_LETTERS = set("ETAOINSHRDLU")
STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "with", "and", "or",
    "is", "are", "was", "were", "be", "by", "from", "as", "at", "into",
    "that", "this", "it", "its", "one", "used", "use"
}

# Pre-compiled regex patterns
VALID_CROSSWORD_CHAR_RE = re.compile(r"[A-Z]+(?:[-'][A-Z]+)*")
CLEAN_ALPHA_RE = re.compile(r"[^a-z]")
TOKENIZE_RE = re.compile(r"[a-z]+")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Crossword Solver",
    page_icon="🧩",
    layout="wide"
)

st.markdown(
    """
    <style>
    .crossword-word { font-size: 17px; font-weight: 600; line-height: 1.25; }
    .crossword-rank { font-size: 12px; color: #777; }
    .crossword-definition { font-size: 13px; line-height: 1.3; padding-top: 3px; }
    .crossword-pos { font-size: 11px; color: #777; font-style: italic; }
    .clue-result-word { font-size: 17px; font-weight: 600; }
    .clue-result-definition { font-size: 13px; line-height: 1.35; }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER / NORMALIZATION FUNCTIONS
# ============================================================

def letters_only(word: str) -> str:
    """Remove punctuation for frequency scoring."""
    return CLEAN_ALPHA_RE.sub("", word.lower())


def tokenize(text: str) -> list[str]:
    """Tokenize text into lower-case words ignoring stop words."""
    words = TOKENIZE_RE.findall(text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def pattern_to_regex(pattern: str) -> str:
    """Convert wildcard pattern to RegEx string."""
    regex_parts = []
    for ch in pattern.upper():
        if ch in ("?", "_"):
            regex_parts.append(".")
        elif ch == "*":
            regex_parts.append(".*")
        else:
            regex_parts.append(re.escape(ch))
    return f"^{''.join(regex_parts)}$"


def frequency_score(word: str) -> float:
    """Return commonness score for a word."""
    clean = letters_only(word)
    if not clean:
        return 0.0

    if WORDFREQ_AVAILABLE:
        try:
            return float(zipf_frequency(clean, "en"))
        except Exception:
            pass

    # Heuristic fallback
    common_words = {
        "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
        "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HAD", "HIS", "HOW",
        "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID",
        "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "YES"
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


def crossword_score(word: str, clean_word: str, dictionary: dict) -> float:
    """Rank a word based on crossword features."""
    score = frequency_score(word) * 10.0

    if word.lower() in dictionary:
        score += 18.0

    length = len(clean_word)
    length_bonuses = {2: 16, 3: 12, 4: 8, 5: 5, 6: 3}
    score += length_bonuses.get(length, 0)

    if clean_word:
        common_count = sum(1 for c in clean_word.upper() if c in COMMON_LETTERS)
        score += common_count * 0.5

        vowels = sum(1 for c in clean_word if c in "aeiou")
        ratio = vowels / length
        if 0.20 <= ratio <= 0.55:
            score += 2.5
        elif ratio < 0.10:
            score -= 2.0

    if "-" in word:
        score -= 3.0
    if "'" in word:
        score -= 2.0

    return score


# ==========================================================
# HIGHLIGHT CLUE MATCHES
# ==========================================================

def highlight_clue_matches(definition, clue):
    clue_tokens = tokenize(clue)

    if not clue_tokens:
        return definition

    highlighted = definition

    for token in sorted(clue_tokens, key=len, reverse=True):

        pattern = rf"\b({re.escape(token)})\b"

        highlighted = re.sub(
            pattern,
            r"<mark>\1</mark>",
            highlighted,
            flags=re.IGNORECASE
        )

    return highlighted

# ============================================================
# DATA LOADERS & INDEXERS
# ============================================================

@st.cache_data
def load_words():
    """Load words and pre-compute metadata for faster pattern matching."""
    words = []
    seen = set()

    try:
        with open(WORDS_FILE, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                word = line.strip().upper()
                if not word or not VALID_CROSSWORD_CHAR_RE.fullmatch(word):
                    continue

                if word not in seen:
                    seen.add(word)
                    clean = letters_only(word)
                    letter_count = len(clean)
                    words.append((word, clean, letter_count))
    except FileNotFoundError:
        st.error(f"Could not find {WORDS_FILE}.")
        return []

    return words


@st.cache_data
def load_dictionary():
    """Load and format local dictionary.json into a standardized map."""
    try:
        with open(DICTIONARY_FILE, "r", encoding="utf-8", errors="ignore") as f:
            raw = json.load(f)
    except FileNotFoundError:
        st.error(f"Could not find {DICTIONARY_FILE}.")
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
            dictionary[word] = {"definition": value, "part_of_speech": ""}
        elif isinstance(value, dict):
            definition = value.get("definition", "")
            if isinstance(definition, list):
                definition = " ".join(str(x) for x in definition if x)
            dictionary[word] = {
                "definition": str(definition),
                "part_of_speech": str(value.get("part_of_speech", ""))
            }
        else:
            dictionary[word] = {"definition": str(value), "part_of_speech": ""}

    return dictionary


@st.cache_data
def build_dictionary_search_index_with_idf(dictionary):
    """
    Builds a reverse index AND pre-calculates Inverse Document Frequency (IDF) 
    scores for every token in the dictionary.
    """
    index = {}
    doc_count = len(dictionary)
    doc_frequencies = Counter()

    for dict_word, info in dictionary.items():
        tokens = set(tokenize(info.get("definition", "")))
        for token in tokens:
            index.setdefault(token, set()).add(dict_word)
            doc_frequencies[token] += 1

    # IDF = log(Total Documents / Documents containing token)
    # Rare words (e.g., 'feline') get high IDF (~8-10)
    # Common words (e.g., 'animal', 'small') get low IDF (~2-3)
    idf_scores = {
        token: math.log(doc_count / count) 
        for token, count in doc_frequencies.items()
    }

    return index, idf_scores


# ============================================================
# SEARCH LOGIC
# ============================================================

def find_pattern_matches(pattern, words_data, dictionary, selected_length="Any"):
    regex_string = pattern_to_regex(pattern)
    try:
        regex = re.compile(regex_string)
    except re.error:
        return []

    matches = []
    for word, clean, letter_count in words_data:
        if selected_length != "Any" and letter_count != selected_length:
            continue

        if not regex.fullmatch(word):
            continue

        score = crossword_score(word, clean, dictionary)
        matches.append((word, score))

    matches.sort(key=lambda x: (-x[1], x[0]))
    return matches[:MAX_PATTERN_RESULTS]

# Common opposite pairs for semantic checks
ANTONYM_PAIRS = {
    "small": {"large", "huge", "giant", "enormous", "big", "great"},
    "tiny": {"large", "huge", "giant", "enormous", "big", "great"},
    "little": {"large", "huge", "giant", "enormous", "big", "great"},
    "large": {"small", "tiny", "little", "minute"},
    "big": {"small", "tiny", "little", "minute"},
    "hot": {"cold", "freezing", "chilly", "ice"},
    "cold": {"hot", "warm", "boiling"},
}

def search_dictionary_for_clue(
    clue, pattern, words_data, dictionary, index_tuple, selected_length="Any"
):
    dictionary_index, idf_scores = index_tuple
    clue_tokens = tokenize(clue)
    if not clue_tokens:
        return []

    num_clue_tokens = len(clue_tokens)
    total_clue_idf = sum(idf_scores.get(t, 1.0) for t in clue_tokens)

    # 1. Identify rare/descriptive terms in clue (IDF >= 4.5)
    rare_clue_tokens = [t for t in clue_tokens if idf_scores.get(t, 0) >= 4.5]

    # Candidate retrieval
    candidate_words = set()
    search_tokens = rare_clue_tokens if rare_clue_tokens else clue_tokens
    for token in search_tokens:
        if token in dictionary_index:
            candidate_words.update(dictionary_index[token])

    # Filter to valid crossword dictionary
    valid_crossword_set = {w[0].lower() for w in words_data}
    candidate_words &= valid_crossword_set

    # Filter by exact word length
    if selected_length != "Any":
        candidate_words = {
            w for w in candidate_words 
            if sum(ch.isalpha() for ch in w) == selected_length
        }

    # Filter by wildcard pattern
    if pattern.strip():
        try:
            regex = re.compile(pattern_to_regex(pattern), re.IGNORECASE)
            candidate_words = {w for w in candidate_words if regex.fullmatch(w)}
        except re.error:
            return []

    opposing_words = set()
    for token in clue_tokens:
        if token in ANTONYM_PAIRS:
            opposing_words.update(ANTONYM_PAIRS[token])

    clue_lower = clue.lower().strip()
    results = []

    for candidate in candidate_words:
        info = dictionary.get(candidate, {})
        definition = info.get("definition", "")
        if not definition:
            continue

        definition_tokens = tokenize(definition)
        def_length = len(definition_tokens)
        if def_length == 0:
            continue

        unique_def_tokens = set(definition_tokens)

        # --- MANDATORY COVERAGE RULES ---
        # 1. Must contain rare clue token if present
        if rare_clue_tokens and not any(rt in unique_def_tokens for rt in rare_clue_tokens):
            continue

        # 2. Count matched clue tokens
        matched_tokens = [t for t in clue_tokens if t in unique_def_tokens]
        matched_count = len(matched_tokens)
        matched_idf = sum(idf_scores.get(t, 1.0) for t in matched_tokens)

        # 3. Clue Concept Coverage Ratio (Count-based & IDF-based)
        token_coverage_ratio = matched_count / num_clue_tokens
        idf_coverage_ratio = matched_idf / total_clue_idf if total_clue_idf > 0 else 0

        # Require matching at least 50% of the clue tokens when clue has multiple words
        if num_clue_tokens >= 2 and token_coverage_ratio < 0.5:
            continue

        # --- SCORING ---
        # Base score driven directly by matched IDF weight
        score = matched_idf * 20.0

        # Smooth length penalty (prevents 2-word definitions like 'CATTISH' from getting 10x score)
        # Ideal definition length is ~8-25 words.
        length_factor = 1.0 / (1.0 + 0.15 * math.log(max(def_length, 1)))
        score *= length_factor

        # Multiply by coverage squared to heavily reward matching MORE clue words
        score *= (token_coverage_ratio ** 2)

        # Antonym / Contradiction penalty
        if opposing_words and any(opp in unique_def_tokens for opp in opposing_words):
            score *= 0.1

        # Proximity boost: clue tokens appear early in definition
        first_8 = set(definition_tokens[:8])
        early_matches = sum(1 for t in clue_tokens if t in first_8)
        if early_matches >= 2:
            score += 15.0

        # Substring / Exact match bonus
        if clue_lower in definition.lower():
            score += 35.0

        # Part of Speech / Noun priority heuristic (prefers concrete nouns)
        pos = info.get("part_of_speech", "").lower()
        if "noun" in pos or "n." in pos:
            score += 5.0
        elif "adj" in pos:
            score -= 5.0  # Penalize adjectives like CATTISH

        # Answer frequency boost
        score += frequency_score(candidate) * 0.5

        if score > 0:
            results.append((
                candidate.upper(),
                score,
                definition,
                info.get("part_of_speech", "")
            ))

    results.sort(key=lambda x: (-x[1], x[0]))
    return results[:MAX_CLUE_RESULTS]


# ============================================================
# INITIALIZE DATA
# ============================================================

words_data = load_words()
dictionary = load_dictionary()
dictionary_index = build_dictionary_search_index_with_idf(dictionary)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧩 Crossword Solver")
st.sidebar.markdown(f"**Words loaded:** {len(words_data):,}")
st.sidebar.markdown(f"**Dictionary entries:** {len(dictionary):,}")
st.sidebar.markdown("### Search rules")
st.sidebar.markdown(
    """
    `?` or `_` → unknown letter  
    `A-Z` → fixed letter  
    `-` → literal hyphen  
    `'` → literal apostrophe  
    """
)

if WORDFREQ_AVAILABLE:
    st.sidebar.success("Word-frequency ranking enabled")
else:
    st.sidebar.info("Install `wordfreq` for improved word commonness ranking.")


# ============================================================
# MAIN APPLICATION
# ============================================================

st.title("🧩 Crossword Solver")
st.caption("Pattern search + local dictionary clue search")

tab1, tab2 = st.tabs(["🔎 Pattern Search", "📖 Dictionary Clue Solver"])


# ------------------------------------------------------------
# TAB 1: PATTERN SEARCH
# ------------------------------------------------------------

with tab1:
    st.subheader("Pattern Search")

    # Wrap inputs inside a form
    with st.form("pattern_search_form"):
        pattern = st.text_input("Enter crossword pattern", placeholder="Example: A??E")
        selected_length = st.selectbox("Exact word length", options=["Any"] + list(range(3, 10)))
        
        # Form submit button replaces regular st.button
        search_clicked = st.form_submit_button("🔍 Search", type="primary")

    # 1. Update session state ONLY when the search button is pressed
    if search_clicked:
        if not pattern.strip():
            st.warning("Please enter a pattern.")
            st.session_state.pattern_results = []
            st.session_state.last_pattern = ""
        else:
            with st.spinner("Searching word list..."):
                st.session_state.pattern_results = find_pattern_matches(
                    pattern, words_data, dictionary, selected_length
                )
                st.session_state.last_pattern = pattern.upper()

    # 2. Always fetch results from session state (outside if search_clicked)
    results = st.session_state.get("pattern_results", [])
    last_pattern = st.session_state.get("last_pattern", "")

    # 3. Render results (outside if search_clicked)
    if results:
        st.markdown(f"**Top {len(results)} possibilities for `{last_pattern}`**")

        for rank, (word, score) in enumerate(results, start=1):
            col1, col2, col3 = st.columns([0.10, 0.30, 0.60])

            with col1:
                st.markdown(f"<div class='crossword-rank'>#{rank}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='crossword-word'>{word}</div>", unsafe_allow_html=True)
                st.caption(f"Score: {score:.1f}")
            with col3:
                info = dictionary.get(word.lower())
                if info:
                    with st.popover("Meaning"):
                        pos = info.get("part_of_speech")
                        if pos:
                            st.markdown(f"<div class='crossword-pos'>{pos}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='crossword-definition'>{info.get('definition', '')}</div>", unsafe_allow_html=True)
                else:
                    st.caption("Definition not available.")

    elif last_pattern:
        st.info("No matching words found.")

# ------------------------------------------------------------
# TAB 2: DICTIONARY CLUE SOLVER
# ------------------------------------------------------------

with tab2:
    st.subheader("Dictionary Clue Solver")
    st.caption("Searches local dictionary only.")

    # Wrap inputs inside an st.form to stop dropdowns from triggering reruns
    with st.form("dictionary_clue_form"):
        clue = st.text_input("Enter clue", placeholder="Example: Small domesticated feline", key="clue_input")
        clue_selected_length = st.selectbox(
            "Number of letters",
            options=["Any"] + list(range(3, 10)),
            index=0,
            key="clue_word_length"
        )
        clue_pattern = st.text_input("Optional pattern", placeholder="Example: C??", key="clue_pattern_input")

        # Form submit button
        clue_search_clicked = st.form_submit_button("📖 Search Dictionary", type="primary")

    # 1. Update session state ONLY when form is submitted
    if clue_search_clicked:
        if not clue.strip():
            st.warning("Please enter a clue.")
            st.session_state.clue_results = []
            st.session_state.last_clue = ""
        else:
            with st.spinner("Searching local dictionary..."):
                st.session_state.clue_results = search_dictionary_for_clue(
                    clue, clue_pattern, words_data, dictionary, dictionary_index, clue_selected_length
                )
                st.session_state.last_clue = clue.strip()

    # 2. Retrieve results from session state (persists across reruns)
    clue_results = st.session_state.get("clue_results", [])
    last_clue = st.session_state.get("last_clue", "")

    # 3. Render results outside if clue_search_clicked
    if clue_results:
        st.markdown(f"**Top {len(clue_results)} dictionary matches**")
        for rank, (word, score, definition, part_of_speech) in enumerate(clue_results, start=1):
            st.markdown(f"<div class='clue-result-word'>#{rank} &nbsp; {word}</div>", unsafe_allow_html=True)
            if part_of_speech:
                st.markdown(f"<div class='crossword-pos'>{part_of_speech}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='clue-result-definition'>{highlight_clue_matches(definition, last_clue)}</div>", unsafe_allow_html=True)
            st.caption(f"Match score: {score:.1f}")
            st.divider()

    elif last_clue:
        st.info("No strong dictionary matches found.")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("Word list: words.txt • Meanings: dictionary.json • Clue search: dictionary only")
