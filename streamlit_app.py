import math
import json
import re
from collections import Counter
import streamlit as st
import nltk

# Auto-download WordNet datasets on Streamlit Cloud
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4.zip')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

from nltk.corpus import wordnet as wn

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
    """Build reverse search index and compute word IDF scores."""
    index = {}
    doc_count = len(dictionary)
    doc_freqs = Counter()

    for dict_word, info in dictionary.items():
        tokens = set(tokenize(info.get("definition", "")))
        for token in tokens:
            if token not in index:
                index[token] = set()
            index[token].add(dict_word)
            doc_freqs[token] += 1

    # Calculate Inverse Document Frequency (IDF) for all tokens
    idf_scores = {}
    for token, freq in doc_freqs.items():
        idf_scores[token] = math.log((doc_count + 1) / (freq + 1)) + 1.0

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
    "large": {"small", "little", "tiny", "miniature", "slight", "petty", "puny"},
    "big": {"small", "little", "tiny", "miniature", "slight"},
    "small": {"large", "big", "huge", "giant", "immense", "enormous", "great"},
    "tiny": {"large", "big", "huge", "giant", "immense", "enormous", "great"},
    "hot": {"cold", "freezing", "chilly", "icy", "cool"},
    "cold": {"hot", "warm", "boiling", "steaming"},
    "high": {"low", "short", "deep"},
    "low": {"high", "tall", "lofty"},
}

GENERIC_CLUE_WORDS = {
    "small", "large", "animal", "figure", "kind", "type", "part",
    "state", "quality", "act", "one", "thing", "make", "cause", "used"
}

COMPARISON_REGEX = re.compile(
    r'\b(size of a|resembling a|resembling the|like a|similar to|called also|allied to|type of)\b',
    re.IGNORECASE
)

def clean_definition_text(text: str) -> str:
    """Removes trailing quotes, author attributions, and bracketed notes."""
    # Remove quotation marks and text inside quotes
    cleaned = re.sub(r'"[^"]*"', '', text)
    # Strip bracketed usage tags like [Obs.] or [Zoöl.]
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    # Cut off text after common author citations or quotation indicators
    cleaned = re.split(r'\b(?:Shak|Bacon|Chaucer|Fuller|Milton|Dryden)\b', cleaned)[0]
    return cleaned.strip()

def get_dynamic_synonyms(word):
    """Dynamically fetches synonyms and lemmas using WordNet for ANY word."""
    synonyms = {word}
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    return synonyms

def get_most_important_token(clue_tokens, idf_scores):
    """Dynamically finds the most descriptive/rare token in any clue based on IDF."""
    if not clue_tokens:
        return None
    return max(clue_tokens, key=lambda t: idf_scores.get(t, 1.0))

# ============================================================
# HELPER FUNCTIONS (Must be defined BEFORE search_dictionary_for_clue)
# ============================================================

def split_into_senses(definition_text):
    """Splits multi-sense dictionary strings cleanly into individual definition senses."""
    if not definition_text:
        return []

    # Split on sense numbers (1., 2.), domain markers, or double hyphens
    raw_senses = re.split(
        r'(?:\r?\n+|\s*\b\d+\.\s*|\s*\b[A-Z]\.\s*|\s*\([a-z0-9\.\s]+\)\s*|\s*--\s*)',
        definition_text
    )
    
    clean_senses = []
    for sense in raw_senses:
        s = sense.strip()
        # Filter out short metadata fragments like "Zoöl." or "Astron."
        if len(s) > 10:
            clean_senses.append(s)

    return clean_senses if clean_senses else [definition_text.strip()]


def is_valid_direct_match(sense_text, concept_term):
    """Ensures concept terms aren't part of comparative clauses (e.g., 'resembling a...')."""
    lower_sense = sense_text.lower()
    if concept_term not in lower_sense:
        return False
    
    for match in COMPARISON_REGEX.finditer(lower_sense):
        end_idx = match.end() + 25
        if lower_sense.find(concept_term, match.start(), end_idx) != -1:
            return False
            
    return True


def search_dictionary_for_clue(
    clue, pattern, words_data, dictionary, index_tuple, selected_length="Any"
):
    if isinstance(index_tuple, tuple):
        dictionary_index, idf_scores = index_tuple
    else:
        dictionary_index, idf_scores = index_tuple, {}

    clue_tokens = tokenize(clue)
    if not clue_tokens:
        return []

    num_clue_tokens = len(clue_tokens)
    clue_lower = clue.lower().strip()

    # 1. Identify mandatory core keywords (non-generic tokens with highest IDF)
    rare_clue_tokens = [
        t for t in clue_tokens 
        if t not in GENERIC_CLUE_WORDS and idf_scores.get(t, 0.0) >= 4.5
    ]
    
    # If no rare tokens pass the threshold, fall back to the single highest-IDF token
    if not rare_clue_tokens and clue_tokens:
        rare_clue_tokens = [max(clue_tokens, key=lambda t: idf_scores.get(t, 0.0))]

    # 2. Gather candidates starting ONLY from the mandatory rare keywords
    candidate_words = set()
    for token in rare_clue_tokens:
        if token in dictionary_index:
            if not candidate_words:
                candidate_words = set(dictionary_index[token])
            else:
                # Intersect to require ALL rare keywords if multiple exist
                candidate_words &= dictionary_index[token]

    # Fallback to standard candidate gathering if intersection yields no results
    if not candidate_words:
        for token in clue_tokens:
            if token in dictionary_index:
                candidate_words.update(dictionary_index[token])

    valid_crossword_set = {w[0].lower() for w in words_data}
    candidate_words &= valid_crossword_set

    # Filter by exact letter count
    if selected_length != "Any":
        candidate_words = {
            w for w in candidate_words 
            if sum(ch.isalpha() for ch in w) == selected_length
        }

    # Filter by optional regex pattern
    if pattern.strip():
        try:
            regex = re.compile(pattern_to_regex(pattern), re.IGNORECASE)
            candidate_words = {w for w in candidate_words if regex.fullmatch(w)}
        except re.error:
            return []

    # Antonym tracking
    contradictory_terms = set()
    for token in clue_tokens:
        if token in ANTONYM_PAIRS:
            contradictory_terms.update(ANTONYM_PAIRS[token])

    results = []

    for candidate in candidate_words:
        info = dictionary.get(candidate, {})
        full_definition = info.get("definition", "")
        if not full_definition:
            continue

        senses = split_into_senses(full_definition)
        best_sense_score = 0.0
        best_sense_def = ""

        for sense in senses:
            cleaned_sense = clean_definition_text(sense)
            sense_tokens = tokenize(cleaned_sense)
            sense_token_set = set(sense_tokens)
            sense_length = len(sense_tokens)

            if sense_length == 0:
                continue

            # MANDATORY CHECK: Definition MUST contain all rare core keywords
            if rare_clue_tokens and not all(rt in sense_token_set for rt in rare_clue_tokens):
                continue

            # Antonym penalty
            if contradictory_terms and any(term in sense_token_set for term in contradictory_terms):
                continue

            matched_tokens = set(clue_tokens) & sense_token_set
            matched_count = len(matched_tokens)

            coverage = matched_count / num_clue_tokens
            if num_clue_tokens >= 2 and coverage < 0.5:
                continue

            # Calculate weighted IDF sum (generic tokens get reduced weight)
            matched_idf = 0.0
            for t in matched_tokens:
                weight = 0.5 if t in GENERIC_CLUE_WORDS else 2.0
                matched_idf += idf_scores.get(t, 1.0) * weight

            sense_score = matched_idf * 10.0 * (coverage ** 2)

            # Length normalization factor
            length_factor = 1.0 / (1.0 + 0.08 * math.log(max(sense_length, 1)))
            sense_score *= length_factor

            # Proximity bonus if core clue terms appear early in the sentence
            first_words = set(sense_tokens[:8])
            if rare_clue_tokens and any(rt in first_words for rt in rare_clue_tokens):
                sense_score += 15.0

            # Substring match bonus
            if clue_lower in cleaned_sense.lower():
                phrase_density = len(clue_lower) / max(len(cleaned_sense), 1)
                sense_score += (15.0 * phrase_density)

            if sense_score > best_sense_score:
                best_sense_score = sense_score
                best_sense_def = cleaned_sense

        if best_sense_score > 0:
            freq = frequency_score(candidate)
            best_sense_score += freq * 1.5

            pos = info.get("part_of_speech", "").lower()
            if "adj" in pos or "adv" in pos:
                best_sense_score *= 0.75

            results.append((
                candidate.upper(),
                round(best_sense_score, 1),
                best_sense_def,
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
