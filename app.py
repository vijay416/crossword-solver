import os
import re
import time
import json
import threading
from functools import lru_cache
from flask import Flask, request, jsonify, render_template_string
import requests

# --- OpenAI (SDK v1) ---
try:
    from openai import OpenAI
    OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception:
    OPENAI_CLIENT = None  # If key/lib missing, we'll degrade gracefully.

app = Flask(__name__)

# =========================
#   Config
# =========================
# Cap pattern results to keep the page fast (tweak as you like)
PATTERN_RESULT_LIMIT = int(os.environ.get("PATTERN_RESULT_LIMIT", "50"))
# Cache TTL for AI results (seconds)
AI_CACHE_TTL = int(os.environ.get("AI_CACHE_TTL", "3600"))  # 1 hour

# =========================
#   Word list
# =========================
with open("words.txt", "r", encoding="utf-8") as f:
    WORD_LIST = [w.strip().lower() for w in f if w.strip()]

# =========================
#   Caches
# =========================
_ai_cache_lock = threading.Lock()
_ai_cache = {}  # { "clue|pattern": {"ts": epoch, "data": [ {word, meaning}, ... ] } }

_dict_cache_lock = threading.Lock()
_dict_cache = {}  # { "word": "meaning" }

# =========================
#   Helpers
# =========================
def _pattern_to_regex(pattern: str) -> str:
    # Support _ or ? as unknowns. Allow letters and hyphens/apostrophes common in crosswords.
    esc = re.escape(pattern.lower())
    esc = esc.replace("\\_", ".").replace("\\?", ".")
    return f"^{esc}$"

def find_matches_with_meanings(pattern: str):
    """Return [{word, meaning}] for words matching pattern (limited for performance)."""
    rx = re.compile(_pattern_to_regex(pattern))
    matches = [w for w in WORD_LIST if rx.match(w)]
    # keep first N to avoid hammering the dictionary API
    matches = matches[:PATTERN_RESULT_LIMIT]
    return [{"word": w, "meaning": get_meaning(w)} for w in matches]

def get_meaning(word: str) -> str:
    """Fetch definition with caching from Free Dictionary API."""
    wl = word.lower()
    with _dict_cache_lock:
        if wl in _dict_cache:
            return _dict_cache[wl]

    meaning = "Meaning not found."
    try:
        r = requests.get(
            f"https://api.dictionaryapi.dev/api/v2/entries/en/{wl}",
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data and "meanings" in data[0]:
                meanings = data[0]["meanings"]
                if meanings and "definitions" in meanings[0] and meanings[0]["definitions"]:
                    meaning = meanings[0]["definitions"][0].get("definition", meaning)
    except Exception:
        pass

    with _dict_cache_lock:
        _dict_cache[wl] = meaning
    return meaning

def ai_guess_clue(clue: str, pattern: str | None):
    """AI guesses (word + meaning). Uses OpenAI if available; falls back gracefully."""
    cache_key = f"{clue.strip()}|{(pattern or '').strip()}".lower()
    now = time.time()
    with _ai_cache_lock:
        entry = _ai_cache.get(cache_key)
        if entry and (now - entry["ts"]) < AI_CACHE_TTL:
            return entry["data"]

    guesses = []

    # If OpenAI not configured, skip to fallback
    if OPENAI_CLIENT is not None and os.environ.get("OPENAI_API_KEY"):
        prompt = f"""
You are an expert crossword solver.
Return up to 5 likely answers for the clue as a JSON array of objects:
[{{"word":"answer","meaning":"brief definition"}}]

Rules:
- If a pattern is provided, each answer MUST match it. Pattern uses "_" or "?" as unknowns.
- Respect given answer length if included in clue like "(5)".
- Keep "meaning" short, factual, dictionary-style.
- No extra commentary, ONLY valid JSON.

Clue: "{clue.strip()}"
Pattern: "{pattern.strip() if pattern else ''}"
"""
        try:
            resp = OPENAI_CLIENT.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            text = resp.choices[0].message.content.strip()
            guesses = json.loads(text)

            # Ensure structure
            if not isinstance(guesses, list):
                guesses = []
        except Exception as e:
            guesses = [{"word": "Error", "meaning": f"AI error: {str(e)}"}]

    # Filter by pattern (AI might ignore)
    if guesses and pattern:
        rx = re.compile(_pattern_to_regex(pattern))
        guesses = [g for g in guesses if isinstance(g, dict)
                   and "word" in g and rx.match(g["word"].lower())]

    # Fallback: if no AI or empty result, try pattern-only from dictionary
    if (not guesses) and pattern:
        guesses = find_matches_with_meanings(pattern)

    # Enrich meanings via dictionary for quality/consistency
    enriched = []
    for g in guesses:
        word = (g.get("word") or "").strip()
        if not word:
            continue
        meaning = g.get("meaning", "").strip()
        # Overwrite with dictionary meaning if missing/weak
        if len(meaning) < 5 or meaning.lower().startswith("ai error"):
            meaning = get_meaning(word)
        enriched.append({"word": word, "meaning": meaning})

    # Cache
    with _ai_cache_lock:
        _ai_cache[cache_key] = {"ts": now, "data": enriched}

    return enriched

# =========================
#   HTML (simple single-file UI)
# =========================
HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Crossword Solver + AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; background:#f6f7fb; margin:0; }
    .wrap { max-width: 680px; margin: 24px auto; padding: 0 16px; }
    h1 { text-align:center; margin: 16px 0 8px; }
    .card { background:#fff; border:1px solid #e6e8ef; border-radius:12px; padding:16px; box-shadow:0 1px 2px rgba(0,0,0,.03); margin-top:16px; }
    .row { display:flex; gap:8px; flex-wrap:wrap; }
    input, button {
      padding:10px; font-size:16px; border:1px solid #cfd4e2; border-radius:8px;
    }
    input { flex:1; min-width:220px; }
    button { cursor:pointer; }
    button.primary { background:#2d6cdf; color:#fff; border-color:#2d6cdf; }
    .hint { color:#6b7280; font-size:14px; margin:6px 0 0; }
    ul { list-style:none; padding:0; margin: 12px 0 0; }
    li { background:#fafbff; border:1px solid #eef0f6; padding:10px; border-radius:8px; margin:6px 0; }
    .word { font-weight:600; }
    .loading { display:none; font-weight:600; color:#2d6cdf; margin-top:8px; }
    hr { border:none; border-top:1px solid #eceef5; margin:20px 0; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Crossword Solver</h1>

    <div class="card">
      <h3>Pattern Search</h3>
      <div class="row">
        <input id="pattern" placeholder="e.g. C_T or ----e-">
        <button class="primary" onclick="solvePattern()">Find words</button>
      </div>
      <p class="hint">Use _ or ? for unknown letters. Shows meanings for each match.</p>
      <div id="loadingPattern" class="loading">🔍 Searching…</div>
      <ul id="patternResults"></ul>
    </div>

    <div class="card">
      <h3>AI Clue Solver</h3>
      <div class="row">
        <input id="clue" placeholder='e.g. "Feline pet (3)"'>
        <input id="ai_pattern" placeholder="Optional pattern, e.g. C_T">
        <button class="primary" onclick="solveClue()">Solve clue</button>
      </div>
      <p class="hint">Returns likely answers with concise meanings. Uses pattern if provided.</p>
      <div id="loadingAI" class="loading">🤖 Thinking…</div>
      <ul id="aiResults"></ul>
    </div>

    <p class="hint">Tip: For large word lists, results are capped to keep it fast (env: PATTERN_RESULT_LIMIT).</p>
  </div>

<script>
async function solvePattern() {
  const pattern = document.getElementById("pattern").value.trim();
  if (!pattern) { alert("Please enter a pattern"); return; }
  document.getElementById("loadingPattern").style.display = "block";
  document.getElementById("patternResults").innerHTML = "";

  const res = await fetch("/solve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pattern })
  });
  const data = await res.json();
  document.getElementById("loadingPattern").style.display = "none";

  if (!data.length) {
    document.getElementById("patternResults").innerHTML = "<li>No matches found</li>";
    return;
  }
  document.getElementById("patternResults").innerHTML =
    data.map(o => `<li><span class="word">${o.word}</span> — ${o.meaning}</li>`).join("");
}

async function solveClue() {
  const clue = document.getElementById("clue").value.trim();
  const pattern = document.getElementById("ai_pattern").value.trim();
  if (!clue) { alert("Please enter a clue"); return; }

  document.getElementById("loadingAI").style.display = "block";
  document.getElementById("aiResults").innerHTML = "";

  const res = await fetch("/solve_clue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clue, pattern })
  });
  const data = await res.json();
  document.getElementById("loadingAI").style.display = "none";

  if (!data.length) {
    document.getElementById("aiResults").innerHTML = "<li>No guesses found</li>";
    return;
  }
  document.getElementById("aiResults").innerHTML =
    data.map(o => `<li><span class="word">${o.word}</span> — ${o.meaning}</li>`).join("");
}
</script>
</body>
</html>
"""

# =========================
#   Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/solve", methods=["POST"])
def solve_pattern():
    data = request.get_json(force=True)
    pattern = (data.get("pattern") or "").strip()
    if not pattern:
        return jsonify([])

    results = find_matches_with_meanings(pattern)
    return jsonify(results)

@app.route("/solve_clue", methods=["POST"])
def solve_clue():
    data = request.get_json(force=True)
    clue = (data.get("clue") or "").strip()
    pattern = (data.get("pattern") or "").strip() or None
    if not clue:
        return jsonify([])

    results = ai_guess_clue(clue, pattern)
    return jsonify(results)

# =========================
#   Entrypoint
# =========================
if __name__ == "__main__":
    # Local dev server; on Render/Heroku use Gunicorn via Procfile
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
