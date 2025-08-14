from flask import Flask, request, jsonify, render_template_string
import re
import os
import requests
import json
import time
from openai import OpenAI

# ======== CONFIGURE OPENAI ========
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ======== LOAD WORD LIST ========
with open("words.txt") as f:
    WORD_LIST = [w.strip().lower() for w in f if w.strip()]

# ======== IN-MEMORY CACHE ========
cache = {}  # { "clue|pattern": { "data": [...], "ts": timestamp } }
CACHE_TTL = 3600  # 1 hour

app = Flask(__name__)

# =========================
#   Helper Functions
# =========================
def find_matches(pattern):
    regex = "^" + pattern.replace("_", ".").replace("?", ".") + "$"
    return [word for word in WORD_LIST if re.match(regex, word)]

def get_meaning_from_api(word):
    """Fallback dictionary API meaning"""
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and "meanings" in data[0]:
                defs = data[0]["meanings"][0]["definitions"]
                if defs and "definition" in defs[0]:
                    return defs[0]["definition"]
    except Exception:
        pass
    return "No definition found"

def ai_guess_clue(clue, pattern=None):
    """Ask AI to guess crossword answer, return word + meaning with fallback dictionary API."""
    cache_key = f"{clue}|{pattern}"
    now = time.time()

    # Check cache
    if cache_key in cache and (now - cache[cache_key]["ts"]) < CACHE_TTL:
        return cache[cache_key]["data"]

    prompt = f"""You are a crossword puzzle solver.
Clue: "{clue}".
If a possible answer length is given in parentheses, respect it.
If a pattern is provided, only give answers that match it.
Respond ONLY in valid JSON array format like:
[{{"word": "cat", "meaning": "A small domesticated feline animal"}}]
Do not include any text outside JSON.
Pattern (optional): {pattern if pattern else "None"}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        ai_text = response.choices[0].message.content.strip()
        guesses = json.loads(ai_text)

        # Optional: Filter by pattern if AI ignored it
        if pattern:
            regex = "^" + pattern.replace("_", ".").replace("?", ".") + "$"
            guesses = [g for g in guesses if re.match(regex, g["word"].lower())]

        # Fallback: Get meaning from dictionary API if missing or too short
        for g in guesses:
            if not g.get("meaning") or len(g["meaning"].strip()) < 5:
                g["meaning"] = get_meaning_from_api(g["word"])

        # Store in cache
        cache[cache_key] = {"data": guesses, "ts": now}
        return guesses
    except Exception as e:
        return [{"word": "Error", "meaning": str(e)}]

# =========================
#   HTML TEMPLATE
# =========================
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Crossword Solver + AI</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
    body { font-family: Arial, sans-serif; text-align: center; margin: 0; padding: 20px; background: #f4f4f4; }
    h1 { color: #333; }
    p { color: #555; }
    input, button {
        padding: 10px;
        font-size: 16px;
        margin-top: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        width: 90%;
        max-width: 350px;
    }
    button {
        background-color: #007bff;
        color: white;
        border: none;
        cursor: pointer;
    }
    button:hover { background-color: #0056b3; }
    ul { list-style-type: none; padding: 0; margin-top: 20px; }
    li { font-size: 18px; background: white; margin: 5px auto; padding: 8px; border-radius: 5px; width: 90%; max-width: 350px; text-align: left; }
    #loading, #loadingAI { display: none; color: #007bff; font-weight: bold; margin-top: 15px; }
</style>
</head>
<body>
<h1>Crossword Solver</h1>

<!-- Pattern solver -->
<p><strong>Pattern Search</strong> (use _ or ? for unknown letters)</p>
<input id="pattern" placeholder="e.g. C_T">
<br>
<button onclick="solvePattern()">Solve Pattern</button>
<p id="loading">🔍 Searching...</p>
<ul id="results"></ul>

<hr style="margin: 30px 0;">

<!-- AI clue solver -->
<p><strong>AI Clue Solver</strong> (optional pattern to narrow down)</p>
<input id="clue" placeholder="e.g. Feline pet (3)">
<br>
<input id="ai_pattern" placeholder="Pattern (optional, e.g. C_T)">
<br>
<button onclick="solveClue()">Solve Clue with AI</button>
<p id="loadingAI">🤖 Thinking...</p>
<ul id="aiResults"></ul>

<script>
async function solvePattern() {
    let pattern = document.getElementById("pattern").value.trim();
    if (!pattern) { alert("Please enter a pattern"); return; }
    document.getElementById("loading").style.display = "block";
    document.getElementById("results").innerHTML = "";
    
    let res = await fetch("/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pattern })
    });
    let data = await res.json();
    document.getElementById("loading").style.display = "none";
    document.getElementById("results").innerHTML =
        data.length ? data.map(w => `<li>${w}</li>`).join("") : "<li>No matches found</li>";
}

async function solveClue() {
    let clue = document.getElementById("clue").value.trim();
    let pattern = document.getElementById("ai_pattern").value.trim();
    if (!clue) { alert("Please enter a clue"); return; }
    document.getElementById("loadingAI").style.display = "block";
    document.getElementById("aiResults").innerHTML = "";
    
    let res = await fetch("/solve_clue", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clue, pattern })
    });
    let data = await res.json();
    document.getElementById("loadingAI").style.display = "none";
    
    if (!data.length) {
        document.getElementById("aiResults").innerHTML = "<li>No guesses found</li>";
    } else {
        document.getElementById("aiResults").innerHTML =
            data.map(obj => `<li><strong>${obj.word}</strong> — ${obj.meaning}</li>`).join("");
    }
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/solve", methods=["POST"])
def solve():
    data = request.json
    pattern = data.get("pattern", "").lower()
    matches = find_matches(pattern)
    return jsonify(matches)

@app.route("/solve_clue", methods=["POST"])
def solve_clue():
    data = request.json
    clue = data.get("clue", "")
    pattern = data.get("pattern", None)
    guesses = ai_guess_clue(clue, pattern)
    return jsonify(guesses)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
