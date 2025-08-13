from flask import Flask, request, jsonify, render_template_string
import re

# Load word list
with open("words.txt") as f:
    WORD_LIST = [w.strip().lower() for w in f if w.strip()]

app = Flask(__name__)

def find_matches(pattern):
    regex = "^" + pattern.replace("_", ".").replace("?", ".") + "$"
    return [word for word in WORD_LIST if re.match(regex, word)]

# HTML Template
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Crossword Solver</title>
<style>
    body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
    input { padding: 8px; font-size: 16px; }
    button { padding: 8px 12px; font-size: 16px; }
    ul { list-style-type: none; padding: 0; margin-top: 20px; }
    li { font-size: 18px; }
</style>
</head>
<body>
<h1>Crossword Solver</h1>
<p>Enter a pattern (use _ or ? for unknown letters)</p>
<input id="pattern" placeholder="e.g. C_T">
<button onclick="solve()">Solve</button>
<ul id="results"></ul>

<script>
async function solve() {
    let pattern = document.getElementById("pattern").value;
    let res = await fetch("/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pattern })
    });
    let data = await res.json();
    document.getElementById("results").innerHTML =
        data.map(w => `<li>${w}</li>`).join("");
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

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # use PORT from Render, fallback 5000 locally
    app.run(host="0.0.0.0", port=port, debug=True)
