# Crossword Solver + AI (Flask)

A tiny web app that helps solve crosswords:
- **Pattern Search**: Enter a pattern like `C_T` → shows matches **with meanings**.
- **AI Clue Solver**: Enter a clue (e.g., `Feline pet (3)`) + optional pattern → gets likely answers **with meanings**.

## Demo (Local)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"   # optional for AI; app still runs without it
python app.py
