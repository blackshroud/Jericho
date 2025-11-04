# autogpt-web-ollama

A minimal web-based AutoGPT-like interface that uses a locally hosted Ollama LLM.

## Prerequisites
- Python 3.10+
- Ollama running locally (default http://localhost:11434)
  - Install a model, e.g.: `ollama pull llama3`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000 and chat.

## Notes
- The backend proxies to Ollama’s `/api/chat` with `stream: false` for simplicity.
- Adjust the default model in `main.py` or pass `model` in the request body.
