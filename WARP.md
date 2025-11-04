# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Commands

- Prerequisites
  - Python 3.10+
  - Ollama running locally (default http://localhost:11434); install a model, e.g.: `ollama pull llama3`

- Setup and run the dev server
  ```bash
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  uvicorn main:app --reload
  ```
  - Open http://127.0.0.1:8000

- Configuration via environment variables
  ```bash
export OLLAMA_HOST=http://127.0.0.1:11434   # change if Ollama runs elsewhere
  export OLLAMA_MODEL=llama3                  # default model override
  ```

- Quick API smoke test (after server is running)
  ```bash
  curl -s http://127.0.0.1:8000/api/chat \
    -H 'Content-Type: application/json' -d '{
      "model":"llama3",
      "messages":[{"role":"user","content":"hello"}],
      "stream":false
    }'
  ```

- Linting / Tests
  - No linter or test suite is configured in this repository.

## Architecture and structure

- Overview: Thin FastAPI backend that proxies to a local Ollama LLM, plus a minimal static web UI served from `web/`.

- Backend (FastAPI, `main.py`)
  - Configuration
    - `OLLAMA_HOST` (default `http://localhost:11434`)
    - `OLLAMA_MODEL` (default `llama3`)
  - Static files: `app.mount("/", StaticFiles(directory="web", html=True))` serves the UI at `/`.
  - API: `POST /api/chat`
    - Request body (Pydantic `ChatRequest`): `messages: list[dict]`, optional `model`, `stream` (unused; backend forces `stream: false`).
    - Behavior: forwards to `{OLLAMA_HOST}/api/chat` with `httpx.AsyncClient`, returns normalized JSON including `content` extracted from the last message if present; HTTP errors from Ollama map to `502`.

- Frontend (`web/index.html`)
  - Simple HTML + JS page; no build step.
  - Inputs: model text field, prompt textarea; button triggers `fetch('/api/chat', { method: 'POST', body: { model, messages, stream:false } })`.
  - Displays `data.content` if available, otherwise the raw JSON.

- Notes
  - There is no database or session state; the app is stateless and relies entirely on the Ollama API.
  - To change the default model without editing code, set `OLLAMA_MODEL`; to point at a remote Ollama instance, set `OLLAMA_HOST`.
