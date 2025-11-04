# Jericho (autogpt-web-ollama)

A thin FastAPI backend plus a modern, zero-build web UI for chatting with local LLMs.
Out of the box it supports:
- Ollama (local)
- GPT4All API servers (OpenAI-compatible; e.g., http://HOST:4891/v1)

Jericho provides selectable “agent” personas, friendly error guidance, a themeable chat UI, and a code editor pane that appears automatically when code is returned.

## Features
- Providers
  - Ollama via `/api/chat`
  - GPT4All via `/v1/chat/completions` with fallbacks to `/chat/completions` and `/v1/completions`
- Agents (personas)
  - Researcher, Analyst, Coder, Customer Service, Financial Planner (select from the “Agent” menu)
- Modern chat UI
  - Theme menu (Dark/Light)
  - Settings drawer: Host/Base URL, Temperature, Max tokens, Server timeout (0 = wait indefinitely)
  - Friendly error messages with tips (e.g., add `/v1` for GPT4All, pull models for Ollama)
  - Code editor pane (right side) with syntax highlighting (CodeMirror), Copy button, Close, animated, theme-aware; auto-opens when a code block (```lang) is detected
  - Chat history persists locally

## Prerequisites
- Python 3.10+
- One of:
  - Ollama running locally (default http://127.0.0.1:11434). Example model: `ollama pull llama3`
  - GPT4All API server (commonly http://127.0.0.1:4891/v1 or `http://LAN_IP:4891/v1`)

## Run (recommended)
```bash
./run.sh
```
What it does:
- Ensures Python, pip, venv, curl (and on Debian/Ubuntu, `python3-venv`) are present
- Optionally installs/starts Ollama and waits for readiness
- Prompts for a port and handles conflicts (auto-kill or auto-free-port)
- Creates/uses `.venv`, installs deps, and runs `uvicorn`

You can configure via env vars before running, e.g.:
```bash
export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3
PORT=9001 RELOAD=0 ./run.sh
```

## Manual setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000.

## Using the UI
- Pick an Agent from the “Agent” menu; Jericho greets you accordingly
- Choose Provider (Ollama or GPT4All) and set Host/Base URL in Settings
  - Examples:
    - Ollama: `http://127.0.0.1:11434`
    - GPT4All: `http://203.30.13.127:4891/v1` (note the `/v1`)
- Adjust Temperature, Max tokens, and Server timeout (sec)
  - Timeout `0` means “wait indefinitely” until the server responds or fails
- When Jericho returns code blocks, the code editor opens on the right; use Copy or Close

## API quick tests
- Ollama
```bash
curl -s http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' -d '{
    "provider":"ollama",
    "model":"llama3",
    "messages":[{"role":"user","content":"hello"}],
    "timeout_sec": 120
  }'
```
- GPT4All
```bash
curl -s http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' -d '{
    "provider":"gpt4all",
    "api_base":"http://203.30.13.127:4891/v1",
    "model":"Phi-3 Mini Instruct",
    "messages":[{"role":"user","content":"Who is Lionel Messi?"}],
    "max_tokens": 200,
    "timeout_sec": 0
  }'
```

## Troubleshooting
- Address already in use: `run.sh` prompts to kill/change port or auto-selects a free port
- HTTP 502: upstream not reachable/slow; verify provider host, model availability, and consider raising `timeout_sec` (or set to 0 for infinite wait)
- HTTP 404 with GPT4All: ensure Base URL includes `/v1` when the server expects it
- No content: some providers may return non-standard shapes; UI will show raw JSON if needed

## Notes
- Stateless backend; no database
- No streaming (requests are made with `stream: false`)
- Frontend is static (no build step); backend is FastAPI/uvicorn
