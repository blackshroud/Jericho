#!/usr/bin/env bash
set -euo pipefail

# Simple Linux helper to ensure system requirements, set up a venv, install deps,
# (byte-)compile, and run the app.
# Configurable via environment variables:
#   OLLAMA_HOST (default: http://localhost:11434)
#   OLLAMA_MODEL (default: llama3)
#   HOST (default: 127.0.0.1)
#   PORT (default: 8000)
#   RELOAD (default: 1; set to 0 to disable --reload)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# --- Package manager detection ---
PKG_MGR=""
if command -v apt-get >/dev/null 2>&1; then
  PKG_MGR="apt"
elif command -v dnf >/dev/null 2>&1; then
  PKG_MGR="dnf"
elif command -v yum >/dev/null 2>&1; then
  PKG_MGR="yum"
elif command -v pacman >/dev/null 2>&1; then
  PKG_MGR="pacman"
elif command -v zypper >/dev/null 2>&1; then
  PKG_MGR="zypper"
fi

confirm() {
  local prompt=${1:-"Proceed?"}
  read -r -p "$prompt [y/N] " reply || reply=""
  case "$reply" in
    [yY][eE][sS]|[yY]) return 0;;
    *) return 1;;
  esac
}

install_packages() {
  # Usage: install_packages pkg1 pkg2 ...
  if [[ -z "$PKG_MGR" ]]; then
    echo "No supported package manager detected. Please install: $*" >&2
    return 1
  fi
  if ! confirm "Missing requirement(s): $*. Install via $PKG_MGR?"; then
    return 1
  fi
  case "$PKG_MGR" in
    apt)
      sudo apt-get update && sudo apt-get install -y "$@"
      ;;
    dnf)
      sudo dnf install -y "$@"
      ;;
    yum)
      sudo yum install -y "$@"
      ;;
    pacman)
      sudo pacman -Syu --needed "$@"
      ;;
    zypper)
      sudo zypper install -y "$@"
      ;;
  esac
}

ensure_cmd() {
  # ensure_cmd <command> <package-names...>
  local cmd="$1"; shift || true
  if command -v "$cmd" >/dev/null 2>&1; then
    return 0
  fi
  install_packages "$@"
}

# --- Ensure core system requirements ---
# Python 3
ensure_cmd python3 \
  $( [[ "$PKG_MGR" == "pacman" ]] && echo python || echo python3 ) || true

# pip for Python 3
if ! command -v pip3 >/dev/null 2>&1; then
  case "$PKG_MGR" in
    apt) install_packages python3-pip || true ;;
    pacman) install_packages python-pip || true ;;
    dnf|yum|zypper) install_packages python3-pip || true ;;
  esac
fi

# venv support (Debian/Ubuntu needs python3-venv)
if ! python3 -m venv --help >/dev/null 2>&1; then
  if [[ "$PKG_MGR" == "apt" ]]; then
    install_packages python3-venv || true
  fi
fi

# curl (for optional checks / installs)
ensure_cmd curl curl || true

# --- Optional: Ollama availability ---
# If not reachable and 'ollama' CLI missing, offer to install via official script.
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
if ! command -v ollama >/dev/null 2>&1; then
  if ! curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "Ollama not found and ${OLLAMA_HOST} is unreachable."
    if confirm "Install Ollama via official script (curl -fsSL https://ollama.com/install.sh | sh)?"; then
      curl -fsSL https://ollama.com/install.sh | sh
      echo "Ollama installed. You may need to restart your shell/session."
    else
      echo "Skipping Ollama install. The app may not function without an Ollama server." >&2
    fi
  fi
fi

# Refresh command hash table in case new tools were added
hash -r || true

start_ollama() {
  # Try starting via systemd if available; otherwise, fallback to foreground server in background.
  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl enable --now ollama 2>/dev/null || sudo systemctl start ollama 2>/dev/null || true
  fi
  if ! curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    if command -v ollama >/dev/null 2>&1; then
      echo "Starting Ollama in the background... (logs: /tmp/ollama.log)"
      nohup ollama serve >/tmp/ollama.log 2>&1 &
    fi
  fi
}

wait_for_ollama() {
  local timeout="${OLLAMA_WAIT_TIMEOUT:-60}"
  echo -n "Waiting for Ollama at ${OLLAMA_HOST} (timeout: ${timeout}s) "
  for ((i=0; i<timeout; i++)); do
    if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      echo "OK"
      return 0
    fi
    echo -n "."
    sleep 1
  done
  echo ""
  echo "Timed out waiting for Ollama at ${OLLAMA_HOST}." >&2
  return 1
}

# Ensure Ollama is running (if local or reachable)
if ! curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  start_ollama
  wait_for_ollama || echo "Warning: Ollama is not reachable; the app may return 502." >&2
fi

# Ensure requested model exists; optionally pull
ensure_model() {
  local model="$1"
  # If local CLI exists, use it to check; otherwise rely on tags endpoint
  if command -v ollama >/dev/null 2>&1; then
    if ! ollama show "$model" >/dev/null 2>&1; then
      if [[ "${AUTO_PULL:-0}" == "1" ]] || confirm "Model '$model' not found. Pull it now?"; then
        ollama pull "$model" || {
          echo "Failed to pull model '$model'." >&2
          return 1
        }
      fi
    fi
  fi
}

# --- Create and activate venv ---
create_venv() {
  local py=${PYTHON:-python3}
  echo "Creating virtual environment in .venv using $py ..."
  if ! "$py" -m venv .venv >/dev/null 2>&1; then
    # Try to install venv support on Debian/Ubuntu
    if [[ "$PKG_MGR" == "apt" ]]; then
      echo "python3-venv may be missing; attempting to install..."
      install_packages python3-venv || true
    fi
    "$py" -m venv .venv
  fi
}

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" || ! -f "$VENV_DIR/bin/activate" ]]; then
  create_venv
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Error: failed to create virtual environment at $VENV_DIR." >&2
  echo "Ensure Python venv support is installed (e.g., python3-venv on Debian/Ubuntu) and rerun." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt

export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-1}"

# Interactive prompt for port (TTY-only, can skip with NO_PORT_PROMPT=1)
if [[ -t 0 && "${NO_PORT_PROMPT:-0}" != "1" ]]; then
  read -r -p "Port to use [${PORT}]: " _ans || _ans=""
  if [[ -n "$_ans" ]]; then
    if [[ "$_ans" =~ ^[0-9]+$ ]] && (( _ans > 0 && _ans < 65536 )); then
      PORT="$_ans"
    else
      echo "Invalid port '$_ans'; keeping ${PORT}" >&2
    fi
  fi
fi

# Ensure model exists (best-effort)
ensure_model "$OLLAMA_MODEL" || true

# Optional: byte-compile Python files
python -m compileall -q main.py || true

# Optional: quick reachability check to Ollama
if command -v curl >/dev/null 2>&1; then
  if ! curl -sSf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "Warning: Cannot reach Ollama at ${OLLAMA_HOST}. The API may fail." >&2
  fi
fi

if [[ "$RELOAD" == "1" ]]; then
  exec uvicorn main:app --host "$HOST" --port "$PORT" --reload
else
  exec uvicorn main:app --host "$HOST" --port "$PORT"
fi
    lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1
  else
    # Fallback: try to bind via Python
    "$PYTHON" - <<'PY' "$p" >/dev/null 2>&1 || return 0
import socket, sys
s = socket.socket()
try:
    s.bind(('127.0.0.1', int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
PY
    return 1
  fi
}

show_port_info() {
  local p="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp | grep -E ":${p} " || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$p" -sTCP:LISTEN || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$p" || true
  fi
}

kill_on_port() {
  local p="$1"
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids=$(lsof -t -iTCP:"$p" -sTCP:LISTEN || true)
    if [[ -n "$pids" ]]; then
      echo "Terminating PIDs on port $p: $pids"
      kill -TERM $pids 2>/dev/null || true
      sleep 1
      # Force kill if still alive
      for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
          echo "Force killing PID $pid"
          kill -9 "$pid" 2>/dev/null || sudo kill -9 "$pid" || true
        fi
      done
      return 0
    fi
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -k -n tcp "$p" 2>/dev/null || sudo fuser -k -n tcp "$p" || true
    return 0
  fi
  echo "Could not kill process on port $p automatically. Please free it manually." >&2
  return 1
}

kill_existing_uvicorn() {
  # Kill any uvicorn serving this app (best effort)
  if command -v pgrep >/dev/null 2>&1; then
    local pids
    pids=$(pgrep -f "uvicorn .*main:app" || true)
    if [[ -n "$pids" ]]; then
      echo "Found existing uvicorn process(es) for main:app: $pids"
      if [[ "${AUTO_KILL:-0}" == "1" ]] || confirm "Kill them now?"; then
        kill -TERM $pids 2>/dev/null || true
        sleep 1
        for pid in $pids; do
          if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing PID $pid"
            kill -9 "$pid" 2>/dev/null || sudo kill -9 "$pid" || true
          fi
        done
      fi
    fi
  fi
}

# Handle occupied port
if port_in_use "$PORT"; then
  echo "Port $PORT is in use on $HOST."
  show_port_info "$PORT"
  if [[ "${AUTO_FREE_PORT:-0}" == "1" ]]; then
    NEW_PORT=$("$PYTHON" - <<'PY'
import socket
s = socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
PY
)
    echo "AUTO_FREE_PORT=1 set; using free port: $NEW_PORT"
    PORT="$NEW_PORT"
  else
    echo "Options: [K]ill process, [C]hoose another port, [A]bort"
    read -r -p "Your choice [k/c/a]: " choice || choice="a"
    case "$choice" in
      [kK])
        kill_existing_uvicorn || true
        kill_on_port "$PORT" || true
        if port_in_use "$PORT"; then
          echo "Port $PORT still in use. Aborting." >&2
          exit 1
        fi
        ;;
      [cC])
        # Find a free port via Python
        NEW_PORT=$("$PYTHON" - <<'PY'
import socket
s = socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
PY
)
        echo "Using free port: $NEW_PORT"
        PORT="$NEW_PORT"
        ;;
      *)
        echo "Aborted due to occupied port." >&2
        exit 1
        ;;
    esac
  fi
fi

# Final preflight: ensure we can bind HOST:PORT; if not, auto-pick a free port
can_bind() {
  local host="$1" port="$2"
  "$PYTHON" - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket, sys
h, p = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind((h, p))
except OSError:
    raise SystemExit(1)
PY
}

if ! can_bind "$HOST" "$PORT"; then
  echo "Selected $HOST:$PORT cannot be bound; choosing a free port..."
  PORT=$("$PYTHON" - <<'PY'
import socket
s = socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
PY
)
  echo "Using free port: $PORT"
fi

echo "Starting server on http://$HOST:$PORT (reload=$RELOAD)"
# As a final safety, ensure no previous uvicorn remains
kill_existing_uvicorn || true

if [[ "$RELOAD" == "1" ]]; then
  exec uvicorn main:app --host "$HOST" --port "$PORT" --reload
else
  exec uvicorn main:app --host "$HOST" --port "$PORT"
fi
