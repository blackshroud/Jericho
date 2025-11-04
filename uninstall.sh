#!/usr/bin/env bash
set -euo pipefail

# Uninstall/cleanup helper for this repo (non-destructive to system packages/models).
# Actions:
#   - Stop uvicorn instances serving this app
#   - Optionally stop a user-launched `ollama serve` (not systemd service)
#   - Remove local venv (.venv) and caches (__pycache__, .pytest_cache, .mypy_cache)
# Flags/env:
#   --yes or FORCE=1       proceed without prompts
#   --keep-venv            do not remove .venv
#   --keep-caches          do not remove caches
#   --stop-ollama          try to stop user-launched `ollama serve`

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

FORCE=${FORCE:-0}
KEEP_VENV=0
KEEP_CACHES=0
STOP_OLLAMA=0

for arg in "${@:-}"; do
  case "${arg:-}" in
    --yes) FORCE=1 ;;
    --keep-venv) KEEP_VENV=1 ;;
    --keep-caches) KEEP_CACHES=1 ;;
    --stop-ollama) STOP_OLLAMA=1 ;;
  esac
done

confirm() {
  local prompt=${1:-"Proceed?"}
  if [[ "$FORCE" == "1" ]]; then return 0; fi
  read -r -p "$prompt [y/N] " reply || reply=""
  case "$reply" in [yY][eE][sS]|[yY]) return 0;; *) return 1;; esac
}

echo "[uninstall] Project: $PROJECT_ROOT"

kill_uvicorn_in_project() {
  local killed=0
  if command -v pgrep >/dev/null 2>&1; then
    local pids
    pids=$(pgrep -f "uvicorn .*main:app" || true)
    for pid in $pids; do
      local cwd
      cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
      if [[ "$cwd" == "$PROJECT_ROOT"* ]]; then
        echo "Stopping uvicorn PID $pid (cwd=$cwd)"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
          echo "Force killing PID $pid"
          kill -9 "$pid" 2>/dev/null || true
        fi
        killed=1
      fi
    done
  fi
  return $killed
}

# 1) Stop uvicorn for this project
kill_uvicorn_in_project || true

# 2) Optionally stop user-launched ollama serve (not touching systemd service)
if [[ "$STOP_OLLAMA" == "1" ]]; then
  if command -v pgrep >/dev/null 2>&1; then
    opp=$(pgrep -u "$USER" -f "ollama serve" || true)
    if [[ -n "$opp" ]]; then
      if confirm "Stop user-launched 'ollama serve' (PIDs: $opp)?"; then
        kill -TERM $opp 2>/dev/null || true
        sleep 1
        for pid in $opp; do
          if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
          fi
        done
        echo "Stopped ollama serve (user)"
      fi
    fi
  fi
fi

# 3) Remove virtualenv
if [[ "$KEEP_VENV" != "1" && -d .venv ]]; then
  if confirm "Remove virtual environment .venv?"; then
    rm -rf .venv
    echo "Removed .venv"
  fi
fi

# 4) Remove caches and compiled files
if [[ "$KEEP_CACHES" != "1" ]]; then
  if confirm "Remove Python caches (__pycache__, .pytest_cache, .mypy_cache) and *.pyc?"; then
    find "$PROJECT_ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
    rm -rf .pytest_cache .mypy_cache 2>/dev/null || true
    find "$PROJECT_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
    echo "Removed caches/pyc"
  fi
fi

# 5) Optional: remove temporary logs created by run.sh
if [[ -f /tmp/ollama.log ]]; then
  if confirm "Remove /tmp/ollama.log?"; then
    rm -f /tmp/ollama.log || true
    echo "Removed /tmp/ollama.log"
  fi
fi

echo "[uninstall] Done."
