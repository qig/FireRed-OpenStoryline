#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

# Load project-level env when present (keeps API keys out of config.toml).
if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi
if [ -f "$ROOT_DIR/../../.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/../../.env"
  set +a
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7860}"

python -m open_storyline.mcp.server &
MCP_PID=$!

uvicorn agent_fastapi:app \
  --host "$HOST" \
  --port "$PORT" &
WEB_PID=$!

trap 'kill $MCP_PID $WEB_PID' INT TERM

wait
