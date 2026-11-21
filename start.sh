#!/usr/bin/env bash
# Start, stop and watch the scalp trading agent and its dashboard.
#
#   ./start.sh              start both in the background
#   ./start.sh agent        start only the trading agent
#   ./start.sh dashboard    start only the dashboard
#   ./start.sh stop         stop both        (stop agent | stop dashboard)
#   ./start.sh restart      stop, then start both
#   ./start.sh status       what is running
#   ./start.sh logs         follow both logs (logs agent | logs dashboard)
#   ./start.sh link         show the dashboard login link
#   ./start.sh finetune check    can this computer fine-tune Kronos, and how long does a run take?
#   ./start.sh finetune status   candles collected and training runs so far
#   ./start.sh finetune train    start a fine-tuning run in the background (logs/finetune.log)
#   ./start.sh finetune export   write the training kit for a computer with a GPU (data/exports/)
#   ./start.sh finetune export --with-model   same, with the Kronos model inside (no Hugging Face needed)
#   ./start.sh finetune import <zip>   bring a model trained on that computer back
#   ./start.sh finetune versions       models saved on GitHub Releases
#   ./start.sh finetune save <run> [note]   keep a training run as the next version (GITHUB_TOKEN in .env)
#   ./start.sh finetune restore <n>    download version n into models/
#
# Works from any folder. Uses the virtualenv that is active, or venv/ or
# .venv/ in this folder or the one above it.
#
# Dashboard address (optional):
#   DASHBOARD_HOST=0.0.0.0 ./start.sh dashboard   # reachable from your phone on a trusted network
#   DASHBOARD_PORT=5001   ./start.sh dashboard
set -u

# a file argument (./start.sh finetune import <zip>) is relative to where you ran the script
FT_FILE="${3:-}"
case "$FT_FILE" in ""|/*|-*) ;; *) FT_FILE="$PWD/$FT_FILE" ;; esac
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")")" && pwd)"
cd "$ROOT" || exit 1

LOG_DIR="$ROOT/logs"
RUN_DIR="$ROOT/data"
DASHBOARD_HOST="${DASHBOARD_HOST:-127.0.0.1}"
DASHBOARD_PORT="${DASHBOARD_PORT:-5010}"   # 5010: runs beside the swing agent (5000)
mkdir -p "$LOG_DIR" "$RUN_DIR"

find_python() {
  local c
  for c in "${VIRTUAL_ENV:-}" "$ROOT/venv" "$ROOT/.venv" "$ROOT/../venv" "$ROOT/../.venv"; do
    if [ -n "$c" ] && [ -x "$c/bin/python" ]; then echo "$c/bin/python"; return 0; fi
  done
  return 1
}

PY="$(find_python)" || {
  echo "No Python virtualenv found. Activate yours first (source venv/bin/activate)," >&2
  echo "or create one in $ROOT/venv or $(dirname "$ROOT")/venv." >&2
  exit 1
}

pid_file() { echo "$RUN_DIR/$1.pid"; }
log_file() { echo "$LOG_DIR/$1.log"; }

running_pid() {   # prints the PID and succeeds if <name> is running
  local f pid
  f="$(pid_file "$1")"
  [ -f "$f" ] || return 1
  pid="$(cat "$f" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "$pid"; return 0; fi
  rm -f "$f"
  return 1
}

show_link() {
  local line
  line="$(grep -a "token=" "$(log_file dashboard)" 2>/dev/null | tail -1 | sed 's/^ *//')"
  if [ -n "$line" ]; then
    echo "  Dashboard login link (keep it private):"
    echo "  $line"
  else
    echo "  No login link yet - is the dashboard running? (./start.sh status)"
  fi
}

start_one() {
  local name="$1" pid
  shift
  if pid="$(running_pid "$name")"; then
    echo "  $name is already running (PID $pid)"
    return 0
  fi
  {
    echo
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') starting $name ====="
  } >> "$(log_file "$name")"
  nohup "$PY" -u "$@" >> "$(log_file "$name")" 2>&1 &
  pid=$!
  echo "$pid" > "$(pid_file "$name")"
  sleep 3
  if kill -0 "$pid" 2>/dev/null; then
    echo "  $name started (PID $pid) - log: logs/$name.log"
  else
    rm -f "$(pid_file "$name")"
    echo "  $name failed to start. Last lines of logs/$name.log:" >&2
    tail -n 15 "$(log_file "$name")" | sed 's/^/    /' >&2
    return 1
  fi
}

start_agent() {
  if [ ! -f "$ROOT/config/config.yaml" ]; then
    echo "  config/config.yaml is missing - copy config/config.example.yaml and edit it first." >&2
    return 1
  fi
  start_one agent main.py
}

start_dashboard() {
  start_one dashboard run_dashboard.py --host "$DASHBOARD_HOST" --port "$DASHBOARD_PORT" || return 1
  show_link
  if [ "$DASHBOARD_HOST" != "127.0.0.1" ] && [ "$DASHBOARD_HOST" != "localhost" ]; then
    echo "  Note: reachable from your network over plain HTTP. Use only on a network you trust."
  fi
}

stop_one() {
  local name="$1" pid i
  if ! pid="$(running_pid "$name")"; then
    echo "  $name is not running"
    return 0
  fi
  kill "$pid" 2>/dev/null
  for i in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "  $name (PID $pid) has not stopped after 20s. It may be finishing an order." >&2
    echo "  Check logs/$name.log, or force it with: kill -9 $pid" >&2
    return 1
  fi
  rm -f "$(pid_file "$name")"
  echo "  $name stopped"
}

status_one() {
  local name="$1" pid
  if pid="$(running_pid "$name")"; then
    echo "  $name: running (PID $pid, since $(ps -o lstart= -p "$pid" 2>/dev/null | sed 's/^ *//'))"
  else
    echo "  $name: stopped"
  fi
}

target="${2:-all}"
case "${1:-start}" in
  start|all) start_agent; a=$?; start_dashboard; d=$?; [ "$a" -eq 0 ] && [ "$d" -eq 0 ] ;;
  agent)     start_agent ;;
  dashboard) start_dashboard ;;
  stop)
    case "$target" in
      agent|dashboard) stop_one "$target" ;;
      all) stop_one dashboard; d=$?; stop_one agent; a=$?; [ "$a" -eq 0 ] && [ "$d" -eq 0 ] ;;
      *) echo "Usage: ./start.sh stop [agent|dashboard]" >&2; exit 2 ;;
    esac ;;
  restart)
    stop_one dashboard; stop_one agent || exit 1
    start_agent; a=$?; start_dashboard; d=$?; [ "$a" -eq 0 ] && [ "$d" -eq 0 ] ;;
  status)    status_one agent; status_one dashboard ;;
  link)      show_link ;;
  finetune)
    case "$target" in
      check|status) exec "$PY" -m src.finetune "$target" ;;
      export|import) exec "$PY" -m src.finetune "$target" ${FT_FILE:+"$FT_FILE"} ;;
      versions|save|restore) shift 2; exec "$PY" -m src.finetune "$target" "$@" ;;
      train)
        nohup "$PY" -u -m src.finetune train >> "$(log_file finetune)" 2>&1 &
        echo "  fine-tuning started (PID $!) - follow it with: tail -f logs/finetune.log" ;;
      *) echo "Usage: ./start.sh finetune check|status|train|export|import <zip>|versions|save <run> [note]|restore <n>" >&2; exit 2 ;;
    esac ;;
  logs)
    case "$target" in
      agent|dashboard) touch "$(log_file "$target")"; exec tail -n 40 -f "$(log_file "$target")" ;;
      all) touch "$(log_file agent)" "$(log_file dashboard)"
           exec tail -n 20 -f "$(log_file agent)" "$(log_file dashboard)" ;;
      *) echo "Usage: ./start.sh logs [agent|dashboard]" >&2; exit 2 ;;
    esac ;;
  -h|--help|help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "Unknown command: $1" >&2; sed -n '4,12p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
