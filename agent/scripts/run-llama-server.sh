#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$AGENT_DIR/.." && pwd)"

ENV_FILE="${ENV_FILE:-$AGENT_DIR/.env.orchestrator}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  . "$ENV_FILE"
  set +ar
fi

BINARY_PATH="${BINARY_PATH:-$AGENT_DIR/llama.cpp/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-$AGENT_DIR/models/gemma-4-E2B-it-Q4_K_M.gguf}"
PROJ_PATH="${PROJ_PATH:-$AGENT_DIR/models/mmproj-F16.gguf}"
# When 0/false/off, omit --mmproj (text-only server; matches orchestrator RESCUE_IMAGE_LLM_ENABLED=0).
RESCUE_IMAGE_LLM_ENABLED="${RESCUE_IMAGE_LLM_ENABLED:-1}"
PORT="${PORT:-8080}"
NGL="${NGL:-99}"
CTX_SIZE="${CTX_SIZE:-2048}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-}"
# macOS: perflevel0 logical CPUs; Linux/Pi: getconf/nproc (hw.ncpu is not set on Linux).
THREADS="${THREADS:-$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 4)}"
PREDICT="${PREDICT:-2048}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-64}"
JINJA="${JINJA:-1}"
FLASH_ATTN="${FLASH_ATTN:-on}"
CNV="${CNV:-1}"
MLOCK="${MLOCK:-0}"
REASONING="${REASONING:-off}"

if [ ! -x "$BINARY_PATH" ]; then
  echo "llama-server binary not found or not executable:"
  echo "  $BINARY_PATH"
  echo "Build it first with: make build-llama"
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "Model file not found: $MODEL_PATH"
  exit 1
fi

_load_mmproj=1
case "$(printf '%s' "${RESCUE_IMAGE_LLM_ENABLED}" | tr '[:upper:]' '[:lower:]')" in
  0|false|no|off) _load_mmproj=0 ;;
esac

if [ "$_load_mmproj" = 1 ]; then
  if [ ! -f "$PROJ_PATH" ]; then
    echo "mmproj file not found: $PROJ_PATH"
    exit 1
  fi
fi

CMD=(
  "$BINARY_PATH"
  -m "$MODEL_PATH"
  --port "$PORT"
  --n-gpu-layers "$NGL"
  -c "$CTX_SIZE"
  -b "$BATCH"
  -n "$PREDICT"
  --temp "$TEMP"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  -t "$THREADS"
  --flash-attn "$FLASH_ATTN"
  --reasoning "$REASONING"
)

if [ -n "$UBATCH" ]; then
  CMD+=(--ubatch-size "$UBATCH")
fi

if [ "$_load_mmproj" = 1 ]; then
  CMD+=(--mmproj "$PROJ_PATH")
else
  echo "RESCUE_IMAGE_LLM_ENABLED is off: starting llama-server without --mmproj (text chat only)."
fi

if [ "$JINJA" = "1" ]; then
  CMD+=(--jinja)
fi

if [ "$MLOCK" = "1" ]; then
  CMD+=(--mlock)
fi

echo "Starting llama.cpp server..."
echo "Project root: $PROJECT_ROOT"
echo "Command: ${CMD[*]}"
exec "${CMD[@]}"
