#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCHESTRATOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$ORCHESTRATOR_DIR/.." && pwd)"

ENV_FILE="${ENV_FILE:-$ORCHESTRATOR_DIR/.env.orchestrator}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  . "$ENV_FILE"
  set +a
fi

BINARY_PATH="${BINARY_PATH:-$ORCHESTRATOR_DIR/llama.cpp/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-$ORCHESTRATOR_DIR/models/gemma-4-E2B-it-Q4_K_M.gguf}"
PROJ_PATH="${PROJ_PATH:-$ORCHESTRATOR_DIR/models/mmproj-F16.gguf}"
PORT="${PORT:-8080}"
NGL="${NGL:-99}"
CTX_SIZE="${CTX_SIZE:-4096}"
BATCH="${BATCH:-1024}"
THREADS="${THREADS:-$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.ncpu)}"
PREDICT="${PREDICT:-1024}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-64}"
JINJA="${JINJA:-1}"
FLASH_ATTN="${FLASH_ATTN:-on}"
CNV="${CNV:-1}"
MLOCK="${MLOCK:-0}"
IMAGE_PATH="${IMAGE_PATH:-}"

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

if [ ! -f "$PROJ_PATH" ]; then
  echo "mmproj file not found: $PROJ_PATH"
  exit 1
fi

IMAGE_ARG=()
if [ -n "$IMAGE_PATH" ]; then
  IMAGE_ARG=(--image "$IMAGE_PATH")
fi

CMD=(
  "$BINARY_PATH"
  -m "$MODEL_PATH"
  --mmproj "$PROJ_PATH"
  "${IMAGE_ARG[@]}"
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
)

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
