#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/agent/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"

if [ ! -d "$LLAMA_DIR" ]; then
  echo "llama.cpp submodule not found."
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" -DGGML_METAL=ON
cmake --build "$BUILD_DIR" -j

echo "llama.cpp build complete."
echo "Server binary path:"
echo "  $BUILD_DIR/bin/llama-server"
