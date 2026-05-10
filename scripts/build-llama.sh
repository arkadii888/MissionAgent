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

OS="$(uname -s 2>/dev/null || echo unknown)"
CMAKE_EXTRA=()
case "$OS" in
  Darwin)
    # Apple GPU acceleration (macOS / iOS toolchains only)
    CMAKE_EXTRA+=(-DGGML_METAL=ON)
    ;;
  Linux)
    # CPU / OpenMP defaults from upstream; Metal is invalid on Linux
    ;;
  MINGW* | MSYS* | CYGWIN*)
    # Git Bash, MSYS2, Cygwin: same as Linux — no Metal
    ;;
  *)
    echo "Warning: unknown OS '$OS'; building without Metal. Use CMAKE_EXTRA_ARGS to add backends." >&2
    ;;
esac

# Optional: extra CMake flags, e.g. -DGGML_CUDA=ON
if [ -n "${CMAKE_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  CMAKE_EXTRA+=(${CMAKE_EXTRA_ARGS})
fi

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" "${CMAKE_EXTRA[@]}"

# Parallel job count (nproc: Linux/MSYS; sysctl: macOS)
if JOBS="$(nproc 2>/dev/null)"; then
  :
elif JOBS="$(sysctl -n hw.ncpu 2>/dev/null)"; then
  :
else
  JOBS=4
fi
cmake --build "$BUILD_DIR" --parallel "$JOBS"

echo "llama.cpp build complete."
echo "Server binary path:"
case "$OS" in
  MINGW* | MSYS* | CYGWIN*)
    echo "  $BUILD_DIR/bin/llama-server.exe"
    ;;
  *)
    echo "  $BUILD_DIR/bin/llama-server"
    ;;
esac
