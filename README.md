# MissionAgent

## llama.cpp build and server

The orchestrator uses [llama.cpp](https://github.com/ggml-org/llama.cpp) as a submodule. On macOS the build script enables Metal (`GGML_METAL=ON`).

### Prerequisites

- **Submodule:** clone with submodules, or after cloning run:

  ```bash
  git submodule update --init --recursive
  ```

- **Models:** place the GGUF model and mmproj under `orchestrator/models/`. The run script defaults to:
  - `orchestrator/models/gemma-4-E2B-it-Q4_K_M.gguf`
  - `orchestrator/models/mmproj-F16.gguf`  
  Set `MODEL_PATH` and `PROJ_PATH` if you use different files (see [Configuration](#configuration)).

### Build

From the repository root:

```bash
make build-llama
```

This runs `scripts/build-llama.sh`, which configures CMake and builds the `llama-server` binary. The output is:

`orchestrator/llama.cpp/build/bin/llama-server`

To remove the build tree and rebuild from scratch:

```bash
make clean-llama
make build-llama
```

### Run the HTTP server

From the repository root:

```bash
make run-llama-server
```

This executes `orchestrator/scripts/run-llama-server.sh`, which starts `llama-server` with the default model paths and options. The process listens on **port 8080** unless you change it (see below).

- **Health check:** with the server up, `GET http://127.0.0.1:8080/health` and `GET http://127.0.0.1:8080/v1/health` return JSON such as `{"status":"ok"}`.

### Configuration

Optional environment file: `orchestrator/.env.orchestrator`. If it exists, the run script sources it. You can set sampling, context, batch size, GPU layers (`NGL`), and other variables supported by the script. Commented examples are in that file for binary and model path overrides.

- **Port:** default is `8080` in the run script. To use another port for a single run:

  ```bash
  PORT=18090 make run-llama-server
  ```

  Environment variables you set in the shell are passed through to the run script. If you put `PORT=...` inside `.env.orchestrator`, that file’s value applies when the shell does not already set `PORT`.

For a full list of variables and their defaults, open `orchestrator/scripts/run-llama-server.sh`.

### Manual commands (without Make)

```bash
./scripts/build-llama.sh
./orchestrator/scripts/run-llama-server.sh
```
