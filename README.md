# MissionAgent

## llama.cpp build and server

The agent uses [llama.cpp](https://github.com/ggml-org/llama.cpp) as a submodule.

### Prerequisites

- **Submodule:** clone with submodules, or after cloning run:

  ```bash
  git submodule update --init --recursive
  ```

- **Models:** place the GGUF model and mmproj under `agent/models/`. The run script defaults to:
  - `agent/models/gemma-4-E2B-it-Q4_K_M.gguf`
  - `agent/models/mmproj-F16.gguf`  
  Set `MODEL_PATH` and `PROJ_PATH` if you use different files (see [Configuration](#configuration)).

### Build

From the repository root:

```bash
make build-llama
```

This runs `scripts/build-llama.sh`, which configures CMake and builds the `llama-server` binary. The output is:

`agent/llama.cpp/build/bin/llama-server`

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

This executes `agent/scripts/run-llama-server.sh`, which starts `llama-server` with the default model paths and options. The process listens on **port 8080** unless you change it (see below).

- **Health check:** with the server up, `GET http://127.0.0.1:8080/health` and `GET http://127.0.0.1:8080/v1/health` return JSON such as `{"status":"ok"}`.

### Configuration

Optional environment file: `agent/.env.orchestrator`. If it exists, the run script sources it. You can set sampling, context, batch size, GPU layers (`NGL`), and other variables supported by the script. Commented examples are in that file for binary and model path overrides.

- **Port:** default is `8080` in the run script. To use another port for a single run:

  ```bash
  PORT=18090 make run-llama-server
  ```

For a full list of variables and their defaults, open `agent/scripts/run-llama-server.sh`.

### Test gRPC manually
Run 
```
source .venv/bin/activate
``` 
and 
```
uv sync
``` 
then 
```
python -m agent.orchestrator.main
```

## Tests (`agent/tests`)

From the repository root, install dependencies and run the orchestrator test suite with `uv` (or use an activated virtualenv and `pytest` on your `PATH`).

```bash
uv sync
uv run pytest -q agent/tests
```

- **Single file:** `uv run pytest -q agent/tests/test_state.py -s`
- **Verbose output:** add `-s` to show prints, or run without `-q` for default verbosity.

`agent/tests/test_llama_client_integration.py` talks to a running `llama-server` when it is up (e.g. `make run-llama-server` in another terminal). It may skip or behave differently if the server is not reachable. Set `LLAMA_CPP_URL` and `MODEL_NAME` if you use a non-default URL or model name (same as `agent/.env.orchestrator`).

## Mission test loop (gRPC + llama)

End-to-end integration: poll `GetPrompt` from your C++ side; when the prompt string changes, pull telemetry, call Gemma, upload a mission with `StartMission`, then keep telemetry and a simple mission progress model updated. Requires both services running (see `make run-llama-server` and your gRPC `InternalService`).

```bash
uv sync
# Optional: set -a; source agent/.env.orchestrator; set +a
export GRPC_TARGET=127.0.0.1:50051
uv run python -m agent.orchestrator.loops
```

From the repo root you can also run the file directly (the script adds the repo root to `sys.path`):

```bash
uv run python agent/orchestrator/loops.py
```

The same string is not processed twice in one run; change the prompt in C++ to request a new mission. Tune `PROMPT_POLL_INTERVAL_S` and `FOLLOW_POLL_INTERVAL_S` (seconds) and `TELEMETRY_POLL_HZ` if needed.