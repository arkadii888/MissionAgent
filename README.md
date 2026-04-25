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

```bash
source .venv/bin/activate
uv sync
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

## Mission DSL pipeline (Gemma 4 E2B)

The orchestrator now uses a two-stage mission pipeline:

1. **NL -> intent DSL** (Gemma 4 E2B)
2. **Intent DSL -> mission points** (deterministic Python expansion)

The LLM does not compute latitude/longitude directly; mission points are computed from telemetry origin and cumulative offsets in `agent/orchestrator/mission_intents/`.

Naming conventions used in this repository for this flow:
- **Mission intent plan**: JSON object produced by Gemma (`mission_name` + `intents`).
- **Mission items / MissionItemList**: protobuf mission points sent to gRPC.
- **Model name** in env/tests: `gemma-4-e2b`.
- **Model file** default in llama server script: `gemma-4-E2B-it-Q4_K_M.gguf`.

### Currently supported mission intents

| Intent type | Required fields | What it does | Implemented in |
| --- | --- | --- | --- |
| `takeoff` | `altitude_m` | Adds a takeoff waypoint at telemetry origin with target relative altitude. | `agent/orchestrator/mission_intents/basic.py` |
| `move` | `north_m`, `east_m`, `up_m` | Updates cumulative north/east/altitude offsets and appends a fly-through waypoint with computed lat/lon. | `agent/orchestrator/mission_intents/basic.py` |
| `move_directional` | `direction` (`north/south/east/west/northeast/northwest/southeast/southwest`) | World-frame directional move. Supports compass synonyms and optional `distance_m` (default `10`). | `agent/orchestrator/mission_intents/basic.py` |
| `move_vertical` | `direction` (`down`) | Vertical descend move. Supports descend/down synonyms and optional `distance_m` (default `5`). | `agent/orchestrator/mission_intents/basic.py` |
| `turn_relative` | none (`type` only) | Turn-around behavior only in phase 1 (180 degrees). Emits a waypoint with updated yaw. | `agent/orchestrator/mission_intents/basic.py` |
| `safety_control` | `action` (`stop/hold/abort/return_home`) | Safety primitive; marks mission as preempted so subsequent movement/sweep intents are skipped (except `land`). | `agent/orchestrator/mission_intents/basic.py` |
| `comb_square_area` | none (`type` only) | Deterministic square comb/sweep pattern with optional `side_m`, `lane_spacing_m`, `altitude_m`, `start_corner`. | `agent/orchestrator/mission_intents/area_patterns.py` |
| `loiter` | `seconds` | Sets loiter duration on the latest waypoint (or creates a stationary waypoint if needed). | `agent/orchestrator/mission_intents/basic.py` |
| `yaw` | `degrees` | Stores yaw to apply to the next emitted waypoint. | `agent/orchestrator/mission_intents/basic.py` |
| `return_to_home` | none (`type` only) | Resets cumulative horizontal offsets to origin and appends return waypoint. | `agent/orchestrator/mission_intents/basic.py` |
| `land` | none (`type` only) | Appends final landing waypoint (`vehicle_action=2`). | `agent/orchestrator/mission_intents/basic.py` |

Phase-1 constraints:
- World-frame compass movement only (no drone-relative `forward/backward/left/right` parsing).
- `turn_relative` is intentionally limited to turn-around (180) semantics.
- `safety_control` acts as a preemption barrier for later movement/sweep intents.

### Mission-item defaults used during conversion

When intents are converted to protobuf mission items, these defaults are applied in `agent/orchestrator/mission_intents/proto.py` unless a handler overrides them:

| Field | Default / rule |
| --- | --- |
| `speed_m_s` | always `1.0` |
| `camera_action` | always `0` |
| `loiter_time_s` | default `1.0` (overridden by `loiter` intent) |
| `is_fly_through` | `true` for `move` and `return_to_home`; `false` for `takeoff`/`land` |
| `vehicle_action` | `1` for `takeoff`, `0` for normal move/return, `2` for `land` |
| `relative_altitude_m` | clamped to `[0, 100]` meters |
| `yaw_deg` | normalized to `[-360, 360]`; can be set via `yaw` intent |
| `gimbal_pitch_deg` / `gimbal_yaw_deg` | `NaN` |
| `camera_photo_interval_s` | `0.1` |
| `acceptance_radius_m` | `0.5` |
| `camera_photo_distance_m` | `NaN` |

Validation contract enforced before upload:
- latitude in `[-90, 90]`, longitude in `[-180, 180]`
- altitude in `[0, 100]`
- `speed_m_s == 1.0`
- `camera_action == 0`
- `vehicle_action in {0,1,2,3,4}`

### Run loop with local test mode

Use `agent/.env.orchestrator`:

```bash
LOCAL_TEST_MODE=1
MODEL_NAME=gemma-4-e2b
MISSION_JSON_LOG_ENABLED=1
MISSION_JSON_LOG_PATH=agent/logs/mission_pipeline.jsonl
```

Run:

```bash
uv sync
uv run python -m agent.orchestrator.loops
```

### Run loop with gRPC controller

Set:

```bash
LOCAL_TEST_MODE=0
GRPC_TARGET=127.0.0.1:50051
MODEL_NAME=gemma-4-e2b
```

Then run:

```bash
uv run python -m agent.orchestrator.loops
```

### JSON pipeline logs

When enabled (`MISSION_JSON_LOG_ENABLED=1`), each prompt writes JSONL records to:

- `agent/logs/mission_pipeline.jsonl` (or `MISSION_JSON_LOG_PATH`)

Events include:
- `prompt_received`
- `intents_generated`
- `intent_handler_called`
- `mission_converted`
- `mission_uploaded` / `mission_upload_failed`

`mission_converted` logs mission items in deterministic protobuf field order for easier diffing/debugging:
`latitude_deg`, `longitude_deg`, `relative_altitude_m`, `speed_m_s`, `is_fly_through`, `gimbal_pitch_deg`,
`gimbal_yaw_deg`, `camera_action`, `loiter_time_s`, `camera_photo_interval_s`, `acceptance_radius_m`,
`yaw_deg`, `camera_photo_distance_m`, `vehicle_action`.

Inspect quickly:

```bash
rg "mission_converted|mission_upload_failed" agent/logs/mission_pipeline.jsonl
```

## Tests for Mission DSL

Run all orchestrator tests:

```bash
uv run pytest -q agent/tests
```

Targeted:

```bash
uv run pytest -q agent/tests/test_mission_intents.py
uv run pytest -q agent/tests/test_json_pipeline_logging.py
uv run pytest -q agent/tests/test_llama_client_integration.py
```

## Add a new mission intent

To add a new intent:

1. Add a new `oneOf` schema branch in `agent/orchestrator/llm/schemas.py`.
2. Add a handler in `agent/orchestrator/mission_intents/` (for example `area_patterns.py`).
3. Register it in `agent/orchestrator/mission_intents/registry.py`.
4. Add/update few-shot examples in `agent/orchestrator/llm/prompts.py` so Gemma 4 E2B emits the new intent.
5. Add tests in `agent/tests/test_mission_intents.py` and, if needed, integration tests.