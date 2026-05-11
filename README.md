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

- **One module:** `uv run pytest -q agent/tests/test_mission_intents.py` or `.../test_state.py`
- **Verbose output:** add `-s` to show prints, or drop `-q` for default verbosity.

End-to-end with a model: `python -m agent.orchestrator.loops` (see Mission test loop below); unit tests stay offline.

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

### ArduCam + YOLO vision (optional)

Set **`ARDUCAM_VISION=1`** to start vision alongside the mission loop (see `agent/orchestrator/loops.py`). That path runs, in order:

1. **`configure_vision_environment()`** — if you have not set **`YOLO_BACKEND`**, it defaults to **`hailo`** for compatibility with older env files (inference is Hailo-only).
2. **`run_rpicam_health_check()`** — quick `rpicam-*` / `libcamera-*` smoke test so the process fails fast if the camera stack is broken.
3. **`CameraManager`** (`agent/orchestrator/camera_manager.py`) — Picamera2 preview stream (dual **main + lores** when possible, else single stream). Frames are exposed as **`latest_frame`** under a lock.
4. **`DetectionManager`** (`agent/orchestrator/inference/detection_manager.py`) — loads the **Hailo** YOLO detector and runs a **background thread** that repeatedly grabs the latest frame, runs **`detector.detect(...)`**, and publishes **`latest`** detections (also under a lock) for the recorder and any other consumer.
5. **`_OverlayRecorder`** (`agent/orchestrator/vision_sidecar.py`) — another thread that copies the preview frame, scales boxes when inference used full-resolution **main** (see below), draws overlays via **`annotate_frame`** (`agent/orchestrator/vision_overlay.py`), and writes an MP4 with **`cv2.VideoWriter`**.

Startup is **fail-fast** if Picamera2 or the detector cannot initialize. Recordings go under **`ARDUCAM_VIDEO_DIR`** (default `/arducamvideos`) as timestamped **`arducam_*.mp4`**.

**Picamera2** is not in `pyproject.toml` (cross-platform lock); on a Pi install it in the venv, e.g. `uv pip install "picamera2>=0.3.31"`, or use the OS package—see **`agent/.env.orchestrator`**.

#### How the YOLO stack works

Inference code lives under **`agent/orchestrator/inference/`**:

| Module | Role |
| --- | --- |
| **`detection_manager.py`** | Loads **`YoloHailoDetector`**, resolves **`YOLO_HEF_PATH`**, starts the inference thread, keeps debug stats / optional JSONL history (`DETECTION_JSONL_PATH`). |
| **`yolo_common.py`** | **`Detection`** dataclass, letterbox / NMS helpers shared with the Hailo path. |
| **`yolo_hailo.py`** | **`YoloHailoDetector`**: HailoRT + **HEF**; supports end-to-end and multi-head HEF layouts; optional tiling via env (see table below). Returns **`Detection`** in image pixel space. |
| **`coco_names.py`** | Static COCO-80 names for class ids. |
| **`paths.py`** | Resolves relative paths like **`models/yolo26n_b8.hef`** by walking up from this package until the file exists (works with different clone layouts). |

Inference is **Hailo-only** (no ONNX runtime in this app). Put the **HEF** under **`agent/models/`** or set absolute **`YOLO_HEF_PATH`** (default relative **`models/yolo26n_b8.hef`**). Legacy **`YOLO_BACKEND`** values other than **`hailo`** are ignored with a console notice.

**Threading contract:** the camera thread only updates **`latest_frame`**. The detection thread **drops frames** if inference is slower than capture (it always processes the most recently acquired buffer for the chosen stream). The recorder thread reads **`latest`** and the same preview frame for drawing; it does not run the network.

**RGB vs BGR:** Picamera **`capture_array`** channel order does **not** always match the stream name. The code uses **`CameraManager.buffer_is_bgr()`** (see `camera_manager.py`): **`RGB888`** streams are treated as **OpenCV BGR** in memory; **`BGR888`** as **RGB** in memory, matching Picamera2’s internal buffer/PIL mapping. **`DetectionManager`** converts **BGR → RGB** before **`detect()`** when needed. For recording, frames are passed to **`VideoWriter`** in **BGR** order. If colors are wrong on your build, set **`ARDUCAM_PIXEL_LAYOUT=bgr`** or **`rgb`** to override.

**Full-resolution inference (optional):** set **`DETECTION_USE_MAIN_STREAM=1`** (and use a **dual** stream) so Hailo can run on **`capture_array("main")`** while the preview/recorder stays on **lores**. **`overlay_scale_for_preview()`** then supplies scale factors so **`vision_sidecar`** can map boxes onto the smaller overlay frame.

#### Environment variables (YOLO + vision)

**Orchestration / Hailo core**

| Variable | Purpose |
| --- | --- |
| **`YOLO_BACKEND`** | Optional; **`hailo`** is the only runtime. Other values are ignored (legacy). |
| **`YOLO_HEF_PATH`** | Hailo **HEF** file; default relative **`models/yolo26n_b8.hef`**. |
| **`DETECTION_USE_MAIN_STREAM`** | `1` / `true` — Hailo uses **main** full-res on dual stream; overlay scales to lores. |
| **`DETECTION_DEBUG_LOG`** | Per-second pipeline log lines. |
| **`DETECTION_DEBUG_STATS`** | Frame mean/std in the debug stats path. |
| **`DETECTION_JSONL_PATH`** | Append per-frame detection JSON lines. |

**Hailo-only (see docstring in `yolo_hailo.py` for detail)**

| Variable | Purpose |
| --- | --- |
| **`YOLO_HAILO_FRAME_MODE`** | Legacy label for **`get_runtime_info()`** / scheduling text; **tiling uses `YOLO_NUMBER_TILES` only**. |
| **`YOLO_NUMBER_TILES`** | Split each frame into an **N = rows×cols** grid (`N` from this int, default **1** = whole image). Layout picks `(rows, cols)` to match frame aspect. **`N` is clamped** per frame to at most **`ceil(W/model_w)×ceil(H/model_h)`** (model input size from the HEF, typically 640). Each tile is letterboxed to the model, then boxes are stitched with offsets + global NMS. |
| **`PERSON_NMS_IOU`**, **`YOLO_MAX_CANDIDATES`**, **`YOLO_MAX_DETECTIONS`**, **`YOLO_PERSON_ONLY`**, **`YOLO_THREE_HEAD_BOX_SCALE`** | Post-process / filtering knobs. |

**Camera / color (when `ARDUCAM_VISION=1`)**

| Variable | Purpose |
| --- | --- |
| **`ARDUCAM_PIXEL_LAYOUT`** | `bgr` or `rgb` — force buffer interpretation if autodetection is wrong. |
| **`ARDUCAM_FORCE_RGB888`** / **`ARDUCAM_FORCE_BGR888`** | Prefer one Picamera pixel format first when negotiating the stream. |

#### Thresholds for logging vs model scores

| What | Where | Default |
| --- | --- | --- |
| **Person** `log.info` after this confidence on **N** consecutive recorder frames | **`ARDUCAM_PERSON_CONF`**, **`ARDUCAM_PERSON_FRAMES`** | `0.5`, `3` |
| **Minimum class score** for a kept box (all classes) | **`conf_threshold`** in **`YoloHailoDetector`** (`0.25`); not env-wired | `0.25` |
| Verbose detection pipeline prints / logs | **`DETECTION_DEBUG_LOG=1`** | off |

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
| `move_bearing` | `distance_m`, `bearing_deg` | World-frame move on a compass bearing: clockwise from north (0°=north, 90°=east). | `agent/orchestrator/mission_intents/basic.py` |
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
| `relative_altitude_m` | clamped to `[0, 50]` meters |
| `yaw_deg` | normalized to `[-360, 360]`; can be set via `yaw` intent |
| `gimbal_pitch_deg` / `gimbal_yaw_deg` | `NaN` |
| `camera_photo_interval_s` | `0.1` |
| `acceptance_radius_m` | `0.5` |
| `camera_photo_distance_m` | `NaN` |

Validation contract enforced before upload:
- latitude in `[-90, 90]`, longitude in `[-180, 180]`
- altitude in `[0, 50]`
- `speed_m_s == 1.75`
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

## Add a new mission intent

To add a new intent:

1. Add an `IntentSpec` entry to `INTENT_SPECS` in `agent/orchestrator/mission_intents/intent_specs.py` (this defines the schema `enum`/`oneOf` branch and registers the handler with `build_default_registry()` via `expand.build_default_registry()`).
2. Implement the handler in `agent/orchestrator/mission_intents/` (for example `area_patterns.py` or `basic.py`).
3. Add/update few-shot examples in `agent/orchestrator/llm/prompts.py` so Gemma 4 E2B emits the new intent.
4. Add tests in `agent/tests/test_mission_intents.py`.

`agent/orchestrator/llm/schemas.py` pulls the JSON schema from the same specs; edit it only if you need non-intent tweaks (aliases, etc.).