# Agent: detailed reference

Operational and developer reference for Agent. For installation, model downloads, `llama-server`, local demo mode, and the orchestrator quick start, see **[README.md](README.md)**.

## Contents

1. [llama-server tuning](#llama-server-tuning)
2. [Orchestrator loop behavior](#orchestrator-loop-behavior)
3. [Mission DSL pipeline](#mission-dsl-pipeline)
4. [JSON pipeline logs](#json-pipeline-logs)
5. [Vision: ArduCam and YOLO](#vision-arducam-and-yolo)
6. [Person rescue trigger](#person-rescue-trigger)
7. [Tests](#tests)
8. [Add a new mission intent](#add-a-new-mission-intent)
9. [gRPC connectivity check](#grpc-connectivity-check)

---

## llama-server tuning

Setup (submodule, models, `make run-llama-server`, Raspberry Pi prebuilt binary) is documented in [README.md](README.md).

`agent/scripts/run-llama-server.sh` sources `agent/.env.orchestrator` when present. Defaults:


| Variable      | Default                                   | Notes                                                |
| ------------- | ----------------------------------------- | ---------------------------------------------------- |
| `BINARY_PATH` | `agent/llama.cpp/build/bin/llama-server`  | Override for custom binary location                  |
| `MODEL_PATH`  | `agent/models/gemma-4-E2B-it-Q4_K_M.gguf` |                                                      |
| `PROJ_PATH`   | `agent/models/mmproj-F16.gguf`            | Omitted when `RESCUE_IMAGE_LLM_ENABLED=0`            |
| `PORT`        | `8080`                                    | Example override: `PORT=18090 make run-llama-server` |


Other knobs (`NGL`, `CTX_SIZE`, `BATCH`, `THREADS`, `TEMP`, `TOP_P`, `FLASH_ATTN`, etc.) are defined in the script. See `agent/scripts/run-llama-server.sh` for the full list.

**Health check:** with the server running, `GET http://127.0.0.1:8080/health` or `/v1/health` returns JSON such as `{"status":"ok"}`.

**Rebuild from source (desktop only):** `make clean-llama && make build-llama` removes `agent/llama.cpp/build` and recompiles. On Pi, use the prebuilt binary per README instead.

---

## Orchestrator loop behavior

Entry point: `agent/orchestrator/loops.py` (`make run-loops` or `python -m agent.orchestrator.loops`).

### gRPC mode

When `LOCAL_TEST_MODE` is off and `GRPC_TARGET` is set (see README):

1. Opens `InternalGrpcClient` and polls `GetTelemetry` in the background into `TelemetryCache`.
2. Polls `GetPrompt` every `PROMPT_POLL_INTERVAL_S` (default 1 s).
3. On a new non-empty prompt (deduplicated per process): fetches telemetry, runs `_plan_from_prompt` (LLM intent plan, deterministic expansion, validation), then `StartMission`.
4. Records home coordinates on the first `takeoff` intent (used by rescue).

The same prompt string is not processed twice in one run. Change the prompt on the controller side to request a new mission.

Tune polling with `PROMPT_POLL_INTERVAL_S`, `FOLLOW_POLL_INTERVAL_S`, and `TELEMETRY_POLL_HZ`.

### Local test mode

With `LOCAL_TEST_MODE=1`, prompts come from stdin, telemetry uses `LOCAL_TEST_`* env defaults, and `StartMission` is skipped. See README.

### Optional vision sidecar

When `ARDUCAM_VISION=1`, the loop also starts camera capture, Hailo YOLO inference, overlay recording, and (if gRPC is configured) the rescue trigger. See [Vision](#vision-arducam-and-yolo) and [Person rescue trigger](#person-rescue-trigger).

---

## Mission DSL pipeline

Two stages:

1. **Natural language to intent DSL** (Gemma 4 E2B via `llama-server`)
2. **Intent DSL to mission points** (deterministic Python in `agent/orchestrator/mission_intents/`)

The LLM never outputs latitude or longitude. Waypoints are computed from telemetry origin and cumulative offsets.

### Naming


| Term                              | Meaning                                     |
| --------------------------------- | ------------------------------------------- |
| Mission intent plan               | JSON from Gemma: `mission_name` + `intents` |
| Mission items / `MissionItemList` | Protobuf waypoints sent over gRPC           |
| `MODEL_NAME` (env)                | `gemma-4-e2b`                               |
| Model file (default)              | `gemma-4-E2B-it-Q4_K_M.gguf`                |


### Supported intents


| Intent type        | Required fields             | Behavior                                                                        | Module             |
| ------------------ | --------------------------- | ------------------------------------------------------------------------------- | ------------------ |
| `takeoff`          | `altitude_m`                | Takeoff waypoint at telemetry origin                                            | `basic.py`         |
| `move`             | `north_m`, `east_m`, `up_m` | Fly-through waypoint from cumulative offsets                                    | `basic.py`         |
| `move_directional` | `direction`                 | Compass move; optional `distance_m` (default 10)                                | `basic.py`         |
| `move_bearing`     | `distance_m`, `bearing_deg` | Move on bearing (0° = north, 90° = east)                                        | `basic.py`         |
| `move_vertical`    | `direction` (`down`)        | Descend; optional `distance_m` (default 5)                                      | `basic.py`         |
| `turn_relative`    | none                        | Turn-around (180°) only in phase 1                                              | `basic.py`         |
| `safety_control`   | `action`                    | Preempts later movement/sweep intents (not `land`)                              | `basic.py`         |
| `comb_square_area` | none                        | Square sweep; optional `side_m`, `lane_spacing_m`, `altitude_m`, `start_corner` | `area_patterns.py` |
| `loiter`           | `seconds`                   | Loiter on latest waypoint                                                       | `basic.py`         |
| `yaw`              | `degrees`                   | Yaw applied to next waypoint                                                    | `basic.py`         |
| `return_to_home`   | none                        | Return waypoint at origin offsets                                               | `basic.py`         |
| `land`             | none                        | Final land (`vehicle_action=2`)                                                 | `basic.py`         |


**Phase-1 constraints**

- World-frame compass movement only (no body-relative forward/back/left/right).
- `turn_relative` is limited to 180° turn-around semantics.
- `safety_control` blocks subsequent movement and sweep intents until `land`.

### Mission-item defaults

Applied in `mission_intents/proto.py` unless a handler overrides:


| Field                                 | Default / rule                                                     |
| ------------------------------------- | ------------------------------------------------------------------ |
| `speed_m_s`                           | `1.75` for takeoff/move handlers                                   |
| `camera_action`                       | `0`                                                                |
| `loiter_time_s`                       | `1.0` (overridden by `loiter`)                                     |
| `is_fly_through`                      | `true` for `move`, `return_to_home`; `false` for `takeoff`, `land` |
| `vehicle_action`                      | `1` takeoff, `0` move/return, `2` land                             |
| `relative_altitude_m`                 | clamped to `[0, 50]` m                                             |
| `yaw_deg`                             | normalized to `[-360, 360]`                                        |
| `gimbal_pitch_deg` / `gimbal_yaw_deg` | `NaN`                                                              |
| `camera_photo_interval_s`             | `0.1`                                                              |
| `acceptance_radius_m`                 | `0.5`                                                              |
| `camera_photo_distance_m`             | `NaN`                                                              |


### Validation before upload


| Check            | Rule                          |
| ---------------- | ----------------------------- |
| Latitude         | `[-90, 90]`                   |
| Longitude        | `[-180, 180]`                 |
| Altitude         | `[0, 50]` m                   |
| `speed_m_s`      | `≤ 4.0` (handlers use `1.75`) |
| `camera_action`  | `0`                           |
| `vehicle_action` | `{0, 1, 2, 3, 4}`             |


Layers: per-item `validate_proto_item`, list `validate_proto_list`, and `expand._validate_contract` before gRPC upload.

---

## JSON pipeline logs

Enable with `MISSION_JSON_LOG_ENABLED=1`. Output path: `MISSION_JSON_LOG_PATH` (default `agent/logs/mission_pipeline.jsonl`).

### Mission planning events


| Event                   | When                                                                 |
| ----------------------- | -------------------------------------------------------------------- |
| `prompt_received`       | New operator prompt                                                  |
| `intents_generated`     | LLM returned intent plan                                             |
| `intent_handler_called` | Per-intent expansion                                                 |
| `mission_converted`     | Protobuf list built                                                  |
| `mission_uploaded`      | `StartMission` succeeded                                             |
| `mission_upload_failed` | Failure (stage: `llm_plan`, `intent_expansion`, `start_mission_rpc`) |


`mission_converted` records fields in fixed order for diffing: `latitude_deg`, `longitude_deg`, `relative_altitude_m`, `speed_m_s`, `is_fly_through`, `gimbal_pitch_deg`, `gimbal_yaw_deg`, `camera_action`, `loiter_time_s`, `camera_photo_interval_s`, `acceptance_radius_m`, `yaw_deg`, `camera_photo_distance_m`, `vehicle_action`.

**Inspect:**

```bash
rg "mission_converted|mission_upload_failed" agent/logs/mission_pipeline.jsonl
```

Rescue-specific events are listed under [Person rescue trigger](#person-rescue-trigger).

---

## Vision: ArduCam and YOLO

Requires `ARDUCAM_VISION=1` in the orchestrator environment (not only in the llama-server env file). See `agent/.env.orchestrator` for examples.

### Startup sequence

1. `configure_vision_environment()` sets `YOLO_BACKEND=hailo` if unset.
2. `run_rpicam_health_check()` verifies `rpicam-*` / `libcamera-*`.
3. `CameraManager` (`camera_manager.py`): Picamera2 preview; dual main + lores when possible.
4. `DetectionManager` (`inference/detection_manager.py`): Hailo YOLO on a background thread.
5. `_OverlayRecorder` (`vision_sidecar.py`): annotated MP4 via `vision_overlay.py`.

Fail-fast if Picamera2 or the detector cannot start. Videos: `ARDUCAM_VIDEO_DIR` (default `/arducamvideos`), files `arducam_*.mp4`.

Picamera2 is not in `pyproject.toml`; on Pi install with `uv pip install "picamera2>=0.3.31"` or the OS package.

### Inference modules


| Module                 | Role                                                           |
| ---------------------- | -------------------------------------------------------------- |
| `detection_manager.py` | Hailo detector thread, optional JSONL (`DETECTION_JSONL_PATH`) |
| `yolo_common.py`       | `Detection` dataclass, letterbox, NMS                          |
| `yolo_hailo.py`        | `YoloHailoDetector` (HEF, tiling, multi-head layouts)          |
| `coco_names.py`        | COCO-80 class names                                            |
| `paths.py`             | Resolves `models/yolo26n_b8.hef` relative to repo layout       |


Hailo-only. Place HEF under `agent/models/` or set `YOLO_HEF_PATH`. Non-`hailo` `YOLO_BACKEND` values are ignored with a notice.

### Threading and color

- Camera thread updates `latest_frame` only.
- Detection thread drops frames if inference lags; always uses the newest buffer.
- Recorder reads `latest` detections and the preview frame; does not run inference.

**RGB vs BGR:** `CameraManager.buffer_is_bgr()` maps Picamera formats to OpenCV memory layout. `DetectionManager` converts BGR to RGB before `detect()` when needed. Override with `ARDUCAM_PIXEL_LAYOUT=bgr` or `rgb` if colors are wrong.

**Full-resolution inference:** `DETECTION_USE_MAIN_STREAM=1` with a dual stream runs Hailo on `main` while preview/recorder use lores. `overlay_scale_for_preview()` maps boxes to the overlay frame.

### Environment variables

**Core**


| Variable                    | Purpose                                    |
| --------------------------- | ------------------------------------------ |
| `YOLO_BACKEND`              | Only `hailo` is supported                  |
| `YOLO_HEF_PATH`             | HEF path (default `models/yolo26n_b8.hef`) |
| `DETECTION_USE_MAIN_STREAM` | `1` / `true`: infer on main stream         |
| `DETECTION_DEBUG_LOG`       | Per-second pipeline logs                   |
| `DETECTION_DEBUG_STATS`     | Frame mean/std in stats                    |
| `DETECTION_JSONL_PATH`      | Per-frame detection JSONL                  |


**Hailo post-process** (see `yolo_hailo.py` docstring)


| Variable                                                                                                        | Purpose                                            |
| --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `YOLO_NUMBER_TILES`                                                                                             | Tile grid size (default 1 = full frame)            |
| `YOLO_HAILO_FRAME_MODE`                                                                                         | Legacy label only; tiling uses `YOLO_NUMBER_TILES` |
| `PERSON_NMS_IOU`, `YOLO_MAX_CANDIDATES`, `YOLO_MAX_DETECTIONS`, `YOLO_PERSON_ONLY`, `YOLO_THREE_HEAD_BOX_SCALE` | Filtering                                          |


**Camera**


| Variable                                        | Purpose               |
| ----------------------------------------------- | --------------------- |
| `ARDUCAM_PIXEL_LAYOUT`                          | Force `bgr` or `rgb`  |
| `ARDUCAM_FORCE_RGB888` / `ARDUCAM_FORCE_BGR888` | Prefer pixel format   |
| `ARDUCAM_VIDEO_DIR`, `ARDUCAM_RECORD_FPS`       | Recording output      |
| `CAPTURE_SIZE`, `STREAM_DISPLAY_SIZE`           | Resolution            |
| `ARDUCAM_PERSON_CONF`, `ARDUCAM_PERSON_FRAMES`  | Person log thresholds |


### Detection thresholds


| Purpose                         | Variables                                      | Default                |
| ------------------------------- | ---------------------------------------------- | ---------------------- |
| Person `log.info` (recorder)    | `ARDUCAM_PERSON_CONF`, `ARDUCAM_PERSON_FRAMES` | `0.5`, `3`             |
| Rescue trigger                  | `RESCUE_PERSON_CONF`, `RESCUE_PERSON_FRAMES`   | `0.75`, `5`            |
| Minimum box score (all classes) | `conf_threshold` in `YoloHailoDetector`        | `0.25` (not env-wired) |


---

## Person rescue trigger

Active when `ARDUCAM_VISION=1` and a gRPC target is configured. Watches YOLO frames for a person and, after criteria are met, uploads a return-home-and-land mission.

### State machine

One-way transitions:

```
SUPPRESSED  ->  ACTIVE  ->  DISABLED
```


| State          | Behavior                                                                                                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **SUPPRESSED** | Ignores person detections until (1) first operator mission uploaded via `StartMission`, and (2) `RESCUE_ARM_DELAY_S` (default 60 s) elapsed. Avoids triggering on the operator at takeoff. |
| **ACTIVE**     | Qualifying frame: highest `person` score >= `RESCUE_PERSON_CONF` (default 0.75). Fires after `RESCUE_PERSON_FRAMES` (default 5) consecutive qualifying frames; a miss resets the counter.  |
| **DISABLED**   | Permanent after one fire. No second rescue mission per flight.                                                                                                                             |


### On trigger

Non-blocking on the vision thread:

1. **Photos** in `RESCUE_PHOTOS_DIR` (default `agent/arducamphotos/`): `*_full.jpg` (annotated frame), `*_person.jpg` (crop, 10 px margin).
2. **Rescue mission** via `StartMission`: `goto_lat_lon` to home at current altitude (floor `RESCUE_MIN_RTH_ALT_M`, default 10 m), then `land`.
3. **Gemma analysis** (async, if `llama-server` has `--mmproj`): cropped image plus estimated `forward_m` / `right_m`. Logs posture, health concern, rescuer action plan. Event: `rescue_analysis_completed`.

### Home location

First `takeoff` in the mission loop records telemetry lat/lon as home. Override with `RESCUE_HOME_LATITUDE_DEG` and `RESCUE_HOME_LONGITUDE_DEG` (both required).

### Person-position estimate

Flat-terrain pinhole projection from bbox center, `CAMERA_MOUNT_PITCH_DEG` (default 90 = nadir), and relative altitude:

- `forward_m`: metres ahead along heading
- `right_m`: metres to the right

World-frame lat/lon from heading is not implemented yet; see `agent/orchestrator/rescue/geo.py`.

### Rescue environment variables


| Variable                          | Default               | Description                                                  |
| --------------------------------- | --------------------- | ------------------------------------------------------------ |
| `RESCUE_IMAGE_LLM_ENABLED`        | `1`                   | `0`: skip Gemma image analysis and omit `--mmproj` on server |
| `RESCUE_PERSON_CONF`              | `0.75`                | Min person score per frame                                   |
| `RESCUE_PERSON_FRAMES`            | `5`                   | Consecutive qualifying frames                                |
| `RESCUE_ARM_DELAY_S`              | `60.0`                | Seconds after first mission before arming                    |
| `RESCUE_PHOTOS_DIR`               | `agent/arducamphotos` | Snapshot directory                                           |
| `RESCUE_MIN_RTH_ALT_M`            | `10.0`                | Min return altitude (m AGL)                                  |
| `RESCUE_HOME_LATITUDE_DEG`        | unset                 | Hard-coded home (pair with longitude)                        |
| `RESCUE_HOME_LONGITUDE_DEG`       | unset                 | Hard-coded home                                              |
| `RESCUE_HOME_RELATIVE_ALTITUDE_M` | `0.0`                 | Home altitude                                                |
| `CAMERA_MOUNT_PITCH_DEG`          | `90.0`                | Camera tilt (90 = nadir)                                     |
| `CAMERA_HFOV_DEG`                 | `66.0`                | Horizontal FOV                                               |
| `CAMERA_VFOV_DEG`                 | `41.0`                | Vertical FOV                                                 |


### Rescue JSON log events


| Event                            | Contents                                      |
| -------------------------------- | --------------------------------------------- |
| `rescue_plan_built`              | `goto_lat_lon` + `land` intent dict           |
| `rescue_mission_converted`       | Expanded protobuf                             |
| `rescue_mission_uploaded`        | `StartMission` OK                             |
| `rescue_person_offset_estimated` | `forward_m`, `right_m`, `relative_altitude_m` |
| `rescue_analysis_completed`      | Gemma response + offset                       |
| `rescue_mission_failed`          | Expansion or gRPC error                       |
| `rescue_analysis_failed`         | Gemma call error                              |


---

## Tests

Unit tests under `agent/tests` (no live model or gRPC required):

```bash
uv run pytest -q agent/tests
uv run pytest -q agent/tests/test_mission_intents.py
uv run pytest -q agent/tests/test_state.py
```

Add `-s` or drop `-q` for verbose output. End-to-end with a running `llama-server` uses the orchestrator loop (see README).

---

## Add a new mission intent

1. Add an `IntentSpec` to `INTENT_SPECS` in `mission_intents/intent_specs.py` (JSON Schema branch + handler registration).
2. Implement the handler in `mission_intents/` (e.g. `basic.py` or `area_patterns.py`).
3. Add few-shot examples in `llm/prompts.py` for Gemma.
4. Add tests in `agent/tests/test_mission_intents.py`.

`llm/schemas.py` imports the schema from intent specs; edit only for non-intent schema tweaks.

---

## gRPC connectivity check

After the controller is running and `GRPC_TARGET` is set (see README):

```bash
python -m agent.orchestrator.main
```

Prints one telemetry sample from `GetTelemetry`.