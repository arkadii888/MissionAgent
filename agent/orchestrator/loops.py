"""Mission planning loop: telemetry, operator prompt, LLM intents, waypoint expansion, optional gRPC upload."""

import asyncio
import contextlib
import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent.orchestrator.config import Settings
from agent.orchestrator.grpc_client import InternalGrpcClient
from agent.orchestrator.llm.client import LlamaClient
from agent.orchestrator.llm.prompts import build_system_prompt, build_user_prompt
from agent.orchestrator.logging import JsonPipelineLogger, log_mission_multipoint_geojson
from agent.orchestrator.mission_intents import expand_intents_to_mission
from agent.orchestrator.mission_intents.proto import mission_list_to_ordered_dict
from agent.orchestrator.rescue.dispatcher import RescueMissionDispatcher
from agent.orchestrator.rescue.home_state import HomeLocationState
from agent.orchestrator.state import MissionState, TelemetryCache
from agent.orchestrator.vision_sidecar import (
    arducam_vision_enabled,
    start_arducam_vision,
    stop_arducam_vision,
)

log = logging.getLogger(__name__)

_DEFAULT_PROMPT_POLL_S = 1
_LOCAL_TEST_TELEMETRY_DEFAULTS: dict[str, float] = {
    "latitude_deg": 47.3977419,
    "longitude_deg": 8.5455938,
    "relative_altitude_m": 0.0,
    "absolute_altitude_m": 488.0,
}


def _load_settings() -> Settings:
    """Environment-backed settings with sane defaults for llama URL, model, and gRPC."""
    base = Settings.from_env()
    out = base
    if out.grpc_target is None:
        out = replace(out, grpc_target="localhost:50051")
    if out.llama_cpp_url is None:
        out = replace(out, llama_cpp_url="http://127.0.0.1:8080")
    if out.model_name is None:
        out = replace(out, model_name="gemma-4-e2b")
    return out


def _env_flag(name: str, default: bool = False) -> bool:
    """True if ``name`` is set to a common truthy string (``1``, ``true``, ``yes``, ``on``)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_local_test_telemetry() -> dict[str, float]:
    """Baseline telemetry map; overridden by ``LOCAL_TEST_*`` env vars when set."""
    out = dict(_LOCAL_TEST_TELEMETRY_DEFAULTS)
    for key, default_value in _LOCAL_TEST_TELEMETRY_DEFAULTS.items():
        raw = os.getenv(f"LOCAL_TEST_{key.upper()}")
        if raw is None or not raw.strip():
            continue
        try:
            out[key] = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid float for LOCAL_TEST_{key.upper()}: {raw!r}") from exc
    return out


async def _plan_from_prompt(
    *,
    llm: LlamaClient,
    mission: MissionState,
    settings: Settings,
    prompt_text: str,
    telemetry_map: dict[str, Any],
    json_logger: JsonPipelineLogger,
    trace_id: str,
    home_state: HomeLocationState,
    rescue_dispatcher: RescueMissionDispatcher | None = None,
) -> tuple[str, Any, str] | None:
    """Run LLM then expand JSON intents to protobuf mission items.

    Returns:
        ``(mission_name, MissionItemList, trace_id)`` on success, or ``None`` if planning failed.

    Side effects:
        Updates ``mission`` phase on error; logs pipeline events to ``json_logger``.
        Calls ``home_state.set_once`` when the first ``takeoff`` intent is processed.
    """
    status_line = await mission.prompt_mission_status()
    await mission.begin_planning()
    system = build_system_prompt(max_waypoints=settings.max_waypoints)
    user = build_user_prompt(
        user_prompt=prompt_text,
        telemetry=telemetry_map,
        mission_status=status_line,
    )
    log.info("LLM system prompt sent to llama-server:\n%s", system)
    log.info("LLM user prompt sent to llama-server:\n%s", user)

    try:
        plan_dict: dict[str, Any] = await llm.plan_mission(system, user)
    except Exception as exc:
        log.exception("LLM plan_mission failed: %s", exc)
        json_logger.log(
            "mission_upload_failed",
            trace_id,
            {"stage": "llm_plan", "error": str(exc)},
        )
        await mission.mark_error(f"llm: {exc}")
        return None

    json_logger.log(
        "intents_generated",
        trace_id,
        {"mission_plan": plan_dict},
    )
    log.info(
        "LLM parsed mission intent plan (dict after JSON parse):\n%s",
        json.dumps(plan_dict, indent=2, ensure_ascii=False),
    )

    try:
        called_handlers: list[dict[str, Any]] = []

        def _on_handler_called(intent_type: str, intent_payload: dict[str, Any]) -> None:
            called_handlers.append(
                {
                    "intent_type": intent_type,
                    "handler_input": dict(intent_payload),
                }
            )
            json_logger.log(
                "intent_handler_called",
                trace_id,
                {"intent_type": intent_type, "intent": dict(intent_payload)},
            )
            # Capture first-takeoff lat/lon as home for rescue RTL.
            if intent_type == "takeoff":
                home_state.set_once(
                    latitude_deg=float(telemetry_map.get("latitude_deg", 0.0)),
                    longitude_deg=float(telemetry_map.get("longitude_deg", 0.0)),
                    relative_altitude_m=float(telemetry_map.get("relative_altitude_m", 0.0)),
                )

        proto = expand_intents_to_mission(
            plan_dict,
            telemetry_map,
            on_handler_called=_on_handler_called,
        )
        json_logger.log(
            "mission_converted",
            trace_id,
            {
                "mission_plan": plan_dict,
                "called_handlers": called_handlers,
                "mission_proto": mission_list_to_ordered_dict(proto),
            },
        )
        log_mission_multipoint_geojson(json_logger, trace_id, proto)
    except Exception as exc:
        log.exception("expand_intents_to_mission: %s", exc)
        json_logger.log(
            "mission_upload_failed",
            trace_id,
            {"stage": "intent_expansion", "error": str(exc), "mission_plan": plan_dict},
        )
        await mission.mark_error(f"map: {exc}")
        return None

    name = str(plan_dict.get("mission_name", "mission"))[:64]
    return name, proto, trace_id


async def _telemetry_poll_loop(
    client: InternalGrpcClient,
    cache: TelemetryCache,
    period_s: float,
    stop: asyncio.Event,
) -> None:
    """Refresh ``cache`` from ``GetTelemetry`` every ``period_s`` until ``stop`` is set."""
    while not stop.is_set():
        try:
            t = await client.get_telemetry()
            await cache.update_from_telemetry(t)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("GetTelemetry failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=period_s)
        except asyncio.TimeoutError:
            pass


async def run_mission_test_loop() -> None:
    """Run until interrupt: poll prompts, plan missions, optionally upload via gRPC.

    With ``LOCAL_TEST_MODE`` set, reads each mission prompt from stdin (blocking ``input``) and
    skips ``StartMission``. Non-interactive use: pipe or redirect stdin (e.g. ``echo '...' | uv run ...``).
    Otherwise opens gRPC and sends each new prompt string as one mission.

    Repeated identical prompts in gRPC mode are ignored for one process lifetime so a failing
    plan does not spin forever.
    """
    settings = _load_settings()
    prompt_interval = float(os.getenv("PROMPT_POLL_INTERVAL_S", str(_DEFAULT_PROMPT_POLL_S)))
    period_telemetry = 1.0 / max(settings.telemetry_poll_hz, 0.1)

    if not settings.llama_cpp_url or not settings.model_name:
        raise ValueError("LLAMA_CPP_URL and MODEL_NAME must be set in the environment or defaults")

    llm = LlamaClient(
        base_url=settings.llama_cpp_url,
        model_name=settings.model_name,
        timeout_s=settings.llm_timeout_s,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )

    cache = TelemetryCache()
    mission = MissionState()
    stop = asyncio.Event()
    json_logger = JsonPipelineLogger(
        path=Path(settings.mission_json_log_path),
        enabled=settings.mission_json_log_enabled,
    )
    home_state = HomeLocationState()
    local_test_mode = _env_flag("LOCAL_TEST_MODE")
    last_seen_prompt: list[str | None] = [None]

    log.info(
        "Starting mission test loop: grpc=%s llama=%s prompt_poll=%.2fs local_test_mode=%s arducam=%s",
        settings.grpc_target,
        settings.llama_cpp_url,
        prompt_interval,
        local_test_mode,
        arducam_vision_enabled(),
    )

    vision_rt = None
    if arducam_vision_enabled():
        log.info("Starting ArduCam vision pipeline (fail-fast on errors)...")
        try:
            vision_rt = await asyncio.to_thread(start_arducam_vision)
        except Exception:
            log.exception("ArduCam vision startup failed")
            raise SystemExit(1) from None

    try:
        if local_test_mode:
            telemetry_map = _load_local_test_telemetry()
            log.info(
                "Running in LOCAL_TEST_MODE with fake telemetry lat=%.7f lon=%.7f rel_alt=%.2f abs_alt=%.2f",
                telemetry_map["latitude_deg"],
                telemetry_map["longitude_deg"],
                telemetry_map["relative_altitude_m"],
                telemetry_map["absolute_altitude_m"],
            )
            while True:
                prompt_text = await asyncio.to_thread(
                    input, "Mission prompt (empty to exit LOCAL_TEST_MODE): "
                )
                prompt_text = prompt_text.strip()
                if not prompt_text:
                    log.info("LOCAL_TEST_MODE prompt empty; stopping.")
                    return

                log.info(
                    "LOCAL_TEST_MODE prompt: %r",
                    prompt_text[:200] + ("..." if len(prompt_text) > 200 else ""),
                )
                trace_id = json_logger.new_trace_id()
                json_logger.log(
                    "prompt_received",
                    trace_id,
                    {"prompt_text": prompt_text, "telemetry": telemetry_map},
                )

                planned = await _plan_from_prompt(
                    llm=llm,
                    mission=mission,
                    settings=settings,
                    prompt_text=prompt_text,
                    telemetry_map=telemetry_map,
                    json_logger=json_logger,
                    trace_id=trace_id,
                    home_state=home_state,
                    rescue_dispatcher=None,  # no gRPC in local-test mode
                )
                if planned is None:
                    continue

                name, proto, planned_trace_id = planned
                await mission.set_mission(name, proto)
                log.info(
                    "LOCAL_TEST_MODE planned mission %r (%d items). No StartMission gRPC call made.",
                    name,
                    len(proto.items),
                )
                log.info(
                    "LOCAL_TEST_MODE mission payload (ordered dict):\n%s",
                    json.dumps(mission_list_to_ordered_dict(proto), indent=2, ensure_ascii=False),
                )
                json_logger.log(
                    "mission_uploaded",
                    planned_trace_id,
                    {
                        "local_test_mode": True,
                        "mission_name": name,
                        "item_count": len(proto.items),
                    },
                )

        async with InternalGrpcClient(settings) as client:
            loop = asyncio.get_running_loop()

            # Build rescue dispatcher now that we have a live gRPC client + event loop.
            rescue_dispatcher = RescueMissionDispatcher(
                loop=loop,
                client=client,
                llm=llm,
                cache=cache,
                home_state=home_state,
                json_logger=json_logger,
                min_rth_alt_m=settings.rescue_min_rth_alt_m,
                camera_mount_pitch_deg=settings.camera_mount_pitch_deg,
                camera_hfov_deg=settings.camera_hfov_deg,
                camera_vfov_deg=settings.camera_vfov_deg,
            )
            # Arm the vision sidecar with the dispatcher so it can trigger rescue.
            if vision_rt is not None:
                vision_rt.recorder.set_rescue_dispatcher(
                    dispatcher=rescue_dispatcher,
                    rescue_person_conf=settings.rescue_person_conf,
                    rescue_person_frames=settings.rescue_person_frames,
                    rescue_arm_delay_s=settings.rescue_arm_delay_s,
                    rescue_photos_dir=Path(settings.rescue_photos_dir),
                )

            telemetry_task = asyncio.create_task(
                _telemetry_poll_loop(client, cache, period_telemetry, stop),
                name="telemetry_poll",
            )
            try:
                while not stop.is_set():
                    try:
                        pr = await client.get_prompt()
                    except Exception:
                        log.exception("GetPrompt failed")
                        await asyncio.sleep(prompt_interval)
                        continue

                    text = (pr.prompt or "").strip()
                    if not text or text == last_seen_prompt[0]:
                        await asyncio.sleep(prompt_interval)
                        continue
                    last_seen_prompt[0] = text

                    log.info("New prompt: %r", text[:200] + ("..." if len(text) > 200 else ""))

                    try:
                        tel = await client.get_telemetry()
                        await cache.update_from_telemetry(tel)
                    except Exception as exc:
                        log.exception("get_telemetry for planning: %s", exc)
                        await mission.mark_error(str(exc))
                        await asyncio.sleep(prompt_interval)
                        continue

                    tel_map = await cache.as_any()
                    trace_id = json_logger.new_trace_id()
                    json_logger.log(
                        "prompt_received",
                        trace_id,
                        {"prompt_text": text, "telemetry": tel_map},
                    )
                    planned = await _plan_from_prompt(
                        llm=llm,
                        mission=mission,
                        settings=settings,
                        prompt_text=text,
                        telemetry_map=tel_map,
                        json_logger=json_logger,
                        trace_id=trace_id,
                        home_state=home_state,
                        rescue_dispatcher=rescue_dispatcher,
                    )
                    if planned is None:
                        await asyncio.sleep(prompt_interval)
                        continue
                    name, proto, planned_trace_id = planned

                    log.info(
                        "gRPC StartMission payload (ordered dict):\n%s",
                        json.dumps(
                            mission_list_to_ordered_dict(proto),
                            indent=2,
                            ensure_ascii=False,
                        ),
                    )

                    try:
                        await client.start_mission(proto)
                    except Exception as exc:
                        log.exception("StartMission failed: %s", exc)
                        json_logger.log(
                            "mission_upload_failed",
                            planned_trace_id,
                            {"stage": "start_mission_rpc", "error": str(exc)},
                        )
                        await mission.mark_error(f"grpc: {exc}")
                        await asyncio.sleep(prompt_interval)
                        continue

                    # Notify the rescue trigger that the first operator mission is airborne.
                    # This starts the arm-delay countdown; the trigger will only become
                    # active after RESCUE_ARM_DELAY_S seconds, giving the drone time to
                    # fly away from the operator before person detection is enabled.
                    if vision_rt is not None:
                        vision_rt.notify_mission_sent()

                    await mission.set_mission(name, proto)
                    log.info("Uploaded mission %r (%d items).", name, len(proto.items))
                    json_logger.log(
                        "mission_uploaded",
                        planned_trace_id,
                        {
                            "local_test_mode": False,
                            "mission_name": name,
                            "item_count": len(proto.items),
                        },
                    )

            except (asyncio.CancelledError, KeyboardInterrupt):
                log.info("Shutting down...")
            finally:
                stop.set()
                telemetry_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await telemetry_task

    finally:
        await asyncio.to_thread(stop_arducam_vision, vision_rt)


__all__ = ["run_mission_test_loop", "_load_settings"]


def main() -> None:
    """CLI entry: configure logging and run :func:`run_mission_test_loop`."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(run_mission_test_loop())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
