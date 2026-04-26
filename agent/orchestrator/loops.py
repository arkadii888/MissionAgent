"""
Basic orchestrator loop for integration testing against a real gRPC server and llama-server.

One asyncio task keeps ``TelemetryCache`` fresh. The main coroutine polls ``GetPrompt``; when
the prompt string changes to a new non-empty value, it fetches cached telemetry, builds prompts,
calls the LLM for mission intents, expands intents into ``MissionItemList``, and calls
``StartMission``.
"""

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
from agent.orchestrator.logging import JsonPipelineLogger
from agent.orchestrator.mission_intents import expand_intents_to_mission
from agent.orchestrator.mission_intents.proto import mission_list_to_ordered_dict
from agent.orchestrator.state import MissionState, TelemetryCache

log = logging.getLogger(__name__)

# Default prompt polling interval in seconds.
_DEFAULT_PROMPT_POLL_S = 1
_LOCAL_TEST_TELEMETRY_DEFAULTS: dict[str, float] = {
    "latitude_deg": 47.3977419,
    "longitude_deg": 8.5455938,
    "relative_altitude_m": 0.0,
    "absolute_altitude_m": 488.0,
}


def _load_settings() -> Settings:
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
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_local_test_telemetry() -> dict[str, float]:
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
) -> tuple[str, Any, str] | None:
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
    """
    Run until SIGINT/KeyboardInterrupt: poll prompts, run LLM, upload mission, and keep
    telemetry current in the background.
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
    local_test_mode = _env_flag("LOCAL_TEST_MODE")
    # Process each unique prompt once per run to avoid hot retry loops when one prompt
    # deterministically fails in LLM expansion or gRPC upload.
    last_seen_prompt: str | None = None

    log.info(
        "Starting mission test loop: grpc=%s llama=%s prompt_poll=%.2fs local_test_mode=%s",
        settings.grpc_target,
        settings.llama_cpp_url,
        prompt_interval,
        local_test_mode,
    )

    if local_test_mode:
        telemetry_map = _load_local_test_telemetry()
        env_prompt = (os.getenv("LOCAL_TEST_PROMPT") or "").strip()
        log.info(
            "Running in LOCAL_TEST_MODE with fake telemetry lat=%.7f lon=%.7f rel_alt=%.2f abs_alt=%.2f",
            telemetry_map["latitude_deg"],
            telemetry_map["longitude_deg"],
            telemetry_map["relative_altitude_m"],
            telemetry_map["absolute_altitude_m"],
        )
        while True:
            prompt_text = env_prompt
            env_prompt = ""
            if not prompt_text:
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
                {"local_test_mode": True, "mission_name": name, "item_count": len(proto.items)},
            )

    async with InternalGrpcClient(settings) as client:
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
                if not text or text == last_seen_prompt:
                    await asyncio.sleep(prompt_interval)
                    continue
                last_seen_prompt = text

                log.info("New prompt: %r", text[:200] + ("..." if len(text) > 200 else ""))

                # Refresh telemetry before planning so expansion uses the latest origin.
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
                )
                if planned is None:
                    await asyncio.sleep(prompt_interval)
                    continue
                name, proto, planned_trace_id = planned

                log.info(
                    "gRPC StartMission payload (ordered dict):\n%s",
                    json.dumps(mission_list_to_ordered_dict(proto), indent=2, ensure_ascii=False),
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

                await mission.set_mission(name, proto)
                log.info("Uploaded mission %r (%d items).", name, len(proto.items))
                json_logger.log(
                    "mission_uploaded",
                    planned_trace_id,
                    {"local_test_mode": False, "mission_name": name, "item_count": len(proto.items)},
                )

        except (asyncio.CancelledError, KeyboardInterrupt):
            log.info("Shutting down...")
        finally:
            stop.set()
            telemetry_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await telemetry_task


__all__ = ["run_mission_test_loop", "_load_settings"]


def main() -> None:
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
