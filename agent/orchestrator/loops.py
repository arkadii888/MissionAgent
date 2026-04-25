"""
Basic orchestrator loop for integration testing against a real gRPC server and llama-server.

One asyncio task keeps ``TelemetryCache`` fresh. The main coroutine polls ``GetPrompt``; when
the prompt string changes to a new non-empty value, it fetches cached telemetry, builds prompts,
calls the LLM, maps JSON to ``MissionItemList``, and ``StartMission``.
"""

import asyncio
import contextlib
import json
import logging
import os
from dataclasses import replace
from typing import Any

from google.protobuf.json_format import MessageToJson

from agent.orchestrator.config import Settings
from agent.orchestrator.grpc_client import InternalGrpcClient
from agent.orchestrator.llm.client import LlamaClient
from agent.orchestrator.llm.mapping import mission_plan_to_proto
from agent.orchestrator.llm.prompts import build_system_prompt, build_user_prompt
from agent.orchestrator.state import MissionState, TelemetryCache

log = logging.getLogger(__name__)

# How often to poll C++ for a new user prompt (seconds between GetPrompt calls).
_DEFAULT_PROMPT_POLL_S = 1


def _load_settings() -> Settings:
    base = Settings.from_env()
    out = base
    if out.grpc_target is None:
        out = replace(out, grpc_target="localhost:50051")
    if out.llama_cpp_url is None:
        out = replace(out, llama_cpp_url="http://127.0.0.1:8080")
    if out.model_name is None:
        out = replace(out, model_name="gemma")
    return out


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
    # Deduplicate by prompt text so each unique controller prompt is attempted once.
    # This avoids hot retry loops when one prompt deterministically fails in LLM/map/gRPC.
    last_seen_prompt: str | None = None

    log.info(
        "Starting mission test loop: grpc=%s llama=%s prompt_poll=%.2fs",
        settings.grpc_target,
        settings.llama_cpp_url,
        prompt_interval,
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

                # Fresh telemetry for planning
                try:
                    tel = await client.get_telemetry()
                    await cache.update_from_telemetry(tel)
                except Exception as exc:
                    log.exception("get_telemetry for planning: %s", exc)
                    await mission.mark_error(str(exc))
                    await asyncio.sleep(prompt_interval)
                    continue

                tel_map = await cache.as_any()
                status_line = await mission.prompt_mission_status()

                await mission.begin_planning()
                system = build_system_prompt(max_waypoints=settings.max_waypoints)
                user = build_user_prompt(
                    user_prompt=text,
                    telemetry=tel_map,
                    mission_status=status_line,
                )
                log.info("LLM system prompt sent to llama-server:\n%s", system)
                log.info("LLM user prompt sent to llama-server:\n%s", user)

                try:
                    plan_dict: dict[str, Any] = await llm.plan_mission(system, user)
                except Exception as exc:
                    log.exception("LLM plan_mission failed: %s", exc)
                    await mission.mark_error(f"llm: {exc}")
                    await asyncio.sleep(prompt_interval)
                    continue

                log.info(
                    "LLM parsed mission plan (dict after JSON parse):\n%s",
                    json.dumps(plan_dict, indent=2, ensure_ascii=False),
                )

                try:
                    proto = mission_plan_to_proto(plan_dict, tel_map, text)
                except Exception as exc:
                    log.exception("mission_plan_to_proto: %s", exc)
                    await mission.mark_error(f"map: {exc}")
                    await asyncio.sleep(prompt_interval)
                    continue

                name = str(plan_dict.get("mission_name", "mission"))[:64]

                log.info(
                    "gRPC StartMission payload (JSON view):\n%s",
                    MessageToJson(
                        proto,
                        preserving_proto_field_name=True,
                        always_print_fields_with_no_presence=True,
                    ),
                )

                try:
                    await client.start_mission(proto)
                except Exception as exc:
                    log.exception("StartMission failed: %s", exc)
                    await mission.mark_error(f"grpc: {exc}")
                    await asyncio.sleep(prompt_interval)
                    continue

                await mission.set_mission(name, proto)
                log.info("Uploaded mission %r (%d items).", name, len(proto.items))

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
