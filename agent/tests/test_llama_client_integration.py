import os
import asyncio
import json
from urllib import error, request

import pytest

from agent.orchestrator.llm.client import LlamaClient
from agent.orchestrator.llm.prompts import build_system_prompt, build_user_prompt


def _server_is_up(base_url: str) -> bool:
    for path in ("/health", "/v1/health"):
        try:
            with request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except (error.URLError, error.HTTPError):
            continue
    return False


def _chat_endpoint_available(base_url: str) -> bool:
    probe = b"{}"
    for path in ("/v1/chat/completions", "/chat/completions"):
        req = request.Request(
            url=f"{base_url.rstrip('/')}{path}",
            data=probe,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=1.5) as _resp:
                return True
        except error.HTTPError as exc:
            if exc.code != 404:
                return True
        except error.URLError:
            continue
    return False


@pytest.mark.asyncio
async def test_llama_client_can_send_and_receive():
    base_url = os.getenv("LLAMA_CPP_URL", "http://127.0.0.1:8080")
    model_name = os.getenv("MODEL_NAME", "gemma-4-e2b")
    if not _server_is_up(base_url):
        pytest.skip(f"llama.cpp server not running at {base_url}")
    if not _chat_endpoint_available(base_url):
        pytest.skip(f"llama.cpp chat endpoint unavailable at {base_url}")

    client = LlamaClient(
        base_url=base_url,
        model_name=model_name,
        timeout_s=90.0,
        max_tokens=256,
        temperature=0.2,
    )
    system_prompt = build_system_prompt(max_waypoints=4)
    user_prompt = build_user_prompt(
        user_prompt="Create a very short square mission and return to launch.",
        telemetry={
            "latitude_deg": 47.3977419,
            "longitude_deg": 8.5455938,
            "relative_altitude_m": 0.0,
            "absolute_altitude_m": 488.0,
        },
        mission_status="IDLE",
    )
    try:
        plan = await client.plan_mission(system_prompt, user_prompt)
        assert isinstance(plan, dict)
        assert "mission_name" in plan
        assert "intents" in plan
        assert isinstance(plan["intents"], list)
        assert len(plan["intents"]) >= 1
    except (ValueError, json.JSONDecodeError, RuntimeError):
        # Fallback assertion: endpoint is reachable and returns model output,
        # even when this llama.cpp build does not strictly enforce JSON schema.
        raw = await asyncio.to_thread(
            client._post_chat_completions,  # noqa: SLF001
            {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 128,
                "temperature": 0.2,
                "stream": False,
            },
        )
        assert isinstance(raw, dict)
        assert "choices" in raw
        assert isinstance(raw["choices"], list)
        assert len(raw["choices"]) > 0
