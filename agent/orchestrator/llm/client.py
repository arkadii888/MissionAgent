import asyncio
import json
from collections.abc import Mapping
from typing import Any
from urllib import error, request

from .schemas import MISSION_PLAN_SCHEMA, MISSION_PLAN_SCHEMA_NAME


class LlamaClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        timeout_s: float = 120.0,
        max_tokens: int = 1024,
        temperature: float = 0.9,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be set")
        if not model_name:
            raise ValueError("model_name must be set")
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def plan_mission(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Keep both formats: different llama.cpp builds enforce one or the other.
            "json_schema": MISSION_PLAN_SCHEMA,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": MISSION_PLAN_SCHEMA_NAME,
                    "schema": MISSION_PLAN_SCHEMA,
                },
            },
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": False,
        }
        response = await asyncio.to_thread(self._post_chat_completions, payload)
        try:
            content = self._extract_content_text(response)
            parsed = self._parse_json_from_text(content)
            if not isinstance(parsed, Mapping):
                raise ValueError("mission plan must be a JSON object")
            return dict(parsed)
        except (ValueError, json.JSONDecodeError):
            # Retry once with stronger output constraints for servers that ignore schema hints.
            retry_payload = dict(payload)
            retry_payload["messages"] = [
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n"
                        "Return only one compact JSON object. "
                        "Do not include markdown, preamble, explanations, or thinking."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ]
            retry_payload["temperature"] = 0.0
            retry_response = await asyncio.to_thread(self._post_chat_completions, retry_payload)
            retry_content = self._extract_content_text(retry_response)
            retry_parsed = self._parse_json_from_text(retry_content)
            if not isinstance(retry_parsed, Mapping):
                raise ValueError(f"mission plan must be a JSON object, got: {type(retry_parsed)}")
            return dict(retry_parsed)

    def _extract_content_text(self, response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"llama response missing choices: {response}")

        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise ValueError(f"llama choice is not an object: {first_choice}")

        message = first_choice.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content

            # Some servers put useful model output here.
            reasoning_content = message.get("reasoning_content")
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                return reasoning_content

        # Fallback for alternate server response formats.
        content = first_choice.get("content")
        if isinstance(content, str) and content.strip():
            return content

        raise ValueError(f"llama returned empty content payload: {response}")

    def _parse_json_from_text(self, text: str) -> Any:
        stripped = text.strip()
        if not stripped:
            raise ValueError("llama response content is empty")

        # Common markdown-wrapped output: ```json ... ```
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            start_obj = stripped.find("{")
            end_obj = stripped.rfind("}")
            if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
                return json.loads(stripped[start_obj : end_obj + 1])
            raise

    def _post_chat_completions(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        errors: list[str] = []
        for path in ("/v1/chat/completions", "/chat/completions"):
            try:
                return self._post_json(path, payload)
            except RuntimeError as exc:
                errors.append(str(exc))
        raise RuntimeError(" ; ".join(errors))

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self._base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp HTTP error {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"llama.cpp connection error: {exc.reason}") from exc

        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("llama.cpp response must be a JSON object")
        return parsed
