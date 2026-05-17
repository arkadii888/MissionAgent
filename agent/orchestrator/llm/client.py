"""HTTP client for OpenAI-style chat completions against llama-server."""

import asyncio
import base64
import json
import logging
from collections.abc import Mapping
from typing import Any
from urllib import error, request

from .schemas import MISSION_INTENT_SCHEMA, MISSION_INTENT_SCHEMA_NAME

log = logging.getLogger(__name__)


class LlamaClient:
    """Call ``/v1/chat/completions`` (or legacy ``/chat/completions``) with a JSON-schema mission plan.

    Retries with stricter instructions if the model returns markdown or invalid JSON.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        timeout_s: float = 120.0,
        max_tokens: int = 1024,
        temperature: float = 0.9,
    ) -> None:
        """Configure the llama-server HTTP client.

        Args:
            base_url: Server root URL (e.g. ``http://127.0.0.1:8080``).
            model_name: Model id passed in chat completion requests.
            timeout_s: HTTP read timeout per request.
            max_tokens: Default completion token cap for mission planning.
            temperature: Sampling temperature for mission planning.

        Raises:
            ValueError: If ``base_url`` or ``model_name`` is empty.
        """
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
        """Return a parsed mission intent plan (JSON object) from the model.

        Args:
            system_prompt: Rules and schema constraints.
            user_prompt: Operator request plus telemetry line.

        Returns:
            Mission plan dict with ``mission_name`` and ``intents`` list.

        Raises:
            ValueError: If the response is not a JSON object or is empty.
            RuntimeError: On HTTP or connection errors from the server.
        """
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "json_schema": MISSION_INTENT_SCHEMA,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": MISSION_INTENT_SCHEMA_NAME,
                    "schema": MISSION_INTENT_SCHEMA,
                },
            },
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": False,
        }
        response = await asyncio.to_thread(self._post_chat_completions, payload)
        try:
            self._log_finish_reason(response, "first attempt")
            content = self._extract_content_text(response)
            log.info("llama-server raw assistant content (first attempt):\n%s", content)
            parsed = self._parse_json_from_text(content)
            if not isinstance(parsed, Mapping):
                raise ValueError("mission intent plan must be a JSON object")
            return dict(parsed)
        except (ValueError, json.JSONDecodeError):
            retry_payload = dict(payload)
            # Low max_tokens truncates structured JSON mid-stream ("Unterminated string"); escalate on repair attempts.
            retry_payload["max_tokens"] = max(self._max_tokens, 1024)
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
            self._log_finish_reason(retry_response, "retry")
            retry_content = self._extract_content_text(retry_response)
            log.info("llama-server raw assistant content (retry):\n%s", retry_content)
            try:
                retry_parsed = self._parse_json_from_text(retry_content)
                if not isinstance(retry_parsed, Mapping):
                    raise ValueError(
                        f"mission intent plan must be a JSON object, got: {type(retry_parsed)}"
                    )
                return dict(retry_parsed)
            except (ValueError, json.JSONDecodeError):
                final_payload = dict(retry_payload)
                final_payload["max_tokens"] = max(self._max_tokens, 2048)
                final_payload["messages"] = [
                    {
                        "role": "system",
                        "content": (
                            f"{system_prompt}\n"
                            "Return exactly one valid JSON object and nothing else. "
                            "No markdown, no prose, no comments, no trailing commas."
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ]
                final_response = await asyncio.to_thread(self._post_chat_completions, final_payload)
                self._log_finish_reason(final_response, "final retry")
                final_content = self._extract_content_text(final_response)
                log.info("llama-server raw assistant content (final retry):\n%s", final_content)
                final_parsed = self._parse_json_from_text(final_content)
                if not isinstance(final_parsed, Mapping):
                    raise ValueError(
                        f"mission intent plan must be a JSON object, got: {type(final_parsed)}"
                    )
                return dict(final_parsed)

    async def analyze_image(
        self,
        *,
        system: str,
        user_text: str,
        image_jpeg: bytes,
        mime: str = "image/jpeg",
        max_tokens: int | None = None,
    ) -> str:
        """Send a multimodal prompt with an inline image and return the assistant text.

        The llama-server must be started with ``--mmproj`` for the vision projector to
        be loaded; otherwise the image data will be ignored by the server.

        Args:
            system: System message text.
            user_text: User message text (describes context for the image).
            image_jpeg: Raw image bytes (JPEG by default).
            mime: MIME type for the data URI, e.g. ``"image/jpeg"``.
            max_tokens: Override for this call; defaults to ``self._max_tokens``.

        Returns:
            Assistant response as a plain string.
        """
        b64 = base64.b64encode(image_jpeg).decode("ascii")
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "temperature": self._temperature,
            "stream": False,
        }
        response = await asyncio.to_thread(self._post_chat_completions, payload)
        self._log_finish_reason(response, "analyze_image")
        return self._extract_content_text(response)

    def _extract_content_text(self, response: Mapping[str, Any]) -> str:
        """Pick assistant text from various OpenAI-compatible response shapes."""
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

            reasoning_content = message.get("reasoning_content")
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                return reasoning_content

        content = first_choice.get("content")
        if isinstance(content, str) and content.strip():
            return content

        raise ValueError(f"llama returned empty content payload: {response}")

    @staticmethod
    def _log_finish_reason(response: Mapping[str, Any], label: str) -> None:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        first = choices[0]
        if not isinstance(first, Mapping):
            return
        fr = first.get("finish_reason")
        if fr is None:
            return
        log.info("llama-server %s finish_reason=%s", label, fr)
        if fr == "length":
            log.warning(
                "llama-server %s stopped at max_tokens (output truncated); raise LLM_MAX_TOKENS if JSON parse fails",
                label,
            )

    def _parse_json_from_text(self, text: str) -> Any:
        """Parse JSON, stripping optional ```json fences and scanning for a first ``{...}`` object."""
        stripped = text.strip()
        if not stripped:
            raise ValueError("llama response content is empty")

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
            first_obj = self._extract_first_balanced_json_object(stripped)
            if first_obj is not None:
                return json.loads(first_obj)
            raise

    def _extract_first_balanced_json_object(self, text: str) -> str | None:
        """Find the first top-level JSON object; respects strings and escapes."""
        start = -1
        depth = 0
        in_string = False
        escaping = False

        for i, ch in enumerate(text):
            if in_string:
                if escaping:
                    escaping = False
                    continue
                if ch == "\\":
                    escaping = True
                    continue
                if ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
                continue
            if ch == "}":
                if depth == 0:
                    continue
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start : i + 1]
        return None

    def _post_chat_completions(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """POST JSON; try OpenAI path first, then legacy path."""
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
