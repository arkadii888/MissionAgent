"""Append-only JSONL logger for mission pipeline events (debugging and replay)."""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JsonPipelineLogger:
    """Writes one JSON object per line: timestamp, event name, trace id, payload.

    Attributes:
        path: Output file; parent directories are created on first write.
        enabled: When false, :meth:`log` is a no-op.
    """

    path: Path
    enabled: bool = True

    def new_trace_id(self) -> str:
        """Return a new hex id to correlate all events for one planning attempt."""
        return uuid.uuid4().hex

    def log(self, event: str, trace_id: str, payload: dict[str, Any]) -> None:
        """Append a single record if logging is enabled.

        Args:
            event: Short event type (e.g. ``"mission_converted"``).
            trace_id: From :meth:`new_trace_id` for this run.
            payload: JSON-serializable details (missions, errors, etc.).
        """
        if not self.enabled:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "trace_id": trace_id,
            "payload": payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
