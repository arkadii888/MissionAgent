"""Append-only JSONL logger for mission pipeline events (debugging and replay)."""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.orchestrator.protoc import internal_communication_pb2


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


def log_mission_multipoint_geojson(
    logger: JsonPipelineLogger,
    trace_id: str,
    mission: "internal_communication_pb2.MissionItemList",
) -> None:
    """Append a ``mission_multipoint_geojson`` record; payload is a GeoJSON MultiPoint geometry."""
    from agent.orchestrator.mission_intents.proto import mission_list_to_multipoint_geometry

    logger.log("mission_multipoint_geojson", trace_id, mission_list_to_multipoint_geometry(mission))
