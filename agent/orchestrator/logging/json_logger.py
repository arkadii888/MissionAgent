import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JsonPipelineLogger:
    path: Path
    enabled: bool = True

    def new_trace_id(self) -> str:
        return uuid.uuid4().hex

    def log(self, event: str, trace_id: str, payload: dict[str, Any]) -> None:
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
