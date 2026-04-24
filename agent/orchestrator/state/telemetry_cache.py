import asyncio
import time
from dataclasses import dataclass
from typing import Any

from agent.orchestrator.protoc import internal_communication_pb2


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """Immutable copy of the last telemetry (safe to pass across async tasks)."""

    latitude_deg: float
    longitude_deg: float
    relative_altitude_m: float
    absolute_altitude_m: float
    monotonic_s: float

    @classmethod
    def from_proto(cls, t: internal_communication_pb2.TelemetryResponse) -> "TelemetrySnapshot":
        return cls(
            latitude_deg=float(t.latitude_deg),
            longitude_deg=float(t.longitude_deg),
            relative_altitude_m=float(t.relative_altitude_m),
            absolute_altitude_m=float(t.absolute_altitude_m),
            monotonic_s=time.monotonic(),
        )

    def to_prompt_map(self) -> dict[str, float]:
        return {
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "relative_altitude_m": self.relative_altitude_m,
            "absolute_altitude_m": self.absolute_altitude_m,
        }


class TelemetryCache:
    """Latest gRPC `GetTelemetry` result, protected by a lock for concurrent poller + reader."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._latest: TelemetrySnapshot | None = None

    async def update_from_telemetry(
        self,
        response: internal_communication_pb2.TelemetryResponse,
    ) -> None:
        snap = TelemetrySnapshot.from_proto(response)
        async with self._lock:
            self._latest = snap

    async def get_snapshot(self) -> TelemetrySnapshot | None:
        async with self._lock:
            return self._latest

    async def get_for_prompt(self) -> dict[str, float] | None:
        snap = await self.get_snapshot()
        if snap is None:
            return None
        return snap.to_prompt_map()

    async def as_any(self) -> dict[str, Any]:
        """Return prompt-shaped telemetry or N/A sentinels when no sample yet."""
        m = await self.get_for_prompt()
        if m is not None:
            return m
        return {
            "latitude_deg": float("nan"),
            "longitude_deg": float("nan"),
            "relative_altitude_m": float("nan"),
            "absolute_altitude_m": float("nan"),
        }
