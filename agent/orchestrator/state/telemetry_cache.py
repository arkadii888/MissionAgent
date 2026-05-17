"""Thread-safe snapshot of recent vehicle telemetry for planning prompts."""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from agent.orchestrator.protoc import internal_communication_pb2


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """Frozen telemetry row plus capture time (:func:`time.monotonic`).

    Attributes:
        yaw_deg: Vehicle heading in degrees from telemetry.
        monotonic_s: When this sample was recorded (seconds, process monotonic clock).
    """

    latitude_deg: float
    longitude_deg: float
    relative_altitude_m: float
    absolute_altitude_m: float
    yaw_deg: float
    monotonic_s: float

    @classmethod
    def from_proto(cls, t: internal_communication_pb2.TelemetryResponse) -> "TelemetrySnapshot":
        """Build from ``GetTelemetry`` response."""
        return cls(
            latitude_deg=float(t.latitude_deg),
            longitude_deg=float(t.longitude_deg),
            relative_altitude_m=float(t.relative_altitude_m),
            absolute_altitude_m=float(t.absolute_altitude_m),
            yaw_deg=float(t.yaw_deg),
            monotonic_s=time.monotonic(),
        )

    def to_prompt_map(self) -> dict[str, float]:
        """Dict keys expected by :func:`~agent.orchestrator.llm.prompts.build_user_prompt`."""
        return {
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "relative_altitude_m": self.relative_altitude_m,
            "absolute_altitude_m": self.absolute_altitude_m,
            "yaw_deg": self.yaw_deg,
        }


class TelemetryCache:
    """Holds latest ``TelemetrySnapshot``; safe for concurrent poller and planners."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._latest: TelemetrySnapshot | None = None

    async def update_from_telemetry(
        self,
        response: internal_communication_pb2.TelemetryResponse,
    ) -> None:
        async with self._lock:
            self._latest = TelemetrySnapshot.from_proto(response)

    async def get_snapshot(self) -> TelemetrySnapshot | None:
        async with self._lock:
            return self._latest

    async def get_for_prompt(self) -> dict[str, float] | None:
        """``None`` until the first telemetry sample arrives."""
        snap = await self.get_snapshot()
        if snap is None:
            return None
        return snap.to_prompt_map()

    async def as_any(self) -> dict[str, Any]:
        """Telemetry dict for prompts, or NaN placeholders until data exists.

        Expansion will reject NaN lat/lon; callers should ensure a fresh sample exists
        before running the LLM when operating on hardware.
        """
        m = await self.get_for_prompt()
        if m is not None:
            return m
        return {
            "latitude_deg": float("nan"),
            "longitude_deg": float("nan"),
            "relative_altitude_m": float("nan"),
            "absolute_altitude_m": float("nan"),
            "yaw_deg": float("nan"),
        }
