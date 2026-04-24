from enum import StrEnum

import asyncio

from agent.orchestrator.protoc import internal_communication_pb2


class MissionPhase(StrEnum):
    """High-level mission lifecycle (orchestrator-owned; not from PX4)."""

    IDLE = "IDLE"
    PLANNING = "PLANNING"
    UPLOADED = "UPLOADED"
    FLYING = "FLYING"
    REPLANNING = "REPLANNING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class MissionState:
    """Tracks the active plan, progress, and phase for prompts, triggers, and the executor."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._phase = MissionPhase.IDLE
        self._mission_name: str | None = None
        self._plan: internal_communication_pb2.MissionItemList | None = None
        self._waypoint_index: int = 0
        self._last_error: str | None = None

    async def get_phase(self) -> MissionPhase:
        async with self._lock:
            return self._phase

    async def set_phase(self, phase: MissionPhase) -> None:
        async with self._lock:
            self._phase = phase

    async def begin_planning(self) -> None:
        await self.set_phase(MissionPhase.PLANNING)
        async with self._lock:
            self._last_error = None

    async def begin_replanning(self) -> None:
        await self.set_phase(MissionPhase.REPLANNING)
        async with self._lock:
            self._last_error = None

    async def set_mission(
        self,
        name: str,
        plan: internal_communication_pb2.MissionItemList,
    ) -> None:
        stored = internal_communication_pb2.MissionItemList()
        stored.CopyFrom(plan)
        async with self._lock:
            self._mission_name = name
            self._plan = stored
            self._waypoint_index = 0
            self._phase = MissionPhase.UPLOADED
            self._last_error = None

    async def mark_flying(self) -> None:
        await self.set_phase(MissionPhase.FLYING)

    async def set_waypoint_index(self, index: int) -> None:
        if index < 0:
            raise ValueError("waypoint index must be non-negative")
        async with self._lock:
            self._waypoint_index = index

    async def advance_waypoint(self) -> None:
        async with self._lock:
            n = len(self._plan.items) if self._plan is not None else 0
            if n == 0:
                return
            self._waypoint_index = min(self._waypoint_index + 1, n - 1)

    async def mark_complete(self) -> None:
        async with self._lock:
            self._phase = MissionPhase.COMPLETE
            self._waypoint_index = 0
            if self._plan is not None and len(self._plan.items) > 0:
                self._waypoint_index = len(self._plan.items) - 1

    async def mark_error(self, message: str) -> None:
        async with self._lock:
            self._phase = MissionPhase.ERROR
            self._last_error = message

    async def clear_mission(self) -> None:
        async with self._lock:
            self._mission_name = None
            self._plan = None
            self._waypoint_index = 0
            self._phase = MissionPhase.IDLE
            self._last_error = None

    async def get_plan(self) -> internal_communication_pb2.MissionItemList | None:
        async with self._lock:
            if self._plan is None:
                return None
            out = internal_communication_pb2.MissionItemList()
            out.CopyFrom(self._plan)
            return out

    async def get_waypoint_progress(self) -> tuple[int, int]:
        """Current waypoint index and total waypoints (0,0 if no plan)."""
        async with self._lock:
            if self._plan is None:
                return (0, 0)
            return (self._waypoint_index, len(self._plan.items))

    async def prompt_mission_status(self) -> str:
        """Single line for `build_user_prompt(mission_status=...)`."""
        async with self._lock:
            phase = self._phase
            name = self._mission_name
            wpi = self._waypoint_index
            n = len(self._plan.items) if self._plan is not None else 0
            err = self._last_error

        parts: list[str] = [f"phase={phase.value}"]
        if name:
            parts.append(f"name={name!r}")
        if n > 0:
            parts.append(f"waypoint={wpi + 1}/{n}")
        if err and phase == MissionPhase.ERROR:
            parts.append(f"error={err!r}")
        return ", ".join(parts)
