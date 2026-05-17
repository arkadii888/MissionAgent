"""In-memory planner state mirrored for prompts and downstream mission execution hooks.

TODO: Track live waypoint progress when the vehicle exposes current mission index over gRPC.
"""

from enum import StrEnum

import asyncio

from agent.orchestrator.protoc import internal_communication_pb2


class MissionPhase(StrEnum):
    """Where the orchestrator is in mission handling (distinct from MAVLink autopilot modes)."""

    IDLE = "IDLE"
    PLANNING = "PLANNING"
    UPLOADED = "UPLOADED"
    ERROR = "ERROR"


class MissionState:
    """Stores the uploaded plan name, protobuf copy, phase, and last error."""

    def __init__(self) -> None:
        """Create an empty mission state in ``IDLE`` phase."""
        self._lock = asyncio.Lock()
        self._phase = MissionPhase.IDLE
        self._mission_name: str | None = None
        self._plan: internal_communication_pb2.MissionItemList | None = None
        self._last_error: str | None = None

    async def get_phase(self) -> MissionPhase:
        """Return the current orchestrator mission phase.

        Returns:
            Current :class:`MissionPhase`.
        """
        async with self._lock:
            return self._phase

    async def begin_planning(self) -> None:
        """Enter ``PLANNING`` and clear any prior error."""
        async with self._lock:
            self._phase = MissionPhase.PLANNING
            self._last_error = None

    async def set_mission(
        self,
        name: str,
        plan: internal_communication_pb2.MissionItemList,
    ) -> None:
        """Persist a deep copy of ``plan`` and mark phase ``UPLOADED``.

        Args:
            name: Mission label from the LLM plan.
            plan: Expanded protobuf waypoint list ready for ``StartMission``.
        """
        stored = internal_communication_pb2.MissionItemList()
        stored.CopyFrom(plan)
        async with self._lock:
            self._mission_name = name
            self._plan = stored
            self._phase = MissionPhase.UPLOADED
            self._last_error = None

    async def mark_error(self, message: str) -> None:
        """Record a failure and set phase to ``ERROR``.

        Args:
            message: Short error text included in :meth:`prompt_mission_status` when in error.
        """
        async with self._lock:
            self._phase = MissionPhase.ERROR
            self._last_error = message

    async def get_plan(self) -> internal_communication_pb2.MissionItemList | None:
        """Return a deep copy of the uploaded plan, or ``None`` if idle.

        Returns:
            Copied ``MissionItemList`` or ``None``.
        """
        async with self._lock:
            if self._plan is None:
                return None
            out = internal_communication_pb2.MissionItemList()
            out.CopyFrom(self._plan)
            return out

    async def prompt_mission_status(self) -> str:
        """Compact line for embedding in LLM prompts (phase, name, item count, error).

        Does not include live waypoint index; the vehicle does not report progress yet.

        Returns:
            Comma-separated status string, e.g. ``phase=UPLOADED, name='patrol', items=5``.
        """
        async with self._lock:
            phase = self._phase
            name = self._mission_name
            n = len(self._plan.items) if self._plan is not None else 0
            err = self._last_error

        parts: list[str] = [f"phase={phase.value}"]
        if name:
            parts.append(f"name={name!r}")
        if n > 0:
            parts.append(f"items={n}")
        if err and phase == MissionPhase.ERROR:
            parts.append(f"error={err!r}")
        return ", ".join(parts)
