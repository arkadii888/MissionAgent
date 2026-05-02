"""Mutable interpreter state while expanding intents to waypoints."""

from dataclasses import dataclass, field

from agent.orchestrator.protoc import internal_communication_pb2


@dataclass
class ExpansionContext:
    """Cumulative planar state and protobuf list built by intent handlers.

    Attributes:
        base_latitude_deg: Mission origin latitude (degrees); fixed for one expansion.
        base_longitude_deg: Mission origin longitude (degrees).
        north_total_m: Displacement north from origin in meters.
        east_total_m: Displacement east from origin in meters.
        current_altitude_m: Relative altitude AMSL/AGL per project convention, clamped in handlers.
        pending_yaw_deg: Heading to apply on the next emitted waypoint if set.
        preempted: Set by ``safety_control`` to skip subsequent movement intents.
        items: Appended ``MissionItem`` protobufs in execution order.
    """

    base_latitude_deg: float
    base_longitude_deg: float
    north_total_m: float = 0.0
    east_total_m: float = 0.0
    current_altitude_m: float = 0.0
    pending_yaw_deg: float | None = None
    preempted: bool = False
    items: list[internal_communication_pb2.MissionItem] = field(default_factory=list)
