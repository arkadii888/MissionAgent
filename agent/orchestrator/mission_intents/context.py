from dataclasses import dataclass, field

from agent.orchestrator.protoc import internal_communication_pb2


@dataclass
class ExpansionContext:
    base_latitude_deg: float
    base_longitude_deg: float
    north_total_m: float = 0.0
    east_total_m: float = 0.0
    current_altitude_m: float = 0.0
    pending_yaw_deg: float | None = None
    preempted: bool = False
    items: list[internal_communication_pb2.MissionItem] = field(default_factory=list)
