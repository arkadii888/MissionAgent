import math
from collections import OrderedDict

from agent.orchestrator.protoc import internal_communication_pb2

from .geometry import MAX_RELATIVE_ALTITUDE_M, MIN_RELATIVE_ALTITUDE_M, normalize_yaw


def nan() -> float:
    return float("nan")


def build_proto_item(
    *,
    latitude_deg: float,
    longitude_deg: float,
    relative_altitude_m: float,
    is_fly_through: bool,
    vehicle_action: int,
    loiter_time_s: float = 1.0,
    yaw_deg: float = 0.0,
    speed_m_s: float = 1.0,
) -> internal_communication_pb2.MissionItem:
    proto_item = internal_communication_pb2.MissionItem()
    proto_item.latitude_deg = latitude_deg
    proto_item.longitude_deg = longitude_deg
    proto_item.relative_altitude_m = relative_altitude_m
    proto_item.speed_m_s = speed_m_s
    proto_item.is_fly_through = is_fly_through
    proto_item.gimbal_pitch_deg = nan()
    proto_item.gimbal_yaw_deg = nan()
    proto_item.camera_action = 0
    proto_item.loiter_time_s = loiter_time_s
    proto_item.camera_photo_interval_s = 0.1
    proto_item.acceptance_radius_m = 0.5
    proto_item.yaw_deg = normalize_yaw(yaw_deg)
    proto_item.camera_photo_distance_m = nan()
    proto_item.vehicle_action = vehicle_action
    return proto_item


def validate_proto_item(item: internal_communication_pb2.MissionItem) -> None:
    if not (-90.0 <= item.latitude_deg <= 90.0):
        raise ValueError("latitude_deg must be in [-90, 90]")
    if not (-180.0 <= item.longitude_deg <= 180.0):
        raise ValueError("longitude_deg must be in [-180, 180]")
    if not (MIN_RELATIVE_ALTITUDE_M <= item.relative_altitude_m <= MAX_RELATIVE_ALTITUDE_M):
        raise ValueError("relative_altitude_m must be in [0, 100]")
    if item.speed_m_s != 1.0:
        raise ValueError("speed_m_s must be 1.0")
    if item.loiter_time_s < 0.0:
        raise ValueError("loiter_time_s must be >= 0")
    if not (-360.0 <= item.yaw_deg <= 360.0):
        raise ValueError("yaw_deg must be in [-360, 360]")
    if item.camera_action != 0:
        raise ValueError("camera_action must be 0")
    if item.vehicle_action not in {0, 1, 2, 3, 4}:
        raise ValueError("vehicle_action must be in {0,1,2,3,4}")


def validate_proto_list(result: internal_communication_pb2.MissionItemList) -> None:
    if not result.items:
        raise ValueError("mission must include at least one item")
    for item in result.items:
        validate_proto_item(item)
    if not math.isnan(result.items[0].gimbal_pitch_deg):
        raise ValueError("expected gimbal_pitch_deg default to NaN")


_MISSION_ITEM_FIELD_ORDER: tuple[str, ...] = (
    "latitude_deg",
    "longitude_deg",
    "relative_altitude_m",
    "speed_m_s",
    "is_fly_through",
    "gimbal_pitch_deg",
    "gimbal_yaw_deg",
    "camera_action",
    "loiter_time_s",
    "camera_photo_interval_s",
    "acceptance_radius_m",
    "yaw_deg",
    "camera_photo_distance_m",
    "vehicle_action",
)


def _format_scalar(value: float | int | bool) -> float | int | bool | str:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def mission_item_to_ordered_dict(
    item: internal_communication_pb2.MissionItem,
) -> OrderedDict[str, float | int | bool | str]:
    out: OrderedDict[str, float | int | bool | str] = OrderedDict()
    for field_name in _MISSION_ITEM_FIELD_ORDER:
        out[field_name] = _format_scalar(getattr(item, field_name))
    return out


def mission_list_to_ordered_dict(
    result: internal_communication_pb2.MissionItemList,
) -> dict[str, list[OrderedDict[str, float | int | bool | str]]]:
    return {"items": [mission_item_to_ordered_dict(item) for item in result.items]}
