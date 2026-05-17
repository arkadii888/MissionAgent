"""Environment-backed configuration for the orchestrator (LLM, gRPC, logging, rescue)."""

import os
from dataclasses import dataclass


def _rescue_photos_mirror_dir_from_env() -> str | None:
    raw = os.getenv("RESCUE_PHOTOS_MIRROR_DIR")
    if raw is None:
        return "/agent/arducamphotos"
    stripped = raw.strip()
    return stripped or None


@dataclass(frozen=True)
class Settings:
    """Orchestrator parameters read from environment variables.

    Attributes:
        llama_cpp_url: Base URL for OpenAI-compatible chat (e.g. llama-server). Env: ``LLAMA_CPP_URL``.
        model_name: Model id passed in API requests. Env: ``MODEL_NAME``.
        grpc_target: Host:port for InternalService when not in local test. Env: ``GRPC_TARGET``.
        grpc_timeout_s: Per-RPC deadline in seconds. Env: ``GRPC_TIMEOUT_S``.
        telemetry_poll_hz: Background telemetry poll rate. Env: ``TELEMETRY_POLL_HZ``.
        llm_timeout_s: HTTP deadline for chat completion. Env: ``LLM_TIMEOUT_S``.
        llm_max_tokens: Completion token cap. Env: ``LLM_MAX_TOKENS``.
        llm_temperature: Sampling temperature. Env: ``LLM_TEMPERATURE``.
        max_waypoints: Caps intents in prompts and schema (see also ``MAX_WAYPOINTS`` env).
        extended_mission_prompts: Use the full intent checklist and few-shot examples in LLM prompts.
            Env: ``EXTENDED_MISSION_PROMPTS`` (default false).
        mission_json_log_enabled: Append JSON pipeline events when true. Env: ``MISSION_JSON_LOG_ENABLED``.
        mission_json_log_path: JSONL destination path. Env: ``MISSION_JSON_LOG_PATH``.
        rescue_person_conf: Min YOLO person confidence to count a qualifying frame. Env: ``RESCUE_PERSON_CONF``.
        rescue_person_frames: Consecutive qualifying frames before trigger fires. Env: ``RESCUE_PERSON_FRAMES``.
        rescue_arm_delay_s: Seconds after the first operator mission is sent before the rescue trigger
            becomes active. Prevents detecting the operator at takeoff. Env: ``RESCUE_ARM_DELAY_S``.
        rescue_photos_dir: Directory for annotated frame + crop JPEG saves. Env: ``RESCUE_PHOTOS_DIR``.
        rescue_photos_mirror_dir: Optional second directory to copy the same JPEGs into after paths
            are final (including geo-based renames). Env: ``RESCUE_PHOTOS_MIRROR_DIR``. When unset,
            defaults to ``/agent/arducamphotos``. Set to empty to disable mirroring.
        rescue_min_rth_alt_m: Minimum cruise altitude for the return-home leg. Env: ``RESCUE_MIN_RTH_ALT_M``.
        camera_mount_pitch_deg: Camera mount angle down from horizontal (90=nadir). Env: ``CAMERA_MOUNT_PITCH_DEG``.
        camera_hfov_deg: Camera horizontal field of view in degrees. Env: ``CAMERA_HFOV_DEG``.
        camera_vfov_deg: Camera vertical field of view in degrees. Env: ``CAMERA_VFOV_DEG``.
        rescue_image_llm_enabled: When false, skip post-rescue multimodal image analysis
            (no second LLM call with the crop). Pair with llama-server started without ``--mmproj``.
            Env: ``RESCUE_IMAGE_LLM_ENABLED`` (default true).
    """

    llama_cpp_url: str | None
    model_name: str | None
    grpc_target: str | None
    grpc_timeout_s: float
    telemetry_poll_hz: float
    llm_timeout_s: float
    llm_max_tokens: int
    llm_temperature: float
    max_waypoints: int
    extended_mission_prompts: bool
    mission_json_log_enabled: bool
    mission_json_log_path: str

    # --- Person rescue trigger ---
    rescue_person_conf: float
    rescue_person_frames: int
    rescue_arm_delay_s: float
    rescue_photos_dir: str
    rescue_photos_mirror_dir: str | None
    rescue_min_rth_alt_m: float

    # --- Camera geometry for person-offset estimation ---
    camera_mount_pitch_deg: float
    camera_hfov_deg: float
    camera_vfov_deg: float
    rescue_image_llm_enabled: bool

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from ``os.environ`` using documented defaults."""
        return cls(
            llama_cpp_url=os.getenv("LLAMA_CPP_URL"),
            model_name=os.getenv("MODEL_NAME"),
            grpc_target=os.getenv("GRPC_TARGET"),
            grpc_timeout_s=float(os.getenv("GRPC_TIMEOUT_S", "4.0")),
            telemetry_poll_hz=float(os.getenv("TELEMETRY_POLL_HZ", "2.0")),
            llm_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "120.0")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_waypoints=int(os.getenv("MAX_WAYPOINTS", "32")),
            extended_mission_prompts=os.getenv("EXTENDED_MISSION_PROMPTS", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            mission_json_log_enabled=os.getenv("MISSION_JSON_LOG_ENABLED", "1").strip().lower()
            in {"1", "true", "yes", "on"},
            mission_json_log_path=os.getenv("MISSION_JSON_LOG_PATH", "agent/logs/mission_pipeline.jsonl"),
            rescue_person_conf=float(os.getenv("RESCUE_PERSON_CONF", "0.75")),
            rescue_person_frames=int(os.getenv("RESCUE_PERSON_FRAMES", "5")),
            rescue_arm_delay_s=float(os.getenv("RESCUE_ARM_DELAY_S", "60.0")),
            rescue_photos_dir=os.getenv("RESCUE_PHOTOS_DIR", "agent/arducamphotos"),
            rescue_photos_mirror_dir=_rescue_photos_mirror_dir_from_env(),
            rescue_min_rth_alt_m=float(os.getenv("RESCUE_MIN_RTH_ALT_M", "10.0")),
            camera_mount_pitch_deg=float(os.getenv("CAMERA_MOUNT_PITCH_DEG", "90.0")),
            camera_hfov_deg=float(os.getenv("CAMERA_HFOV_DEG", "66.0")),
            camera_vfov_deg=float(os.getenv("CAMERA_VFOV_DEG", "41.0")),
            rescue_image_llm_enabled=os.getenv("RESCUE_IMAGE_LLM_ENABLED", "1").strip().lower()
            in {"1", "true", "yes", "on"},
        )
