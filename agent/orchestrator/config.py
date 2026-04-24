import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llama_cpp_url: str | None
    model_name: str | None
    grpc_target: str | None
    grpc_timeout_s: float
    telemetry_poll_hz: float
    llm_timeout_s: float
    llm_max_tokens: int
    llm_temperature: float
    max_waypoints: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llama_cpp_url=os.getenv("LLAMA_CPP_URL"),
            model_name=os.getenv("MODEL_NAME"),
            grpc_target=os.getenv("GRPC_TARGET"),
            grpc_timeout_s=float(os.getenv("GRPC_TIMEOUT_S", "4.0")),
            telemetry_poll_hz=float(os.getenv("TELEMETRY_POLL_HZ", "2.0")),
            llm_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "120.0")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_waypoints=int(os.getenv("MAX_WAYPOINTS", "16")),
        )
