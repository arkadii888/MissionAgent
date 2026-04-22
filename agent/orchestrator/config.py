import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llama_cpp_url: str | None
    model_name: str | None
    grpc_target: str | None
    grpc_timeout_s: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llama_cpp_url=os.getenv("LLAMA_CPP_URL"),
            model_name=os.getenv("MODEL_NAME"),
            grpc_target=os.getenv("GRPC_TARGET"),
            grpc_timeout_s=float(os.getenv("GRPC_TIMEOUT_S", "4.0")),
        )
