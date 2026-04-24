import grpc
import grpc.aio
from typing import Any

from .config import Settings
from .protoc import internal_communication_pb2
from .protoc import internal_communication_pb2_grpc

# Defaults for a long‑lived client: PINGs keep the connection warm behind NATs / LBs.
_DEFAULT_CHANNEL_OPTIONS: tuple[tuple[str, int], ...] = (
    ("grpc.keepalive_time_ms", 10_000),
    ("grpc.keepalive_timeout_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
)


class InternalGrpcClient:
    """Async gRPC client for ``InternalService``. RPCs await without blocking the event loop."""

    def __init__(
        self,
        settings: Settings,
        *,
        channel_options: tuple[tuple[str, int], ...] = _DEFAULT_CHANNEL_OPTIONS,
    ) -> None:
        if not settings.grpc_target:
            raise ValueError("Settings.grpc_target must be set (e.g. GRPC_TARGET=host:port)")
        self._settings = settings
        self._channel: grpc.aio.Channel = grpc.aio.insecure_channel(
            settings.grpc_target,
            options=channel_options,
        )
        self._stub = internal_communication_pb2_grpc.InternalServiceStub(self._channel)

    @property
    def channel(self) -> grpc.aio.Channel:
        return self._channel

    async def get_telemetry(
        self,
        timeout: float | None = None,
    ) -> internal_communication_pb2.TelemetryResponse:
        """Call ``GetTelemetry``; uses ``Settings.grpc_timeout_s`` when ``timeout`` is omitted."""
        t = self._settings.grpc_timeout_s if timeout is None else timeout
        return await self._stub.GetTelemetry(
            internal_communication_pb2.Empty(),
            timeout=t,
        )

    async def get_prompt(
        self,
        timeout: float | None = None,
    ) -> internal_communication_pb2.PromptResponse:
        t = self._settings.grpc_timeout_s if timeout is None else timeout
        return await self._stub.GetPrompt(
            internal_communication_pb2.Empty(),
            timeout=t,
        )

    async def start_mission(
        self,
        mission: internal_communication_pb2.MissionItemList,
        timeout: float | None = None,
    ) -> internal_communication_pb2.Empty:
        t = self._settings.grpc_timeout_s if timeout is None else timeout
        return await self._stub.StartMission(
            mission,
            timeout=t,
        )

    async def close(self) -> None:
        await self._channel.close()

    async def __aenter__(self) -> "InternalGrpcClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
