"""Async gRPC client for the vehicle / companion ``InternalService`` stubs."""

import grpc.aio
from typing import Any

from .config import Settings
from .protoc import internal_communication_pb2
from .protoc import internal_communication_pb2_grpc

_DEFAULT_CHANNEL_OPTIONS: tuple[tuple[str, int], ...] = (
    ("grpc.keepalive_time_ms", 10_000),
    ("grpc.keepalive_timeout_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
)


class InternalGrpcClient:
    """Thin async wrapper around ``InternalService`` (telemetry, prompt, mission upload).

    Uses an insecure channel with keepalive options suitable for NATs or load balancers.
    Prefer TLS or Unix sockets where your deployment supports them.

    Raises:
        ValueError: If ``settings.grpc_target`` is empty.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        channel_options: tuple[tuple[str, int], ...] = _DEFAULT_CHANNEL_OPTIONS,
    ) -> None:
        """Open an insecure gRPC channel to ``settings.grpc_target``.

        Args:
            settings: Must include a non-empty ``grpc_target``.
            channel_options: Optional gRPC channel option tuples.

        Raises:
            ValueError: If ``settings.grpc_target`` is empty.
        """
        if not settings.grpc_target:
            raise ValueError("Settings.grpc_target must be set (e.g. GRPC_TARGET=host:port)")
        self._settings = settings
        self._channel: grpc.aio.Channel = grpc.aio.insecure_channel(
            settings.grpc_target,
            options=channel_options,
        )
        self._stub = internal_communication_pb2_grpc.InternalServiceStub(self._channel)

    async def get_telemetry(
        self,
        timeout: float | None = None,
    ) -> internal_communication_pb2.TelemetryResponse:
        """Fetch latest vehicle telemetry.

        Args:
            timeout: RPC deadline in seconds; defaults to ``Settings.grpc_timeout_s``.
        """
        t = self._settings.grpc_timeout_s if timeout is None else timeout
        return await self._stub.GetTelemetry(
            internal_communication_pb2.Empty(),
            timeout=t,
        )

    async def get_prompt(
        self,
        timeout: float | None = None,
    ) -> internal_communication_pb2.PromptResponse:
        """Return the operator mission prompt string from the vehicle side.

        Args:
            timeout: RPC deadline in seconds; defaults to ``Settings.grpc_timeout_s``.
        """
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
        """Upload a full mission (waypoint list) to the vehicle.

        Args:
            mission: Populated ``MissionItemList`` protobuf.
            timeout: RPC deadline in seconds; defaults to ``Settings.grpc_timeout_s``.
        """
        t = self._settings.grpc_timeout_s if timeout is None else timeout
        return await self._stub.StartMission(
            mission,
            timeout=t,
        )

    async def close(self) -> None:
        """Close the channel; safe to call multiple times from ``__aexit__``."""
        await self._channel.close()

    async def __aenter__(self) -> "InternalGrpcClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
