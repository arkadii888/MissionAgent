"""Small CLI to verify gRPC telemetry connectivity (standalone smoke test)."""

import asyncio
from dataclasses import replace

import grpc

from .config import Settings
from .grpc_client import InternalGrpcClient


def _load_settings() -> Settings:
    base = Settings.from_env()
    if base.grpc_target is None:
        return replace(base, grpc_target="localhost:50051")
    return base


async def get_telemetry() -> None:
    """Print one ``GetTelemetry`` response or the RPC status on failure."""
    settings = _load_settings()
    async with InternalGrpcClient(settings) as client:
        try:
            response = await client.get_telemetry()
            print("Data:")
            print(f"Latitude: {response.latitude_deg}")
            print(f"Longitude: {response.longitude_deg}")
        except grpc.RpcError as e:
            print(f"Error: {e.code()}")


if __name__ == "__main__":
    asyncio.run(get_telemetry())
