from .config import Settings

__all__ = ["Settings", "InternalGrpcClient"]


def __getattr__(name: str):
    if name == "InternalGrpcClient":
        from .grpc_client import InternalGrpcClient

        return InternalGrpcClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
