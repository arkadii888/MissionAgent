"""Float parsing helpers for telemetry and intent payloads (reject NaN/inf)."""

from collections.abc import Mapping
from typing import Any


def _finite(value: float, key: str) -> float:
    if value != value:
        raise ValueError(f"{key} must be finite")
    return value


def require_float(mapping: Mapping[str, Any], key: str) -> float:
    """Require ``mapping[key]`` and return it as a finite float.

    Raises:
        ValueError: Missing key or non-finite value.
    """
    if key not in mapping:
        raise ValueError(f"intent field {key!r} is required")
    return _finite(float(mapping[key]), key)


def optional_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
    """Like ``mapping.get(key, default)`` but requires a finite float.

    Raises:
        ValueError: If the resolved value is not finite.
    """
    value = mapping.get(key, default)
    return _finite(float(value), key)
