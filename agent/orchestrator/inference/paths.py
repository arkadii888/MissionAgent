"""Locate files under ``models/`` across different clone layouts (single or nested ``agent/``)."""

from __future__ import annotations

from pathlib import Path

_INFERENCE_PKG = Path(__file__).resolve().parent


def resolve_model_file(relative: str | Path) -> Path:
    """Resolve a path like ``models/yolo26n_b8.hef`` to an absolute path.

    Walks upward from this package directory and returns the first place where the file
    exists. If it is not found, returns a legacy default under the agent package root.

    Args:
        relative: Path relative to a discovered ``models/`` directory, or absolute.

    Returns:
        Resolved absolute path (may not exist if lookup failed).
    """
    rel = Path(relative)
    if rel.is_absolute():
        return rel.resolve()
    cur: Path = _INFERENCE_PKG
    for _ in range(14):
        cand = (cur / rel).resolve()
        if cand.is_file():
            return cand
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return (_INFERENCE_PKG.parents[2] / rel).resolve()
