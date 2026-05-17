"""Registry mapping intent ``type`` strings to expansion callables."""

from collections.abc import Callable, Mapping
from typing import Any

from .context import ExpansionContext

IntentHandler = Callable[[ExpansionContext, Mapping[str, Any]], None]


class IntentRegistry:
    """Stores one handler per DSL intent ``type``; duplicate registration fails."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._handlers: dict[str, IntentHandler] = {}

    def register(self, intent_type: str, handler: IntentHandler) -> None:
        """Add ``handler`` for ``intent_type``.

        Raises:
            ValueError: If ``intent_type`` is already registered.
        """
        if intent_type in self._handlers:
            raise ValueError(f"duplicate handler registration for intent type {intent_type!r}")
        self._handlers[intent_type] = handler

    def resolve(self, intent_type: str) -> IntentHandler:
        """Look up handler by intent type string.

        Raises:
            ValueError: If unknown, listing supported types in the message.
        """
        try:
            return self._handlers[intent_type]
        except KeyError as exc:
            supported = ", ".join(sorted(self._handlers))
            raise ValueError(f"unsupported intent type {intent_type!r}; supported: {supported}") from exc
