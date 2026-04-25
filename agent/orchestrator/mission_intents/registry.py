from collections.abc import Callable, Mapping
from typing import Any

from .context import ExpansionContext

IntentHandler = Callable[[ExpansionContext, Mapping[str, Any]], None]


class IntentRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, IntentHandler] = {}

    def register(self, intent_type: str, handler: IntentHandler) -> None:
        if intent_type in self._handlers:
            raise ValueError(f"duplicate handler registration for intent type {intent_type!r}")
        self._handlers[intent_type] = handler

    def resolve(self, intent_type: str) -> IntentHandler:
        try:
            return self._handlers[intent_type]
        except KeyError as exc:
            supported = ", ".join(sorted(self._handlers))
            raise ValueError(f"unsupported intent type {intent_type!r}; supported: {supported}") from exc

    @property
    def handlers(self) -> dict[str, IntentHandler]:
        return dict(self._handlers)
