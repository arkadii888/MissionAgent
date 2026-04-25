from collections.abc import Mapping
from typing import Any

from .context import ExpansionContext


def handle_comb_square_area(ctx: ExpansionContext, intent: Mapping[str, Any]) -> None:
    del ctx, intent
    raise NotImplementedError(
        "comb_square_area intent is scaffolded but not implemented yet. "
        "Add schema branch, register this handler, and add tests before enabling."
    )
