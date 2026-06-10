from __future__ import annotations

from .state import autobox
from .state.context import current_context


def after(dependency, then):
    """
    Encode an explicit execution-order dependency and return `then` unchanged.

    When called during route compilation (inside an active TinyChain context), this
    helper binds `dependency` into the current context form so `then` is evaluated
    after it. Returning `then` preserves the caller's type wrapper for method chaining.
    """
    ctx = current_context()
    if ctx is not None:
        ctx.bind_auto(autobox(dependency), prefix="_after")
    return then
