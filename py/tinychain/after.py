from __future__ import annotations

from . import state


def after(dependency, then):
    """
    Encode an explicit execution-order dependency and return `then` unchanged.

    When called during route compilation (inside an active TinyChain context), this
    helper binds `dependency` into the current context form so `then` is evaluated
    after it. Returning `then` preserves the caller's type wrapper for method chaining.
    """
    state.after(dependency, then)
    return then
