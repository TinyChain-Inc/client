from __future__ import annotations

from .state import autobox, cond as state_cond


def cond(condition, then, or_else=None):
    """
    Resolve either `then` or `or_else` based on the resolved value of `condition`.

    For symbolic conditions this delegates to `tc.state.cond`.
    """
    if isinstance(condition, bool):
        return then if condition else or_else

    return state_cond(condition, autobox(then), autobox(or_else))
