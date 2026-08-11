from __future__ import annotations

from .state import autobox, cond as state_cond


def cond(condition, then, or_else=None):
    """
    Resolve either `then` or `or_else` based on the resolved value of `condition`.

    A concrete Python condition returns its selected branch. Symbolic conditions
    compile through the canonical scalar control-flow helper.
    """
    if isinstance(condition, bool):
        return then if condition else or_else
    return state_cond(condition, autobox(then), autobox(or_else))
