from __future__ import annotations

from typing import Type

from .state import Scalar, autobox


def _gcs(*types: Type) -> Type:
    if not types:
        return object

    mros = [list(t.__mro__) for t in types]
    for candidate in mros[0]:
        if all(candidate in mro for mro in mros[1:]):
            return candidate
    return object


def cond(condition, then, or_else=None):
    """
    Resolve either `then` or `or_else` based on the resolved value of `condition`.

    Returns the most specific common subtype when possible; falls back to `Scalar`.
    """
    if or_else is None:
        rtype = type(then) if isinstance(then, Scalar) else Scalar
    elif isinstance(then, Scalar) and isinstance(or_else, Scalar):
        rtype = _gcs(type(then), type(or_else))
    else:
        rtype = Scalar

    result = Scalar.if_ref(condition, autobox(then), autobox(or_else))
    return result if rtype is Scalar else result
