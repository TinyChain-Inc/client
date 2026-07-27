from __future__ import annotations

from ...state.base import State
from ...uri import uri


TENSOR_CLASS_URI = uri(State, "collection", "tensor")


def tensor_route(segment: str) -> str:
    return str(uri(TENSOR_CLASS_URI, segment))


__all__ = ["TENSOR_CLASS_URI", "tensor_route"]
