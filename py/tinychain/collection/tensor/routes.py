from __future__ import annotations

from ...state.base import State
from ...uri import URI

TENSOR_CLASS_URI = URI(path=URI.of(State, "collection", "tensor"))


def tensor_route(segment: str) -> str:
    return str(URI(path=URI.of(TENSOR_CLASS_URI, segment)))


__all__ = ["TENSOR_CLASS_URI", "tensor_route"]
