from __future__ import annotations

from ...uri import uri


TENSOR_CLASS_URI = uri("state", "collection", "tensor")


def tensor_route(segment: str) -> str:
    return str(uri("state", "collection", "tensor", segment))


__all__ = ["TENSOR_CLASS_URI", "tensor_route"]
