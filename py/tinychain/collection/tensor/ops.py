from __future__ import annotations

from ...state.scalar import Scalar, autobox
from .core import Tensor
from .routes import tensor_route


def split(tensor: "Tensor", num_or_size_splits: object, axis: object = 0) -> Scalar:
    return Scalar._post_ref(
        tensor_route("split"),
        {
            "tensor": tensor,
            "num_or_size_splits": autobox(num_or_size_splits),
            "axis": autobox(axis),
        },
    )


def concatenate(tensors: object, axis: object = None) -> "Tensor":
    params = {"tensors": autobox(tensors)}
    if axis is not None:
        params["axis"] = autobox(axis)
    return Tensor._post_ref(tensor_route("concatenate"), params)


def einsum(format: str, tensors: object) -> "Tensor":
    return Tensor._post_ref(tensor_route("einsum"), {"format": autobox(format), "tensors": autobox(tensors)})


def tile(tensor: "Tensor", multiples: object) -> "Tensor":
    return tensor.tile(multiples)


__all__ = ["split", "concatenate", "einsum", "tile"]
