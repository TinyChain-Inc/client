from __future__ import annotations

from collections.abc import Iterable as IterableABC
from ..uri import URI, uri
from .scalar import Bool, Comparable, Number, Scalar, Tuple, autobox


def _params(**kwargs: object) -> dict[str, Scalar]:
    return {key: autobox(value) for key, value in kwargs.items() if value is not None}


def _reduce_args(axes: object = None, keepdims: bool = False) -> dict[str, Scalar]:
    params: dict[str, Scalar] = {}
    if axes is not None:
        if isinstance(axes, IterableABC) and not isinstance(axes, (str, bytes, bytearray)):
            params["axes"] = autobox(list(axes))
        else:
            params["axes"] = autobox([axes])
    if keepdims:
        params["keepdims"] = autobox(keepdims)
    return params


class Tensor(Comparable):
    """Symbolic TinyChain tensor.

    This is the Python authoring facade for `/state/collection/tensor`. Its
    methods mirror the v1 Tensor surface but only build canonical TinyChain op
    references; execution mode is still controlled by the active backend.
    """

    __uri__: URI = uri("state", "collection", "tensor")

    @property
    def dtype(self) -> Scalar:
        return self._post("dtype", rtype=Scalar)

    @property
    def ndim(self) -> Number:
        return self._post("ndim", rtype=Number)

    @property
    def shape(self) -> Tuple:
        return self._post("shape", rtype=Tuple)

    @property
    def size(self) -> Number:
        return self._post("size", rtype=Number)

    def all(self) -> Bool:
        return self._post("all", rtype=Bool)

    def any(self) -> Bool:
        return self._post("any", rtype=Bool)

    def broadcast(self, shape: object) -> "Tensor":
        return self._get("broadcast", autobox(shape), rtype=Tensor)

    def cast(self, number_type: object) -> "Tensor":
        return self._get("cast", autobox(number_type), rtype=Tensor)

    def copy(self) -> "Tensor":
        return Tensor._post_ref(str(uri(Tensor, "copy_from")), {"tensor": self})

    def expand_dims(self, axes: object = None) -> "Tensor":
        return self._get("expand_dims", autobox(axes), rtype=Tensor)

    def cond(self, then: object, or_else: object) -> "Tensor":
        return self._post("cond", {"then": autobox(then), "or_else": autobox(or_else)}, rtype=Tensor)

    def max(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("max", _reduce_args(axes, keepdims), rtype=Scalar)

    def min(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("min", _reduce_args(axes, keepdims), rtype=Scalar)

    def mean(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("mean", _reduce_args(axes, keepdims), rtype=Scalar)

    def norm(self, axis: object = None, keepdims: bool = False) -> Scalar:
        return self._post("norm", _params(axis=axis, keepdims=keepdims if keepdims else None), rtype=Scalar)

    def product(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("product", _reduce_args(axes, keepdims), rtype=Scalar)

    def reshape(self, shape: object) -> "Tensor":
        return self._get("reshape", autobox(shape), rtype=Tensor)

    def slice(self, bounds: object) -> "Tensor":
        return self._get(key=autobox(bounds), rtype=Tensor)

    def std(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("std", _reduce_args(axes, keepdims), rtype=Scalar)

    def sum(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("sum", _reduce_args(axes, keepdims), rtype=Scalar)

    def transpose(self, permutation: object = None) -> "Tensor":
        return self._get("transpose", autobox(permutation), rtype=Tensor)

    def write(self, value: object) -> "Tensor":
        return self._put(autobox(value), rtype=Tensor)

    def matmul(self, other: object) -> "Tensor":
        return self._post("matmul", {"r": autobox(other)}, rtype=Tensor)

    def logical_and(self, other: object) -> "Tensor":
        return self._post("and", {"r": autobox(other)}, rtype=Tensor)

    def logical_not(self) -> "Tensor":
        return self._post("not", rtype=Tensor)

    def logical_or(self, other: object) -> "Tensor":
        return self._post("or", {"r": autobox(other)}, rtype=Tensor)

    def logical_xor(self, other: object) -> "Tensor":
        return self._post("xor", {"r": autobox(other)}, rtype=Tensor)

    def tile(self, multiples: object) -> "Tensor":
        return Tensor._post_ref(str(uri(Tensor, "tile")), {"tensor": self, "multiples": autobox(multiples)})

    def __getitem__(self, bounds: object) -> "Tensor":
        return self.slice(bounds)

    def __matmul__(self, other: object) -> "Tensor":
        return self.matmul(other)

    def _binary_tensor(self, op_name: str, other: object) -> "Tensor":
        return self._post(op_name, {"r": autobox(other)}, rtype=Tensor)

    def __add__(self, other: object) -> "Tensor":
        return self._binary_tensor("add", other)

    def __sub__(self, other: object) -> "Tensor":
        return self._binary_tensor("sub", other)

    def __mul__(self, other: object) -> "Tensor":
        return self._binary_tensor("mul", other)

    def __truediv__(self, other: object) -> "Tensor":
        return self._binary_tensor("div", other)

    def __radd__(self, other: object) -> "Tensor":
        return autobox(other)._post("add", {"r": self}, rtype=Tensor)

    def __rsub__(self, other: object) -> "Tensor":
        return autobox(other)._post("sub", {"r": self}, rtype=Tensor)

    def __rmul__(self, other: object) -> "Tensor":
        return autobox(other)._post("mul", {"r": self}, rtype=Tensor)

    def __rtruediv__(self, other: object) -> "Tensor":
        return autobox(other)._post("div", {"r": self}, rtype=Tensor)


def split(tensor: Tensor, num_or_size_splits: object, axis: object = 0) -> Scalar:
    return Scalar._post_ref(
        str(uri(Tensor, "split")),
        {
            "tensor": tensor,
            "num_or_size_splits": autobox(num_or_size_splits),
            "axis": autobox(axis),
        },
    )


def concatenate(tensors: object, axis: object = None) -> Tensor:
    params = {"tensors": autobox(tensors)}
    if axis is not None:
        params["axis"] = autobox(axis)
    return Tensor._post_ref(str(uri(Tensor, "concatenate")), params)


def einsum(format: str, tensors: object) -> Tensor:
    return Tensor._post_ref(str(uri(Tensor, "einsum")), {"format": autobox(format), "tensors": autobox(tensors)})


def tile(tensor: Tensor, multiples: object) -> Tensor:
    return tensor.tile(multiples)


__all__ = ["Tensor", "concatenate", "einsum", "split", "tile"]
