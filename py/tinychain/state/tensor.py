from __future__ import annotations

from collections.abc import Iterable as IterableABC
from numbers import Number as NumberABC
from ..uri import URI, uri
from .scalar import (
    Bool,
    Comparable,
    Number,
    Scalar,
    Tuple,
    autobox,
)


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
    """TinyChain tensor.

    A tensor may be symbolic (backed by an IR ref/op) or materialized (backed by
    eager backend data). User code should not need to distinguish those cases.
    """

    __slots__ = ("_native",)

    __uri__: URI = uri("state", "collection", "tensor")

    def __init__(
        self,
        form: object = None,
        *,
        ref=None,
        native: object = None,
    ):
        if native is not None and (form is not None or ref is not None):
            raise TypeError("Tensor accepts either native data or symbolic form/ref")
        super().__init__(form, ref=ref)
        self._native = native

    def _tensor_post(
        self,
        operator_segment: str,
        params: dict[str, object],
        *,
        rtype,
    ):
        return self._post(operator_segment, params, rtype=rtype)

    @property
    def native(self) -> object | None:
        return self._native

    def _native_attr(self, name: str) -> object | None:
        if self._native is None:
            return None
        attr = getattr(self._native, name)
        return attr() if callable(attr) else attr

    @property
    def dtype(self) -> object:
        dtype = self._native_attr("dtype")
        if dtype is not None:
            return dtype
        return self._post("dtype", rtype=Scalar)

    @property
    def ndim(self) -> Number:
        return self._post("ndim", rtype=Number)

    @property
    def shape(self) -> object:
        shape = self._native_attr("shape")
        if shape is not None:
            return shape
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
        right = autobox(other)
        result = self._tensor_post("matmul", {"r": right}, rtype=Tensor)
        return result

    def logical_and(self, other: object) -> "Tensor":
        return self._binary_tensor("and", other)

    def logical_not(self) -> "Tensor":
        result = self._tensor_post("not", {}, rtype=Tensor)
        return result

    def logical_or(self, other: object) -> "Tensor":
        return self._binary_tensor("or", other)

    def logical_xor(self, other: object) -> "Tensor":
        return self._binary_tensor("xor", other)

    def tile(self, multiples: object) -> "Tensor":
        return Tensor._post_ref(str(uri(Tensor, "tile")), {"tensor": self, "multiples": autobox(multiples)})

    def __getitem__(self, bounds: object) -> "Tensor":
        return self.slice(bounds)

    def __matmul__(self, other: object) -> "Tensor":
        return self.matmul(other)

    def _binary_tensor(self, operator_segment: str, other: object) -> "Tensor":
        left = self
        right = autobox(other)
        return left._tensor_post(operator_segment, {"r": right}, rtype=Tensor)

    def __add__(self, other: object) -> "Tensor":
        return self._binary_tensor("add", other)

    def __sub__(self, other: object) -> "Tensor":
        return self._binary_tensor("sub", other)

    def __mul__(self, other: object) -> "Tensor":
        return self._binary_tensor("mul", other)

    def __truediv__(self, other: object) -> "Tensor":
        return self._binary_tensor("div", other)

    def __radd__(self, other: object) -> "Tensor":
        return self._binary_tensor("add", other)

    def __rsub__(self, other: object) -> "Tensor":
        left = autobox(other)
        if isinstance(left, Tensor):
            return left._binary_tensor("sub", self)
        if isinstance(other, NumberABC):
            raise TypeError("Tensor reverse subtraction requires Tensor lhs; literal promotion is not implemented")
        raise TypeError(f"unsupported Tensor reverse subtraction operand {type(other).__name__}")

    def __rmul__(self, other: object) -> "Tensor":
        return self._binary_tensor("mul", other)

    def __rtruediv__(self, other: object) -> "Tensor":
        left = autobox(other)
        if isinstance(left, Tensor):
            return left._binary_tensor("div", self)
        if isinstance(other, NumberABC):
            raise TypeError("Tensor reverse division requires Tensor lhs; literal promotion is not implemented")
        raise TypeError(f"unsupported Tensor reverse division operand {type(other).__name__}")

    @property
    def values(self) -> object:
        native = self._native_attr("values")
        if native is None:
            raise AttributeError("symbolic Tensor has no materialized values")
        return native

    def to_json(self) -> object:
        if self._native is None:
            return super().to_json()

        return {
            str(uri("state", "collection", "tensor")): [
                [self.dtype, self.shape],
                self.values,
            ]
        }


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
