from __future__ import annotations

from numbers import Number as NumberABC

from ...uri import URI
from ...state.collection import Collection
from ...state.scalar import (
    Bool,
    Comparable,
    Number,
    Scalar,
    Tuple,
    autobox,
)
from ...state.value import Number as ValueNumber
from ...autodiff.tracing import record_operation
from ._common import params, reduce_args
from .routes import TENSOR_CLASS_URI, tensor_route

_DTYPE_URIS = {
    f"{short}{bits}": str(URI(ValueNumber, family, str(bits)))
    for short, family, widths in (
        ("i", "int", (8, 16, 32, 64)),
        ("u", "uint", (8, 16, 32, 64)),
        ("f", "float", (32, 64)),
        ("c", "complex", (32, 64)),
    )
    for bits in widths
}
_DTYPES_BY_URI = {uri: dtype for dtype, uri in _DTYPE_URIS.items()}
_DTYPE_ALIASES = {
    f"{family}{dtype[1:]}": dtype
    for dtype in _DTYPE_URIS
    for family in ({"i": "int", "u": "uint", "f": "float", "c": "complex"}[dtype[0]],)
}


def _normalize_dtype(dtype: str) -> str:
    dtype = _DTYPES_BY_URI.get(dtype, dtype.strip().lower())
    dtype = _DTYPE_ALIASES.get(dtype, dtype)
    if dtype not in _DTYPE_URIS:
        raise TypeError(f"unsupported tensor dtype {dtype}")
    return dtype


def _decode_number(dtype: str, value: object) -> int | float | complex:
    if dtype.startswith("c"):
        if not isinstance(value, list) or len(value) != 2:
            raise TypeError("complex Tensor values must be [real, imaginary]")
        return complex(float(value[0]), float(value[1]))
    return float(value) if dtype.startswith("f") else int(value)


def _encode_number(value: int | float | complex) -> object:
    return [value.real, value.imag] if isinstance(value, complex) else value


def _is_sparse(values: list) -> bool:
    return not values or (
        isinstance(values[0], list) and bool(values[0]) and isinstance(values[0][0], list)
    )


def _validate_shape(shape):
    shape = list(shape)
    if not shape or len(shape) > 4096 or any(
        type(dim) is not int or not 0 < dim < 2**64 for dim in shape
    ):
        raise ValueError("Tensor shape must contain at most 4096 positive u64 dimensions")
    size = 1
    for dim in shape:
        size *= dim
    if size >= 2**64:
        raise ValueError("Tensor logical size exceeds u64")
    return shape, size


class Tensor(Collection, Comparable):
    """TinyChain tensor.

    A tensor may be symbolic (backed by an IR ref/op) or materialized (backed by
    a validated native payload). User code should not need to distinguish those cases.
    """

    __slots__ = ()

    __uri__: URI = TENSOR_CLASS_URI

    @classmethod
    def dense(
        cls,
        dtype: str,
        shape: list[int] | tuple[int, ...],
        values: list[int | float | complex] | tuple[int | float | complex, ...],
    ) -> "Tensor":
        dtype = _normalize_dtype(dtype)
        if dtype.startswith("c"):
            values = [complex(value) for value in values]
        shape, size = _validate_shape(shape)
        if len(values) != size:
            raise ValueError(f"dense Tensor shape requires {size} values, found {len(values)}")
        return cls({str(TENSOR_CLASS_URI): [[_DTYPE_URIS[dtype], shape], [_encode_number(value) for value in values]]})

    @classmethod
    def sparse(
        cls,
        dtype: str,
        shape: list[int] | tuple[int, ...],
        entries: list[tuple[list[int] | tuple[int, ...], int | float | complex]],
    ) -> "Tensor":
        """Construct a tensor from ordered nonzero entries with implicit zeros."""
        dtype = _normalize_dtype(dtype)
        shape, _ = _validate_shape(shape)
        values = []
        previous = None
        for coord, value in entries:
            coord = list(coord)
            if len(coord) != len(shape) or any(type(i) is not int or not 0 <= i < dim for i, dim in zip(coord, shape)):
                raise ValueError("sparse Tensor coordinate is outside its shape")
            if previous is not None and previous >= coord:
                raise ValueError("sparse Tensor coordinates must be ordered and unique")
            if value == 0:
                raise ValueError("sparse Tensor entries must be nonzero")
            previous = coord
            if dtype.startswith("c"):
                value = complex(value)
            values.append([coord, _encode_number(value)])
        return cls({str(TENSOR_CLASS_URI): [[_DTYPE_URIS[dtype], shape], values]})

    @classmethod
    def _from_payload(cls, payload: object) -> "Tensor":
        if not (isinstance(payload, list) and len(payload) == 2):
            raise TypeError("Tensor payload must be [metadata, values]")
        meta, values = payload
        if not (
            isinstance(meta, list)
            and len(meta) == 2
            and isinstance(meta[0], str)
            and isinstance(meta[1], list)
            and isinstance(values, list)
        ):
            raise TypeError("invalid Tensor payload")

        dtype = _DTYPES_BY_URI.get(meta[0])
        if dtype is None:
            raise TypeError(f"unsupported tensor dtype {meta[0]}")
        if not _is_sparse(values):
            return cls.dense(dtype, meta[1], [_decode_number(dtype, value) for value in values])
        entries = []
        for entry in values:
            if not (isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], list)):
                raise TypeError("sparse Tensor entries must be [coordinate, value]")
            entries.append((entry[0], _decode_number(dtype, entry[1])))
        return cls.sparse(dtype, meta[1], entries)

    @property
    def dtype(self) -> object:
        if isinstance(self._form, dict) and str(TENSOR_CLASS_URI) in self._form:
            return _DTYPES_BY_URI[self._form[str(TENSOR_CLASS_URI)][0][0]]
        return self._post("dtype", rtype=Scalar)

    @property
    def ndim(self) -> Number:
        return self._post("ndim", rtype=Number)

    @property
    def shape(self) -> object:
        if isinstance(self._form, dict) and str(TENSOR_CLASS_URI) in self._form:
            return list(self._form[str(TENSOR_CLASS_URI)][0][1])
        return self._post("shape", rtype=Tuple)

    @property
    def size(self) -> Number:
        return self._post("size", rtype=Number)

    def all(self) -> Bool:
        return self._post("all", rtype=Bool)

    def any(self) -> Bool:
        return self._post("any", rtype=Bool)

    def as_dense(self) -> "Tensor":
        """Include implicit zeros in operations which require a dense result."""
        return self._get("as_dense", rtype=Tensor)

    def broadcast(self, shape: object) -> "Tensor":
        return self._get("broadcast", autobox(shape), rtype=Tensor)

    def cast(self, number_type: object) -> "Tensor":
        return self._get("cast", autobox(number_type), rtype=Tensor)

    def copy(self) -> "Tensor":
        return Tensor._post_ref(tensor_route("copy_from"), {"tensor": self})

    def exp(self) -> "Tensor":
        """Exponentiate dense values; convert sparse inputs with ``as_dense`` first."""
        return self._post("exp", {}, rtype=Tensor)

    def expand_dims(self, axes: object = None) -> "Tensor":
        return self._get("expand_dims", autobox(axes), rtype=Tensor)

    def cond(self, then: object, or_else: object) -> "Tensor":
        return self._post("cond", {"then": autobox(then), "or_else": autobox(or_else)}, rtype=Tensor)

    def _reduction_post(
        self,
        operator_segment: str,
        axes: object,
        keepdims: bool,
        *,
        capture_operation: str | None = None,
    ) -> Scalar:
        """Build the canonical reduction result, then record it if capture is requested.

        Shared choke point for reduction operations: the symbolic/eager result is
        constructed exactly as before via the ordinary route helper, so inactive
        behavior is unchanged. When ``capture_operation`` is given and a trace is
        active, the recorder captures one node after the result is built.
        """
        result = self._post(operator_segment, reduce_args(axes, keepdims), rtype=Scalar)
        if capture_operation is not None:
            record_operation(capture_operation, (self,), result, {"axes": axes, "keepdims": keepdims})
        return result

    def max(self, axes: object = None, keepdims: bool = False) -> Scalar:
        """Returns Scalar. Autodiff (VJP) for reductions is unsupported in Phase 1."""
        return self._post("max", reduce_args(axes, keepdims), rtype=Scalar)

    def min(self, axes: object = None, keepdims: bool = False) -> Scalar:
        """Returns Scalar. Autodiff (VJP) for reductions is unsupported in Phase 1."""
        return self._post("min", reduce_args(axes, keepdims), rtype=Scalar)

    def mean(self, axes: object = None, keepdims: bool = False) -> Scalar:
        """Returns Scalar. Under an active typed trace, mean is captured as a
        MeanOperator node with normalized axes and keepdims; the public return
        type is still Scalar."""
        return self._reduction_post("mean", axes, keepdims, capture_operation="mean")

    def norm(self, axis: object = None, keepdims: bool = False) -> Scalar:
        """Returns Scalar. Autodiff (VJP) for reductions is unsupported in Phase 1."""
        return self._post("norm", params(axis=axis, keepdims=keepdims if keepdims else None), rtype=Scalar)

    def product(self, axes: object = None, keepdims: bool = False) -> Scalar:
        """Returns Scalar. Autodiff (VJP) for reductions is unsupported in Phase 1."""
        return self._post("product", reduce_args(axes, keepdims), rtype=Scalar)

    def reshape(self, shape: object) -> "Tensor":
        return self._get("reshape", autobox(shape), rtype=Tensor)

    def slice(self, bounds: object) -> "Tensor":
        """Select values; ``write`` addresses this Tensor with the same bounds.

        The returned native view is read-only. Further selections or transforms
        address that view, so their writes are rejected by the host.
        """
        parent = self
        bounds = autobox(bounds)

        class Selection(Tensor):
            __slots__ = ()

            def write(self, value: object) -> Tensor:
                return parent._put(autobox(value), key=bounds, rtype=Tensor)

        return self._get(key=bounds, rtype=Selection)

    def std(self, axes: object = None, keepdims: bool = False) -> Scalar:
        """Returns Scalar. Autodiff (VJP) for reductions is unsupported in Phase 1."""
        return self._post("std", reduce_args(axes, keepdims), rtype=Scalar)

    def sum(self, axes: object = None, keepdims: bool = False) -> Scalar:
        """Returns Scalar. Autodiff (VJP) for reductions is unsupported in Phase 1."""
        return self._post("sum", reduce_args(axes, keepdims), rtype=Scalar)

    def transpose(self, permutation: object = None) -> "Tensor":
        result = self._get("transpose", autobox(permutation), rtype=Tensor)
        record_operation("transpose", (self,), result, {"permutation": permutation})
        return result

    def write(self, value: object) -> "Tensor":
        return self._put(autobox(value), rtype=Tensor)

    def matmul(self, other: object) -> "Tensor":
        right = autobox(other)
        result = self._post("matmul", {"r": right}, rtype=Tensor)
        record_operation("matmul", (self, other), result)
        return result

    def logical_and(self, other: object) -> "Tensor":
        return self._binary_tensor("and", other)

    def logical_not(self) -> "Tensor":
        result = self._post("not", {}, rtype=Tensor)
        return result

    def logical_or(self, other: object) -> "Tensor":
        return self._binary_tensor("or", other)

    def logical_xor(self, other: object) -> "Tensor":
        return self._binary_tensor("xor", other)

    def tile(self, multiples: object) -> "Tensor":
        return Tensor._post_ref(tensor_route("tile"), {"tensor": self, "multiples": autobox(multiples)})

    def __getitem__(self, bounds: object) -> "Tensor":
        return self.slice(bounds)

    def __matmul__(self, other: object) -> "Tensor":
        return self.matmul(other)

    def _binary_tensor(self, operator_segment: str, other: object) -> "Tensor":
        left = self
        right = autobox(other)
        return left._post(operator_segment, {"r": right}, rtype=Tensor)

    def __add__(self, other: object) -> "Tensor":
        result = self._binary_tensor("add", other)
        record_operation("add", (self, other), result)
        return result

    def __sub__(self, other: object) -> "Tensor":
        result = self._binary_tensor("sub", other)
        record_operation("sub", (self, other), result)
        return result

    def __mul__(self, other: object) -> "Tensor":
        result = self._binary_tensor("mul", other)
        record_operation("mul", (self, other), result)
        return result

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
    def _payload(self):
        if isinstance(self._form, dict) and str(TENSOR_CLASS_URI) in self._form:
            return self._form[str(TENSOR_CLASS_URI)]
        raise AttributeError("Tensor payload requires a materialized client value")

    @property
    def layout(self) -> str:
        return "sparse" if _is_sparse(self._payload[1]) else "dense"

    @property
    def entries(self) -> list[tuple[list[int], int | float | complex]]:
        """Return the stored entries of a materialized sparse Tensor."""
        if self.layout != "sparse":
            raise AttributeError("Tensor entries require a materialized sparse value")
        return [(list(coord), _decode_number(self.dtype, value)) for coord, value in self._payload[1]]

    @property
    def values(self) -> object:
        """Return stored values; sparse values exclude implicit zeros."""
        if self.layout == "sparse":
            return [value for _, value in self.entries]
        return [_decode_number(self.dtype, value) for value in self._payload[1]]


__all__ = ["Tensor"]
