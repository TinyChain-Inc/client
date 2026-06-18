from __future__ import annotations

from collections.abc import Callable
from numbers import Number as NumberABC
from typing import Literal

from ...uri import URI
from ..scalar import (
    Bool,
    Comparable,
    IdRef,
    Number,
    Scalar,
    Tuple,
    autobox,
    tcref_form_of,
)
from ._common import infer_broadcast_axes, normalize_permutation, normalize_shape, params, reduce_args
from .backend import FensorWireTensorBackend, TensorBackend
from .routes import TENSOR_CLASS_URI, tensor_route
from .schema import FensorLayoutSchema, FensorTensorSchema, FensorViewSchema
from .view_ops import BroadcastViewOp, ReshapeViewOp, SliceViewOp, TensorViewOp, TransposeViewOp
from .view_spec import TensorViewSpec


class Tensor(Comparable):
    """TinyChain tensor.

    A tensor may be symbolic (backed by an IR ref/op) or materialized (backed by
    eager backend data). User code should not need to distinguish those cases.
    """

    __slots__ = ("_native", "_view_ops", "_subject_root", "_view_ops_materialized")

    __uri__: URI = TENSOR_CLASS_URI

    def __init__(
        self,
        form: object = None,
        *,
        ref=None,
        native: object = None,
        view_ops: tuple[TensorViewOp, ...] = (),
        subject_root: str | None = None,
        view_ops_materialized: bool | None = None,
    ):
        if native is not None and (form is not None or ref is not None):
            raise TypeError("Tensor accepts either native data or symbolic form/ref")
        super().__init__(form, ref=ref)
        self._native = native
        self._view_ops = view_ops
        if view_ops_materialized is not None:
            self._view_ops_materialized = view_ops_materialized
        else:
            self._view_ops_materialized = native is not None

        if subject_root is not None:
            self._subject_root = subject_root
        elif ref is not None:
            ref_form = tcref_form_of(ref)
            self._subject_root = ref_form.key() if isinstance(ref_form, IdRef) else None
        else:
            self._subject_root = None

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

    @property
    def view_ops(self) -> tuple[TensorViewOp, ...]:
        return self._view_ops

    def _append_view_op(self, op: TensorViewOp, result: "Tensor") -> "Tensor":
        result._view_ops = self._view_ops + (op,)
        result._subject_root = self._subject_root
        result._view_ops_materialized = False
        return result

    def view_spec(self) -> TensorViewSpec:
        return TensorViewSpec(ops=self._view_ops)

    def to_fensor_view_schema(self, *, base_shape: object | None = None) -> FensorViewSchema:
        if base_shape is None:
            inferred = self._native_attr("shape")
            if inferred is None:
                raise TypeError("base_shape is required for symbolic tensors")
            base_shape = inferred

        return self.view_spec().to_fensor_view_schema(base_shape=base_shape)

    def to_fensor_view_wire(self, *, base_shape: object | None = None) -> tuple[int, list[tuple[int, tuple[int, list[int]]]], list[int | None]]:
        return self.to_fensor_view_schema(base_shape=base_shape).to_wire()

    def to_fensor_tensor_schema(
        self,
        *,
        base_shape: object | None = None,
        dtype: str = "f32",
        layout: Literal["dense", "sparse"] = "dense",
        sparse_axis: int | None = None,
    ) -> FensorTensorSchema:
        if base_shape is None:
            inferred = self._native_attr("shape")
            if inferred is None:
                raise TypeError("base_shape is required for symbolic tensors")
            base_shape = inferred

        normalized = normalize_shape(base_shape)
        if normalized is None:
            raise TypeError("base_shape must be an iterable of integer dimensions")
        if not normalized:
            raise ValueError("base_shape must not be empty")

        layout_schema = FensorLayoutSchema(kind=layout, sparse_axis=sparse_axis)
        return FensorTensorSchema(dtype=dtype, shape=normalized, layout=layout_schema)

    def to_fensor_tensor_wire(
        self,
        *,
        base_shape: object | None = None,
        dtype: str = "f32",
        layout: Literal["dense", "sparse"] = "dense",
        sparse_axis: int | None = None,
    ) -> tuple[str, list[int], tuple[int, int | None]]:
        return self.to_fensor_tensor_schema(
            base_shape=base_shape,
            dtype=dtype,
            layout=layout,
            sparse_axis=sparse_axis,
        ).to_wire()

    def _transform_get(self, method: str, arg: object) -> "Tensor":
        if self._subject_root is None:
            return self._get(method, autobox(arg), rtype=Tensor)

        return Tensor._get_ref(f"{self._subject_root}/{method}", autobox(arg))

    def _native_transform(self, method: str, *args: object) -> object | None:
        if self._native is None:
            return None

        if isinstance(self._native, TensorBackend):
            op = {
                "transpose": TransposeViewOp(kind="transpose", permutation=normalize_permutation(args[0] if args else None)),
                "broadcast": BroadcastViewOp(kind="broadcast", shape=normalize_shape(args[0] if args else ()) or (), broadcast_axes=None),
                "reshape": ReshapeViewOp(kind="reshape", shape=normalize_shape(args[0] if args else ()) or ()),
                "slice": SliceViewOp(kind="slice", bounds=args[0] if args else None),
            }.get(method)

            if op is not None:
                try:
                    return self._native.apply_view_op(op)
                except NotImplementedError:
                    pass

        native_method = getattr(self._native, method, None)
        if native_method is None or not callable(native_method):
            return None

        try:
            return native_method(*args)
        except NotImplementedError:
            return None

    def _native_attr(self, name: str) -> object | None:
        if self._native is None:
            return None
        attr = getattr(self._native, name)
        return attr() if callable(attr) else attr

    def _apply_view_transform(
        self,
        *,
        method: str,
        arg: object,
        op: TensorViewOp,
        symbolic_builder: Callable[[], "Tensor"] | None = None,
    ) -> "Tensor":
        native = self._native_transform(method, arg)
        if native is not None:
            return Tensor(native=native, view_ops=self._view_ops + (op,), view_ops_materialized=True)

        result = symbolic_builder() if symbolic_builder is not None else self._transform_get(method, arg)
        return self._append_view_op(op, result)

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
        normalized_shape = normalize_shape(shape)
        source_shape = self._native_attr("shape")
        broadcast_axes = (
            infer_broadcast_axes(source_shape, normalized_shape)
            if normalized_shape is not None
            else None
        )
        op = BroadcastViewOp(
            kind="broadcast",
            shape=normalized_shape or (),
            broadcast_axes=broadcast_axes,
        )

        return self._apply_view_transform(method="broadcast", arg=shape, op=op)

    def cast(self, number_type: object) -> "Tensor":
        return self._get("cast", autobox(number_type), rtype=Tensor)

    def copy(self) -> "Tensor":
        return Tensor._post_ref(tensor_route("copy_from"), {"tensor": self})

    def expand_dims(self, axes: object = None) -> "Tensor":
        return self._get("expand_dims", autobox(axes), rtype=Tensor)

    def cond(self, then: object, or_else: object) -> "Tensor":
        return self._post("cond", {"then": autobox(then), "or_else": autobox(or_else)}, rtype=Tensor)

    def max(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("max", reduce_args(axes, keepdims), rtype=Scalar)

    def min(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("min", reduce_args(axes, keepdims), rtype=Scalar)

    def mean(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("mean", reduce_args(axes, keepdims), rtype=Scalar)

    def norm(self, axis: object = None, keepdims: bool = False) -> Scalar:
        return self._post("norm", params(axis=axis, keepdims=keepdims if keepdims else None), rtype=Scalar)

    def product(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("product", reduce_args(axes, keepdims), rtype=Scalar)

    def reshape(self, shape: object) -> "Tensor":
        op = ReshapeViewOp(kind="reshape", shape=normalize_shape(shape) or ())

        return self._apply_view_transform(method="reshape", arg=shape, op=op)

    def slice(self, bounds: object) -> "Tensor":
        op = SliceViewOp(kind="slice", bounds=bounds)

        def _symbolic_slice() -> "Tensor":
            if self._subject_root is None:
                return self._get(key=autobox(bounds), rtype=Tensor)
            return Tensor._get_ref(self._subject_root, autobox(bounds))

        return self._apply_view_transform(
            method="slice",
            arg=bounds,
            op=op,
            symbolic_builder=_symbolic_slice,
        )

    def std(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("std", reduce_args(axes, keepdims), rtype=Scalar)

    def sum(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("sum", reduce_args(axes, keepdims), rtype=Scalar)

    def transpose(self, permutation: object = None) -> "Tensor":
        op = TransposeViewOp(kind="transpose", permutation=normalize_permutation(permutation))

        return self._apply_view_transform(method="transpose", arg=permutation, op=op)

    def materialize_view_spec(self) -> "Tensor":
        """Apply the current view spec through the active backend if supported."""
        if self._native is None or not isinstance(self._native, TensorBackend):
            return self

        if self._view_ops_materialized:
            return self

        spec = self.view_spec()
        if not spec.ops:
            return self

        if isinstance(self._native, FensorWireTensorBackend):
            try:
                base_shape = self._native_attr("shape")
                if base_shape is not None:
                    wire = spec.to_fensor_view_wire(base_shape=base_shape)
                    native = self._native.apply_fensor_view_wire(wire)
                    return Tensor(
                        native=native,
                        view_ops=self._view_ops,
                        subject_root=self._subject_root,
                        view_ops_materialized=True,
                    )
            except NotImplementedError:
                pass

        try:
            native = self._native.apply_view_spec(spec)
        except NotImplementedError:
            return self

        return Tensor(
            native=native,
            view_ops=self._view_ops,
            subject_root=self._subject_root,
            view_ops_materialized=True,
        )

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
        return Tensor._post_ref(tensor_route("tile"), {"tensor": self, "multiples": autobox(multiples)})

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
            str(TENSOR_CLASS_URI): [
                [self.dtype, self.shape],
                self.values,
            ]
        }


__all__ = ["Tensor"]
