from __future__ import annotations

from collections.abc import Callable
from numbers import Number as NumberABC
from typing import Literal

from ...uri import URI, path
from ...state.scalar import (
    Bool,
    Comparable,
    IdRef,
    Number,
    Scalar,
    Tuple,
    autobox,
    tcref_form_of,
)
from ...autodiff.graph import AddOperator, MatmulOperator, TensorNodeRecord, TransposeOperator, get_active_builder
from ._common import infer_broadcast_axes, normalize_permutation, normalize_shape, params, reduce_args
from ._wire import encode_view_schema
from .backend import TensorBackend, TensorWireTensorBackend
from .routes import TENSOR_CLASS_URI, tensor_route
from .schema import TensorStorageLayout, TensorStorageSchema, TensorViewSchema
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

    def to_view_schema(self, *, base_shape: object | None = None) -> TensorViewSchema:
        if base_shape is None:
            inferred = self._native_attr("shape")
            if inferred is None:
                raise TypeError("base_shape is required for symbolic tensors")
            base_shape = inferred

        return self.view_spec().to_view_schema(base_shape=base_shape)

    def to_storage_schema(
        self,
        *,
        base_shape: object | None = None,
        dtype: str = "f32",
        layout: Literal["dense", "sparse"] = "dense",
        sparse_axis: int | None = None,
    ) -> TensorStorageSchema:
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

        layout_schema = TensorStorageLayout(kind=layout, sparse_axis=sparse_axis)
        return TensorStorageSchema(dtype=dtype, shape=normalized, layout=layout_schema)


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
        """Returns Scalar. Autodiff (VJP) for reductions is unsupported in Phase 1."""
        return self._post("max", reduce_args(axes, keepdims), rtype=Scalar)

    def min(self, axes: object = None, keepdims: bool = False) -> Scalar:
        """Returns Scalar. Autodiff (VJP) for reductions is unsupported in Phase 1."""
        return self._post("min", reduce_args(axes, keepdims), rtype=Scalar)

    def mean(self, axes: object = None, keepdims: bool = False) -> Scalar:
        """Returns Scalar. Autodiff (VJP) for reductions is unsupported in Phase 1."""
        return self._post("mean", reduce_args(axes, keepdims), rtype=Scalar)

    def norm(self, axis: object = None, keepdims: bool = False) -> Scalar:
        """Returns Scalar. Autodiff (VJP) for reductions is unsupported in Phase 1."""
        return self._post("norm", params(axis=axis, keepdims=keepdims if keepdims else None), rtype=Scalar)

    def product(self, axes: object = None, keepdims: bool = False) -> Scalar:
        """Returns Scalar. Autodiff (VJP) for reductions is unsupported in Phase 1."""
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
        """Returns Scalar. Autodiff (VJP) for reductions is unsupported in Phase 1."""
        return self._post("std", reduce_args(axes, keepdims), rtype=Scalar)

    def sum(self, axes: object = None, keepdims: bool = False) -> Scalar:
        """Returns Scalar. Autodiff (VJP) for reductions is unsupported in Phase 1."""
        return self._post("sum", reduce_args(axes, keepdims), rtype=Scalar)

    def transpose(self, permutation: object = None) -> "Tensor":
        op = TransposeViewOp(kind="transpose", permutation=normalize_permutation(permutation))
        result = self._apply_view_transform(method="transpose", arg=permutation, op=op)
        _builder = get_active_builder()
        if _builder is not None:
            in_vid = _builder.register_value(self)
            out_vid = _builder.register_value(result)
            perm_list = list(permutation) if permutation is not None else []
            _builder.record(TensorNodeRecord(
                node_id=_builder._next_node_id(),
                output_value_id=out_vid,
                operator=TransposeOperator(),
                op_params={"perm": perm_list},
                input_value_ids=[in_vid],
            ))
        return result

    def materialize_view_spec(self) -> "Tensor":
        """Apply the current view spec through the active backend if supported."""
        if self._native is None or not isinstance(self._native, TensorBackend):
            return self

        if self._view_ops_materialized:
            return self

        spec = self.view_spec()
        if not spec.ops:
            return self

        if isinstance(self._native, TensorWireTensorBackend):
            try:
                base_shape = self._native_attr("shape")
                if base_shape is not None:
                    wire = encode_view_schema(spec.to_view_schema(base_shape=base_shape))
                    native = self._native.apply_view_wire(wire)
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
        _builder = get_active_builder()
        if _builder is not None:
            lhs_vid = _builder.register_value(self)
            rhs_vid = _builder.register_value(other) if isinstance(other, Tensor) else f"const_{id(other)}"
            out_vid = _builder.register_value(result)
            _builder.record(TensorNodeRecord(
                node_id=_builder._next_node_id(),
                output_value_id=out_vid,
                operator=MatmulOperator(),
                op_params={},
                input_value_ids=[lhs_vid, rhs_vid],
            ))
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
        result = self._binary_tensor("add", other)
        _builder = get_active_builder()
        if _builder is not None:
            lhs_vid = _builder.register_value(self)
            rhs_vid = _builder.register_value(other) if isinstance(other, Tensor) else f"const_{id(other)}"
            out_vid = _builder.register_value(result)
            _builder.record(TensorNodeRecord(
                node_id=_builder._next_node_id(),
                output_value_id=out_vid,
                operator=AddOperator(),
                op_params={},
                input_value_ids=[lhs_vid, rhs_vid],
            ))
        return result

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

        dtype = self.dtype
        if isinstance(dtype, str):
            from ...state.value import Number

            normalized = dtype.strip().lower()
            dtype = {
                "f32": path(Number, "float", "32"),
                "float32": path(Number, "float", "32"),
                "f64": path(Number, "float", "64"),
                "float64": path(Number, "float", "64"),
                "u64": path(Number, "uint", "64"),
                "uint64": path(Number, "uint", "64"),
            }.get(normalized, dtype)

        return {
            str(TENSOR_CLASS_URI): [
                [dtype, self.shape],
                self.values,
            ]
        }


__all__ = ["Tensor"]
