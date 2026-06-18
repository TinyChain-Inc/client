from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable as IterableABC
from numbers import Number as NumberABC
from typing import Literal, Protocol, runtime_checkable
from ..uri import URI, uri
from .scalar import (
    Bool,
    Comparable,
    IdRef,
    Number,
    Scalar,
    TCRef,
    Tuple,
    autobox,
    form_of,
    tcref_form_of,
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


@dataclass(frozen=True, slots=True)
class TensorViewOp:
    kind: str


@dataclass(frozen=True, slots=True)
class TransposeViewOp(TensorViewOp):
    permutation: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class BroadcastViewOp(TensorViewOp):
    shape: tuple[int, ...]
    broadcast_axes: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class SliceViewOp(TensorViewOp):
    bounds: object


@dataclass(frozen=True, slots=True)
class ReshapeViewOp(TensorViewOp):
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FensorLayoutSchema:
    kind: Literal["dense", "sparse"]
    sparse_axis: int | None = None

    def to_wire(self) -> tuple[int, int | None]:
        if self.kind == "dense":
            return (0, None)

        return (1, self.sparse_axis)

    @staticmethod
    def from_wire(payload: object) -> "FensorLayoutSchema":
        if not isinstance(payload, (tuple, list)) or len(payload) != 2:
            raise TypeError("layout wire payload must be a (tag, axis) pair")

        tag, axis = payload
        if tag == 0:
            if axis is not None:
                raise ValueError("dense layout wire payload must not include axis")
            return FensorLayoutSchema(kind="dense")

        if tag == 1:
            if axis is not None:
                axis = int(axis)
            return FensorLayoutSchema(kind="sparse", sparse_axis=axis)

        raise ValueError(f"unknown layout tag {tag}")


@dataclass(frozen=True, slots=True)
class FensorTensorSchema:
    dtype: str
    shape: tuple[int, ...]
    layout: FensorLayoutSchema

    def to_wire(self) -> tuple[str, list[int], tuple[int, int | None]]:
        return (self.dtype, list(self.shape), self.layout.to_wire())

    @staticmethod
    def from_wire(payload: object) -> "FensorTensorSchema":
        if not isinstance(payload, (tuple, list)) or len(payload) != 3:
            raise TypeError("tensor schema wire payload must be (dtype, shape, layout)")

        dtype, shape, layout = payload
        if not isinstance(dtype, str):
            raise TypeError("tensor schema dtype must be string")
        if not isinstance(shape, (tuple, list)):
            raise TypeError("tensor schema shape must be a list")

        normalized_shape = tuple(int(dim) for dim in shape)
        if not normalized_shape:
            raise ValueError("tensor schema shape must not be empty")
        if any(dim <= 0 for dim in normalized_shape):
            raise ValueError("tensor schema shape dimensions must be positive")

        return FensorTensorSchema(
            dtype=dtype,
            shape=normalized_shape,
            layout=FensorLayoutSchema.from_wire(layout),
        )


@dataclass(frozen=True, slots=True)
class FensorViewAxisMap:
    kind: Literal["identity", "affine", "gather"]
    start: int = 0
    step: int = 1
    gather: tuple[int, ...] = ()

    def to_wire(self) -> tuple[int, list[int]]:
        if self.kind == "identity":
            return (0, [])

        if self.kind == "affine":
            return (1, [self.start, self.step])

        return (2, list(self.gather))

    @staticmethod
    def from_wire(payload: object) -> "FensorViewAxisMap":
        if not isinstance(payload, (tuple, list)) or len(payload) != 2:
            raise TypeError("axis map wire payload must be (tag, payload)")

        tag, data = payload
        if not isinstance(data, (tuple, list)):
            raise TypeError("axis map payload data must be a list")

        if tag == 0:
            if len(data) != 0:
                raise ValueError("identity axis map payload must be empty")
            return FensorViewAxisMap(kind="identity")

        if tag == 1:
            if len(data) != 2:
                raise ValueError("affine axis map payload must contain [start, step]")
            return FensorViewAxisMap(kind="affine", start=int(data[0]), step=int(data[1]))

        if tag == 2:
            return FensorViewAxisMap(kind="gather", gather=tuple(int(index) for index in data))

        raise ValueError(f"unknown axis map tag {tag}")


@dataclass(frozen=True, slots=True)
class FensorViewAxis:
    base_axis: int
    map: FensorViewAxisMap

    def to_wire(self) -> tuple[int, tuple[int, list[int]]]:
        return (self.base_axis, self.map.to_wire())

    @staticmethod
    def from_wire(payload: object) -> "FensorViewAxis":
        if not isinstance(payload, (tuple, list)) or len(payload) != 2:
            raise TypeError("view axis wire payload must be (base_axis, axis_map)")

        base_axis, axis_map = payload
        return FensorViewAxis(base_axis=int(base_axis), map=FensorViewAxisMap.from_wire(axis_map))


@dataclass(frozen=True, slots=True)
class FensorViewSchema:
    base_rank: int
    axes: tuple[FensorViewAxis, ...]
    base_fixed: tuple[int | None, ...]

    def to_wire(self) -> tuple[int, list[tuple[int, tuple[int, list[int]]]], list[int | None]]:
        return (
            self.base_rank,
            [axis.to_wire() for axis in self.axes],
            list(self.base_fixed),
        )

    @staticmethod
    def from_wire(payload: object) -> "FensorViewSchema":
        if not isinstance(payload, (tuple, list)) or len(payload) != 3:
            raise TypeError("view schema wire payload must be (base_rank, axes, base_fixed)")

        base_rank, axes, base_fixed = payload
        if not isinstance(axes, (tuple, list)):
            raise TypeError("view schema axes must be a list")
        if not isinstance(base_fixed, (tuple, list)):
            raise TypeError("view schema base_fixed must be a list")

        return FensorViewSchema(
            base_rank=int(base_rank),
            axes=tuple(FensorViewAxis.from_wire(axis) for axis in axes),
            base_fixed=tuple(None if item is None else int(item) for item in base_fixed),
        )

    def to_json(self) -> dict[str, object]:
        def map_to_json(axis_map: FensorViewAxisMap) -> dict[str, object]:
            if axis_map.kind == "identity":
                return {"identity": True}
            if axis_map.kind == "affine":
                return {"affine": {"start": axis_map.start, "step": axis_map.step}}
            return {"gather": list(axis_map.gather)}

        return {
            "base_rank": self.base_rank,
            "axes": [
                {
                    "base_axis": axis.base_axis,
                    "map": map_to_json(axis.map),
                }
                for axis in self.axes
            ],
            "base_fixed": list(self.base_fixed),
        }


@dataclass(frozen=True, slots=True)
class TensorViewSpec:
    """Canonical client-side tensor view plan.

    This remains backend-agnostic: local shims can execute it eagerly while
    deferred backends can serialize and forward it to runtime routes.
    """

    ops: tuple[TensorViewOp, ...]

    def to_fensor_view_schema(self, *, base_shape: object) -> FensorViewSchema:
        """Compile this view plan to a fensor-compatible view schema payload.

        This intentionally supports the minimal, high-value subset needed for
        current Autodiff planning: `transpose` and rank-preserving `broadcast`.
        """

        normalized = _normalize_shape(base_shape)
        if normalized is None:
            raise TypeError("base_shape must be an iterable of integer dimensions")
        if not normalized:
            raise ValueError("base_shape must not be empty")

        base_rank = len(normalized)
        current_shape = normalized
        axes: list[_ViewAxisState] = [
            _ViewAxisState(base_axis=axis, map=_AxisMapState.identity())
            for axis in range(base_rank)
        ]
        base_fixed: list[int | None] = [None] * base_rank

        for op in self.ops:
            if isinstance(op, TransposeViewOp):
                permutation = op.permutation or tuple(reversed(range(len(axes))))
                if len(permutation) != len(axes):
                    raise ValueError("transpose permutation rank must match tensor rank")
                if sorted(permutation) != list(range(len(axes))):
                    raise ValueError("transpose permutation must be a valid axis permutation")

                axes = [axes[index] for index in permutation]
                current_shape = tuple(current_shape[index] for index in permutation)
                continue

            if isinstance(op, BroadcastViewOp):
                target_shape = op.shape
                if len(target_shape) != len(current_shape):
                    raise NotImplementedError(
                        "broadcast schema compilation currently requires rank-preserving broadcast"
                    )

                next_axes: list[_ViewAxisState] = []
                for axis, (source_dim, target_dim) in enumerate(zip(current_shape, target_shape)):
                    state = axes[axis]
                    if source_dim == target_dim:
                        next_axes.append(state)
                        continue

                    if source_dim == 1 and target_dim > 1:
                        next_axes.append(state.broadcasted())
                        continue

                    raise ValueError(
                        f"invalid broadcast from shape {current_shape} to {target_shape} at axis {axis}"
                    )

                axes = next_axes
                current_shape = target_shape
                continue

            if isinstance(op, SliceViewOp):
                axes, current_shape, base_fixed = _compile_slice(
                    op.bounds,
                    axes,
                    current_shape,
                    base_fixed,
                )
                continue

            if isinstance(op, ReshapeViewOp):
                raise NotImplementedError("reshape schema compilation is not implemented yet")

            raise TypeError(f"unsupported view op type {type(op).__name__}")

        return FensorViewSchema(
            base_rank=base_rank,
            axes=tuple(axis.to_schema() for axis in axes),
            base_fixed=tuple(base_fixed),
        )

    def to_fensor_view_wire(self, *, base_shape: object) -> tuple[int, list[tuple[int, tuple[int, list[int]]]], list[int | None]]:
        return self.to_fensor_view_schema(base_shape=base_shape).to_wire()


@dataclass(frozen=True, slots=True)
class _AxisMapState:
    kind: str
    start: int = 0
    step: int = 1
    gather: tuple[int, ...] = ()

    @staticmethod
    def identity() -> "_AxisMapState":
        return _AxisMapState(kind="identity")

    def broadcasted(self) -> "_AxisMapState":
        if self.kind == "identity":
            return _AxisMapState(kind="affine", start=0, step=0)

        if self.kind == "affine":
            return _AxisMapState(kind="affine", start=self.start, step=0)

        if self.kind == "gather":
            if not self.gather:
                raise ValueError("cannot broadcast empty gather axis")
            return _AxisMapState(kind="affine", start=self.gather[0], step=0)

        raise TypeError(f"unsupported axis map kind {self.kind}")

    def resolve(self, index: int) -> int:
        if self.kind == "identity":
            return index

        if self.kind == "affine":
            return self.start + index * self.step

        if self.kind == "gather":
            if index < 0 or index >= len(self.gather):
                raise ValueError("slice coordinate out of bounds for gathered axis")
            return self.gather[index]

        raise TypeError(f"unsupported axis map kind {self.kind}")

    def compose_range(self, *, start: int, step: int, extent: int) -> "_AxisMapState":
        if self.kind == "identity":
            return _AxisMapState(kind="affine", start=start, step=step)

        if self.kind == "affine":
            return _AxisMapState(
                kind="affine",
                start=self.start + start * self.step,
                step=self.step * step,
            )

        if self.kind == "gather":
            indices = [self.resolve(start + i * step) for i in range(extent)]
            return _AxisMapState(kind="gather", gather=tuple(indices))

        raise TypeError(f"unsupported axis map kind {self.kind}")

    def to_json(self) -> dict[str, object]:
        if self.kind == "identity":
            return {"identity": True}

        if self.kind == "affine":
            return {"affine": {"start": self.start, "step": self.step}}

        if self.kind == "gather":
            return {"gather": list(self.gather)}

        raise TypeError(f"unsupported axis map kind {self.kind}")

    def to_schema(self) -> FensorViewAxisMap:
        if self.kind == "identity":
            return FensorViewAxisMap(kind="identity")

        if self.kind == "affine":
            return FensorViewAxisMap(kind="affine", start=self.start, step=self.step)

        if self.kind == "gather":
            return FensorViewAxisMap(kind="gather", gather=self.gather)

        raise TypeError(f"unsupported axis map kind {self.kind}")


@dataclass(frozen=True, slots=True)
class _ViewAxisState:
    base_axis: int
    map: _AxisMapState

    def broadcasted(self) -> "_ViewAxisState":
        return _ViewAxisState(base_axis=self.base_axis, map=self.map.broadcasted())

    def to_json(self) -> dict[str, object]:
        return {"base_axis": self.base_axis, "map": self.map.to_json()}

    def to_schema(self) -> FensorViewAxis:
        return FensorViewAxis(base_axis=self.base_axis, map=self.map.to_schema())


def _normalize_slice_axis_bound(bound: object, axis_dim: int) -> tuple[str, int, int, int]:
    if isinstance(bound, int):
        if bound < 0 or bound >= axis_dim:
            raise ValueError("slice index out of bounds")
        return ("at", bound, bound + 1, 1)

    if isinstance(bound, slice):
        start = 0 if bound.start is None else int(bound.start)
        stop = axis_dim if bound.stop is None else int(bound.stop)
        step = 1 if bound.step is None else int(bound.step)
    elif isinstance(bound, (tuple, list)):
        if len(bound) == 0 or len(bound) > 3:
            raise ValueError("slice range must have 1 to 3 components")
        start = int(bound[0])
        stop = axis_dim if len(bound) < 2 else int(bound[1])
        step = 1 if len(bound) < 3 else int(bound[2])
    else:
        raise TypeError("slice bound must be int, slice, or tuple/list range")

    if step <= 0:
        raise ValueError("slice step must be positive")
    if start < 0 or stop < start or stop > axis_dim:
        raise ValueError("slice range out of bounds")

    return ("in", start, stop, step)


def _compile_slice(
    bounds: object,
    axes: list[_ViewAxisState],
    shape: tuple[int, ...],
    base_fixed: list[int | None],
) -> tuple[list[_ViewAxisState], tuple[int, ...], list[int | None]]:
    if not isinstance(bounds, IterableABC) or isinstance(bounds, (str, bytes, bytearray)):
        raise TypeError("slice bounds must be an iterable")

    bounds_list = list(bounds)
    if len(bounds_list) != len(shape):
        raise ValueError("slice bounds rank must match tensor rank")

    next_axes: list[_ViewAxisState] = []
    next_shape: list[int] = []
    next_fixed = list(base_fixed)

    for axis, bound in enumerate(bounds_list):
        axis_state = axes[axis]
        mode, start, stop, step = _normalize_slice_axis_bound(bound, shape[axis])

        if mode == "at":
            next_fixed[axis_state.base_axis] = axis_state.map.resolve(start)
            continue

        extent = 0 if stop == start else (stop - start + step - 1) // step
        next_axes.append(
            _ViewAxisState(
                base_axis=axis_state.base_axis,
                map=axis_state.map.compose_range(start=start, step=step, extent=extent),
            )
        )
        next_shape.append(extent)

    return next_axes, tuple(next_shape), next_fixed


@runtime_checkable
class TensorBackend(Protocol):
    """Minimal backend adapter contract for tensor view transforms."""

    def apply_view_op(self, op: TensorViewOp) -> object: ...

    def apply_view_spec(self, spec: TensorViewSpec) -> object: ...


@runtime_checkable
class FensorWireTensorBackend(Protocol):
    """Optional backend hook for canonical fensor wire view transforms."""

    def apply_fensor_view_wire(
        self,
        wire: tuple[int, list[tuple[int, tuple[int, list[int]]]], list[int | None]],
    ) -> object: ...


def _normalize_shape(shape: object) -> tuple[int, ...] | None:
    if isinstance(shape, IterableABC) and not isinstance(shape, (str, bytes, bytearray)):
        dims: list[int] = []
        for dim in shape:
            try:
                dims.append(int(dim))
            except (TypeError, ValueError):
                return None
        return tuple(dims)

    return None


def _normalize_permutation(permutation: object) -> tuple[int, ...] | None:
    if permutation is None:
        return None

    if isinstance(permutation, IterableABC) and not isinstance(permutation, (str, bytes, bytearray)):
        axes: list[int] = []
        for axis in permutation:
            try:
                axes.append(int(axis))
            except (TypeError, ValueError):
                return None
        return tuple(axes)

    return None


def _infer_broadcast_axes(source_shape: object, target_shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if not isinstance(source_shape, IterableABC) or isinstance(source_shape, (str, bytes, bytearray)):
        return None

    try:
        source = [int(dim) for dim in source_shape]
    except (TypeError, ValueError):
        return None

    if len(source) > len(target_shape):
        return None

    pad = len(target_shape) - len(source)
    padded = [1] * pad + source
    broadcast_axes: list[int] = []

    for axis, (src, dst) in enumerate(zip(padded, target_shape)):
        if src == 1 and dst > 1:
            broadcast_axes.append(axis)
        elif src != dst:
            return None

    return tuple(broadcast_axes)


class Tensor(Comparable):
    """TinyChain tensor.

    A tensor may be symbolic (backed by an IR ref/op) or materialized (backed by
    eager backend data). User code should not need to distinguish those cases.
    """

    __slots__ = ("_native", "_view_ops", "_subject_root", "_view_ops_materialized")

    __uri__: URI = uri("state", "collection", "tensor")

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

        normalized = _normalize_shape(base_shape)
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
                "transpose": TransposeViewOp(kind="transpose", permutation=_normalize_permutation(args[0] if args else None)),
                "broadcast": BroadcastViewOp(kind="broadcast", shape=_normalize_shape(args[0] if args else ()) or (), broadcast_axes=None),
                "reshape": ReshapeViewOp(kind="reshape", shape=_normalize_shape(args[0] if args else ()) or ()),
                "slice": SliceViewOp(kind="slice", bounds=args[0] if args else None),
            }.get(method)

            if op is not None and hasattr(self._native, "apply_view_op"):
                try:
                    return self._native.apply_view_op(op)
                except (TypeError, ValueError, NotImplementedError):
                    pass

        native_method = getattr(self._native, method, None)
        if native_method is None or not callable(native_method):
            return None

        try:
            return native_method(*args)
        except (TypeError, ValueError):
            return None

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
        normalized_shape = _normalize_shape(shape)
        source_shape = self._native_attr("shape")
        broadcast_axes = (
            _infer_broadcast_axes(source_shape, normalized_shape)
            if normalized_shape is not None
            else None
        )
        op = BroadcastViewOp(
            kind="broadcast",
            shape=normalized_shape or (),
            broadcast_axes=broadcast_axes,
        )

        native = self._native_transform("broadcast", shape)
        if native is not None:
            return Tensor(native=native, view_ops=self._view_ops + (op,), view_ops_materialized=True)

        return self._append_view_op(op, self._transform_get("broadcast", shape))

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
        op = ReshapeViewOp(kind="reshape", shape=_normalize_shape(shape) or ())

        native = self._native_transform("reshape", shape)
        if native is not None:
            return Tensor(native=native, view_ops=self._view_ops + (op,), view_ops_materialized=True)

        return self._append_view_op(op, self._transform_get("reshape", shape))

    def slice(self, bounds: object) -> "Tensor":
        op = SliceViewOp(kind="slice", bounds=bounds)

        native = self._native_transform("slice", bounds)
        if native is not None:
            return Tensor(native=native, view_ops=self._view_ops + (op,), view_ops_materialized=True)

        if self._subject_root is None:
            result = self._get(key=autobox(bounds), rtype=Tensor)
        else:
            result = Tensor._get_ref(self._subject_root, autobox(bounds))

        return self._append_view_op(op, result)

    def std(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("std", _reduce_args(axes, keepdims), rtype=Scalar)

    def sum(self, axes: object = None, keepdims: bool = False) -> Scalar:
        return self._post("sum", _reduce_args(axes, keepdims), rtype=Scalar)

    def transpose(self, permutation: object = None) -> "Tensor":
        op = TransposeViewOp(kind="transpose", permutation=_normalize_permutation(permutation))

        native = self._native_transform("transpose", permutation)
        if native is not None:
            return Tensor(native=native, view_ops=self._view_ops + (op,), view_ops_materialized=True)

        return self._append_view_op(op, self._transform_get("transpose", permutation))

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
            except (TypeError, ValueError, NotImplementedError):
                pass

        try:
            native = self._native.apply_view_spec(spec)
        except (TypeError, ValueError, NotImplementedError):
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


__all__ = [
    "Tensor",
    "FensorLayoutSchema",
    "FensorTensorSchema",
    "FensorViewAxisMap",
    "FensorViewAxis",
    "FensorViewSchema",
    "TensorViewSpec",
    "TensorBackend",
    "FensorWireTensorBackend",
    "TensorViewOp",
    "TransposeViewOp",
    "BroadcastViewOp",
    "SliceViewOp",
    "ReshapeViewOp",
    "concatenate",
    "einsum",
    "split",
    "tile",
]
