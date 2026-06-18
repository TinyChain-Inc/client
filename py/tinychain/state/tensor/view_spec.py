from __future__ import annotations

from collections.abc import Iterable as IterableABC
from dataclasses import dataclass

from ._common import normalize_shape
from .schema import FensorViewAxis, FensorViewAxisMap, FensorViewSchema
from .view_ops import BroadcastViewOp, ReshapeViewOp, SliceViewOp, TensorViewOp, TransposeViewOp


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

        normalized = normalize_shape(base_shape)
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


__all__ = ["TensorViewSpec"]
