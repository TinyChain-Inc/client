from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from .view_ops import TensorViewOp


@runtime_checkable
class TensorBackend(Protocol):
    """Minimal backend adapter contract for tensor view transforms."""

    def apply_view_op(self, op: TensorViewOp) -> object: ...


@runtime_checkable
class TensorWireTensorBackend(Protocol):
    """Optional backend hook for canonical tensor wire view transforms."""

    def apply_view_wire(
        self,
        wire: tuple[int, list[tuple[int, tuple[int, list[int]]]], list[int | None]],
    ) -> object: ...


@dataclass(frozen=True)
class DenseTensor:
    """Canonical in-memory tensor payload used by the Python client.

    This is deliberately a client value, not a PyO3 wrapper. The local backend
    receives the same JSON representation as an HTTP backend.
    """

    dtype: str
    shape: tuple[int, ...]
    values: tuple[int | float, ...]

    @classmethod
    def new(
        cls,
        dtype: str,
        shape: Sequence[int],
        values: Sequence[int | float],
    ) -> "DenseTensor":
        shape = tuple(int(dim) for dim in shape)
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError("dense Tensor shape must contain positive dimensions")

        size = 1
        for dim in shape:
            size *= dim
        values = tuple(values)
        if len(values) != size:
            raise ValueError(f"dense Tensor shape requires {size} values, found {len(values)}")

        return cls(dtype, shape, values)


__all__ = ["DenseTensor", "TensorBackend", "TensorWireTensorBackend"]
