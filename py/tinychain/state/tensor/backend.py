from __future__ import annotations

from typing import Protocol, runtime_checkable

from .view_ops import TensorViewOp
from .view_spec import TensorViewSpec


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


__all__ = ["TensorBackend", "FensorWireTensorBackend"]
