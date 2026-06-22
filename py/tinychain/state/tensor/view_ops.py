from __future__ import annotations

from dataclasses import dataclass


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


__all__ = [
    "TensorViewOp",
    "TransposeViewOp",
    "BroadcastViewOp",
    "SliceViewOp",
    "ReshapeViewOp",
]
