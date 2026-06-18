from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TensorStorageLayout:
    kind: Literal["dense", "sparse"]
    sparse_axis: int | None = None


@dataclass(frozen=True, slots=True)
class TensorStorageSchema:
    dtype: str
    shape: tuple[int, ...]
    layout: TensorStorageLayout


@dataclass(frozen=True, slots=True)
class TensorViewAxisMap:
    kind: Literal["identity", "affine", "gather"]
    start: int = 0
    step: int = 1
    gather: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TensorViewAxis:
    base_axis: int
    map: TensorViewAxisMap


@dataclass(frozen=True, slots=True)
class TensorViewSchema:
    base_rank: int
    axes: tuple[TensorViewAxis, ...]
    base_fixed: tuple[int | None, ...]

    def to_json(self) -> dict[str, object]:
        def map_to_json(axis_map: TensorViewAxisMap) -> dict[str, object]:
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


__all__ = [
    "TensorStorageLayout",
    "TensorStorageSchema",
    "TensorViewAxisMap",
    "TensorViewAxis",
    "TensorViewSchema",
]
