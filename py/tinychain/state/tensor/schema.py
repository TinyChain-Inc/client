from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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


__all__ = [
    "FensorLayoutSchema",
    "FensorTensorSchema",
    "FensorViewAxisMap",
    "FensorViewAxis",
    "FensorViewSchema",
]
