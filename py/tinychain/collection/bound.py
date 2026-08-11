from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class In:
    value: object

    def to_json(self) -> list[object]:
        return ["in", self.value]


@dataclass(frozen=True)
class Ex:
    value: object

    def to_json(self) -> list[object]:
        return ["ex", self.value]


@dataclass(frozen=True)
class Range:
    start: In | Ex | None = None
    stop: In | Ex | None = None

    @classmethod
    def from_slice(cls, bound: slice) -> "Range":
        if bound.step not in (None, 1):
            raise ValueError("collection slice bounds require a unit step")
        start = None if bound.start is None else In(bound.start)
        stop = None if bound.stop is None else Ex(bound.stop)
        return cls(start, stop)

    def to_json(self) -> list[object]:
        return [
            None if self.start is None else self.start.to_json(),
            None if self.stop is None else self.stop.to_json(),
        ]
