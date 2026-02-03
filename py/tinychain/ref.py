from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .opref import OpRef


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Ref(Generic[T]):
    op: OpRef[T]


class String(Ref[str]):
    pass


class Json(Ref[Any]):
    pass

