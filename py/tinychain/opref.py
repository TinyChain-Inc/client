from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Iterable, Optional, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class OpRef(Generic[T]):
    method: str
    path: str
    headers: tuple[tuple[str, str], ...] = ()
    body: Any = None

    def with_headers(self, headers: Optional[Iterable[tuple[str, str]]]) -> "OpRef[T]":
        if not headers:
            return self
        return OpRef(
            method=self.method,
            path=self.path,
            headers=self.headers + tuple(headers),
            body=self.body,
        )
