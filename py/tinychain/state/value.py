from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from ..uri import URI, uri

Json: TypeAlias = object

VALUE_NONE: str = uri("state", "scalar", "value", "none").path
VALUE_NUMBER: str = uri("state", "scalar", "value", "number").path
VALUE_STRING: str = uri("state", "scalar", "value", "string").path
VALUE_LINK: str = uri("state", "scalar", "value", "link").path


@dataclass(frozen=True, slots=True)
class Value:
    kind: Literal["none", "number", "string", "link"]
    value: None | bool | int | float | str = None

    @staticmethod
    def none() -> "Value":
        return Value(kind="none", value=None)

    @staticmethod
    def number(value: bool | int | float) -> "Value":
        return Value(kind="number", value=value)

    @staticmethod
    def string(value: str) -> "Value":
        return Value(kind="string", value=value)

    @staticmethod
    def link(value: URI | str) -> "Value":
        return Value(kind="link", value=str(value))

    def to_json(self) -> Json:
        if self.kind == "none":
            return None
        if self.kind == "number":
            return self.value
        if self.kind == "string":
            return self.value
        if self.kind == "link":
            return {VALUE_LINK: str(self.value)}
        raise AssertionError(f"unexpected Value.kind {self.kind}")

    @staticmethod
    def from_json(obj: Any) -> "Value":
        if obj is None:
            return Value.none()
        if isinstance(obj, (bool, int, float)):
            return Value.number(obj)
        if isinstance(obj, str):
            return Value.string(obj)

        if isinstance(obj, dict) and len(obj) == 1:
            (key, value), = obj.items()
            if key == VALUE_NONE:
                return Value.none()
            if key == VALUE_NUMBER:
                if not isinstance(value, (bool, int, float)):
                    raise TypeError("expected number value")
                return Value.number(value)
            if key == VALUE_STRING:
                if not isinstance(value, str):
                    raise TypeError("expected string value")
                return Value.string(value)
            if key == VALUE_LINK:
                if not isinstance(value, str):
                    raise TypeError("expected link value")
                return Value.link(value)
            if isinstance(key, str) and (key.startswith("/") or "://" in key):
                if value != []:
                    raise TypeError("expected link value to be an empty list")
                return Value.link(key)

        raise TypeError(f"cannot decode Value from {type(obj).__name__}")
