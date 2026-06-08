from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..uri import URI, uri

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
    def string(value: str) -> "String":
        return String(value)

    @staticmethod
    def link(value: URI | str) -> "Value":
        return Value(kind="link", value=str(value))

    def to_json(self) -> object:
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


class String(Value):
    __slots__ = ("op",)

    def __init__(self, value: str | object):
        from ..opref import OpRef as RuntimeOpRef
        from .scalar import OpRef as StateOpRef

        if isinstance(value, (RuntimeOpRef, StateOpRef)):
            super().__init__(kind="string", value=None)
            object.__setattr__(self, "op", value)
        else:
            super().__init__(kind="string", value=value)
            object.__setattr__(self, "op", None)

    def render(self, params: dict[str, object] | None = None, **kwargs: object) -> "String":
        if params is not None and kwargs:
            raise ValueError("String.render accepts a dict or kwargs, not both")

        from .scalar import autobox

        render_params = kwargs if params is None else params
        if self.op is None and all(_is_literal_render_value(value) for value in render_params.values()):
            rendered = str(self.value)
            for key, value in render_params.items():
                rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
            return String(rendered)

        return String(autobox(self)._string_render(render_params).ref.op)


def _is_literal_render_value(value: object) -> bool:
    return isinstance(value, (str, bool, int, float, URI))
