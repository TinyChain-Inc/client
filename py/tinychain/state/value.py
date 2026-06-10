from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

from ..uri import URI, uri

VALUE_NONE: str = uri("state", "scalar", "value", "none").path
VALUE_BOOL: str = uri("state", "scalar", "value", "bool").path
VALUE_NUMBER: str = uri("state", "scalar", "value", "number").path
VALUE_STRING: str = uri("state", "scalar", "value", "string").path
VALUE_LINK: str = uri("state", "scalar", "value", "link").path
VALUE_MAP: str = uri("state", "scalar", "value", "map").path
VALUE_TUPLE: str = uri("state", "scalar", "value", "tuple").path


@dataclass(frozen=True, slots=True)
class Value:
    kind: Literal["none", "bool", "number", "string", "link", "map", "tuple"]
    value: None | bool | int | float | str | dict[str, "Value"] | list["Value"] = None

    @staticmethod
    def none() -> "Value":
        return Value(kind="none", value=None)

    @staticmethod
    def bool(value: bool | object) -> "Bool":
        return Bool(value)

    @staticmethod
    def number(value: int | float | object) -> "Number":
        return Number(value)

    @staticmethod
    def string(value: str) -> "String":
        return String(value)

    @staticmethod
    def link(value: URI | str) -> "Value":
        return Value(kind="link", value=str(value))

    @staticmethod
    def map_of(value: Mapping[str, object] | object) -> "Map":
        return Map(value)

    @staticmethod
    def tuple_of(value: Sequence[object] | object) -> "Tuple":
        return Tuple(value)

    def to_json(self) -> object:
        if self.kind == "none":
            return None
        if self.kind == "bool":
            return self.value
        if self.kind == "number":
            return self.value
        if self.kind == "string":
            return self.value
        if self.kind == "link":
            return {VALUE_LINK: str(self.value)}
        if self.kind == "map":
            assert isinstance(self.value, dict)
            return {key: _encode_value_item(item) for key, item in self.value.items()}
        if self.kind == "tuple":
            assert isinstance(self.value, list)
            return [_encode_value_item(item) for item in self.value]
        raise AssertionError(f"unexpected Value.kind {self.kind}")

    @staticmethod
    def from_json(obj: Any) -> "Value":
        if obj is None:
            return Value.none()
        if isinstance(obj, bool):
            return Value.bool(obj)
        if isinstance(obj, (int, float)):
            return Value.number(obj)
        if isinstance(obj, str):
            return Value.string(obj)
        if isinstance(obj, list):
            return Value.tuple_of(obj)

        if isinstance(obj, dict) and len(obj) == 1:
            (key, value), = obj.items()
            if key == VALUE_NONE:
                return Value.none()
            if key == VALUE_BOOL:
                if not isinstance(value, bool):
                    raise TypeError("expected bool value")
                return Value.bool(value)
            if key == VALUE_NUMBER:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
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
            if key == VALUE_MAP:
                if not isinstance(value, dict):
                    raise TypeError("expected map value")
                return Value.map_of(value)
            if key == VALUE_TUPLE:
                if not isinstance(value, list):
                    raise TypeError("expected tuple value")
                return Value.tuple_of(value)
            if isinstance(key, str) and (key.startswith("/") or "://" in key):
                if value != []:
                    raise TypeError("expected link value to be an empty list")
                return Value.link(key)

        if isinstance(obj, dict):
            return Value.map_of(obj)

        raise TypeError(f"cannot decode Value from {type(obj).__name__}")


def _decode_value_item(obj: object) -> Value:
    if isinstance(obj, Value):
        return obj

    return Value.from_json(obj)


def _encode_value_item(value: Value) -> object:
    return value.to_json()


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


class Bool(Value):
    __slots__ = ("op",)

    def __init__(self, value: bool | object):
        op = _as_opref(value)
        if op is not None:
            super().__init__(kind="bool", value=None)
            object.__setattr__(self, "op", op)
            return

        if not isinstance(value, bool):
            raise TypeError("expected bool value")

        super().__init__(kind="bool", value=value)
        object.__setattr__(self, "op", None)


class Number(Value):
    __slots__ = ("op",)

    def __init__(self, value: int | float | object):
        op = _as_opref(value)
        if op is not None:
            super().__init__(kind="number", value=None)
            object.__setattr__(self, "op", op)
            return

        if isinstance(value, bool):
            raise TypeError("bool is not a number; use Value.bool")
        if not isinstance(value, (int, float)):
            raise TypeError("expected number value")

        super().__init__(kind="number", value=value)
        object.__setattr__(self, "op", None)


class Map(Value):
    __slots__ = ("op",)

    def __init__(self, value: Mapping[str, object] | object):
        op = _as_opref(value)
        if op is not None:
            super().__init__(kind="map", value=None)
            object.__setattr__(self, "op", op)
            return

        if not isinstance(value, Mapping):
            raise TypeError("expected map value")

        decoded: dict[str, Value] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("expected map key to be a string")
            decoded[key] = _decode_value_item(item)

        super().__init__(kind="map", value=decoded)
        object.__setattr__(self, "op", None)

    def __iter__(self) -> Iterator[str]:
        if self.value is None:
            raise TypeError("cannot iterate a deferred Map without execution")
        assert isinstance(self.value, dict)
        return iter(self.value)

    def keys(self) -> Iterable[str]:
        if self.value is None:
            raise TypeError("cannot read keys from a deferred Map without execution")
        assert isinstance(self.value, dict)
        return self.value.keys()

    def values(self) -> Iterable[Value]:
        if self.value is None:
            raise TypeError("cannot read values from a deferred Map without execution")
        assert isinstance(self.value, dict)
        return self.value.values()

    def items(self) -> Iterable[tuple[str, Value]]:
        if self.value is None:
            raise TypeError("cannot read items from a deferred Map without execution")
        assert isinstance(self.value, dict)
        return self.value.items()


class Tuple(Value):
    __slots__ = ("op",)

    def __init__(self, value: Sequence[object] | object):
        op = _as_opref(value)
        if op is not None:
            super().__init__(kind="tuple", value=None)
            object.__setattr__(self, "op", op)
            return

        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError("expected tuple value")

        super().__init__(kind="tuple", value=[_decode_value_item(item) for item in value])
        object.__setattr__(self, "op", None)

    def __iter__(self) -> Iterator[Value]:
        if self.value is None:
            raise TypeError("cannot iterate a deferred Tuple without execution")
        assert isinstance(self.value, list)
        return iter(self.value)

    def __len__(self) -> int:
        if self.value is None:
            raise TypeError("cannot get len of a deferred Tuple without execution")
        assert isinstance(self.value, list)
        return len(self.value)

    def __getitem__(self, index: int) -> Value:
        if self.value is None:
            raise TypeError("cannot index a deferred Tuple without execution")
        assert isinstance(self.value, list)
        return self.value[index]


def _as_opref(value: object) -> object | None:
    from ..opref import OpRef as RuntimeOpRef
    from .scalar import OpRef as StateOpRef

    if isinstance(value, (RuntimeOpRef, StateOpRef)):
        return value

    return None
