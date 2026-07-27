from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Sequence

from ...uri import path, uri
from ..base import State

if TYPE_CHECKING:
    from . import Scalar


class OpDef:
    __slots__ = ()

    METHOD: str = ""

    @property
    def method(self) -> str:
        return type(self).METHOD

    @property
    def form(self) -> list[tuple[str, "Scalar"]]:
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.to_json() == other.to_json()

    def __hash__(self) -> int:
        return hash((type(self), repr(self.to_json())))

    def last_id(self) -> str | None:
        if not self.form:
            return None
        return self.form[-1][0]

    def walk_scalars(self) -> Iterator["Scalar"]:
        from . import _iter_scalar_nodes

        for _, scalar in self.form:
            yield from _iter_scalar_nodes(scalar)

    def _reflect(self, subject: str) -> "Scalar":
        from . import PostOpRef, Scalar

        return Scalar(ref=PostOpRef(subject, {"op": self}))

    def reflect_form(self) -> "Scalar":
        return self._reflect(path(uri(State, "scalar", "op", "reflect", "form")))

    def reflect_last_id(self) -> "Scalar":
        return self._reflect(path(uri(State, "scalar", "op", "reflect", "last_id")))

    def reflect_scalars(self) -> "Scalar":
        return self._reflect(path(uri(State, "scalar", "op", "reflect", "scalars")))

    def class_(self) -> "Scalar":
        from . import Scalar

        return Scalar(self).class_()

    def to_json(self) -> dict[str, object]:
        raise NotImplementedError()

    @staticmethod
    def from_json(obj: Any) -> "OpDef":
        from . import _decode_form

        if not isinstance(obj, dict) or len(obj) != 1:
            raise TypeError("expected an OpDef map")

        (key, value), = obj.items()
        if not isinstance(key, str):
            raise TypeError("expected OpDef map key to be a string")

        if key == path(uri(State, "scalar", "op", "get")):
            if not isinstance(value, list) or len(value) != 2:
                raise TypeError("invalid GET opdef encoding")
            key_name, form = value
            if not isinstance(key_name, str):
                raise TypeError("expected GET key name to be a string")
            return GetOpDef(key_name, _decode_form(form))

        if key == path(uri(State, "scalar", "op", "put")):
            if not isinstance(value, list) or len(value) != 3:
                raise TypeError("invalid PUT opdef encoding")
            key_name, value_name, form = value
            if not isinstance(key_name, str) or not isinstance(value_name, str):
                raise TypeError("expected PUT key/value names to be strings")
            return PutOpDef(key_name, value_name, _decode_form(form))

        if key == path(uri(State, "scalar", "op", "post")):
            return PostOpDef(_decode_form(value))

        if key == path(uri(State, "scalar", "op", "delete")):
            if not isinstance(value, list) or len(value) != 2:
                raise TypeError("invalid DELETE opdef encoding")
            key_name, form = value
            if not isinstance(key_name, str):
                raise TypeError("expected DELETE key name to be a string")
            return DeleteOpDef(key_name, _decode_form(form))

        raise TypeError("unexpected OpDef map key")


class GetOpDef(OpDef):
    __slots__ = ("key", "_form")

    METHOD = "GET"

    def __init__(self, key: str, form: Sequence[tuple[str, object]]):
        from . import _normalize_opdef_form

        if not isinstance(key, str):
            raise TypeError("GET OpDef requires a string key")

        self.key = key
        self._form = _normalize_opdef_form(form)

    @property
    def form(self) -> list[tuple[str, "Scalar"]]:
        return self._form

    def to_json(self) -> dict[str, object]:
        from . import _encode_form

        return {path(uri(State, "scalar", "op", "get")): [self.key, _encode_form(self.form)]}


class PutOpDef(OpDef):
    __slots__ = ("key", "value", "_form")

    METHOD = "PUT"

    def __init__(self, key: str, value: str, form: Sequence[tuple[str, object]]):
        from . import _normalize_opdef_form

        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("PUT OpDef requires string key and value bindings")

        self.key = key
        self.value = value
        self._form = _normalize_opdef_form(form)

    @property
    def form(self) -> list[tuple[str, "Scalar"]]:
        return self._form

    def to_json(self) -> dict[str, object]:
        from . import _encode_form

        return {path(uri(State, "scalar", "op", "put")): [self.key, self.value, _encode_form(self.form)]}


class PostOpDef(OpDef):
    __slots__ = ("_form",)

    METHOD = "POST"

    def __init__(self, form: Sequence[tuple[str, object]]):
        from . import _normalize_opdef_form

        self._form = _normalize_opdef_form(form)

    @property
    def form(self) -> list[tuple[str, "Scalar"]]:
        return self._form

    def to_json(self) -> dict[str, object]:
        from . import _encode_form

        return {path(uri(State, "scalar", "op", "post")): _encode_form(self.form)}


class DeleteOpDef(OpDef):
    __slots__ = ("key", "_form")

    METHOD = "DELETE"

    def __init__(self, key: str, form: Sequence[tuple[str, object]]):
        from . import _normalize_opdef_form

        if not isinstance(key, str):
            raise TypeError("DELETE OpDef requires a string key")

        self.key = key
        self._form = _normalize_opdef_form(form)

    @property
    def form(self) -> list[tuple[str, "Scalar"]]:
        return self._form

    def to_json(self) -> dict[str, object]:
        from . import _encode_form

        return {path(uri(State, "scalar", "op", "delete")): [self.key, _encode_form(self.form)]}
