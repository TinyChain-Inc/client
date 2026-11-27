from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Sequence

from ...uri import URI
from ..base import State

if TYPE_CHECKING:
    from . import Scalar


class OpDef:
    __slots__ = ()

    METHOD: str = ""
    __uri__: URI = URI(State, "scalar", "op")

    @property
    def method(self) -> str:
        return type(self).METHOD

    @property
    def form(self) -> tuple[tuple[str, "Scalar"], ...]:
        raise NotImplementedError()

    def last_id(self) -> str | None:
        if not self.form:
            return None
        return self.form[-1][0]

    def _parameters(self) -> tuple[str, ...]:
        if isinstance(self, (GetOpDef, DeleteOpDef)):
            return (self.key,)
        if isinstance(self, PutOpDef):
            return (self.key, self.value)
        return ()

    def requires(self, required: set[str]) -> None:
        from ..._lexical import requires

        requires(self, required)

    def validate(self) -> None:
        from ..._lexical import validate

        validate(self)

    def walk_scalars(self) -> Iterator["Scalar"]:
        from . import _iter_scalar_nodes

        for _, scalar in self.form:
            yield from _iter_scalar_nodes(scalar)

    def _reflect(self, subject: str) -> "Scalar":
        from . import PostOpRef, Scalar

        return Scalar._from_opref(PostOpRef(subject, {"op": self}))

    def reflect_form(self) -> "Scalar":
        return self._reflect(str(URI(type(self), "reflect", "form")))

    def reflect_last_id(self) -> "Scalar":
        return self._reflect(str(URI(type(self), "reflect", "last_id")))

    def reflect_scalars(self) -> "Scalar":
        return self._reflect(str(URI(type(self), "reflect", "scalars")))

    def class_(self) -> "Scalar":
        from . import Scalar

        return Scalar(self).class_()

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)

    @staticmethod
    def from_json(obj: Any) -> "OpDef":
        from ... import _native

        return _native.Codec().opdef_from_wire(obj)


class GetOpDef(OpDef):
    __slots__ = ("key", "_form")

    METHOD = "GET"

    def __init__(self, key: str, form: Sequence[tuple[str, object]]):
        from . import _normalize_opdef_form

        if not isinstance(key, str):
            raise TypeError("GET OpDef requires a string key")

        self.key = key
        self._form = _normalize_opdef_form(form)
        self.validate()

    @property
    def form(self) -> tuple[tuple[str, "Scalar"], ...]:
        return self._form

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)


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
        self.validate()

    @property
    def form(self) -> tuple[tuple[str, "Scalar"], ...]:
        return self._form

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)


class PostOpDef(OpDef):
    __slots__ = ("_form",)

    METHOD = "POST"

    def __init__(self, form: Sequence[tuple[str, object]]):
        from . import _normalize_opdef_form

        self._form = _normalize_opdef_form(form)
        self.validate()

    @property
    def form(self) -> tuple[tuple[str, "Scalar"], ...]:
        return self._form

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)


class DeleteOpDef(OpDef):
    __slots__ = ("key", "_form")

    METHOD = "DELETE"

    def __init__(self, key: str, form: Sequence[tuple[str, object]]):
        from . import _normalize_opdef_form

        if not isinstance(key, str):
            raise TypeError("DELETE OpDef requires a string key")

        self.key = key
        self._form = _normalize_opdef_form(form)
        self.validate()

    @property
    def form(self) -> tuple[tuple[str, "Scalar"], ...]:
        return self._form

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)
