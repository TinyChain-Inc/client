from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from . import Scalar
from ...uri import URI

if TYPE_CHECKING:
    from . import OpDef
    from ..value import Value


_RESERVED_ID_PATTERNS = (
    "/", "..", "~", "$", "`", "&", "|", "=", "^", "{", "}", "<", ">",
    "'", '"', "?", ":", "@", "#", "(", ")",
)


def _validate_id(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("TinyChain IDs must be strings")
    if not name:
        raise ValueError("TinyChain IDs cannot be empty")
    if any(ord(char) < 0x20 for char in name):
        raise ValueError("TinyChain IDs cannot contain ASCII control characters")
    if any(char.isspace() for char in name):
        raise ValueError("TinyChain IDs cannot contain whitespace")
    for pattern in _RESERVED_ID_PATTERNS:
        if pattern in name:
            raise ValueError(f"TinyChain ID {name!r} contains reserved pattern {pattern!r}")
    return name


def _sorted_items(obj: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return sorted(obj.items(), key=lambda kv: kv[0])


def _subject_str(subject: str | URI) -> str:
    if isinstance(subject, Scalar):
        from .._ops import subject_of

        return subject_of(subject)
    return str(subject)


class TCRef(Scalar):
    __uri__: URI = URI(Scalar, "ref")

    def __init__(self, form: "TCRef | Scalar"):
        if form is self:
            Scalar.__init__(self, self)
            return

        if isinstance(form, TCRef):
            super().__init__(form._form)
            return

        if not isinstance(form, Scalar):
            raise TypeError("TCRef form must be TCRef or Scalar")

        super().__init__(form)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TCRef) and self._form == other._form

    def __hash__(self) -> int:
        return hash((type(self._form), self._form))

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)

    def requires(self, required: set[str]) -> None:
        from ..._lexical import requires

        requires(self, required)

    @staticmethod
    def from_json(obj: Any) -> "TCRef":
        from ... import _native

        return _native.Codec().ref_from_wire(obj)


class OpRef(TCRef):
    """
    An IR-shaped OpRef encoding used by `tc-ir` and op-graph payloads.

    This is distinct from the runtime `tinychain.OpRef` request stub (HTTP method/path/body).
    """

    __uri__: URI = URI(TCRef, "op")

    METHOD: str = ""

    def __init__(self):
        super().__init__(self)

    @property
    def method(self) -> str:
        return type(self).METHOD

    @property
    def subject(self) -> str:
        raise NotImplementedError()

    @property
    def args(self) -> object:
        raise NotImplementedError()

    __eq__ = object.__eq__
    __hash__ = object.__hash__

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)

    def requires(self, required: set[str]) -> None:
        from ..._lexical import requires

        requires(self, required)

    def _argument_form(self) -> object:
        if isinstance(self, (GetOpRef, DeleteOpRef)):
            return self._key
        if isinstance(self, PutOpRef):
            return (self._key, self._value)
        if isinstance(self, PostOpRef):
            return self._params
        raise TypeError(f"unsupported OpRef type {type(self).__name__}")

    @staticmethod
    def from_json(obj: Any) -> "OpRef":
        from ... import _native

        value = _native.Codec().ref_from_wire(obj)
        if not isinstance(value, OpRef):
            raise TypeError("expected an operation reference")
        return value


class GetOpRef(OpRef):
    METHOD = "GET"

    def __init__(self, subject: str | URI, key: "Scalar | Value | object" = None):
        from . import autobox

        super().__init__()
        subject = _subject_str(subject)
        if not isinstance(subject, str):
            raise TypeError(f"expected op subject to be str, got {type(subject).__name__}")
        self._subject = subject
        self._key = autobox(key)

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def args(self) -> object:
        return [self._key.to_json()]

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)


class PutOpRef(OpRef):
    METHOD = "PUT"

    def __init__(self, subject: str | URI, key: "Scalar | Value | object", value: "Scalar | Value | object"):
        from . import autobox

        super().__init__()
        subject = _subject_str(subject)
        if not isinstance(subject, str):
            raise TypeError(f"expected op subject to be str, got {type(subject).__name__}")
        self._subject = subject
        self._key = autobox(key)
        self._value = autobox(value)

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def args(self) -> object:
        return [self._key.to_json(), self._value.to_json()]

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)


class PostOpRef(OpRef):
    METHOD = "POST"

    def __init__(self, subject: str | URI, params: Mapping[str, "Scalar | Value | object"]):
        from . import autobox

        super().__init__()
        encoded: dict[str, Scalar] = {}
        for key, value in _sorted_items(params):
            encoded[key] = autobox(value)

        subject = _subject_str(subject)
        if not isinstance(subject, str):
            raise TypeError(f"expected op subject to be str, got {type(subject).__name__}")
        self._subject = subject
        self._params = encoded

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def args(self) -> object:
        return {key: value.to_json() for key, value in _sorted_items(self._params)}

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)


class DeleteOpRef(OpRef):
    METHOD = "DELETE"

    def __init__(self, subject: str | URI, key: "Scalar | Value | object" = None):
        from . import autobox

        super().__init__()
        subject = _subject_str(subject)
        if not isinstance(subject, str):
            raise TypeError(f"expected op subject to be str, got {type(subject).__name__}")
        self._subject = subject
        self._key = autobox(key)

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def args(self) -> object:
        return self._key.to_json()

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)


class IdRef(TCRef):
    def __init__(self, name: str):
        super().__init__(self)
        self.name = _validate_id(name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IdRef) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def key(self) -> str:
        return f"${self.name}"

    def requires(self, required: set[str]) -> None:
        from ..._lexical import requires

        requires(self, required)


class ControlRef(TCRef):
    def __init__(self):
        super().__init__(self)


class After(ControlRef):
    def __init__(self, when: "Scalar", then: "Scalar"):
        super().__init__()
        self.when = when
        self.then = then

    def __eq__(self, other: object) -> bool:
        return isinstance(other, After) and self.when == other.when and self.then == other.then

    def __hash__(self) -> int:
        return hash((self.when, self.then))

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)

    def requires(self, required: set[str]) -> None:
        from ..._lexical import requires

        requires(self, required)


class While(ControlRef):
    def __init__(self, cond: "Scalar", op: "Scalar", state: "Scalar"):
        super().__init__()
        self.cond = cond
        self.op = op
        self.state = state

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, While)
            and self.cond == other.cond
            and self.op == other.op
            and self.state == other.state
        )

    def __hash__(self) -> int:
        return hash((self.cond, self.op, self.state))

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)

    def requires(self, required: set[str]) -> None:
        from ..._lexical import requires

        requires(self, required)


class Cond(ControlRef):
    def __init__(self, cond: "TCRef", then: "Scalar", or_else: "Scalar"):
        super().__init__()
        self.cond = cond
        self.then = then
        self.or_else = or_else

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Cond)
            and self.cond == other.cond
            and self.then == other.then
            and self.or_else == other.or_else
        )

    def __hash__(self) -> int:
        return hash((self.cond, self.then, self.or_else))

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)

    def requires(self, required: set[str]) -> None:
        from ..._lexical import requires

        requires(self, required)


class ForEach(ControlRef):
    def __init__(self, items: "Scalar", op: "Scalar", item_name: str):
        super().__init__()
        self.items = items
        self.op = op
        self.item_name = _validate_id(item_name)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ForEach)
            and self.items == other.items
            and self.op == other.op
            and self.item_name == other.item_name
        )

    def __hash__(self) -> int:
        return hash((self.items, self.op, self.item_name))

    def to_json(self) -> dict[str, object]:
        from ... import _native

        return _native.to_wire(self)

    def requires(self, required: set[str]) -> None:
        from ..._lexical import requires

        requires(self, required)


def tcref_form_of(value: "TCRef | object") -> object:
    if isinstance(value, TCRef):
        return value._form
    return value
