from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from . import Scalar
from ...uri import URI, path, uri

if TYPE_CHECKING:
    from . import OpDef
    from ..value import Value


def _tcref_uri(*segments: str) -> URI:
    return uri(Scalar, "ref", *segments)


def _opref_uri(*segments: str) -> URI:
    return uri(_tcref_uri(), "op", *segments)


def _sorted_items(obj: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return sorted(obj.items(), key=lambda kv: kv[0])


def _looks_like_tcref_map(obj: Mapping[str, object]) -> bool:
    if len(obj) != 1:
        return False

    (key, _value), = obj.items()
    return isinstance(key, str) and (key == path(_opref_uri("delete")) or key.startswith("/") or key.startswith("$"))


class TCRef(Scalar):
    __uri__: URI = _tcref_uri()

    def __init__(self, form: "TCRef | Scalar"):
        if form is self:
            Scalar.__init__(self, ref=self)
            return

        if isinstance(form, TCRef):
            super().__init__(ref=form._form, ctx=form._ctx)
            return

        if not isinstance(form, Scalar):
            raise TypeError("TCRef form must be TCRef or Scalar")

        super().__init__(form, ctx=form._ctx)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TCRef) and self._form == other._form

    def __hash__(self) -> int:
        return hash((type(self._form), self._form))

    def to_json(self) -> dict[str, object]:
        from . import _json_of, form_of

        raw = _json_of(form_of(self))
        if not isinstance(raw, dict):
            raise TypeError("TCRef form must encode to a map")
        return raw

    @staticmethod
    def from_json(obj: Any) -> "TCRef":
        from . import Scalar

        if not isinstance(obj, dict) or len(obj) != 1:
            raise TypeError("expected a TCRef op map")

        (key, value), = obj.items()
        if not isinstance(key, str):
            raise TypeError("expected TCRef map key to be a string")

        if not _looks_like_tcref_map(obj):
            raise TypeError("not a TCRef op map")

        if key == path(_tcref_uri("cond")):
            if not isinstance(value, list) or len(value) != 3:
                raise TypeError("invalid Cond ref encoding")
            raw_cond, raw_then, raw_or_else = value
            cond = TCRef.from_json(raw_cond)
            return Cond(
                cond,
                Scalar.from_json(raw_then),
                Scalar.from_json(raw_or_else),
            )

        if key == path(_tcref_uri("while")):
            if not isinstance(value, list) or len(value) != 3:
                raise TypeError("invalid While ref encoding")
            cond, op, state = value
            return While(
                Scalar.from_json(cond),
                Scalar.from_json(op),
                Scalar.from_json(state),
            )
        if key == path(_tcref_uri("for_each")):
            if not isinstance(value, list) or len(value) != 3:
                raise TypeError("invalid ForEach ref encoding")
            items, op, item_name = value
            if not isinstance(item_name, str):
                raise TypeError("expected ForEach item_name to be a string")
            return ForEach(
                Scalar.from_json(items),
                Scalar.from_json(op),
                item_name,
            )

        if key.startswith("$") and isinstance(value, list) and not value:
            return IdRef(key[1:])

        return OpRef.from_json(obj)


class OpRef(TCRef):
    """
    An IR-shaped OpRef encoding used by `tc-ir` and op-graph payloads.

    This is distinct from the runtime `tinychain.OpRef` request stub (HTTP method/path/body).
    """

    __uri__: URI = _opref_uri()

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

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.subject == other.subject and self.args == other.args

    def __hash__(self) -> int:
        return hash((type(self), repr(self.to_json())))

    def to_json(self) -> dict[str, object]:
        raise NotImplementedError()

    @staticmethod
    def from_runtime(obj: Any) -> "OpRef | None":
        from ...opref import DeleteOpRef as RuntimeDeleteOpRef
        from ...opref import GetOpRef as RuntimeGetOpRef
        from ...opref import PostOpRef as RuntimePostOpRef
        from ...opref import PutOpRef as RuntimePutOpRef

        if isinstance(obj, RuntimeGetOpRef):
            return GetOpRef(obj.path, obj.body)

        if isinstance(obj, RuntimePutOpRef):
            body = obj.body
            if isinstance(body, (list, tuple)) and len(body) == 2:
                return PutOpRef(obj.path, body[0], body[1])
            raise TypeError("runtime PUT op requires [key, value] body for IR conversion")

        if isinstance(obj, RuntimePostOpRef):
            if obj.body is None:
                return PostOpRef(obj.path, {})
            if not isinstance(obj.body, dict):
                raise TypeError("runtime POST op requires object body for IR conversion")
            return PostOpRef(obj.path, obj.body)

        if isinstance(obj, RuntimeDeleteOpRef):
            return DeleteOpRef(obj.path, obj.body)

        return None

    @staticmethod
    def from_json(obj: Any) -> "OpRef":
        from . import Scalar

        if not isinstance(obj, dict) or len(obj) != 1:
            raise TypeError("expected an OpRef map")

        (key, value), = obj.items()
        if key == path(_opref_uri("delete")):
            if not isinstance(value, list) or len(value) != 2:
                raise TypeError("invalid DELETE opref encoding")
            subject, raw_key = value
            if not isinstance(subject, str):
                raise TypeError("expected DELETE subject to be a string")
            return DeleteOpRef(subject, Scalar.from_json(raw_key))

        if not isinstance(key, str):
            raise TypeError("expected OpRef subject key to be a string")

        if isinstance(value, list):
            if len(value) == 1:
                return GetOpRef(key, Scalar.from_json(value[0]))
            if len(value) == 2:
                return PutOpRef(key, Scalar.from_json(value[0]), Scalar.from_json(value[1]))
            raise TypeError("invalid opref params (expected 1 or 2 elements)")

        if isinstance(value, dict):
            params: dict[str, Scalar] = {}
            for name, param_value in _sorted_items(value):
                params[name] = Scalar.from_json(param_value)
            return PostOpRef(key, params)

        raise TypeError("invalid OpRef args (expected list or dict)")


class GetOpRef(OpRef):
    METHOD = "GET"

    def __init__(self, subject: str, key: "Scalar | Value | object" = None):
        from . import autobox

        super().__init__()
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
        return {self.subject: self.args}


class PutOpRef(OpRef):
    METHOD = "PUT"

    def __init__(self, subject: str, key: "Scalar | Value | object", value: "Scalar | Value | object"):
        from . import autobox

        super().__init__()
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
        return {self.subject: self.args}


class PostOpRef(OpRef):
    METHOD = "POST"

    def __init__(self, subject: str, params: Mapping[str, "Scalar | Value | object"]):
        from . import autobox

        super().__init__()
        encoded: dict[str, Scalar] = {}
        for key, value in _sorted_items(params):
            encoded[key] = autobox(value)

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
        return {self.subject: self.args}


class DeleteOpRef(OpRef):
    METHOD = "DELETE"

    def __init__(self, subject: str, key: "Scalar | Value | object" = None):
        from . import autobox

        super().__init__()
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
        return {path(_opref_uri("delete")): [self.subject, self.args]}


class IdRef(TCRef):
    def __init__(self, name: str):
        super().__init__(self)
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IdRef) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def key(self) -> str:
        return f"${self.name}"


class ControlRef(TCRef):
    def __init__(self):
        super().__init__(self)


class While(ControlRef):
    def __init__(self, cond: "Scalar", op: "Scalar", state: "Scalar"):
        super().__init__()
        self.cond = cond
        self.op = op
        self.state = state
        self._ctx = None

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
        return {
            path(_tcref_uri("while")): [
                self.cond.to_json(),
                self.op.to_json(),
                self.state.to_json(),
            ]
        }


class Cond(ControlRef):
    def __init__(self, cond: "TCRef", then: "Scalar", or_else: "Scalar"):
        super().__init__()
        self.cond = cond
        self.then = then
        self.or_else = or_else
        self._ctx = None

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
        return {
            path(_tcref_uri("cond")): [
                self.cond.to_json(),
                self.then.to_json(),
                self.or_else.to_json(),
            ]
        }


class ForEach(ControlRef):
    def __init__(self, items: "Scalar", op: "Scalar", item_name: str):
        super().__init__()
        self.items = items
        self.op = op
        self.item_name = item_name
        self._ctx = None

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
        return {
            path(_tcref_uri("for_each")): [
                self.items.to_json(),
                self.op.to_json(),
                self.item_name,
            ]
        }


def tcref_form_of(value: "TCRef | object") -> object:
    if isinstance(value, TCRef):
        return value._form
    return value
