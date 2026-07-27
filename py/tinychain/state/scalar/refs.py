from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ...uri import path

if TYPE_CHECKING:
    from . import OpDef, Scalar
    from ..value import Value


OPREF_DELETE_TAG: str = path("state", "scalar", "ref", "op", "delete")
TCREF_COND: str = path("state", "scalar", "ref", "cond")
TCREF_WHILE: str = path("state", "scalar", "ref", "while")
TCREF_FOR_EACH: str = path("state", "scalar", "ref", "for_each")


def _sorted_items(obj: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return sorted(obj.items(), key=lambda kv: kv[0])


def _freeze_shape(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    if isinstance(value, Mapping):
        return tuple((key, _freeze_shape(val)) for key, val in _sorted_items(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_shape(item) for item in value)
    return value


def _looks_like_tcref_map(obj: Mapping[str, object]) -> bool:
    if len(obj) != 1:
        return False

    (key, _value), = obj.items()
    return isinstance(key, str) and (key == OPREF_DELETE_TAG or key.startswith("/") or key.startswith("$"))


def coerce_subject(subject: object) -> str:
    if isinstance(subject, str):
        return subject

    while hasattr(subject, "_form"):
        next_subject = getattr(subject, "_form")
        if next_subject is subject:
            break
        subject = next_subject

    if isinstance(subject, IdRef):
        return subject.key()

    if isinstance(subject, OpRef):
        return coerce_subject(subject.subject)

    if isinstance(subject, TCRef):
        inner = subject._form
        if inner is subject:
            raise TypeError(f"expected op subject to be str, got {type(subject).__name__}")
        return coerce_subject(inner)

    try:
        from . import Scalar, form_of

        if isinstance(subject, Scalar):
            return coerce_subject(form_of(subject))
    except (ImportError, AttributeError):
        pass

    if hasattr(subject, "to_json"):
        try:
            as_json = subject.to_json()
            if isinstance(as_json, str):
                return as_json
        except (TypeError, ValueError, AttributeError):
            pass

    raise TypeError(f"expected op subject to be str, got {type(subject).__name__}")


# Backward-compatible alias for existing internal call sites.
_coerce_subject = coerce_subject


class TCRef:
    __slots__ = ("_form",)

    def __init__(self, form: "TCRef | Scalar | IdRef | Cond | While | ForEach"):
        from . import Scalar

        if isinstance(form, TCRef):
            self._form = form._form
            return

        if not isinstance(form, (Scalar, IdRef, Cond, While, ForEach)):
            raise TypeError("TCRef form must be TCRef, Scalar, IdRef, Cond, While, or ForEach")

        self._form = form

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
    def id(name: str) -> "TCRef":
        return TCRef(IdRef(name))

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

        if key == TCREF_COND:
            if not isinstance(value, list) or len(value) != 3:
                raise TypeError("invalid Cond ref encoding")
            raw_cond, raw_then, raw_or_else = value
            cond = TCRef.from_json(raw_cond)
            return TCRef(
                Cond(
                    cond,
                    Scalar.from_json(raw_then),
                    Scalar.from_json(raw_or_else),
                )
            )

        if key == TCREF_WHILE:
            if not isinstance(value, list) or len(value) != 3:
                raise TypeError("invalid While ref encoding")
            cond, op, state = value
            return TCRef(
                While(
                    Scalar.from_json(cond),
                    Scalar.from_json(op),
                    Scalar.from_json(state),
                )
            )
        if key == TCREF_FOR_EACH:
            if not isinstance(value, list) or len(value) != 3:
                raise TypeError("invalid ForEach ref encoding")
            items, op, item_name = value
            if not isinstance(item_name, str):
                raise TypeError("expected ForEach item_name to be a string")
            return TCRef(
                ForEach(
                    Scalar.from_json(items),
                    Scalar.from_json(op),
                    item_name,
                )
            )

        if key.startswith("$") and isinstance(value, list) and not value:
            return TCRef.id(key[1:])

        return OpRef.from_json(obj)


class OpRef(TCRef):
    """
    An IR-shaped OpRef encoding used by `tc-ir` and op-graph payloads.

    This is distinct from the runtime `tinychain.OpRef` request stub (HTTP method/path/body).
    """

    __slots__ = ()

    METHOD: str = ""

    def __init__(self):
        self._form = self

    @property
    def method(self) -> str:
        return type(self).METHOD

    @property
    def subject(self) -> str:
        raise NotImplementedError()

    @property
    def args(self) -> object:
        raise NotImplementedError()

    def _cmp_key(self) -> object:
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self._cmp_key() == other._cmp_key()

    def __hash__(self) -> int:
        return hash((type(self), _freeze_shape(self._cmp_key())))

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
        if key == OPREF_DELETE_TAG:
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
    __slots__ = ("_subject", "_key")

    METHOD = "GET"

    def __init__(self, subject: str, key: "Scalar | Value | object" = None):
        from . import autobox

        super().__init__()
        self._subject = coerce_subject(subject)
        self._key = autobox(key)

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def args(self) -> object:
        return [self._key.to_json()]

    def _cmp_key(self) -> object:
        from . import form_of

        return self._subject, form_of(self._key)

    def to_json(self) -> dict[str, object]:
        return {self.subject: self.args}


class PutOpRef(OpRef):
    __slots__ = ("_subject", "_key", "_value")

    METHOD = "PUT"

    def __init__(self, subject: str, key: "Scalar | Value | object", value: "Scalar | Value | object"):
        from . import autobox

        super().__init__()
        self._subject = coerce_subject(subject)
        self._key = autobox(key)
        self._value = autobox(value)

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def args(self) -> object:
        return [self._key.to_json(), self._value.to_json()]

    def _cmp_key(self) -> object:
        from . import form_of

        return self._subject, form_of(self._key), form_of(self._value)

    def to_json(self) -> dict[str, object]:
        return {self.subject: self.args}


class PostOpRef(OpRef):
    __slots__ = ("_subject", "_params")

    METHOD = "POST"

    def __init__(self, subject: str, params: Mapping[str, "Scalar | Value | object"]):
        from . import autobox

        super().__init__()
        encoded: dict[str, Scalar] = {}
        for key, value in _sorted_items(params):
            encoded[key] = autobox(value)

        self._subject = coerce_subject(subject)
        self._params = encoded

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def args(self) -> object:
        return {key: value.to_json() for key, value in _sorted_items(self._params)}

    def _cmp_key(self) -> object:
        from . import form_of

        return self._subject, tuple((key, form_of(value)) for key, value in _sorted_items(self._params))

    def to_json(self) -> dict[str, object]:
        return {self.subject: self.args}


class DeleteOpRef(OpRef):
    __slots__ = ("_subject", "_key")

    METHOD = "DELETE"

    def __init__(self, subject: str, key: "Scalar | Value | object" = None):
        from . import autobox

        super().__init__()
        self._subject = coerce_subject(subject)
        self._key = autobox(key)

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def args(self) -> object:
        return self._key.to_json()

    def _cmp_key(self) -> object:
        from . import form_of

        return self._subject, form_of(self._key)

    def to_json(self) -> dict[str, object]:
        return {OPREF_DELETE_TAG: [self.subject, self.args]}


class IdRef:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IdRef) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def key(self) -> str:
        return f"${self.name}"


class While:
    __slots__ = ("cond", "op", "state", "_ctx")

    def __init__(self, cond: "Scalar", op: "Scalar", state: "Scalar", *, ctx: object | None = None):
        self.cond = cond
        self.op = op
        self.state = state
        self._ctx = ctx

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
            TCREF_WHILE: [
                self.cond.to_json(),
                self.op.to_json(),
                self.state.to_json(),
            ]
        }


class Cond:
    __slots__ = ("cond", "then", "or_else", "_ctx")

    def __init__(self, cond: "TCRef", then: "Scalar", or_else: "Scalar", *, ctx: object | None = None):
        self.cond = cond
        self.then = then
        self.or_else = or_else
        self._ctx = ctx

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
            TCREF_COND: [
                self.cond.to_json(),
                self.then.to_json(),
                self.or_else.to_json(),
            ]
        }


class ForEach:
    __slots__ = ("items", "op", "item_name", "_ctx")

    def __init__(self, items: "Scalar", op: "Scalar", item_name: str, *, ctx: object | None = None):
        self.items = items
        self.op = op
        self.item_name = item_name
        self._ctx = ctx

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
            TCREF_FOR_EACH: [
                self.items.to_json(),
                self.op.to_json(),
                self.item_name,
            ]
        }

