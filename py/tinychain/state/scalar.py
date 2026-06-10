from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from ..uri import URI, uri
from .value import Value

OPREF_DELETE_TAG: str = uri("state", "scalar", "ref", "op", "delete").path
TCREF_IF: str = uri("state", "scalar", "ref", "if").path
TCREF_COND: str = uri("state", "scalar", "ref", "cond").path
TCREF_WHILE: str = uri("state", "scalar", "ref", "while").path
TCREF_FOR_EACH: str = uri("state", "scalar", "ref", "for_each").path
OPDEF_GET: str = uri("state", "scalar", "op", "get").path
OPDEF_PUT: str = uri("state", "scalar", "op", "put").path
OPDEF_POST: str = uri("state", "scalar", "op", "post").path
OPDEF_DELETE: str = uri("state", "scalar", "op", "delete").path
SCALAR_REFLECT_CLASS: str = uri("state", "scalar", "reflect", "class").path
SCALAR_REFLECT_REF_PARTS: str = uri("state", "scalar", "reflect", "ref_parts").path
OPDEF_REFLECT_FORM: str = uri("state", "scalar", "op", "reflect", "form").path
OPDEF_REFLECT_LAST_ID: str = uri("state", "scalar", "op", "reflect", "last_id").path
OPDEF_REFLECT_SCALARS: str = uri("state", "scalar", "op", "reflect", "scalars").path


def _sorted_items(obj: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return sorted(obj.items(), key=lambda kv: kv[0])


@dataclass(frozen=True, slots=True)
class OpRef:
    """
    An IR-shaped OpRef encoding used by `tc-ir` and op-graph payloads.

    This is distinct from the runtime `tinychain.OpRef` request stub (HTTP method/path/body).
    """

    method: str
    subject: str
    args: object

    @staticmethod
    def get(subject: str, key: "Scalar | Value | object" = None) -> "OpRef":
        return OpRef(method="GET", subject=subject, args=[autobox(key).to_json()])

    @staticmethod
    def put(subject: str, key: "Scalar | Value | object", value: "Scalar | Value | object") -> "OpRef":
        return OpRef(
            method="PUT",
            subject=subject,
            args=[
                autobox(key).to_json(),
                autobox(value).to_json(),
            ],
        )

    @staticmethod
    def post(subject: str, params: Mapping[str, "Scalar | Value | object"]) -> "OpRef":
        encoded: dict[str, object] = {}
        for key, value in _sorted_items(params):
            encoded[key] = autobox(value).to_json()
        return OpRef(method="POST", subject=subject, args=encoded)

    @staticmethod
    def delete(subject: str, key: "Scalar | Value | object" = None) -> "OpRef":
        return OpRef(method="DELETE", subject=subject, args=autobox(key).to_json())

    def to_json(self) -> dict[str, object]:
        if self.method == "GET":
            return {self.subject: self.args}
        if self.method == "PUT":
            return {self.subject: self.args}
        if self.method == "POST":
            return {self.subject: self.args}
        if self.method == "DELETE":
            return {OPREF_DELETE_TAG: [self.subject, self.args]}
        raise AssertionError(f"unexpected OpRef.method {self.method}")

    @staticmethod
    def from_json(obj: Any) -> "OpRef":
        if not isinstance(obj, dict) or len(obj) != 1:
            raise TypeError("expected an OpRef map")

        (key, value), = obj.items()
        if key == OPREF_DELETE_TAG:
            if not isinstance(value, list) or len(value) != 2:
                raise TypeError("invalid DELETE opref encoding")
            subject, raw_key = value
            if not isinstance(subject, str):
                raise TypeError("expected DELETE subject to be a string")
            return OpRef(method="DELETE", subject=subject, args=Scalar.from_json(raw_key).to_json())

        if not isinstance(key, str):
            raise TypeError("expected OpRef subject key to be a string")

        if isinstance(value, list):
            if len(value) == 1:
                return OpRef(method="GET", subject=key, args=value)
            if len(value) == 2:
                return OpRef(method="PUT", subject=key, args=value)
            raise TypeError("invalid opref params (expected 1 or 2 elements)")

        if isinstance(value, dict):
            return OpRef(method="POST", subject=key, args=value)

        raise TypeError("invalid OpRef args (expected list or dict)")


@dataclass(frozen=True, slots=True)
class IdRef:
    name: str

    def key(self) -> str:
        return f"${self.name}"


@dataclass(frozen=True, slots=True)
class While:
    cond: "Scalar"
    op: "Scalar"
    state: "Scalar"

    def to_json(self) -> dict[str, object]:
        return {
            TCREF_WHILE: [
                self.cond.to_json(),
                self.op.to_json(),
                self.state.to_json(),
            ]
        }


@dataclass(frozen=True, slots=True)
class Cond:
    cond: "TCRef"
    then: "Scalar"
    or_else: "Scalar"

    def to_json(self) -> dict[str, object]:
        return {
            TCREF_COND: [
                self.cond.to_json(),
                self.then.to_json(),
                self.or_else.to_json(),
            ]
        }


@dataclass(frozen=True, slots=True)
class TCRef:
    op: OpRef | None = None
    id: IdRef | None = None
    cond: Cond | None = None
    while_loop: While | None = None
    for_each: "ForEach | None" = None

    @staticmethod
    def for_id(name: str) -> "TCRef":
        return TCRef(id=IdRef(name))

    def to_json(self) -> dict[str, object]:
        if self.op is not None:
            return self.op.to_json()
        if self.id is not None:
            return {self.id.key(): []}
        if self.cond is not None:
            return self.cond.to_json()
        if self.while_loop is not None:
            return self.while_loop.to_json()
        if self.for_each is not None:
            return self.for_each.to_json()
        return {}

    @staticmethod
    def from_json(obj: Any) -> "TCRef":
        if not isinstance(obj, dict) or len(obj) != 1:
            raise TypeError("expected a TCRef op map")

        (key, value), = obj.items()
        if not isinstance(key, str):
            raise TypeError("expected TCRef map key to be a string")

        # Mirror `tc-ir` decoding rules: only treat `"/..."` or `"$..."` map keys
        # (or the special DELETE tag) as OpRefs/TCRefs. Any other single-entry map
        # is a general-purpose scalar map.
        if not (key == OPREF_DELETE_TAG or key.startswith("/") or key.startswith("$")):
            raise TypeError("not a TCRef op map")

        if key == TCREF_IF or key == TCREF_COND:
            if not isinstance(value, list) or len(value) != 3:
                raise TypeError("invalid Cond ref encoding")
            raw_cond, raw_then, raw_or_else = value
            cond = TCRef.from_json(raw_cond)
            return TCRef(
                cond=Cond(
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
                while_loop=While(
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
                for_each=ForEach(
                    Scalar.from_json(items),
                    Scalar.from_json(op),
                    item_name,
                )
            )

        if key.startswith("$") and isinstance(value, list) and not value:
            return TCRef(id=IdRef(key[1:]))

        return TCRef(op=OpRef.from_json(obj))


def autobox(
    obj: "Scalar | TCRef | OpRef | IdRef | OpDef | Value | Cond | While | ForEach | object",
) -> "Scalar":
    if isinstance(obj, Scalar):
        return obj
    if isinstance(obj, Value):
        op = getattr(obj, "op", None)
        if isinstance(op, OpRef):
            return Scalar(ref=TCRef(op=op))
        return Scalar(value=obj)
    if isinstance(obj, URI):
        return Scalar(value=Value.link(obj))
    if isinstance(obj, TCRef):
        return Scalar(ref=obj)
    if isinstance(obj, Cond):
        return Scalar(ref=TCRef(cond=obj))
    if isinstance(obj, While):
        return Scalar(ref=TCRef(while_loop=obj))
    if isinstance(obj, ForEach):
        return Scalar(ref=TCRef(for_each=obj))
    if isinstance(obj, IdRef):
        return Scalar(ref=TCRef(id=obj))
    if isinstance(obj, OpDef):
        return Scalar(op=obj)
    if isinstance(obj, OpRef):
        return Scalar(ref=TCRef(op=obj))
    if isinstance(obj, dict):
        return Scalar.map_of({k: autobox(v) for k, v in _sorted_items(obj)})
    if isinstance(obj, (list, tuple)):
        return Scalar.tuple_of([autobox(v) for v in obj])

    return Scalar(value=Value.from_json(obj))


def _is_string_scalar(obj: object) -> bool:
    return (
        isinstance(obj, str)
        or (isinstance(obj, Value) and obj.kind == "string")
        or (isinstance(obj, Scalar) and obj.value is not None and obj.value.kind == "string")
    )


def _coerce_form(form: Sequence[tuple[str, object]]) -> list[tuple[str, "Scalar"]]:
    out: list[tuple[str, Scalar]] = []
    for name, value in form:
        if not isinstance(name, str):
            raise TypeError("OpDef form entries must use string ids")
        out.append((name, autobox(value)))
    return out


def id(name: str) -> "Scalar":
    return Scalar.id(name)


def while_loop(
    cond: "Scalar | Value | object",
    op: "Scalar | Value | object",
    state: "Scalar | Value | object",
) -> "Scalar":
    return Scalar.while_loop(cond, op, state)


def after(
    dependency: "Scalar | Value | object",
    then: "Scalar | Value | object",
) -> "Scalar":
    return Scalar.after(dependency, then)


def cond_op(
    cond: "TCRef | Scalar | OpRef | IdRef | object",
    then: "OpDef",
    or_else: "OpDef",
) -> "Scalar":
    return Scalar.cond_op(cond, then, or_else)


def for_each(
    items: "Scalar | Value | object",
    *,
    item_name: str,
    op: "OpDef",
) -> "Scalar":
    return autobox(ForEach(autobox(items), autobox(op), item_name))


@dataclass(frozen=True, slots=True)
class Scalar:
    """
    Minimal v2 Scalar mirror for Python-side reflection and static analysis.

    Encodes/decodes to the same JSON shapes understood by `tc-ir`:
    - scalar values (plain JSON literals)
    - scalar maps/tuples (plain JSON objects/arrays)
    - scalar refs (an OpRef/TCRef single-entry map)
    - scalar op defs (typed `/state/scalar/op/*` maps)
    """

    value: Value | None = None
    ref: TCRef | None = None
    op: "OpDef | None" = None
    map: Mapping[str, "Scalar"] | None = None
    tuple: Sequence["Scalar"] | None = None

    @staticmethod
    def map_of(items: Mapping[str, "Scalar | Value | object"]) -> "Scalar":
        return Scalar(map={key: autobox(value) for key, value in _sorted_items(items)})

    @staticmethod
    def tuple_of(items: Sequence["Scalar | Value | object"]) -> "Scalar":
        return Scalar(tuple=[autobox(item) for item in items])

    @staticmethod
    def id(name: str) -> "Scalar":
        return Scalar(ref=TCRef.for_id(name))

    @staticmethod
    def while_loop(
        cond: "Scalar | Value | object",
        op: "Scalar | Value | object",
        state: "Scalar | Value | object",
    ) -> "Scalar":
        return autobox(
            While(
                autobox(cond),
                autobox(op),
                autobox(state),
            )
        )

    @staticmethod
    def after(
        dependency: "Scalar | Value | object",
        then: "Scalar | Value | object",
    ) -> "Scalar":
        from .context import current_context

        bound_then = autobox(then)
        ctx = current_context()
        if ctx is not None:
            # Bind an explicit dependency edge so side-effect order is encoded in the OpDef form.
            ctx.bind_auto(autobox(dependency), prefix="_after")
        return bound_then

    @staticmethod
    def cond(
        cond: "TCRef | Scalar | OpRef | IdRef | object",
        then: "Scalar | Value | object",
        or_else: "Scalar | Value | object",
    ) -> "Scalar":
        cond_ref = cond if isinstance(cond, TCRef) else autobox(cond).ref
        if cond_ref is None:
            raise TypeError("cond condition must be a ref")
        return autobox(
            Cond(
                cond_ref,
                autobox(then),
                autobox(or_else),
            )
        )

    @staticmethod
    def cond_op(
        cond: "TCRef | Scalar | OpRef | IdRef | object",
        then: "OpDef",
        or_else: "OpDef",
    ) -> "Scalar":
        return Scalar.cond(cond, Scalar(op=then), Scalar(op=or_else))

    def to_json(self) -> object:
        if self.ref is not None:
            return self.ref.to_json()
        if self.op is not None:
            return self.op.to_json()
        if self.map is not None:
            return {k: v.to_json() for k, v in _sorted_items(self.map)}
        if self.tuple is not None:
            return [v.to_json() for v in self.tuple]
        if self.value is not None:
            return self.value.to_json()
        return None

    def class_(self) -> "Scalar":
        opref = OpRef.post(SCALAR_REFLECT_CLASS, {"scalar": self})
        return Scalar(ref=TCRef(op=opref))

    def ref_parts(self) -> "Scalar":
        opref = OpRef.post(SCALAR_REFLECT_REF_PARTS, {"scalar": self})
        return Scalar(ref=TCRef(op=opref))

    def reflect_form(self) -> "Scalar":
        opref = OpRef.post(OPDEF_REFLECT_FORM, {"op": self})
        return Scalar(ref=TCRef(op=opref))

    def reflect_last_id(self) -> "Scalar":
        opref = OpRef.post(OPDEF_REFLECT_LAST_ID, {"op": self})
        return Scalar(ref=TCRef(op=opref))

    def reflect_scalars(self) -> "Scalar":
        opref = OpRef.post(OPDEF_REFLECT_SCALARS, {"op": self})
        return Scalar(ref=TCRef(op=opref))

    def _subject(self) -> str:
        if self.ref is not None and self.ref.id is not None:
            return self.ref.id.key()
        try:
            from .context import current_context  # local import to avoid cycles
        except Exception:
            current_context = None
        if current_context is not None:
            ctx = current_context()
            if ctx is not None:
                bound = ctx.bind_auto(self)
                if bound.ref is not None and bound.ref.id is not None:
                    return bound.ref.id.key()
        raise TypeError("expected a Scalar id ref for an op subject")

    def eq(self, other: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/eq"
        opref = OpRef.post(subject, {"r": autobox(other)})
        return Scalar(ref=TCRef(op=opref))

    def ne(self, other: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/ne"
        opref = OpRef.post(subject, {"r": autobox(other)})
        return Scalar(ref=TCRef(op=opref))

    def add(self, other: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/add"
        opref = OpRef.post(subject, {"r": autobox(other)})
        return Scalar(ref=TCRef(op=opref))

    def gt(self, other: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/gt"
        opref = OpRef.post(subject, {"r": autobox(other)})
        return Scalar(ref=TCRef(op=opref))

    def ge(self, other: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/ge"
        opref = OpRef.post(subject, {"r": autobox(other)})
        return Scalar(ref=TCRef(op=opref))

    def lt(self, other: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/lt"
        opref = OpRef.post(subject, {"r": autobox(other)})
        return Scalar(ref=TCRef(op=opref))

    def le(self, other: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/le"
        opref = OpRef.post(subject, {"r": autobox(other)})
        return Scalar(ref=TCRef(op=opref))

    def logical_and(self, other: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/and"
        opref = OpRef.post(subject, {"r": autobox(other)})
        return Scalar(ref=TCRef(op=opref))

    def logical_not(self) -> "Scalar":
        subject = f"{self._subject()}/not"
        opref = OpRef.post(subject, {})
        return Scalar(ref=TCRef(op=opref))

    def logical_or(self, other: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/or"
        opref = OpRef.post(subject, {"r": autobox(other)})
        return Scalar(ref=TCRef(op=opref))

    def logical_xor(self, other: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/xor"
        opref = OpRef.post(subject, {"r": autobox(other)})
        return Scalar(ref=TCRef(op=opref))

    def len(self) -> "Scalar":
        subject = f"{self._subject()}/len"
        opref = OpRef.post(subject, {})
        return Scalar(ref=TCRef(op=opref))

    def head(self) -> "Scalar":
        subject = f"{self._subject()}/head"
        opref = OpRef.post(subject, {})
        return Scalar(ref=TCRef(op=opref))

    def tail(self) -> "Scalar":
        subject = f"{self._subject()}/tail"
        opref = OpRef.post(subject, {})
        return Scalar(ref=TCRef(op=opref))

    def concat(self, other: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/concat"
        opref = OpRef.post(subject, {"r": autobox(other)})
        return Scalar(ref=TCRef(op=opref))

    def _string_render(self, params: Mapping[str, object]) -> "Scalar":
        subject = f"{self._subject()}/render"
        opref = OpRef.post(subject, params)
        return Scalar(ref=TCRef(op=opref))

    def get(self, index: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/get"
        opref = OpRef.post(subject, {"i": autobox(index)})
        return Scalar(ref=TCRef(op=opref))

    def slice(self, start: "Scalar | Value | object", stop: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/slice"
        opref = OpRef.post(subject, {"start": autobox(start), "stop": autobox(stop)})
        return Scalar(ref=TCRef(op=opref))

    def __getitem__(self, index: "Scalar | Value | object") -> "Scalar":
        if isinstance(index, slice):
            if index.step is not None:
                raise NotImplementedError(f"slice with step: {index}")
            start = 0 if index.start is None else index.start
            stop = self.len() if index.stop is None else index.stop
            return self.slice(start, stop)
        return self.get(index)

    def __add__(self, other: object) -> "Scalar":
        if _is_string_scalar(self) or _is_string_scalar(other):
            return self.concat(other)
        if isinstance(other, (list, tuple)):
            return self.concat(other)
        if isinstance(other, Scalar) and other.tuple is not None:
            return self.concat(other)
        return self.add(other)

    def __radd__(self, other: object) -> "Scalar":
        if _is_string_scalar(other) or _is_string_scalar(self):
            return autobox(other).concat(self)
        if isinstance(other, (list, tuple)):
            return autobox(other).concat(self)
        if isinstance(other, Scalar) and other.tuple is not None:
            return other.concat(self)
        return autobox(other).add(self)

    def __gt__(self, other: object) -> "Scalar":
        return self.gt(other)

    def __ge__(self, other: object) -> "Scalar":
        return self.ge(other)

    def __lt__(self, other: object) -> "Scalar":
        return self.lt(other)

    def __le__(self, other: object) -> "Scalar":
        return self.le(other)

    def __eq__(self, other: object) -> object:
        try:
            from .context import current_context  # local import to avoid cycles
        except Exception:
            current_context = None

        if current_context is not None and current_context() is not None:
            return self.eq(other)
        if isinstance(other, Scalar):
            return (
                self.value,
                self.ref,
                self.op,
                self.map,
                self.tuple,
            ) == (
                other.value,
                other.ref,
                other.op,
                other.map,
                other.tuple,
            )
        return False

    def __ne__(self, other: object) -> object:
        result = self.__eq__(other)
        if isinstance(result, bool):
            return not result
        return self.ne(other)


    def reduce(
        self,
        *,
        item_name: str,
        op: "OpDef",
        value: "Scalar | Value | object",
    ) -> "Scalar":
        subject = f"{self._subject()}/reduce"
        opref = OpRef.post(
            subject,
            {
                "item_name": item_name,
                "op": op,
                "value": autobox(value),
            },
        )
        return Scalar(ref=TCRef(op=opref))

    @staticmethod
    def from_json(obj: Any) -> "Scalar":
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return Scalar(value=Value.from_json(obj))

        if isinstance(obj, list):
            return Scalar.tuple_of([Scalar.from_json(v) for v in obj])

        if isinstance(obj, dict):
            if not obj:
                return Scalar.map_of({})

            if len(obj) == 1:
                (key, _), = obj.items()
                if isinstance(key, str) and key.startswith(uri("state", "scalar", "op").path):
                    return Scalar(op=OpDef.from_json(obj))

            # Decode TCRef/OpRef maps before generic Value maps to avoid
            # treating single-entry refs (e.g. {"$id": []}) as plain maps.
            if len(obj) == 1:
                try:
                    return Scalar(ref=TCRef.from_json(obj))
                except Exception:
                    pass

            # Try to decode as a Value-typed map.
            try:
                return Scalar(value=Value.from_json(obj))
            except Exception:
                pass

            return Scalar.map_of({k: Scalar.from_json(v) for k, v in _sorted_items(obj)})

        raise TypeError(f"cannot decode Scalar from {type(obj).__name__}")

    def walk(self) -> Iterator["Scalar"]:
        stack = [self]
        while stack:
            node = stack.pop()
            if node.map is not None:
                for value in reversed(list(node.map.values())):
                    stack.append(value)
            elif node.tuple is not None:
                for value in reversed(list(node.tuple)):
                    stack.append(value)

            yield node

    def walk_tcref(self) -> Iterator[TCRef]:
        for node in self.walk():
            if node.ref is not None:
                yield node.ref


@dataclass(frozen=True, slots=True)
class Op:
    """
    A callable op identity which can be invoked via GET/PUT/POST/DELETE.

    This provides Python client ergonomics while emitting canonical `Scalar`/`TCRef`/`OpRef`
    encodings.
    """

    subject: str

    def get(self) -> "Get":
        return Get(self.subject)

    def put(self) -> "Put":
        return Put(self.subject)

    def post(self) -> "Post":
        return Post(self.subject)

    def delete(self) -> "Delete":
        return Delete(self.subject)


@dataclass(frozen=True, slots=True)
class Get:
    subject: str

    def __call__(self, key: Scalar | Value | object = None) -> Scalar:
        return Scalar(ref=TCRef(op=OpRef.get(self.subject, key)))


@dataclass(frozen=True, slots=True)
class Put:
    subject: str

    def __call__(self, key: Scalar | Value | object, value: Scalar | Value | object) -> Scalar:
        return Scalar(ref=TCRef(op=OpRef.put(self.subject, key, value)))


@dataclass(frozen=True, slots=True)
class Post:
    subject: str

    def __call__(self, **params: Scalar | Value | object) -> Scalar:
        return Scalar(ref=TCRef(op=OpRef.post(self.subject, params)))


@dataclass(frozen=True, slots=True)
class Delete:
    subject: str

    def __call__(self, key: Scalar | Value | object = None) -> Scalar:
        return Scalar(ref=TCRef(op=OpRef.delete(self.subject, key)))


@dataclass(frozen=True, slots=True)
class OpDef:
    method: str
    form: list[tuple[str, Scalar]]
    key: str | None = None
    value: str | None = None

    @staticmethod
    def get(key: str, form: Sequence[tuple[str, object]]) -> "OpDef":
        return OpDef(method="GET", key=key, form=_coerce_form(form))

    @staticmethod
    def put(key: str, value: str, form: Sequence[tuple[str, object]]) -> "OpDef":
        return OpDef(method="PUT", key=key, value=value, form=_coerce_form(form))

    @staticmethod
    def post(form: Sequence[tuple[str, object]]) -> "OpDef":
        return OpDef(method="POST", form=_coerce_form(form))

    @staticmethod
    def delete(key: str, form: Sequence[tuple[str, object]]) -> "OpDef":
        return OpDef(method="DELETE", key=key, form=_coerce_form(form))

    def last_id(self) -> str | None:
        if not self.form:
            return None
        return self.form[-1][0]

    def walk_scalars(self) -> Iterator[Scalar]:
        for _, scalar in self.form:
            yield from scalar.walk()

    def reflect_form(self) -> "Scalar":
        opref = OpRef.post(OPDEF_REFLECT_FORM, {"op": self})
        return Scalar(ref=TCRef(op=opref))

    def reflect_last_id(self) -> "Scalar":
        opref = OpRef.post(OPDEF_REFLECT_LAST_ID, {"op": self})
        return Scalar(ref=TCRef(op=opref))

    def reflect_scalars(self) -> "Scalar":
        opref = OpRef.post(OPDEF_REFLECT_SCALARS, {"op": self})
        return Scalar(ref=TCRef(op=opref))

    def class_(self) -> "Scalar":
        return Scalar(op=self).class_()

    def to_json(self) -> dict[str, object]:
        if self.method == "GET":
            return {OPDEF_GET: [self.key, _encode_form(self.form)]}
        if self.method == "PUT":
            return {OPDEF_PUT: [self.key, self.value, _encode_form(self.form)]}
        if self.method == "POST":
            return {OPDEF_POST: _encode_form(self.form)}
        if self.method == "DELETE":
            return {OPDEF_DELETE: [self.key, _encode_form(self.form)]}
        raise AssertionError(f"unexpected OpDef.method {self.method}")

    @staticmethod
    def from_json(obj: Any) -> "OpDef":
        if not isinstance(obj, dict) or len(obj) != 1:
            raise TypeError("expected an OpDef map")

        (key, value), = obj.items()
        if not isinstance(key, str):
            raise TypeError("expected OpDef map key to be a string")

        if key == OPDEF_GET:
            if not isinstance(value, list) or len(value) != 2:
                raise TypeError("invalid GET opdef encoding")
            key_name, form = value
            if not isinstance(key_name, str):
                raise TypeError("expected GET key name to be a string")
            return OpDef.get(key_name, _decode_form(form))

        if key == OPDEF_PUT:
            if not isinstance(value, list) or len(value) != 3:
                raise TypeError("invalid PUT opdef encoding")
            key_name, value_name, form = value
            if not isinstance(key_name, str) or not isinstance(value_name, str):
                raise TypeError("expected PUT key/value names to be strings")
            return OpDef.put(key_name, value_name, _decode_form(form))

        if key == OPDEF_POST:
            return OpDef.post(_decode_form(value))

        if key == OPDEF_DELETE:
            if not isinstance(value, list) or len(value) != 2:
                raise TypeError("invalid DELETE opdef encoding")
            key_name, form = value
            if not isinstance(key_name, str):
                raise TypeError("expected DELETE key name to be a string")
            return OpDef.delete(key_name, _decode_form(form))

        raise TypeError("unexpected OpDef map key")


@dataclass(frozen=True, slots=True)
class ForEach:
    items: "Scalar"
    op: "Scalar"
    item_name: str

    def to_json(self) -> dict[str, object]:
        return {
            TCREF_FOR_EACH: [
                self.items.to_json(),
                self.op.to_json(),
                self.item_name,
            ]
        }


def _encode_form(form: Sequence[tuple[str, Scalar]]) -> list[list[object]]:
    return [[name, scalar.to_json()] for name, scalar in form]


def _decode_form(raw: Any) -> list[tuple[str, Scalar]]:
    if not isinstance(raw, list):
        raise TypeError("expected OpDef form to be a list")

    out: list[tuple[str, Scalar]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError("expected OpDef form entries to be [id, scalar]")
        name, scalar = item
        if not isinstance(name, str):
            raise TypeError("expected OpDef form id to be a string")
        out.append((name, Scalar.from_json(scalar)))
    return out
