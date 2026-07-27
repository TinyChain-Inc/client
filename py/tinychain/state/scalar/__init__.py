from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence, cast

from ...uri import URI, path, uri
from ..base import State
from ..collection import Collection
from .opdef import (
    DeleteOpDef,
    GetOpDef,
    OpDef,
    PostOpDef,
    PutOpDef,
)
from .ops import Delete, Get, Op, Post, Put
from .refs import (
    Cond,
    DeleteOpRef,
    ForEach,
    GetOpRef,
    IdRef,
    OpRef,
    PostOpRef,
    PutOpRef,
    TCRef,
    While,
    _looks_like_tcref_map,
    tcref_form_of,
)

if TYPE_CHECKING:
    from ..context import Context


@lru_cache(maxsize=1)
def _value_runtime():
    from ..value import Bool as value_bool
    from ..value import Link as value_link
    from ..value import Map as value_map
    from ..value import Number as value_number
    from ..value import String as value_string
    from ..value import Tuple as value_tuple
    from ..value import Value
    from ..value import form_of as value_form_of

    return Value, value_bool, value_link, value_map, value_number, value_string, value_tuple, value_form_of

SCALAR_ROOT_URI: URI = uri(State, "scalar")
SCALAR_OP_ROOT_URI: URI = uri(SCALAR_ROOT_URI, "op")
SCALAR_REFLECT_ROOT_URI: URI = uri(SCALAR_ROOT_URI, "reflect")
SCALAR_OP_REFLECT_ROOT_URI: URI = uri(SCALAR_OP_ROOT_URI, "reflect")

OPDEF_ROOT_PATH: str = path(SCALAR_OP_ROOT_URI)
OPDEF_GET_PATH: str = path(SCALAR_OP_ROOT_URI, "get")
OPDEF_PUT_PATH: str = path(SCALAR_OP_ROOT_URI, "put")
OPDEF_POST_PATH: str = path(SCALAR_OP_ROOT_URI, "post")
OPDEF_DELETE_PATH: str = path(SCALAR_OP_ROOT_URI, "delete")
SCALAR_REFLECT_CLASS_PATH: str = path(SCALAR_REFLECT_ROOT_URI, "class")
SCALAR_REFLECT_REF_PARTS_PATH: str = path(SCALAR_REFLECT_ROOT_URI, "ref_parts")
OPDEF_REFLECT_FORM_PATH: str = path(SCALAR_OP_REFLECT_ROOT_URI, "form")
OPDEF_REFLECT_LAST_ID_PATH: str = path(SCALAR_OP_REFLECT_ROOT_URI, "last_id")
OPDEF_REFLECT_SCALARS_PATH: str = path(SCALAR_OP_REFLECT_ROOT_URI, "scalars")


def _sorted_items(obj: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return sorted(obj.items(), key=lambda kv: kv[0])


def _json_of(form: object) -> object:
    Value, _, _, _, _, _, _, _ = _value_runtime()

    runtime_op = OpRef.from_runtime(form)
    if runtime_op is not None:
        return runtime_op.to_json()

    if isinstance(form, (OpRef, OpDef, Cond, While, ForEach)):
        return form.to_json()
    if isinstance(form, Value):
        return form.to_json()
    if isinstance(form, IdRef):
        return {form.key(): []}
    if isinstance(form, Scalar):
        return _json_of(form_of(form))
    if isinstance(form, TCRef):
        ref_form = form_of(form)
        if ref_form is form:
            return form.to_json()
        return _json_of(ref_form)
    if isinstance(form, Mapping):
        return {k: _json_of(v) for k, v in _sorted_items(form)}
    if isinstance(form, Sequence) and not isinstance(form, (str, bytes, bytearray)):
        return [_json_of(v) for v in form]
    if form is None or isinstance(form, (bool, int, float, str)):
        return form

    raise TypeError(f"cannot encode form of type {type(form).__name__}")


def autobox(
    obj: "Scalar | TCRef | OpRef | IdRef | OpDef | Value | Cond | While | ForEach | object",
) -> "Scalar":
    Value, _, value_link, _, _, _, _, _ = _value_runtime()

    if isinstance(obj, Value):
        from ...opref import OpRef as RuntimeOpRef

        value_ctx = getattr(obj, "_ctx", None)
        op = getattr(obj, "op", None)
        if isinstance(op, OpRef):
            return _scalar_like(obj, ref=op, ctx=value_ctx)
        runtime_op = OpRef.from_runtime(op)
        if runtime_op is not None:
            return _scalar_like(obj, ref=runtime_op, ctx=value_ctx)
        if isinstance(op, RuntimeOpRef):
            raise TypeError(f"unsupported runtime OpRef type {type(op).__name__}")
        return _scalar_like(obj, value=obj, ctx=value_ctx)

    if isinstance(obj, Scalar):
        return obj
    runtime_op = OpRef.from_runtime(obj)
    if runtime_op is not None:
        return _typed_from_op_ref(runtime_op)
    if isinstance(obj, Collection):
        return Scalar(form_of(obj))
    if isinstance(obj, URI):
        return Scalar(value_link(obj))
    if isinstance(obj, TCRef):
        return _typed_from_tcref(obj)
    if isinstance(obj, Cond):
        return _typed_from_cond(obj)
    if isinstance(obj, While):
        ref = obj
        state = autobox(obj.state)
        return _typed_from_ref_like(ref, state)
    if isinstance(obj, ForEach):
        return Scalar(ref=obj)
    if isinstance(obj, IdRef):
        return Scalar(ref=obj)
    if isinstance(obj, OpDef):
        return Scalar(obj)
    if isinstance(obj, OpRef):
        return _typed_from_op_ref(obj)
    if isinstance(obj, dict):
        return map_of({k: autobox(v) for k, v in _sorted_items(obj)})
    if isinstance(obj, (list, tuple)):
        return tuple_of([autobox(v) for v in obj])

    value_obj = Value.from_json(obj)
    return _scalar_like(value_obj, value=value_obj)


def _is_string_scalar(obj: object) -> bool:
    _, _, _, _, _, value_string, _, _ = _value_runtime()

    scalar_form = form_of(obj) if isinstance(obj, Scalar) else None
    return (
        isinstance(obj, str)
        or isinstance(obj, value_string)
        or isinstance(scalar_form, value_string)
    )


def _normalize_opdef_form(form: Sequence[tuple[str, object]]) -> list[tuple[str, "Scalar"]]:
    out: list[tuple[str, Scalar]] = []
    for name, value in form:
        if not isinstance(name, str):
            raise TypeError("OpDef form entries must use string ids")
        out.append((name, autobox(value)))
    return out


def _context_from_values(*values: object) -> "Context | None":
    for value in values:
        if isinstance(value, Scalar) and value._ctx is not None:
            return value._ctx

        if isinstance(value, Collection):
            value_ctx = getattr(value, "_ctx", None)
            if value_ctx is not None:
                return value_ctx

    return None


def _resolve_context(ctx: "Context | None" = None) -> "Context | None":
    from ..context import resolve_context

    return resolve_context(ctx)


def _literal_number(form: object) -> int | float | bool | None:
    if isinstance(form, (int, float, bool)):
        return form

    _, value_bool, _, _, value_number, _, _, _ = _value_runtime()
    if isinstance(form, (value_number, value_bool)):
        try:
            json_value = form.to_json()
        except (TypeError, ValueError, AttributeError):
            return None
        if isinstance(json_value, (int, float, bool)):
            return json_value

    return None


def id(name: str) -> "Scalar":
    active_ctx = _resolve_context()
    if active_ctx is not None:
        try:
            return getattr(active_ctx, name)
        except AttributeError:
            pass

    # Unbound ids are represented as a generic symbolic ref.
    return Symbol(ref=IdRef(name), ctx=active_ctx)


def map_of(items: Mapping[str, "Scalar | Value | object"]) -> "Scalar":
    boxed = {key: autobox(value) for key, value in _sorted_items(items)}
    return Map(boxed, ctx=_context_from_values(*boxed.values()))


def tuple_of(items: Sequence["Scalar | Value | object"]) -> "Scalar":
    boxed = [autobox(item) for item in items]
    return Tuple(boxed, ctx=_context_from_values(*boxed))


def scalar_for_hint(name: str, hint: object) -> "Scalar":
    base = IdRef(name)
    cls = _scalar_class_for_hint(hint)
    return cls(ref=base)


def _scalar_class_for_hint(hint: object) -> type["Scalar"]:
    Value, value_bool, _, value_map, value_number, value_string, value_tuple, _ = _value_runtime()

    if isinstance(hint, type):
        if issubclass(hint, Collection):
            return hint
        if issubclass(hint, value_number):
            return Number
        if issubclass(hint, value_bool):
            return Bool
        if issubclass(hint, value_tuple):
            return Tuple
        if issubclass(hint, value_map):
            return Map
        if issubclass(hint, value_string):
            return String
        if issubclass(hint, Value):
            return Scalar
        if issubclass(hint, Scalar):
            return hint

    if hint is value_number:
        return Number
    if hint is value_bool:
        return Bool
    if hint is value_tuple:
        return Tuple
    if hint is value_map:
        return Map
    if hint is value_string:
        return String
    return Scalar


def _scalar_like(
    value_obj: Value,
    *,
    value: Value | None = None,
    ref: TCRef | None = None,
    op: "OpDef | None" = None,
    map: Mapping[str, "Scalar"] | None = None,
    tuple: Sequence["Scalar"] | None = None,
    ctx: "Context | None" = None,
) -> "Scalar":
    _, value_bool, _, value_map, value_number, value_string, value_tuple, _ = _value_runtime()

    scalar_type: type[Scalar] = Scalar
    if isinstance(value_obj, value_number):
        scalar_type = Number
    elif isinstance(value_obj, value_bool):
        scalar_type = Bool
    elif isinstance(value_obj, value_tuple):
        scalar_type = Tuple
    elif isinstance(value_obj, value_map):
        scalar_type = Map
    elif isinstance(value_obj, value_string):
        scalar_type = String

    # Preserve tuple/map symbolic shape where available.
    if scalar_type is Map and map is not None:
        scalar = Map(dict(map))
        scalar._ctx = ctx
        return scalar
    if scalar_type is Tuple and tuple is not None:
        scalar = Tuple(list(tuple))
        scalar._ctx = ctx
        return scalar

    if ref is not None:
        scalar = scalar_type(ref=ref)
        scalar._ctx = ctx
        return scalar
    if op is not None:
        scalar = scalar_type(op)
        scalar._ctx = ctx
        return scalar

    if scalar_type is Map and map is not None:
        scalar = Map(dict(map))
        scalar._ctx = ctx
        return scalar
    if scalar_type is Tuple and tuple is not None:
        scalar = Tuple(list(tuple))
        scalar._ctx = ctx
        return scalar

    if value is not None:
        scalar = scalar_type(value)
        scalar._ctx = ctx
        return scalar

    scalar = scalar_type()
    scalar._ctx = ctx
    return scalar


def _typed_template(value: Scalar) -> Scalar:
    constructors: dict[type[Scalar], type[Scalar]] = {
        Number: Number,
        Bool: Bool,
        Tuple: Tuple,
        Map: Map,
        String: String,
        Numeric: Numeric,
        Iterable: Iterable,
        Comparable: Comparable,
    }
    scalar_type = constructors.get(type(value), Scalar)
    return scalar_type()


def _merge_map_shape(left: Mapping[str, Scalar] | None, right: Mapping[str, Scalar] | None) -> Mapping[str, Scalar] | None:
    if left is None or right is None:
        return None

    merged: dict[str, Scalar] = {}
    for key in left.keys() & right.keys():
        lv = left[key]
        rv = right[key]
        merged[key] = _typed_template(lv) if type(lv) is type(rv) else Scalar()

    return merged if merged else None


def _merge_tuple_shape(left: Sequence[Scalar] | None, right: Sequence[Scalar] | None) -> Sequence[Scalar] | None:
    if left is None or right is None:
        return None

    merged: list[Scalar] = []
    for lv, rv in zip(left, right):
        merged.append(_typed_template(lv) if type(lv) is type(rv) else Scalar())

    return merged if merged else None


def _typed_from_ref_like(ref: TCRef, exemplar: Scalar) -> Scalar:
    exemplar_ctx = getattr(exemplar, "_ctx", None)
    exemplar_form = form_of(exemplar)
    if isinstance(exemplar_form, OpDef):
        return Iterable(ref=ref, ctx=exemplar_ctx)
    if isinstance(exemplar, Number):
        return Number(ref=ref, ctx=exemplar_ctx)
    if isinstance(exemplar, Bool):
        return Bool(ref=ref, ctx=exemplar_ctx)
    if isinstance(exemplar, String):
        return String(ref=ref, ctx=exemplar_ctx)
    if isinstance(exemplar, Map):
        if isinstance(exemplar_form, Mapping):
            return Map(dict(exemplar_form), ctx=exemplar_ctx)
        return Map(ref=ref, ctx=exemplar_ctx)
    if isinstance(exemplar, Tuple):
        if isinstance(exemplar_form, Sequence) and not isinstance(exemplar_form, (str, bytes, bytearray)):
            return Tuple(list(exemplar_form), ctx=exemplar_ctx)
        return Tuple(ref=ref, ctx=exemplar_ctx)
    if isinstance(exemplar, Numeric):
        return Symbol(ref=ref, ctx=exemplar_ctx)
    if isinstance(exemplar, Iterable):
        return Iterable(ref=ref, ctx=exemplar_ctx)
    if isinstance(exemplar, Comparable):
        return Comparable(ref=ref, ctx=exemplar_ctx)
    return Symbol(ref=ref, ctx=exemplar_ctx)


def _typed_from_cond(cond_ref: Cond) -> Scalar:
    then_value = autobox(cond_ref.then)
    else_value = autobox(cond_ref.or_else)
    ref = cond_ref
    active_ctx = getattr(cond_ref, "_ctx", None) or _context_from_values(cond_ref.cond, cond_ref.then, cond_ref.or_else)

    if type(then_value) is type(else_value):
        if isinstance(then_value, Map) and isinstance(else_value, Map):
            then_form = form_of(then_value)
            else_form = form_of(else_value)
            merged = _merge_map_shape(
                then_form if isinstance(then_form, Mapping) else None,
                else_form if isinstance(else_form, Mapping) else None,
            )
            if merged is not None:
                return Map(dict(merged), ctx=active_ctx)
            return Map(ref=ref, ctx=active_ctx)
        if isinstance(then_value, Tuple) and isinstance(else_value, Tuple):
            then_form = form_of(then_value)
            else_form = form_of(else_value)
            merged = _merge_tuple_shape(
                then_form if isinstance(then_form, Sequence) and not isinstance(then_form, (str, bytes, bytearray)) else None,
                else_form if isinstance(else_form, Sequence) and not isinstance(else_form, (str, bytes, bytearray)) else None,
            )
            if merged is not None:
                return Tuple(list(merged), ctx=active_ctx)
            return Tuple(ref=ref, ctx=active_ctx)
        return _typed_from_ref_like(ref, then_value)

    return Iterable(ref=ref, ctx=active_ctx)


def _typed_from_op_ref(op_ref: OpRef) -> Scalar:
    subject = op_ref.subject
    if not isinstance(subject, str):
        return Symbol(ref=op_ref)
    ref = op_ref

    exact_dispatch: dict[str, type[Scalar]] = {
        SCALAR_REFLECT_REF_PARTS_PATH: Tuple,
        OPDEF_REFLECT_FORM_PATH: Tuple,
        OPDEF_REFLECT_SCALARS_PATH: Tuple,
        OPDEF_REFLECT_LAST_ID_PATH: String,
    }
    wrapper = exact_dispatch.get(subject)
    if wrapper is not None:
        return wrapper(ref=ref)

    return Symbol(ref=op_ref)


def _typed_from_tcref(ref: TCRef) -> Scalar:
    ref_form = form_of(ref)
    if isinstance(ref_form, OpRef):
        return _typed_from_op_ref(ref_form)
    if isinstance(ref_form, Cond):
        return _typed_from_cond(ref_form)
    if isinstance(ref_form, While):
        state = autobox(ref_form.state)
        return _typed_from_ref_like(ref, state)
    return Symbol(ref=ref)


def while_loop(
    cond: "Scalar | Value | object",
    op: "Scalar | Value | object",
    state: "Scalar | Value | object",
) -> "Scalar":
    active_ctx = _context_from_values(cond, op, state)
    while_ref = While(
        autobox(cond),
        autobox(op),
        autobox(state),
    )
    while_ref._ctx = active_ctx
    result = autobox(
        while_ref
    )
    result._ctx = active_ctx
    return result


def cond(
    condition: "TCRef | Scalar | OpRef | IdRef | object",
    then: "Scalar | Value | object",
    or_else: "Scalar | Value | object",
) -> "Scalar":
    if isinstance(condition, TCRef):
        cond_ref = condition
    else:
        cond_form = form_of(autobox(condition))
        cond_ref = cond_form if isinstance(cond_form, TCRef) else None
    if cond_ref is None:
        raise TypeError("cond condition must be a ref")
    active_ctx = _context_from_values(condition, then, or_else)
    cond_node = Cond(
        cond_ref,
        autobox(then),
        autobox(or_else),
    )
    cond_node._ctx = active_ctx
    result = autobox(
        cond_node
    )
    result._ctx = active_ctx
    return result


def after(
    dependency: "Scalar | Value | object",
    then: "Scalar | Value | object",
) -> "Scalar":
    bound_then = autobox(then)
    active_ctx = _resolve_context() or _context_from_values(dependency, then)
    if active_ctx is not None:
        # Bind an explicit dependency edge so side-effect order is encoded in the OpDef form.
        active_ctx.bind_auto(autobox(dependency), prefix="_after")
        bound_then._ctx = active_ctx
    return bound_then


def for_each(
    items: "Scalar | Value | object",
    *,
    item_name: str,
    op: "OpDef",
) -> "Scalar":
    active_ctx = _context_from_values(items, op)
    foreach_ref = ForEach(autobox(items), autobox(op), item_name)
    foreach_ref._ctx = active_ctx
    result = autobox(foreach_ref)
    result._ctx = active_ctx
    return result


def form_of(value: "Scalar | object") -> object:
    Value, _, _, _, _, _, _, value_form_of = _value_runtime()

    if isinstance(value, Value):
        return value_form_of(value)
    if isinstance(value, TCRef):
        return value._form
    if isinstance(value, Scalar):
        return value._form
    if isinstance(value, Collection):
        return value._form
    return value


class Scalar(State):
    """
    Minimal v2 Scalar mirror for Python-side reflection and static analysis.

    Encodes/decodes to the same JSON shapes understood by `tc-ir`:
    - scalar values (plain JSON literals)
    - scalar maps/tuples (plain JSON objects/arrays)
    - scalar refs (an OpRef/TCRef single-entry map)
    - scalar op defs (typed `/state/scalar/op/*` maps)
    """

    def __init__(self, form: object = None, *, ref: TCRef | None = None, ctx: "Context | None" = None):
        super().__init__(form, ref=ref, ctx=ctx)

    def to_json(self) -> object:
        return _json_of(form_of(self))

    def _reflect(self, subject: str, payload_key: str, payload_value: object, *, rtype: type["Scalar"]) -> "Scalar":
        return rtype._post_ref(subject, {payload_key: payload_value}, ctx=self._ctx)

    def class_(self) -> "Scalar":
        return self._reflect(SCALAR_REFLECT_CLASS_PATH, "scalar", self, rtype=Scalar)

    def ref_parts(self) -> "Tuple":
        return cast(Tuple, self._reflect(SCALAR_REFLECT_REF_PARTS_PATH, "scalar", self, rtype=Tuple))

    def reflect_form(self) -> "Tuple":
        return cast(Tuple, self._reflect(OPDEF_REFLECT_FORM_PATH, "op", self, rtype=Tuple))

    def reflect_last_id(self) -> "String":
        return cast(String, self._reflect(OPDEF_REFLECT_LAST_ID_PATH, "op", self, rtype=String))

    def reflect_scalars(self) -> "Tuple":
        return cast(Tuple, self._reflect(OPDEF_REFLECT_SCALARS_PATH, "op", self, rtype=Tuple))

    @staticmethod
    def from_json(obj: Any) -> "Scalar":
        Value, _, _, _, _, _, _, _ = _value_runtime()

        if obj is None or isinstance(obj, (bool, int, float, str)):
            value = Value.from_json(obj)
            return _scalar_like(value, value=value)

        if isinstance(obj, list):
            return tuple_of([Scalar.from_json(v) for v in obj])

        if isinstance(obj, dict):
            if not obj:
                return map_of({})

            if len(obj) == 1:
                (key, _), = obj.items()
                if isinstance(key, str) and key.startswith(OPDEF_ROOT_PATH):
                    return Scalar(OpDef.from_json(obj))

            # Decode TCRef/OpRef maps before generic Value maps to avoid
            # treating single-entry refs (e.g. {"$id": []}) as plain maps.
            if len(obj) == 1:
                if _looks_like_tcref_map(obj):
                    return Scalar(ref=TCRef.from_json(obj))

            # Try to decode as a Value-typed map.
            try:
                value = Value.from_json(obj)
                return _scalar_like(value, value=value)
            except (TypeError, ValueError):
                pass

            return map_of({k: Scalar.from_json(v) for k, v in _sorted_items(obj)})

        raise TypeError(f"cannot decode Scalar from {type(obj).__name__}")


def _post_ref_call(
    owner: Scalar,
    method: str,
    params: Mapping[str, object],
    *,
    rtype: type[Scalar],
    ctx: "Context | None" = None,
) -> Scalar:
    subject = _subject_method(owner, method, ctx=ctx)
    active_ctx = owner._ctx if ctx is None else ctx
    return rtype(ref=PostOpRef(subject, params), ctx=active_ctx)


def _post_binary_call(owner: Scalar, method: str, other: "Scalar | Value | object", *, rtype: type[Scalar]) -> Scalar:
    return _post_ref_call(owner, method, {"r": autobox(other)}, rtype=rtype)


def _post_unary_call(owner: Scalar, method: str, *, rtype: type[Scalar]) -> Scalar:
    return _post_ref_call(owner, method, {}, rtype=rtype)


def _subject_method(owner: Scalar, method: str, *, ctx: "Context | None" = None) -> str:
    return f"{owner._subject(ctx=ctx)}/{method}"


def _is_concat_operand(left: object, raw: object) -> bool:
    return isinstance(left, (Tuple, Map, String)) or isinstance(raw, (list, tuple, dict, str))


def _reverse_post_call(
    owner: Scalar,
    other: object,
    *,
    method: str,
    rtype: type[Scalar],
    chain_type: type[Scalar] | None = None,
) -> Scalar:
    left = autobox(other)
    if chain_type is not None and isinstance(left, chain_type):
        return getattr(left, method)(owner)
    return rtype(ref=PostOpRef(f"{left._subject()}/{method}", {"r": owner}), ctx=getattr(left, "_ctx", None))


def _seq_form_or_none(obj: object) -> Sequence | None:
    return obj if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)) else None


def _map_form_or_none(obj: object) -> Mapping | None:
    return obj if isinstance(obj, Mapping) else None


class Comparable(Scalar):
    def eq(self, other: "Scalar | Value | object") -> "Bool":
        return cast(Bool, _post_binary_call(self, "eq", other, rtype=Bool))

    def ne(self, other: "Scalar | Value | object") -> "Bool":
        return cast(Bool, _post_binary_call(self, "ne", other, rtype=Bool))

    def gt(self, other: "Scalar | Value | object") -> "Bool":
        return cast(Bool, _post_binary_call(self, "gt", other, rtype=Bool))

    def ge(self, other: "Scalar | Value | object") -> "Bool":
        return cast(Bool, _post_binary_call(self, "ge", other, rtype=Bool))

    def lt(self, other: "Scalar | Value | object") -> "Bool":
        return cast(Bool, _post_binary_call(self, "lt", other, rtype=Bool))

    def le(self, other: "Scalar | Value | object") -> "Bool":
        return cast(Bool, _post_binary_call(self, "le", other, rtype=Bool))

    def __gt__(self, other: object) -> "Bool":
        return self.gt(other)

    def __ge__(self, other: object) -> "Bool":
        return self.ge(other)

    def __lt__(self, other: object) -> "Bool":
        return self.lt(other)

    def __le__(self, other: object) -> "Bool":
        return self.le(other)

    def __eq__(self, other: object) -> "Bool":  # type: ignore[override]
        return self.eq(other)

    def __ne__(self, other: object) -> "Bool":  # type: ignore[override]
        return self.ne(other)


class Numeric(Comparable):
    def add(self, other: "Scalar | Value | object") -> "Numeric":
        return Numeric(ref=PostOpRef(f"{self._subject()}/add", {"r": autobox(other)}), ctx=self._ctx)

    def __add__(self, other: object) -> "Numeric":
        return self.add(other)

    def __radd__(self, other: object) -> "Numeric":
        return cast(Numeric, _reverse_post_call(self, other, method="add", rtype=Numeric, chain_type=Numeric))


class Iterable(Comparable):
    def len(self) -> "Number":
        subject = f"{self._subject()}/len"
        return Number(ref=PostOpRef(subject, {}), ctx=self._ctx)

    def get(self, index: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/get"
        return Symbol(ref=PostOpRef(subject, {"i": autobox(index)}), ctx=self._ctx)

    def __getitem__(self, index: "Scalar | Value | object") -> "Scalar":
        if isinstance(index, slice):
            if index.step is not None:
                raise NotImplementedError(f"slice with step: {index}")
            start = 0 if index.start is None else index.start
            stop = self.len() if index.stop is None else index.stop
            subject = f"{self._subject()}/slice"
            return Iterable(ref=PostOpRef(subject, {"start": autobox(start), "stop": autobox(stop)}), ctx=self._ctx)
        return self.get(index)

    def concat(self, other: "Scalar | Value | object") -> "Iterable":
        return Iterable(ref=PostOpRef(f"{self._subject()}/concat", {"r": autobox(other)}), ctx=self._ctx)

    def __add__(self, other: object) -> "Iterable":
        right = autobox(other)
        if _is_concat_operand(right, other):
            return self.concat(right)
        return Iterable(ref=PostOpRef(f"{self._subject()}/add", {"r": right}), ctx=self._ctx)

    def __radd__(self, other: object) -> "Iterable":
        left = autobox(other)
        method = "add"
        if _is_concat_operand(left, other):
            method = "concat"
        return cast(Iterable, _reverse_post_call(self, other, method=method, rtype=Iterable))


class Symbol(Numeric, Iterable):
    def _string_render(self, params: Mapping[str, object]) -> "String":
        return String(ref=PostOpRef(f"{self._subject()}/render", params), ctx=self._ctx)

    def __add__(self, other: object) -> "Symbol":
        right = autobox(other)
        if _is_concat_operand(right, other):
            return Symbol(ref=PostOpRef(f"{self._subject()}/concat", {"r": right}), ctx=self._ctx)
        return Symbol(ref=PostOpRef(f"{self._subject()}/add", {"r": right}), ctx=self._ctx)

    def __radd__(self, other: object) -> "Symbol":
        left = autobox(other)
        method = "add"
        if _is_concat_operand(left, other):
            method = "concat"
        return cast(Symbol, _reverse_post_call(self, other, method=method, rtype=Symbol))


class Number(Numeric):
    def _compare(self, method: str, other: "Scalar | Value | object", *, reverse_method: str, literal_cmp) -> "Bool":
        right = autobox(other)
        left_form = form_of(self)
        right_form = form_of(right)
        left_literal = _literal_number(left_form)
        right_literal = _literal_number(right_form)
        if left_literal is not None and right_literal is not None:
            return Bool(literal_cmp(left_literal, right_literal))
        if left_literal is not None and isinstance(right, Comparable):
            return getattr(right, reverse_method)(left_literal)
        return Bool(ref=PostOpRef(f"{self._subject()}/{method}", {"r": right}), ctx=self._ctx)

    def _arithmetic(self, method: str, other: "Scalar | Value | object", *, literal_op) -> "Number":
        right = autobox(other)
        left_form = form_of(self)
        right_form = form_of(right)
        left_literal = _literal_number(left_form)
        right_literal = _literal_number(right_form)
        if left_literal is not None and right_literal is not None:
            return Number(literal_op(left_literal, right_literal))
        return Number(ref=PostOpRef(f"{self._subject()}/{method}", {"r": right}), ctx=self._ctx)

    def _reverse_arithmetic(self, method: str, other: object) -> "Number":
        left = autobox(other)
        if isinstance(left, Number):
            return getattr(left, method)(self)
        return Number(ref=PostOpRef(f"{left._subject()}/{method}", {"r": self}), ctx=getattr(left, "_ctx", None))

    def eq(self, other: "Scalar | Value | object") -> "Bool":
        return self._compare("eq", other, reverse_method="eq", literal_cmp=lambda l, r: l == r)

    def ne(self, other: "Scalar | Value | object") -> "Bool":
        return self._compare("ne", other, reverse_method="ne", literal_cmp=lambda l, r: l != r)

    def gt(self, other: "Scalar | Value | object") -> "Bool":
        return self._compare("gt", other, reverse_method="lt", literal_cmp=lambda l, r: l > r)

    def ge(self, other: "Scalar | Value | object") -> "Bool":
        return self._compare("ge", other, reverse_method="le", literal_cmp=lambda l, r: l >= r)

    def lt(self, other: "Scalar | Value | object") -> "Bool":
        return self._compare("lt", other, reverse_method="gt", literal_cmp=lambda l, r: l < r)

    def le(self, other: "Scalar | Value | object") -> "Bool":
        return self._compare("le", other, reverse_method="ge", literal_cmp=lambda l, r: l <= r)

    def add(self, other: "Scalar | Value | object") -> "Number":
        return self._arithmetic("add", other, literal_op=lambda l, r: l + r)

    def sub(self, other: "Scalar | Value | object") -> "Number":
        return self._arithmetic("sub", other, literal_op=lambda l, r: l - r)

    def mul(self, other: "Scalar | Value | object") -> "Number":
        return self._arithmetic("mul", other, literal_op=lambda l, r: l * r)

    def div(self, other: "Scalar | Value | object") -> "Number":
        return self._arithmetic("div", other, literal_op=lambda l, r: l / r)

    def __add__(self, other: object) -> "Number":
        return self.add(other)

    def __radd__(self, other: object) -> "Number":
        return self._reverse_arithmetic("add", other)

    def __sub__(self, other: object) -> "Number":
        return self.sub(other)

    def __rsub__(self, other: object) -> "Number":
        return self._reverse_arithmetic("sub", other)

    def __mul__(self, other: object) -> "Number":
        return self.mul(other)

    def __rmul__(self, other: object) -> "Number":
        return self._reverse_arithmetic("mul", other)

    def __truediv__(self, other: object) -> "Number":
        return self.div(other)

    def __rtruediv__(self, other: object) -> "Number":
        return self._reverse_arithmetic("div", other)

    def __gt__(self, other: object) -> "Bool":
        return self.gt(other)

    def __ge__(self, other: object) -> "Bool":
        return self.ge(other)

    def __lt__(self, other: object) -> "Bool":
        return self.lt(other)

    def __le__(self, other: object) -> "Bool":
        return self.le(other)

class Bool(Comparable):
    def __bool__(self) -> bool:
        raise TypeError(
            "TinyChain Bool has no Python truth value; use tc.state.cond or boolean ops"
        )

    def logical_and(self, other: "Scalar | Value | object") -> "Bool":
        return cast(Bool, _post_binary_call(self, "and", other, rtype=Bool))

    def logical_not(self) -> "Bool":
        return cast(Bool, _post_unary_call(self, "not", rtype=Bool))

    def logical_or(self, other: "Scalar | Value | object") -> "Bool":
        return cast(Bool, _post_binary_call(self, "or", other, rtype=Bool))

    def logical_xor(self, other: "Scalar | Value | object") -> "Bool":
        return cast(Bool, _post_binary_call(self, "xor", other, rtype=Bool))


def _reduce_scalar(
    subject: str,
    op: "OpDef | Scalar | object",
    value: "Scalar | Value | object",
    *,
    ctx: "Context | None" = None,
) -> Scalar:
    from .reduce import infer_reduce_item_name

    infer_ctx = _context_from_values(op, value)
    if infer_ctx is None:
        infer_ctx = ctx
    item_name = infer_reduce_item_name(op, value, ctx=infer_ctx)
    opref = PostOpRef(
        subject,
        {
            "item_name": item_name,
            "op": op,
            "value": autobox(value),
        },
    )
    return Scalar(ref=opref, ctx=ctx)


class Tuple(Iterable):
    def len(self) -> "Number":
        form = _seq_form_or_none(form_of(self))
        if form is not None:
            return Number(len(form))
        subject = f"{self._subject()}/len"
        return Number(ref=PostOpRef(subject, {}), ctx=self._ctx)

    def head(self) -> Scalar:
        subject = f"{self._subject()}/head"
        return Symbol(ref=PostOpRef(subject, {}), ctx=self._ctx)

    def tail(self) -> "Tuple":
        subject = f"{self._subject()}/tail"
        return Tuple(ref=PostOpRef(subject, {}), ctx=self._ctx)

    def concat(self, other: "Scalar | Value | object") -> "Tuple":
        form = _seq_form_or_none(form_of(self))
        right = autobox(other)
        right_form = _seq_form_or_none(form_of(right))
        if form is not None and right_form is not None:
            return Tuple(list(form) + list(right_form))
        subject = f"{self._subject()}/concat"
        return Tuple(ref=PostOpRef(subject, {"r": right}), ctx=self._ctx)

    def get(self, index: "Scalar | Value | object") -> Scalar:
        form = _seq_form_or_none(form_of(self))
        if form is not None and isinstance(index, int):
            return autobox(form[index])
        subject = f"{self._subject()}/get"
        return Symbol(ref=PostOpRef(subject, {"i": autobox(index)}), ctx=self._ctx)

    def slice(self, start: "Scalar | Value | object", stop: "Scalar | Value | object") -> "Tuple":
        form = _seq_form_or_none(form_of(self))
        start_literal = _literal_number(form_of(autobox(start)))
        stop_literal = _literal_number(form_of(autobox(stop)))
        if form is not None and start_literal is not None and stop_literal is not None:
            return Tuple(list(form)[int(start_literal) : int(stop_literal)])
        subject = f"{self._subject()}/slice"
        return Tuple(ref=PostOpRef(subject, {"start": autobox(start), "stop": autobox(stop)}), ctx=self._ctx)

    def __getitem__(self, index: "Scalar | Value | object") -> Scalar:
        if isinstance(index, slice):
            if index.step is not None:
                raise NotImplementedError(f"slice with step: {index}")
            start = 0 if index.start is None else index.start
            stop = self.len() if index.stop is None else index.stop
            return self.slice(start, stop)
        return self.get(index)

    def __add__(self, other: object) -> "Tuple":
        return self.concat(other)

    def __radd__(self, other: object) -> "Tuple":
        return cast(Tuple, _reverse_post_call(self, other, method="concat", rtype=Tuple, chain_type=Tuple))

    def reduce(
        self,
        *,
        op: "OpDef | Scalar | object",
        value: "Scalar | Value | object",
    ) -> Scalar:
        return _reduce_scalar(_subject_method(self, "reduce"), op, value, ctx=self._ctx)


class Map(Comparable):
    def len(self) -> "Number":
        form = _map_form_or_none(form_of(self))
        if form is not None:
            return Number(len(form))
        subject = f"{self._subject()}/len"
        return Number(ref=PostOpRef(subject, {}), ctx=self._ctx)

    def get(self, index: "Scalar | Value | object") -> Scalar:
        form = _map_form_or_none(form_of(self))
        if form is not None and isinstance(index, str) and index in form:
            return autobox(form[index])
        subject = f"{self._subject()}/get"
        return Symbol(ref=PostOpRef(subject, {"i": autobox(index)}), ctx=self._ctx)

    def __getitem__(self, index: "Scalar | Value | object") -> Scalar:
        return self.get(index)

    def reduce(
        self,
        *,
        op: "OpDef | Scalar | object",
        value: "Scalar | Value | object",
    ) -> Scalar:
        return _reduce_scalar(_subject_method(self, "reduce"), op, value, ctx=self._ctx)


class String(Comparable):
    def concat(self, other: "Scalar | Value | object") -> "String":
        subject = f"{self._subject()}/concat"
        return String(ref=PostOpRef(subject, {"r": autobox(other)}), ctx=self._ctx)

    def _string_render(self, params: Mapping[str, object]) -> "String":
        active_ctx = self._ctx or _context_from_values(*params.values())
        subject = f"{self._subject(ctx=active_ctx)}/render"
        return String(ref=PostOpRef(subject, params), ctx=active_ctx)

    def __add__(self, other: object) -> "String":
        return self.concat(other)

    def __radd__(self, other: object) -> "String":
        return cast(String, _reverse_post_call(self, other, method="concat", rtype=String, chain_type=String))


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


def _iter_scalar_nodes(root: Scalar) -> Iterator[Scalar]:
    stack = [root]
    while stack:
        node = stack.pop()
        node_form = form_of(node)
        if isinstance(node_form, Mapping):
            for value in reversed(list(node_form.values())):
                stack.append(value)
        elif isinstance(node_form, Sequence) and not isinstance(node_form, (str, bytes, bytearray)):
            for value in reversed(list(node_form)):
                stack.append(value)

        yield node
