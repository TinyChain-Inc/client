from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence, cast

from ...uri import URI, path, uri
from .opdef import (
    DeleteOpDef,
    GetOpDef,
    OpDef,
    PostOpDef,
    PutOpDef,
)
from .ops import Delete, Get, Op, Post, Put
from .refs import (
    OPREF_DELETE_TAG,
    TCREF_COND,
    TCREF_FOR_EACH,
    TCREF_WHILE,
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
    coerce_subject as coerce_op_subject,
    _looks_like_tcref_map,
)

if TYPE_CHECKING:
    from ..context import Context


@lru_cache(maxsize=1)
def _value_runtime():
    from ..value import Bool as ValueBool
    from ..value import Link as ValueLink
    from ..value import Map as ValueMap
    from ..value import Number as ValueNumber
    from ..value import String as ValueString
    from ..value import Tuple as ValueTuple
    from ..value import Value
    from ..value import form_of as value_form_of

    return Value, ValueBool, ValueLink, ValueMap, ValueNumber, ValueString, ValueTuple, value_form_of

OPDEF_GET: str = path("state", "scalar", "op", "get")
OPDEF_PUT: str = path("state", "scalar", "op", "put")
OPDEF_POST: str = path("state", "scalar", "op", "post")
OPDEF_DELETE: str = path("state", "scalar", "op", "delete")
SCALAR_REFLECT_CLASS: str = path("state", "scalar", "reflect", "class")
SCALAR_REFLECT_REF_PARTS: str = path("state", "scalar", "reflect", "ref_parts")
OPDEF_REFLECT_FORM: str = path("state", "scalar", "op", "reflect", "form")
OPDEF_REFLECT_LAST_ID: str = path("state", "scalar", "op", "reflect", "last_id")
OPDEF_REFLECT_SCALARS: str = path("state", "scalar", "op", "reflect", "scalars")


def _sorted_items(obj: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return sorted(obj.items(), key=lambda kv: kv[0])


def _json_of(form: object) -> object:
    Value, _, _, _, _, _, _, _ = _value_runtime()

    runtime_op = OpRef.from_runtime(form)
    if runtime_op is not None:
        return runtime_op.to_json()

    if isinstance(form, Value):
        return form.to_json()
    if isinstance(form, Scalar):
        return _json_of(form_of(form))
    if isinstance(form, TCRef):
        return _json_of(form_of(form))
    if isinstance(form, IdRef):
        return {form.key(): []}
    if isinstance(form, (OpRef, OpDef, Cond, While, ForEach)):
        return form.to_json()
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
    Value, _, ValueLink, _, _, _, _, _ = _value_runtime()

    if isinstance(obj, Value):
        from ...opref import OpRef as RuntimeOpRef

        value_ctx = getattr(obj, "_ctx", None)
        op = getattr(obj, "op", None)
        if isinstance(op, OpRef):
            return _scalar_like(obj, ref=TCRef(op), ctx=value_ctx)
        runtime_op = OpRef.from_runtime(op)
        if runtime_op is not None:
            return _scalar_like(obj, ref=TCRef(runtime_op), ctx=value_ctx)
        if isinstance(op, RuntimeOpRef):
            raise TypeError(f"unsupported runtime OpRef type {type(op).__name__}")
        return _scalar_like(obj, value=obj, ctx=value_ctx)

    if isinstance(obj, Scalar):
        return obj
    runtime_op = OpRef.from_runtime(obj)
    if runtime_op is not None:
        return _typed_from_op_ref(runtime_op)
    try:
        from ..collection import Collection

        if isinstance(obj, Collection):
            return Scalar(form_of(obj))
    except ImportError:
        pass
    if isinstance(obj, URI):
        return Scalar(ValueLink(obj))
    if isinstance(obj, TCRef):
        return _typed_from_tcref(obj)
    if isinstance(obj, Cond):
        return _typed_from_cond(obj)
    if isinstance(obj, While):
        ref = TCRef(obj)
        state = autobox(obj.state)
        return _typed_from_ref_like(ref, state)
    if isinstance(obj, ForEach):
        return Scalar(ref=TCRef(obj))
    if isinstance(obj, IdRef):
        return Scalar(ref=TCRef(obj))
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
    _, _, _, _, _, ValueString, _, _ = _value_runtime()

    scalar_form = form_of(obj) if isinstance(obj, Scalar) else None
    return (
        isinstance(obj, str)
        or isinstance(obj, ValueString)
        or isinstance(scalar_form, ValueString)
    )


def _coerce_form(form: Sequence[tuple[str, object]]) -> list[tuple[str, "Scalar"]]:
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

        try:
            from ..collection import Collection

            if isinstance(value, Collection):
                value_ctx = getattr(value, "_ctx", None)
                if value_ctx is not None:
                    return value_ctx
        except ImportError:
            pass

    return None


def _resolve_context(ctx: "Context | None" = None) -> "Context | None":
    return ctx


def _literal_number(form: object) -> int | float | bool | None:
    if isinstance(form, (int, float, bool)):
        return form

    _, ValueBool, _, _, ValueNumber, _, _, _ = _value_runtime()
    if isinstance(form, (ValueNumber, ValueBool)):
        try:
            json_value = form.to_json()
        except (TypeError, ValueError, AttributeError):
            return None
        if isinstance(json_value, (int, float, bool)):
            return json_value

    return None


def id(name: str, *, ctx: "Context | None" = None) -> "Scalar":
    active_ctx = _resolve_context(ctx)
    if active_ctx is not None:
        try:
            return getattr(active_ctx, name)
        except AttributeError:
            pass

    # Unbound ids are represented as a generic symbolic ref.
    return Symbol(ref=TCRef(IdRef(name)), ctx=active_ctx)


def map_of(items: Mapping[str, "Scalar | Value | object"]) -> "Scalar":
    boxed = {key: autobox(value) for key, value in _sorted_items(items)}
    return Map(boxed, ctx=_context_from_values(*boxed.values()))


def tuple_of(items: Sequence["Scalar | Value | object"]) -> "Scalar":
    boxed = [autobox(item) for item in items]
    return Tuple(boxed, ctx=_context_from_values(*boxed))


def scalar_for_hint(name: str, hint: object) -> "Scalar":
    base = TCRef(IdRef(name))
    cls = _scalar_class_for_hint(hint)
    return cls(ref=base)


def _scalar_class_for_hint(hint: object) -> type["Scalar"]:
    ValueBase, ValueBool, _, ValueMap, ValueNumber, ValueString, ValueTuple, _ = _value_runtime()

    if isinstance(hint, type):
        try:
            from ..collection import Collection
        except ImportError:
            Collection = None

        if Collection is not None and issubclass(hint, Collection):
            return hint

        try:
            from ...collection.tensor import Tensor
        except ImportError:
            Tensor = None

        if Tensor is not None and issubclass(hint, Tensor):
            return Tensor
        if issubclass(hint, ValueNumber):
            return Number
        if issubclass(hint, ValueBool):
            return Bool
        if issubclass(hint, ValueTuple):
            return Tuple
        if issubclass(hint, ValueMap):
            return Map
        if issubclass(hint, ValueString):
            return String
        if issubclass(hint, ValueBase):
            return Scalar
        if issubclass(hint, Scalar):
            return hint

    if hint is ValueNumber:
        return Number
    if hint is ValueBool:
        return Bool
    if hint is ValueTuple:
        return Tuple
    if hint is ValueMap:
        return Map
    if hint is ValueString:
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
    _, ValueBool, _, ValueMap, ValueNumber, ValueString, ValueTuple, _ = _value_runtime()

    scalar_type: type[Scalar] = Scalar
    if isinstance(value_obj, ValueNumber):
        scalar_type = Number
    elif isinstance(value_obj, ValueBool):
        scalar_type = Bool
    elif isinstance(value_obj, ValueTuple):
        scalar_type = Tuple
    elif isinstance(value_obj, ValueMap):
        scalar_type = Map
    elif isinstance(value_obj, ValueString):
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
    ref = TCRef(cond_ref)
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
        return Symbol(ref=TCRef(op_ref))
    ref = TCRef(op_ref)

    exact_dispatch: dict[str, type[Scalar]] = {
        SCALAR_REFLECT_REF_PARTS: Tuple,
        OPDEF_REFLECT_FORM: Tuple,
        OPDEF_REFLECT_SCALARS: Tuple,
        OPDEF_REFLECT_LAST_ID: String,
    }
    wrapper = exact_dispatch.get(subject)
    if wrapper is not None:
        return wrapper(ref=ref)

    return Symbol(ref=TCRef(op_ref))


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
    result = autobox(
        While(
            autobox(cond),
            autobox(op),
            autobox(state),
            ctx=active_ctx,
        )
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
    result = autobox(
        Cond(
            cond_ref,
            autobox(then),
            autobox(or_else),
            ctx=active_ctx,
        )
    )
    result._ctx = active_ctx
    return result


def after(
    dependency: "Scalar | Value | object",
    then: "Scalar | Value | object",
    *,
    ctx: "Context | None" = None,
) -> "Scalar":
    bound_then = autobox(then)
    active_ctx = _resolve_context(ctx) or _context_from_values(dependency, then)
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
    result = autobox(ForEach(autobox(items), autobox(op), item_name, ctx=active_ctx))
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
    try:
        from ..collection import Collection

        if isinstance(value, Collection):
            return value._form
    except ImportError:
        pass
    return value


class Scalar:
    """
    Minimal v2 Scalar mirror for Python-side reflection and static analysis.

    Encodes/decodes to the same JSON shapes understood by `tc-ir`:
    - scalar values (plain JSON literals)
    - scalar maps/tuples (plain JSON objects/arrays)
    - scalar refs (an OpRef/TCRef single-entry map)
    - scalar op defs (typed `/state/scalar/op/*` maps)
    """

    __slots__ = ("_form", "_ctx")

    def __init__(self, form: object = None, *, ref: TCRef | None = None, ctx: "Context | None" = None):
        if form is not None and ref is not None:
            raise TypeError("Scalar accepts either form or ref, not both")

        self._form = ref if ref is not None else form
        self._ctx = ctx

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Scalar):
            return False

        return form_of(self) == form_of(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        # Scalar instances may appear transiently as symbolic map keys during
        # route compilation; hash on encoded shape to keep semantics stable.
        return hash(repr(self.to_json()))

    def to_json(self) -> object:
        return _json_of(form_of(self))

    @classmethod
    def _from_opref(cls, opref: OpRef, *, ctx: "Context | None" = None) -> "Scalar":
        try:
            return cls(ref=TCRef(opref), ctx=ctx)
        except TypeError:
            # Value subclasses (e.g. Tuple, Null) carry deferred ops in their
            # own constructor form and do not accept ref=...
            scalar = cls(opref)
            if isinstance(scalar, Scalar):
                scalar._ctx = ctx
            return scalar

    @classmethod
    def _get_ref(
        cls,
        subject: str,
        key: "Scalar | Value | object" = None,
        *,
        ctx: "Context | None" = None,
    ) -> "Scalar":
        return cls._from_opref(GetOpRef(subject, key), ctx=ctx)

    @classmethod
    def _put_ref(
        cls,
        subject: str,
        key: "Scalar | Value | object",
        value: "Scalar | Value | object",
        *,
        ctx: "Context | None" = None,
    ) -> "Scalar":
        return cls._from_opref(PutOpRef(subject, key, value), ctx=ctx)

    @classmethod
    def _post_ref(
        cls,
        subject: str,
        params: Mapping[str, "Scalar | Value | object"] | None = None,
        *,
        ctx: "Context | None" = None,
    ) -> "Scalar":
        return cls._from_opref(PostOpRef(subject, params or {}), ctx=ctx)

    @classmethod
    def _delete_ref(
        cls,
        subject: str,
        key: "Scalar | Value | object" = None,
        *,
        ctx: "Context | None" = None,
    ) -> "Scalar":
        return cls._from_opref(DeleteOpRef(subject, key), ctx=ctx)

    def _subject_ref(self, method: str | None = None, *, ctx: "Context | None" = None) -> str:
        subject = self._subject(ctx=ctx)
        return f"{subject}/{method}" if method else subject

    def _get(
        self,
        method: str | None = None,
        key: "Scalar | Value | object" = None,
        *,
        rtype: type["Scalar"] | None = None,
        ctx: "Context | None" = None,
    ) -> "Scalar":
        cls = rtype or type(self)
        active_ctx = ctx if ctx is not None else self._ctx
        return cls._get_ref(self._subject_ref(method, ctx=active_ctx), key, ctx=active_ctx)

    def _put(
        self,
        value: "Scalar | Value | object",
        method: str | None = None,
        key: "Scalar | Value | object" = None,
        *,
        rtype: type["Scalar"] | None = None,
        ctx: "Context | None" = None,
    ) -> "Scalar":
        cls = rtype or type(self)
        active_ctx = ctx if ctx is not None else self._ctx
        return cls._put_ref(self._subject_ref(method, ctx=active_ctx), key, value, ctx=active_ctx)

    def _post(
        self,
        method: str | None = None,
        params: Mapping[str, "Scalar | Value | object"] | None = None,
        *,
        rtype: type["Scalar"] | None = None,
        ctx: "Context | None" = None,
    ) -> "Scalar":
        cls = rtype or type(self)
        active_ctx = ctx if ctx is not None else self._ctx
        return cls._post_ref(self._subject_ref(method, ctx=active_ctx), params, ctx=active_ctx)

    def _delete(
        self,
        method: str | None = None,
        key: "Scalar | Value | object" = None,
        *,
        rtype: type["Scalar"] | None = None,
        ctx: "Context | None" = None,
    ) -> "Scalar":
        cls = rtype or type(self)
        active_ctx = ctx if ctx is not None else self._ctx
        return cls._delete_ref(self._subject_ref(method, ctx=active_ctx), key, ctx=active_ctx)

    def class_(self) -> "Scalar":
        return Scalar._post_ref(SCALAR_REFLECT_CLASS, {"scalar": self}, ctx=self._ctx)

    def ref_parts(self) -> "Tuple":
        return Tuple._post_ref(SCALAR_REFLECT_REF_PARTS, {"scalar": self}, ctx=self._ctx)

    def reflect_form(self) -> "Tuple":
        return Tuple._post_ref(OPDEF_REFLECT_FORM, {"op": self}, ctx=self._ctx)

    def reflect_last_id(self) -> "String":
        return String._post_ref(OPDEF_REFLECT_LAST_ID, {"op": self}, ctx=self._ctx)

    def reflect_scalars(self) -> "Tuple":
        return Tuple._post_ref(OPDEF_REFLECT_SCALARS, {"op": self}, ctx=self._ctx)

    def _subject(self, *, ctx: "Context | None" = None) -> str:
        form = form_of(self)
        active_ctx = ctx if ctx is not None else self._ctx

        def _stage_subject() -> str | None:
            if active_ctx is None:
                return None

            bound = active_ctx.bind_auto(self)
            bound_form = form_of(bound)
            if isinstance(bound_form, TCRef):
                bound_ref_form = form_of(bound_form)
                if isinstance(bound_ref_form, IdRef):
                    return bound_ref_form.key()

            return None

        def _subject_from_op(candidate: object) -> str | None:
            if isinstance(candidate, OpRef):
                subject = coerce_op_subject(candidate.subject)
            else:
                runtime_candidate = OpRef.from_runtime(candidate)
                if runtime_candidate is None:
                    return None
                subject = coerce_op_subject(runtime_candidate.subject)

            staged = _stage_subject()
            return staged if staged is not None else subject

        if isinstance(form, OpRef):
            subject = _subject_from_op(form)
            if subject is not None:
                return subject

        subject = _subject_from_op(form)
        if subject is not None:
            return subject

        if isinstance(form, Mapping) and len(form) == 1:
            (key, _value), = form.items()
            if isinstance(key, str) and key.startswith("$"):
                return key

        if isinstance(form, Sequence) and not isinstance(form, (str, bytes, bytearray)):
            staged = _stage_subject()
            if staged is not None:
                return staged

        if isinstance(form, TCRef):
            ref_form = form_of(form)
            if isinstance(ref_form, IdRef):
                return ref_form.key()

            subject = _subject_from_op(ref_form)
            if subject is not None:
                return subject

            if isinstance(ref_form, Cond):
                return TCREF_COND
            if isinstance(ref_form, While):
                return TCREF_WHILE
            if isinstance(ref_form, ForEach):
                return TCREF_FOR_EACH

        staged = _stage_subject()
        if staged is not None:
            return staged

        inner = form_of(form) if isinstance(form, TCRef) else None
        raise TypeError(
            "expected a Scalar id/op ref for an op subject, got "
            f"{type(form).__name__}: {form!r}; inner={type(inner).__name__ if inner is not None else None}: {inner!r}"
        )

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
                if isinstance(key, str) and key.startswith(path("state", "scalar", "op")):
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


class Comparable(Scalar):
    def eq(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/eq", {"r": autobox(other)})), ctx=self._ctx)

    def ne(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/ne", {"r": autobox(other)})), ctx=self._ctx)

    def gt(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/gt", {"r": autobox(other)})), ctx=self._ctx)

    def ge(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/ge", {"r": autobox(other)})), ctx=self._ctx)

    def lt(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/lt", {"r": autobox(other)})), ctx=self._ctx)

    def le(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/le", {"r": autobox(other)})), ctx=self._ctx)

    def __gt__(self, other: object) -> "Bool":
        return self.gt(other)

    def __ge__(self, other: object) -> "Bool":
        return self.ge(other)

    def __lt__(self, other: object) -> "Bool":
        return self.lt(other)

    def __le__(self, other: object) -> "Bool":
        return self.le(other)

    def __eq__(self, other: object) -> Any:  # type: ignore[override]
        form = form_of(self)
        if self._ctx is not None or isinstance(form, (TCRef, OpRef)) or OpRef.from_runtime(form) is not None:
            return self.eq(other)

        if isinstance(other, Scalar):
            return form_of(self) == form_of(other)

        return False

    def __ne__(self, other: object) -> Any:  # type: ignore[override]
        result = self.__eq__(other)
        if isinstance(result, bool):
            return not result
        return self.ne(other)


class Numeric(Comparable):
    def add(self, other: "Scalar | Value | object") -> "Numeric":
        return Numeric(ref=TCRef(PostOpRef(f"{self._subject()}/add", {"r": autobox(other)})), ctx=self._ctx)

    def __add__(self, other: object) -> "Numeric":
        return self.add(other)

    def __radd__(self, other: object) -> "Numeric":
        left = autobox(other)
        if isinstance(left, Numeric):
            return left.add(self)
        return Numeric(ref=TCRef(PostOpRef(f"{left._subject()}/add", {"r": self})), ctx=getattr(left, "_ctx", None))


class Iterable(Comparable):
    def len(self) -> "Number":
        subject = f"{self._subject()}/len"
        return Number(ref=TCRef(PostOpRef(subject, {})), ctx=self._ctx)

    def get(self, index: "Scalar | Value | object") -> "Scalar":
        subject = f"{self._subject()}/get"
        return Symbol(ref=TCRef(PostOpRef(subject, {"i": autobox(index)})), ctx=self._ctx)

    def __getitem__(self, index: "Scalar | Value | object") -> "Scalar":
        if isinstance(index, slice):
            if index.step is not None:
                raise NotImplementedError(f"slice with step: {index}")
            start = 0 if index.start is None else index.start
            stop = self.len() if index.stop is None else index.stop
            subject = f"{self._subject()}/slice"
            return Iterable(ref=TCRef(PostOpRef(subject, {"start": autobox(start), "stop": autobox(stop)})), ctx=self._ctx)
        return self.get(index)

    def concat(self, other: "Scalar | Value | object") -> "Iterable":
        return Iterable(ref=TCRef(PostOpRef(f"{self._subject()}/concat", {"r": autobox(other)})), ctx=self._ctx)

    def __add__(self, other: object) -> "Iterable":
        right = autobox(other)
        if isinstance(right, (Tuple, Map, String)) or isinstance(other, (list, tuple, dict, str)):
            return self.concat(right)
        return Iterable(ref=TCRef(PostOpRef(f"{self._subject()}/add", {"r": right})), ctx=self._ctx)

    def __radd__(self, other: object) -> "Iterable":
        left = autobox(other)
        method = "add"
        if isinstance(left, (Tuple, Map, String)) or isinstance(other, (list, tuple, dict, str)):
            method = "concat"
        return Iterable(ref=TCRef(PostOpRef(f"{left._subject()}/{method}", {"r": self})), ctx=getattr(left, "_ctx", None))


class Symbol(Numeric, Iterable):
    def _string_render(self, params: Mapping[str, object]) -> "String":
        return String(ref=TCRef(PostOpRef(f"{self._subject()}/render", params)), ctx=self._ctx)

    def __add__(self, other: object) -> "Symbol":
        right = autobox(other)
        if isinstance(right, (Tuple, Map, String)) or isinstance(other, (list, tuple, dict, str)):
            return Symbol(ref=TCRef(PostOpRef(f"{self._subject()}/concat", {"r": right})), ctx=self._ctx)
        return Symbol(ref=TCRef(PostOpRef(f"{self._subject()}/add", {"r": right})), ctx=self._ctx)

    def __radd__(self, other: object) -> "Symbol":
        left = autobox(other)
        method = "add"
        if isinstance(left, (Tuple, Map, String)) or isinstance(other, (list, tuple, dict, str)):
            method = "concat"
        return Symbol(ref=TCRef(PostOpRef(f"{left._subject()}/{method}", {"r": self})), ctx=getattr(left, "_ctx", None))


class Number(Numeric):
    def eq(self, other: "Scalar | Value | object") -> "Bool":
        right = autobox(other)
        left_form = form_of(self)
        right_form = form_of(right)
        left_literal = _literal_number(left_form)
        right_literal = _literal_number(right_form)
        if left_literal is not None and right_literal is not None:
            return Bool(left_literal == right_literal)
        if left_literal is not None and isinstance(right, Comparable):
            return right.eq(left_literal)
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/eq", {"r": right})), ctx=self._ctx)

    def ne(self, other: "Scalar | Value | object") -> "Bool":
        right = autobox(other)
        left_form = form_of(self)
        right_form = form_of(right)
        left_literal = _literal_number(left_form)
        right_literal = _literal_number(right_form)
        if left_literal is not None and right_literal is not None:
            return Bool(left_literal != right_literal)
        if left_literal is not None and isinstance(right, Comparable):
            return right.ne(left_literal)
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/ne", {"r": right})), ctx=self._ctx)

    def gt(self, other: "Scalar | Value | object") -> "Bool":
        right = autobox(other)
        left_form = form_of(self)
        right_form = form_of(right)
        left_literal = _literal_number(left_form)
        right_literal = _literal_number(right_form)
        if left_literal is not None and right_literal is not None:
            return Bool(left_literal > right_literal)
        if left_literal is not None and isinstance(right, Comparable):
            return right.lt(left_literal)
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/gt", {"r": right})), ctx=self._ctx)

    def ge(self, other: "Scalar | Value | object") -> "Bool":
        right = autobox(other)
        left_form = form_of(self)
        right_form = form_of(right)
        left_literal = _literal_number(left_form)
        right_literal = _literal_number(right_form)
        if left_literal is not None and right_literal is not None:
            return Bool(left_literal >= right_literal)
        if left_literal is not None and isinstance(right, Comparable):
            return right.le(left_literal)
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/ge", {"r": right})), ctx=self._ctx)

    def lt(self, other: "Scalar | Value | object") -> "Bool":
        right = autobox(other)
        left_form = form_of(self)
        right_form = form_of(right)
        left_literal = _literal_number(left_form)
        right_literal = _literal_number(right_form)
        if left_literal is not None and right_literal is not None:
            return Bool(left_literal < right_literal)
        if left_literal is not None and isinstance(right, Comparable):
            return right.gt(left_literal)
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/lt", {"r": right})), ctx=self._ctx)

    def le(self, other: "Scalar | Value | object") -> "Bool":
        right = autobox(other)
        left_form = form_of(self)
        right_form = form_of(right)
        left_literal = _literal_number(left_form)
        right_literal = _literal_number(right_form)
        if left_literal is not None and right_literal is not None:
            return Bool(left_literal <= right_literal)
        if left_literal is not None and isinstance(right, Comparable):
            return right.ge(left_literal)
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/le", {"r": right})), ctx=self._ctx)

    def add(self, other: "Scalar | Value | object") -> "Number":
        right = autobox(other)
        left_form = form_of(self)
        right_form = form_of(right)
        left_literal = _literal_number(left_form)
        right_literal = _literal_number(right_form)
        if left_literal is not None and right_literal is not None:
            return Number(left_literal + right_literal)
        subject = f"{self._subject()}/add"
        opref = PostOpRef(subject, {"r": right})
        return Number(ref=TCRef(opref), ctx=self._ctx)

    def sub(self, other: "Scalar | Value | object") -> "Number":
        right = autobox(other)
        left_form = form_of(self)
        right_form = form_of(right)
        left_literal = _literal_number(left_form)
        right_literal = _literal_number(right_form)
        if left_literal is not None and right_literal is not None:
            return Number(left_literal - right_literal)
        subject = f"{self._subject()}/sub"
        opref = PostOpRef(subject, {"r": right})
        return Number(ref=TCRef(opref), ctx=self._ctx)

    def mul(self, other: "Scalar | Value | object") -> "Number":
        right = autobox(other)
        left_form = form_of(self)
        right_form = form_of(right)
        left_literal = _literal_number(left_form)
        right_literal = _literal_number(right_form)
        if left_literal is not None and right_literal is not None:
            return Number(left_literal * right_literal)
        subject = f"{self._subject()}/mul"
        opref = PostOpRef(subject, {"r": right})
        return Number(ref=TCRef(opref), ctx=self._ctx)

    def div(self, other: "Scalar | Value | object") -> "Number":
        right = autobox(other)
        left_form = form_of(self)
        right_form = form_of(right)
        left_literal = _literal_number(left_form)
        right_literal = _literal_number(right_form)
        if left_literal is not None and right_literal is not None:
            return Number(left_literal / right_literal)
        subject = f"{self._subject()}/div"
        opref = PostOpRef(subject, {"r": right})
        return Number(ref=TCRef(opref), ctx=self._ctx)

    def __add__(self, other: object) -> "Number":
        return self.add(other)

    def __radd__(self, other: object) -> "Number":
        left = autobox(other)
        if isinstance(left, Number):
            return left.add(self)
        return Number(ref=TCRef(PostOpRef(f"{left._subject()}/add", {"r": self})), ctx=getattr(left, "_ctx", None))

    def __sub__(self, other: object) -> "Number":
        return self.sub(other)

    def __rsub__(self, other: object) -> "Number":
        left = autobox(other)
        if isinstance(left, Number):
            return left.sub(self)
        return Number(ref=TCRef(PostOpRef(f"{left._subject()}/sub", {"r": self})), ctx=getattr(left, "_ctx", None))

    def __mul__(self, other: object) -> "Number":
        return self.mul(other)

    def __rmul__(self, other: object) -> "Number":
        left = autobox(other)
        if isinstance(left, Number):
            return left.mul(self)
        return Number(ref=TCRef(PostOpRef(f"{left._subject()}/mul", {"r": self})), ctx=getattr(left, "_ctx", None))

    def __truediv__(self, other: object) -> "Number":
        return self.div(other)

    def __rtruediv__(self, other: object) -> "Number":
        left = autobox(other)
        if isinstance(left, Number):
            return left.div(self)
        return Number(ref=TCRef(PostOpRef(f"{left._subject()}/div", {"r": self})), ctx=getattr(left, "_ctx", None))

    def __gt__(self, other: object) -> "Bool":
        return self.gt(other)

    def __ge__(self, other: object) -> "Bool":
        return self.ge(other)

    def __lt__(self, other: object) -> "Bool":
        return self.lt(other)

    def __le__(self, other: object) -> "Bool":
        return self.le(other)

class Bool(Comparable):
    def eq(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/eq", {"r": autobox(other)})), ctx=self._ctx)

    def ne(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/ne", {"r": autobox(other)})), ctx=self._ctx)

    def logical_and(self, other: "Scalar | Value | object") -> "Bool":
        subject = f"{self._subject()}/and"
        opref = PostOpRef(subject, {"r": autobox(other)})
        return Bool(ref=TCRef(opref), ctx=self._ctx)

    def logical_not(self) -> "Bool":
        subject = f"{self._subject()}/not"
        opref = PostOpRef(subject, {})
        return Bool(ref=TCRef(opref), ctx=self._ctx)

    def logical_or(self, other: "Scalar | Value | object") -> "Bool":
        subject = f"{self._subject()}/or"
        opref = PostOpRef(subject, {"r": autobox(other)})
        return Bool(ref=TCRef(opref), ctx=self._ctx)

    def logical_xor(self, other: "Scalar | Value | object") -> "Bool":
        subject = f"{self._subject()}/xor"
        opref = PostOpRef(subject, {"r": autobox(other)})
        return Bool(ref=TCRef(opref), ctx=self._ctx)


def _reduce_scalar(
    subject: str,
    op: "OpDef | Scalar | object",
    value: "Scalar | Value | object",
    *,
    ctx: "Context | None" = None,
) -> Scalar:
    infer_ctx = _context_from_values(op, value)
    if infer_ctx is None:
        infer_ctx = ctx
    item_name = _infer_reduce_item_name(op, value, ctx=infer_ctx)
    opref = PostOpRef(
        subject,
        {
            "item_name": item_name,
            "op": op,
            "value": autobox(value),
        },
    )
    return Scalar(ref=TCRef(opref), ctx=ctx)


class Tuple(Iterable):
    def eq(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/eq", {"r": autobox(other)})), ctx=self._ctx)

    def ne(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/ne", {"r": autobox(other)})), ctx=self._ctx)

    def len(self) -> "Number":
        form = form_of(self)
        if isinstance(form, Sequence) and not isinstance(form, (str, bytes, bytearray)):
            return Number(len(form))
        subject = f"{self._subject()}/len"
        return Number(ref=TCRef(PostOpRef(subject, {})), ctx=self._ctx)

    def head(self) -> Scalar:
        subject = f"{self._subject()}/head"
        return Symbol(ref=TCRef(PostOpRef(subject, {})), ctx=self._ctx)

    def tail(self) -> "Tuple":
        subject = f"{self._subject()}/tail"
        return Tuple(ref=TCRef(PostOpRef(subject, {})), ctx=self._ctx)

    def concat(self, other: "Scalar | Value | object") -> "Tuple":
        form = form_of(self)
        right = autobox(other)
        right_form = form_of(right)
        if (
            isinstance(form, Sequence)
            and not isinstance(form, (str, bytes, bytearray))
            and isinstance(right_form, Sequence)
            and not isinstance(right_form, (str, bytes, bytearray))
        ):
            return Tuple(list(form) + list(right_form))
        subject = f"{self._subject()}/concat"
        return Tuple(ref=TCRef(PostOpRef(subject, {"r": right})), ctx=self._ctx)

    def get(self, index: "Scalar | Value | object") -> Scalar:
        form = form_of(self)
        if isinstance(form, Sequence) and not isinstance(form, (str, bytes, bytearray)) and isinstance(index, int):
            return autobox(form[index])
        subject = f"{self._subject()}/get"
        return Symbol(ref=TCRef(PostOpRef(subject, {"i": autobox(index)})), ctx=self._ctx)

    def slice(self, start: "Scalar | Value | object", stop: "Scalar | Value | object") -> "Tuple":
        form = form_of(self)
        start_literal = _literal_number(form_of(autobox(start)))
        stop_literal = _literal_number(form_of(autobox(stop)))
        if (
            isinstance(form, Sequence)
            and not isinstance(form, (str, bytes, bytearray))
            and start_literal is not None
            and stop_literal is not None
        ):
            return Tuple(list(form)[int(start_literal) : int(stop_literal)])
        subject = f"{self._subject()}/slice"
        return Tuple(ref=TCRef(PostOpRef(subject, {"start": autobox(start), "stop": autobox(stop)})), ctx=self._ctx)

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
        left = autobox(other)
        if isinstance(left, Tuple):
            return left.concat(self)
        return Tuple(ref=TCRef(PostOpRef(f"{left._subject()}/concat", {"r": self})))

    def reduce(
        self,
        *,
        op: "OpDef | Scalar | object",
        value: "Scalar | Value | object",
    ) -> Scalar:
        return _reduce_scalar(f"{self._subject()}/reduce", op, value, ctx=self._ctx)


class Map(Comparable):
    def eq(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/eq", {"r": autobox(other)})), ctx=self._ctx)

    def ne(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/ne", {"r": autobox(other)})), ctx=self._ctx)

    def len(self) -> "Number":
        form = form_of(self)
        if isinstance(form, Mapping):
            return Number(len(form))
        subject = f"{self._subject()}/len"
        return Number(ref=TCRef(PostOpRef(subject, {})), ctx=self._ctx)

    def get(self, index: "Scalar | Value | object") -> Scalar:
        form = form_of(self)
        if isinstance(form, Mapping) and isinstance(index, str) and index in form:
            return autobox(form[index])
        subject = f"{self._subject()}/get"
        return Symbol(ref=TCRef(PostOpRef(subject, {"i": autobox(index)})), ctx=self._ctx)

    def __getitem__(self, index: "Scalar | Value | object") -> Scalar:
        return self.get(index)

    def reduce(
        self,
        *,
        op: "OpDef | Scalar | object",
        value: "Scalar | Value | object",
    ) -> Scalar:
        return _reduce_scalar(f"{self._subject()}/reduce", op, value, ctx=self._ctx)


class String(Comparable):
    def eq(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/eq", {"r": autobox(other)})), ctx=self._ctx)

    def ne(self, other: "Scalar | Value | object") -> "Bool":
        return Bool(ref=TCRef(PostOpRef(f"{self._subject()}/ne", {"r": autobox(other)})), ctx=self._ctx)

    def concat(self, other: "Scalar | Value | object") -> "String":
        subject = f"{self._subject()}/concat"
        return String(ref=TCRef(PostOpRef(subject, {"r": autobox(other)})), ctx=self._ctx)

    def _string_render(self, params: Mapping[str, object]) -> "String":
        active_ctx = self._ctx or _context_from_values(*params.values())
        subject = f"{self._subject(ctx=active_ctx)}/render"
        return String(ref=TCRef(PostOpRef(subject, params)), ctx=active_ctx)

    def __add__(self, other: object) -> "String":
        return self.concat(other)

    def __radd__(self, other: object) -> "String":
        left = autobox(other)
        if isinstance(left, String):
            return left.concat(self)
        return String(ref=TCRef(PostOpRef(f"{left._subject()}/concat", {"r": self})))


def _reduce_state_keys(value: "Scalar | Value | object") -> set[str]:
    Value, _, _, ValueMap, _, _, _, value_form_of = _value_runtime()

    if isinstance(value, Scalar):
        value_form = form_of(value)
        if isinstance(value_form, Mapping):
            return set(value_form.keys())
        if isinstance(value_form, ValueMap):
            map_form = value_form_of(value_form)
            if isinstance(map_form, dict):
                return set(map_form.keys())
        return set()

    if isinstance(value, Value):
        if isinstance(value, ValueMap):
            map_form = value_form_of(value)
            if isinstance(map_form, dict):
                return set(map_form.keys())
        return set()

    if isinstance(value, Mapping):
        keys: set[str] = set()
        for key in value.keys():
            if not isinstance(key, str):
                raise TypeError("reduce state map keys must be strings")
            keys.add(key)
        return keys

    return set()


def _infer_reduce_item_name(
    op: "OpDef | Scalar | object",
    value: "Scalar | Value | object",
    *,
    ctx: "Context | None" = None,
) -> str:
    resolved_op = _resolve_reduce_opdef(op, ctx=ctx)
    if resolved_op is None:
        raise TypeError("reduce requires a concrete OpDef to infer item binding")

    state_keys = _reduce_state_keys(value)
    defined_ids = {name for name, _ in resolved_op.form}
    referenced_ids: set[str] = set()
    subject_ids: set[str] = set()
    for _, scalar in resolved_op.form:
        _collect_ref_ids_from_json(scalar.to_json(), referenced_ids, subject_ids)

    subject_candidates = sorted(
        name for name in subject_ids if name not in defined_ids and name not in state_keys
    )
    if len(subject_candidates) == 1:
        return subject_candidates[0]
    if len(subject_candidates) > 1:
        raise TypeError(
            "reduce item binding is ambiguous; reducer references multiple subject inputs: "
            + ", ".join(subject_candidates)
        )

    external_ids = sorted(name for name in referenced_ids if name not in defined_ids)
    candidates = [name for name in external_ids if name not in state_keys]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise TypeError(
            "reduce could not infer item binding: reducer must reference exactly one external input not present in state"
        )

    raise TypeError(
        "reduce item binding is ambiguous; reducer references multiple external inputs: "
        + ", ".join(candidates)
    )


def _resolve_reduce_opdef(op: "OpDef | Scalar | object", *, ctx: "Context | None" = None) -> "OpDef | None":
    if isinstance(op, OpDef):
        return op

    if not isinstance(op, Scalar):
        return None

    op_form = form_of(op)
    if isinstance(op_form, OpDef):
        return op_form

    if not isinstance(op_form, TCRef):
        return None
    op_ref_form = form_of(op_form)
    if not isinstance(op_ref_form, IdRef):
        return None

    active_ctx = ctx
    if active_ctx is None:
        return None

    target = op_ref_form.name
    for name, scalar in active_ctx.form():
        if name != target:
            continue
        scalar_form = form_of(scalar)
        if isinstance(scalar_form, OpDef):
            return scalar_form
        return _resolve_reduce_opdef(scalar, ctx=active_ctx)

    return None


def _collect_ref_ids_from_json(node: object, out: set[str], subject_ids: set[str]) -> None:
    if isinstance(node, list):
        for item in node:
            _collect_ref_ids_from_json(item, out, subject_ids)
        return

    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and key.startswith("$"):
                head, sep, _tail = key[1:].partition("/")
                token = head
                if token:
                    out.add(token)
                    if sep:
                        subject_ids.add(token)
            _collect_ref_ids_from_json(value, out, subject_ids)


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
