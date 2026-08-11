from __future__ import annotations

from typing import Any, Iterator, Mapping, Sequence, cast

from ...uri import URI
from ..base import State
from ..collection import Collection
from .opdef import (
    OpDef,
)


def _sorted_items(obj: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return sorted(obj.items(), key=lambda kv: kv[0])


def _json_of(form: object) -> object:
    from ..value import Value

    runtime_op = OpRef.from_runtime(form)
    if runtime_op is not None:
        return runtime_op.to_json()

    if isinstance(form, (OpRef, OpDef, After, Cond, While, ForEach)):
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


def autobox(obj: object) -> State:
    from ..value import Link as value_link
    from ..value import Value

    if isinstance(obj, Value):
        from ...opref import OpRef as RuntimeOpRef

        op = getattr(obj, "op", None)
        if isinstance(op, OpRef):
            return _scalar_like(obj, form=op)
        runtime_op = OpRef.from_runtime(op)
        if runtime_op is not None:
            return _scalar_like(obj, form=runtime_op)
        if isinstance(op, RuntimeOpRef):
            raise TypeError(f"unsupported runtime OpRef type {type(op).__name__}")
        return _scalar_like(obj, value=obj)

    runtime_op = OpRef.from_runtime(obj)
    if runtime_op is not None:
        return _typed_from_op_ref(runtime_op)
    tensor = _tensor_from_native(obj)
    if tensor is not None:
        return tensor

    if isinstance(obj, URI):
        return Scalar(value_link(obj))
    if isinstance(obj, OpRef):
        return _typed_from_op_ref(obj)
    if isinstance(obj, After):
        return _typed_from_after(obj)
    if isinstance(obj, Cond):
        return _typed_from_cond(obj)
    if isinstance(obj, While):
        ref = obj
        state = autobox(obj.state)
        return _typed_from_ref_like(ref, state)
    if isinstance(obj, ForEach):
        return Scalar(obj)
    if isinstance(obj, IdRef):
        return Symbol(obj)
    if isinstance(obj, TCRef):
        return _typed_from_tcref(obj)
    if isinstance(obj, Scalar):
        return obj
    if isinstance(obj, OpDef):
        return Scalar(obj)
    if isinstance(obj, State):
        return obj
    if isinstance(obj, dict):
        return map_of({k: autobox(v) for k, v in _sorted_items(obj)})
    if isinstance(obj, (list, tuple)):
        return tuple_of([autobox(v) for v in obj])

    value_obj = Value.from_json(obj)
    return _scalar_like(value_obj, value=value_obj)


def _tensor_from_native(obj: object) -> "State | None":
    # Treat numpy-like dense arrays as Tensor-native inputs.
    module = type(obj).__module__
    if not (module == "numpy" or module.startswith("numpy.")):
        return None

    if not hasattr(obj, "shape") or not hasattr(obj, "dtype"):
        return None

    from ...collection.tensor import Tensor

    return Tensor(native=_NumpyTensorNative(obj))


class _NumpyTensorNative:
    __slots__ = ("_array",)

    def __init__(self, array: object):
        self._array = array

    @property
    def dtype(self) -> object:
        return getattr(self._array, "dtype")

    @property
    def shape(self) -> object:
        return getattr(self._array, "shape")

    @property
    def values(self) -> object:
        ravel = getattr(self._array, "ravel", None)
        if callable(ravel):
            flat = ravel()
            tolist = getattr(flat, "tolist", None)
            if callable(tolist):
                return tolist()

        tolist = getattr(self._array, "tolist", None)
        if callable(tolist):
            return _flatten_dense_values(tolist())

        raise TypeError("numpy-like tensor value must expose ravel/tolist")


def _flatten_dense_values(value: object) -> list[object]:
    if isinstance(value, list):
        flat: list[object] = []
        for item in value:
            flat.extend(_flatten_dense_values(item))
        return flat

    return [value]


def _is_string_scalar(obj: object) -> bool:
    from ..value import String as value_string

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


def _literal_number(form: object) -> int | float | bool | None:
    if isinstance(form, (int, float, bool)):
        return form

    from ..value import Bool as value_bool
    from ..value import Number as value_number
    if isinstance(form, (value_number, value_bool)):
        try:
            json_value = form.to_json()
        except (TypeError, ValueError, AttributeError):
            return None
        if isinstance(json_value, (int, float, bool)):
            return json_value

    return None


def id(name: str) -> "Scalar":
    from ..context import current_context

    active_ctx = current_context()
    if active_ctx is not None:
        try:
            return getattr(active_ctx, name)
        except AttributeError:
            pass

    # Unbound ids are represented as a generic symbolic ref.
    return Symbol(IdRef(name))


def map_of(items: Mapping[str, "Scalar | Value | object"]) -> "Scalar":
    boxed = {key: autobox(value) for key, value in _sorted_items(items)}
    return Map(boxed)


def tuple_of(items: Sequence["Scalar | Value | object"]) -> "Scalar":
    boxed = [autobox(item) for item in items]
    return Tuple(boxed)


def scalar_for_hint(name: str, hint: object) -> "Scalar":
    base = IdRef(name)
    cls = _scalar_class_for_hint(hint)
    return cls(base)


def _scalar_class_for_hint(hint: object) -> type["Scalar"]:
    from ..value import Bool as value_bool
    from ..value import Map as value_map
    from ..value import Number as value_number
    from ..value import String as value_string
    from ..value import Tuple as value_tuple
    from ..value import Value

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
    form: TCRef | None = None,
    op: "OpDef | None" = None,
    map: Mapping[str, "Scalar"] | None = None,
    tuple: Sequence["Scalar"] | None = None,
) -> "Scalar":
    from ..value import Bool as value_bool
    from ..value import Map as value_map
    from ..value import Number as value_number
    from ..value import String as value_string
    from ..value import Tuple as value_tuple

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
        return scalar
    if scalar_type is Tuple and tuple is not None:
        scalar = Tuple(list(tuple))
        return scalar

    if form is not None:
        scalar = scalar_type(form)
        return scalar
    if op is not None:
        scalar = scalar_type(op)
        return scalar

    if scalar_type is Map and map is not None:
        scalar = Map(dict(map))
        return scalar
    if scalar_type is Tuple and tuple is not None:
        scalar = Tuple(list(tuple))
        return scalar

    if value is not None:
        scalar = scalar_type(value)
        return scalar

    scalar = scalar_type()
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
    exemplar_form = form_of(exemplar)
    if isinstance(exemplar_form, OpDef):
        return Iterable(ref)
    if isinstance(exemplar, Number):
        return Number(ref)
    if isinstance(exemplar, Bool):
        return Bool(ref)
    if isinstance(exemplar, String):
        return String(ref)
    if isinstance(exemplar, Map):
        if isinstance(exemplar_form, Mapping):
            return Map(dict(exemplar_form))
        return Map(ref)
    if isinstance(exemplar, Tuple):
        if isinstance(exemplar_form, Sequence) and not isinstance(exemplar_form, (str, bytes, bytearray)):
            return Tuple(list(exemplar_form))
        return Tuple(ref)
    if isinstance(exemplar, Numeric):
        return Symbol(ref)
    if isinstance(exemplar, Iterable):
        return Iterable(ref)
    if isinstance(exemplar, Comparable):
        return Comparable(ref)
    return Symbol(ref)


def _typed_from_cond(cond_ref: Cond) -> Scalar:
    then_value = autobox(cond_ref.then)
    else_value = autobox(cond_ref.or_else)
    ref = cond_ref

    if type(then_value) is type(else_value):
        if isinstance(then_value, Map) and isinstance(else_value, Map):
            then_form = form_of(then_value)
            else_form = form_of(else_value)
            merged = _merge_map_shape(
                then_form if isinstance(then_form, Mapping) else None,
                else_form if isinstance(else_form, Mapping) else None,
            )
            if merged is not None:
                return Map(dict(merged))
            return Map(ref)
        if isinstance(then_value, Tuple) and isinstance(else_value, Tuple):
            then_form = form_of(then_value)
            else_form = form_of(else_value)
            merged = _merge_tuple_shape(
                then_form if isinstance(then_form, Sequence) and not isinstance(then_form, (str, bytes, bytearray)) else None,
                else_form if isinstance(else_form, Sequence) and not isinstance(else_form, (str, bytes, bytearray)) else None,
            )
            if merged is not None:
                return Tuple(list(merged))
            return Tuple(ref)
        return _typed_from_ref_like(ref, then_value)

    return Iterable(ref)


def _typed_from_after(after_ref: After) -> State:
    then = autobox(after_ref.then)
    return type(then)(after_ref)


def _typed_from_op_ref(op_ref: OpRef) -> Scalar:
    subject = op_ref.subject
    if not isinstance(subject, str):
        return Symbol(op_ref)
    ref = op_ref

    exact_dispatch: dict[str, type[Scalar]] = {
        str(URI(Scalar, "reflect", "ref_parts")): Tuple,
        str(URI(OpDef, "reflect", "form")): Tuple,
        str(URI(OpDef, "reflect", "scalars")): Tuple,
        str(URI(OpDef, "reflect", "last_id")): String,
    }
    wrapper = exact_dispatch.get(subject)
    if wrapper is not None:
        return wrapper(ref)

    return Symbol(op_ref)


def _typed_from_tcref(ref: TCRef) -> Scalar:
    ref_form = form_of(ref)
    if isinstance(ref_form, OpRef):
        return _typed_from_op_ref(ref_form)
    if isinstance(ref_form, Cond):
        return _typed_from_cond(ref_form)
    if isinstance(ref_form, After):
        return _typed_from_after(ref_form)
    if isinstance(ref_form, While):
        state = autobox(ref_form.state)
        return _typed_from_ref_like(ref, state)
    return Symbol(ref)


def while_loop(
    cond: "Scalar | Value | object",
    op: "Scalar | Value | object",
    state: "Scalar | Value | object",
) -> "Scalar":
    while_ref = While(
        autobox(cond),
        autobox(op),
        autobox(state),
    )
    result = autobox(
        while_ref
    )
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
    cond_node = Cond(
        cond_ref,
        autobox(then),
        autobox(or_else),
    )
    result = autobox(
        cond_node
    )
    return result


def after(
    dependency: "Scalar | Value | object",
    then: "Scalar | Value | object",
) -> "State":
    after_ref = After(autobox(dependency), autobox(then))
    return _typed_from_after(after_ref)


def for_each(
    items: "Scalar | Value | object",
    *,
    item_name: str,
    op: "OpDef",
) -> "Scalar":
    foreach_ref = ForEach(autobox(items), autobox(op), item_name)
    result = autobox(foreach_ref)
    return result


def form_of(value: "Scalar | object") -> object:
    from ..value import Value
    from ..value import form_of as value_form_of

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

    __uri__: URI = URI(State, "scalar")

    def __init__(self, form: object = None):
        super().__init__(form)

    def to_json(self) -> object:
        return _json_of(form_of(self))

    def _reflect(self, subject: str, payload_key: str, payload_value: object, *, rtype: type["Scalar"]) -> "Scalar":
        return rtype._post_ref(subject, {payload_key: payload_value})

    def class_(self) -> "Scalar":
        return self._reflect(str(URI(Scalar, "reflect", "class")), "scalar", self, rtype=Scalar)

    def ref_parts(self) -> "Tuple":
        return cast(Tuple, self._reflect(str(URI(Scalar, "reflect", "ref_parts")), "scalar", self, rtype=Tuple))

    def reflect_form(self) -> "Tuple":
        return cast(Tuple, self._reflect(str(URI(OpDef, "reflect", "form")), "op", self, rtype=Tuple))

    def reflect_last_id(self) -> "String":
        return cast(String, self._reflect(str(URI(OpDef, "reflect", "last_id")), "op", self, rtype=String))

    def reflect_scalars(self) -> "Tuple":
        return cast(Tuple, self._reflect(str(URI(OpDef, "reflect", "scalars")), "op", self, rtype=Tuple))

    @staticmethod
    def from_json(obj: Any) -> "Scalar":
        from ..value import Value

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
                if isinstance(key, str) and key.startswith(str(URI(OpDef))):
                    return Scalar(OpDef.from_json(obj))

            # Decode TCRef/OpRef maps before generic Value maps to avoid
            # treating single-entry refs (e.g. {"$id": []}) as plain maps.
            if len(obj) == 1:
                if _looks_like_tcref_map(obj):
                    return Scalar(TCRef.from_json(obj))

            # Try to decode as a Value-typed map.
            try:
                value = Value.from_json(obj)
                return _scalar_like(value, value=value)
            except (TypeError, ValueError):
                pass

            return map_of({k: Scalar.from_json(v) for k, v in _sorted_items(obj)})

        raise TypeError(f"cannot decode Scalar from {type(obj).__name__}")


# Import ref types after Scalar is defined so TCRef can subclass Scalar without a cycle.
from .refs import (
    After,
    Cond,
    ForEach,
    IdRef,
    OpRef,
    PostOpRef,
    TCRef,
    While,
    _looks_like_tcref_map,
    tcref_form_of,
)


def _post_ref_call(
    owner: Scalar,
    method: str,
    params: Mapping[str, object],
    *,
    rtype: type[Scalar],
) -> Scalar:
    subject = _subject_method(owner, method)
    return rtype(PostOpRef(subject, params))


def _post_binary_call(owner: Scalar, method: str, other: "Scalar | Value | object", *, rtype: type[Scalar]) -> Scalar:
    return _post_ref_call(owner, method, {"r": autobox(other)}, rtype=rtype)


def _post_unary_call(owner: Scalar, method: str, *, rtype: type[Scalar]) -> Scalar:
    return _post_ref_call(owner, method, {}, rtype=rtype)


def _subject_of(owner: Scalar) -> str:
    from .._ops import subject_of

    return subject_of(owner)


def _subject_method(owner: Scalar, method: str) -> str:
    return f"{_subject_of(owner)}/{method}"


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
    return rtype(PostOpRef(f"{_subject_of(left)}/{method}", {"r": owner}))


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
        return Numeric(PostOpRef(f"{_subject_of(self)}/add", {"r": autobox(other)}))

    def __add__(self, other: object) -> "Numeric":
        return self.add(other)

    def __radd__(self, other: object) -> "Numeric":
        return cast(Numeric, _reverse_post_call(self, other, method="add", rtype=Numeric, chain_type=Numeric))


class Iterable(Comparable):
    def len(self) -> "Number":
        subject = f"{_subject_of(self)}/len"
        return Number(PostOpRef(subject, {}))

    def get(self, index: "Scalar | Value | object") -> "Scalar":
        subject = f"{_subject_of(self)}/get"
        return Symbol(PostOpRef(subject, {"i": autobox(index)}))

    def __getitem__(self, index: "Scalar | Value | object") -> "Scalar":
        if isinstance(index, slice):
            if index.step is not None:
                raise NotImplementedError(f"slice with step: {index}")
            start = 0 if index.start is None else index.start
            stop = self.len() if index.stop is None else index.stop
            subject = f"{_subject_of(self)}/slice"
            return Iterable(PostOpRef(subject, {"start": autobox(start), "stop": autobox(stop)}))
        return self.get(index)

    def concat(self, other: "Scalar | Value | object") -> "Iterable":
        return Iterable(PostOpRef(f"{_subject_of(self)}/concat", {"r": autobox(other)}))

    def __add__(self, other: object) -> "Iterable":
        right = autobox(other)
        if _is_concat_operand(right, other):
            return self.concat(right)
        return Iterable(PostOpRef(f"{_subject_of(self)}/add", {"r": right}))

    def __radd__(self, other: object) -> "Iterable":
        left = autobox(other)
        method = "add"
        if _is_concat_operand(left, other):
            method = "concat"
        return cast(Iterable, _reverse_post_call(self, other, method=method, rtype=Iterable))


class Symbol(Numeric, Iterable):
    def _string_render(self, params: Mapping[str, object]) -> "String":
        return String(PostOpRef(f"{_subject_of(self)}/render", params))

    def __add__(self, other: object) -> "Symbol":
        right = autobox(other)
        if _is_concat_operand(right, other):
            return Symbol(PostOpRef(f"{_subject_of(self)}/concat", {"r": right}))
        return Symbol(PostOpRef(f"{_subject_of(self)}/add", {"r": right}))

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
        return Bool(PostOpRef(f"{_subject_of(self)}/{method}", {"r": right}))

    def _arithmetic(self, method: str, other: "Scalar | Value | object", *, literal_op) -> "Number":
        right = autobox(other)
        left_form = form_of(self)
        right_form = form_of(right)
        left_literal = _literal_number(left_form)
        right_literal = _literal_number(right_form)
        if left_literal is not None and right_literal is not None:
            return Number(literal_op(left_literal, right_literal))
        return Number(PostOpRef(f"{_subject_of(self)}/{method}", {"r": right}))

    def _reverse_arithmetic(self, method: str, other: object) -> "Number":
        left = autobox(other)
        if isinstance(left, Number):
            return getattr(left, method)(self)
        return Number(PostOpRef(f"{_subject_of(left)}/{method}", {"r": self}))

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
) -> Scalar:
    from .reduce import infer_reduce_item_name

    item_name = infer_reduce_item_name(op, value)
    opref = PostOpRef(
        subject,
        {
            "item_name": item_name,
            "op": op,
            "value": autobox(value),
        },
    )
    return Scalar(opref)


class Tuple(Iterable):
    def len(self) -> "Number":
        form = _seq_form_or_none(form_of(self))
        if form is not None:
            return Number(len(form))
        subject = f"{_subject_of(self)}/len"
        return Number(PostOpRef(subject, {}))

    def head(self) -> Scalar:
        subject = f"{_subject_of(self)}/head"
        return Symbol(PostOpRef(subject, {}))

    def tail(self) -> "Tuple":
        subject = f"{_subject_of(self)}/tail"
        return Tuple(PostOpRef(subject, {}))

    def concat(self, other: "Scalar | Value | object") -> "Tuple":
        form = _seq_form_or_none(form_of(self))
        right = autobox(other)
        right_form = _seq_form_or_none(form_of(right))
        if form is not None and right_form is not None:
            return Tuple(list(form) + list(right_form))
        subject = f"{_subject_of(self)}/concat"
        return Tuple(PostOpRef(subject, {"r": right}))

    def get(self, index: "Scalar | Value | object") -> Scalar:
        form = _seq_form_or_none(form_of(self))
        if form is not None and isinstance(index, int):
            return autobox(form[index])
        subject = f"{_subject_of(self)}/get"
        return Symbol(PostOpRef(subject, {"i": autobox(index)}))

    def slice(self, start: "Scalar | Value | object", stop: "Scalar | Value | object") -> "Tuple":
        form = _seq_form_or_none(form_of(self))
        start_literal = _literal_number(form_of(autobox(start)))
        stop_literal = _literal_number(form_of(autobox(stop)))
        if form is not None and start_literal is not None and stop_literal is not None:
            return Tuple(list(form)[int(start_literal) : int(stop_literal)])
        subject = f"{_subject_of(self)}/slice"
        return Tuple(PostOpRef(subject, {"start": autobox(start), "stop": autobox(stop)}))

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
        return _reduce_scalar(_subject_method(self, "reduce"), op, value)


class Map(Comparable):
    def len(self) -> "Number":
        form = _map_form_or_none(form_of(self))
        if form is not None:
            return Number(len(form))
        subject = f"{_subject_of(self)}/len"
        return Number(PostOpRef(subject, {}))

    def get(self, index: "Scalar | Value | object") -> Scalar:
        form = _map_form_or_none(form_of(self))
        if form is not None and isinstance(index, str) and index in form:
            return autobox(form[index])
        subject = f"{_subject_of(self)}/get"
        return Symbol(PostOpRef(subject, {"i": autobox(index)}))

    def __getitem__(self, index: "Scalar | Value | object") -> Scalar:
        return self.get(index)

    def reduce(
        self,
        *,
        op: "OpDef | Scalar | object",
        value: "Scalar | Value | object",
    ) -> Scalar:
        return _reduce_scalar(_subject_method(self, "reduce"), op, value)


class String(Comparable):
    def concat(self, other: "Scalar | Value | object") -> "String":
        subject = f"{_subject_of(self)}/concat"
        return String(PostOpRef(subject, {"r": autobox(other)}))

    def _string_render(self, params: Mapping[str, object]) -> "String":
        subject = f"{_subject_of(self)}/render"
        return String(PostOpRef(subject, params))

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
