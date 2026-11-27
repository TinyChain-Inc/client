"""The one Python-to-native normalization boundary; wire grammar lives in Rust."""

from __future__ import annotations

from tinychain_local import Codec, ScalarHandle


def lower(obj: object, *, _depth: int = 0) -> ScalarHandle:
    from .opref import OpRef as RuntimeOpRef
    from .classdef import Class
    from .state.base import State
    from .state.scalar import OpDef, Scalar, TCRef, form_of
    from .state.scalar.refs import (
        After, Cond, DeleteOpRef, ForEach, GetOpRef, IdRef, PostOpRef,
        PutOpRef, While,
    )
    from .state.value import Link, Map, Number, Tuple, Value
    from .state.value import form_of as value_form
    from .uri import URI

    if _depth > 16:
        raise ValueError("native Scalar nesting exceeds 16")

    def child(value: object) -> ScalarHandle:
        return lower(value, _depth=_depth + 1)

    if isinstance(obj, ScalarHandle):
        return obj
    if isinstance(obj, RuntimeOpRef):
        raise TypeError("runtime requests must be lowered by the Library compiler")
    if isinstance(obj, IdRef):
        return ScalarHandle.idref(obj.key())
    if isinstance(obj, GetOpRef):
        return ScalarHandle.get(obj.subject, child(obj._key))
    if isinstance(obj, PutOpRef):
        return ScalarHandle.put(obj.subject, child(obj._key), child(obj._value))
    if isinstance(obj, PostOpRef):
        return ScalarHandle.post(obj.subject, [(k, child(v)) for k, v in obj._params.items()])
    if isinstance(obj, DeleteOpRef):
        return ScalarHandle.delete(obj.subject, child(obj._key))
    if isinstance(obj, Cond):
        return ScalarHandle.cond(child(obj.cond), child(obj.then), child(obj.or_else))
    if isinstance(obj, After):
        return ScalarHandle.after(child(obj.when), child(obj.then))
    if isinstance(obj, While):
        return ScalarHandle.while_loop(child(obj.cond), child(obj.op), child(obj.state))
    if isinstance(obj, ForEach):
        return ScalarHandle.for_each(child(obj.items), child(obj.op), obj.item_name)
    if isinstance(obj, OpDef):
        return ScalarHandle.opdef(obj.method, list(obj._parameters()), [(k, child(v)) for k, v in obj.form])
    if isinstance(obj, Class):
        return lower(obj._form, _depth=_depth)
    if isinstance(obj, Value):
        op = getattr(obj, "op", None)
        if op is not None:
            return lower(op, _depth=_depth)
        raw = value_form(obj)
        if isinstance(obj, Link):
            return ScalarHandle.link(raw)
        if isinstance(obj, Map):
            return ScalarHandle.map([(k, child(v)) for k, v in raw.items()])
        if isinstance(obj, Tuple):
            return ScalarHandle.value_tuple([child(v) for v in raw])
        if isinstance(obj, Number):
            return ScalarHandle.number(raw, str(URI(type(obj))))
        return ScalarHandle.literal(raw)
    if isinstance(obj, (Scalar, TCRef)):
        raw = form_of(obj)
        if raw is obj:
            raise TypeError(f"unsupported canonical reference {type(obj).__name__}")
        return lower(raw, _depth=_depth)
    if isinstance(obj, State):
        return lower(obj._form, _depth=_depth)
    if isinstance(obj, URI):
        return ScalarHandle.link(str(obj))
    if isinstance(obj, dict):
        return ScalarHandle.map([(k, child(v)) for k, v in obj.items()])
    if isinstance(obj, (list, tuple)):
        return ScalarHandle.tuple([child(v) for v in obj])
    return ScalarHandle.literal(obj)


def to_wire(value: object) -> object:
    return Codec().to_wire(lower(value))


def value_to_wire(value: object) -> object:
    return Codec().to_value_wire(lower(value))


def wire_leaf(value: object) -> object:
    """Normalize one authoring leaf during the bounded native wire traversal."""
    from .state.base import State
    from .state.scalar import OpDef
    from .uri import URI

    if isinstance(value, (State, OpDef, URI)):
        from .state.collection import Collection
        if isinstance(value, Collection):
            return value._form
        return lower(value)
    if isinstance(value, (complex, bytes)):
        return lower(value)
    raise TypeError(f"cannot encode form {type(value).__name__}")


def encode(value: object, codec: Codec | None = None) -> bytes:
    return bytes((codec or Codec()).encode_wire(value))


def local_input(value: object, *, _depth: int = 0) -> object:
    """Construct State composition, preserving collection and native leaves."""
    from tinychain_local import StateHandle

    from .state.collection import Collection

    if _depth > 16:
        raise ValueError("State composition exceeds nesting limit")
    if isinstance(value, (Collection, StateHandle)):
        return value
    if isinstance(value, dict):
        return {k: local_input(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [local_input(v, _depth=_depth + 1) for v in value]
    return lower(value, _depth=_depth)
