from __future__ import annotations

from .autodiff.callsite import grad
from .library import Library, delete, get, install, post, put
from .classdef import Class, ClassError, InvalidClassParent, MissingClassMember, UnsupportedClassOverride, class_definition, validate_class_definition
from .codec import decode_response_body
from .executor import Executor, backend
from .executor import execute as _dispatch_execute
from .opref import OpRef
from . import opref
from .ref import Ref
from .collection import Tensor, concatenate, einsum, split, tile
from .state.value import Bool, C64, C128, Complex, F32, F64, Float, I64, Integer, Link, Map, Null, Number, String, Tuple, U64
from .uri import URI, authority, origin, uri
from . import compute
from . import collection
from . import state
from . import kernel
from . import auth
from . import std
from .cond import cond
from .state import after
from .host import Host

__all__ = [
    "Library",
    "Class",
    "ClassError",
    "InvalidClassParent",
    "MissingClassMember",
    "UnsupportedClassOverride",
    "class_definition",
    "validate_class_definition",
    "Executor",
    "backend",
    "OpRef",
    "opref",
    "Ref",
    "Bool",
    "Null",
    "Link",
    "Number",
    "Integer",
    "I64",
    "U64",
    "Float",
    "F32",
    "F64",
    "Complex",
    "C64",
    "C128",
    "Map",
    "Tuple",
    "String",
    "Tensor",
    "concatenate",
    "einsum",
    "split",
    "tile",
    "URI",
    "compute",
    "collection",
    "state",
    "kernel",
    "install",
    "std",
    "get",
    "grad",
    "put",
    "post",
    "delete",
    "cond",
    "after",
    "uri",
    "authority",
    "origin",
    "Host",
    "auth",
]

globals().pop("testing", None)
globals().pop("wasm", None)


def execute(op: "OpRef | Ref") -> object:
    if hasattr(op, "op"):
        op = op.op
    elif hasattr(op, "_form"):
        form = getattr(op, "_form")
        if isinstance(form, (OpRef, Ref)):
            op = form
    if not isinstance(op, (OpRef, Ref)):
        raise TypeError(f"expected OpRef or Ref, got {type(op).__name__}")

    response = _dispatch_execute(op)
    status = getattr(response, "status", None)
    if status is None:
        # HTTP-host execution paths may already return decoded JSON-like values.
        return response
    if status == 200:
        return decode_response_body(response)
    if status == 204:
        return None
    message = None
    try:
        body = getattr(response, "body", None)
        if body is not None:
            value = body.value()
            text = value.to_json() if hasattr(value, "to_json") else value
            if isinstance(text, (bytes, bytearray)):
                text = text.decode("utf-8", errors="replace")
            message = text if isinstance(text, str) else str(text)
    except (AttributeError, TypeError, ValueError):
        message = None
    if message:
        raise AssertionError(f"unexpected status {status}: {message}")
    raise AssertionError(f"unexpected status {status}")
