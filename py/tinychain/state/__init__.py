from typing import TypeAlias

from .scalar import (
    Scalar,
    autobox,
    after,
    cond,
    id,
    map_of,
    scalar_for_hint,
    tuple_of,
    while_loop,
    for_each,
    form_of,
)
from .scalar.opdef import DeleteOpDef, GetOpDef, OpDef, PostOpDef, PutOpDef
from .scalar.ops import Delete, Get, Op, Post, Put
from .scalar.refs import After, Cond, DeleteOpRef, ForEach, GetOpRef, IdRef, OpRef, PostOpRef, PutOpRef, TCRef, While
from .context import Context, ContextResult, context, scoped_context, current_context
from ._ops import subject_of
from . import collection
from .collection import Collection
from ..collection.tensor import Tensor as _Tensor
from .value import Bool, C64, C128, Complex, F32, F64, Float, I64, Integer, Link, Map, Null, Number, String, Tuple, U64, Value

Numeric: TypeAlias = Scalar | _Tensor

__all__ = [
    "Op",
    "Cond",
    "After",
    "Delete",
    "DeleteOpDef",
    "DeleteOpRef",
    "Get",
    "GetOpDef",
    "GetOpRef",
    "OpRef",
    "IdRef",
    "OpDef",
    "Post",
    "PostOpDef",
    "PostOpRef",
    "Put",
    "PutOpDef",
    "PutOpRef",
    "Scalar",
    "Numeric",
    "TCRef",
    "While",
    "ForEach",
    "Value",
    "Null",
    "Link",
    "Bool",
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
    "autobox",
    "after",
    "cond",
    "Context",
    "ContextResult",
    "context",
    "scoped_context",
    "current_context",
    "subject_of",
    "collection",
    "id",
    "map_of",
    "scalar_for_hint",
    "tuple_of",
    "while_loop",
    "for_each",
    "form_of",
    "Collection",
]
