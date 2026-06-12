from __future__ import annotations

from abc import ABC

import tinychain as tc


class Operator(ABC):
    """Autodiff operator descriptor.

    This is a transform-time abstraction used by the runtime autodiff library
    to map forward graph structure to derivative rules. It is not part of
    ordinary runtime execution payloads.

    Subclass guidance:
    1. Define concrete typed fields on each subclass (no generic metadata bags).
    2. Keep operator model concerns local to autodiff routes.
    3. Do not require operator objects in standard execution payloads.
    """

    __slots__ = ()

    def __init__(self):
        if type(self) is Operator:
            raise TypeError("Operator is abstract and must be subclassed")


class DualOperator(Operator):
    """Minimal binary operator example for autodiff rule authoring.

    Developers should define additional Operator subclasses following this
    typed-field pattern as autodiff rule coverage expands.
    """

    __slots__ = ("left", "right")

    def __init__(self, *, left: tc.state.Numeric, right: tc.state.Numeric):
        self.left = left
        self.right = right


class Autodiff(tc.Library):
    """Runtime autodiff library stub.

    This is a fill-in skeleton for runtime autodiff routes. The default
    implementations are no-op placeholders so the manifest surface can be
     installed and iterated independently of client-side compiler logic.

     Operator handling guidance:
     1. Use `Operator` subclasses in this module to encode transform-time rule
         semantics with explicit typed fields per operator form.
     2. Keep operator objects local to autodiff transform routes (`trace`, `grad`,
         `vjp`) and never require them in ordinary execution payloads.
     3. Runtime execution remains route+params driven; operator classes are
         an autodiff planning concern, not a general execution envelope.
    """

    publisher = "std"
    version = "0.1.0"

    @tc.post
    def trace(self, op: tc.state.OpDef) -> tc.state.OpDef:
        # Placeholder: return normalized forward graph once implemented.
        return op

    @tc.post
    def grad(self, target: tc.state.Scalar, wrt: tc.Tuple) -> tc.state.OpDef:
        # Placeholder: return a gradient OpDef transform for `target` and `wrt`.
        return tc.state.PostOpDef([("result", target)])

    @tc.post
    def vjp(self, target: tc.state.Scalar, wrt: tc.Tuple, cotangent: tc.state.Scalar) -> tc.state.OpDef:
        # Placeholder: return vector-Jacobian product OpDef once implemented.
        return tc.state.PostOpDef([("result", cotangent)])


__all__ = [
    "Operator",
    "DualOperator",
    "Autodiff",
]
