"""The shared generic reference consumer for the test tree.

Multiple specifications (client issue #128, #129) need a generic backend
proving structural and numerical equivalence over the lowering seam
(`tinychain.autodiff.lower_graph` / `OperationHandlerRegistry`). This module
is the single shared consumer both use, so that exactly one dense-array
execution semantics and exactly one generic reference backend exist in the
test tree. Whichever specification's task lands first creates and owns this
module; later tasks extend it rather than building a second one.

Every handler below delegates to `tests.autodiff_execution.NumpyAutodiffDispatcher`
for its actual dense-array result -- the single source of operator meaning the
node-level executor and this lowering-level registry agree on. A handler here
never restates matmul, mean, broadcast, division, or fill semantics; it only
adapts an `OperationContext` into the shape `NumpyAutodiffDispatcher` expects
(it reads only `.operator` and `.op_params`, so an `OperationContext` -- which
carries both -- can stand in for a `TensorNodeRecord` directly) and, for the
trivial-reshape handler, adds one extra structural check the dispatcher itself
does not perform.

This module imports only public `tinychain.autodiff` names and `numpy`, and
the `tests.autodiff_execution` sibling module it extends with `FillOperator`
support.

`test_autodiff_fake_consumer.py` is a separate, narrow, self-contained proof
of the generic lowering seam and is not related to this module -- it stays
untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tinychain.autodiff import (
    AutodiffError,
    BroadcastOperator,
    DivOperator,
    FillOperator,
    MatmulOperator,
    MeanOperator,
    MulOperator,
    OperationContext,
    OperationHandler,
    OperationHandlerRegistry,
    ReshapeOperator,
    TensorOperator,
)

from tests.autodiff_execution import NumpyAutodiffDispatcher

_dispatcher = NumpyAutodiffDispatcher()


def _delegate(context: OperationContext) -> np.ndarray:
    """Execute *context* through the single dense-array execution semantics.

    `NumpyAutodiffDispatcher.__call__` reads only `node.operator` and
    `node.op_params`, both of which `OperationContext` also carries, so the
    context itself can be handed to the dispatcher directly.
    """
    return _dispatcher(context, list(context.inputs))


class _FillHandler:
    """Lowers a fill node to the constant array `NumpyAutodiffDispatcher` builds."""

    operator_type = FillOperator

    def lower(self, context: OperationContext) -> np.ndarray:
        return _delegate(context)


class _MatmulHandler:
    """Lowers a two-operand matmul, delegating to the shared dense semantics."""

    operator_type = MatmulOperator

    def lower(self, context: OperationContext) -> np.ndarray:
        return _delegate(context)


class _MulHandler:
    """Lowers a mul (two-operand or `right_literal` form) via the shared semantics."""

    operator_type = MulOperator

    def lower(self, context: OperationContext) -> np.ndarray:
        return _delegate(context)


class _MeanHandler:
    """Lowers a mean reduction via the shared semantics."""

    operator_type = MeanOperator

    def lower(self, context: OperationContext) -> np.ndarray:
        return _delegate(context)


class _BroadcastHandler:
    """Lowers a broadcast via the shared semantics."""

    operator_type = BroadcastOperator

    def lower(self, context: OperationContext) -> np.ndarray:
        return _delegate(context)


class _DivHandler:
    """Lowers a division (two-operand or `right_literal` form) via the shared semantics."""

    operator_type = DivOperator

    def lower(self, context: OperationContext) -> np.ndarray:
        return _delegate(context)


class _TrivialReshapeHandler:
    """Lowers a reshape that preserves element count; rejects one that does not.

    This is the one handler in the module that adds a check beyond the shared
    dense-array semantics: `NumpyAutodiffDispatcher`'s own `np.reshape` call
    would raise a bare `ValueError` for a target shape with a different
    element count, so this handler validates the element count itself and
    reports a categorized failure before ever reaching `np.reshape`.
    """

    operator_type = ReshapeOperator

    def lower(self, context: OperationContext) -> np.ndarray:
        source = np.asarray(context.inputs[0])
        target_shape = tuple(int(dimension) for dimension in context.op_params["shape"])
        target_size = int(np.prod(target_shape, dtype=np.int64)) if target_shape else 1
        if source.size != target_size:
            raise AutodiffError(
                "shape_mismatch",
                f"trivial reshape handler rejects node {context.node_id!r}: "
                f"input has {source.size} elements, target shape {target_shape!r} "
                f"has {target_size}",
            )
        return _delegate(context)


def limited_operation_registry(*, include_trivial_reshape: bool = False) -> OperationHandlerRegistry:
    """Return a registry resolving exactly `FillOperator`, `MatmulOperator`, `MulOperator`.

    `ReshapeOperator`, `MeanOperator`, `BroadcastOperator`, and `DivOperator`
    lookups fail with `unsupported_operator`, proving the rank-preserving tier
    can be exercised without the trivial-reshape handler.

    Passing `include_trivial_reshape=True` additionally registers the
    element-count-checking trivial-reshape handler -- an explicit opt-in
    rather than a default, so a caller must ask for reshape support.
    """
    registry = OperationHandlerRegistry()
    registry.register(_FillHandler())
    registry.register(_MatmulHandler())
    registry.register(_MulHandler())
    if include_trivial_reshape:
        registry.register(_TrivialReshapeHandler())
    return registry


def reduction_capable_registry() -> OperationHandlerRegistry:
    """Return a registry additionally resolving `MeanOperator`, `BroadcastOperator`,
    and `DivOperator`, all through the same dense semantics `NumpyAutodiffDispatcher`
    uses -- so this registry can serve as the numerical control for an equivalent
    fill/matmul/mul-expanded artifact.
    """
    registry = limited_operation_registry()
    registry.register(_MeanHandler())
    registry.register(_BroadcastHandler())
    registry.register(_DivHandler())
    return registry


@dataclass(frozen=True)
class RecordedInvocation:
    """One handler call captured by `recording_registry`, in call order."""

    node_id: str
    operator_type: type[TensorOperator]


class _RecordingHandler:
    """Wraps one `OperationHandler`, recording an invocation before delegating.

    The record is appended inside `lower`, at the point the wrapped handler is
    actually called -- not at registration or lookup time -- so a lowering
    that fails closed before any handler runs (`OperationHandlerRegistry.lookup`
    raising `unsupported_operator` during the pre-flight support check) leaves
    the record empty, which is exactly the proof a fail-closed test needs.
    """

    def __init__(self, wrapped: OperationHandler, invocations: list[RecordedInvocation]) -> None:
        self.operator_type = wrapped.operator_type
        self._wrapped = wrapped
        self._invocations = invocations

    def lower(self, context: OperationContext) -> object:
        self._invocations.append(
            RecordedInvocation(node_id=context.node_id, operator_type=type(context.operator))
        )
        return self._wrapped.lower(context)


@dataclass
class RecordingRegistry:
    """An `OperationHandlerRegistry` wrapper exposing every handler call in order."""

    registry: OperationHandlerRegistry
    invocations: list[RecordedInvocation] = field(default_factory=list)


def recording_registry(base: OperationHandlerRegistry) -> RecordingRegistry:
    """Wrap every handler in *base* with an invocation recorder.

    The returned `.registry` behaves exactly like *base* for lowering
    purposes; `.invocations` accumulates a `RecordedInvocation` each time a
    wrapped handler's `lower` actually runs, in call order.
    """
    invocations: list[RecordedInvocation] = []
    wrapped_registry = OperationHandlerRegistry()
    for operator_type in base.supported_types():
        handler = base.lookup(operator_type())
        wrapped_registry.register(_RecordingHandler(handler, invocations))
    return RecordingRegistry(registry=wrapped_registry, invocations=invocations)
