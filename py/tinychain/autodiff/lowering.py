"""Framework-owned program construction over consumer-supplied lowering handlers.

A consumer that wants to turn a traced forward graph or a generated derivative
program into its own target representation needs the graph walked, sliced, and
validated first. Doing that in the consumer makes a specific application
responsible for reachability, topological order, dependency binding, and output
selection -- generic mechanics that belong to the framework. This module owns
them and leaves the consumer exactly two decisions: how one concrete operation
becomes a target value, and which local patterns are worth fusing.

The seam is deliberately opaque in both directions. The framework never imports,
inspects, compares, hashes, or iterates a target value; it only carries it from
the handler that produced it to the handlers that consume it and, finally, to
the selected outputs. The consumer never receives a traversal responsibility: it
sees one :class:`OperationContext` at a time, with its operands already lowered.

Operations are identified by concrete :class:`TensorOperator` type, never by
route name. Two operators that spell ``route_name`` identically are two
operations, and a handler registered for one is not reached by the other.

Exactly-once lowering
---------------------
Every reachable operation is lowered by exactly one handler, replaced by exactly
one fusion, or reported unsupported. That invariant is not spread over the
dispatch path, the fusion hook, and output selection; enforcing it per step
would leave each pair of steps as its own possible gap. Instead:

* :func:`_claim` is the single writer. It is the only way an operation becomes
  lowered and the only way a produced value enters the value environment, and it
  rejects any operation that already carries a claim. The single-handler path
  and the fusion path both go through it, so a doubly-lowered operation fails
  identically whichever route reached it.
* :func:`_require_every_operation_claimed` closes the other half. A per-operation
  guard can only ever observe double-claiming; one set comparison over the whole
  reachable region is what makes "not skipped" and "not repeated" a single
  enforced property rather than two half-checks.
* Output selection reads the value environment and nothing else, so it cannot
  introduce, substitute, or re-lower an operation.

Fusion legality -- that a fused region discards no value still needed outside it
-- is a *different* invariant, checked separately and named separately. Folding
it into the exactly-once check would produce one guard that appears to cover both
and covers neither.

Failures are categorized :class:`AutodiffError` values raised before any target
program is handed back: ``unsupported_operator`` for an operation no handler
claims, ``handler_contract_violation`` for a consumer that breaks the seam
contract, and the dependency-analysis categories (``missing_dependency``,
``ambiguous_producer``, ``invalid_selected_output``, ``malformed_derivative_ir``,
``missing_dtype_metadata``, ``missing_shape_metadata``) for a malformed selection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from .dependencies import (
    DEPENDENCY_PROVENANCE_LOCAL_VALUE,
    DependencyAnalysis,
    ValueDependency,
    analyze_derivative_dependencies,
    analyze_graph_dependencies,
)
from .graph import TensorGraph, TensorNodeRecord, TensorOperator
from .protocol import AutodiffError

if TYPE_CHECKING:
    from .reverse import DerivativeProgram


#: An operation lowered by the handler registered for its concrete type.
LOWERING_CLAIM_HANDLER = "handler"

#: An operation replaced, together with its fused neighbours, by one instruction.
LOWERING_CLAIM_FUSION = "fusion"

#: Exceptions a consumer callback may leak that become a contract violation.
_CONSUMER_FAILURES = (
    AssertionError,
    AttributeError,
    IndexError,
    KeyError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class OperationContext:
    """One reachable operation, normalized for a consumer handler.

    ``inputs`` holds the target values of ``input_value_ids`` in the same order
    and with the same repetition: an operand used twice appears twice, as the
    same object. ``input_provenance`` gives each operand's dependency provenance,
    so a handler can tell a declared input from a seed, a forward capture, or a
    value the selection produced itself. ``op_params`` and ``output_typespec``
    are read-only views over copies, so a handler can neither mutate the analyzed
    graph nor see a later mutation of it.
    """

    node_id: str
    operator: TensorOperator
    op_params: Mapping[str, object]
    input_value_ids: tuple[str, ...]
    inputs: tuple[object, ...]
    input_provenance: tuple[str, ...]
    output_value_id: str
    output_typespec: Optional[Mapping[str, object]]


@runtime_checkable
class OperationHandler(Protocol):
    """Consumer-supplied lowering rule for one concrete operator type."""

    operator_type: type[TensorOperator]

    def lower(self, context: OperationContext) -> object:
        """Return the consumer's target value for *context*.

        Reporting the operation unsupported means raising
        ``AutodiffError("unsupported_operator", ...)``. Returning nothing, or
        failing with an uncategorized exception, is a contract violation.
        """


@dataclass(frozen=True)
class FusionContext:
    """The bounded look-ahead window a fusion hook may claim from.

    ``candidates`` are the next unclaimed operations in traversal order, capped
    by the hook's declared ``lookahead``; the first is always the operation the
    hook is being offered. ``value_of`` resolves an already-bound value id to its
    target value, which is how a fusion reaches the operands entering the region
    it wants to collapse.
    """

    candidates: tuple[TensorNodeRecord, ...]
    value_of: Callable[[str], object]


@dataclass(frozen=True)
class FusionResult:
    """One consumer instruction replacing a contiguous run of operations.

    ``consumed_node_ids`` must be a subset of the offered candidates, must be
    free of repeats, and must include the offered operation. Exactly one of the
    consumed operations may have a value that is still needed outside the region
    -- that value becomes ``value``; a region that would discard a second such
    value is rejected rather than silently dropped.
    """

    value: object
    consumed_node_ids: Sequence[str]


@runtime_checkable
class FusionHook(Protocol):
    """Consumer-owned pattern recognition over a bounded look-ahead window."""

    lookahead: int

    def fuse(self, context: FusionContext) -> Optional[FusionResult]:
        """Return a fusion for ``context.candidates[0]``, or ``None`` to decline."""


@dataclass(frozen=True)
class LoweredOperation:
    """One emitted target value and the source operations behind it."""

    output_value_id: str
    value: object
    source_node_ids: tuple[str, ...]
    source_operators: tuple[TensorOperator, ...]
    is_fused: bool


@dataclass(frozen=True)
class LoweredProgram:
    """The consumer's target program for a selected region of a graph.

    ``operations`` follow traversal order, ``values`` maps every bound value id
    to its target value, and ``dependencies`` is the framework analysis the
    traversal was derived from, so a consumer binds runtime inputs from
    provenance instead of from a graph scan.
    """

    selected_outputs: tuple[str, ...]
    operations: tuple[LoweredOperation, ...]
    values: Mapping[str, object]
    dependencies: DependencyAnalysis

    @property
    def output_values(self) -> tuple[object, ...]:
        """The target values of :attr:`selected_outputs`, in selection order."""
        return tuple(self.values[value_id] for value_id in self.selected_outputs)


class OperationHandlerRegistry:
    """Consumer-owned mapping from concrete operator types to lowering handlers.

    Registration is keyed by ``handler.operator_type`` and dispatch is by exact
    concrete type, so operator identity -- never a route name, a constant, or any
    other string -- decides which handler runs. One operator type accepts one
    handler: a second registration is rejected rather than silently overwriting
    the first, because "exactly one handler" cannot hold if the registry itself
    can hold two.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[TensorOperator], OperationHandler] = {}

    def register(self, handler: OperationHandler) -> None:
        """Register *handler* for its declared concrete operator type."""
        operator_type = getattr(handler, "operator_type", None)
        if not isinstance(operator_type, type) or not issubclass(operator_type, TensorOperator):
            raise AutodiffError(
                "handler_contract_violation",
                f"handler {type(handler).__name__!r} must declare a concrete "
                "TensorOperator subclass as its operator_type",
            )
        if not callable(getattr(handler, "lower", None)):
            raise AutodiffError(
                "handler_contract_violation",
                f"handler {type(handler).__name__!r} must provide a callable lower(context)",
            )
        existing = self._handlers.get(operator_type)
        if existing is not None:
            raise AutodiffError(
                "handler_contract_violation",
                f"operator type {operator_type.__name__!r} already has handler "
                f"{type(existing).__name__!r}; one operator type accepts one handler",
            )
        self._handlers[operator_type] = handler

    def handler(self, operator_type: type[TensorOperator]) -> Callable[[type], type]:
        """Class decorator registering the decorated handler for *operator_type*."""

        def decorator(handler_cls: type) -> type:
            instance = handler_cls()
            if getattr(instance, "operator_type", None) is None:
                object.__setattr__(instance, "operator_type", operator_type)
            if getattr(instance, "operator_type") is not operator_type:
                raise AutodiffError(
                    "handler_contract_violation",
                    f"handler {handler_cls.__name__!r} declares operator type "
                    f"{getattr(instance, 'operator_type')!r} but was registered for "
                    f"{operator_type!r}",
                )
            self.register(instance)
            return handler_cls

        return decorator

    def lookup(self, operator: TensorOperator) -> OperationHandler:
        """Return the handler for *operator*'s concrete type, or fail closed."""
        handler = self._handlers.get(type(operator))
        if handler is None:
            raise AutodiffError(
                "unsupported_operator",
                f"no lowering handler registered for operator type "
                f"{type(operator).__name__!r}",
            )
        return handler

    def has_handler(self, operator: TensorOperator | type[TensorOperator]) -> bool:
        """Report whether a handler is registered for *operator*'s concrete type."""
        operator_type = operator if isinstance(operator, type) else type(operator)
        return operator_type in self._handlers

    def supported_types(self) -> list[type[TensorOperator]]:
        """Return every concrete operator type this registry can lower."""
        return list(self._handlers)


def lower_graph(
    graph: TensorGraph,
    *,
    handlers: OperationHandlerRegistry,
    outputs: Optional[Sequence[str]] = None,
    fusion: Optional[FusionHook] = None,
    bind_input: Optional[Callable[[ValueDependency], object]] = None,
) -> LoweredProgram:
    """Lower the region of a forward *graph* reachable from its selected outputs.

    ``outputs`` defaults to the graph's own outputs. ``bind_input`` turns each
    analyzed free dependency into the consumer's target value for it; the default
    binds the :class:`ValueDependency` itself, which already carries the value
    id, provenance, and type metadata a consumer needs.
    """
    analysis = analyze_graph_dependencies(graph, outputs=outputs)
    return _lower(
        analysis=analysis,
        nodes=graph.nodes,
        handlers=handlers,
        fusion=fusion,
        bind_input=bind_input,
    )


def lower_derivative_program(
    program: "DerivativeProgram",
    *,
    forward_graph: TensorGraph,
    seed_value_ids: Sequence[str],
    handlers: OperationHandlerRegistry,
    outputs: Optional[Sequence[str]] = None,
    fusion: Optional[FusionHook] = None,
    bind_input: Optional[Callable[[ValueDependency], object]] = None,
) -> LoweredProgram:
    """Lower the region of a derivative *program* reachable from its selection.

    ``forward_graph`` and ``seed_value_ids`` are what let the analysis tell a
    seed and a forward capture apart from a genuinely missing value, so the
    consumer learns which inputs to materialize and which to retain from the
    forward run. ``outputs`` defaults to the program's output gradients.
    """
    analysis = analyze_derivative_dependencies(
        program,
        forward_graph=forward_graph,
        seed_value_ids=seed_value_ids,
        outputs=outputs,
    )
    return _lower(
        analysis=analysis,
        nodes=program.nodes,
        handlers=handlers,
        fusion=fusion,
        bind_input=bind_input,
    )


def _lower(
    *,
    analysis: DependencyAnalysis,
    nodes: Sequence[TensorNodeRecord],
    handlers: OperationHandlerRegistry,
    fusion: Optional[FusionHook],
    bind_input: Optional[Callable[[ValueDependency], object]],
) -> LoweredProgram:
    """Walk the analyzed region once, claiming every operation exactly once."""
    if not isinstance(handlers, OperationHandlerRegistry):
        raise AutodiffError(
            "handler_contract_violation",
            "lowering requires an OperationHandlerRegistry of concrete operator handlers",
        )
    _validate_fusion_hook(fusion)

    ordered_nodes = _ordered_operations(analysis, nodes)
    # Nothing is lowered until the whole region is known to be lowerable, so an
    # unsupported operation cannot leave a consumer holding a half-built program.
    _require_every_operation_supported(ordered_nodes, handlers)

    provenance = {
        dependency.value_id: dependency.provenance
        for dependency in analysis.dependencies
    }
    bind = bind_input if bind_input is not None else _default_bind_input

    values: dict[str, object] = {}
    for dependency in analysis.required_inputs:
        _bind_value(values, dependency.value_id, _call_consumer(bind, dependency))

    nodes_by_id = {node.node_id: node for node in ordered_nodes}
    consumers = _consumers_by_value(ordered_nodes)
    selected_outputs = set(analysis.selected_outputs)

    claims: dict[str, str] = {}
    operations: list[LoweredOperation] = []
    for position, node in enumerate(ordered_nodes):
        if node.node_id in claims:
            continue
        operation = _lower_one(
            node,
            position=position,
            ordered_nodes=ordered_nodes,
            nodes_by_id=nodes_by_id,
            consumers=consumers,
            selected_outputs=selected_outputs,
            handlers=handlers,
            fusion=fusion,
            provenance=provenance,
            claims=claims,
            values=values,
        )
        operations.append(operation)

    _require_every_operation_claimed(ordered_nodes, claims)
    _require_selected_outputs_bound(analysis.selected_outputs, values)

    return LoweredProgram(
        selected_outputs=tuple(analysis.selected_outputs),
        operations=tuple(operations),
        values=MappingProxyType(values),
        dependencies=analysis,
    )


def _lower_one(
    node: TensorNodeRecord,
    *,
    position: int,
    ordered_nodes: Sequence[TensorNodeRecord],
    nodes_by_id: Mapping[str, TensorNodeRecord],
    consumers: Mapping[str, tuple[str, ...]],
    selected_outputs: set[str],
    handlers: OperationHandlerRegistry,
    fusion: Optional[FusionHook],
    provenance: Mapping[str, str],
    claims: dict[str, str],
    values: dict[str, object],
) -> LoweredOperation:
    """Lower *node* through the fusion hook if it claims it, else its handler."""
    if fusion is not None:
        candidates = _fusion_candidates(ordered_nodes, position, claims, fusion.lookahead)
        result = _call_consumer(fusion.fuse, FusionContext(candidates, lambda value_id: _value_of(values, value_id)))
        if result is not None:
            return _lower_fused(
                result,
                offered=node,
                candidates=candidates,
                nodes_by_id=nodes_by_id,
                consumers=consumers,
                selected_outputs=selected_outputs,
                claims=claims,
                values=values,
            )

    handler = handlers.lookup(node.operator)
    context = _operation_context(node, values=values, provenance=provenance)
    value = _call_consumer(handler.lower, context)
    if value is None:
        raise AutodiffError(
            "handler_contract_violation",
            f"handler {type(handler).__name__!r} returned no target value for "
            f"operation {node.node_id!r}",
        )
    return _claim(
        claims,
        values,
        node_ids=(node.node_id,),
        claim=LOWERING_CLAIM_HANDLER,
        nodes_by_id=nodes_by_id,
        result_value_id=node.output_value_id,
        value=value,
        is_fused=False,
    )


def _lower_fused(
    result: object,
    *,
    offered: TensorNodeRecord,
    candidates: tuple[TensorNodeRecord, ...],
    nodes_by_id: Mapping[str, TensorNodeRecord],
    consumers: Mapping[str, tuple[str, ...]],
    selected_outputs: set[str],
    claims: dict[str, str],
    values: dict[str, object],
) -> LoweredOperation:
    """Validate one fusion and claim the operations it replaces."""
    if not isinstance(result, FusionResult):
        raise AutodiffError(
            "handler_contract_violation",
            "a fusion hook must return a FusionResult or None, got "
            f"{type(result).__name__!r}",
        )
    if result.value is None:
        raise AutodiffError(
            "handler_contract_violation",
            f"the fusion offered operation {offered.node_id!r} returned no target value",
        )

    consumed = _fusion_consumed_node_ids(result, offered=offered, candidates=candidates)
    result_value_id = _fusion_result_value_id(
        consumed,
        offered=offered,
        nodes_by_id=nodes_by_id,
        consumers=consumers,
        selected_outputs=selected_outputs,
    )
    return _claim(
        claims,
        values,
        node_ids=consumed,
        claim=LOWERING_CLAIM_FUSION,
        nodes_by_id=nodes_by_id,
        result_value_id=result_value_id,
        value=result.value,
        is_fused=True,
    )


def _claim(
    claims: dict[str, str],
    values: dict[str, object],
    *,
    node_ids: tuple[str, ...],
    claim: str,
    nodes_by_id: Mapping[str, TensorNodeRecord],
    result_value_id: str,
    value: object,
    is_fused: bool,
) -> LoweredOperation:
    """Claim *node_ids* for one emitted value; the sole way an operation is lowered.

    This is the single point where the "one reachable operation, one lowering"
    invariant is enforced. Both the single-handler path and the fusion path pass
    through it, so a doubly-lowered operation is rejected identically whichever
    route reached it, and no target value can enter the value environment without
    a claim first being recorded for the operation that produced it.
    """
    for node_id in node_ids:
        existing = claims.get(node_id)
        if existing is not None:
            raise AutodiffError(
                "handler_contract_violation",
                f"operation {node_id!r} is already lowered by a {existing} and "
                f"cannot also be lowered by a {claim}",
            )
        claims[node_id] = claim

    _bind_value(values, result_value_id, value)
    source_nodes = tuple(nodes_by_id[node_id] for node_id in node_ids)
    return LoweredOperation(
        output_value_id=result_value_id,
        value=value,
        source_node_ids=node_ids,
        source_operators=tuple(source.operator for source in source_nodes),
        is_fused=is_fused,
    )


def _require_every_operation_claimed(
    ordered_nodes: Sequence[TensorNodeRecord],
    claims: Mapping[str, str],
) -> None:
    """Assert the claims cover the reachable region exactly, once, as one check.

    :func:`_claim` can only ever observe an operation lowered twice. One set
    comparison over the whole region is what also rules out an operation that no
    handler and no fusion ever reached, so "exactly once" stays a single enforced
    property instead of two guards that each cover half of it.
    """
    reachable = {node.node_id for node in ordered_nodes}
    claimed = set(claims)
    if claimed == reachable:
        return
    unclaimed = sorted(reachable - claimed)
    if unclaimed:
        joined = ", ".join(repr(node_id) for node_id in unclaimed)
        raise AutodiffError(
            "unsupported_operator",
            f"reachable operation(s) {joined} were not lowered by any handler or fusion",
        )
    unknown = ", ".join(repr(node_id) for node_id in sorted(claimed - reachable))
    raise AutodiffError(
        "handler_contract_violation",
        f"operation(s) {unknown} were lowered but are not part of the selected region",
    )


def _require_every_operation_supported(
    ordered_nodes: Sequence[TensorNodeRecord],
    handlers: OperationHandlerRegistry,
) -> None:
    """Fail closed before any handler runs if the region is not fully supported.

    Every reachable operation must be individually lowerable. Fusion may then
    replace a run of supported operations with one instruction, but it is never
    the only way an operation can be expressed, so support does not depend on
    which patterns a hook happens to recognize.
    """
    for node in ordered_nodes:
        handlers.lookup(node.operator)


def _require_selected_outputs_bound(
    selected_outputs: Sequence[str],
    values: Mapping[str, object],
) -> None:
    """Assert output selection can read every selection from the environment.

    Selection reads the value environment and nothing else, so it cannot become a
    second route to lowering an operation. This only reports the impossible.
    """
    missing = [value_id for value_id in selected_outputs if value_id not in values]
    if missing:
        joined = ", ".join(repr(value_id) for value_id in missing)
        raise AutodiffError(
            "malformed_derivative_ir",
            f"selected output(s) {joined} were never bound to a target value",
        )


def _bind_value(values: dict[str, object], value_id: str, value: object) -> None:
    """Bind one target value, rejecting a value id that already carries one."""
    if value_id in values:
        raise AutodiffError(
            "handler_contract_violation",
            f"value {value_id!r} is already bound to a target value",
        )
    values[value_id] = value


def _value_of(values: Mapping[str, object], value_id: str) -> object:
    """Resolve an already-bound value id for a consumer, failing closed."""
    if value_id not in values:
        raise AutodiffError(
            "handler_contract_violation",
            f"value {value_id!r} is not bound yet and cannot be used as a fusion operand",
        )
    return values[value_id]


def _ordered_operations(
    analysis: DependencyAnalysis,
    nodes: Sequence[TensorNodeRecord],
) -> tuple[TensorNodeRecord, ...]:
    """Return the reachable operations in the analysis's own traversal order.

    Reachability and topological order come from the dependency analysis, which
    already rejects cycles, ambiguous producers, unknown selections, and missing
    values. The producer index below is only a lookup from an analyzed local
    value back to the operation that produced it.
    """
    producers = {node.output_value_id: node for node in nodes}
    ordered: list[TensorNodeRecord] = []
    for dependency in analysis.with_provenance(DEPENDENCY_PROVENANCE_LOCAL_VALUE):
        node = producers.get(dependency.value_id)
        if node is None:
            raise AutodiffError(
                "malformed_derivative_ir",
                f"analyzed local value {dependency.value_id!r} has no producing operation",
            )
        ordered.append(node)
    return tuple(ordered)


def _consumers_by_value(
    ordered_nodes: Sequence[TensorNodeRecord],
) -> dict[str, tuple[str, ...]]:
    """Index which reachable operations consume each value."""
    consumers: dict[str, list[str]] = {}
    for node in ordered_nodes:
        for value_id in node.input_value_ids:
            consumers.setdefault(value_id, []).append(node.node_id)
    return {value_id: tuple(node_ids) for value_id, node_ids in consumers.items()}


def _operation_context(
    node: TensorNodeRecord,
    *,
    values: Mapping[str, object],
    provenance: Mapping[str, str],
) -> OperationContext:
    """Normalize one operation and its already-lowered operands for a handler."""
    input_value_ids = tuple(node.input_value_ids)
    inputs = tuple(_value_of(values, value_id) for value_id in input_value_ids)
    typespec = node.output_typespec
    return OperationContext(
        node_id=node.node_id,
        operator=node.operator,
        op_params=MappingProxyType(dict(node.op_params)),
        input_value_ids=input_value_ids,
        inputs=inputs,
        input_provenance=tuple(provenance.get(value_id, "") for value_id in input_value_ids),
        output_value_id=node.output_value_id,
        output_typespec=None if typespec is None else MappingProxyType(dict(typespec)),
    )


def _validate_fusion_hook(fusion: Optional[FusionHook]) -> None:
    """Reject a fusion hook that declares no usable bound or no callable."""
    if fusion is None:
        return
    lookahead = getattr(fusion, "lookahead", None)
    if isinstance(lookahead, bool) or not isinstance(lookahead, int) or lookahead < 1:
        raise AutodiffError(
            "handler_contract_violation",
            f"fusion hook {type(fusion).__name__!r} must declare a positive integer "
            f"lookahead, got {lookahead!r}",
        )
    if not callable(getattr(fusion, "fuse", None)):
        raise AutodiffError(
            "handler_contract_violation",
            f"fusion hook {type(fusion).__name__!r} must provide a callable fuse(context)",
        )


def _fusion_candidates(
    ordered_nodes: Sequence[TensorNodeRecord],
    position: int,
    claims: Mapping[str, str],
    lookahead: int,
) -> tuple[TensorNodeRecord, ...]:
    """Return the next unclaimed operations from *position*, capped by *lookahead*."""
    candidates: list[TensorNodeRecord] = []
    for node in ordered_nodes[position:]:
        if node.node_id in claims:
            continue
        candidates.append(node)
        if len(candidates) == lookahead:
            break
    return tuple(candidates)


def _fusion_consumed_node_ids(
    result: FusionResult,
    *,
    offered: TensorNodeRecord,
    candidates: Sequence[TensorNodeRecord],
) -> tuple[str, ...]:
    """Normalize and validate the operations one fusion claims."""
    consumed = result.consumed_node_ids
    if isinstance(consumed, (str, bytes)) or not isinstance(consumed, Sequence):
        raise AutodiffError(
            "handler_contract_violation",
            "a fusion must claim a non-string sequence of operation ids",
        )

    offered_ids = [candidate.node_id for candidate in candidates]
    claimed: list[str] = []
    for node_id in consumed:
        if not isinstance(node_id, str) or not node_id:
            raise AutodiffError(
                "handler_contract_violation",
                f"fusion claimed {node_id!r}, which is not an operation id",
            )
        if node_id in claimed:
            raise AutodiffError(
                "handler_contract_violation",
                f"fusion claimed operation {node_id!r} more than once",
            )
        if node_id not in offered_ids:
            raise AutodiffError(
                "handler_contract_violation",
                f"fusion claimed operation {node_id!r}, which was not among the "
                f"operations it was offered",
            )
        claimed.append(node_id)

    if not claimed:
        raise AutodiffError(
            "handler_contract_violation",
            f"fusion offered operation {offered.node_id!r} claimed no operations",
        )
    if offered.node_id not in claimed:
        raise AutodiffError(
            "handler_contract_violation",
            f"fusion was offered operation {offered.node_id!r} but did not claim it",
        )
    # Traversal order, not claim order, so the emitted provenance of a fused
    # region is the source order regardless of how the hook listed it.
    return tuple(node_id for node_id in offered_ids if node_id in claimed)


def _fusion_result_value_id(
    consumed: tuple[str, ...],
    *,
    offered: TensorNodeRecord,
    nodes_by_id: Mapping[str, TensorNodeRecord],
    consumers: Mapping[str, tuple[str, ...]],
    selected_outputs: set[str],
) -> str:
    """Return the one value a fused region still owes the rest of the program.

    This is fusion *legality*, deliberately a separate invariant from
    exactly-once lowering: a fused region emits one target value, so every other
    value it produces disappears into the instruction. That is only sound while
    no operation outside the region, and no selected output, still needs one of
    them. A region that would strand a second live value is rejected instead of
    silently dropping it.
    """
    consumed_ids = set(consumed)
    escaping = [
        nodes_by_id[node_id].output_value_id
        for node_id in consumed
        if _value_escapes(
            nodes_by_id[node_id].output_value_id,
            consumed_ids=consumed_ids,
            consumers=consumers,
            selected_outputs=selected_outputs,
        )
    ]
    if len(escaping) == 1:
        return escaping[0]
    joined = ", ".join(repr(value_id) for value_id in escaping)
    raise AutodiffError(
        "handler_contract_violation",
        f"the fusion offered operation {offered.node_id!r} emits one target value "
        f"but {len(escaping)} of its values are still needed outside it: {joined}",
    )


def _value_escapes(
    value_id: str,
    *,
    consumed_ids: set[str],
    consumers: Mapping[str, tuple[str, ...]],
    selected_outputs: set[str],
) -> bool:
    """Report whether *value_id* is needed outside a fused region."""
    if value_id in selected_outputs:
        return True
    return any(
        node_id not in consumed_ids for node_id in consumers.get(value_id, ())
    )


def _default_bind_input(dependency: ValueDependency) -> object:
    """Bind a free dependency to the analysis row describing it."""
    return dependency


def _call_consumer(callback: Callable[..., object], argument: object) -> object:
    """Invoke a consumer callback, categorizing anything uncategorized it leaks."""
    try:
        return callback(argument)
    except AutodiffError:
        raise
    except _CONSUMER_FAILURES as exc:
        raise AutodiffError(
            "handler_contract_violation",
            f"consumer callback {getattr(callback, '__qualname__', callback)!r} "
            f"failed with {type(exc).__name__}: {exc}",
        ) from exc
