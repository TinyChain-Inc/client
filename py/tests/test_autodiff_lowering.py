"""Unit tests for the framework-owned program construction and lowering seam.

These tests pin the contract that a consumer supplies only concrete-operator
handlers and an optional fusion hook, while the framework owns reachability,
traversal order, dependency validation, output selection, and every failure
mode. The fake consumer below emits its own representation and the framework
never imports, inspects, or interprets it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest
import tinychain as tc
from tinychain.autodiff import (
    AUTODIFF_ERROR_CATEGORIES,
    AddOperator,
    AutodiffError,
    BroadcastOperator,
    BroadcastReduceOperator,
    DivOperator,
    MatmulOperator,
    MaxOperator,
    MeanOperator,
    MinOperator,
    MulOperator,
    ProductOperator,
    ReshapeOperator,
    SubOperator,
    SumOperator,
    TensorGraph,
    TensorGraphBuilder,
    TensorNodeRecord,
    TensorOperator,
    TransposeOperator,
)
from tinychain.autodiff.dependencies import (
    DEPENDENCY_PROVENANCE_DECLARED_INPUT,
    DEPENDENCY_PROVENANCE_FORWARD_CAPTURE,
    DEPENDENCY_PROVENANCE_SEED_INPUT,
    ValueDependency,
)
from tinychain.autodiff.lowering import (
    FusionContext,
    FusionResult,
    LoweredOperation,
    LoweredProgram,
    OperationContext,
    OperationHandlerRegistry,
    lower_derivative_program,
    lower_graph,
)


# --------------------------------------------------------------------------
# fake consumer — no framework type appears in its target representation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeTarget:
    """A target value owned entirely by the fake consumer."""

    kind: str
    operands: tuple[object, ...] = ()


@dataclass(frozen=True)
class FakeHandler:
    """Fake consumer handler keyed by a concrete operator type."""

    operator_type: type
    kind: str
    calls: list[OperationContext] = field(default_factory=list)

    def lower(self, context: OperationContext) -> object:
        self.calls.append(context)
        return FakeTarget(self.kind, tuple(context.inputs))


@dataclass(frozen=True)
class BrokenHandler:
    """Handler that violates the return contract by emitting nothing."""

    operator_type: type

    def lower(self, context: OperationContext) -> object:
        return None


@dataclass(frozen=True)
class RaisingHandler:
    """Handler that fails with an uncategorized container exception."""

    operator_type: type

    def lower(self, context: OperationContext) -> object:
        raise KeyError("consumer lookup failed")


@dataclass(frozen=True)
class TransposeMatmulFusion:
    """Fuse a transpose feeding a matmul into one consumer instruction."""

    lookahead: int = 2
    windows: list[tuple[str, ...]] = field(default_factory=list)

    def fuse(self, context: FusionContext) -> Optional[FusionResult]:
        self.windows.append(tuple(node.node_id for node in context.candidates))
        if len(context.candidates) < 2:
            return None
        first, second = context.candidates[0], context.candidates[1]
        if not isinstance(first.operator, TransposeOperator):
            return None
        if not isinstance(second.operator, MatmulOperator):
            return None
        if first.output_value_id not in second.input_value_ids:
            return None
        operands = tuple(
            context.value_of(value_id)
            for value_id in second.input_value_ids
            if value_id != first.output_value_id
        )
        fused = FakeTarget("fake.transposed_matmul", (context.value_of(first.input_value_ids[0]),) + operands)
        return FusionResult(value=fused, consumed_node_ids=(first.node_id, second.node_id))


@dataclass(frozen=True)
class ScriptedFusion:
    """Fusion hook that returns a caller-supplied result at the first offer."""

    result: FusionResult
    lookahead: int = 2
    offers: list[int] = field(default_factory=list)

    def fuse(self, context: FusionContext) -> Optional[FusionResult]:
        self.offers.append(len(context.candidates))
        if len(self.offers) > 1:
            return None
        return self.result


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


CONCRETE_OPERATOR_TYPES = (
    AddOperator,
    SubOperator,
    MulOperator,
    DivOperator,
    SumOperator,
    MeanOperator,
    MaxOperator,
    MinOperator,
    ProductOperator,
    ReshapeOperator,
    BroadcastOperator,
    BroadcastReduceOperator,
    MatmulOperator,
    TransposeOperator,
)


def _typespec(dtype: str, shape: list[object]) -> dict[str, object]:
    return {"dtype": dtype, "shape": list(shape)}


def _node(
    node_id: str,
    output: str,
    operator: TensorOperator,
    inputs: list[str],
    typespec: object,
    op_params: Optional[dict] = None,
) -> TensorNodeRecord:
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output,
        operator=operator,
        op_params=dict(op_params or {}),
        input_value_ids=list(inputs),
        output_typespec=typespec,
    )


def _registry(*handlers: object) -> OperationHandlerRegistry:
    registry = OperationHandlerRegistry()
    for handler in handlers:
        registry.register(handler)
    return registry


def _full_registry() -> OperationHandlerRegistry:
    registry = OperationHandlerRegistry()
    for operator_type in CONCRETE_OPERATOR_TYPES:
        registry.register(FakeHandler(operator_type, f"fake.{operator_type.__name__}"))
    return registry


def _bind_input(dependency: ValueDependency) -> object:
    return FakeTarget(f"fake.input:{dependency.value_id}")


def _matrix(dtype: str = "f32") -> dict[str, object]:
    return _typespec(dtype, [2, 2])


def _binary_graph() -> TensorGraph:
    """alpha + beta, then that sum matmul beta."""
    return TensorGraph(
        nodes=[
            _node("n0", "sum", AddOperator(), ["alpha", "beta"], _matrix()),
            _node("n1", "out", MatmulOperator(), ["sum", "beta"], _matrix()),
        ],
        inputs=[("alpha", _matrix()), ("beta", _matrix())],
        outputs=["out"],
    )


def _repeated_value_graph() -> TensorGraph:
    return TensorGraph(
        nodes=[_node("n0", "out", AddOperator(), ["alpha", "alpha"], _matrix())],
        inputs=[("alpha", _matrix())],
        outputs=["out"],
    )


def _two_output_graph() -> TensorGraph:
    """One selected output reaches the add/mul chain, the other only the matmul."""
    return TensorGraph(
        nodes=[
            _node("n0", "sum", AddOperator(), ["alpha", "beta"], _matrix()),
            _node("n1", "scaled", MulOperator(), ["sum", "beta"], _matrix()),
            _node("n2", "product", MatmulOperator(), ["alpha", "beta"], _matrix()),
        ],
        inputs=[("alpha", _matrix()), ("beta", _matrix())],
        outputs=["scaled", "product"],
    )


def _fusion_graph() -> TensorGraph:
    """transpose(alpha) @ beta — the canonical local fusion candidate."""
    return TensorGraph(
        nodes=[
            _node(
                "n0",
                "alpha_t",
                TransposeOperator(),
                ["alpha"],
                _typespec("f32", [3, 2]),
                {"perm": [1, 0]},
            ),
            _node("n1", "out", MatmulOperator(), ["alpha_t", "beta"], _typespec("f32", [3, 4])),
        ],
        inputs=[("alpha", _typespec("f32", [2, 3])), ("beta", _typespec("f32", [2, 4]))],
        outputs=["out"],
    )


def _fusion_graph_with_live_intermediate() -> TensorGraph:
    """The transpose result is also consumed outside the fusion candidate."""
    return TensorGraph(
        nodes=[
            _node(
                "n0",
                "alpha_t",
                TransposeOperator(),
                ["alpha"],
                _typespec("f32", [3, 2]),
                {"perm": [1, 0]},
            ),
            _node("n1", "out", MatmulOperator(), ["alpha_t", "beta"], _typespec("f32", [3, 4])),
            _node("n2", "doubled", AddOperator(), ["alpha_t", "alpha_t"], _typespec("f32", [3, 2])),
        ],
        inputs=[("alpha", _typespec("f32", [2, 3])), ("beta", _typespec("f32", [2, 4]))],
        outputs=["out", "doubled"],
    )


def _fusion_graph_with_trailing_operation() -> TensorGraph:
    """transpose(alpha) @ beta, then that result added to itself."""
    return TensorGraph(
        nodes=[
            _node(
                "n0",
                "alpha_t",
                TransposeOperator(),
                ["alpha"],
                _typespec("f32", [3, 2]),
                {"perm": [1, 0]},
            ),
            _node("n1", "out", MatmulOperator(), ["alpha_t", "beta"], _typespec("f32", [3, 4])),
            _node("n2", "doubled", AddOperator(), ["out", "out"], _typespec("f32", [3, 4])),
        ],
        inputs=[("alpha", _typespec("f32", [2, 3])), ("beta", _typespec("f32", [2, 4]))],
        outputs=["doubled"],
    )


def _lower(graph: TensorGraph, registry: OperationHandlerRegistry, **kwargs) -> LoweredProgram:
    return lower_graph(graph, handlers=registry, bind_input=_bind_input, **kwargs)


def _assert_category(category: str, operation) -> None:
    with pytest.raises(AutodiffError) as error:
        operation()
    assert error.value.category == category


def _trace_linear_mse(dtype: str = "f32"):
    trace = TensorGraphBuilder()
    with tc.state.scoped_context():
        with trace:
            images = trace.input("images", dtype=dtype, shape=(2, 3))
            weights = trace.input("weights", dtype=dtype, shape=(3, 4))
            labels = trace.input("labels", dtype=dtype, shape=(2, 4))
            residual = (images @ weights) - labels
            loss = (residual * residual).mean([0, 1])
    return trace, images, weights, labels, loss


# --------------------------------------------------------------------------
# AC1 — the framework, not the consumer, owns reachability and traversal
# --------------------------------------------------------------------------


def test_only_operations_reachable_from_the_selected_output_are_lowered():
    graph = _two_output_graph()
    registry = _full_registry()

    lowered = _lower(graph, registry, outputs=["product"])

    assert [operation.output_value_id for operation in lowered.operations] == ["product"]
    assert [operation.source_node_ids for operation in lowered.operations] == [("n2",)]


def test_selected_output_slicing_reports_only_the_selected_outputs():
    graph = _two_output_graph()

    lowered = _lower(graph, _full_registry(), outputs=["scaled"])

    assert lowered.selected_outputs == ("scaled",)
    assert [operation.output_value_id for operation in lowered.operations] == ["sum", "scaled"]
    assert lowered.output_values == (lowered.values["scaled"],)


def test_operations_are_lowered_in_dependency_order_with_inputs_already_bound():
    graph = _binary_graph()
    add_handler = FakeHandler(AddOperator, "fake.add")
    matmul_handler = FakeHandler(MatmulOperator, "fake.matmul")

    lowered = _lower(graph, _registry(add_handler, matmul_handler))

    assert [operation.output_value_id for operation in lowered.operations] == ["sum", "out"]
    (matmul_call,) = matmul_handler.calls
    assert matmul_call.inputs[0] == FakeTarget("fake.add", (FakeTarget("fake.input:alpha"), FakeTarget("fake.input:beta")))
    assert matmul_call.inputs[1] == FakeTarget("fake.input:beta")


def test_a_repeated_input_value_is_bound_once_and_passed_twice():
    graph = _repeated_value_graph()
    add_handler = FakeHandler(AddOperator, "fake.add")

    lowered = _lower(graph, _registry(add_handler))

    (call,) = add_handler.calls
    assert call.input_value_ids == ("alpha", "alpha")
    assert call.inputs[0] is call.inputs[1]
    assert len(lowered.operations) == 1


def test_lowering_exposes_the_framework_owned_dependency_analysis():
    graph = _binary_graph()

    lowered = _lower(graph, _full_registry())

    provenance = {
        dependency.value_id: dependency.provenance
        for dependency in lowered.dependencies.required_inputs
    }
    assert provenance == {
        "alpha": DEPENDENCY_PROVENANCE_DECLARED_INPUT,
        "beta": DEPENDENCY_PROVENANCE_DECLARED_INPUT,
    }
    assert lowered.values["alpha"] == FakeTarget("fake.input:alpha")


def test_every_reachable_operation_is_lowered_exactly_once():
    graph = _fusion_graph_with_live_intermediate()

    lowered = _lower(graph, _full_registry())

    claimed = [
        node_id
        for operation in lowered.operations
        for node_id in operation.source_node_ids
    ]
    assert sorted(claimed) == ["n0", "n1", "n2"]
    assert len(claimed) == len(set(claimed))


def test_input_binding_defaults_to_the_analyzed_dependency():
    graph = _repeated_value_graph()

    lowered = lower_graph(graph, handlers=_full_registry())

    assert isinstance(lowered.values["alpha"], ValueDependency)
    assert lowered.values["alpha"].value_id == "alpha"


# --------------------------------------------------------------------------
# AC2 — handlers are keyed by concrete operator identity
# --------------------------------------------------------------------------


def test_handlers_dispatch_by_concrete_operator_type():
    graph = _binary_graph()
    add_handler = FakeHandler(AddOperator, "fake.add")
    matmul_handler = FakeHandler(MatmulOperator, "fake.matmul")

    lowered = _lower(graph, _registry(add_handler, matmul_handler))

    assert len(add_handler.calls) == 1
    assert len(matmul_handler.calls) == 1
    assert lowered.values["out"].kind == "fake.matmul"
    assert lowered.values["sum"].kind == "fake.add"


def test_a_look_alike_operator_route_name_is_not_dispatched_to_the_add_handler():
    @dataclass(frozen=True)
    class LookAlikeOperator(TensorOperator):
        def __init__(self) -> None:
            object.__setattr__(self, "route_name", "add")

    graph = TensorGraph(
        nodes=[_node("n0", "out", LookAlikeOperator(), ["alpha"], _matrix())],
        inputs=[("alpha", _matrix())],
        outputs=["out"],
    )

    _assert_category(
        "unsupported_operator",
        lambda: _lower(graph, _registry(FakeHandler(AddOperator, "fake.add"))),
    )


def test_registering_two_handlers_for_one_operator_type_is_rejected():
    registry = OperationHandlerRegistry()
    registry.register(FakeHandler(AddOperator, "fake.add"))

    _assert_category(
        "handler_contract_violation",
        lambda: registry.register(FakeHandler(AddOperator, "fake.other_add")),
    )


def test_registry_reports_its_supported_concrete_operator_types():
    registry = _registry(
        FakeHandler(AddOperator, "fake.add"),
        FakeHandler(MatmulOperator, "fake.matmul"),
    )

    assert set(registry.supported_types()) == {AddOperator, MatmulOperator}
    assert registry.has_handler(AddOperator())
    assert not registry.has_handler(MulOperator())


# --------------------------------------------------------------------------
# AC3 — a fake consumer emits its own representation
# --------------------------------------------------------------------------


def test_the_fake_consumer_target_values_are_returned_verbatim_and_in_output_order():
    graph = _two_output_graph()

    lowered = _lower(graph, _full_registry())

    assert lowered.selected_outputs == ("scaled", "product")
    assert lowered.output_values == (lowered.values["scaled"], lowered.values["product"])
    assert all(isinstance(value, FakeTarget) for value in lowered.output_values)


def test_the_framework_never_inspects_an_opaque_target_value():
    class Opaque:
        def __eq__(self, other: object) -> bool:
            raise AssertionError("the framework compared an opaque target value")

        def __hash__(self) -> int:
            raise AssertionError("the framework hashed an opaque target value")

        def __iter__(self):
            raise AssertionError("the framework iterated an opaque target value")

        def __bool__(self) -> bool:
            raise AssertionError("the framework tested an opaque target value")

    @dataclass(frozen=True)
    class OpaqueHandler:
        operator_type: type

        def lower(self, context: OperationContext) -> object:
            return Opaque()

    graph = _repeated_value_graph()

    lowered = lower_graph(
        graph,
        handlers=_registry(OpaqueHandler(AddOperator)),
        bind_input=lambda dependency: Opaque(),
    )

    assert isinstance(lowered.output_values[0], Opaque)


def test_operation_context_normalizes_the_operation_without_exposing_mutable_state():
    graph = _fusion_graph()
    transpose_handler = FakeHandler(TransposeOperator, "fake.transpose")

    _lower(graph, _registry(transpose_handler, FakeHandler(MatmulOperator, "fake.matmul")))

    (call,) = transpose_handler.calls
    assert call.node_id == "n0"
    assert isinstance(call.operator, TransposeOperator)
    assert call.output_value_id == "alpha_t"
    assert call.op_params["perm"] == [1, 0]
    assert call.output_typespec == _typespec("f32", [3, 2])
    assert call.input_provenance == (DEPENDENCY_PROVENANCE_DECLARED_INPUT,)
    graph_params = graph.nodes[0].op_params
    with pytest.raises(TypeError):
        call.op_params["perm"] = [0, 1]
    assert graph_params == {"perm": [1, 0]}


# --------------------------------------------------------------------------
# AC4 — fusion is optional, explicit, deterministic, provenance-preserving
# --------------------------------------------------------------------------


def test_without_a_fusion_hook_every_operation_is_lowered_separately():
    graph = _fusion_graph()

    lowered = _lower(graph, _full_registry())

    assert [operation.source_node_ids for operation in lowered.operations] == [("n0",), ("n1",)]
    assert all(not operation.is_fused for operation in lowered.operations)


def test_a_fusion_hook_lowers_a_local_pattern_as_one_operation():
    graph = _fusion_graph()
    fusion = TransposeMatmulFusion()

    lowered = _lower(graph, _full_registry(), fusion=fusion)

    assert len(lowered.operations) == 1
    (operation,) = lowered.operations
    assert operation.is_fused
    assert operation.output_value_id == "out"
    assert lowered.values["out"].kind == "fake.transposed_matmul"


def test_a_fused_operation_preserves_its_source_provenance():
    graph = _fusion_graph()

    lowered = _lower(graph, _full_registry(), fusion=TransposeMatmulFusion())

    (operation,) = lowered.operations
    assert operation.source_node_ids == ("n0", "n1")
    assert [type(operator) for operator in operation.source_operators] == [
        TransposeOperator,
        MatmulOperator,
    ]


def test_fusion_is_deterministic_across_repeated_lowering():
    graph = _fusion_graph()

    first = _lower(graph, _full_registry(), fusion=TransposeMatmulFusion())
    second = _lower(graph, _full_registry(), fusion=TransposeMatmulFusion())

    def digest(lowered: LoweredProgram) -> list[tuple[object, ...]]:
        return [
            (operation.output_value_id, operation.source_node_ids, operation.is_fused)
            for operation in lowered.operations
        ]

    assert digest(first) == digest(second)
    assert first.output_values == second.output_values


def test_the_fusion_candidate_window_is_bounded_by_the_declared_lookahead():
    graph = _fusion_graph_with_trailing_operation()
    fusion = TransposeMatmulFusion(lookahead=2)

    _lower(graph, _full_registry(), fusion=fusion)

    assert fusion.windows
    assert all(len(window) <= 2 for window in fusion.windows)
    assert fusion.windows[0] == ("n0", "n1")


def test_a_fusion_that_would_drop_a_value_used_elsewhere_is_rejected():
    graph = _fusion_graph_with_live_intermediate()

    _assert_category(
        "handler_contract_violation",
        lambda: _lower(graph, _full_registry(), fusion=TransposeMatmulFusion()),
    )


def test_a_fusion_claiming_the_same_operation_twice_is_rejected():
    graph = _fusion_graph()
    fusion = ScriptedFusion(FusionResult(value=FakeTarget("fake.fused"), consumed_node_ids=("n0", "n0")))

    _assert_category(
        "handler_contract_violation",
        lambda: _lower(graph, _full_registry(), fusion=fusion),
    )


def test_a_fusion_must_claim_the_operation_it_was_offered():
    graph = _fusion_graph()
    fusion = ScriptedFusion(FusionResult(value=FakeTarget("fake.fused"), consumed_node_ids=("n1",)))

    _assert_category(
        "handler_contract_violation",
        lambda: _lower(graph, _full_registry(), fusion=fusion),
    )


def test_a_fusion_claiming_an_operation_outside_the_offered_window_is_rejected():
    graph = _fusion_graph_with_live_intermediate()
    fusion = ScriptedFusion(
        FusionResult(value=FakeTarget("fake.fused"), consumed_node_ids=("n0", "n2")),
        lookahead=2,
    )

    _assert_category(
        "handler_contract_violation",
        lambda: _lower(graph, _full_registry(), fusion=fusion),
    )


def test_a_fusion_hook_with_a_non_positive_lookahead_is_rejected():
    graph = _fusion_graph()
    fusion = ScriptedFusion(FusionResult(value=FakeTarget("fake.fused"), consumed_node_ids=("n0",)), lookahead=0)

    _assert_category(
        "handler_contract_violation",
        lambda: _lower(graph, _full_registry(), fusion=fusion),
    )


def test_a_declined_fusion_falls_back_to_per_operation_handlers():
    graph = _binary_graph()
    fusion = TransposeMatmulFusion()

    lowered = _lower(graph, _full_registry(), fusion=fusion)

    assert [operation.source_node_ids for operation in lowered.operations] == [("n0",), ("n1",)]
    assert fusion.windows


# --------------------------------------------------------------------------
# AC5 — unsupported or malformed inputs fail with categorized errors
# --------------------------------------------------------------------------


def test_an_operation_without_a_registered_handler_raises_unsupported_operator():
    graph = _binary_graph()

    _assert_category(
        "unsupported_operator",
        lambda: _lower(graph, _registry(FakeHandler(AddOperator, "fake.add"))),
    )


def test_an_unsupported_operation_fails_before_any_handler_emits_a_value():
    graph = _binary_graph()
    add_handler = FakeHandler(AddOperator, "fake.add")

    with pytest.raises(AutodiffError):
        _lower(graph, _registry(add_handler))

    assert add_handler.calls == []


def test_a_handler_returning_no_target_value_raises_handler_contract_violation():
    graph = _repeated_value_graph()

    _assert_category(
        "handler_contract_violation",
        lambda: _lower(graph, _registry(BrokenHandler(AddOperator))),
    )


def test_a_handler_raising_an_uncategorized_error_is_reported_as_a_contract_violation():
    graph = _repeated_value_graph()

    _assert_category(
        "handler_contract_violation",
        lambda: _lower(graph, _registry(RaisingHandler(AddOperator))),
    )


def test_a_handler_raising_a_categorized_autodiff_error_is_propagated_unchanged():
    @dataclass(frozen=True)
    class UnsupportedHandler:
        operator_type: type

        def lower(self, context: OperationContext) -> object:
            raise AutodiffError("unsupported_operator", "the consumer target cannot express add")

    graph = _repeated_value_graph()

    _assert_category(
        "unsupported_operator",
        lambda: _lower(graph, _registry(UnsupportedHandler(AddOperator))),
    )


def test_registering_a_handler_without_a_concrete_operator_type_is_rejected():
    registry = OperationHandlerRegistry()

    _assert_category(
        "handler_contract_violation",
        lambda: registry.register(FakeHandler(str, "fake.not_an_operator")),
    )


def test_a_cycle_fails_before_any_handler_runs():
    graph = TensorGraph(
        nodes=[
            _node("n0", "left", AddOperator(), ["right"], _matrix()),
            _node("n1", "right", AddOperator(), ["left"], _matrix()),
        ],
        inputs=[],
        outputs=["left"],
    )
    add_handler = FakeHandler(AddOperator, "fake.add")

    _assert_category("malformed_derivative_ir", lambda: _lower(graph, _registry(add_handler)))
    assert add_handler.calls == []


def test_an_unknown_selected_output_raises_invalid_selected_output():
    graph = _binary_graph()

    _assert_category(
        "invalid_selected_output",
        lambda: _lower(graph, _full_registry(), outputs=["nowhere"]),
    )


def test_two_operations_producing_one_value_raise_ambiguous_producer():
    graph = TensorGraph(
        nodes=[
            _node("n0", "out", AddOperator(), ["alpha"], _matrix()),
            _node("n1", "out", MulOperator(), ["alpha"], _matrix()),
        ],
        inputs=[("alpha", _matrix())],
        outputs=["out"],
    )
    add_handler = FakeHandler(AddOperator, "fake.add")

    _assert_category("ambiguous_producer", lambda: _lower(graph, _registry(add_handler)))
    assert add_handler.calls == []


def test_a_reachable_value_with_no_producer_raises_missing_dependency():
    graph = TensorGraph(
        nodes=[_node("n0", "out", AddOperator(), ["alpha", "ghost"], _matrix())],
        inputs=[("alpha", _matrix())],
        outputs=["out"],
    )

    _assert_category("missing_dependency", lambda: _lower(graph, _full_registry()))


def test_handler_contract_violation_is_a_known_autodiff_error_category():
    assert "handler_contract_violation" in AUTODIFF_ERROR_CATEGORIES


# --------------------------------------------------------------------------
# derivative programs use the same seam
# --------------------------------------------------------------------------


def test_a_derivative_program_lowers_with_seed_and_forward_capture_bindings():
    trace, _images, weights, _labels, loss = _trace_linear_mse()
    graph = trace.build(outputs=loss)
    program = trace.vjp(loss, wrt=[weights], seed="seed")

    lowered = lower_derivative_program(
        program,
        forward_graph=graph,
        seed_value_ids=("seed",),
        handlers=_full_registry(),
        bind_input=_bind_input,
    )

    provenance = {
        dependency.value_id: dependency.provenance
        for dependency in lowered.dependencies.required_inputs
    }
    assert provenance["seed"] == DEPENDENCY_PROVENANCE_SEED_INPUT
    assert DEPENDENCY_PROVENANCE_FORWARD_CAPTURE in provenance.values()
    assert all(value_id in lowered.values for value_id in provenance)
    assert lowered.selected_outputs == tuple(program.output_gradients)
    assert len(lowered.output_values) == len(program.output_gradients)


def test_a_derivative_program_operation_without_a_handler_raises_unsupported_operator():
    trace, _images, weights, _labels, loss = _trace_linear_mse()
    graph = trace.build(outputs=loss)
    program = trace.vjp(loss, wrt=[weights], seed="seed")

    _assert_category(
        "unsupported_operator",
        lambda: lower_derivative_program(
            program,
            forward_graph=graph,
            seed_value_ids=("seed",),
            handlers=_registry(FakeHandler(AddOperator, "fake.add")),
            bind_input=_bind_input,
        ),
    )


def test_every_derivative_operation_is_claimed_exactly_once():
    trace, _images, weights, _labels, loss = _trace_linear_mse()
    graph = trace.build(outputs=loss)
    program = trace.vjp(loss, wrt=[weights], seed="seed")

    lowered = lower_derivative_program(
        program,
        forward_graph=graph,
        seed_value_ids=("seed",),
        handlers=_full_registry(),
        bind_input=_bind_input,
    )

    claimed = [
        node_id
        for operation in lowered.operations
        for node_id in operation.source_node_ids
    ]
    assert len(claimed) == len(set(claimed))
    assert set(claimed) <= {node.node_id for node in program.nodes}
    assert isinstance(lowered.operations[0], LoweredOperation)
