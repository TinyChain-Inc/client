"""Fail-closed and equivalence proof for the matmul-based mean expansion.

These are the two properties the whole expansion exists to deliver, and both
are only observable through a real backend registry:

* **Fail closed.** Every construct the expansion introduces is a distinct
  concrete operator type, so a backend without a
  handler for it is rejected by the lowering pre-flight *before any handler
  runs*. The error alone does not establish that; the proof is a recording
  registry that stayed empty, and an error message that names the offending
  node rather than only the operator type.
* **Equivalence.** The expanded artifact lowered by the limited-operation
  registry computes the same
  dtype, the same shape *including rank*, and values within tolerance of the
  unexpanded artifact lowered by the reduction-capable control registry. The
  control that the unexpanded artifacts *fail* against the limited-operation
  registry is what makes those comparisons mean anything: without it they would
  not show that the expansion is what enabled the lowering.

Both registries and the single dense-array execution semantics behind them come
from `tests.autodiff_reference_consumer` and `tests.autodiff_execution`; nothing
here defines a second reference backend or a second operator semantics. The
registries composed below are assembled out of those handlers -- never out of
new ones -- so that "a registry lacking exactly one handler" is expressible.

Nothing here asserts that differentiating an expanded graph must fail. Reverse
traversal skips a node whose inputs contain no requested differentiation target
before looking up a rule, so a zero-operand constant provides no such guarantee
and the source graph remains the canonical differentiation input.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tinychain.autodiff import (
    AutodiffError,
    DEPENDENCY_PROVENANCE_ORDER,
    FillOperator,
    MatmulOperator,
    OperationContext,
    OperationHandler,
    OperationHandlerRegistry,
    ReshapeOperator,
    TensorGraph,
    TensorGraphBuilder,
    TensorOperator,
    analyze_derivative_dependencies,
    analyze_graph_dependencies,
    expand_mean_derivative_program,
    expand_mean_graph,
    generate,
    lower_derivative_program,
    lower_graph,
)

from tests.autodiff_reference_consumer import (
    limited_operation_registry,
    recording_registry,
    reduction_capable_registry,
)

# Values are compared within a relative tolerance; dtype and shape compare
# exactly.
_RELATIVE_TOLERANCE = {"f32": 1e-6, "f64": 1e-12}
_NUMPY_DTYPE = {"f32": np.float32, "f64": np.float64}

_SHAPES = [(1, 1), (3, 5), (5, 3)]


# --------------------------------------------------------------------------
# fixtures: traced artifacts, expanded and not
# --------------------------------------------------------------------------


def _traced_mean(*, shape=(3, 5), dtype="f64", keepdims=True):
    """Trace `value.mean([0, 1], keepdims=...)` and generate its derivative."""
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype=dtype, shape=shape)
        output = value.mean([0, 1], keepdims=keepdims)
    graph = trace.build(outputs=output)
    program = generate(graph, graph.outputs[0], [graph.inputs[0][0]], "seed")
    return graph, program


def _traced_mean_with_a_forward_capture(*, shape=(3, 5), dtype="f64", keepdims=True):
    """Trace a mean whose derivative genuinely reads a forward intermediate.

    `(v + v) * (v + v)` makes the reverse transform capture the intermediate
    rather than only the declared input, so the forward-capture comparison of
    the forward-capture comparison uses a non-empty set.
    """
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype=dtype, shape=shape)
        doubled = value + value
        output = (doubled * doubled).mean([0, 1], keepdims=keepdims)
    graph = trace.build(outputs=output)
    program = generate(graph, graph.outputs[0], [graph.inputs[0][0]], "seed")
    return graph, program


def _operand(shape, dtype):
    """A non-uniform operand, so a wrong reduction cannot match by symmetry."""
    count = int(np.prod(shape))
    values = np.linspace(-1.5, 2.5, count, dtype=_NUMPY_DTYPE[dtype])
    return values.reshape(shape)


def _seed(*, keepdims, dtype):
    numpy_dtype = _NUMPY_DTYPE[dtype]
    return np.array([[2.5]], dtype=numpy_dtype) if keepdims else np.array(2.5, dtype=numpy_dtype)


# --------------------------------------------------------------------------
# registry composition over the shared reference consumer's handlers
# --------------------------------------------------------------------------


def _rebuild(*sources: OperationHandlerRegistry, without: type[TensorOperator] | None = None):
    """Compose one registry out of the handlers *sources* already registered.

    Earlier sources win, a handler for *without* is dropped, and no handler is
    defined here -- every one of them is a `tests.autodiff_reference_consumer`
    handler delegating to the single dense-array execution semantics.
    """
    composed = OperationHandlerRegistry()
    seen: set[type[TensorOperator]] = set()
    for source in sources:
        for operator_type in source.supported_types():
            if operator_type is without or operator_type in seen:
                continue
            seen.add(operator_type)
            composed.register(source.lookup(operator_type()))
    return composed


def _limited(*, reshape: bool) -> OperationHandlerRegistry:
    return limited_operation_registry(include_trivial_reshape=reshape)


def _control(*, reshape: bool) -> OperationHandlerRegistry:
    """The reduction-capable control, with the trivial reshape handler on demand."""
    if not reshape:
        return reduction_capable_registry()
    return _rebuild(reduction_capable_registry(), _limited(reshape=True))


class _TwoOperandMatmulHandler:
    """A matmul handler that asserts the two-operand contract itself."""

    operator_type = MatmulOperator

    def __init__(self) -> None:
        self.operand_counts: list[int] = []

    def lower(self, context: OperationContext) -> object:
        operands = list(context.inputs)
        assert len(operands) == 2, (
            f"matmul node {context.node_id!r} received {len(operands)} operands, not 2"
        )
        self.operand_counts.append(len(operands))
        return np.matmul(np.asarray(operands[0]), np.asarray(operands[1]))


def _registry_with(handler: OperationHandler, base: OperationHandlerRegistry):
    replaced = _rebuild(base, without=type(handler).operator_type)
    replaced.register(handler)
    return replaced


# --------------------------------------------------------------------------
# lowering helpers
# --------------------------------------------------------------------------


def _lower_forward(graph: TensorGraph, registry, values):
    lowered = lower_graph(
        graph,
        handlers=registry,
        bind_input=lambda dependency: values[dependency.value_id],
    )
    (result,) = lowered.output_values
    return np.asarray(result)


def _lower_gradient(program, *, forward_graph, registry, values, wrt="v0"):
    lowered = lower_derivative_program(
        program,
        forward_graph=forward_graph,
        seed_value_ids=("seed",),
        handlers=registry,
        bind_input=lambda dependency: values[dependency.value_id],
    )
    return np.asarray(lowered.values[program.gradients[wrt]])


def _assert_equivalent(expanded, control, *, dtype):
    """Compare dtype and shape exactly, and values within tolerance."""
    assert expanded.dtype == control.dtype
    assert expanded.shape == control.shape
    np.testing.assert_allclose(expanded, control, rtol=_RELATIVE_TOLERANCE[dtype])


# ==========================================================================
# Fail closed before any handler runs
# ==========================================================================


@pytest.mark.parametrize("keepdims", [True, False])
def test_expanded_graph_without_a_fill_handler_fails_closed_naming_the_fill_node(keepdims):
    graph, _program = _traced_mean(keepdims=keepdims)
    expanded = expand_mean_graph(graph)
    fill_node_ids = [
        node.node_id for node in expanded.nodes if isinstance(node.operator, FillOperator)
    ]
    assert fill_node_ids, "the expanded graph must contain a fill node"
    unaware = recording_registry(_rebuild(_limited(reshape=True), without=FillOperator))
    # The empty record below only proves something if the registry could have
    # recorded: every other emitted operator type does have a handler.
    assert unaware.registry.has_handler(MatmulOperator())

    with pytest.raises(AutodiffError) as excinfo:
        lower_graph(
            graph=expanded,
            handlers=unaware.registry,
            bind_input=lambda dependency: _operand((3, 5), "f64"),
        )

    assert excinfo.value.category == "unsupported_operator"
    assert "FillOperator" in excinfo.value.message
    assert fill_node_ids[0] in excinfo.value.message
    assert unaware.invocations == []


def test_rank_reducing_expanded_graph_without_a_reshape_handler_fails_closed_naming_the_node():
    graph, _program = _traced_mean(keepdims=False)
    expanded = expand_mean_graph(graph)
    reshape_node_ids = [
        node.node_id for node in expanded.nodes if isinstance(node.operator, ReshapeOperator)
    ]
    assert reshape_node_ids, "the rank-reducing tier must emit a reshape node"
    unaware = recording_registry(_limited(reshape=False))
    assert unaware.registry.has_handler(FillOperator())
    assert unaware.registry.has_handler(MatmulOperator())

    with pytest.raises(AutodiffError) as excinfo:
        lower_graph(
            graph=expanded,
            handlers=unaware.registry,
            bind_input=lambda dependency: _operand((3, 5), "f64"),
        )

    assert excinfo.value.category == "unsupported_operator"
    assert "ReshapeOperator" in excinfo.value.message
    assert reshape_node_ids[0] in excinfo.value.message
    assert unaware.invocations == []


def test_rank_reducing_expanded_derivative_program_without_a_reshape_handler_fails_closed():
    """The documented most-likely omission, on the artifact it is easiest to miss.

    Section 9.3 says the rank-reducing tier needs a trivial-reshape handler in
    *both* artifacts. On the gradient path the reshape is not emitted by the
    expansion at all -- it is the seed reshape the VJP already produced, which
    the broadcast-and-scale rewrite leaves in place -- so a backend that added a
    fill handler and stopped is rejected by that surviving node.
    """
    graph, program = _traced_mean(keepdims=False)
    expanded = expand_mean_derivative_program(program)
    reshape_node_ids = [
        node.node_id for node in expanded.nodes if isinstance(node.operator, ReshapeOperator)
    ]
    assert reshape_node_ids, "the rank-reducing gradient path must retain a reshape node"
    unaware = recording_registry(_limited(reshape=False))
    assert unaware.registry.has_handler(FillOperator())
    assert unaware.registry.has_handler(MatmulOperator())
    values = {"seed": _seed(keepdims=False, dtype="f64"), "v0": _operand((3, 5), "f64")}

    with pytest.raises(AutodiffError) as excinfo:
        _lower_gradient(
            expanded, forward_graph=graph, registry=unaware.registry, values=values
        )

    assert excinfo.value.category == "unsupported_operator"
    assert "ReshapeOperator" in excinfo.value.message
    assert reshape_node_ids[0] in excinfo.value.message
    assert unaware.invocations == []


def test_rank_reducing_expanded_derivative_program_lowers_once_a_reshape_handler_exists():
    """The positive control: the same artifact and registry, reshape handler added."""
    graph, program = _traced_mean(keepdims=False)
    expanded = expand_mean_derivative_program(program)
    values = {"seed": _seed(keepdims=False, dtype="f64"), "v0": _operand((3, 5), "f64")}

    gradient = _lower_gradient(
        expanded, forward_graph=graph, registry=_limited(reshape=True), values=values
    )

    np.testing.assert_allclose(gradient, np.full((3, 5), 2.5 / 15.0), rtol=1e-12)


def test_registry_lookup_message_names_only_the_operator_type():
    """`lookup`'s own message stays byte-identical; only the pre-flight enriches it."""
    with pytest.raises(AutodiffError) as excinfo:
        limited_operation_registry().lookup(ReshapeOperator())

    assert excinfo.value.category == "unsupported_operator"
    assert excinfo.value.message == (
        "no lowering handler registered for operator type 'ReshapeOperator'"
    )


# ==========================================================================
# The two-operand matmul contract still holds
# ==========================================================================


@pytest.mark.parametrize("keepdims", [True, False])
def test_a_two_operand_asserting_matmul_handler_lowers_an_expanded_graph(keepdims):
    graph, _program = _traced_mean(keepdims=keepdims)
    expanded = expand_mean_graph(graph)
    handler = _TwoOperandMatmulHandler()
    registry = _registry_with(handler, _limited(reshape=True))
    operand = _operand((3, 5), "f64")

    result = _lower_forward(expanded, registry, {"v0": operand})

    assert handler.operand_counts == [2, 2]
    np.testing.assert_allclose(np.asarray(result).reshape(()), operand.mean(), rtol=1e-12)


@pytest.mark.parametrize("keepdims", [True, False])
def test_a_two_operand_asserting_matmul_handler_lowers_an_expanded_derivative_program(keepdims):
    graph, program = _traced_mean(keepdims=keepdims)
    expanded = expand_mean_derivative_program(program)
    handler = _TwoOperandMatmulHandler()
    registry = _registry_with(handler, _limited(reshape=True))
    values = {"seed": _seed(keepdims=keepdims, dtype="f64"), "v0": _operand((3, 5), "f64")}

    gradient = _lower_gradient(
        expanded, forward_graph=graph, registry=registry, values=values
    )

    assert handler.operand_counts == [2, 2]
    np.testing.assert_allclose(gradient, np.full((3, 5), 2.5 / 15.0), rtol=1e-12)


# ==========================================================================
# Control: the unexpanded artifacts do not lower
# ==========================================================================


@pytest.mark.parametrize("keepdims", [True, False])
def test_unexpanded_forward_graph_is_rejected_by_the_limited_operation_registry(keepdims):
    graph, _program = _traced_mean(keepdims=keepdims)

    with pytest.raises(AutodiffError) as excinfo:
        lower_graph(
            graph,
            handlers=_limited(reshape=True),
            bind_input=lambda dependency: _operand((3, 5), "f64"),
        )

    assert excinfo.value.category == "unsupported_operator"


@pytest.mark.parametrize("keepdims", [True, False])
def test_unexpanded_derivative_program_is_rejected_by_the_limited_operation_registry(keepdims):
    graph, program = _traced_mean(keepdims=keepdims)
    values = {"seed": _seed(keepdims=keepdims, dtype="f64"), "v0": _operand((3, 5), "f64")}

    with pytest.raises(AutodiffError) as excinfo:
        _lower_gradient(
            program,
            forward_graph=graph,
            registry=_limited(reshape=True),
            values=values,
        )

    assert excinfo.value.category == "unsupported_operator"


# ==========================================================================
# Expanded and control artifacts are numerically equivalent
# ==========================================================================


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", ["f32", "f64"])
@pytest.mark.parametrize("keepdims", [True, False])
def test_expanded_forward_result_equals_the_control_result(keepdims, dtype, shape):
    graph, _program = _traced_mean(shape=shape, dtype=dtype, keepdims=keepdims)
    expanded = expand_mean_graph(graph)
    values = {"v0": _operand(shape, dtype)}

    control = _lower_forward(graph, _control(reshape=True), values)
    result = _lower_forward(expanded, _limited(reshape=True), values)

    _assert_equivalent(result, control, dtype=dtype)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", ["f32", "f64"])
@pytest.mark.parametrize("keepdims", [True, False])
def test_expanded_gradient_equals_the_control_gradient(keepdims, dtype, shape):
    graph, program = _traced_mean(shape=shape, dtype=dtype, keepdims=keepdims)
    expanded = expand_mean_derivative_program(program)
    values = {"seed": _seed(keepdims=keepdims, dtype=dtype), "v0": _operand(shape, dtype)}

    control = _lower_gradient(
        program, forward_graph=graph, registry=_control(reshape=True), values=values
    )
    result = _lower_gradient(
        expanded, forward_graph=graph, registry=_limited(reshape=True), values=values
    )

    _assert_equivalent(result, control, dtype=dtype)


# ==========================================================================
# A scalar and a [1, 1] value are never interchangeable
# ==========================================================================


def _assert_rank_preserving_shape(result) -> None:
    assert result.shape == (1, 1)


def _assert_rank_reducing_shape(result) -> None:
    assert result.shape == ()


def test_the_two_tiers_produce_shapes_that_never_satisfy_each_others_assertion():
    values = {"v0": _operand((3, 5), "f64")}
    rank_preserving_graph, _ = _traced_mean(keepdims=True)
    rank_reducing_graph, _ = _traced_mean(keepdims=False)
    registry = _limited(reshape=True)

    rank_preserving = _lower_forward(expand_mean_graph(rank_preserving_graph), registry, values)
    rank_reducing = _lower_forward(expand_mean_graph(rank_reducing_graph), registry, values)

    _assert_rank_preserving_shape(rank_preserving)
    _assert_rank_reducing_shape(rank_reducing)
    with pytest.raises(AssertionError):
        _assert_rank_reducing_shape(rank_preserving)
    with pytest.raises(AssertionError):
        _assert_rank_preserving_shape(rank_reducing)


# ==========================================================================
# Dependency analysis accepts the expanded artifacts unchanged
# ==========================================================================


def test_dependency_analysis_entry_points_gained_no_argument():
    assert list(inspect.signature(analyze_graph_dependencies).parameters) == [
        "graph",
        "outputs",
    ]
    assert list(inspect.signature(analyze_derivative_dependencies).parameters) == [
        "program",
        "forward_graph",
        "seed_value_ids",
        "outputs",
    ]


@pytest.mark.parametrize("keepdims", [True, False])
def test_analyze_graph_dependencies_accepts_the_expanded_graph(keepdims):
    graph, _program = _traced_mean(keepdims=keepdims)
    expanded = expand_mean_graph(graph)

    analysis = analyze_graph_dependencies(expanded)

    assert analysis.selected_outputs == tuple(expanded.outputs)
    assert {dependency.provenance for dependency in analysis.dependencies} <= set(
        DEPENDENCY_PROVENANCE_ORDER
    )


@pytest.mark.parametrize("keepdims", [True, False])
def test_analyze_derivative_dependencies_accepts_the_expanded_program(keepdims):
    graph, program = _traced_mean_with_a_forward_capture(keepdims=keepdims)
    expanded_program = expand_mean_derivative_program(program)

    analysis = analyze_derivative_dependencies(
        expanded_program, forward_graph=graph, seed_value_ids=("seed",)
    )

    assert analysis.selected_outputs == tuple(expanded_program.output_gradients)
    assert {dependency.provenance for dependency in analysis.dependencies} <= set(
        DEPENDENCY_PROVENANCE_ORDER
    )


@pytest.mark.parametrize("keepdims", [True, False])
def test_forward_captures_are_the_same_for_the_original_and_the_expanded_forward_graph(keepdims):
    graph, program = _traced_mean_with_a_forward_capture(keepdims=keepdims)
    expanded_graph = expand_mean_graph(graph)

    def captures(forward_graph):
        analysis = analyze_derivative_dependencies(
            program, forward_graph=forward_graph, seed_value_ids=("seed",)
        )
        return {
            dependency.value_id
            for dependency in analysis.dependencies
            if dependency.provenance == "forward_capture"
        }

    original_captures = captures(graph)
    assert original_captures, "the fixture must produce at least one forward capture"
    assert captures(expanded_graph) == original_captures
