"""Unit tests for framework-owned structured dependency analysis.

These tests pin the contract that a consumer can bind every derivative input
from provenance alone: without scanning graph nodes, without a private producer
map, and without any convention about how value ids are spelled.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import tinychain as tc
from tinychain.autodiff import (
    AddOperator,
    AutodiffError,
    DerivativeMetadata,
    DerivativeProgram,
    MulOperator,
    TensorGraph,
    TensorGraphBuilder,
    TensorNodeRecord,
    compile_derivative_program,
)
from tinychain.autodiff.dependencies import (
    DEPENDENCY_PROVENANCE_DECLARED_INPUT,
    DEPENDENCY_PROVENANCE_FORWARD_CAPTURE,
    DEPENDENCY_PROVENANCE_LOCAL_VALUE,
    DEPENDENCY_PROVENANCE_ORDER,
    DEPENDENCY_PROVENANCE_SEED_INPUT,
    DependencyAnalysis,
    ValueDependency,
    analyze_derivative_dependencies,
    analyze_graph_dependencies,
)
from tinychain.autodiff.dependencies import _order_reachable_nodes


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _typespec(dtype: str, shape: list[int]) -> dict[str, object]:
    return {"dtype": dtype, "shape": list(shape)}


def _trace_linear_mse(dtype: str = "f32"):
    """Trace a small linear-regression loss and return the builder and values."""
    trace = TensorGraphBuilder()
    with tc.state.scoped_context():
        with trace:
            images = trace.input("images", dtype=dtype, shape=(2, 3))
            weights = trace.input("weights", dtype=dtype, shape=(3, 4))
            labels = trace.input("labels", dtype=dtype, shape=(2, 4))
            residual = (images @ weights) - labels
            loss = (residual * residual).mean([0, 1])
    return trace, images, weights, labels, loss


def _provenance_map(analysis: DependencyAnalysis) -> dict[str, str]:
    return {
        dependency.value_id: dependency.provenance
        for dependency in analysis.dependencies
    }


def _node(node_id: str, output: str, inputs: list[str], typespec: object) -> TensorNodeRecord:
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output,
        operator=AddOperator() if len(inputs) == 2 else MulOperator(),
        op_params={},
        input_value_ids=list(inputs),
        output_typespec=typespec,
    )


def _derivative_program(nodes: list[TensorNodeRecord], gradients: dict[str, str]) -> DerivativeProgram:
    return DerivativeProgram(
        nodes=list(nodes),
        gradients=dict(gradients),
        output_gradients=list(gradients.values()),
        metadata=DerivativeMetadata(
            source_graph_id="test-graph",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=tuple(gradients),
            seed_contract="seed matches out",
        ),
        value_typespecs={},
    )


def _assert_category(category: str, operation) -> None:
    with pytest.raises(AutodiffError) as error:
        operation()
    assert error.value.category == category


# --------------------------------------------------------------------------
# AC1 — consumers can bind every derivative input from provenance alone
# --------------------------------------------------------------------------


def test_derivative_analysis_classifies_declared_seed_and_capture_inputs():
    trace, images, weights, _labels, loss = _trace_linear_mse()
    graph = trace.build(outputs=loss)
    program = trace.vjp(loss, wrt=[weights], seed="seed")

    analysis = analyze_derivative_dependencies(
        program, forward_graph=graph, seed_value_ids=("seed",)
    )

    provenance = _provenance_map(analysis)
    images_id = graph.inputs[0][0]
    assert provenance[images_id] == DEPENDENCY_PROVENANCE_DECLARED_INPUT
    assert provenance["seed"] == DEPENDENCY_PROVENANCE_SEED_INPUT

    captures = [dependency.value_id for dependency in analysis.forward_captures]
    assert captures, "the squared-residual gradient must capture a forward intermediate"
    forward_outputs = {node.output_value_id for node in graph.nodes}
    assert set(captures) <= forward_outputs

    locals_ = {dependency.value_id for dependency in analysis.local_values}
    assert locals_ == {node.output_value_id for node in program.nodes}


def test_required_inputs_match_the_compiled_route_parameters():
    trace, _images, weights, _labels, loss = _trace_linear_mse()
    graph = trace.build(outputs=loss)
    program = trace.vjp(loss, wrt=[weights], seed="seed")

    analysis = analyze_derivative_dependencies(
        program, forward_graph=graph, seed_value_ids=("seed",)
    )
    compiled = compile_derivative_program(program)

    required = [dependency.value_id for dependency in analysis.required_inputs]
    assert set(required) == set(compiled.params)
    assert len(required) == len(set(required))


def test_every_dependency_carries_normalized_dtype_and_shape():
    trace, _images, weights, _labels, loss = _trace_linear_mse()
    graph = trace.build(outputs=loss)
    program = trace.vjp(loss, wrt=[weights], seed="seed")

    analysis = analyze_derivative_dependencies(
        program, forward_graph=graph, seed_value_ids=("seed",)
    )

    for dependency in analysis.dependencies:
        assert isinstance(dependency, ValueDependency)
        assert dependency.dtype == "f32"
        assert isinstance(dependency.shape, tuple)

    by_id = {dependency.value_id: dependency for dependency in analysis.dependencies}
    assert by_id[graph.inputs[0][0]].shape == (2, 3)
    assert by_id["seed"].shape == ()


def test_forward_graph_analysis_reports_declared_inputs_and_local_values():
    trace, _images, _weights, _labels, loss = _trace_linear_mse()
    graph = trace.build(outputs=loss)

    analysis = analyze_graph_dependencies(graph)

    assert analysis.selected_outputs == tuple(graph.outputs)
    assert [dependency.value_id for dependency in analysis.declared_inputs] == [
        value_id for value_id, _typespec in graph.inputs
    ]
    assert [dependency.value_id for dependency in analysis.local_values] == [
        node.output_value_id for node in graph.nodes
    ]
    assert analysis.seed_inputs == ()
    assert analysis.forward_captures == ()


# --------------------------------------------------------------------------
# AC2 — deterministic ordering across repeated and equivalent analyses
# --------------------------------------------------------------------------


def test_repeated_analysis_of_the_same_program_is_identical():
    trace, _images, weights, _labels, loss = _trace_linear_mse()
    graph = trace.build(outputs=loss)
    program = trace.vjp(loss, wrt=[weights], seed="seed")

    first = analyze_derivative_dependencies(
        program, forward_graph=graph, seed_value_ids=("seed",)
    )
    second = analyze_derivative_dependencies(
        program, forward_graph=graph, seed_value_ids=("seed",)
    )

    assert first == second
    assert first.dependencies == second.dependencies


def test_equivalent_traces_produce_identical_ordered_results():
    first_trace, _images, first_weights, _labels, first_loss = _trace_linear_mse()
    first_graph = first_trace.build(outputs=first_loss)
    first_program = first_trace.vjp(first_loss, wrt=[first_weights], seed="seed")

    second_trace, _images2, second_weights, _labels2, second_loss = _trace_linear_mse()
    second_graph = second_trace.build(outputs=second_loss)
    second_program = second_trace.vjp(second_loss, wrt=[second_weights], seed="seed")

    first = analyze_derivative_dependencies(
        first_program, forward_graph=first_graph, seed_value_ids=("seed",)
    )
    second = analyze_derivative_dependencies(
        second_program, forward_graph=second_graph, seed_value_ids=("seed",)
    )

    assert first == second


def test_dependencies_are_grouped_by_documented_provenance_order():
    trace, _images, weights, _labels, loss = _trace_linear_mse()
    graph = trace.build(outputs=loss)
    program = trace.vjp(loss, wrt=[weights], seed="seed")

    analysis = analyze_derivative_dependencies(
        program, forward_graph=graph, seed_value_ids=("seed",)
    )

    assert DEPENDENCY_PROVENANCE_ORDER == (
        DEPENDENCY_PROVENANCE_DECLARED_INPUT,
        DEPENDENCY_PROVENANCE_SEED_INPUT,
        DEPENDENCY_PROVENANCE_FORWARD_CAPTURE,
        DEPENDENCY_PROVENANCE_LOCAL_VALUE,
    )
    ranks = [
        DEPENDENCY_PROVENANCE_ORDER.index(dependency.provenance)
        for dependency in analysis.dependencies
    ]
    assert ranks == sorted(ranks)
    assert analysis.dependencies == (
        *analysis.declared_inputs,
        *analysis.seed_inputs,
        *analysis.forward_captures,
        *analysis.local_values,
    )


def test_local_values_follow_program_topological_order():
    nodes = [
        _node("n1", "second", ["first"], _typespec("f32", [2])),
        _node("n0", "first", ["alpha"], _typespec("f32", [2])),
    ]
    graph = TensorGraph(
        nodes=nodes,
        inputs=[("alpha", _typespec("f32", [2]))],
        outputs=["second"],
    )

    analysis = analyze_graph_dependencies(graph)

    assert [dependency.value_id for dependency in analysis.local_values] == [
        "first",
        "second",
    ]


def test_analysis_does_not_mutate_the_analyzed_graph():
    trace, _images, _weights, _labels, loss = _trace_linear_mse()
    graph = trace.build(outputs=loss)
    before = (list(graph.inputs), list(graph.outputs), [node.node_id for node in graph.nodes])

    analyze_graph_dependencies(graph)

    assert (list(graph.inputs), list(graph.outputs), [node.node_id for node in graph.nodes]) == before


# --------------------------------------------------------------------------
# AC3 — no id-spelling, prefix convention, or private node map required
# --------------------------------------------------------------------------


def test_classification_ignores_value_id_spelling():
    forward_nodes = [
        _node("fn0", "zzz_intermediate", ["parameter"], _typespec("f32", [2])),
    ]
    forward_graph = TensorGraph(
        nodes=forward_nodes,
        inputs=[("parameter", _typespec("f32", [2]))],
        outputs=["zzz_intermediate"],
    )
    program = _derivative_program(
        nodes=[
            _node("an0", "aaa_local", ["cotangent", "zzz_intermediate"], _typespec("f32", [2])),
        ],
        gradients={"parameter": "aaa_local"},
    )

    analysis = analyze_derivative_dependencies(
        program, forward_graph=forward_graph, seed_value_ids=("cotangent",)
    )

    assert _provenance_map(analysis) == {
        "cotangent": DEPENDENCY_PROVENANCE_SEED_INPUT,
        "zzz_intermediate": DEPENDENCY_PROVENANCE_FORWARD_CAPTURE,
        "aaa_local": DEPENDENCY_PROVENANCE_LOCAL_VALUE,
    }


def test_only_nodes_reachable_from_selected_outputs_are_analyzed():
    nodes = [
        _node("n0", "reachable", ["alpha"], _typespec("f32", [2])),
        _node("n1", "unreachable", ["ghost"], _typespec("f32", [2])),
    ]
    graph = TensorGraph(
        nodes=nodes,
        inputs=[("alpha", _typespec("f32", [2]))],
        outputs=["reachable", "unreachable"],
    )

    analysis = analyze_graph_dependencies(graph, outputs=["reachable"])

    assert analysis.selected_outputs == ("reachable",)
    assert set(_provenance_map(analysis)) == {"alpha", "reachable"}


# --------------------------------------------------------------------------
# AC4 — malformed cases fail with categorized AutodiffError values
# --------------------------------------------------------------------------


def test_missing_producer_raises_missing_dependency():
    graph = TensorGraph(
        nodes=[_node("n0", "out", ["ghost"], _typespec("f32", [2]))],
        inputs=[],
        outputs=["out"],
    )

    _assert_category("missing_dependency", lambda: analyze_graph_dependencies(graph))


def test_derivative_free_input_outside_forward_graph_and_seeds_raises_missing_dependency():
    forward_graph = TensorGraph(
        nodes=[],
        inputs=[("parameter", _typespec("f32", [2]))],
        outputs=["parameter"],
    )
    program = _derivative_program(
        nodes=[_node("an0", "local", ["ghost"], _typespec("f32", [2]))],
        gradients={"parameter": "local"},
    )

    _assert_category(
        "missing_dependency",
        lambda: analyze_derivative_dependencies(
            program, forward_graph=forward_graph, seed_value_ids=("cotangent",)
        ),
    )


def test_duplicate_declared_inputs_raise_ambiguous_producer():
    graph = TensorGraph(
        nodes=[_node("n0", "out", ["alpha"], _typespec("f32", [2]))],
        inputs=[("alpha", _typespec("f32", [2])), ("alpha", _typespec("f32", [9]))],
        outputs=["out"],
    )

    with pytest.raises(AutodiffError) as error:
        analyze_graph_dependencies(graph)

    assert error.value.category == "ambiguous_producer"
    assert "alpha" in error.value.message


def test_duplicate_declared_inputs_in_forward_graph_raise_ambiguous_producer():
    forward_graph = TensorGraph(
        nodes=[],
        inputs=[("parameter", _typespec("f32", [2])), ("parameter", _typespec("f32", [9]))],
        outputs=["parameter"],
    )
    program = _derivative_program(
        nodes=[_node("an0", "local", ["cotangent", "parameter"], _typespec("f32", [2]))],
        gradients={"parameter": "local"},
    )

    with pytest.raises(AutodiffError) as error:
        analyze_derivative_dependencies(
            program, forward_graph=forward_graph, seed_value_ids=("cotangent",)
        )

    assert error.value.category == "ambiguous_producer"
    assert "parameter" in error.value.message


def test_declared_input_that_is_also_produced_raises_ambiguous_producer():
    graph = TensorGraph(
        nodes=[
            _node("n0", "alpha", ["beta"], _typespec("f32", [2])),
            _node("n1", "out", ["alpha"], _typespec("f32", [2])),
        ],
        inputs=[("alpha", _typespec("f32", [9])), ("beta", _typespec("f32", [2]))],
        outputs=["out"],
    )

    with pytest.raises(AutodiffError) as error:
        analyze_graph_dependencies(graph)

    assert error.value.category == "ambiguous_producer"
    assert "alpha" in error.value.message
    assert DEPENDENCY_PROVENANCE_DECLARED_INPUT in error.value.message
    assert DEPENDENCY_PROVENANCE_LOCAL_VALUE in error.value.message


def test_forward_declared_input_that_is_also_produced_raises_ambiguous_producer():
    forward_graph = TensorGraph(
        nodes=[_node("f0", "alpha", ["beta"], _typespec("f32", [2]))],
        inputs=[("alpha", _typespec("f32", [9])), ("beta", _typespec("f32", [2]))],
        outputs=["alpha"],
    )
    program = _derivative_program(
        nodes=[_node("an0", "local", ["cotangent", "alpha"], _typespec("f32", [2]))],
        gradients={"beta": "local"},
    )

    with pytest.raises(AutodiffError) as error:
        analyze_derivative_dependencies(
            program, forward_graph=forward_graph, seed_value_ids=("cotangent",)
        )

    assert error.value.category == "ambiguous_producer"
    assert "alpha" in error.value.message
    assert DEPENDENCY_PROVENANCE_DECLARED_INPUT in error.value.message
    assert DEPENDENCY_PROVENANCE_FORWARD_CAPTURE in error.value.message


def test_duplicate_producers_raise_ambiguous_producer():
    graph = TensorGraph(
        nodes=[
            _node("n0", "out", ["alpha"], _typespec("f32", [2])),
            _node("n1", "out", ["alpha"], _typespec("f32", [2])),
        ],
        inputs=[("alpha", _typespec("f32", [2]))],
        outputs=["out"],
    )

    _assert_category("ambiguous_producer", lambda: analyze_graph_dependencies(graph))


def test_seed_colliding_with_a_forward_value_raises_ambiguous_producer():
    forward_graph = TensorGraph(
        nodes=[],
        inputs=[("parameter", _typespec("f32", [2]))],
        outputs=["parameter"],
    )
    program = _derivative_program(
        nodes=[_node("an0", "local", ["parameter"], _typespec("f32", [2]))],
        gradients={"parameter": "local"},
    )

    _assert_category(
        "ambiguous_producer",
        lambda: analyze_derivative_dependencies(
            program, forward_graph=forward_graph, seed_value_ids=("parameter",)
        ),
    )


def test_seed_colliding_with_a_produced_forward_value_raises_ambiguous_producer():
    """Regression guard: the seed/produced half of the seed collision check.

    The sibling seed/declared case is covered elsewhere; this pins the case
    where the colliding forward value is produced by a node rather than
    declared as an input.
    """
    forward_graph = TensorGraph(
        nodes=[_node("f0", "mid", ["beta"], _typespec("f32", [2]))],
        inputs=[("beta", _typespec("f32", [2]))],
        outputs=["mid"],
    )
    program = _derivative_program(
        nodes=[_node("an0", "local", ["mid"], _typespec("f32", [2]))],
        gradients={"beta": "local"},
    )

    with pytest.raises(AutodiffError) as error:
        analyze_derivative_dependencies(
            program, forward_graph=forward_graph, seed_value_ids=("mid",)
        )

    assert error.value.category == "ambiguous_producer"
    assert "mid" in error.value.message


def test_cycle_raises_malformed_derivative_ir():
    graph = TensorGraph(
        nodes=[
            _node("n0", "left", ["right"], _typespec("f32", [2])),
            _node("n1", "right", ["left"], _typespec("f32", [2])),
        ],
        inputs=[],
        outputs=["left"],
    )

    _assert_category("malformed_derivative_ir", lambda: analyze_graph_dependencies(graph))


def test_cycle_within_the_derivative_program_raises_malformed_derivative_ir():
    """A cycle reachable from a selected output, entirely inside the derivative
    program itself, still fails -- this path never routed through the
    forward-graph topological sort and is unaffected by skipping it."""
    forward_graph = TensorGraph(
        nodes=[],
        inputs=[("parameter", _typespec("f32", [2]))],
        outputs=["parameter"],
    )
    program = _derivative_program(
        nodes=[
            _node("an0", "left", ["right"], _typespec("f32", [2])),
            _node("an1", "right", ["left"], _typespec("f32", [2])),
        ],
        gradients={"parameter": "left"},
    )

    _assert_category(
        "malformed_derivative_ir",
        lambda: analyze_derivative_dependencies(
            program, forward_graph=forward_graph, seed_value_ids=()
        ),
    )


def test_forward_graph_cycle_is_reported_when_the_selection_captures_any_forward_value():
    """A cycle in the forward graph still fails when the selection captures
    any forward value: capturing anything at all -- not just a value
    downstream of the cycle -- gates the whole-forward-graph walk back on,
    and that walk checks every node in the forward graph, so the cycle here
    (which happens to also produce the captured value) is found regardless
    of the exact relationship between the two."""
    forward_graph = TensorGraph(
        nodes=[
            _node("f0", "left", ["right"], _typespec("f32", [2])),
            _node("f1", "right", ["left"], _typespec("f32", [2])),
        ],
        inputs=[],
        outputs=["left"],
    )
    program = _derivative_program(
        nodes=[_node("an0", "local", ["left"], _typespec("f32", [2]))],
        gradients={"unused": "local"},
    )

    _assert_category(
        "malformed_derivative_ir",
        lambda: analyze_derivative_dependencies(
            program, forward_graph=forward_graph, seed_value_ids=()
        ),
    )


def test_forward_graph_cycle_is_not_reported_when_the_selection_captures_nothing():
    """A forward-graph cycle is not reported when the selection captures no
    forward value at all: the derivative selection here depends only on a
    declared forward input, so no forward capture exists, the whole-forward-
    graph topological sort is skipped by design, and the cycle inside it goes
    unreported as a deliberate consequence -- not because the cycle happens
    to be unreachable from what was captured. (When at least one forward
    value is captured, the whole forward graph is walked and any cycle in it
    is found, including one unrelated to what was captured -- see the
    forward-capture test above.)"""
    forward_graph = TensorGraph(
        nodes=[
            _node("f0", "left", ["right"], _typespec("f32", [2])),
            _node("f1", "right", ["left"], _typespec("f32", [2])),
        ],
        inputs=[("parameter", _typespec("f32", [2]))],
        outputs=["left"],
    )
    program = _derivative_program(
        nodes=[_node("an0", "local", ["parameter"], _typespec("f32", [2]))],
        gradients={"parameter": "local"},
    )

    analysis = analyze_derivative_dependencies(
        program, forward_graph=forward_graph, seed_value_ids=()
    )

    provenance = _provenance_map(analysis)
    assert provenance["parameter"] == DEPENDENCY_PROVENANCE_DECLARED_INPUT
    assert "left" not in provenance
    assert "right" not in provenance


def test_unknown_selected_output_raises_invalid_selected_output():
    graph = TensorGraph(
        nodes=[_node("n0", "out", ["alpha"], _typespec("f32", [2]))],
        inputs=[("alpha", _typespec("f32", [2]))],
        outputs=["out"],
    )

    _assert_category(
        "invalid_selected_output",
        lambda: analyze_graph_dependencies(graph, outputs=["nope"]),
    )


def test_empty_selected_outputs_raise_invalid_selected_output():
    graph = TensorGraph(
        nodes=[_node("n0", "out", ["alpha"], _typespec("f32", [2]))],
        inputs=[("alpha", _typespec("f32", [2]))],
        outputs=["out"],
    )

    _assert_category(
        "invalid_selected_output",
        lambda: analyze_graph_dependencies(graph, outputs=[]),
    )


def test_missing_dtype_metadata_raises_missing_dtype_metadata():
    graph = TensorGraph(
        nodes=[_node("n0", "out", ["alpha"], _typespec("f32", [2]))],
        inputs=[("alpha", None)],
        outputs=["out"],
    )

    _assert_category("missing_dtype_metadata", lambda: analyze_graph_dependencies(graph))


def test_missing_shape_metadata_raises_missing_shape_metadata():
    graph = TensorGraph(
        nodes=[_node("n0", "out", ["alpha"], _typespec("f32", [2]))],
        inputs=[("alpha", {"dtype": "f32"})],
        outputs=["out"],
    )

    _assert_category("missing_shape_metadata", lambda: analyze_graph_dependencies(graph))


def test_node_output_without_metadata_raises_missing_dtype_metadata():
    graph = TensorGraph(
        nodes=[_node("n0", "out", ["alpha"], None)],
        inputs=[("alpha", _typespec("f32", [2]))],
        outputs=["out"],
    )

    _assert_category("missing_dtype_metadata", lambda: analyze_graph_dependencies(graph))


# --------------------------------------------------------------------------
# AC4b — malformed seed identifier shapes fail with a categorized error, not
# a raw TypeError, matching the sibling selected-output guard's convention
# --------------------------------------------------------------------------


def _minimal_derivative_setup() -> tuple[TensorGraph, DerivativeProgram]:
    forward_graph = TensorGraph(
        nodes=[],
        inputs=[("parameter", _typespec("f32", [2]))],
        outputs=["parameter"],
    )
    program = _derivative_program(
        nodes=[_node("an0", "local", ["parameter"], _typespec("f32", [2]))],
        gradients={"parameter": "local"},
    )
    return forward_graph, program


@pytest.mark.parametrize(
    "malformed_seed_value_ids",
    [None, "not-a-sequence", [123], [""]],
    ids=["none", "bare-string", "non-string-element", "empty-string-element"],
)
def test_malformed_seed_value_ids_raise_invalid_selected_output(malformed_seed_value_ids):
    forward_graph, program = _minimal_derivative_setup()

    _assert_category(
        "invalid_selected_output",
        lambda: analyze_derivative_dependencies(
            program,
            forward_graph=forward_graph,
            seed_value_ids=malformed_seed_value_ids,
        ),
    )


# --------------------------------------------------------------------------
# AC — reachability over a deep chain does not exhaust the recursion limit
# --------------------------------------------------------------------------


def _deep_chain_graph(depth: int) -> TensorGraph:
    typespec = _typespec("f32", [2])
    nodes: list[TensorNodeRecord] = []
    previous = "v0"
    for index in range(1, depth + 1):
        current = f"v{index}"
        nodes.append(_node(f"n{index}", current, [previous], typespec))
        previous = current
    return TensorGraph(nodes=nodes, inputs=[("v0", typespec)], outputs=[previous])


def test_deep_forward_graph_analyzes_without_recursion_error():
    depth = 1200
    graph = _deep_chain_graph(depth)

    analysis = analyze_graph_dependencies(graph)

    assert [dependency.value_id for dependency in analysis.local_values] == [
        f"v{index}" for index in range(1, depth + 1)
    ]


# --------------------------------------------------------------------------
# an unrelated forward-graph cycle is still reported once anything is captured
# --------------------------------------------------------------------------


def test_an_unrelated_forward_graph_cycle_is_reported_when_a_forward_value_is_captured():
    """A forward-graph cycle with no relationship at all to the captured value
    is still reported. The sibling tests cover a cycle that itself produces the
    captured value; this one separates the two, because the rule is gated on
    whether the selection captures anything -- not on whether the cycle is
    reachable from what was captured. Without this, the surprising half of the
    rule is described in prose and constrained by nothing."""
    forward_graph = TensorGraph(
        nodes=[
            _node("f0", "left", ["right"], _typespec("f32", [2])),
            _node("f1", "right", ["left"], _typespec("f32", [2])),
            _node("f2", "h", ["p"], _typespec("f32", [2])),
        ],
        inputs=[("p", _typespec("f32", [2]))],
        outputs=["h"],
    )
    program = _derivative_program(
        nodes=[_node("an0", "local", ["h"], _typespec("f32", [2]))],
        gradients={"p": "local"},
    )

    _assert_category(
        "malformed_derivative_ir",
        lambda: analyze_derivative_dependencies(
            program, forward_graph=forward_graph, seed_value_ids=()
        ),
    )


# --------------------------------------------------------------------------
# emission order: each operation is scheduled as late as its consumers allow
# --------------------------------------------------------------------------


def _emission_order(graph: TensorGraph, outputs: list[str] | None = None) -> list[str]:
    """The value ids of the analyzed selection's own operations, in emission order."""
    analysis = analyze_graph_dependencies(graph, outputs=outputs)
    return [dependency.value_id for dependency in analysis.local_values]


def _independent_operand_graph() -> TensorGraph:
    """transpose(a) @ (b + b): the matmul's operands come from two independent chains."""
    typespec = _typespec("f32", [2, 2])
    return TensorGraph(
        nodes=[
            _node("n0", "a_t", ["a"], typespec),
            _node("n1", "bb", ["b", "b"], typespec),
            _node("n2", "out", ["a_t", "bb"], typespec),
        ],
        inputs=[("a", typespec), ("b", typespec)],
        outputs=["out"],
    )


def _split_chain_graph(transpose_operand_position: int) -> TensorGraph:
    """One long producer chain and one single operation, joined by one consumer.

    ``transpose_operand_position`` decides whether the single operation is the
    consumer's first or second operand. Nothing about the graph's meaning
    changes with it, so nothing about the emission order should either -- that
    is exactly the asymmetry a differentiated-parameter change introduces.
    """
    typespec = _typespec("f32", [2, 2])
    operands = ["t", "c3"] if transpose_operand_position == 0 else ["c3", "t"]
    return TensorGraph(
        nodes=[
            _node("nc1", "c1", ["p"], typespec),
            _node("nc2", "c2", ["c1"], typespec),
            _node("nc3", "c3", ["c2"], typespec),
            _node("nt", "t", ["x"], typespec),
            _node("nout", "out", operands, typespec),
        ],
        inputs=[("p", typespec), ("x", typespec)],
        outputs=["out"],
    )


def test_a_producer_is_emitted_immediately_before_the_consumer_that_reads_it():
    """The transpose is emitted next to the matmul that consumes it, and the
    independent chain producing the other operand is pushed ahead of both. A
    producer emitted as early as its own operands allow instead lands at the
    front, separated from its consumer by work that has nothing to do with it."""
    order = _emission_order(_independent_operand_graph())

    assert order == ["bb", "a_t", "out"]


@pytest.mark.parametrize("transpose_operand_position", [0, 1])
def test_the_shortest_producer_chain_is_emitted_next_to_its_consumer(
    transpose_operand_position: int,
):
    """Two graphs that differ only in operand order schedule identically: the
    long chain is emitted first because it needs the room, and the single
    operation stays beside the consumer that reads it."""
    graph = _split_chain_graph(transpose_operand_position)

    order = _emission_order(graph)

    assert order == ["c1", "c2", "c3", "t", "out"]


def _ordering_shape_catalogue() -> list[tuple[str, TensorGraph, list[str] | None]]:
    """Graph shapes whose emission order the scheduler must keep executable."""
    typespec = _typespec("f32", [2, 2])
    chain = TensorGraph(
        nodes=[
            _node("n0", "v1", ["v0"], typespec),
            _node("n1", "v2", ["v1"], typespec),
            _node("n2", "v3", ["v2"], typespec),
        ],
        inputs=[("v0", typespec)],
        outputs=["v3"],
    )
    diamond = TensorGraph(
        nodes=[
            _node("n0", "left", ["root"], typespec),
            _node("n1", "right", ["root"], typespec),
            _node("n2", "joined", ["left", "right"], typespec),
        ],
        inputs=[("root", typespec)],
        outputs=["joined"],
    )
    fan_in = TensorGraph(
        nodes=[
            _node("n0", "one", ["alpha"], typespec),
            _node("n1", "two", ["beta"], typespec),
            _node("n2", "three", ["gamma"], typespec),
            _node("n3", "pair", ["one", "two"], typespec),
            _node("n4", "all", ["pair", "three"], typespec),
        ],
        inputs=[("alpha", typespec), ("beta", typespec), ("gamma", typespec)],
        outputs=["all"],
    )
    repeated_operand = TensorGraph(
        nodes=[
            _node("n0", "doubled", ["alpha", "alpha"], typespec),
            _node("n1", "again", ["doubled", "doubled"], typespec),
        ],
        inputs=[("alpha", typespec)],
        outputs=["again"],
    )
    shared_producer = TensorGraph(
        nodes=[
            _node("n0", "shared", ["alpha"], typespec),
            _node("n1", "first", ["shared", "beta"], typespec),
            _node("n2", "second", ["shared", "gamma"], typespec),
        ],
        inputs=[("alpha", typespec), ("beta", typespec), ("gamma", typespec)],
        outputs=["first", "second"],
    )
    partially_selected = TensorGraph(
        nodes=[
            _node("n0", "used", ["alpha"], typespec),
            _node("n1", "unused", ["beta"], typespec),
            _node("n2", "out", ["used", "gamma"], typespec),
        ],
        inputs=[("alpha", typespec), ("beta", typespec), ("gamma", typespec)],
        outputs=["out", "unused"],
    )
    return [
        ("chain", chain, None),
        ("diamond", diamond, None),
        ("fan-in", fan_in, None),
        ("repeated operand", repeated_operand, None),
        ("shared producer", shared_producer, None),
        ("independent operands", _independent_operand_graph(), None),
        ("split chain, first operand", _split_chain_graph(0), None),
        ("split chain, second operand", _split_chain_graph(1), None),
        ("narrowed selection", partially_selected, ["out"]),
        ("deep chain", _deep_chain_graph(64), None),
    ]


def test_emission_order_places_every_operand_before_the_operation_reading_it():
    """Topological validity checked as a property of the order itself, not
    inferred from a consumer that happens to succeed: at each position, every
    value the operation reads is either free or already emitted."""
    for label, graph, outputs in _ordering_shape_catalogue():
        analysis = analyze_graph_dependencies(graph, outputs=outputs)
        emitted = [dependency.value_id for dependency in analysis.local_values]
        produced_by = {node.output_value_id: node for node in graph.nodes}

        available: set[str] = set()
        for value_id in emitted:
            node = produced_by[value_id]
            for input_value_id in node.input_value_ids:
                if input_value_id in produced_by and input_value_id in emitted:
                    assert input_value_id in available, (
                        f"{label}: {value_id!r} reads {input_value_id!r}, "
                        f"which is emitted later at {emitted.index(input_value_id)}"
                    )
            available.add(value_id)


def test_emission_order_contains_exactly_the_reachable_operations():
    """Reordering may not add, drop, or duplicate an operation. The reachable
    set is recomputed here from the graph alone, so the assertion does not
    depend on the traversal it is checking."""
    for label, graph, outputs in _ordering_shape_catalogue():
        analysis = analyze_graph_dependencies(graph, outputs=outputs)
        emitted = [dependency.value_id for dependency in analysis.local_values]
        produced_by = {node.output_value_id: node for node in graph.nodes}

        expected: set[str] = set()
        pending = list(analysis.selected_outputs)
        while pending:
            value_id = pending.pop()
            if value_id in expected or value_id not in produced_by:
                continue
            expected.add(value_id)
            pending.extend(produced_by[value_id].input_value_ids)

        assert len(emitted) == len(set(emitted)), f"{label}: an operation is emitted twice"
        assert set(emitted) == expected, f"{label}: the emitted set is not the reachable set"


def test_repeated_analyses_emit_an_identical_order():
    """Equivalent graphs schedule identically, so a consumer may bind to the
    order across separate analyses of the same trace."""
    for _label, graph, outputs in _ordering_shape_catalogue():
        first = _emission_order(graph, outputs)
        second = _emission_order(graph, outputs)

        assert first == second


def test_emission_order_is_identical_under_varied_hash_seeds():
    """The tie-break must not read set iteration, hash order, or object
    identity. Each seed runs in its own interpreter, because PYTHONHASHSEED is
    fixed at startup and cannot be varied in-process."""
    snippet = textwrap.dedent(
        """
        from tinychain.autodiff import TensorGraph, TensorNodeRecord, AddOperator, MulOperator
        from tinychain.autodiff.dependencies import analyze_graph_dependencies

        typespec = {"dtype": "f32", "shape": [2, 2]}

        def node(node_id, output, inputs):
            return TensorNodeRecord(
                node_id=node_id,
                output_value_id=output,
                operator=AddOperator() if len(inputs) == 2 else MulOperator(),
                op_params={},
                input_value_ids=list(inputs),
                output_typespec=typespec,
            )

        graph = TensorGraph(
            nodes=[
                node("n0", "a_t", ["a"]),
                node("n1", "bb", ["b", "b"]),
                node("nc1", "c1", ["c"]),
                node("nc2", "c2", ["c1"]),
                node("n2", "out", ["a_t", "bb"]),
                node("n3", "joined", ["out", "c2"]),
            ],
            inputs=[("a", typespec), ("b", typespec), ("c", typespec)],
            outputs=["joined"],
        )
        analysis = analyze_graph_dependencies(graph)
        print(",".join(d.value_id for d in analysis.local_values))
        """
    )

    orders = []
    for seed in range(5):
        environment = dict(os.environ, PYTHONHASHSEED=str(seed))
        completed = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
        orders.append(completed.stdout.strip())

    assert len(set(orders)) == 1, f"emission order varied across hash seeds: {orders}"


def test_sequencing_a_cyclic_reachable_set_is_reported_rather_than_dropped():
    """The scheduler cannot place an operation whose consumers never all get
    placed. Reaching this through a public entry point is impossible, because
    the reachability walk rejects a cycle first, so the set is handed to the
    sequencing step directly. It must report rather than emit the operations
    it did manage to place, which would be a program in an order no consumer
    can run."""
    typespec = _typespec("f32", [2])
    left = _node("n0", "left", ["right"], typespec)
    right = _node("n1", "right", ["left"], typespec)

    _assert_category(
        "malformed_derivative_ir",
        lambda: _order_reachable_nodes({"n0": left, "n1": right}),
    )
