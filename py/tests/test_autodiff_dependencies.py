"""Unit tests for framework-owned structured dependency analysis.

These tests pin the contract that a consumer can bind every derivative input
from provenance alone: without scanning graph nodes, without a private producer
map, and without any convention about how value ids are spelled.
"""

from __future__ import annotations

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
