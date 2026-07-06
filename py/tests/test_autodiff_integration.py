"""Integration tests exercising Phase 2 autodiff subsystems together.

These tests cover FR-P2-007: generate(), serialization, reflection, graph
identity, the builder guard, and the VJP registry in combination, with zero
Phase 1 regressions.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tinychain.autodiff import (
    AddOperator,
    AutodiffError,
    AutodiffResult,
    DerivativeMetadata,
    MatmulOperator,
    TensorGraph,
    TensorGraphBuilder,
    TensorNodeRecord,
    TensorOperator,
    TransposeOperator,
    VjpRegistry,
    generate,
    get_active_builder,
    reflect_derivative_program,
)
from tinychain.autodiff.vjp import default_vjp_registry
from tinychain.graph_reflection import TypeSpec, TypedValueRef


PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _add_graph(lhs_shape=(2, 3), rhs_shape=(2, 3), out_shape=(2, 3)):
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=AddOperator(),
        op_params={},
        input_value_ids=["v0", "v1"],
        output_typespec=_typespec(out_shape),
    )
    return TensorGraph(
        nodes=[node],
        inputs=[("v0", _typespec(lhs_shape)), ("v1", _typespec(rhs_shape))],
        outputs=["v2"],
    )


def _matmul_graph():
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=MatmulOperator(),
        op_params={},
        input_value_ids=["v0", "v1"],
        output_typespec=_typespec((2, 3)),
    )
    return TensorGraph(
        nodes=[node],
        inputs=[("v0", _typespec((2, 3))), ("v1", _typespec((3, 2)))],
        outputs=["v2"],
    )


# --- Step 1: generate() + serialization integration ---


def test_derivative_program_to_dict():
    """generate() produces a DerivativeProgram whose to_dict() is JSON-serializable
    and contains the operator type name from the derivative nodes."""
    graph = _matmul_graph()
    program = generate(graph, "v2", ["v0", "v1"], "seed")

    result = program.to_dict()

    json_str = json.dumps(result)
    assert isinstance(json_str, str)

    assert len(result["nodes"]) > 0
    nodes_json = json.dumps(result["nodes"])
    assert "MatmulOperator" in nodes_json
    assert "TransposeOperator" in nodes_json


# --- Step 2: generate() + reflection integration ---


def test_reflect_derivative_program():
    """reflect_derivative_program() on a generated DerivativeProgram returns one
    TypedValueRef per derivative node with correct value/output/namespace/value_type."""
    graph = _add_graph(rhs_shape=(1, 3), out_shape=(2, 3))
    program = generate(graph, "v2", ["v0", "v1"], "seed")

    refs = reflect_derivative_program(program)

    assert len(refs) == len(program.nodes)
    for ref, node in zip(refs, program.nodes):
        assert isinstance(ref, TypedValueRef)
        assert ref.value == node.output_value_id
        assert ref.output == node.node_id
        assert ref.namespace == program.metadata.source_graph_id
        assert ref.value_type.class_uri == "/state/collection/tensor"


# --- Step 3: graph identity stability ---


def test_source_graph_id_stability():
    """Two generate() calls on structurally identical graphs produce the same
    source_graph_id."""
    graph1 = _add_graph()
    graph2 = _add_graph()
    program1 = generate(graph1, "v2", ["v0", "v1"], "seed")
    program2 = generate(graph2, "v2", ["v0", "v1"], "seed")

    assert program1.metadata.source_graph_id == program2.metadata.source_graph_id


# --- Step 4: graph_id override ---


def test_source_graph_id_override():
    """graph_id='custom_id' propagates to DerivativeMetadata.source_graph_id."""
    graph = _add_graph()
    program = generate(graph, "v2", ["v0", "v1"], "seed", graph_id="custom_id")

    assert program.metadata.source_graph_id == "custom_id"


# --- Step 5: nested builder guard ---


def test_nested_builder_guard():
    """Entering TensorGraphBuilder twice (nested) raises RuntimeError."""
    with TensorGraphBuilder() as outer:
        with pytest.raises(RuntimeError, match="Nested TensorGraphBuilder"):
            with TensorGraphBuilder() as inner:
                pass
    assert get_active_builder() is None


# --- Step 6: VjpRegistry.has_rule() ---


def test_vjp_registry_has_rule():
    """has_rule() returns True for registered operators and False for unregistered ones."""
    registry = default_vjp_registry()

    assert registry.has_rule(AddOperator()) is True
    assert registry.has_rule(MatmulOperator()) is True
    assert registry.has_rule(TransposeOperator()) is True

    class UnknownOperator(TensorOperator):
        pass

    assert registry.has_rule(UnknownOperator("unknown")) is False


# --- Step 7: VjpRegistry.supported_types() ---


def test_vjp_registry_supported_types():
    """supported_types() returns all 3 current operators (Add, Matmul, Transpose)."""
    registry = default_vjp_registry()
    supported = registry.supported_types()

    assert set(supported) == {AddOperator, MatmulOperator, TransposeOperator}
    assert len(supported) == 3


# --- Step 8: missing derivative behavior ---


def test_missing_derivative_behavior():
    """An unregistered operator in the graph causes AutodiffError with category
    'missing_derivative_behavior' when generate() is called."""
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=TensorOperator("mystery"),
                op_params={},
                input_value_ids=["v0"],
                output_typespec=_typespec((2, 3)),
            )
        ],
        inputs=[("v0", _typespec((2, 3)))],
        outputs=["v1"],
    )

    with pytest.raises(AutodiffError) as exc:
        generate(graph, "v1", ["v0"], "seed")

    assert exc.value.category == "missing_derivative_behavior"
    assert "mystery" in exc.value.message


# --- Step 9: to_dict() snapshot compatibility ---


def test_to_dict_snapshot_compatibility():
    """AutodiffError, AutodiffResult, and DerivativeMetadata to_dict() outputs
    match the expected pre-refactor snapshots after migration to serialize()."""

    error = AutodiffError(
        category="missing_derivative_behavior",
        message="no VJP rule for operator",
    )
    assert error.to_dict() == {
        "category": "missing_derivative_behavior",
        "message": "no VJP rule for operator",
    }

    metadata = DerivativeMetadata(
        source_graph_id="graph_abc",
        transform_version="0.1.0",
        tensor_op_contract_version="0.1.0",
        wrt_signature=("alpha", "beta"),
        seed_contract="seed_v0",
    )
    assert metadata.to_dict() == {
        "source_graph_id": "graph_abc",
        "transform_version": "0.1.0",
        "tensor_op_contract_version": "0.1.0",
        "wrt_signature": ["alpha", "beta"],
        "seed_contract": "seed_v0",
    }

    result_obj = AutodiffResult(
        gradients=[1.5, 2.5],
        metadata=metadata,
    )
    assert result_obj.to_dict() == {
        "gradients": [1.5, 2.5],
        "metadata": {
            "source_graph_id": "graph_abc",
            "transform_version": "0.1.0",
            "tensor_op_contract_version": "0.1.0",
            "wrt_signature": ["alpha", "beta"],
            "seed_contract": "seed_v0",
        },
    }


# --- Step 10: subprocess determinism ---


def test_source_graph_id_subprocess_determinism():
    """A subprocess running generate() on the same graph structure produces the
    same source_graph_id as the parent process."""
    graph = _add_graph()
    program = generate(graph, "v2", ["v0", "v1"], "seed")
    expected_id = program.metadata.source_graph_id

    script = """
from tinychain.autodiff import AddOperator, TensorGraph, TensorNodeRecord, generate

node = TensorNodeRecord(
    node_id="n0",
    output_value_id="v2",
    operator=AddOperator(),
    op_params={},
    input_value_ids=["v0", "v1"],
    output_typespec={"shape": [2, 3], "dtype": "f32"},
)
graph = TensorGraph(
    nodes=[node],
    inputs=[
        ("v0", {"shape": [2, 3], "dtype": "f32"}),
        ("v1", {"shape": [2, 3], "dtype": "f32"}),
    ],
    outputs=["v2"],
)
program = generate(graph, "v2", ["v0", "v1"], "seed")
print(program.metadata.source_graph_id)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = PY_DIR + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    actual_id = result.stdout.strip()
    assert actual_id == expected_id


# --- Step 11: VJP rule declaration order independence ---


def test_vjp_rule_declaration_order():
    """Reversed-order rule definitions in a temp registry match normal-order
    coverage: all three operator types are registered regardless of decorator
    application order."""

    reversed_registry = VjpRegistry()

    @reversed_registry.rule(TransposeOperator)
    class TransposeRule:
        operator_type = TransposeOperator

        def apply(self, context):
            ...

    @reversed_registry.rule(MatmulOperator)
    class MatmulRule:
        operator_type = MatmulOperator

        def apply(self, context):
            ...

    @reversed_registry.rule(AddOperator)
    class AddRule:
        operator_type = AddOperator

        def apply(self, context):
            ...

    normal_registry = default_vjp_registry()

    assert set(reversed_registry.supported_types()) == set(normal_registry.supported_types())
    assert reversed_registry.has_rule(AddOperator())
    assert reversed_registry.has_rule(MatmulOperator())
    assert reversed_registry.has_rule(TransposeOperator())


# --- Step 12: builder guard released on exception ---


def test_builder_guard_released_on_exit():
    """An exception inside a TensorGraphBuilder context block still releases the
    guard, allowing a subsequent builder to succeed."""
    try:
        with TensorGraphBuilder() as builder:
            raise ValueError("test exception")
    except ValueError:
        pass

    assert get_active_builder() is None

    with TensorGraphBuilder() as builder:
        assert get_active_builder() is builder
    assert get_active_builder() is None


# --- Step 13: sequential builders ---


def test_sequential_builders():
    """Two sequential (non-nested) TensorGraphBuilder contexts both succeed."""
    with TensorGraphBuilder() as first:
        assert get_active_builder() is first
    assert get_active_builder() is None

    with TensorGraphBuilder() as second:
        assert get_active_builder() is second
    assert get_active_builder() is None
