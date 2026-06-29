"""Tests for transpose VJP and T-04c autodiff finalization."""
from __future__ import annotations

import pytest

import tinychain as tc
from tinychain.autodiff import (
    AddOperator,
    AutodiffError,
    BroadcastReduceOperator,
    DerivativeProgram,
    GradientAccumulator,
    MatmulOperator,
    TensorGraph,
    TensorNodeRecord,
    TransposeOperator,
    TransposeVjpRule,
    generate,
)
from tinychain.autodiff.vjp import VjpContext


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _transpose_graph(input_shape=(2, 3), perm=(1, 0), dtype="f32"):
    output_shape = tuple(input_shape[axis] for axis in perm)
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v1",
        operator=TransposeOperator(),
        op_params={"perm": list(perm)},
        input_value_ids=["v0"],
        output_typespec=_typespec(output_shape, dtype),
    )
    return TensorGraph(
        nodes=[node],
        inputs=[("v0", _typespec(input_shape, dtype))],
        outputs=["v1"],
    )


def _add_graph_with_disconnected_input():
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=AddOperator(),
        op_params={},
        input_value_ids=["v0", "v1"],
        output_typespec=_typespec((2, 3)),
    )
    return TensorGraph(
        nodes=[node],
        inputs=[("v0", _typespec((2, 3))), ("v1", _typespec((2, 3))), ("v9", _typespec((2, 3)))],
        outputs=["v2"],
    )


def _matmul_graph():
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=MatmulOperator(),
        op_params={},
        input_value_ids=["v0", "v1"],
        output_typespec=_typespec((3, 5)),
    )
    return TensorGraph(
        nodes=[node],
        inputs=[("v0", _typespec((3, 4))), ("v1", _typespec((4, 5)))],
        outputs=["v2"],
    )


def _counter(prefix):
    value = {"n": 0}

    def next_id():
        item = f"{prefix}{value['n']}"
        value["n"] += 1
        return item

    return next_id


def _transpose_context(perm, *, input_shape=(2, 3, 4), output_shape=(4, 2, 3)):
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v1",
        operator=TransposeOperator(),
        op_params={"perm": perm},
        input_value_ids=["v0"],
        output_typespec=_typespec(output_shape, "f64"),
    )
    return VjpContext(
        upstream_value_id="dZ",
        node=node,
        value_typespecs={
            "v0": _typespec(input_shape, "f64"),
            "v1": _typespec(output_shape, "f64"),
            "dZ": _typespec(output_shape, "f64"),
        },
        next_value_id=_counter("dv"),
        next_node_id=_counter("dn"),
    )


def test_transpose_vjp_inverse_permutation_shape_and_dtype():
    result = TransposeVjpRule().apply(_transpose_context([2, 0, 1]))

    assert list(result.gradients) == ["v0"]
    node = result.derivative_nodes[0]
    assert isinstance(node.operator, TransposeOperator)
    assert node.op_params["perm"] == [1, 2, 0]
    assert node.input_value_ids == ["dZ"]
    assert node.output_typespec == _typespec((2, 3, 4), "f64")


@pytest.mark.parametrize(
    "perm",
    [None, "102", 1, [0, 0], [0, 1, 3], [0, 2], [-1, 0], [0, "1"], [True, 0]],
)
def test_transpose_vjp_invalid_permutation_errors(perm):
    ctx = _transpose_context([1, 0], input_shape=(2, 3), output_shape=(3, 2))
    bad_node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v1",
        operator=TransposeOperator(),
        op_params={"perm": perm},
        input_value_ids=["v0"],
        output_typespec=_typespec((3, 2), "f64"),
    )
    bad_ctx = VjpContext(
        upstream_value_id=ctx.upstream_value_id,
        node=bad_node,
        value_typespecs=ctx.value_typespecs,
        next_value_id=ctx.next_value_id,
        next_node_id=ctx.next_node_id,
    )

    with pytest.raises(AutodiffError) as exc:
        TransposeVjpRule().apply(bad_ctx)
    assert exc.value.category == "invalid_permutation"


def test_reverse_traversal_dispatches_transpose_rule():
    program = generate(_transpose_graph(), "v1", ["v0"], "seed")

    assert len(program.nodes) == 1
    assert isinstance(program.nodes[0].operator, TransposeOperator)
    assert program.nodes[0].op_params["perm"] == [1, 0]
    assert program.output_gradients == [program.nodes[0].output_value_id]


def test_gradient_accumulator_reduces_and_combines_fanout_deterministically():
    value_typespecs = {
        "v0": _typespec((2, 3)),
        "g_big": _typespec((4, 2, 3)),
        "g_exact": _typespec((2, 3)),
    }
    accumulator = GradientAccumulator(value_typespecs=value_typespecs)
    accumulator.add("v0", "g_exact")
    accumulator.add("v0", "g_big")

    result_id, nodes = accumulator.result_for(
        "v0",
        next_value_id=_counter("d"),
        next_node_id=_counter("dn"),
    )

    assert result_id == nodes[-1].output_value_id
    assert [type(node.operator) for node in nodes] == [BroadcastReduceOperator, AddOperator]
    assert nodes[0].op_params["target_shape"] == [2, 3]
    assert nodes[1].input_value_ids == [nodes[0].output_value_id, "g_exact"]


def test_fanout_graph_build_is_deterministic_across_runs():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v2",
                operator=AddOperator(),
                op_params={},
                input_value_ids=["v0", "v1"],
                output_typespec=_typespec((2, 3)),
            ),
            TensorNodeRecord(
                node_id="n1",
                output_value_id="v3",
                operator=AddOperator(),
                op_params={},
                input_value_ids=["v0", "v2"],
                output_typespec=_typespec((2, 3)),
            ),
        ],
        inputs=[("v0", _typespec((2, 3))), ("v1", _typespec((2, 3)))],
        outputs=["v3"],
    )

    first = generate(graph, "v3", ["v0", "v1"], "seed")
    second = generate(graph, "v3", ["v0", "v1"], "seed")

    assert [(node.operator.route_name, node.input_value_ids, node.output_value_id) for node in first.nodes] == [
        (node.operator.route_name, node.input_value_ids, node.output_value_id) for node in second.nodes
    ]


def test_ordered_wrt_output_uses_caller_order():
    program = generate(_matmul_graph(), "v2", ["v1", "v0"], "seed")

    assert program.output_gradients == [program.gradients["v1"], program.gradients["v0"]]


def test_missing_disconnected_wrt_raises_missing_derivative_behavior():
    with pytest.raises(AutodiffError) as exc:
        generate(_add_graph_with_disconnected_input(), "v2", ["v9"], "seed")
    assert exc.value.category == "missing_derivative_behavior"


def test_generate_returns_structured_experimental_program_not_callbacks():
    program = generate(_transpose_graph(), "v1", ["v0"], "seed")

    assert isinstance(program, DerivativeProgram)
    assert isinstance(program.nodes, list)
    assert all(isinstance(node, TensorNodeRecord) for node in program.nodes)
    assert not callable(program)
    assert all(not callable(node) for node in program.nodes)


def test_tc_grad_delegates_to_real_tensor_graph_engine():
    graph = _transpose_graph()

    program = tc.grad(graph, wrt=["v0"], seed_typespec=_typespec((3, 2)))

    assert isinstance(program, DerivativeProgram)
    assert isinstance(program.nodes[0].operator, TransposeOperator)
    assert program.output_gradients != ["v0"]
