from __future__ import annotations

import pytest
import tinychain as tc

from tinychain.autodiff import (
    AddOperator,
    AddVjpRule,
    AutodiffError,
    BroadcastReduceOperator,
    BroadcastReductionPlanner,
    GradientAccumulator,
    ReverseTraversal,
    SeedValidator,
    TensorGraph,
    TensorNodeRecord,
    TensorOperator,
    generate,
)


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _counter(prefix):
    value = {"n": 0}

    def next_id():
        item = f"{prefix}{value['n']}"
        value["n"] += 1
        return item

    return next_id


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


def test_broadcast_reduction_axes_exact_shape():
    plan = BroadcastReductionPlanner().plan(
        result_shape=(2, 3),
        operand_shape=(2, 3),
    )

    assert plan.axes == ()


def test_broadcast_reduction_axes_right_aligned_broadcast():
    plan = BroadcastReductionPlanner().plan(
        result_shape=(2, 3),
        operand_shape=(1, 3),
    )

    assert plan.axes == (0,)


def test_broadcast_reduction_axes_missing_leading_dims():
    plan = BroadcastReductionPlanner().plan(
        result_shape=(4, 2, 3),
        operand_shape=(2, 1),
    )

    assert plan.axes == (0, 2)


def test_broadcast_reduction_rejects_incompatible_shapes():
    with pytest.raises(AutodiffError) as exc:
        BroadcastReductionPlanner().plan(
            result_shape=(2, 3),
            operand_shape=(2, 2),
        )

    assert exc.value.category == "broadcast_shape_mismatch"


def test_add_vjp_no_broadcast_passes_upstream_to_both_operands():
    graph = _add_graph()
    program = generate(graph, "v2", ["v0", "v1"], "seed")

    assert program.nodes == []
    assert program.output_gradients == ["seed", "seed"]


def test_add_vjp_right_aligned_broadcast_emits_reduce_for_broadcast_operand():
    graph = _add_graph(lhs_shape=(2, 3), rhs_shape=(1, 3), out_shape=(2, 3))
    program = generate(graph, "v2", ["v0", "v1"], "seed")

    assert program.output_gradients[0] == "seed"
    assert len(program.nodes) == 1
    reduce = program.nodes[0]
    assert isinstance(reduce.operator, BroadcastReduceOperator)
    assert reduce.op_params == {"target_shape": [1, 3]}
    assert program.output_gradients[1] == reduce.output_value_id


def test_add_vjp_missing_leading_dims_emits_reduce():
    graph = _add_graph(lhs_shape=(4, 2, 3), rhs_shape=(2, 1), out_shape=(4, 2, 3))
    program = generate(graph, "v2", ["v1"], "seed")

    reduce = program.nodes[0]
    assert reduce.op_params == {"target_shape": [2, 1]}
    assert program.output_gradients == [reduce.output_value_id]


def test_add_single_wrt_prunes_unused_operand_gradient():
    graph = _add_graph()
    program = generate(graph, "v2", ["v1"], "seed")

    assert set(program.gradients) == {"v1"}
    assert program.output_gradients == ["seed"]
    assert "v0" not in program.gradients
    assert program.nodes == []


def test_unsupported_non_wrt_producer_is_skipped():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v0",
                operator=TensorOperator("constant"),
                op_params={},
                input_value_ids=[],
                output_typespec=_typespec((2, 3)),
            ),
            TensorNodeRecord(
                node_id="n1",
                output_value_id="v2",
                operator=AddOperator(),
                op_params={},
                input_value_ids=["v0", "v1"],
                output_typespec=_typespec((2, 3)),
            ),
        ],
        inputs=[("v1", _typespec((2, 3)))],
        outputs=["v2"],
    )

    program = generate(graph, "v2", ["v1"], "seed")

    assert set(program.gradients) == {"v1"}
    assert program.output_gradients == ["seed"]
    assert "v0" not in program.gradients
    assert program.nodes == []


def test_seed_validator_rejects_non_floating_dtype():
    with pytest.raises(AutodiffError) as exc:
        SeedValidator().validate(
            seed_typespec=_typespec((2, 3), dtype="u64"),
            output_typespec=_typespec((2, 3)),
        )

    assert exc.value.category == "dtype_not_differentiable"


def test_seed_validator_rejects_shape_mismatch():
    with pytest.raises(AutodiffError) as exc:
        SeedValidator().validate(
            seed_typespec=_typespec((3, 2)),
            output_typespec=_typespec((2, 3)),
        )

    assert exc.value.category == "seed_shape_mismatch"


def test_reverse_traversal_single_add_node():
    graph = _add_graph(lhs_shape=(2, 3), rhs_shape=(1, 3), out_shape=(2, 3))
    program = ReverseTraversal().build(
        graph=graph,
        output_value_id="v2",
        wrt=["v0", "v1"],
        seed_value_id="seed",
        seed_typespec=_typespec((2, 3)),
    )

    assert program.gradients["v0"] == "seed"
    assert program.gradients["v1"] == program.nodes[0].output_value_id


def test_gradient_accumulator_single_contribution_passthrough():
    accumulator = GradientAccumulator(value_typespecs={"v0": _typespec((2, 3))})
    accumulator.add("v0", "dv0")

    assert accumulator.result_for("v0") == ("dv0", [])


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


def test_reverse_traversal_missing_disconnected_wrt_raises():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v2",
                operator=AddOperator(),
                op_params={},
                input_value_ids=["v0", "v1"],
                output_typespec=_typespec((2, 3)),
            )
        ],
        inputs=[("v0", _typespec((2, 3))), ("v1", _typespec((2, 3))), ("v9", _typespec((2, 3)))],
        outputs=["v2"],
    )

    with pytest.raises(AutodiffError) as exc:
        generate(graph, "v2", ["v9"], "seed")

    assert exc.value.category == "missing_derivative_behavior"


def test_reverse_traversal_unknown_operator_fails():
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


def test_tc_grad_graph_target_delegates_to_generate():
    graph = _add_graph()

    program = tc.grad(graph, wrt=("v0", "v1"), seed="seed")

    assert program.output_gradients == ["seed", "seed"]


def test_tc_grad_unsupported_target_fails_without_placeholder_opref():
    with pytest.raises(AutodiffError) as exc:
        tc.grad(object(), wrt=("v0",))

    assert exc.value.category == "autodiff_not_implemented"
