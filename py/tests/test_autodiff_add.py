from __future__ import annotations

import pytest
import tinychain as tc

from tinychain.autodiff import (
    OP_ADD,
    OP_BROADCAST_REDUCE,
    AddVjpRule,
    AutodiffError,
    BroadcastReductionPlanner,
    GradientAccumulator,
    ReverseTraversal,
    SeedValidator,
    TensorGraph,
    TensorNodeRecord,
    generate,
)


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _add_graph(lhs_shape=(2, 3), rhs_shape=(2, 3), out_shape=(2, 3)):
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="z",
        op_kind=OP_ADD,
        op_params={},
        input_value_ids=["x", "y"],
        output_typespec=_typespec(out_shape),
    )
    return TensorGraph(
        nodes=[node],
        inputs=[("x", _typespec(lhs_shape)), ("y", _typespec(rhs_shape))],
        outputs=["z"],
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
    program = generate(graph, "z", ["x", "y"], "seed")

    assert program.nodes == []
    assert program.output_gradients == ["seed", "seed"]


def test_add_vjp_right_aligned_broadcast_emits_reduce_for_broadcast_operand():
    graph = _add_graph(lhs_shape=(2, 3), rhs_shape=(1, 3), out_shape=(2, 3))
    program = generate(graph, "z", ["x", "y"], "seed")

    assert program.output_gradients[0] == "seed"
    assert len(program.nodes) == 1
    reduce = program.nodes[0]
    assert reduce.op_kind == OP_BROADCAST_REDUCE
    assert reduce.op_params == {"target_shape": [1, 3]}
    assert program.output_gradients[1] == reduce.output_value_id


def test_add_vjp_missing_leading_dims_emits_reduce():
    graph = _add_graph(lhs_shape=(4, 2, 3), rhs_shape=(2, 1), out_shape=(4, 2, 3))
    program = generate(graph, "z", ["y"], "seed")

    reduce = program.nodes[0]
    assert reduce.op_params == {"target_shape": [2, 1]}
    assert program.output_gradients == [reduce.output_value_id]


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
        output_value_id="z",
        wrt=["x", "y"],
        seed_value_id="seed",
        seed_typespec=_typespec((2, 3)),
    )

    assert program.gradients["x"] == "seed"
    assert program.gradients["y"] == program.nodes[0].output_value_id


def test_gradient_accumulator_single_contribution_passthrough():
    accumulator = GradientAccumulator(value_typespecs={"x": _typespec((2, 3))})
    accumulator.add("x", "dx")

    assert accumulator.result_for("x") == ("dx", [])


def test_reverse_traversal_unknown_operator_fails():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="z",
                op_kind="mystery",
                op_params={},
                input_value_ids=["x"],
                output_typespec=_typespec((2, 3)),
            )
        ],
        inputs=[("x", _typespec((2, 3)))],
        outputs=["z"],
    )

    with pytest.raises(AutodiffError) as exc:
        generate(graph, "z", ["x"], "seed")

    assert exc.value.category == "unsupported_operator"


def test_tc_grad_graph_target_delegates_to_generate():
    graph = _add_graph()

    program = tc.grad(graph, wrt=("x", "y"), seed="seed")

    assert program.output_gradients == ["seed", "seed"]


def test_tc_grad_unsupported_target_fails_without_placeholder_opref():
    with pytest.raises(AutodiffError) as exc:
        tc.grad(object(), wrt=("x",))

    assert exc.value.category == "autodiff_not_implemented"
