"""Unit tests for MatmulVjpRule and the transpose-last-two helper.

End-to-end matmul gradient tests marked ``tc_server`` require ``tc-server``.
"""
from __future__ import annotations

import pytest

from tinychain.autodiff import (
    AddOperator,
    AutodiffError,
    BroadcastReduceOperator,
    MatmulOperator,
    TransposeOperator,
    MatmulVjpRule,
    ReverseTraversal,
    TensorGraph,
    TensorNodeRecord,
    TensorOperator,
    generate,
)
from tinychain.autodiff.vjp import (
    BroadcastReductionPlanner,
    VjpContext,
    _swap_last_two_dims,
    _transpose_last_two_perm,
)


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _matmul_graph(lhs_shape, rhs_shape, out_shape):
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=MatmulOperator(),
        op_params={},
        input_value_ids=["v0", "v1"],
        output_typespec=_typespec(out_shape),
    )
    return TensorGraph(
        nodes=[node],
        inputs=[("v0", _typespec(lhs_shape)), ("v1", _typespec(rhs_shape))],
        outputs=["v2"],
    )


# --- transpose_last_two helpers ---

def test_swap_last_two_dims_rank2():
    assert _swap_last_two_dims((3, 4)) == (4, 3)


def test_swap_last_two_dims_batched():
    assert _swap_last_two_dims((2, 5, 3, 4)) == (2, 5, 4, 3)


def test_transpose_last_two_perm_rank2():
    assert _transpose_last_two_perm(2) == [1, 0]


def test_transpose_last_two_perm_rank4():
    assert _transpose_last_two_perm(4) == [0, 1, 3, 2]


# --- MatmulVjpRule unit tests ---

def _operator_types(nodes):
    return [type(node.operator) for node in nodes]


def _make_context(lhs_shape, rhs_shape, out_shape, dz_id="dZ", needed_input_value_ids=None):
    value_typespecs = {
        "v0": _typespec(lhs_shape),
        "v1": _typespec(rhs_shape),
        "v2": _typespec(out_shape),
    }
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=MatmulOperator(),
        op_params={},
        input_value_ids=["v0", "v1"],
        output_typespec=_typespec(out_shape),
    )
    counter = [0]

    def nv():
        vid = f"dv{counter[0]}"
        counter[0] += 1
        return vid

    counter2 = [0]

    def nn():
        nid = f"dn{counter2[0]}"
        counter2[0] += 1
        return nid

    return VjpContext(
        upstream_value_id=dz_id,
        node=node,
        value_typespecs=value_typespecs,
        needed_input_value_ids=frozenset({"v0", "v1"} if needed_input_value_ids is None else needed_input_value_ids),
        next_value_id=nv,
        next_node_id=nn,
    )


def test_matmul_vjp_rank2_exact_shapes():
    # A: (3, 4), B: (4, 5), Z: (3, 5)
    ctx = _make_context((3, 4), (4, 5), (3, 5))
    result = MatmulVjpRule().apply(ctx)

    # dA = matmul(dZ, B^T), dB = matmul(A^T, dZ)
    # No broadcast reduction needed — no batch dims
    assert _operator_types(result.derivative_nodes) == [TransposeOperator, MatmulOperator, TransposeOperator, MatmulOperator]

    # B^T node: perm [1, 0], shape (5, 4)
    b_t_node = result.derivative_nodes[0]
    assert b_t_node.op_params["perm"] == [1, 0]
    assert b_t_node.output_typespec["shape"] == [5, 4]

    # dA node shape: (3, 4)
    da_node = result.derivative_nodes[1]
    assert da_node.output_typespec["shape"] == [3, 4]
    assert da_node.input_value_ids == ["dZ", b_t_node.output_value_id]

    # A^T node: perm [1, 0], shape (4, 3)
    a_t_node = result.derivative_nodes[2]
    assert a_t_node.op_params["perm"] == [1, 0]
    assert a_t_node.output_typespec["shape"] == [4, 3]

    # dB node shape: (4, 5)
    db_node = result.derivative_nodes[3]
    assert db_node.output_typespec["shape"] == [4, 5]
    assert db_node.input_value_ids == [a_t_node.output_value_id, "dZ"]

    # Gradients point to the matmul outputs (no reduction)
    assert result.gradients["v0"] == da_node.output_value_id
    assert result.gradients["v1"] == db_node.output_value_id


def test_matmul_vjp_batched_no_broadcast():
    # A: (2, 3, 4), B: (2, 4, 5), Z: (2, 3, 5)
    ctx = _make_context((2, 3, 4), (2, 4, 5), (2, 3, 5))
    result = MatmulVjpRule().apply(ctx)

    assert _operator_types(result.derivative_nodes) == [TransposeOperator, MatmulOperator, TransposeOperator, MatmulOperator]

    b_t_node = result.derivative_nodes[0]
    assert b_t_node.op_params["perm"] == [0, 2, 1]
    assert b_t_node.output_typespec["shape"] == [2, 5, 4]

    da_node = result.derivative_nodes[1]
    assert da_node.output_typespec["shape"] == [2, 3, 4]

    a_t_node = result.derivative_nodes[2]
    assert a_t_node.op_params["perm"] == [0, 2, 1]
    assert a_t_node.output_typespec["shape"] == [2, 4, 3]

    db_node = result.derivative_nodes[3]
    assert db_node.output_typespec["shape"] == [2, 4, 5]

    assert result.gradients["v0"] == da_node.output_value_id
    assert result.gradients["v1"] == db_node.output_value_id


def test_matmul_vjp_batched_with_batch_broadcast_reduction():
    # A: (3, 4) (no batch), B: (2, 4, 5) (batch=2), Z: (2, 3, 5)
    # dA intermediate: (2, 3, 4) → must reduce to (3, 4)
    # dB intermediate: (2, 4, 5) → no reduction needed
    ctx = _make_context((3, 4), (2, 4, 5), (2, 3, 5))
    result = MatmulVjpRule().apply(ctx)

    assert _operator_types(result.derivative_nodes) == [TransposeOperator, MatmulOperator, BroadcastReduceOperator, TransposeOperator, MatmulOperator]

    da_reduced = result.derivative_nodes[2]
    assert isinstance(da_reduced.operator, BroadcastReduceOperator)
    assert da_reduced.op_params["target_shape"] == [3, 4]

    assert result.gradients["v0"] == da_reduced.output_value_id
    db_node = result.derivative_nodes[4]
    assert result.gradients["v1"] == db_node.output_value_id
    assert db_node.output_typespec["shape"] == [2, 4, 5]


def test_matmul_vjp_both_operands_broadcast_reduced():
    # A: (3, 4), B: (4, 5), Z: (2, 3, 5) — both missing leading batch dim
    ctx = _make_context((3, 4), (4, 5), (2, 3, 5))
    result = MatmulVjpRule().apply(ctx)

    assert _operator_types(result.derivative_nodes) == [TransposeOperator, MatmulOperator, BroadcastReduceOperator, TransposeOperator, MatmulOperator, BroadcastReduceOperator]

    da_reduced = result.derivative_nodes[2]
    assert da_reduced.op_params["target_shape"] == [3, 4]
    db_reduced = result.derivative_nodes[5]
    assert db_reduced.op_params["target_shape"] == [4, 5]


def test_matmul_shape_mismatch_inner_dims():
    # A: (3, 4), B: (3, 5) — inner dims 4 != 3 → matmul_shape_mismatch
    ctx = _make_context((3, 4), (3, 5), (3, 5))
    with pytest.raises(AutodiffError) as exc:
        MatmulVjpRule().apply(ctx)
    assert exc.value.category == "matmul_shape_mismatch"


def test_matmul_shape_mismatch_rank_too_low():
    # A rank-1 operand is invalid
    ctx = _make_context((4,), (4, 5), (5,))
    with pytest.raises(AutodiffError) as exc:
        MatmulVjpRule().apply(ctx)
    assert exc.value.category == "matmul_shape_mismatch"


def test_transpose_node_has_correct_dtype_propagated():
    ctx = _make_context((2, 3), (3, 4), (2, 4))
    result = MatmulVjpRule().apply(ctx)
    for node in result.derivative_nodes:
        assert node.output_typespec.get("dtype") == "f32"


def test_matmul_vjp_only_lhs_needed_emits_lhs_branch():
    ctx = _make_context((3, 4), (4, 5), (3, 5), needed_input_value_ids={"v0"})
    result = MatmulVjpRule().apply(ctx)

    assert set(result.gradients) == {"v0"}
    assert _operator_types(result.derivative_nodes) == [TransposeOperator, MatmulOperator]
    assert result.derivative_nodes[0].input_value_ids == ["v1"]
    assert result.derivative_nodes[1].input_value_ids == ["dZ", result.derivative_nodes[0].output_value_id]


def test_matmul_vjp_only_rhs_needed_emits_rhs_branch():
    ctx = _make_context((3, 4), (4, 5), (3, 5), needed_input_value_ids={"v1"})
    result = MatmulVjpRule().apply(ctx)

    assert set(result.gradients) == {"v1"}
    assert _operator_types(result.derivative_nodes) == [TransposeOperator, MatmulOperator]
    assert result.derivative_nodes[0].input_value_ids == ["v0"]
    assert result.derivative_nodes[1].input_value_ids == [result.derivative_nodes[0].output_value_id, "dZ"]


def test_matmul_vjp_no_needed_inputs_emits_nothing():
    ctx = _make_context((3, 4), (4, 5), (3, 5), needed_input_value_ids=set())
    result = MatmulVjpRule().apply(ctx)

    assert result.gradients == {}
    assert result.derivative_nodes == []


# --- ReverseTraversal dispatches MatmulOperator ---

def test_reverse_traversal_matmul_rank2():
    graph = _matmul_graph((3, 4), (4, 5), (3, 5))
    program = generate(graph, "v2", ["v0", "v1"], "seed")

    # No batch broadcast → 4 derivative nodes (2 transposes + 2 matmuls)
    operator_types = _operator_types(program.nodes)
    assert TransposeOperator in operator_types
    assert MatmulOperator in operator_types
    assert program.gradients["v0"] is not None
    assert program.gradients["v1"] is not None


def test_reverse_traversal_matmul_does_not_break_add_path():
    add_graph = TensorGraph(
        nodes=[TensorNodeRecord(
            node_id="n0",
            output_value_id="v2",
            operator=AddOperator(),
            op_params={},
            input_value_ids=["v0", "v1"],
            output_typespec=_typespec((2, 3)),
        )],
        inputs=[("v0", _typespec((2, 3))), ("v1", _typespec((2, 3)))],
        outputs=["v2"],
    )
    program = generate(add_graph, "v2", ["v0", "v1"], "seed")
    assert program.output_gradients == ["seed", "seed"]


def test_reverse_traversal_matmul_single_wrt_lhs_emits_only_lhs_branch():
    graph = _matmul_graph((3, 4), (4, 5), (3, 5))
    program = generate(graph, "v2", ["v0"], "seed")

    assert set(program.gradients) == {"v0"}
    assert _operator_types(program.nodes) == [TransposeOperator, MatmulOperator]
    assert program.nodes[0].input_value_ids == ["v1"]
    assert program.nodes[1].input_value_ids == ["seed", program.nodes[0].output_value_id]


def test_reverse_traversal_matmul_single_wrt_rhs_emits_only_rhs_branch():
    graph = _matmul_graph((3, 4), (4, 5), (3, 5))
    program = generate(graph, "v2", ["v1"], "seed")

    assert set(program.gradients) == {"v1"}
    assert _operator_types(program.nodes) == [TransposeOperator, MatmulOperator]
    assert program.nodes[0].input_value_ids == ["v0"]
    assert program.nodes[1].input_value_ids == [program.nodes[0].output_value_id, "seed"]


def test_reverse_traversal_matmul_skips_unsupported_non_wrt_producer():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v0",
                operator=TensorOperator("constant"),
                op_params={},
                input_value_ids=[],
                output_typespec=_typespec((3, 4)),
            ),
            TensorNodeRecord(
                node_id="n1",
                output_value_id="v2",
                operator=MatmulOperator(),
                op_params={},
                input_value_ids=["v0", "v1"],
                output_typespec=_typespec((3, 5)),
            ),
        ],
        inputs=[("v1", _typespec((4, 5)))],
        outputs=["v2"],
    )

    program = generate(graph, "v2", ["v1"], "seed")

    assert set(program.gradients) == {"v1"}
    assert _operator_types(program.nodes) == [TransposeOperator, MatmulOperator]
    assert program.nodes[0].input_value_ids == ["v0"]


# Transport-neutral numerical matmul tests live in test_execution_matmul_gradient.py.
