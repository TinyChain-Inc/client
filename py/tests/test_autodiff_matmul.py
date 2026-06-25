"""Unit tests for MatmulVjpRule and transpose_last_two helper (T-04b).

E2e matmul gradient tests (marked tc_server) require tc-server + T-05b installed.
"""
from __future__ import annotations

import pytest

from tinychain.autodiff import (
    OP_BROADCAST_REDUCE,
    OP_MATMUL,
    OP_TRANSPOSE,
    AutodiffError,
    MatmulVjpRule,
    ReverseTraversal,
    TensorGraph,
    TensorNodeRecord,
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
        operator=OP_MATMUL,
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

def _make_context(lhs_shape, rhs_shape, out_shape, dz_id="dZ"):
    value_typespecs = {
        "v0": _typespec(lhs_shape),
        "v1": _typespec(rhs_shape),
        "v2": _typespec(out_shape),
    }
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=OP_MATMUL,
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
        next_value_id=nv,
        next_node_id=nn,
    )


def test_matmul_vjp_rank2_exact_shapes():
    # A: (3, 4), B: (4, 5), Z: (3, 5)
    ctx = _make_context((3, 4), (4, 5), (3, 5))
    result = MatmulVjpRule().apply(ctx)

    # dA = matmul(dZ, B^T), dB = matmul(A^T, dZ)
    # No broadcast reduction needed — no batch dims
    ops = [n.operator for n in result.derivative_nodes]
    assert ops == [OP_TRANSPOSE, OP_MATMUL, OP_TRANSPOSE, OP_MATMUL]

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

    ops = [n.operator for n in result.derivative_nodes]
    assert ops == [OP_TRANSPOSE, OP_MATMUL, OP_TRANSPOSE, OP_MATMUL]

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

    ops = [n.operator for n in result.derivative_nodes]
    # transpose B, matmul dA-intermediate, reduce dA, transpose A, matmul dB
    assert ops == [OP_TRANSPOSE, OP_MATMUL, OP_BROADCAST_REDUCE, OP_TRANSPOSE, OP_MATMUL]

    da_reduced = result.derivative_nodes[2]
    assert da_reduced.operator == OP_BROADCAST_REDUCE
    assert da_reduced.op_params["target_shape"] == [3, 4]

    assert result.gradients["v0"] == da_reduced.output_value_id
    db_node = result.derivative_nodes[4]
    assert result.gradients["v1"] == db_node.output_value_id
    assert db_node.output_typespec["shape"] == [2, 4, 5]


def test_matmul_vjp_both_operands_broadcast_reduced():
    # A: (3, 4), B: (4, 5), Z: (2, 3, 5) — both missing leading batch dim
    ctx = _make_context((3, 4), (4, 5), (2, 3, 5))
    result = MatmulVjpRule().apply(ctx)

    ops = [n.operator for n in result.derivative_nodes]
    # transpose B, matmul dA-int, reduce dA, transpose A, matmul dB-int, reduce dB
    assert ops == [OP_TRANSPOSE, OP_MATMUL, OP_BROADCAST_REDUCE, OP_TRANSPOSE, OP_MATMUL, OP_BROADCAST_REDUCE]

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


# --- ReverseTraversal dispatches OP_MATMUL ---

def test_reverse_traversal_matmul_rank2():
    graph = _matmul_graph((3, 4), (4, 5), (3, 5))
    program = generate(graph, "v2", ["v0", "v1"], "seed")

    # No batch broadcast → 4 derivative nodes (2 transposes + 2 matmuls)
    ops = [n.operator for n in program.nodes]
    assert OP_TRANSPOSE in ops
    assert OP_MATMUL in ops
    assert program.gradients["v0"] is not None
    assert program.gradients["v1"] is not None


def test_reverse_traversal_matmul_does_not_break_add_path():
    from tinychain.autodiff.graph import OP_ADD
    add_graph = TensorGraph(
        nodes=[TensorNodeRecord(
            node_id="n0",
            output_value_id="v2",
            operator=OP_ADD,
            op_params={},
            input_value_ids=["v0", "v1"],
            output_typespec=_typespec((2, 3)),
        )],
        inputs=[("v0", _typespec((2, 3))), ("v1", _typespec((2, 3)))],
        outputs=["v2"],
    )
    program = generate(add_graph, "v2", ["v0", "v1"], "seed")
    assert program.output_gradients == ["seed", "seed"]


def test_reverse_traversal_matmul_single_wrt():
    graph = _matmul_graph((3, 4), (4, 5), (3, 5))
    program = generate(graph, "v2", ["v0"], "seed")
    assert "v0" in program.gradients
    assert "v1" not in program.gradients


# E2e matmul gradient tests live in test_e2e_matmul_gradient.py.
