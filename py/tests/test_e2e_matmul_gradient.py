"""E2E tests for matmul-gradient execution via tc-server.

These tests run the full autodiff pipeline:
  generate() → DerivativeProgram → ExecutionScheduler.execute() → gradient tensors

Tests are skipped automatically when tc-server is not running at localhost:8702
or when the installed autodiff OpDef-backed route library is absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from tinychain.autodiff import (
    ExecutionScheduler,
    OP_MATMUL,
    TensorGraph,
    TensorNodeRecord,
    generate,
)
from tinychain.autodiff.http_dispatcher import TcServerDispatcher, TensorLiteral


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


def test_e2e_matmul_gradient_rank2(tc_server_url, tc_autodiff_route_root):
    """Rank-2 matmul gradient: Z = A@B, A=(2,3), B=(3,2), seed=(2,2).

    dA = dZ @ B^T = ones(2,2) @ ones(2,3) = 2*ones(2,3)
    dB = A^T @ dZ = ones(3,2) @ ones(2,2) = 2*ones(3,2)
    """
    a_shape, b_shape, z_shape = (2, 3), (3, 2), (2, 2)
    graph = _matmul_graph(a_shape, b_shape, z_shape)
    program = generate(graph, "v2", ["v0", "v1"], "seed", seed_typespec=_typespec(z_shape))

    a = TensorLiteral.from_numpy(np.ones(a_shape, dtype=np.float32))
    b = TensorLiteral.from_numpy(np.ones(b_shape, dtype=np.float32))
    seed = TensorLiteral.from_numpy(np.ones(z_shape, dtype=np.float32))

    dispatcher = TcServerDispatcher(tc_server_url, route_root=tc_autodiff_route_root)
    result = ExecutionScheduler(dispatcher).execute(program, values={"v0": a, "v1": b, "seed": seed})

    da, db = result.gradients
    np.testing.assert_allclose(np.asarray(da), 2.0 * np.ones(a_shape, dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(db), 2.0 * np.ones(b_shape, dtype=np.float32), rtol=1e-5)


def test_e2e_matmul_gradient_rank3_no_broadcast(tc_server_url, tc_autodiff_route_root):
    """Batched rank-3 matmul gradient (no broadcast): A=(2,3,4), B=(2,4,5), seed=(2,3,5).

    dA = dZ @ B^T = ones(2,3,5) @ ones(2,5,4) = 5*ones(2,3,4)  [inner dim=5]
    dB = A^T @ dZ = ones(2,4,3) @ ones(2,3,5) = 3*ones(2,4,5)  [inner dim=3]
    """
    a_shape, b_shape, z_shape = (2, 3, 4), (2, 4, 5), (2, 3, 5)
    graph = _matmul_graph(a_shape, b_shape, z_shape)
    program = generate(graph, "v2", ["v0", "v1"], "seed", seed_typespec=_typespec(z_shape))

    a = TensorLiteral.from_numpy(np.ones(a_shape, dtype=np.float32))
    b = TensorLiteral.from_numpy(np.ones(b_shape, dtype=np.float32))
    seed = TensorLiteral.from_numpy(np.ones(z_shape, dtype=np.float32))

    dispatcher = TcServerDispatcher(tc_server_url, route_root=tc_autodiff_route_root)
    result = ExecutionScheduler(dispatcher).execute(program, values={"v0": a, "v1": b, "seed": seed})

    da, db = result.gradients
    np.testing.assert_allclose(np.asarray(da), 5.0 * np.ones(a_shape, dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(db), 3.0 * np.ones(b_shape, dtype=np.float32), rtol=1e-5)


def test_e2e_matmul_gradient_batch_broadcast(tc_server_url, tc_autodiff_route_root):
    """Batch-broadcast matmul: A=(3,4) (no batch), B=(2,4,5), Z=(2,3,5), seed=(2,3,5).

    dA_intermediate = dZ @ B^T = ones(2,3,5) @ ones(2,5,4) = 5*ones(2,3,4)  [inner dim=5]
    dA = sum(dA_intermediate, axis=0) = 10*ones(3,4)  [broadcast-reduce over batch]
    dB = A^T @ dZ = ones(4,3) @ ones(2,3,5) = 3*ones(2,4,5)  [no reduction for B, inner dim=3]
    """
    a_shape, b_shape, z_shape = (3, 4), (2, 4, 5), (2, 3, 5)
    graph = _matmul_graph(a_shape, b_shape, z_shape)
    program = generate(graph, "v2", ["v0", "v1"], "seed", seed_typespec=_typespec(z_shape))

    a = TensorLiteral.from_numpy(np.ones(a_shape, dtype=np.float32))
    b = TensorLiteral.from_numpy(np.ones(b_shape, dtype=np.float32))
    seed = TensorLiteral.from_numpy(np.ones(z_shape, dtype=np.float32))

    dispatcher = TcServerDispatcher(tc_server_url, route_root=tc_autodiff_route_root)
    result = ExecutionScheduler(dispatcher).execute(program, values={"v0": a, "v1": b, "seed": seed})

    da, db = result.gradients
    np.testing.assert_allclose(np.asarray(da), 10.0 * np.ones(a_shape, dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(db), 3.0 * np.ones(b_shape, dtype=np.float32), rtol=1e-5)
