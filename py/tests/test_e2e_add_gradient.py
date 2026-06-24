"""E2E tests for add-gradient execution via tc-server.

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
    OP_ADD,
    TensorGraph,
    TensorNodeRecord,
    generate,
)
from tinychain.autodiff.http_dispatcher import TcServerDispatcher, TensorLiteral


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _add_graph(lhs_shape, rhs_shape, out_shape):
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=OP_ADD,
        op_params={},
        input_value_ids=["v0", "v1"],
        output_typespec=_typespec(out_shape),
    )
    return TensorGraph(
        nodes=[node],
        inputs=[("v0", _typespec(lhs_shape)), ("v1", _typespec(rhs_shape))],
        outputs=["v2"],
    )


def test_e2e_add_gradient_no_broadcast(tc_server_url, tc_autodiff_route_root):
    """Exact-shape add: dA = dB = seed (no server ops needed, just passthrough)."""
    graph = _add_graph(lhs_shape=(2, 3), rhs_shape=(2, 3), out_shape=(2, 3))
    program = generate(graph, "v2", ["v0", "v1"], "seed")

    seed_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    seed = TensorLiteral.from_numpy(seed_array)
    dispatcher = TcServerDispatcher(tc_server_url, route_root=tc_autodiff_route_root)
    result = ExecutionScheduler(dispatcher).execute(program, values={"seed": seed})

    dx, dy = result.gradients
    np.testing.assert_array_equal(np.asarray(dx), seed_array)
    np.testing.assert_array_equal(np.asarray(dy), seed_array)


def test_e2e_add_gradient_broadcast_rhs(tc_server_url, tc_autodiff_route_root):
    """rhs=(1,3) broadcast to (2,3): dA=seed, dB=sum(seed, axis=0, keepdims=True)."""
    graph = _add_graph(lhs_shape=(2, 3), rhs_shape=(1, 3), out_shape=(2, 3))
    program = generate(graph, "v2", ["v0", "v1"], "seed")

    seed_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    seed = TensorLiteral.from_numpy(seed_array)
    dispatcher = TcServerDispatcher(tc_server_url, route_root=tc_autodiff_route_root)
    result = ExecutionScheduler(dispatcher).execute(program, values={"seed": seed})

    dx, dy = result.gradients
    np.testing.assert_array_equal(np.asarray(dx), seed_array)
    expected_dy = np.sum(seed_array, axis=0, keepdims=True)  # [[5., 7., 9.]]
    np.testing.assert_allclose(dy, expected_dy, rtol=1e-5)


def test_e2e_add_gradient_missing_leading_dims(tc_server_url, tc_autodiff_route_root):
    """rhs=(2,1) missing leading dim: dB reduced over axes (0, 2) to shape (2,1)."""
    graph = _add_graph(lhs_shape=(4, 2, 3), rhs_shape=(2, 1), out_shape=(4, 2, 3))
    program = generate(graph, "v2", ["v1"], "seed")

    seed_array = np.ones((4, 2, 3), dtype=np.float32)
    seed = TensorLiteral.from_numpy(seed_array)
    dispatcher = TcServerDispatcher(tc_server_url, route_root=tc_autodiff_route_root)
    result = ExecutionScheduler(dispatcher).execute(program, values={"seed": seed})

    (dy,) = result.gradients
    # sum over leading axis (0) and last axis (2), result shape (2, 1)
    expected_dy = np.sum(seed_array, axis=(0, 2), keepdims=False).reshape(2, 1)
    np.testing.assert_allclose(dy, expected_dy, rtol=1e-5)
