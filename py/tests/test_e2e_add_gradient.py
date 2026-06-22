"""E2E tests for add-gradient execution via tc-server.

These tests run the full autodiff pipeline:
  generate() → DerivativeProgram → ExecutionScheduler.execute() → gradient tensors

Tests are skipped automatically when tc-server is not running at localhost:8702.
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
from tinychain.autodiff.http_dispatcher import TcServerDispatcher


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _add_graph(lhs_shape, rhs_shape, out_shape):
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


def test_e2e_add_gradient_no_broadcast(tc_server_url):
    """Exact-shape add: dA = dB = seed (no server ops needed, just passthrough)."""
    graph = _add_graph(lhs_shape=(2, 3), rhs_shape=(2, 3), out_shape=(2, 3))
    program = generate(graph, "z", ["x", "y"], "seed")

    seed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    dispatcher = TcServerDispatcher(tc_server_url)
    result = ExecutionScheduler(dispatcher).execute(program, values={"seed": seed})

    dx, dy = result.gradients
    np.testing.assert_array_equal(dx, seed)
    np.testing.assert_array_equal(dy, seed)


def test_e2e_add_gradient_broadcast_rhs(tc_server_url):
    """rhs=(1,3) broadcast to (2,3): dA=seed, dB=sum(seed, axis=0, keepdims=True)."""
    graph = _add_graph(lhs_shape=(2, 3), rhs_shape=(1, 3), out_shape=(2, 3))
    program = generate(graph, "z", ["x", "y"], "seed")

    seed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    dispatcher = TcServerDispatcher(tc_server_url)
    result = ExecutionScheduler(dispatcher).execute(program, values={"seed": seed})

    dx, dy = result.gradients
    np.testing.assert_array_equal(dx, seed)
    expected_dy = np.sum(seed, axis=0, keepdims=True)  # [[5., 7., 9.]]
    np.testing.assert_allclose(dy, expected_dy, rtol=1e-5)


def test_e2e_add_gradient_missing_leading_dims(tc_server_url):
    """rhs=(2,1) missing leading dim: dB reduced over axes (0, 2) to shape (2,1)."""
    graph = _add_graph(lhs_shape=(4, 2, 3), rhs_shape=(2, 1), out_shape=(4, 2, 3))
    program = generate(graph, "z", ["y"], "seed")

    seed = np.ones((4, 2, 3), dtype=np.float32)
    dispatcher = TcServerDispatcher(tc_server_url)
    result = ExecutionScheduler(dispatcher).execute(program, values={"seed": seed})

    (dy,) = result.gradients
    # sum over leading axis (0) and last axis (2), result shape (2, 1)
    expected_dy = np.sum(seed, axis=(0, 2), keepdims=False).reshape(2, 1)
    np.testing.assert_allclose(dy, expected_dy, rtol=1e-5)
