"""Transport-neutral execution tests for add derivative programs."""
from __future__ import annotations

import numpy as np

from tinychain.autodiff import AddOperator, ExecutionScheduler, TensorGraph, TensorNodeRecord, generate
from tests.autodiff_execution import NumpyAutodiffDispatcher


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _add_graph(lhs_shape, rhs_shape, out_shape):
    return TensorGraph(
        nodes=[TensorNodeRecord(
            node_id="n0",
            output_value_id="v2",
            operator=AddOperator(),
            op_params={},
            input_value_ids=["v0", "v1"],
            output_typespec=_typespec(out_shape),
        )],
        inputs=[("v0", _typespec(lhs_shape)), ("v1", _typespec(rhs_shape))],
        outputs=["v2"],
    )


def test_execution_add_gradient_no_broadcast():
    graph = _add_graph(lhs_shape=(2, 3), rhs_shape=(2, 3), out_shape=(2, 3))
    program = generate(graph, "v2", ["v0", "v1"], "seed")

    seed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values={"seed": seed})

    dx, dy = result.gradients
    np.testing.assert_array_equal(dx, seed)
    np.testing.assert_array_equal(dy, seed)


def test_execution_add_gradient_broadcast_rhs():
    graph = _add_graph(lhs_shape=(2, 3), rhs_shape=(1, 3), out_shape=(2, 3))
    program = generate(graph, "v2", ["v0", "v1"], "seed")

    seed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values={"seed": seed})

    dx, dy = result.gradients
    np.testing.assert_array_equal(dx, seed)
    np.testing.assert_allclose(dy, np.sum(seed, axis=0, keepdims=True), rtol=1e-5)


def test_execution_add_gradient_missing_leading_dims():
    graph = _add_graph(lhs_shape=(4, 2, 3), rhs_shape=(2, 1), out_shape=(4, 2, 3))
    program = generate(graph, "v2", ["v1"], "seed")

    seed = np.ones((4, 2, 3), dtype=np.float32)
    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values={"seed": seed})

    (dy,) = result.gradients
    expected = np.sum(seed, axis=(0, 2), keepdims=False).reshape(2, 1)
    np.testing.assert_allclose(dy, expected, rtol=1e-5)
