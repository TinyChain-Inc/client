"""Transport-neutral execution tests for matmul derivative programs."""
from __future__ import annotations

import numpy as np

from tinychain.autodiff import (
    ExecutionScheduler,
    MatmulOperator,
    TensorGraph,
    TensorNodeRecord,
    TransposeOperator,
    generate,
)
from tests.autodiff_execution import NumpyAutodiffDispatcher


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _matmul_graph(lhs_shape, rhs_shape, out_shape):
    return TensorGraph(
        nodes=[TensorNodeRecord(
            node_id="n0",
            output_value_id="v2",
            operator=MatmulOperator(),
            op_params={},
            input_value_ids=["v0", "v1"],
            output_typespec=_typespec(out_shape),
        )],
        inputs=[("v0", _typespec(lhs_shape)), ("v1", _typespec(rhs_shape))],
        outputs=["v2"],
    )


def _execute(graph, wrt, values, seed_shape):
    program = generate(graph, "v2", wrt, "seed", seed_typespec=_typespec(seed_shape))
    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values=values)
    return program, result


def test_execution_matmul_gradient_rank2():
    a_shape, b_shape, z_shape = (2, 3), (3, 2), (2, 2)
    graph = _matmul_graph(a_shape, b_shape, z_shape)
    values = {
        "v0": np.ones(a_shape, dtype=np.float32),
        "v1": np.ones(b_shape, dtype=np.float32),
        "seed": np.ones(z_shape, dtype=np.float32),
    }

    _, result = _execute(graph, ["v0", "v1"], values, z_shape)

    da, db = result.gradients
    np.testing.assert_allclose(da, 2.0 * np.ones(a_shape, dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(db, 2.0 * np.ones(b_shape, dtype=np.float32), rtol=1e-5)


def test_execution_matmul_gradient_single_wrt_fixed_rhs():
    # v1 remains a runtime input during execution, not a literal captured by the graph.
    a_shape, b_shape, z_shape = (2, 3), (3, 2), (2, 2)
    graph = _matmul_graph(a_shape, b_shape, z_shape)
    values = {
        "v0": np.ones(a_shape, dtype=np.float32),
        "v1": np.ones(b_shape, dtype=np.float32),
        "seed": np.ones(z_shape, dtype=np.float32),
    }

    program, result = _execute(graph, ["v0"], values, z_shape)

    assert set(program.gradients) == {"v0"}
    assert [type(node.operator) for node in program.nodes] == [TransposeOperator, MatmulOperator]
    (da,) = result.gradients
    np.testing.assert_allclose(da, 2.0 * np.ones(a_shape, dtype=np.float32), rtol=1e-5)


def test_execution_matmul_gradient_single_wrt_fixed_lhs():
    # v0 remains a runtime input during execution, not a literal captured by the graph.
    a_shape, b_shape, z_shape = (2, 3), (3, 2), (2, 2)
    graph = _matmul_graph(a_shape, b_shape, z_shape)
    values = {
        "v0": np.ones(a_shape, dtype=np.float32),
        "v1": np.ones(b_shape, dtype=np.float32),
        "seed": np.ones(z_shape, dtype=np.float32),
    }

    program, result = _execute(graph, ["v1"], values, z_shape)

    assert set(program.gradients) == {"v1"}
    assert [type(node.operator) for node in program.nodes] == [TransposeOperator, MatmulOperator]
    (db,) = result.gradients
    np.testing.assert_allclose(db, 2.0 * np.ones(b_shape, dtype=np.float32), rtol=1e-5)


def test_execution_matmul_gradient_rank3_no_broadcast():
    a_shape, b_shape, z_shape = (2, 3, 4), (2, 4, 5), (2, 3, 5)
    graph = _matmul_graph(a_shape, b_shape, z_shape)
    values = {
        "v0": np.ones(a_shape, dtype=np.float32),
        "v1": np.ones(b_shape, dtype=np.float32),
        "seed": np.ones(z_shape, dtype=np.float32),
    }

    _, result = _execute(graph, ["v0", "v1"], values, z_shape)

    da, db = result.gradients
    np.testing.assert_allclose(da, 5.0 * np.ones(a_shape, dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(db, 3.0 * np.ones(b_shape, dtype=np.float32), rtol=1e-5)


def test_execution_matmul_gradient_batch_broadcast():
    a_shape, b_shape, z_shape = (3, 4), (2, 4, 5), (2, 3, 5)
    graph = _matmul_graph(a_shape, b_shape, z_shape)
    values = {
        "v0": np.ones(a_shape, dtype=np.float32),
        "v1": np.ones(b_shape, dtype=np.float32),
        "seed": np.ones(z_shape, dtype=np.float32),
    }

    _, result = _execute(graph, ["v0", "v1"], values, z_shape)

    da, db = result.gradients
    np.testing.assert_allclose(da, 10.0 * np.ones(a_shape, dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(db, 3.0 * np.ones(b_shape, dtype=np.float32), rtol=1e-5)
