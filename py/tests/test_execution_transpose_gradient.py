"""Transport-neutral execution tests for transpose derivative programs."""
from __future__ import annotations

import numpy as np

from tinychain.autodiff import (
    AddOperator,
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


def _transpose_graph(input_shape, perm, dtype="f32"):
    output_shape = tuple(input_shape[axis] for axis in perm)
    return TensorGraph(
        nodes=[TensorNodeRecord(
            node_id="n0",
            output_value_id="v1",
            operator=TransposeOperator(),
            op_params={"perm": list(perm)},
            input_value_ids=["v0"],
            output_typespec=_typespec(output_shape, dtype),
        )],
        inputs=[("v0", _typespec(input_shape, dtype))],
        outputs=["v1"],
    )


def test_execution_transpose_gradient_2d():
    graph = _transpose_graph((2, 3), [1, 0])
    seed = np.arange(6, dtype=np.float32).reshape(3, 2)
    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec(seed.shape))

    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values={"seed": seed})

    (dx,) = result.gradients
    np.testing.assert_allclose(dx, seed.T, rtol=1e-5)


def test_execution_transpose_gradient_nontrivial_3d_permutation():
    graph = _transpose_graph((2, 3, 4), [2, 0, 1])
    seed = np.ones((4, 2, 3), dtype=np.float32)
    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec(seed.shape))

    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values={"seed": seed})

    (dx,) = result.gradients
    np.testing.assert_allclose(dx, np.ones((2, 3, 4), dtype=np.float32), rtol=1e-5)


def test_execution_transpose_gradient_preserves_f64_values():
    graph = _transpose_graph((3, 3), [1, 0], dtype="f64")
    seed = np.eye(3, dtype=np.float64)
    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec(seed.shape, "f64"))

    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values={"seed": seed})

    (dx,) = result.gradients
    np.testing.assert_allclose(dx, seed, rtol=1e-12)
    assert dx.dtype == np.float64


def test_execution_composite_add_transpose_matmul_gradient():
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
                operator=TransposeOperator(),
                op_params={"perm": [1, 0]},
                input_value_ids=["v2"],
                output_typespec=_typespec((3, 2)),
            ),
            TensorNodeRecord(
                node_id="n2",
                output_value_id="v5",
                operator=MatmulOperator(),
                op_params={},
                input_value_ids=["v3", "v4"],
                output_typespec=_typespec((3, 4)),
            ),
        ],
        inputs=[("v0", _typespec((2, 3))), ("v1", _typespec((2, 3))), ("v4", _typespec((2, 4)))],
        outputs=["v5"],
    )
    seed = np.ones((3, 4), dtype=np.float32)
    values = {
        "v3": 2.0 * np.ones((3, 2), dtype=np.float32),
        "v4": np.ones((2, 4), dtype=np.float32),
        "seed": seed,
    }
    program = generate(graph, "v5", ["v0", "v1", "v4"], "seed", seed_typespec=_typespec(seed.shape))

    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values=values)

    da, db, dc = result.gradients
    np.testing.assert_allclose(da, 4.0 * np.ones((2, 3), dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(db, 4.0 * np.ones((2, 3), dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(dc, 6.0 * np.ones((2, 4), dtype=np.float32), rtol=1e-5)
