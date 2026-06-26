"""E2E tests for transpose-gradient execution via tc-server.

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
    OP_TRANSPOSE,
    TensorGraph,
    TensorNodeRecord,
    generate,
)
from tinychain.autodiff.http_dispatcher import TcServerDispatcher, TensorLiteral


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def test_e2e_transpose_gradient_2d_f32(tc_server_url, tc_autodiff_route_root):
    """2D f32 transpose gradient: Z = transpose(A, [1,0]), A=(2,3), seed=arange(6).reshape(3,2).

    dA = transpose(seed, [1,0]) = seed.T = arange(6).reshape(2,3)
    """
    input_shape = (2, 3)
    perm = [1, 0]
    output_shape = (3, 2)
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v1",
        operator=OP_TRANSPOSE,
        op_params={"perm": perm},
        input_value_ids=["v0"],
        output_typespec=_typespec(output_shape),
    )
    graph = TensorGraph(nodes=[node], inputs=[("v0", _typespec(input_shape))], outputs=["v1"])
    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec(output_shape))

    seed_array = np.arange(6, dtype=np.float32).reshape(3, 2)
    dispatcher = TcServerDispatcher(tc_server_url, route_root=tc_autodiff_route_root)
    result = ExecutionScheduler(dispatcher).execute(
        program,
        values={"seed": TensorLiteral.from_numpy(seed_array)},
    )

    (da,) = result.gradients
    np.testing.assert_allclose(np.asarray(da), seed_array.T, rtol=1e-5)


def test_e2e_transpose_gradient_3d_nontrivial_perm(tc_server_url, tc_autodiff_route_root):
    """3D f32 transpose gradient: Z = transpose(A, [2,0,1]), A=(2,3,4), seed=ones(4,2,3).

    Inverse perm of [2,0,1] is [1,2,0].
    dA = transpose(seed, [1,2,0]) = ones(2,3,4)
    """
    input_shape = (2, 3, 4)
    perm = [2, 0, 1]
    output_shape = (4, 2, 3)
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v1",
        operator=OP_TRANSPOSE,
        op_params={"perm": perm},
        input_value_ids=["v0"],
        output_typespec=_typespec(output_shape),
    )
    graph = TensorGraph(nodes=[node], inputs=[("v0", _typespec(input_shape))], outputs=["v1"])
    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec(output_shape))

    seed_array = np.ones((4, 2, 3), dtype=np.float32)
    dispatcher = TcServerDispatcher(tc_server_url, route_root=tc_autodiff_route_root)
    result = ExecutionScheduler(dispatcher).execute(
        program,
        values={"seed": TensorLiteral.from_numpy(seed_array)},
    )

    (da,) = result.gradients
    np.testing.assert_allclose(np.asarray(da), np.ones(input_shape, dtype=np.float32), rtol=1e-5)


def test_e2e_transpose_gradient_f64_dtype(tc_server_url, tc_autodiff_route_root):
    """f64 dtype preservation: Z = transpose(A, [1,0]), A=(3,3) f64, seed=eye(3) f64.

    dA = transpose(seed, [1,0]) = eye(3) (symmetric); dtype must be f64.
    """
    input_shape = (3, 3)
    perm = [1, 0]
    output_shape = (3, 3)
    dtype = "f64"
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v1",
        operator=OP_TRANSPOSE,
        op_params={"perm": perm},
        input_value_ids=["v0"],
        output_typespec=_typespec(output_shape, dtype=dtype),
    )
    graph = TensorGraph(
        nodes=[node],
        inputs=[("v0", _typespec(input_shape, dtype=dtype))],
        outputs=["v1"],
    )
    program = generate(
        graph,
        "v1",
        ["v0"],
        "seed",
        seed_typespec=_typespec(output_shape, dtype=dtype),
    )

    seed_array = np.eye(3, dtype=np.float64)
    dispatcher = TcServerDispatcher(tc_server_url, route_root=tc_autodiff_route_root)
    result = ExecutionScheduler(dispatcher).execute(
        program,
        values={"seed": TensorLiteral.from_numpy(seed_array)},
    )

    (da,) = result.gradients
    np.testing.assert_allclose(np.asarray(da), np.eye(3, dtype=np.float64), rtol=1e-12)
    assert np.asarray(da).dtype == np.float64


def test_e2e_transpose_gradient_identity_perm(tc_server_url, tc_autodiff_route_root):
    """Identity permutation gradient: Z = transpose(A, [0,1]), A=(2,3), seed=arange(6).reshape(2,3).

    Inverse perm of [0,1] is [0,1].
    dA = transpose(seed, [0,1]) = seed
    """
    input_shape = (2, 3)
    perm = [0, 1]
    output_shape = (2, 3)
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v1",
        operator=OP_TRANSPOSE,
        op_params={"perm": perm},
        input_value_ids=["v0"],
        output_typespec=_typespec(output_shape),
    )
    graph = TensorGraph(nodes=[node], inputs=[("v0", _typespec(input_shape))], outputs=["v1"])
    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec(output_shape))

    seed_array = np.arange(6, dtype=np.float32).reshape(2, 3)
    dispatcher = TcServerDispatcher(tc_server_url, route_root=tc_autodiff_route_root)
    result = ExecutionScheduler(dispatcher).execute(
        program,
        values={"seed": TensorLiteral.from_numpy(seed_array)},
    )

    (da,) = result.gradients
    np.testing.assert_allclose(np.asarray(da), seed_array, rtol=1e-5)


def test_e2e_transpose_add_chain_gradient(tc_server_url, tc_autodiff_route_root):
    """Transpose → add chain: v0=(2,3), v1=(3,2), v2=transpose(v0,[1,0]) → v3=add(v2,v1), seed=ones(3,2).

    dv2 = seed = ones(3,2)
    dv0 = transpose(dv2, [1,0]) = ones(2,3)
    dv1 = dv2 = ones(3,2)
    """
    node0 = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=OP_TRANSPOSE,
        op_params={"perm": [1, 0]},
        input_value_ids=["v0"],
        output_typespec=_typespec((3, 2)),
    )
    node1 = TensorNodeRecord(
        node_id="n1",
        output_value_id="v3",
        operator=OP_ADD,
        op_params={},
        input_value_ids=["v2", "v1"],
        output_typespec=_typespec((3, 2)),
    )
    graph = TensorGraph(
        nodes=[node0, node1],
        inputs=[("v0", _typespec((2, 3))), ("v1", _typespec((3, 2)))],
        outputs=["v3"],
    )
    program = generate(graph, "v3", ["v0", "v1"], "seed", seed_typespec=_typespec((3, 2)))

    seed_array = np.ones((3, 2), dtype=np.float32)
    dispatcher = TcServerDispatcher(tc_server_url, route_root=tc_autodiff_route_root)
    result = ExecutionScheduler(dispatcher).execute(
        program,
        values={"seed": TensorLiteral.from_numpy(seed_array)},
    )

    dv0, dv1 = result.gradients
    np.testing.assert_allclose(np.asarray(dv0), np.ones((2, 3), dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(dv1), np.ones((3, 2), dtype=np.float32), rtol=1e-5)
