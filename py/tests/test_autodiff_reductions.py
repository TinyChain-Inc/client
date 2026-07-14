from __future__ import annotations

import numpy as np
import pytest

from tinychain.autodiff import (
    AutodiffError,
    BroadcastOperator,
    DerivativeMetadata,
    DerivativeProgram,
    DivOperator,
    ExecutionScheduler,
    MaxOperator,
    MeanOperator,
    MinOperator,
    ProductOperator,
    ReshapeOperator,
    SumOperator,
    TensorGraph,
    TensorNodeRecord,
    compile_derivative_program,
    generate,
)
from tinychain.autodiff.vjp import default_vjp_registry
from tests.autodiff_execution import NumpyAutodiffDispatcher


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _metadata() -> DerivativeMetadata:
    return DerivativeMetadata(
        source_graph_id="graph",
        transform_version="0.1.0",
        tensor_op_contract_version="0.1.0",
        wrt_signature=("v0",),
        seed_contract="seed matches output",
    )


def _program(nodes: list[TensorNodeRecord], outputs: list[str]) -> DerivativeProgram:
    return DerivativeProgram(
        nodes=nodes,
        gradients={"v0": outputs[0]},
        output_gradients=outputs,
        metadata=_metadata(),
    )


def _reduction_graph(operator, axes, keepdims, output_shape):
    return TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=operator,
                op_params={"axes": list(axes), "keepdims": keepdims},
                input_value_ids=["v0"],
                output_typespec=_typespec(output_shape),
            )
        ],
        inputs=[("v0", _typespec((2, 3, 4)))],
        outputs=["v1"],
    )


def _operator_types(nodes):
    return [type(node.operator) for node in nodes]


@pytest.mark.parametrize(
    ("operator_type", "route_name"),
    [
        (SumOperator, "sum"),
        (MeanOperator, "mean"),
        (MaxOperator, "max"),
        (MinOperator, "min"),
        (ProductOperator, "product"),
    ],
)
def test_reduction_operator_route_names(operator_type, route_name):
    assert operator_type().route_name == route_name


@pytest.mark.parametrize("operator", [SumOperator(), MeanOperator(), MaxOperator(), MinOperator(), ProductOperator()])
def test_default_registry_has_reduction_rules(operator):
    assert default_vjp_registry().has_rule(operator)


def test_sum_vjp_expands_explicit_axes_without_keepdims():
    graph = _reduction_graph(SumOperator(), axes=(1,), keepdims=False, output_shape=(2, 4))
    program = generate(graph, "v1", ["v0"], "seed")

    assert _operator_types(program.nodes) == [ReshapeOperator, BroadcastOperator]
    assert program.nodes[0].op_params == {"shape": [2, 1, 4]}
    assert program.nodes[1].op_params == {"shape": [2, 3, 4]}

    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    seed = np.arange(8, dtype=np.float32).reshape(2, 4)
    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values={"seed": seed, "v0": source})

    (gradient,) = result.gradients
    np.testing.assert_allclose(gradient, np.broadcast_to(seed.reshape(2, 1, 4), source.shape), rtol=1e-5)


def test_sum_vjp_keepdims_broadcasts_upstream():
    graph = _reduction_graph(SumOperator(), axes=(1,), keepdims=True, output_shape=(2, 1, 4))
    program = generate(graph, "v1", ["v0"], "seed")

    assert _operator_types(program.nodes) == [BroadcastOperator]
    assert program.nodes[0].op_params == {"shape": [2, 3, 4]}

    seed = np.arange(8, dtype=np.float32).reshape(2, 1, 4)
    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values={"seed": seed})

    (gradient,) = result.gradients
    np.testing.assert_allclose(gradient, np.broadcast_to(seed, (2, 3, 4)), rtol=1e-5)


def test_mean_vjp_divides_expanded_gradient_by_reduction_size():
    graph = _reduction_graph(MeanOperator(), axes=(0, 2), keepdims=False, output_shape=(3,))
    program = generate(graph, "v1", ["v0"], "seed")

    assert _operator_types(program.nodes) == [ReshapeOperator, BroadcastOperator, DivOperator]
    assert program.nodes[-1].op_params == {"right_literal": 8.0}

    seed = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values={"seed": seed})

    (gradient,) = result.gradients
    expected = np.broadcast_to(seed.reshape(1, 3, 1), (2, 3, 4)) / 8.0
    np.testing.assert_allclose(gradient, expected, rtol=1e-5)


def test_compile_derivative_program_maps_reduction_and_broadcast_routes():
    nodes = [
        TensorNodeRecord(
            node_id="n0",
            output_value_id="summed",
            operator=SumOperator(),
            op_params={"axes": [1], "keepdims": False},
            input_value_ids=["v0"],
        ),
        TensorNodeRecord(
            node_id="n1",
            output_value_id="expanded",
            operator=BroadcastOperator(),
            op_params={"shape": [2, 3, 4]},
            input_value_ids=["summed"],
        ),
        TensorNodeRecord(
            node_id="n2",
            output_value_id="averaged",
            operator=MeanOperator(),
            op_params={"axes": [0], "keepdims": True},
            input_value_ids=["expanded"],
        ),
    ]

    compiled = compile_derivative_program(_program(nodes, ["averaged"]))

    assert compiled.opdef.to_json() == {
        "/state/scalar/op/post": [
            ["summed", {"$v0/sum": {"axes": [1], "keepdims": False}}],
            ["expanded", {"$summed/broadcast": {"shape": [2, 3, 4]}}],
            ["averaged", {"$expanded/mean": {"axes": [0], "keepdims": True}}],
            ["result", [{"$averaged": []}]],
        ]
    }


@pytest.mark.parametrize(
    "operator",
    [MaxOperator(), MinOperator(), ProductOperator()],
)
def test_max_min_product_fail_with_explicit_unsupported_reduction_error(operator):
    graph = _reduction_graph(operator, axes=(1,), keepdims=False, output_shape=(2, 4))

    with pytest.raises(AutodiffError) as exc:
        generate(graph, "v1", ["v0"], "seed")

    assert exc.value.category == "unsupported_reduction"
    assert operator.route_name in exc.value.message


def test_sum_vjp_rejects_axes_none_until_shape_contract_exists():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=SumOperator(),
                op_params={"axes": None, "keepdims": False},
                input_value_ids=["v0"],
                output_typespec=_typespec(()),
            )
        ],
        inputs=[("v0", _typespec((2, 3)))],
        outputs=["v1"],
    )

    with pytest.raises(AutodiffError) as exc:
        generate(graph, "v1", ["v0"], "seed")

    assert exc.value.category == "unsupported_reduction"
    assert "explicit axes" in exc.value.message
