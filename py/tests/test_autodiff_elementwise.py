from __future__ import annotations

import numpy as np
import pytest

from tinychain.autodiff import (
    AddOperator,
    AutodiffError,
    BroadcastReduceOperator,
    DerivativeMetadata,
    DerivativeProgram,
    DivOperator,
    ExecutionScheduler,
    MulOperator,
    SubOperator,
    TensorGraph,
    TensorNodeRecord,
    compile_derivative_program,
    generate,
)
from tinychain.autodiff.vjp import default_vjp_registry
from tests.autodiff_execution import NumpyAutodiffDispatcher


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _elementwise_graph(operator, lhs_shape=(2, 3), rhs_shape=(2, 3), out_shape=(2, 3)):
    return TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v2",
                operator=operator,
                op_params={},
                input_value_ids=["v0", "v1"],
                output_typespec=_typespec(out_shape),
            )
        ],
        inputs=[("v0", _typespec(lhs_shape)), ("v1", _typespec(rhs_shape))],
        outputs=["v2"],
    )


def _metadata() -> DerivativeMetadata:
    return DerivativeMetadata(
        source_graph_id="graph",
        transform_version="0.1.0",
        tensor_op_contract_version="0.1.0",
        wrt_signature=("x",),
        seed_contract="seed matches output",
    )


def _program(nodes: list[TensorNodeRecord], outputs: list[str]) -> DerivativeProgram:
    return DerivativeProgram(
        nodes=nodes,
        gradients={"x": outputs[0]},
        output_gradients=outputs,
        metadata=_metadata(),
    )


def _operator_types(nodes):
    return [type(node.operator) for node in nodes]


@pytest.mark.parametrize(
    ("operator_type", "route_name"),
    [
        (SubOperator, "sub"),
        (MulOperator, "mul"),
        (DivOperator, "div"),
    ],
)
def test_elementwise_operator_route_names(operator_type, route_name):
    assert operator_type().route_name == route_name


@pytest.mark.parametrize("operator", [SubOperator(), MulOperator(), DivOperator()])
def test_default_registry_has_elementwise_rules(operator):
    assert default_vjp_registry().has_rule(operator)


def test_sub_vjp_same_shape_executes_correct_gradients():
    graph = _elementwise_graph(SubOperator())
    program = generate(graph, "v2", ["v0", "v1"], "seed")

    assert _operator_types(program.nodes) == [MulOperator]
    assert program.nodes[0].op_params == {"right_literal": -1.0}

    seed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values={"seed": seed})

    dx, dy = result.gradients
    np.testing.assert_array_equal(dx, seed)
    np.testing.assert_array_equal(dy, -seed)


def test_sub_vjp_broadcast_rhs_reduces_negated_gradient():
    graph = _elementwise_graph(SubOperator(), lhs_shape=(2, 3), rhs_shape=(1, 3), out_shape=(2, 3))
    program = generate(graph, "v2", ["v1"], "seed")

    assert _operator_types(program.nodes) == [MulOperator, BroadcastReduceOperator]
    assert program.nodes[1].op_params == {"target_shape": [1, 3]}

    seed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values={"seed": seed})

    (dy,) = result.gradients
    np.testing.assert_allclose(dy, -np.sum(seed, axis=0, keepdims=True), rtol=1e-5)


def test_mul_vjp_repeated_input_accumulates_both_partials():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=MulOperator(),
                op_params={},
                input_value_ids=["v0", "v0"],
                output_typespec=_typespec((2, 3)),
            )
        ],
        inputs=[("v0", _typespec((2, 3)))],
        outputs=["v1"],
    )
    program = generate(graph, "v1", ["v0"], "seed")

    assert _operator_types(program.nodes) == [MulOperator, MulOperator, AddOperator]
    lhs_partial, rhs_partial, accumulated = program.nodes
    assert accumulated.input_value_ids == [lhs_partial.output_value_id, rhs_partial.output_value_id]
    assert program.gradients == {"v0": accumulated.output_value_id}
    assert program.output_gradients == [accumulated.output_value_id]

    seed = np.array([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]], dtype=np.float32)
    value = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], dtype=np.float32)
    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(
        program,
        values={"seed": seed, "v0": value},
    )

    (gradient,) = result.gradients
    np.testing.assert_allclose(gradient, 2 * seed * value, rtol=1e-5)


def test_mul_vjp_broadcast_rhs_executes_correct_gradients():
    graph = _elementwise_graph(MulOperator(), lhs_shape=(2, 3), rhs_shape=(1, 3), out_shape=(2, 3))
    program = generate(graph, "v2", ["v0", "v1"], "seed")

    assert _operator_types(program.nodes) == [MulOperator, MulOperator, BroadcastReduceOperator]

    seed = np.array([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]], dtype=np.float32)
    lhs = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], dtype=np.float32)
    rhs = np.array([[3.0, 5.0, 7.0]], dtype=np.float32)
    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(
        program,
        values={"seed": seed, "v0": lhs, "v1": rhs},
    )

    dx, dy = result.gradients
    np.testing.assert_allclose(dx, seed * rhs, rtol=1e-5)
    np.testing.assert_allclose(dy, np.sum(seed * lhs, axis=0, keepdims=True), rtol=1e-5)


def test_div_vjp_same_shape_executes_correct_gradients():
    graph = _elementwise_graph(DivOperator())
    program = generate(graph, "v2", ["v0", "v1"], "seed")

    assert _operator_types(program.nodes) == [DivOperator, MulOperator, MulOperator, DivOperator, MulOperator]

    seed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    lhs = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], dtype=np.float32)
    rhs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(
        program,
        values={"seed": seed, "v0": lhs, "v1": rhs},
    )

    dx, dy = result.gradients
    np.testing.assert_allclose(dx, seed / rhs, rtol=1e-5)
    np.testing.assert_allclose(dy, -(seed * lhs) / (rhs * rhs), rtol=1e-5)


def test_div_single_wrt_prunes_unrequested_rhs_branch():
    graph = _elementwise_graph(DivOperator())
    program = generate(graph, "v2", ["v0"], "seed")

    assert set(program.gradients) == {"v0"}
    assert _operator_types(program.nodes) == [DivOperator]
    assert program.nodes[0].input_value_ids == ["seed", "v1"]
    assert "v1" not in program.gradients


def test_compile_derivative_program_maps_elementwise_routes():
    nodes = [
        TensorNodeRecord(
            node_id="n0",
            output_value_id="subbed",
            operator=SubOperator(),
            op_params={},
            input_value_ids=["seed", "other"],
        ),
        TensorNodeRecord(
            node_id="n1",
            output_value_id="negated",
            operator=MulOperator(),
            op_params={"right_literal": -1.0},
            input_value_ids=["subbed"],
        ),
        TensorNodeRecord(
            node_id="n2",
            output_value_id="divided",
            operator=DivOperator(),
            op_params={},
            input_value_ids=["negated", "denom"],
        ),
    ]
    compiled = compile_derivative_program(_program(nodes, ["divided"]))

    assert compiled.params == ("seed", "other", "denom")
    assert compiled.opdef.to_json() == {
        "/state/scalar/op/post": [
            ["subbed", {"$seed/sub": {"r": {"$other": []}}}],
            ["negated", {"$subbed/mul": {"r": -1.0}}],
            ["divided", {"$negated/div": {"r": {"$denom": []}}}],
            ["result", [{"$divided": []}]],
        ]
    }


@pytest.mark.parametrize("operator", [SubOperator(), MulOperator(), DivOperator()])
def test_elementwise_vjp_rejects_malformed_input_count(operator):
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v2",
                operator=operator,
                op_params={},
                input_value_ids=["v0"],
                output_typespec=_typespec((2, 3)),
            )
        ],
        inputs=[("v0", _typespec((2, 3)))],
        outputs=["v2"],
    )

    with pytest.raises(AutodiffError, match="requires exactly two inputs") as exc:
        generate(graph, "v2", ["v0"], "seed")

    assert exc.value.category == "malformed_derivative_ir"


def test_elementwise_vjp_rejects_broadcast_shape_mismatch():
    graph = _elementwise_graph(MulOperator(), lhs_shape=(2, 2), rhs_shape=(2, 3), out_shape=(2, 3))

    with pytest.raises(AutodiffError) as exc:
        generate(graph, "v2", ["v0"], "seed")

    assert exc.value.category == "broadcast_shape_mismatch"


def test_elementwise_seed_rejects_non_floating_dtype():
    graph = _elementwise_graph(SubOperator())

    with pytest.raises(AutodiffError) as exc:
        generate(graph, "v2", ["v0"], "seed", seed_typespec=_typespec((2, 3), dtype="u64"))

    assert exc.value.category == "dtype_not_differentiable"
