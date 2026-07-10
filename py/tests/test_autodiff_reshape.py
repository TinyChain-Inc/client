from __future__ import annotations

import numpy as np
import pytest

from tinychain.autodiff import (
    AutodiffError,
    DerivativeMetadata,
    DerivativeProgram,
    ExecutionScheduler,
    ReshapeOperator,
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


def _reshape_graph():
    return TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=ReshapeOperator(),
                op_params={"shape": [3, 2]},
                input_value_ids=["v0"],
                output_typespec=_typespec((3, 2)),
            )
        ],
        inputs=[("v0", _typespec((2, 3)))],
        outputs=["v1"],
    )


def test_reshape_operator_route_name_and_registry():
    assert ReshapeOperator().route_name == "reshape"
    assert default_vjp_registry().has_rule(ReshapeOperator())


def test_reshape_vjp_reshapes_upstream_back_to_input_shape():
    graph = _reshape_graph()
    program = generate(graph, "v1", ["v0"], "seed")

    assert len(program.nodes) == 1
    node = program.nodes[0]
    assert isinstance(node.operator, ReshapeOperator)
    assert node.op_params == {"shape": [2, 3]}

    seed = np.arange(6, dtype=np.float32).reshape(3, 2)
    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(program, values={"seed": seed})

    (gradient,) = result.gradients
    np.testing.assert_allclose(gradient, seed.reshape(2, 3), rtol=1e-5)


def test_compile_derivative_program_maps_reshape_route():
    nodes = [
        TensorNodeRecord(
            node_id="n0",
            output_value_id="reshaped",
            operator=ReshapeOperator(),
            op_params={"shape": [2, 3]},
            input_value_ids=["seed"],
        )
    ]

    compiled = compile_derivative_program(_program(nodes, ["reshaped"]))

    assert compiled.opdef.to_json() == {
        "/state/scalar/op/post": [
            ["reshaped", {"$seed/reshape": {"shape": [2, 3]}}],
            ["result", [{"$reshaped": []}]],
        ]
    }


def test_reshape_vjp_rejects_malformed_input_count():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=ReshapeOperator(),
                op_params={"shape": [2, 3]},
                input_value_ids=["v0", "v2"],
                output_typespec=_typespec((2, 3)),
            )
        ],
        inputs=[("v0", _typespec((6,))), ("v2", _typespec((1,)))],
        outputs=["v1"],
    )

    with pytest.raises(AutodiffError) as exc:
        generate(graph, "v1", ["v0"], "seed")

    assert exc.value.category == "malformed_derivative_ir"
    assert "reshape VJP requires exactly one input" in exc.value.message
