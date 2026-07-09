from __future__ import annotations

import pytest
import tinychain as tc

from tinychain.autodiff import (
    AddOperator,
    AutodiffError,
    BroadcastReduceOperator,
    DerivativeMetadata,
    DerivativeProgram,
    MatmulOperator,
    TensorNodeRecord,
    TensorOperator,
    TransposeOperator,
    compile_derivative_program,
)


def _metadata() -> DerivativeMetadata:
    return DerivativeMetadata(
        source_graph_id="graph",
        transform_version="0.1.0",
        tensor_op_contract_version="0.1.0",
        wrt_signature=("x",),
        seed_contract="seed matches output",
    )


def _program(nodes: list[TensorNodeRecord], outputs: list[str | None]) -> DerivativeProgram:
    return DerivativeProgram(
        nodes=nodes,
        gradients={"x": outputs[0]} if outputs and outputs[0] is not None else {},
        output_gradients=outputs,
        metadata=_metadata(),
    )


def _json(program: DerivativeProgram) -> dict[str, object]:
    return compile_derivative_program(program).opdef.to_json()


def test_compile_single_add_program_to_route_ir() -> None:
    program = _program(
        [
            TensorNodeRecord(
                node_id="n0",
                output_value_id="dx",
                operator=AddOperator(),
                op_params={},
                input_value_ids=["seed", "other"],
            )
        ],
        ["dx"],
    )

    compiled = compile_derivative_program(program)

    assert compiled.params == ("seed", "other")
    assert compiled.results == ("dx",)
    assert compiled.opdef.to_json() == {
        "/state/scalar/op/post": [
            ["dx", {"$seed/add": {"r": {"$other": []}}}],
            ["result", [{"$dx": []}]],
        ]
    }


def test_compile_multi_node_existing_operators_deterministically() -> None:
    nodes = [
        TensorNodeRecord(
            node_id="n0",
            output_value_id="bt",
            operator=BroadcastReduceOperator(),
            op_params={"target_shape": [2, 3]},
            input_value_ids=["seed"],
        ),
        TensorNodeRecord(
            node_id="n1",
            output_value_id="tr",
            operator=TransposeOperator(),
            op_params={"permutation": [1, 0]},
            input_value_ids=["bt"],
        ),
        TensorNodeRecord(
            node_id="n2",
            output_value_id="mm",
            operator=MatmulOperator(),
            op_params={},
            input_value_ids=["tr", "weight"],
        ),
    ]
    first = _program(nodes, ["mm", "seed"])
    second = _program(list(nodes), ["mm", "seed"])

    assert _json(first) == _json(second)
    assert compile_derivative_program(first).params == ("seed", "weight")
    assert _json(first) == {
        "/state/scalar/op/post": [
            ["bt", {"$seed/broadcast_reduce": {"target_shape": [2, 3]}}],
            ["tr", {"$bt/transpose": {"permutation": [1, 0]}}],
            ["mm", {"$tr/matmul": {"r": {"$weight": []}}}],
            ["result", [{"$mm": []}, {"$seed": []}]],
        ]
    }


def test_compile_transpose_accepts_current_perm_metadata() -> None:
    program = _program(
        [
            TensorNodeRecord(
                node_id="n0",
                output_value_id="transposed",
                operator=TransposeOperator(),
                op_params={"perm": [1, 0]},
                input_value_ids=["seed"],
            )
        ],
        ["transposed"],
    )

    assert _json(program) == {
        "/state/scalar/op/post": [
            ["transposed", {"$seed/transpose": {"permutation": [1, 0]}}],
            ["result", [{"$transposed": []}]],
        ]
    }


def test_compile_exports_api_from_autodiff_package() -> None:
    assert tc.autodiff.compile_derivative_program is compile_derivative_program


@pytest.mark.parametrize(
    "program, match",
    [
        (
            _program(
                [
                    TensorNodeRecord(
                        node_id="n0",
                        output_value_id="dx",
                        operator=TensorOperator("unknown"),
                        op_params={},
                        input_value_ids=["seed"],
                    )
                ],
                ["dx"],
            ),
            "unsupported derivative operator",
        ),
        (
            _program(
                [
                    TensorNodeRecord(
                        node_id="n0",
                        output_value_id="dx",
                        operator=AddOperator(),
                        op_params={},
                        input_value_ids=["seed"],
                    )
                ],
                ["dx"],
            ),
            "expected 2 inputs",
        ),
        (
            _program(
                [
                    TensorNodeRecord(
                        node_id="n0",
                        output_value_id="dx",
                        operator=AddOperator(),
                        op_params={},
                        input_value_ids=["seed", "other"],
                    ),
                    TensorNodeRecord(
                        node_id="n1",
                        output_value_id="dx",
                        operator=AddOperator(),
                        op_params={},
                        input_value_ids=["seed", "other"],
                    ),
                ],
                ["dx"],
            ),
            "duplicate output value id",
        ),
        (
            _program(
                [
                    TensorNodeRecord(
                        node_id="n0",
                        output_value_id="dx",
                        operator=TransposeOperator(),
                        op_params={},
                        input_value_ids=["seed"],
                    )
                ],
                ["dx"],
            ),
            "missing param 'perm'",
        ),
        (_program([], [None]), "output gradients cannot contain None"),
    ],
)
def test_compile_rejects_malformed_ir(program: DerivativeProgram, match: str) -> None:
    with pytest.raises(AutodiffError, match=match) as exc:
        compile_derivative_program(program)

    assert exc.value.category == "malformed_derivative_ir"
