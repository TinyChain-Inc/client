from __future__ import annotations

import numpy as np
import pytest

from tinychain.autodiff import (
    AutodiffError,
    BroadcastOperator,
    DerivativeExecutionDispatcher,
    DerivativeMetadata,
    DerivativeProgram,
    ExecutionScheduler,
    ReshapeOperator,
    SumOperator,
    TensorGraph,
    TensorNodeRecord,
    build_derivative_execution_library,
    compile_derivative_program,
    generate,
)
from tinychain.autodiff.seed import SeedValidator
from tinychain.library import library_definition
from tests.autodiff_execution import NumpyAutodiffDispatcher


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def test_seed_validator_preserves_concrete_shape_mismatch():
    with pytest.raises(AutodiffError) as exc:
        SeedValidator().validate(
            seed_typespec=_typespec((3, 2)),
            output_typespec=_typespec((2, 3)),
        )

    assert exc.value.category == "seed_shape_mismatch"


def test_seed_validator_accepts_concrete_seed_for_symbolic_output_rank():
    SeedValidator().validate(
        seed_typespec=_typespec((2, 3)),
        output_typespec=_typespec(("N", "D")),
    )


def test_sum_vjp_allows_symbolic_rank_for_non_keepdims_expansion():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=SumOperator(),
                op_params={"axes": [1], "keepdims": False},
                input_value_ids=["v0"],
                output_typespec=_typespec(("N",)),
            )
        ],
        inputs=[("v0", _typespec(("N", "D")))],
        outputs=["v1"],
    )

    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec((2,)))

    assert [type(node.operator) for node in program.nodes] == [ReshapeOperator, BroadcastOperator]
    assert program.nodes[0].op_params == {"shape": ["N", 1]}
    assert program.nodes[1].op_params == {"shape": ["N", "D"]}


def test_compile_rejects_unresolved_symbolic_route_shapes():
    program = generate(
        TensorGraph(
            nodes=[
                TensorNodeRecord(
                    node_id="n0",
                    output_value_id="v1",
                    operator=SumOperator(),
                    op_params={"axes": [1], "keepdims": False},
                    input_value_ids=["v0"],
                    output_typespec=_typespec(("N",)),
                )
            ],
            inputs=[("v0", _typespec(("N", "D")))],
            outputs=["v1"],
        ),
        "v1",
        ["v0"],
        "seed",
        seed_typespec=_typespec((2,)),
    )

    with pytest.raises(AutodiffError) as exc:
        compile_derivative_program(program)

    assert exc.value.category == "unresolved_symbolic_shape"
    assert "N" in exc.value.message


def test_compile_resolves_symbolic_route_shapes_from_explicit_bindings():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=SumOperator(),
                op_params={"axes": [1], "keepdims": False},
                input_value_ids=["v0"],
                output_typespec=_typespec(("N",)),
            )
        ],
        inputs=[("v0", _typespec(("N", "D")))],
        outputs=["v1"],
    )
    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec((2,)))

    compiled = compile_derivative_program(program, symbol_bindings={"N": 2, "D": 3})

    assert compiled.opdef.to_json() == {
        "/state/scalar/op/post": [
            ["d0", {"$seed/reshape": {"shape": [2, 1]}}],
            ["d1", {"$d0/broadcast": {"shape": [2, 3]}}],
            ["result", [{"$d1": []}]],
        ]
    }


def test_execution_library_defers_symbolic_shape_params_to_runtime():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=ReshapeOperator(),
                op_params={"shape": ["M", "N"]},
                input_value_ids=["v0"],
                output_typespec=_typespec(("D", "M")),
            )
        ],
        inputs=[("v0", _typespec(("M", "N")))],
        outputs=["v1"],
    )
    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec((3, 2)))

    library_cls = build_derivative_execution_library(
        publisher="autodiff-devco",
        class_name="SymbolicShapeDerivativeExecution",
        version="0.1.0",
        program=program,
    )

    assert getattr(library_cls, "__tc_derivative_params__") == ("seed", "__tc_shape_M", "__tc_shape_N")
    assert getattr(library_cls, "__tc_derivative_shape_params__") == {
        "__tc_shape_M": "M",
        "__tc_shape_N": "N",
    }
    assert library_definition(library_cls) == {
        "/lib/autodiff-devco/symbolic_shape_derivative_execution/0.1.0": {
            "execute": {
                "/state/scalar/op/post": [
                    ["d0", {"$seed/reshape": {"shape": [{"$__tc_shape_M": []}, {"$__tc_shape_N": []}]}}],
                    ["result", [{"$d0": []}]],
                ]
            }
        }
    }

    route_calls: list[dict[str, object]] = []

    def execute_route(call_values: dict[str, object]) -> list[object]:
        route_calls.append(call_values)
        return [np.reshape(call_values["seed"], (call_values["__tc_shape_M"], call_values["__tc_shape_N"]))]

    result = DerivativeExecutionDispatcher(
        route_executor=execute_route,
        params=tuple(getattr(library_cls, "__tc_derivative_params__")),
        shape_params=getattr(library_cls, "__tc_derivative_shape_params__"),
    ).execute(
        program,
        values={
            "seed": np.arange(6, dtype=np.float32).reshape(3, 2),
            "v0": np.arange(6, dtype=np.float32).reshape(2, 3),
        },
    )

    assert len(route_calls) == 1
    assert route_calls[0]["__tc_shape_M"] == 2
    assert route_calls[0]["__tc_shape_N"] == 3
    np.testing.assert_allclose(
        result.gradients[0],
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )


def test_execution_library_namespaces_shape_params_away_from_value_ids():
    program = DerivativeProgram(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="gradient",
                operator=ReshapeOperator(),
                op_params={"shape": ["M", "N"]},
                input_value_ids=["M"],
                output_typespec=_typespec(("M", "N")),
            )
        ],
        gradients={"x": "gradient"},
        output_gradients=["gradient"],
        metadata=DerivativeMetadata(
            source_graph_id="graph",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=("x",),
            seed_contract="seed matches output",
        ),
        value_typespecs={"M": _typespec(("M", "N"))},
    )

    library_cls = build_derivative_execution_library(
        publisher="autodiff-devco",
        class_name="SymbolicShapeParamCollisionExecution",
        version="0.1.0",
        program=program,
    )

    assert getattr(library_cls, "__tc_derivative_params__") == (
        "M",
        "__tc_shape_M",
        "__tc_shape_N",
    )
    assert getattr(library_cls, "__tc_derivative_shape_params__") == {
        "__tc_shape_M": "M",
        "__tc_shape_N": "N",
    }

    route_calls: list[dict[str, object]] = []
    value = np.arange(6, dtype=np.float32).reshape(2, 3)

    def execute_route(call_values: dict[str, object]) -> list[object]:
        route_calls.append(call_values)
        assert call_values["M"] is value
        assert call_values["__tc_shape_M"] == 2
        assert call_values["__tc_shape_N"] == 3
        return [np.reshape(call_values["M"], (call_values["__tc_shape_M"], call_values["__tc_shape_N"]))]

    result = DerivativeExecutionDispatcher(
        route_executor=execute_route,
        params=tuple(getattr(library_cls, "__tc_derivative_params__")),
        shape_params=getattr(library_cls, "__tc_derivative_shape_params__"),
    ).execute(program, values={"M": value})

    assert len(route_calls) == 1
    np.testing.assert_allclose(result.gradients[0], value)


def test_execution_library_reuses_duplicate_symbol_shape_params_and_rejects_conflicts():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=ReshapeOperator(),
                op_params={"shape": ["N", "N"]},
                input_value_ids=["v0"],
                output_typespec=_typespec(("D", "M")),
            )
        ],
        inputs=[("v0", _typespec(("N", "N")))],
        outputs=["v1"],
    )
    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec((3, 2)))

    library_cls = build_derivative_execution_library(
        publisher="autodiff-devco",
        class_name="DuplicateSymbolShapeExecution",
        version="0.1.0",
        program=program,
    )

    assert getattr(library_cls, "__tc_derivative_params__") == ("seed", "__tc_shape_N")
    assert getattr(library_cls, "__tc_derivative_shape_params__") == {"__tc_shape_N": "N"}
    assert library_definition(library_cls)[
        "/lib/autodiff-devco/duplicate_symbol_shape_execution/0.1.0"
    ]["execute"]["/state/scalar/op/post"][0] == [
        "d0",
        {"$seed/reshape": {"shape": [{"$__tc_shape_N": []}, {"$__tc_shape_N": []}]}},
    ]

    route_calls: list[dict[str, object]] = []
    dispatcher = DerivativeExecutionDispatcher(
        route_executor=lambda call_values: route_calls.append(call_values) or [],
        params=tuple(getattr(library_cls, "__tc_derivative_params__")),
        shape_params=getattr(library_cls, "__tc_derivative_shape_params__"),
    )

    with pytest.raises(AutodiffError) as exc:
        dispatcher.execute(
            program,
            values={
                "seed": np.arange(6, dtype=np.float32).reshape(3, 2),
                "v0": np.arange(6, dtype=np.float32).reshape(2, 3),
            },
        )

    assert exc.value.category == "symbolic_shape_mismatch"
    assert route_calls == []


def test_scheduler_resolves_symbolic_shape_params_from_explicit_bindings():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=ReshapeOperator(),
                op_params={"shape": ["D", "N"]},
                input_value_ids=["v0"],
                output_typespec=_typespec(("D", "N")),
            )
        ],
        inputs=[("v0", _typespec(("N", "D")))],
        outputs=["v1"],
    )
    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec((3, 2)))

    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(
        program,
        values={"seed": np.arange(6, dtype=np.float32).reshape(3, 2)},
        shape_bindings={"N": 2, "D": 3},
    )

    (gradient,) = result.gradients
    np.testing.assert_allclose(gradient, np.arange(6, dtype=np.float32).reshape(2, 3))


def test_scheduler_resolves_symbolic_shape_params_from_runtime_values():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=ReshapeOperator(),
                op_params={"shape": ["M", "N"]},
                input_value_ids=["v0"],
                output_typespec=_typespec(("D", "M")),
            )
        ],
        inputs=[("v0", _typespec(("M", "N")))],
        outputs=["v1"],
    )
    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec((3, 2)))

    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(
        program,
        values={
            "seed": np.arange(6, dtype=np.float32).reshape(3, 2),
            "v0": np.arange(6, dtype=np.float32).reshape(2, 3),
        },
    )

    (gradient,) = result.gradients
    np.testing.assert_allclose(gradient, np.arange(6, dtype=np.float32).reshape(2, 3))


def test_scheduler_rejects_runtime_shape_symbol_conflict_before_dispatch():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=ReshapeOperator(),
                op_params={"shape": ["N", "N"]},
                input_value_ids=["v0"],
                output_typespec=_typespec(("D", "M")),
            )
        ],
        inputs=[("v0", _typespec(("N", "N")))],
        outputs=["v1"],
    )
    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec((3, 2)))

    with pytest.raises(AutodiffError) as exc:
        ExecutionScheduler(NumpyAutodiffDispatcher()).execute(
            program,
            values={
                "seed": np.arange(6, dtype=np.float32).reshape(3, 2),
                "v0": np.arange(6, dtype=np.float32).reshape(2, 3),
            },
        )

    assert exc.value.category == "symbolic_shape_mismatch"


def test_scheduler_rejects_unresolved_symbolic_shape_params_before_dispatch():
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="v1",
                operator=ReshapeOperator(),
                op_params={"shape": ["D", "N"]},
                input_value_ids=["v0"],
                output_typespec=_typespec(("D", "N")),
            )
        ],
        inputs=[("v0", _typespec(("N", "D")))],
        outputs=["v1"],
    )
    program = generate(graph, "v1", ["v0"], "seed", seed_typespec=_typespec((3, 2)))

    with pytest.raises(AutodiffError) as exc:
        ExecutionScheduler(NumpyAutodiffDispatcher()).execute(
            program,
            values={"seed": np.arange(6, dtype=np.float32).reshape(3, 2)},
        )

    assert exc.value.category == "unresolved_symbolic_shape"
