from __future__ import annotations

import numpy as np
import pytest

from tinychain.autodiff import (
    AutodiffError,
    BroadcastOperator,
    ExecutionScheduler,
    ReshapeOperator,
    SumOperator,
    TensorGraph,
    TensorNodeRecord,
    compile_derivative_program,
    generate,
)
from tinychain.autodiff.seed import SeedValidator
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
