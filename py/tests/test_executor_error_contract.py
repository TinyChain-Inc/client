from __future__ import annotations

import pytest

from tinychain.autodiff import AutodiffError
from tinychain.autodiff.executor import ExecutionScheduler
from tinychain.autodiff.graph import AddOperator, TensorNodeRecord
from tinychain.autodiff.protocol import DerivativeMetadata
from tinychain.autodiff.reverse import DerivativeProgram


def _metadata() -> DerivativeMetadata:
    return DerivativeMetadata(
        source_graph_id="g0",
        transform_version="0.1.0",
        tensor_op_contract_version="0.1.0",
        wrt_signature=("x",),
        seed_contract="scalar",
    )


def _node(node_id: str, input_value_ids: list[str], output_value_id: str) -> TensorNodeRecord:
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=AddOperator(),
        op_params={},
        input_value_ids=input_value_ids,
    )


def _dummy_dispatch(node: TensorNodeRecord, args: list[object]) -> object:
    return args[0]


def test_missing_input_value_raises_autodiff_error() -> None:
    node = _node("n0", ["missing_id"], "out0")
    program = DerivativeProgram(
        nodes=[node],
        gradients={},
        output_gradients=["out0"],
        metadata=_metadata(),
    )
    scheduler = ExecutionScheduler(dispatch=_dummy_dispatch)

    with pytest.raises(AutodiffError) as exc_info:
        scheduler.execute(program, values={})

    err = exc_info.value
    assert err.category == "missing_derivative_ir"
    assert "missing_id" in err.message


def test_missing_seed_value_raises_autodiff_error() -> None:
    node = _node("n0", ["seed_v0"], "out0")
    program = DerivativeProgram(
        nodes=[node],
        gradients={},
        output_gradients=["out0"],
        metadata=_metadata(),
    )
    scheduler = ExecutionScheduler(dispatch=_dummy_dispatch)

    with pytest.raises(AutodiffError) as exc_info:
        scheduler.execute(program, values={})

    err = exc_info.value
    assert err.category == "missing_derivative_ir"
    assert "seed_v0" in err.message


def test_missing_output_gradient_raises_autodiff_error() -> None:
    node = _node("n0", ["v0"], "out0")
    program = DerivativeProgram(
        nodes=[node],
        gradients={},
        output_gradients=["nonexistent_grad"],
        metadata=_metadata(),
    )
    scheduler = ExecutionScheduler(dispatch=_dummy_dispatch)

    with pytest.raises(AutodiffError) as exc_info:
        scheduler.execute(program, values={"v0": 1.0})

    err = exc_info.value
    assert err.category == "missing_derivative_ir"
    assert "nonexistent_grad" in err.message
