"""Smoke tests for the shared generic reference consumer.

Each registry defined in ``tests.autodiff_reference_consumer`` is proven here
by lowering a hand-built graph end to end through ``lower_graph`` and checking
the executed value against ``NumpyAutodiffDispatcher`` -- the single
dense-array execution semantics every registry and the node-level executor
agree on.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinychain.autodiff import (
    AutodiffError,
    BroadcastOperator,
    DivOperator,
    FillOperator,
    MatmulOperator,
    MeanOperator,
    MulOperator,
    ReshapeOperator,
    TensorGraph,
    TensorNodeRecord,
    lower_graph,
)
from tests.autodiff_execution import NumpyAutodiffDispatcher, numpy_dtype_for_fill
from tests.autodiff_reference_consumer import (
    limited_operation_registry,
    reduction_capable_registry,
    recording_registry,
)


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _fill_node(node_id, output_value_id, *, fill, dtype, shape):
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=FillOperator(),
        op_params={"fill": fill, "dtype": dtype, "shape": list(shape)},
        input_value_ids=[],
        output_typespec=_typespec(shape, dtype),
    )


def _matmul_node(node_id, output_value_id, input_value_ids, out_shape, dtype="f32"):
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=MatmulOperator(),
        op_params={},
        input_value_ids=list(input_value_ids),
        output_typespec=_typespec(out_shape, dtype),
    )


def _mul_node(node_id, output_value_id, input_value_ids, out_shape, dtype="f32", right_literal=None):
    op_params = {} if right_literal is None else {"right_literal": right_literal}
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=MulOperator(),
        op_params=op_params,
        input_value_ids=list(input_value_ids),
        output_typespec=_typespec(out_shape, dtype),
    )


def _reshape_node(node_id, output_value_id, input_value_ids, target_shape, dtype="f32"):
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=ReshapeOperator(),
        op_params={"shape": list(target_shape)},
        input_value_ids=list(input_value_ids),
        output_typespec=_typespec(target_shape, dtype),
    )


def _mean_node(node_id, output_value_id, input_value_ids, axes, keepdims, out_shape, dtype="f32"):
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=MeanOperator(),
        op_params={"axes": list(axes), "keepdims": keepdims},
        input_value_ids=list(input_value_ids),
        output_typespec=_typespec(out_shape, dtype),
    )


def _broadcast_node(node_id, output_value_id, input_value_ids, shape, dtype="f32"):
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=BroadcastOperator(),
        op_params={"shape": list(shape)},
        input_value_ids=list(input_value_ids),
        output_typespec=_typespec(shape, dtype),
    )


def _div_node(node_id, output_value_id, input_value_ids, out_shape, dtype="f32", right_literal=None):
    op_params = {} if right_literal is None else {"right_literal": right_literal}
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=DivOperator(),
        op_params=op_params,
        input_value_ids=list(input_value_ids),
        output_typespec=_typespec(out_shape, dtype),
    )


def _fill_matmul_mul_graph():
    fill_node = _fill_node("n0", "vfill", fill=2.0, dtype="f32", shape=(2, 2))
    matmul_node = _matmul_node("n1", "vmatmul", ["v0", "vfill"], (2, 2))
    mul_node = _mul_node("n2", "vmul", ["vmatmul"], (2, 2), right_literal=3.0)
    return TensorGraph(
        nodes=[fill_node, matmul_node, mul_node],
        inputs=[("v0", _typespec((2, 2)))],
        outputs=["vmul"],
    )


# -- AC: NumpyAutodiffDispatcher executes FillOperator ------------------------


def test_numpy_dispatcher_executes_fill_node_to_constant_array():
    node = _fill_node("n0", "v0", fill=2.5, dtype="f32", shape=(2, 3))

    result = NumpyAutodiffDispatcher()(node, [])

    expected = np.full((2, 3), 2.5, dtype=numpy_dtype_for_fill("f32"))
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype


def test_numpy_dispatcher_executes_fill_node_with_a_one_by_one_shape():
    node = _fill_node("n0", "v0", fill=-1.0, dtype="f64", shape=(1, 1))

    result = NumpyAutodiffDispatcher()(node, [])

    expected = np.full((1, 1), -1.0, dtype=numpy_dtype_for_fill("f64"))
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype


def test_numpy_dispatcher_executes_fill_node_with_a_zero_dimensional_shape():
    node = _fill_node("n0", "v0", fill=7.0, dtype="f32", shape=())

    result = NumpyAutodiffDispatcher()(node, [])

    expected = np.full((), 7.0, dtype=numpy_dtype_for_fill("f32"))
    np.testing.assert_array_equal(result, expected)
    assert result.shape == ()


# -- AC: limited-operation registry lowers fill/matmul/mul --------------------


def test_limited_operation_registry_lowers_fill_matmul_mul_graph():
    graph = _fill_matmul_mul_graph()
    operand = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    lowered = lower_graph(
        graph,
        handlers=limited_operation_registry(),
        outputs=["vmul"],
        bind_input=lambda dependency: operand,
    )

    (output_value,) = lowered.output_values
    fill_value = np.full((2, 2), 2.0, dtype=np.float32)
    expected = np.matmul(operand, fill_value) * 3.0
    np.testing.assert_allclose(output_value, expected, rtol=1e-6)


def test_limited_operation_registry_lowers_a_non_square_matmul():
    node = _matmul_node("n0", "vout", ["v0", "v1"], (2, 4))
    graph = TensorGraph(
        nodes=[node],
        inputs=[("v0", _typespec((2, 3))), ("v1", _typespec((3, 4)))],
        outputs=["vout"],
    )
    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = np.arange(12, dtype=np.float32).reshape(3, 4)
    values = {"v0": left, "v1": right}

    lowered = lower_graph(
        graph,
        handlers=limited_operation_registry(),
        outputs=["vout"],
        bind_input=lambda dependency: values[dependency.value_id],
    )

    (output_value,) = lowered.output_values
    np.testing.assert_allclose(output_value, np.matmul(left, right), rtol=1e-6)


# -- AC: limited-operation registry raises unsupported_operator for the rest --


@pytest.mark.parametrize(
    "operator_type",
    [ReshapeOperator, MeanOperator, BroadcastOperator, DivOperator],
)
def test_limited_operation_registry_rejects_unhandled_operator_types(operator_type):
    registry = limited_operation_registry()

    with pytest.raises(AutodiffError) as excinfo:
        registry.lookup(operator_type())

    assert excinfo.value.category == "unsupported_operator"


# -- AC: trivial-reshape opt-in variant ---------------------------------------


def test_trivial_reshape_variant_lowers_an_element_count_preserving_reshape():
    node = _reshape_node("n0", "vout", ["v0"], (3, 2))
    graph = TensorGraph(nodes=[node], inputs=[("v0", _typespec((2, 3)))], outputs=["vout"])
    operand = np.arange(6, dtype=np.float32).reshape(2, 3)

    lowered = lower_graph(
        graph,
        handlers=limited_operation_registry(include_trivial_reshape=True),
        outputs=["vout"],
        bind_input=lambda dependency: operand,
    )

    (output_value,) = lowered.output_values
    np.testing.assert_array_equal(output_value, operand.reshape(3, 2))


def test_trivial_reshape_variant_rejects_a_reshape_that_changes_element_count():
    node = _reshape_node("n0", "vout", ["v0"], (4,))
    graph = TensorGraph(nodes=[node], inputs=[("v0", _typespec((2, 3)))], outputs=["vout"])
    operand = np.arange(6, dtype=np.float32).reshape(2, 3)

    with pytest.raises(AutodiffError) as excinfo:
        lower_graph(
            graph,
            handlers=limited_operation_registry(include_trivial_reshape=True),
            outputs=["vout"],
            bind_input=lambda dependency: operand,
        )

    assert excinfo.value.category == "shape_mismatch"


def test_default_limited_operation_registry_still_rejects_reshape():
    registry = limited_operation_registry()

    with pytest.raises(AutodiffError) as excinfo:
        registry.lookup(ReshapeOperator())

    assert excinfo.value.category == "unsupported_operator"


# -- AC: reduction-capable control registry matches the dispatcher -----------


def test_reduction_capable_registry_matches_dispatcher_for_mean():
    node = _mean_node("n0", "vout", ["v0"], axes=[1], keepdims=False, out_shape=(2,))
    graph = TensorGraph(nodes=[node], inputs=[("v0", _typespec((2, 3)))], outputs=["vout"])
    operand = np.arange(6, dtype=np.float32).reshape(2, 3)

    lowered = lower_graph(
        graph,
        handlers=reduction_capable_registry(),
        outputs=["vout"],
        bind_input=lambda dependency: operand,
    )

    (output_value,) = lowered.output_values
    expected = NumpyAutodiffDispatcher()(node, [operand])
    np.testing.assert_allclose(output_value, expected, rtol=1e-6)


def test_reduction_capable_registry_matches_dispatcher_for_broadcast():
    node = _broadcast_node("n0", "vout", ["v0"], shape=(2, 3))
    graph = TensorGraph(nodes=[node], inputs=[("v0", _typespec((1, 3)))], outputs=["vout"])
    operand = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    lowered = lower_graph(
        graph,
        handlers=reduction_capable_registry(),
        outputs=["vout"],
        bind_input=lambda dependency: operand,
    )

    (output_value,) = lowered.output_values
    expected = NumpyAutodiffDispatcher()(node, [operand])
    np.testing.assert_allclose(output_value, expected, rtol=1e-6)


def test_reduction_capable_registry_matches_dispatcher_for_div():
    node = _div_node("n0", "vout", ["v0"], out_shape=(2, 2), right_literal=4.0)
    graph = TensorGraph(nodes=[node], inputs=[("v0", _typespec((2, 2)))], outputs=["vout"])
    operand = np.array([[8.0, 4.0], [2.0, 1.0]], dtype=np.float32)

    lowered = lower_graph(
        graph,
        handlers=reduction_capable_registry(),
        outputs=["vout"],
        bind_input=lambda dependency: operand,
    )

    (output_value,) = lowered.output_values
    expected = NumpyAutodiffDispatcher()(node, [operand])
    np.testing.assert_allclose(output_value, expected, rtol=1e-6)


def test_reduction_capable_registry_also_lowers_fill_matmul_mul():
    graph = _fill_matmul_mul_graph()
    operand = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    lowered = lower_graph(
        graph,
        handlers=reduction_capable_registry(),
        outputs=["vmul"],
        bind_input=lambda dependency: operand,
    )

    (output_value,) = lowered.output_values
    fill_value = np.full((2, 2), 2.0, dtype=np.float32)
    expected = np.matmul(operand, fill_value) * 3.0
    np.testing.assert_allclose(output_value, expected, rtol=1e-6)


# -- AC: the recording registry records invocations in call order ------------


def test_recording_registry_records_every_invocation_in_call_order():
    graph = _fill_matmul_mul_graph()
    operand = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    recording = recording_registry(limited_operation_registry())

    lower_graph(
        graph,
        handlers=recording.registry,
        outputs=["vmul"],
        bind_input=lambda dependency: operand,
    )

    recorded_node_ids = [invocation.node_id for invocation in recording.invocations]
    assert recorded_node_ids == ["n0", "n1", "n2"]
    recorded_operator_types = [invocation.operator_type for invocation in recording.invocations]
    assert recorded_operator_types == [FillOperator, MatmulOperator, MulOperator]


def test_recording_registry_stays_empty_when_lowering_fails_before_any_handler_runs():
    node = _reshape_node("n0", "vout", ["v0"], (4,))
    graph = TensorGraph(nodes=[node], inputs=[("v0", _typespec((2, 3)))], outputs=["vout"])
    recording = recording_registry(limited_operation_registry())

    with pytest.raises(AutodiffError) as excinfo:
        lower_graph(
            graph,
            handlers=recording.registry,
            outputs=["vout"],
            bind_input=lambda dependency: np.zeros((2, 3), dtype=np.float32),
        )

    assert excinfo.value.category == "unsupported_operator"
    assert recording.invocations == []


# -- Edge case: wrong type reaching a handler ---------------------------------


def test_fill_handler_rejects_a_fill_node_with_a_non_numeric_fill_value():
    node = _fill_node("n0", "v0", fill="not-a-number", dtype="f32", shape=(2, 2))
    graph = TensorGraph(nodes=[node], inputs=[], outputs=["v0"])

    with pytest.raises(AutodiffError) as excinfo:
        lower_graph(graph, handlers=limited_operation_registry(), outputs=["v0"])

    assert excinfo.value.category == "malformed_derivative_ir"
