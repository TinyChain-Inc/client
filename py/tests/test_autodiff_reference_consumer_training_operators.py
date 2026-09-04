"""Unit tests for the shared reference consumer's training-step registry.

Issue #128 built the shared generic reference consumer around a rank-preserving
tier (`limited_operation_registry`) and a reduction-capable tier
(`reduction_capable_registry`). This module's own subtask extends that same
consumer -- per §17.1's ownership rule -- with the handlers a full
training-step compilation needs: `AddOperator`, `SubOperator`,
`TransposeOperator`, and a general `ReshapeOperator`, plus a factory
(`training_step_registry`) returning a registry that resolves every concrete
operator §17.5's loss produces across its forward, derivative, and update
programs.

The operator set asserted here was measured directly, not assumed: compiling
`d = x @ w - y`, `d * d`, `mean(...)` with `SGD` and a permissive recording
registry (see the module docstring investigation this test file's author
performed) shows the three lowered programs together produce exactly
`AddOperator`, `BroadcastOperator`, `DivOperator`, `MatmulOperator`,
`MeanOperator`, `MulOperator`, `ReshapeOperator`, `SubOperator`, and
`TransposeOperator` -- nine concrete operator types, matching §17.1's prose
account operator-for-operator (the "literal-scaled multiply" §17.1 describes
materializes as `DivOperator(right_literal=...)` in this trace, not a second
`MulOperator` shape; `MulOperator` itself is already needed for the forward
trace and reappears, unchanged in kind, in the derivative and the update).

Every assertion here compares a handler's own output against
`NumpyAutodiffDispatcher`'s output for the *same node* -- the only way "no
handler computes a result itself" is a real test rather than a claim about the
shape of the code.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from tinychain.autodiff import (
    AddOperator,
    AutodiffError,
    BroadcastOperator,
    DivOperator,
    FillOperator,
    MatmulOperator,
    MeanOperator,
    MulOperator,
    OperationContext,
    ReshapeOperator,
    SubOperator,
    SumOperator,
    TensorNodeRecord,
    TransposeOperator,
)

from tests.autodiff_execution import NumpyAutodiffDispatcher
from tests.autodiff_reference_consumer import (
    limited_operation_registry,
    reduction_capable_registry,
    recording_registry,
    training_step_registry,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _node(node_id, output_value_id, operator, op_params, input_value_ids, out_shape, dtype="f32"):
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=operator,
        op_params=op_params,
        input_value_ids=list(input_value_ids),
        output_typespec=_typespec(out_shape, dtype),
    )


def _context(node: TensorNodeRecord, inputs: list[object]) -> OperationContext:
    """Build the `OperationContext` `lower_graph` would hand a handler for *node*."""
    return OperationContext(
        node_id=node.node_id,
        operator=node.operator,
        op_params=MappingProxyType(dict(node.op_params)),
        input_value_ids=tuple(node.input_value_ids),
        inputs=tuple(inputs),
        input_provenance=tuple("local_value" for _ in node.input_value_ids),
        output_value_id=node.output_value_id,
        output_typespec=MappingProxyType(dict(node.output_typespec)),
    )


def _assert_handler_matches_dispatcher(
    node: TensorNodeRecord, inputs: list[object], *, registry=None
) -> None:
    """The AC-56 delegation proof: a handler's own output vs. the dispatcher's, same node.

    Defaults to `training_step_registry()`; `FillOperator` is outside that
    registry's measured set (the §17.5 loss never generates a fill node), so
    its delegation is instead proven against `reduction_capable_registry()`,
    which already resolves `FillOperator` unchanged from #128.
    """
    if registry is None:
        registry = training_step_registry()
    handler = registry.lookup(node.operator)
    context = _context(node, inputs)

    handler_result = handler.lower(context)
    dispatcher_result = NumpyAutodiffDispatcher()(node, inputs)

    np.testing.assert_array_equal(np.asarray(handler_result), np.asarray(dispatcher_result))


# The complete operator set §17.5's loss produces across all three programs,
# measured by compiling it with a permissive recording registry (see the
# module docstring). Nine concrete operator types.
MEASURED_OPERATOR_SET = frozenset(
    {
        AddOperator,
        BroadcastOperator,
        DivOperator,
        MatmulOperator,
        MeanOperator,
        MulOperator,
        ReshapeOperator,
        SubOperator,
        TransposeOperator,
    }
)


# --------------------------------------------------------------------------
# AC: every operator the loss produces resolves in the new registry, as a
# concrete operator instance
# --------------------------------------------------------------------------


def test_training_step_registry_resolves_every_measured_operator_as_a_concrete_instance():
    registry = training_step_registry()

    for operator_type in MEASURED_OPERATOR_SET:
        handler = registry.lookup(operator_type())
        assert handler.operator_type is operator_type
        assert isinstance(operator_type(), operator_type)


def test_training_step_registry_supported_types_equals_the_measured_set():
    registry = training_step_registry()

    assert set(registry.supported_types()) == MEASURED_OPERATOR_SET


# --------------------------------------------------------------------------
# AC: no handler computes a result itself -- delegation proof per operator
# --------------------------------------------------------------------------


def test_add_handler_matches_dispatcher_for_same_shape_operands():
    node = _node("n0", "v2", AddOperator(), {}, ["v0", "v1"], out_shape=(2, 3))
    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = left + 1.0

    _assert_handler_matches_dispatcher(node, [left, right])


def test_add_handler_matches_dispatcher_for_operands_with_different_shapes():
    # Edge case: an add whose operands broadcast rather than matching exactly --
    # the accumulation add the derivative produces for a repeated operand can
    # combine a captured value with a reduced gradient of a different rank.
    node = _node("n0", "v2", AddOperator(), {}, ["v0", "v1"], out_shape=(2, 3))
    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    _assert_handler_matches_dispatcher(node, [left, right])


def test_sub_handler_matches_dispatcher():
    node = _node("n0", "v2", SubOperator(), {}, ["v0", "v1"], out_shape=(2, 4))
    left = np.arange(8, dtype=np.float32).reshape(2, 4)
    right = np.ones((2, 4), dtype=np.float32) * 3.0

    _assert_handler_matches_dispatcher(node, [left, right])


def test_transpose_handler_matches_dispatcher_for_a_non_trivial_permutation():
    # Edge case: a non-trivial permutation, not just a plain 2-D swap.
    node = _node(
        "n0",
        "v1",
        TransposeOperator(),
        {"perm": [2, 0, 1]},
        ["v0"],
        out_shape=(4, 2, 3),
    )
    operand = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    _assert_handler_matches_dispatcher(node, [operand])


def test_reshape_handler_matches_dispatcher_for_a_reshape_that_changes_rank():
    # Edge case: a reshape that changes rank, e.g. the derivative's seed
    # reshape from a scalar-shaped value to a 2-D one.
    node = _node("n0", "v1", ReshapeOperator(), {"shape": [1, 1]}, ["v0"], out_shape=(1, 1))
    operand = np.array(7.0, dtype=np.float32)

    _assert_handler_matches_dispatcher(node, [operand])


def test_reshape_handler_matches_dispatcher_for_a_flatten_to_matrix_reshape():
    node = _node("n0", "v1", ReshapeOperator(), {"shape": [2, 3]}, ["v0"], out_shape=(2, 3))
    operand = np.arange(6, dtype=np.float32)

    _assert_handler_matches_dispatcher(node, [operand])


def test_matmul_handler_matches_dispatcher():
    node = _node("n0", "v2", MatmulOperator(), {}, ["v0", "v1"], out_shape=(2, 4))
    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = np.arange(12, dtype=np.float32).reshape(3, 4)

    _assert_handler_matches_dispatcher(node, [left, right])


def test_mul_handler_matches_dispatcher_for_two_operands():
    node = _node("n0", "v2", MulOperator(), {}, ["v0", "v1"], out_shape=(2, 3))
    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = left + 2.0

    _assert_handler_matches_dispatcher(node, [left, right])


def test_mul_handler_matches_dispatcher_for_a_right_literal():
    # Edge case: a multiply carrying `right_literal`.
    node = _node("n0", "v1", MulOperator(), {"right_literal": 2.0}, ["v0"], out_shape=(2, 3))
    operand = np.arange(6, dtype=np.float32).reshape(2, 3)

    _assert_handler_matches_dispatcher(node, [operand])


def test_mean_handler_matches_dispatcher():
    node = _node(
        "n0", "v1", MeanOperator(), {"axes": [0, 1], "keepdims": False}, ["v0"], out_shape=()
    )
    operand = np.arange(8, dtype=np.float32).reshape(2, 4)

    _assert_handler_matches_dispatcher(node, [operand])


def test_broadcast_handler_matches_dispatcher():
    node = _node("n0", "v1", BroadcastOperator(), {"shape": [2, 4]}, ["v0"], out_shape=(2, 4))
    operand = np.array([[1.0]], dtype=np.float32)

    _assert_handler_matches_dispatcher(node, [operand])


def test_div_handler_matches_dispatcher_for_a_right_literal():
    node = _node("n0", "v1", DivOperator(), {"right_literal": 8.0}, ["v0"], out_shape=(2, 4))
    operand = np.arange(8, dtype=np.float32).reshape(2, 4)

    _assert_handler_matches_dispatcher(node, [operand])


def test_fill_handler_matches_dispatcher():
    # FillOperator is outside `training_step_registry`'s measured set (the
    # §17.5 loss never generates a fill node), so this delegation proof runs
    # against `reduction_capable_registry`, which already resolves it.
    node = _node(
        "n0",
        "v0",
        FillOperator(),
        {"fill": 2.5, "dtype": "f32", "shape": [2, 2]},
        [],
        out_shape=(2, 2),
    )

    _assert_handler_matches_dispatcher(node, [], registry=reduction_capable_registry())


# --------------------------------------------------------------------------
# AC: a node whose operator has no handler fails closed before any handler runs
# --------------------------------------------------------------------------


def test_training_step_registry_rejects_an_operator_the_loss_never_produces():
    registry = training_step_registry()

    with pytest.raises(AutodiffError) as excinfo:
        registry.lookup(SumOperator())

    assert excinfo.value.category == "unsupported_operator"


# --------------------------------------------------------------------------
# AC: limited_operation_registry / reduction_capable_registry unchanged
# --------------------------------------------------------------------------


def test_limited_operation_registry_supported_types_unchanged():
    assert set(limited_operation_registry().supported_types()) == {
        FillOperator,
        MatmulOperator,
        MulOperator,
    }


def test_limited_operation_registry_with_trivial_reshape_supported_types_unchanged():
    assert set(limited_operation_registry(include_trivial_reshape=True).supported_types()) == {
        FillOperator,
        MatmulOperator,
        MulOperator,
        ReshapeOperator,
    }


def test_reduction_capable_registry_supported_types_unchanged():
    assert set(reduction_capable_registry().supported_types()) == {
        FillOperator,
        MatmulOperator,
        MulOperator,
        MeanOperator,
        BroadcastOperator,
        DivOperator,
    }


# --------------------------------------------------------------------------
# AC: recording_registry wraps the new registry and records invocation order
# --------------------------------------------------------------------------


def test_recording_registry_wraps_training_step_registry_in_call_order():
    from tinychain.autodiff import TensorGraph, lower_graph

    matmul_node = _node("n0", "vmatmul", MatmulOperator(), {}, ["v0", "v1"], out_shape=(2, 4))
    sub_node = _node("n1", "vsub", SubOperator(), {}, ["vmatmul", "v2"], out_shape=(2, 4))
    graph = TensorGraph(
        nodes=[matmul_node, sub_node],
        inputs=[
            ("v0", _typespec((2, 3))),
            ("v1", _typespec((3, 4))),
            ("v2", _typespec((2, 4))),
        ],
        outputs=["vsub"],
    )
    values = {
        "v0": np.arange(6, dtype=np.float32).reshape(2, 3),
        "v1": np.arange(12, dtype=np.float32).reshape(3, 4),
        "v2": np.ones((2, 4), dtype=np.float32),
    }
    recording = recording_registry(training_step_registry())

    lower_graph(
        graph,
        handlers=recording.registry,
        outputs=["vsub"],
        bind_input=lambda dependency: values[dependency.value_id],
    )

    recorded_node_ids = [invocation.node_id for invocation in recording.invocations]
    assert recorded_node_ids == ["n0", "n1"]
    recorded_operator_types = [invocation.operator_type for invocation in recording.invocations]
    assert recorded_operator_types == [MatmulOperator, SubOperator]
