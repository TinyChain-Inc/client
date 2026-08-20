"""Unit tests for tracing a parameter update as an ordinary Tensor callable.

These tests pin the contract that an optimizer update -- an SGD step in
particular -- is authored as plain Tensor code, traced through the same
public builder path as an application loss, and validated/compiled through
the existing structured dependency analysis (T-01) and extensible lowering
seam (T-02). No manual `TensorNodeRecord`/`TensorOperator` construction is
allowed in production or example code (spec invariant 6).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from tinychain.autodiff import (
    AutodiffError,
    MulOperator,
    SubOperator,
    TensorGraphBuilder,
    TensorNodeRecord,
    TensorOperator,
    get_active_builder,
)
from tinychain.autodiff.dependencies import analyze_graph_dependencies
from tinychain.autodiff.lowering import (
    OperationHandlerRegistry,
    lower_graph,
)
from tinychain.autodiff import training
from tinychain.autodiff.training import TracedUpdate, sgd_update, trace_parameter_update

from tests.autodiff_execution import NumpyAutodiffDispatcher


PARAMETER_SPEC = {"dtype": "f32", "shape": (2, 3)}
GRADIENT_SPEC = {"dtype": "f32", "shape": (2, 3)}
LEARNING_RATE_SPEC = {"dtype": "f32", "shape": ()}


def _trace_sgd() -> TracedUpdate:
    return trace_parameter_update(
        sgd_update,
        parameter=PARAMETER_SPEC,
        gradient=GRADIENT_SPEC,
        optimizer_inputs={"learning_rate": LEARNING_RATE_SPEC},
    )


class _NumpyOperationHandler:
    """Adapts the shared NumPy dispatcher fixture to the lowering handler seam."""

    def __init__(self, operator_type: type) -> None:
        self.operator_type = operator_type
        self._dispatcher = NumpyAutodiffDispatcher()

    def lower(self, context) -> object:
        node_like = SimpleNamespace(operator=context.operator, op_params=context.op_params)
        return self._dispatcher(node_like, list(context.inputs))


def _numeric_registry() -> OperationHandlerRegistry:
    registry = OperationHandlerRegistry()
    registry.register(_NumpyOperationHandler(MulOperator))
    registry.register(_NumpyOperationHandler(SubOperator))
    return registry


# --------------------------------------------------------------------------
# structure and numerical execution
# --------------------------------------------------------------------------


def test_sgd_update_traces_rank2_floating_structure() -> None:
    traced = _trace_sgd()

    assert isinstance(traced, TracedUpdate)
    assert traced.graph.outputs == [traced.updated_parameter_id]
    assert [type(node.operator) for node in traced.graph.nodes] == [MulOperator, SubOperator]

    output_node = traced.graph.nodes[-1]
    assert output_node.output_value_id == traced.updated_parameter_id
    assert output_node.output_typespec == {"dtype": "f32", "shape": [2, 3]}

    assert set(traced.input_value_ids) == {"parameter", "gradient", "learning_rate"}
    declared_ids = {value_id for value_id, _typespec in traced.graph.inputs}
    assert set(traced.input_value_ids.values()) == declared_ids


def test_sgd_update_dependency_analysis_has_no_forward_captures_or_seeds() -> None:
    traced = _trace_sgd()
    analysis = analyze_graph_dependencies(traced.graph, outputs=[traced.updated_parameter_id])

    assert analysis.forward_captures == ()
    assert analysis.seed_inputs == ()
    assert {dependency.value_id for dependency in analysis.declared_inputs} == set(
        traced.input_value_ids.values()
    )


def test_sgd_update_numerical_execution_matches_analytical_update() -> None:
    traced = _trace_sgd()

    parameter_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    gradient_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    learning_rate_value = np.float32(0.5)

    bound_values = {
        traced.input_value_ids["parameter"]: parameter_value,
        traced.input_value_ids["gradient"]: gradient_value,
        traced.input_value_ids["learning_rate"]: learning_rate_value,
    }

    lowered = lower_graph(
        traced.graph,
        handlers=_numeric_registry(),
        outputs=[traced.updated_parameter_id],
        bind_input=lambda dependency: bound_values[dependency.value_id],
    )

    (updated_parameter,) = lowered.output_values
    expected = parameter_value - learning_rate_value * gradient_value
    np.testing.assert_allclose(np.asarray(updated_parameter), expected)


def test_sgd_update_lowering_fails_closed_for_unsupported_operator() -> None:
    traced = _trace_sgd()
    partial_registry = OperationHandlerRegistry()
    partial_registry.register(_NumpyOperationHandler(SubOperator))

    with pytest.raises(AutodiffError) as error:
        lower_graph(
            traced.graph,
            handlers=partial_registry,
            outputs=[traced.updated_parameter_id],
            bind_input=lambda dependency: 0.0,
        )
    assert error.value.category == "unsupported_operator"


# --------------------------------------------------------------------------
# metadata mismatch
# --------------------------------------------------------------------------


def test_trace_parameter_update_rejects_incompatible_gradient_shape() -> None:
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            sgd_update,
            parameter={"dtype": "f32", "shape": (2, 3)},
            gradient={"dtype": "f32", "shape": (4, 3)},
            optimizer_inputs={"learning_rate": LEARNING_RATE_SPEC},
        )
    assert error.value.category == "broadcast_shape_mismatch"
    assert get_active_builder() is None


# --------------------------------------------------------------------------
# non-Tensor output
# --------------------------------------------------------------------------


def test_trace_parameter_update_rejects_non_tensor_output() -> None:
    def bad_update(*, parameter: object, gradient: object, learning_rate: object) -> object:
        return 5.0

    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            bad_update,
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"learning_rate": LEARNING_RATE_SPEC},
        )
    assert error.value.category == "invalid_update_output"
    assert get_active_builder() is None


# --------------------------------------------------------------------------
# invalid signatures
# --------------------------------------------------------------------------


def test_trace_parameter_update_rejects_callable_missing_gradient_parameter() -> None:
    def bad_update(*, parameter: object, learning_rate: object) -> object:
        return parameter

    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            bad_update,
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"learning_rate": LEARNING_RATE_SPEC},
        )
    assert error.value.category == "invalid_update_signature"
    assert get_active_builder() is None


def test_trace_parameter_update_rejects_callable_with_unexpected_required_parameter() -> None:
    def bad_update(*, parameter: object, gradient: object, learning_rate: object, momentum: object) -> object:
        return parameter

    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            bad_update,
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"learning_rate": LEARNING_RATE_SPEC},
        )
    assert error.value.category == "invalid_update_signature"


def test_trace_parameter_update_rejects_non_callable_update() -> None:
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            object(),
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"learning_rate": LEARNING_RATE_SPEC},
        )
    assert error.value.category == "invalid_update_signature"


def test_trace_parameter_update_signature_failure_precedes_tracing() -> None:
    """Invalid update callables fail before any node is recorded (AC4)."""
    calls: list[object] = []

    def bad_update(*, parameter: object, gradient: object) -> object:
        calls.append(True)
        return parameter

    with pytest.raises(AutodiffError):
        trace_parameter_update(
            bad_update,
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"learning_rate": LEARNING_RATE_SPEC},
        )
    assert calls == []


# --------------------------------------------------------------------------
# inactive tracing precondition
# --------------------------------------------------------------------------


def test_trace_parameter_update_rejects_nested_active_trace() -> None:
    with TensorGraphBuilder():
        with pytest.raises(RuntimeError, match="Nested"):
            trace_parameter_update(
                sgd_update,
                parameter=PARAMETER_SPEC,
                gradient=GRADIENT_SPEC,
                optimizer_inputs={"learning_rate": LEARNING_RATE_SPEC},
            )
    assert get_active_builder() is None


# --------------------------------------------------------------------------
# no manual graph-record construction (invariant 6)
# --------------------------------------------------------------------------


def test_training_module_contains_no_manual_graph_record_construction() -> None:
    source = inspect.getsource(training)
    forbidden_constructors = (
        "TensorNodeRecord(",
        "AddOperator(",
        "SubOperator(",
        "MulOperator(",
        "DivOperator(",
        "MatmulOperator(",
        "MeanOperator(",
        "SumOperator(",
        "ReshapeOperator(",
        "TransposeOperator(",
        "MaxOperator(",
        "MinOperator(",
        "ProductOperator(",
        "BroadcastOperator(",
        "BroadcastReduceOperator(",
    )
    for token in forbidden_constructors:
        assert token not in source, f"found manual graph-record construction: {token!r}"

    # Token matching over source text is evadable: an import alias
    # (`from .graph import MulOperator as Scale`) or construction moved into a
    # module-level helper reached through attribute access neither contains a
    # forbidden literal call token in this module's own source. Neither can
    # avoid binding a name in this module's namespace, though -- a
    # `TensorOperator` subclass or `TensorNodeRecord` itself would still show
    # up there under whatever name it was imported or assigned as. Checking
    # the namespace directly catches both evasions the token scan above
    # cannot.
    for name, value in vars(training).items():
        if name.startswith("__"):
            continue
        assert value is not TensorNodeRecord, (
            f"training module namespace binds TensorNodeRecord as {name!r}"
        )
        if isinstance(value, type) and issubclass(value, TensorOperator):
            raise AssertionError(
                "training module namespace binds TensorOperator subclass "
                f"{value!r} as {name!r}"
            )


def test_sgd_update_example_contains_no_manual_graph_record_construction() -> None:
    source = inspect.getsource(sgd_update)
    forbidden_constructors = ("TensorNodeRecord(", "SubOperator(", "MulOperator(")
    for token in forbidden_constructors:
        assert token not in source, f"found manual graph-record construction: {token!r}"
