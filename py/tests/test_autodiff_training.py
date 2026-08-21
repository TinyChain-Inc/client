"""Unit tests for tracing a parameter update as an ordinary Tensor callable.

These tests pin the contract that an optimizer update -- an SGD step in
particular -- is authored as plain Tensor code, traced through the same
public builder path as an application loss, and validated/compiled through
the existing structured dependency analysis and extensible program lowering
seam. No manual `TensorNodeRecord`/`TensorOperator` construction is
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

    # The token scan above only catches a literal, unaliased construction
    # call written directly in this module's source. The namespace scan below
    # additionally catches an import alias bound in this module (for example
    # `from .graph import MulOperator as Scale`, which binds the class object
    # itself under the name `Scale`) and any other name in this module's own
    # namespace that resolves to `TensorNodeRecord` or a `TensorOperator`
    # subclass.
    #
    # Neither scan catches construction that lives in another module and is
    # only reached at call time -- a helper module built and imported for
    # this purpose (`from . import _update_helpers`, or
    # `from ._update_helpers import build_scaled_record`) binds a module
    # object or a function in this namespace, not the class or record type
    # itself, so it is invisible to both checks. Nor does either scan catch a
    # function-local aliased import inside this module (an import statement
    # nested inside a function body binds the alias in that function's local
    # scope, which `vars(training)` never sees). This test is aimed at the
    # realistic accident -- a careless direct construction call, or an
    # import alias created without thinking through what it evades -- not at
    # deliberate circumvention by someone with commit access to this module
    # or a new module beside it.
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


# --------------------------------------------------------------------------
# Every malformed typed-input spec is categorized.
#
# The specs are unpacked as `**dict(spec)` before `TensorGraphBuilder.input`
# is reached, so the unpack -- not the builder's dtype/shape validation --
# decides what a consumer sees. Every malformation therefore has to be
# categorized here, using the same categories the sibling analysis module
# already uses for an incomplete type spec.
# --------------------------------------------------------------------------


_MALFORMED_SPEC_CASES = [
    ("parameter_is_none", {"parameter": None}, "missing_dtype_metadata", "parameter"),
    ("parameter_is_empty", {"parameter": {}}, "missing_dtype_metadata", "parameter"),
    (
        "parameter_dtype_key_typo",
        {"parameter": {"dtpye": "f32", "shape": (2, 3)}},
        "missing_dtype_metadata",
        "parameter",
    ),
    ("parameter_is_an_int", {"parameter": 5}, "missing_dtype_metadata", "parameter"),
    ("parameter_is_a_string", {"parameter": "f32"}, "missing_dtype_metadata", "parameter"),
    # Already categorized today, by the builder's own dtype validation: pinned
    # here so the new spec validation does not re-categorize it.
    (
        "parameter_dtype_value_is_not_a_dtype",
        {"parameter": {"dtype": 32, "shape": (2, 3)}},
        "dtype_not_differentiable",
        "32",
    ),
    (
        "gradient_has_no_shape",
        {"gradient": {"dtype": "f32"}},
        "missing_shape_metadata",
        "gradient",
    ),
    (
        "gradient_shape_is_not_a_sequence",
        {"gradient": {"dtype": "f32", "shape": 3}},
        "missing_shape_metadata",
        "gradient",
    ),
    (
        "gradient_shape_has_a_negative_dimension",
        {"gradient": {"dtype": "f32", "shape": (-1, 3)}},
        "missing_shape_metadata",
        "gradient",
    ),
    (
        "optimizer_input_spec_is_none",
        {"optimizer_inputs": {"learning_rate": None}},
        "missing_dtype_metadata",
        "learning_rate",
    ),
    (
        "optimizer_inputs_is_not_a_mapping",
        {"optimizer_inputs": 5},
        "invalid_update_signature",
        "optimizer_inputs",
    ),
]


def _malformed_call_kwargs(overrides: dict) -> dict:
    kwargs = {
        "parameter": PARAMETER_SPEC,
        "gradient": GRADIENT_SPEC,
        "optimizer_inputs": {"learning_rate": LEARNING_RATE_SPEC},
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("overrides", "category", "named"),
    [case[1:] for case in _MALFORMED_SPEC_CASES],
    ids=[case[0] for case in _MALFORMED_SPEC_CASES],
)
def test_malformed_typed_input_specs_are_categorized(overrides, category, named) -> None:
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(sgd_update, **_malformed_call_kwargs(overrides))

    assert error.value.category == category
    assert named in error.value.message


@pytest.mark.parametrize(
    "overrides",
    [case[1] for case in _MALFORMED_SPEC_CASES],
    ids=[case[0] for case in _MALFORMED_SPEC_CASES],
)
def test_a_malformed_typed_input_spec_fails_before_the_update_runs(overrides) -> None:
    """The same structural property the signature check already guarantees.

    A rejected declaration must never reach the consumer's callable body, so
    the rejection cannot depend on statement order inside the trace.
    """
    invocations: list[dict] = []

    def recording_update(**kwargs):
        invocations.append(kwargs)
        return sgd_update(**kwargs)

    with pytest.raises(AutodiffError):
        trace_parameter_update(recording_update, **_malformed_call_kwargs(overrides))

    assert invocations == []


@pytest.mark.parametrize(
    "overrides",
    [case[1] for case in _MALFORMED_SPEC_CASES],
    ids=[case[0] for case in _MALFORMED_SPEC_CASES],
)
def test_a_malformed_typed_input_spec_leaves_no_builder_active(overrides) -> None:
    with pytest.raises(AutodiffError):
        trace_parameter_update(sgd_update, **_malformed_call_kwargs(overrides))

    assert get_active_builder() is None


def test_a_typed_input_spec_carrying_an_extra_key_is_accepted() -> None:
    """Pins a deliberate widening, not pre-existing behaviour.

    Before this, the specs were unpacked as ``**dict(spec)``, so an unknown
    key raised a raw ``TypeError``. Reading ``dtype`` and ``shape`` by key
    instead -- the only way to categorize a malformed spec without inventing
    a category for "unexpected key" -- makes an extra key ignored, matching
    the structured dependency analysis, which has always read a type spec by
    key and ignored the rest. A consumer must not find one spec accepted by
    the analysis and rejected by the update tracer.

    If that widening is ever silently reversed, this test is what fails.
    """
    traced = trace_parameter_update(
        sgd_update,
        parameter={"dtype": "f32", "shape": (2, 3), "device": "cpu"},
        gradient={"dtype": "f32", "shape": (2, 3), "layout": "dense"},
        optimizer_inputs={"learning_rate": {"dtype": "f32", "shape": (), "device": "cpu"}},
    )

    assert set(traced.input_value_ids) == {"parameter", "gradient", "learning_rate"}
    parameter_id = traced.input_value_ids["parameter"]
    declared_parameter = dict(traced.graph.inputs)[parameter_id]
    assert set(declared_parameter) == {"dtype", "shape"}, (
        "the extra key must be ignored, not recorded on the declared input"
    )
    assert declared_parameter["dtype"] == "f32"
    assert tuple(declared_parameter["shape"]) == (2, 3)


@pytest.mark.parametrize(
    "falsy_container",
    [0, "", [], ()],
    ids=["zero", "empty_string", "empty_list", "empty_tuple"],
)
def test_a_falsy_non_mapping_optimizer_inputs_is_rejected(falsy_container) -> None:
    """Pins a deliberate rejection, and pins *which* argument is blamed.

    ``optimizer_inputs`` used to be resolved as ``optimizer_inputs or {}``, so
    a falsy non-mapping was silently read as "no optimizer inputs". With an
    update that needs one, as here, the caller was then told their *callable*
    was missing a ``learning_rate`` parameter -- blaming the function for a
    fault in the argument beside it.

    The category alone does not discriminate the two: it is
    ``invalid_update_signature`` either way. The message assertion below is
    what does, which is why it is not decoration. Restoring
    ``optimizer_inputs or {}`` for readability leaves the category assertion
    passing and fails on the message.

    ``None`` is excluded deliberately -- it is the documented default for
    "no optimizer inputs" and must keep tracing successfully, which a sibling
    test below pins.
    """
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            sgd_update,
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs=falsy_container,
        )

    assert error.value.category == "invalid_update_signature"
    assert "optimizer_inputs" in error.value.message


@pytest.mark.parametrize(
    "falsy_container",
    [0, "", [], ()],
    ids=["zero", "empty_string", "empty_list", "empty_tuple"],
)
def test_a_falsy_non_mapping_optimizer_inputs_no_longer_traces_silently(
    falsy_container,
) -> None:
    """Pins the rejection of something that previously succeeded outright.

    This is the half where the old ``optimizer_inputs or {}`` produced no
    diagnostic at all: an update needing no optimizer inputs traced happily,
    so a caller who passed a malformed container got a finished graph and no
    hint that what they declared had been discarded.

    A previously succeeding call that now fails is the direction that breaks
    a caller relying on it, even accidentally, so it is pinned as a decision
    rather than left to read as an accident.
    """

    def update_without_optimizer_inputs(*, parameter, gradient):
        return parameter - gradient

    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            update_without_optimizer_inputs,
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs=falsy_container,
        )

    assert error.value.category == "invalid_update_signature"
    assert "optimizer_inputs" in error.value.message


def test_optimizer_inputs_defaulting_to_none_still_declares_no_optimizer_inputs() -> None:
    """The other half of the rule above: `None` remains the way to declare none."""

    def update_without_optimizer_inputs(*, parameter, gradient):
        return parameter - gradient

    traced = trace_parameter_update(
        update_without_optimizer_inputs,
        parameter=PARAMETER_SPEC,
        gradient=GRADIENT_SPEC,
        optimizer_inputs=None,
    )

    assert set(traced.input_value_ids) == {"parameter", "gradient"}
