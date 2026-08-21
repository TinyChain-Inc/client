"""Unit tests for tracing a parameter update as an ordinary Tensor callable.

These tests pin the contract that an optimizer update -- an SGD step in
particular -- is authored as plain Tensor code, traced through the same
public builder path as an application loss, and validated/compiled through
the existing structured dependency analysis and extensible program lowering
seam. No manual `TensorNodeRecord`/`TensorOperator` construction is
allowed in production or example code (spec invariant 6).
"""

from __future__ import annotations

import enum
import inspect
from collections.abc import Mapping, Sequence
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
from tinychain.autodiff.dependencies import analyze_graph_dependencies, _complete_typespec
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


# --------------------------------------------------------------------------
# The declared optimizer input NAMES are part of the declared input set.
#
# `TensorGraphBuilder.input` rejects a name that duplicates an already
# declared input, or that is not a valid non-keyword identifier, with a raw
# `ValueError`/`TypeError`. Whether that leak is visible depends entirely on
# the update callable: one declaring exact parameters cannot bind the bad
# name, so the signature check masks it, but a callable taking `**kwargs`
# binds anything and the raw error escapes. A `**kwargs` update is an
# ordinary consumer shape, so the name set must be validated in its own right.
# --------------------------------------------------------------------------


_OPTIMIZER_NAME_CASES = [
    ("collides_with_parameter", "parameter"),
    ("collides_with_gradient", "gradient"),
    ("not_an_identifier", "a b"),
    ("python_keyword", "class"),
    ("empty_name", ""),
]


def _kwargs_update(*, parameter, gradient, **optimizer_inputs):
    """An update binding any optimizer input by name -- masks no bad name."""
    return parameter - gradient


def _exact_params_update(*, parameter, gradient, learning_rate):
    """An update declaring exact parameters -- its signature masks a bad name."""
    return parameter - learning_rate * gradient


@pytest.mark.parametrize(
    "name",
    [case[1] for case in _OPTIMIZER_NAME_CASES],
    ids=[case[0] for case in _OPTIMIZER_NAME_CASES],
)
def test_a_malformed_optimizer_input_name_is_categorized_for_a_kwargs_update(name) -> None:
    """The shape where nothing masks the leak."""
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            _kwargs_update,
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={name: LEARNING_RATE_SPEC},
        )

    assert error.value.category == "invalid_update_signature"


@pytest.mark.parametrize(
    "name",
    [case[1] for case in _OPTIMIZER_NAME_CASES],
    ids=[case[0] for case in _OPTIMIZER_NAME_CASES],
)
def test_a_malformed_optimizer_input_name_is_categorized_for_an_exact_update(name) -> None:
    """The shape where the signature check masks it: it must stay categorized."""
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            _exact_params_update,
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={name: LEARNING_RATE_SPEC},
        )

    assert error.value.category == "invalid_update_signature"


@pytest.mark.parametrize("colliding_name", ["parameter", "gradient"])
@pytest.mark.parametrize(
    "update",
    [_kwargs_update, _exact_params_update],
    ids=["kwargs_update", "exact_params_update"],
)
def test_a_colliding_optimizer_input_name_is_blamed_by_name(update, colliding_name) -> None:
    """The message must name the collision, not blame the callable.

    With an exact-parameter update this case is already categorized, but for
    the wrong reason and with a misleading message: deduplicating the declared
    names leaves the callable short of an argument, so the caller is told
    their *function* is missing `learning_rate` when what is actually wrong is
    that they declared an optimizer input called `parameter`.
    """
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            update,
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={colliding_name: LEARNING_RATE_SPEC},
        )

    assert error.value.category == "invalid_update_signature"
    assert "optimizer_inputs" in error.value.message
    assert colliding_name in error.value.message
    assert "learning_rate" not in error.value.message, (
        "the message must not blame the callable for the collision"
    )


# --------------------------------------------------------------------------
# The spec read must not let a raw container exception escape, exactly as the
# analysis helper it mirrors does not.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raised",
    [TypeError, ValueError],
    ids=["type_error", "value_error"],
)
def test_a_shape_that_raises_while_being_read_is_categorized(raised) -> None:
    """A `Sequence` shape whose iteration raises must not escape raw.

    `dependencies._complete_typespec` wraps its shape read in
    `except (IndexError, TypeError, ValueError)` for exactly this reason, and
    the tracer's helper claims to mirror it.
    """

    class RaisingShape(Sequence):
        def __getitem__(self, index):
            raise raised("boom while reading the shape")

        def __len__(self) -> int:
            return 2

    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            sgd_update,
            parameter={"dtype": "f32", "shape": RaisingShape()},
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"learning_rate": LEARNING_RATE_SPEC},
        )

    assert error.value.category == "missing_shape_metadata"
    assert "parameter" in error.value.message


# --------------------------------------------------------------------------
# The pre-check must accept exactly what the builder accepts.
# --------------------------------------------------------------------------


def test_an_int_subclass_shape_dimension_is_rejected_at_the_named_input() -> None:
    """An `int` subclass dimension is rejected, as it always has been.

    `TensorGraphBuilder.input` accepts any non-`bool` `int` instance, so it
    looked as though the spec pre-check had narrowed what a dimension may be.
    It has not: `parse_shape` requires ``type(dimension) is int`` and governs
    every later read of the recorded metadata, so an `IntEnum` dimension has
    always failed with this same category -- before the pre-check existed it
    simply failed a moment later, during finalization.

    What the pre-check does change is *where* it is reported, and this test
    pins that. Failing at the declaration names the offending input; failing
    downstream reports only ``tensor shape metadata``, which does not tell a
    caller which of their declarations was wrong. Loosening the pre-check to
    accept an `int` subclass would move the failure back to the vaguer site
    without making anything traceable, so this test fails if that is done.
    """

    class Dimension(enum.IntEnum):
        ROWS = 2
        COLUMNS = 3

    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            sgd_update,
            parameter={"dtype": "f32", "shape": (Dimension.ROWS, Dimension.COLUMNS)},
            gradient={"dtype": "f32", "shape": (Dimension.ROWS, Dimension.COLUMNS)},
            optimizer_inputs={"learning_rate": LEARNING_RATE_SPEC},
        )

    assert error.value.category == "missing_shape_metadata"
    assert "parameter" in error.value.message, (
        "the failure must name the declaration at fault, not just 'tensor "
        "shape metadata'"
    )


@pytest.mark.parametrize("boolean_dimension", [True, False], ids=["true", "false"])
def test_a_bool_shape_dimension_is_still_rejected(boolean_dimension) -> None:
    """The non-vacuity guard on the case above: `bool` is an `int` subclass.

    Widening the pre-check to every `int` instance must not quietly admit
    `bool`, which the builder rejects explicitly.
    """
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            sgd_update,
            parameter={"dtype": "f32", "shape": (boolean_dimension, 3)},
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"learning_rate": LEARNING_RATE_SPEC},
        )

    assert error.value.category == "missing_shape_metadata"


# --------------------------------------------------------------------------
# The tracer's spec read must behave exactly as the analysis helper it claims
# to mirror. Asserting only the tracer's own outcome would keep passing while
# the parity claim was false, so this is written as a differential: the claim
# is about agreement, so the test is about agreement.
# --------------------------------------------------------------------------


def _spec_whose_shape_lookup_raises(raised: type[BaseException]) -> Mapping:
    """A typed input spec whose ``"shape"`` lookup raises, and whose dtype does not.

    Note that this raises from ``__getitem__``, so it is reached by a
    membership test (`Mapping.__contains__` calls ``__getitem__``) just as
    much as by an explicit read.
    """

    class ShapeLookupRaises(Mapping):
        def __getitem__(self, key):
            if key == "shape":
                raise raised("boom from the shape lookup")
            return "f32"

        def __iter__(self):
            return iter(("dtype", "shape"))

        def __len__(self) -> int:
            return 2

    return ShapeLookupRaises()


def _spec_read_outcome(read, spec) -> tuple[str, str]:
    try:
        read(spec, label="parameter")
    except AutodiffError as error:
        return ("categorized", error.category)
    except Exception as error:  # noqa: BLE001 - the outcome under test
        return ("raw", type(error).__name__)
    return ("no raise", "")


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (IndexError, ("categorized", "missing_shape_metadata")),
        (TypeError, ("categorized", "missing_shape_metadata")),
        (ValueError, ("categorized", "missing_shape_metadata")),
        # Outside the three types either side normalizes: a shared limit,
        # not a divergence. Pinned so it stays shared.
        (RuntimeError, ("raw", "RuntimeError")),
    ],
    ids=["index_error", "type_error", "value_error", "runtime_error"],
)
def test_the_spec_read_matches_the_analysis_helper(raised, expected) -> None:
    tracer_outcome = _spec_read_outcome(
        training._typed_input_spec, _spec_whose_shape_lookup_raises(raised)
    )
    analysis_outcome = _spec_read_outcome(
        _complete_typespec, _spec_whose_shape_lookup_raises(raised)
    )

    assert tracer_outcome == analysis_outcome, (
        "the tracer's spec read and the analysis helper it mirrors must agree"
    )
    assert tracer_outcome == expected


@pytest.mark.parametrize(
    "raised",
    [IndexError, TypeError, ValueError],
    ids=["index_error", "type_error", "value_error"],
)
def test_a_shape_lookup_that_raises_is_categorized_through_the_public_api(raised) -> None:
    """The same defect as it actually reaches a consumer."""
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            sgd_update,
            parameter=_spec_whose_shape_lookup_raises(raised),
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"learning_rate": LEARNING_RATE_SPEC},
        )

    assert error.value.category == "missing_shape_metadata"
    assert "parameter" in error.value.message
