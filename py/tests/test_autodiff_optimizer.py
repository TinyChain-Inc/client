"""Unit tests for the abstract optimizer contract and its first implementation.

An update expression and the names of the inputs that expression reads are two
facts that must agree. Today a consumer keeps them in agreement by hand: it
imports an update function by name and separately knows to declare a learning
rate, and nothing checks the pair. These tests pin the contract that binds
them -- an abstract optimizer owning the update expression, the names of the
optimizer inputs it reads, and the validation of its own configuration, with
stochastic gradient descent as the one concrete implementation.

They also pin the sharper reason the declared names exist. An optimizer
instance is callable through a `__call__` that accepts arbitrary keywords, so
`inspect.signature(...).bind(...)` accepts *any* declaration for it; the
existing signature validation is therefore vacuous for exactly the values this
contract introduces. `test_signature_binding_alone_would_accept_a_declaration_
the_optimizer_rejects` demonstrates the vacuity directly and then shows the
declared-name check catching what binding let through.

No optimizer implementation constructs a `TensorNodeRecord` or a concrete
`TensorOperator`; the update is authored in ordinary Tensor operations only.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import tinychain as tc
from tinychain.autodiff import (
    AUTODIFF_ERROR_CATEGORIES,
    AutodiffError,
    MulOperator,
    SubOperator,
    TensorGraphBuilder,
    TensorNodeRecord,
    TensorOperator,
    get_active_builder,
)
from tinychain.autodiff.lowering import OperationHandlerRegistry, lower_graph
from tinychain.autodiff import training
from tinychain.autodiff.training import (
    SGD,
    Optimizer,
    TracedUpdate,
    sgd_update,
    trace_parameter_update,
)

from tests.autodiff_execution import NumpyAutodiffDispatcher


PARAMETER_SPEC = {"dtype": "f32", "shape": (2, 3)}
GRADIENT_SPEC = {"dtype": "f32", "shape": (2, 3)}
RATE_SPEC = {"dtype": "f32", "shape": ()}


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


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


def _graph_shape(traced: TracedUpdate) -> dict[str, object]:
    """A comparable structural description of one traced update.

    Value and node ids are assigned by a per-builder counter in declaration
    order, so two traces declaring the same inputs in the same order produce
    identical ids. Comparing this description therefore compares the graphs
    themselves, not merely their silhouettes.
    """
    return {
        "inputs": [(value_id, typespec) for value_id, typespec in traced.graph.inputs],
        "outputs": list(traced.graph.outputs),
        "updated_parameter_id": traced.updated_parameter_id,
        "input_value_ids": dict(traced.input_value_ids),
        "nodes": [
            (
                node.node_id,
                type(node.operator).__name__,
                dict(node.op_params),
                list(node.input_value_ids),
                node.output_value_id,
                node.output_typespec,
            )
            for node in traced.graph.nodes
        ],
    }


def _execute(traced: TracedUpdate, bindings: dict[str, object]) -> object:
    bound_values = {
        traced.input_value_ids[name]: value for name, value in bindings.items()
    }
    lowered = lower_graph(
        traced.graph,
        handlers=_numeric_registry(),
        outputs=[traced.updated_parameter_id],
        bind_input=lambda dependency: bound_values[dependency.value_id],
    )
    (updated_parameter,) = lowered.output_values
    return updated_parameter


PARAMETER_VALUE = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
GRADIENT_VALUE = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
RATE_VALUE = np.float32(0.5)


# --------------------------------------------------------------------------
# consumer-written optimizers, authored against the public contract alone
# --------------------------------------------------------------------------


class _ScaledStep(Optimizer):
    """The smallest optimizer a consumer can write against the public contract."""

    required_optimizer_inputs = ("step_size",)

    def update(self, *, parameter, gradient, step_size):
        return parameter - step_size * gradient


class _CorrectedStep(Optimizer):
    """A consumer optimizer that has configuration, and validates it itself.

    Its configuration decides which optimizer inputs its expression reads, so
    the declared-name check has to agree with a per-instance answer rather
    than a per-class one.
    """

    def __init__(self, *, corrected: bool) -> None:
        if not isinstance(corrected, bool):
            raise ValueError(
                f"corrected must be a bool, got {type(corrected).__name__!r}"
            )
        self._corrected = corrected

    @property
    def required_optimizer_inputs(self):
        if self._corrected:
            return ("step_size", "correction")
        return ("step_size",)

    def update(self, *, parameter, gradient, step_size, **optimizer_inputs):
        stepped = parameter - step_size * gradient
        if self._corrected:
            return stepped - optimizer_inputs["correction"]
        return stepped


# --------------------------------------------------------------------------
# the abstract contract
# --------------------------------------------------------------------------


def test_the_optimizer_contract_has_exactly_the_two_members_a_consumer_needs() -> None:
    assert inspect.isabstract(Optimizer)
    assert set(Optimizer.__abstractmethods__) == {"required_optimizer_inputs", "update"}
    # The call path is concrete: an instance is usable wherever a callable is.
    assert callable(Optimizer.__call__)
    assert "__call__" not in Optimizer.__abstractmethods__


def test_an_optimizer_missing_its_update_expression_cannot_be_instantiated() -> None:
    class _NoUpdate(Optimizer):
        required_optimizer_inputs = ("step_size",)

    with pytest.raises(TypeError, match="update"):
        _NoUpdate()


def test_an_optimizer_missing_its_declared_input_names_cannot_be_instantiated() -> None:
    class _NoNames(Optimizer):
        def update(self, *, parameter, gradient, step_size):
            return parameter - step_size * gradient

    with pytest.raises(TypeError, match="required_optimizer_inputs"):
        _NoNames()


def test_an_optimizer_instance_is_callable_and_delegates_to_its_update() -> None:
    optimizer = _ScaledStep()
    calls: list[dict[str, object]] = []

    class _Recording(_ScaledStep):
        def update(self, **inputs):
            calls.append(inputs)
            return "updated"

    assert callable(optimizer)
    assert _Recording()(parameter="p", gradient="g", step_size="s") == "updated"
    assert calls == [{"parameter": "p", "gradient": "g", "step_size": "s"}]


def test_an_optimizer_validates_its_own_configuration() -> None:
    with pytest.raises(ValueError, match="corrected"):
        _CorrectedStep(corrected="yes")

    assert _CorrectedStep(corrected=False).required_optimizer_inputs == ("step_size",)
    assert _CorrectedStep(corrected=True).required_optimizer_inputs == (
        "step_size",
        "correction",
    )


# --------------------------------------------------------------------------
# stochastic gradient descent, the first concrete implementation
# --------------------------------------------------------------------------


def test_sgd_is_a_concrete_optimizer_declaring_one_required_input() -> None:
    optimizer = SGD()

    assert isinstance(optimizer, Optimizer)
    assert tuple(optimizer.required_optimizer_inputs) == ("learning_rate",)


def test_sgd_traces_the_same_graph_as_the_reference_expression() -> None:
    traced_optimizer = trace_parameter_update(
        SGD(),
        parameter=PARAMETER_SPEC,
        gradient=GRADIENT_SPEC,
        optimizer_inputs={"learning_rate": RATE_SPEC},
    )
    traced_reference = trace_parameter_update(
        sgd_update,
        parameter=PARAMETER_SPEC,
        gradient=GRADIENT_SPEC,
        optimizer_inputs={"learning_rate": RATE_SPEC},
    )

    assert isinstance(traced_optimizer, TracedUpdate)
    assert [type(node.operator) for node in traced_optimizer.graph.nodes] == [
        MulOperator,
        SubOperator,
    ]
    assert _graph_shape(traced_optimizer) == _graph_shape(traced_reference)


def test_sgd_execution_matches_the_analytical_update() -> None:
    traced_optimizer = trace_parameter_update(
        SGD(),
        parameter=PARAMETER_SPEC,
        gradient=GRADIENT_SPEC,
        optimizer_inputs={"learning_rate": RATE_SPEC},
    )
    traced_reference = trace_parameter_update(
        sgd_update,
        parameter=PARAMETER_SPEC,
        gradient=GRADIENT_SPEC,
        optimizer_inputs={"learning_rate": RATE_SPEC},
    )

    bindings = {
        "parameter": PARAMETER_VALUE,
        "gradient": GRADIENT_VALUE,
        "learning_rate": RATE_VALUE,
    }
    expected = PARAMETER_VALUE - RATE_VALUE * GRADIENT_VALUE
    # Not tautological: `expected` is computed in NumPy from the same inputs,
    # independently of anything the traced graph did.
    assert not np.allclose(np.asarray(expected), PARAMETER_VALUE)

    np.testing.assert_allclose(np.asarray(_execute(traced_optimizer, bindings)), expected)
    np.testing.assert_allclose(
        np.asarray(_execute(traced_optimizer, bindings)),
        np.asarray(_execute(traced_reference, bindings)),
    )


# --------------------------------------------------------------------------
# a consumer optimizer traces through the same public path
# --------------------------------------------------------------------------


def test_a_consumer_written_optimizer_traces_through_the_same_helper() -> None:
    traced = tc.autodiff.trace_parameter_update(
        _ScaledStep(),
        parameter=PARAMETER_SPEC,
        gradient=GRADIENT_SPEC,
        optimizer_inputs={"step_size": RATE_SPEC},
    )

    assert set(traced.input_value_ids) == {"parameter", "gradient", "step_size"}
    assert [type(node.operator) for node in traced.graph.nodes] == [
        MulOperator,
        SubOperator,
    ]

    updated = _execute(
        traced,
        {
            "parameter": PARAMETER_VALUE,
            "gradient": GRADIENT_VALUE,
            "step_size": RATE_VALUE,
        },
    )
    np.testing.assert_allclose(
        np.asarray(updated), PARAMETER_VALUE - RATE_VALUE * GRADIENT_VALUE
    )


def test_a_configured_consumer_optimizer_traces_the_inputs_its_configuration_declares() -> None:
    correction_value = np.full((2, 3), 0.25, dtype=np.float32)
    traced = trace_parameter_update(
        _CorrectedStep(corrected=True),
        parameter=PARAMETER_SPEC,
        gradient=GRADIENT_SPEC,
        optimizer_inputs={"step_size": RATE_SPEC, "correction": GRADIENT_SPEC},
    )

    assert set(traced.input_value_ids) == {
        "parameter",
        "gradient",
        "step_size",
        "correction",
    }

    updated = _execute(
        traced,
        {
            "parameter": PARAMETER_VALUE,
            "gradient": GRADIENT_VALUE,
            "step_size": RATE_VALUE,
            "correction": correction_value,
        },
    )
    np.testing.assert_allclose(
        np.asarray(updated),
        PARAMETER_VALUE - RATE_VALUE * GRADIENT_VALUE - correction_value,
    )


# --------------------------------------------------------------------------
# the declared names are what restores a real check on the optimizer path
# --------------------------------------------------------------------------


def test_signature_binding_alone_would_accept_a_declaration_the_optimizer_rejects() -> None:
    """Binding the *call path* is vacuous; binding is not useless in general.

    An optimizer instance is invoked through `Optimizer.__call__`, which
    accepts arbitrary keywords, so binding *that* signature accepts any
    declared input set at all -- which is why the declared names are checked
    as well. It does not follow that binding has nothing to say here: binding
    the `update` method, which is what actually runs, rejects the same
    declaration. The two checks catch different mistakes, so the fix is to
    bind `update` *and* check the names, not to drop binding.
    """
    optimizer = SGD()

    # Vacuous on the call path: this raises nothing at all.
    inspect.signature(optimizer).bind(parameter=None, gradient=None, momentum=None)

    # Not vacuous on the method that actually runs.
    with pytest.raises(TypeError, match="learning_rate"):
        inspect.signature(optimizer.update).bind(
            parameter=None, gradient=None, momentum=None
        )

    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            optimizer,
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"momentum": RATE_SPEC},
        )
    assert error.value.category == "invalid_update_signature"
    assert "learning_rate" in error.value.message
    assert "momentum" in error.value.message


def test_a_declaration_missing_a_required_optimizer_input_is_rejected() -> None:
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            SGD(),
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
        )
    assert error.value.category == "invalid_update_signature"
    assert "learning_rate" in error.value.message
    assert get_active_builder() is None


def test_a_declaration_with_an_extra_optimizer_input_is_rejected() -> None:
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            SGD(),
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"learning_rate": RATE_SPEC, "momentum": RATE_SPEC},
        )
    assert error.value.category == "invalid_update_signature"
    assert "momentum" in error.value.message
    assert get_active_builder() is None


def test_a_mismatched_declaration_fails_before_the_update_expression_runs() -> None:
    calls: list[object] = []

    class _Recording(_ScaledStep):
        def update(self, **inputs):
            calls.append(inputs)
            raise AssertionError("the update expression must not have run")

    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            _Recording(),
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"learning_rate": RATE_SPEC},
        )
    assert error.value.category == "invalid_update_signature"
    assert calls == []
    assert get_active_builder() is None


def test_a_mismatched_declaration_is_rejected_before_a_malformed_spec_is_read() -> None:
    """Ordering is unchanged: the declared-input check still precedes spec reading."""
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            SGD(),
            parameter=None,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"momentum": RATE_SPEC},
        )
    assert error.value.category == "invalid_update_signature"


def test_a_reserved_optimizer_input_name_is_still_rejected_for_an_optimizer() -> None:
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            SGD(),
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"parameter": PARAMETER_SPEC},
        )
    assert error.value.category == "invalid_update_signature"
    assert get_active_builder() is None


def test_the_optimizer_path_needs_no_new_error_category() -> None:
    # A declared-input mismatch is an update-signature failure by any other
    # name, so the public category surface is unchanged by this contract.
    assert "invalid_update_signature" in AUTODIFF_ERROR_CATEGORIES


# --------------------------------------------------------------------------
# the plain-callable path is unchanged
# --------------------------------------------------------------------------


def test_a_plain_callable_is_still_validated_by_its_signature() -> None:
    def missing_gradient(*, parameter, learning_rate):
        return parameter

    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            missing_gradient,
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"learning_rate": RATE_SPEC},
        )
    assert error.value.category == "invalid_update_signature"
    assert get_active_builder() is None


def test_a_keyword_absorbing_callable_still_binds_any_declaration() -> None:
    """Pins the callable path as it is: this is the hole the names close.

    A plain callable taking `**kwargs` binds any declared input set, so it
    reaches tracing and its body decides. That is existing behaviour and this
    change does not alter it -- it is precisely why an optimizer, which is
    always keyword-absorbing at its call path, needs a different check.
    """

    def absorbing(**inputs):
        return inputs["parameter"] - inputs["momentum"] * inputs["gradient"]

    traced = trace_parameter_update(
        absorbing,
        parameter=PARAMETER_SPEC,
        gradient=GRADIENT_SPEC,
        optimizer_inputs={"momentum": RATE_SPEC},
    )
    assert set(traced.input_value_ids) == {"parameter", "gradient", "momentum"}


def test_a_non_callable_non_optimizer_update_is_still_rejected() -> None:
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            object(),
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"learning_rate": RATE_SPEC},
        )
    assert error.value.category == "invalid_update_signature"


def test_tracing_an_optimizer_still_requires_an_inactive_trace() -> None:
    with TensorGraphBuilder():
        with pytest.raises(RuntimeError, match="Nested"):
            trace_parameter_update(
                SGD(),
                parameter=PARAMETER_SPEC,
                gradient=GRADIENT_SPEC,
                optimizer_inputs={"learning_rate": RATE_SPEC},
            )
    assert get_active_builder() is None


def test_an_optimizer_returning_a_non_tensor_is_still_rejected() -> None:
    class _BadOutput(_ScaledStep):
        def update(self, *, parameter, gradient, step_size):
            return 5.0

    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            _BadOutput(),
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"step_size": RATE_SPEC},
        )
    assert error.value.category == "invalid_update_output"
    assert get_active_builder() is None


# --------------------------------------------------------------------------
# a failure message names whichever form actually failed
#
# The category is shared, and must stay shared -- the two paths fail for the
# same reason. Only the noun differs, because a consumer reading the message
# at failure time needs to know whether to look at the callable they passed or
# at their optimizer's update method. Nothing pinned this text before, which is
# how it came to name the wrong one.
# --------------------------------------------------------------------------


def _error_from(update, **optimizer_inputs) -> AutodiffError:
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            update,
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs=optimizer_inputs or None,
        )
    assert get_active_builder() is None
    return error.value


def test_a_non_tensor_output_names_whichever_form_returned_it() -> None:
    class _BadOptimizerOutput(Optimizer):
        required_optimizer_inputs = ("step_size",)

        def update(self, *, parameter, gradient, step_size):
            return 5.0

    def bad_callable(*, parameter, gradient, step_size):
        return 5.0

    optimizer_error = _error_from(_BadOptimizerOutput(), step_size=RATE_SPEC)
    callable_error = _error_from(bad_callable, step_size=RATE_SPEC)

    # One category for one kind of mistake, on both paths.
    assert (
        optimizer_error.category
        == callable_error.category
        == "invalid_update_output"
    )

    # Both still report what was actually returned.
    assert "'float'" in optimizer_error.message
    assert "'float'" in callable_error.message

    # The optimizer path names the optimizer and its update method, and does
    # not call it a callable the consumer passed.
    assert "_BadOptimizerOutput" in optimizer_error.message
    assert "update method" in optimizer_error.message
    assert "update callable" not in optimizer_error.message

    # The plain-callable path is unchanged, and does not borrow the optimizer
    # wording.
    assert "update callable" in callable_error.message
    assert "update method" not in callable_error.message

    # They cannot converge again without this failing.
    assert optimizer_error.message != callable_error.message


# --------------------------------------------------------------------------
# the existing reference function is a compatibility path
# --------------------------------------------------------------------------


def test_the_reference_update_function_is_still_importable_and_unchanged() -> None:
    from tinychain.autodiff import sgd_update as exported_sgd_update

    assert exported_sgd_update is sgd_update
    assert exported_sgd_update is training.sgd_update

    signature = inspect.signature(sgd_update)
    assert set(signature.parameters) == {"parameter", "gradient", "learning_rate"}
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


def test_the_reference_update_function_delegates_to_the_concrete_optimizer() -> None:
    traced_reference = trace_parameter_update(
        sgd_update,
        parameter=PARAMETER_SPEC,
        gradient=GRADIENT_SPEC,
        optimizer_inputs={"learning_rate": RATE_SPEC},
    )
    traced_optimizer = trace_parameter_update(
        SGD(),
        parameter=PARAMETER_SPEC,
        gradient=GRADIENT_SPEC,
        optimizer_inputs={"learning_rate": RATE_SPEC},
    )
    assert _graph_shape(traced_reference) == _graph_shape(traced_optimizer)

    bindings = {
        "parameter": PARAMETER_VALUE,
        "gradient": GRADIENT_VALUE,
        "learning_rate": RATE_VALUE,
    }
    np.testing.assert_allclose(
        np.asarray(_execute(traced_reference, bindings)),
        PARAMETER_VALUE - RATE_VALUE * GRADIENT_VALUE,
    )


# --------------------------------------------------------------------------
# no optimizer implementation constructs a graph, a record, or an operator
# --------------------------------------------------------------------------


def test_no_optimizer_implementation_constructs_a_graph_record_or_operator() -> None:
    forbidden_constructors = (
        "TensorNodeRecord(",
        "TensorGraph(",
        "TensorGraphBuilder(",
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
    for implementation in (Optimizer, SGD):
        source = inspect.getsource(implementation)
        for token in forbidden_constructors:
            assert token not in source, (
                f"{implementation.__name__} constructs {token!r} directly"
            )

    # An import alias bound in the class namespace would evade the token scan
    # above, so the class namespaces are scanned for the types themselves too.
    for implementation in (Optimizer, SGD):
        for name, value in vars(implementation).items():
            if name.startswith("__"):
                continue
            assert value is not TensorNodeRecord, (
                f"{implementation.__name__} binds TensorNodeRecord as {name!r}"
            )
            assert not (isinstance(value, type) and issubclass(value, TensorOperator)), (
                f"{implementation.__name__} binds a TensorOperator subclass as {name!r}"
            )


# --------------------------------------------------------------------------
# the optimizer path is checked at least as strictly as the callable path
#
# Binding and the declared-name check cover different mistakes. Binding
# catches an implementation whose `update` parameters do not match the
# declared inputs; the name check catches a declaration naming inputs the
# expression never reads. An optimizer must get both, because dropping
# binding would make this path strictly weaker than the plain-callable path
# for exactly the mistakes it was introduced to catch. Each test below states
# one mistake in both forms and requires the same category from each.
# --------------------------------------------------------------------------


class _UpdateMissingRequiredParameter(Optimizer):
    """Declares an input its update method does not accept."""

    required_optimizer_inputs = ("learning_rate",)

    def update(self, *, parameter, gradient):
        return parameter - gradient


class _UpdateWithAnUndeclaredRequiredParameter(Optimizer):
    """Its update method mandates a parameter nothing declares."""

    required_optimizer_inputs = ("learning_rate",)

    def update(self, *, parameter, gradient, learning_rate, momentum):
        return parameter - learning_rate * gradient


def _category_of(update, **optimizer_inputs) -> str:
    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            update,
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs=optimizer_inputs or None,
        )
    assert get_active_builder() is None
    return error.value.category


def test_an_update_method_missing_a_declared_input_matches_the_callable_path() -> None:
    def equivalent_callable(*, parameter, gradient):
        return parameter - gradient

    optimizer_category = _category_of(
        _UpdateMissingRequiredParameter(), learning_rate=RATE_SPEC
    )
    callable_category = _category_of(equivalent_callable, learning_rate=RATE_SPEC)

    assert optimizer_category == callable_category == "invalid_update_signature"


def test_an_update_method_with_an_undeclared_parameter_matches_the_callable_path() -> None:
    def equivalent_callable(*, parameter, gradient, learning_rate, momentum):
        return parameter - learning_rate * gradient

    optimizer_category = _category_of(
        _UpdateWithAnUndeclaredRequiredParameter(), learning_rate=RATE_SPEC
    )
    callable_category = _category_of(equivalent_callable, learning_rate=RATE_SPEC)

    assert optimizer_category == callable_category == "invalid_update_signature"


def test_no_optimizer_declaration_fault_escapes_uncategorized() -> None:
    """Every fault on this path leaves through AutodiffError, never raw.

    The optimizer path must not be the one entry into this function that lets
    a raw TypeError out, which is what the three changes beneath this one were
    for.
    """
    for optimizer in (
        _UpdateMissingRequiredParameter(),
        _UpdateWithAnUndeclaredRequiredParameter(),
    ):
        with pytest.raises(AutodiffError):
            trace_parameter_update(
                optimizer,
                parameter=PARAMETER_SPEC,
                gradient=GRADIENT_SPEC,
                optimizer_inputs={"learning_rate": RATE_SPEC},
            )


def test_binding_the_update_method_does_not_make_the_name_check_redundant() -> None:
    """Both checks are needed: this one binds anything, and still must fail."""

    class _AbsorbingUpdate(Optimizer):
        required_optimizer_inputs = ("learning_rate",)

        def update(self, **optimizer_inputs):
            return optimizer_inputs["parameter"]

    # Binding says nothing about this implementation.
    inspect.signature(_AbsorbingUpdate().update).bind(
        parameter=None, gradient=None, momentum=None
    )

    assert (
        _category_of(_AbsorbingUpdate(), momentum=RATE_SPEC)
        == "invalid_update_signature"
    )


# --------------------------------------------------------------------------
# a malformed declaration of the required names is categorized too
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "declared",
    [
        pytest.param(None, id="none"),
        pytest.param(42, id="int"),
        pytest.param(object(), id="not_iterable"),
        pytest.param("learning_rate", id="a_bare_string_is_not_a_name_collection"),
        pytest.param(("learning_rate", 7), id="a_name_that_is_not_a_string"),
        pytest.param(("",), id="an_empty_name"),
    ],
)
def test_a_malformed_required_input_declaration_is_categorized(declared) -> None:
    class _MalformedDeclaration(Optimizer):
        required_optimizer_inputs = declared

        def update(self, *, parameter, gradient, **optimizer_inputs):
            return parameter - gradient

    with pytest.raises(AutodiffError) as error:
        trace_parameter_update(
            _MalformedDeclaration(),
            parameter=PARAMETER_SPEC,
            gradient=GRADIENT_SPEC,
            optimizer_inputs={"learning_rate": RATE_SPEC},
        )
    assert error.value.category == "invalid_update_signature"
    assert get_active_builder() is None


# --------------------------------------------------------------------------
# public export surface
# --------------------------------------------------------------------------


def test_the_optimizer_contract_is_exported_from_the_autodiff_package() -> None:
    import tinychain.autodiff as autodiff

    assert {"Optimizer", "SGD"}.issubset(set(autodiff.__all__))
    assert autodiff.Optimizer is training.Optimizer
    assert autodiff.SGD is training.SGD

    assert not hasattr(tc, "Optimizer")
    assert not hasattr(tc, "SGD")
