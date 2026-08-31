"""Unit tests for training-step declaration and loss-signature validation.

Pins the contract that every declaration mistake -- the declaration set
itself, the loss signature bound against the declared input names, and the
optimizer contract in both directions -- fails before the caller's loss body
is ever entered. Optimizer-contract and typed-input-spec checking is
delegated to the training module's own validators (one owner per check); this
module only owns the declaration set and the loss-signature binding.
"""

from __future__ import annotations

import pytest

from tinychain.autodiff.protocol import AUTODIFF_ERROR_CATEGORIES, AutodiffError
from tinychain.autodiff.training import SGD, Optimizer
from tinychain.autodiff.training_step import validate_declaration


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

INPUTS = {
    "x": {"dtype": "f32", "shape": (2, 3)},
    "w": {"dtype": "f32", "shape": (3, 4)},
}
PARAMETERS = ("w",)
OPTIMIZER_INPUTS = {"learning_rate": {"dtype": "f32", "shape": ()}}


class _CountingLoss:
    """A loss callable that records whether its body was ever entered."""

    def __init__(self, *, accepted_names: tuple[str, ...] = ("x", "w")) -> None:
        self.calls = 0
        self._accepted_names = accepted_names

    def __call__(self, **kwargs: object) -> object:
        self.calls += 1
        raise AssertionError(
            "loss body must not be invoked during declaration validation"
        )


def _valid_loss(*, x: object, w: object) -> object:
    raise AssertionError("loss body must not be invoked during declaration validation")


class _MismatchedSignatureOptimizer(Optimizer):
    """An optimizer whose declared inputs match SGD but whose update does not."""

    required_optimizer_inputs: tuple[str, ...] = ("learning_rate",)

    def update(self, *, parameter: object, gradient: object, wrong_name: object) -> object:
        return parameter


def _plain_update(*, parameter: object, gradient: object, learning_rate: object) -> object:
    return parameter


def _plain_update_wrong_signature(
    *, parameter: object, gradient: object, unexpected: object
) -> object:
    return parameter


def _call(
    *,
    inputs: object = INPUTS,
    parameters: object = PARAMETERS,
    loss: object = _valid_loss,
    optimizer: object = SGD(),
    optimizer_inputs: object = OPTIMIZER_INPUTS,
) -> None:
    validate_declaration(
        inputs=inputs,
        parameters=parameters,
        loss=loss,
        optimizer=optimizer,
        optimizer_inputs=optimizer_inputs,
    )


# --------------------------------------------------------------------------
# AC: AUTODIFF_ERROR_CATEGORIES grows by exactly four appended members
# --------------------------------------------------------------------------

_PRE_EXISTING_CATEGORIES = (
    "unsupported_operator",
    "missing_derivative_behavior",
    "missing_derivative_ir",
    "non_differentiable_route",
    "missing_shape_metadata",
    "missing_dtype_metadata",
    "dtype_not_differentiable",
    "shape_mismatch",
    "unresolved_symbolic_shape",
    "symbolic_shape_mismatch",
    "broadcast_shape_mismatch",
    "matmul_shape_mismatch",
    "invalid_permutation",
    "reduction_shape_mismatch",
    "unsupported_reduction",
    "seed_shape_mismatch",
    "malformed_derivative_ir",
    "side_effecting_route_unsupported",
    "autodiff_not_implemented",
    "dtype_mismatch",
    "missing_dependency",
    "ambiguous_producer",
    "invalid_selected_output",
    "handler_contract_violation",
    "invalid_update_signature",
    "invalid_update_output",
)

_NEW_CATEGORIES = (
    "invalid_training_declaration",
    "invalid_loss_signature",
    "invalid_loss_output",
    "expansion_contract_violation",
)


def test_autodiff_error_categories_appends_four_new_members_preserving_existing_order() -> None:
    assert len(AUTODIFF_ERROR_CATEGORIES) == len(_PRE_EXISTING_CATEGORIES) + 4
    assert (
        AUTODIFF_ERROR_CATEGORIES[: len(_PRE_EXISTING_CATEGORIES)]
        == _PRE_EXISTING_CATEGORIES
    )
    assert AUTODIFF_ERROR_CATEGORIES[len(_PRE_EXISTING_CATEGORIES) :] == _NEW_CATEGORIES


@pytest.mark.parametrize("category", _NEW_CATEGORIES)
def test_autodiff_error_accepts_each_new_category(category: str) -> None:
    error = AutodiffError(category, "message")
    assert error.category == category


# --------------------------------------------------------------------------
# AC: declaration-set violations raise invalid_training_declaration
# --------------------------------------------------------------------------


def test_validate_declaration_empty_inputs_raises_invalid_training_declaration() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _call(inputs={})

    assert excinfo.value.category == "invalid_training_declaration"
    assert "inputs" in excinfo.value.message


def test_validate_declaration_inputs_not_a_mapping_raises_invalid_training_declaration() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _call(inputs=["x", "w"])

    assert excinfo.value.category == "invalid_training_declaration"


def test_validate_declaration_empty_parameters_raises_invalid_training_declaration() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _call(parameters=())

    assert excinfo.value.category == "invalid_training_declaration"
    assert "parameters" in excinfo.value.message


def test_validate_declaration_parameters_bare_string_raises_invalid_training_declaration() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _call(parameters="w")

    assert excinfo.value.category == "invalid_training_declaration"


def test_validate_declaration_unknown_parameter_name_raises_invalid_training_declaration() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _call(parameters=("not_declared",))

    assert excinfo.value.category == "invalid_training_declaration"
    assert "not_declared" in excinfo.value.message


def test_validate_declaration_repeated_parameter_name_raises_invalid_training_declaration() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _call(parameters=("w", "w"))

    assert excinfo.value.category == "invalid_training_declaration"
    assert "w" in excinfo.value.message


def test_validate_declaration_parameter_repeated_three_times_raises_invalid_training_declaration() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _call(parameters=("w", "w", "w"))

    assert excinfo.value.category == "invalid_training_declaration"


# --------------------------------------------------------------------------
# AC: malformed typed input specs are rejected by the training module's own
# typed-input-spec validator (categories it already owns)
# --------------------------------------------------------------------------


def test_validate_declaration_input_missing_dtype_raises_missing_dtype_metadata() -> None:
    inputs = {
        "x": {"shape": (2, 3)},
        "w": {"dtype": "f32", "shape": (3, 4)},
    }
    with pytest.raises(AutodiffError) as excinfo:
        _call(inputs=inputs)

    assert excinfo.value.category == "missing_dtype_metadata"


def test_validate_declaration_input_missing_shape_raises_missing_shape_metadata() -> None:
    inputs = {
        "x": {"dtype": "f32"},
        "w": {"dtype": "f32", "shape": (3, 4)},
    }
    with pytest.raises(AutodiffError) as excinfo:
        _call(inputs=inputs)

    assert excinfo.value.category == "missing_shape_metadata"


def test_validate_declaration_input_malformed_shape_raises_missing_shape_metadata() -> None:
    inputs = {
        "x": {"dtype": "f32", "shape": object()},
        "w": {"dtype": "f32", "shape": (3, 4)},
    }
    with pytest.raises(AutodiffError) as excinfo:
        _call(inputs=inputs)

    assert excinfo.value.category == "missing_shape_metadata"


# --------------------------------------------------------------------------
# AC: optimizer contract validated in both directions, through the training
# module's existing validators -- both raise invalid_update_signature
# --------------------------------------------------------------------------


def test_validate_declaration_optimizer_required_inputs_disagree_raises_invalid_update_signature() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _call(optimizer=SGD(), optimizer_inputs={"momentum": {"dtype": "f32", "shape": ()}})

    assert excinfo.value.category == "invalid_update_signature"


def test_validate_declaration_optimizer_update_signature_mismatch_raises_invalid_update_signature() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _call(optimizer=_MismatchedSignatureOptimizer())

    assert excinfo.value.category == "invalid_update_signature"


def test_validate_declaration_plain_callable_optimizer_matching_signature_succeeds() -> None:
    _call(optimizer=_plain_update)


def test_validate_declaration_plain_callable_optimizer_wrong_signature_raises_invalid_update_signature() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _call(optimizer=_plain_update_wrong_signature)

    assert excinfo.value.category == "invalid_update_signature"


# --------------------------------------------------------------------------
# AC: loss-signature binding against exactly the declared input names
# --------------------------------------------------------------------------


def test_validate_declaration_loss_wrong_names_raises_invalid_loss_signature() -> None:
    def loss(*, x: object, other: object) -> object:
        raise AssertionError("must not be invoked")

    with pytest.raises(AutodiffError) as excinfo:
        _call(loss=loss)

    assert excinfo.value.category == "invalid_loss_signature"
    assert "x" in excinfo.value.message or "w" in excinfo.value.message


def test_validate_declaration_loss_too_few_names_raises_invalid_loss_signature() -> None:
    def loss(*, x: object) -> object:
        raise AssertionError("must not be invoked")

    with pytest.raises(AutodiffError) as excinfo:
        _call(loss=loss)

    assert excinfo.value.category == "invalid_loss_signature"


def test_validate_declaration_loss_too_many_names_raises_invalid_loss_signature() -> None:
    def loss(*, x: object, w: object, extra: object) -> object:
        raise AssertionError("must not be invoked")

    with pytest.raises(AutodiffError) as excinfo:
        _call(loss=loss)

    assert excinfo.value.category == "invalid_loss_signature"


def test_validate_declaration_loss_only_var_args_raises_invalid_loss_signature() -> None:
    def loss(*args: object) -> object:
        raise AssertionError("must not be invoked")

    with pytest.raises(AutodiffError) as excinfo:
        _call(loss=loss)

    assert excinfo.value.category == "invalid_loss_signature"


def test_validate_declaration_loss_not_callable_raises_invalid_loss_signature() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _call(loss="not callable")

    assert excinfo.value.category == "invalid_loss_signature"


def test_validate_declaration_loss_keyword_only_matching_names_binds_without_error() -> None:
    def loss(*, x: object, w: object) -> object:
        raise AssertionError("must not be invoked")

    _call(loss=loss)


def test_validate_declaration_loss_with_default_arguments_for_declared_names_binds_without_error() -> None:
    def loss(x: object, w: object = None) -> object:
        raise AssertionError("must not be invoked")

    _call(loss=loss)


def test_validate_declaration_valid_declaration_raises_nothing() -> None:
    _call()


# --------------------------------------------------------------------------
# AC: the loss body is never invoked for any declaration failure
# --------------------------------------------------------------------------

_FAILURE_SCENARIOS = {
    "empty_inputs": {"inputs": {}},
    "empty_parameters": {"parameters": ()},
    "unknown_parameter": {"parameters": ("not_declared",)},
    "repeated_parameter": {"parameters": ("w", "w")},
    "missing_dtype": {
        "inputs": {"x": {"shape": (2, 3)}, "w": {"dtype": "f32", "shape": (3, 4)}}
    },
    "missing_shape": {
        "inputs": {"x": {"dtype": "f32"}, "w": {"dtype": "f32", "shape": (3, 4)}}
    },
    "optimizer_inputs_disagree": {
        "optimizer_inputs": {"momentum": {"dtype": "f32", "shape": ()}}
    },
    "optimizer_update_signature_mismatch": {"optimizer": _MismatchedSignatureOptimizer()},
}


@pytest.mark.parametrize("scenario", _FAILURE_SCENARIOS, ids=list(_FAILURE_SCENARIOS))
def test_validate_declaration_never_invokes_loss_on_failure(scenario: str) -> None:
    counting_loss = _CountingLoss()
    overrides = dict(_FAILURE_SCENARIOS[scenario])
    overrides["loss"] = counting_loss

    with pytest.raises(AutodiffError):
        _call(**overrides)

    assert counting_loss.calls == 0


def test_validate_declaration_never_invokes_loss_on_loss_signature_mismatch() -> None:
    counting_loss = _CountingLoss()

    def mismatched_loss(*, only_one: object) -> object:
        counting_loss.calls += 1
        raise AssertionError("must not be invoked")

    with pytest.raises(AutodiffError) as excinfo:
        _call(loss=mismatched_loss)

    assert excinfo.value.category == "invalid_loss_signature"
    assert counting_loss.calls == 0


# --------------------------------------------------------------------------
# AC: no framework-raised failure is a bare KeyError/TypeError/ValueError/
# AssertionError
# --------------------------------------------------------------------------

_BARE_EXCEPTION_SCENARIOS = {
    "inputs_not_a_mapping": {"inputs": ["x", "w"]},
    "parameters_bare_string": {"parameters": "w"},
    "parameter_repeated_three_times": {"parameters": ("w", "w", "w")},
    "loss_not_callable": {"loss": "not callable"},
    "plain_optimizer_wrong_signature": {"optimizer": _plain_update_wrong_signature},
}


@pytest.mark.parametrize(
    "scenario", _BARE_EXCEPTION_SCENARIOS, ids=list(_BARE_EXCEPTION_SCENARIOS)
)
def test_validate_declaration_never_raises_bare_python_exception(scenario: str) -> None:
    overrides = _BARE_EXCEPTION_SCENARIOS[scenario]

    try:
        _call(**overrides)
    except AutodiffError:
        pass
    except (KeyError, TypeError, ValueError, AssertionError) as exc:
        pytest.fail(f"bare {type(exc).__name__} escaped: {exc}")
