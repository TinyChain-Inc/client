from __future__ import annotations

import contextlib
import dataclasses

import pytest

from tinychain.autodiff import (
    AUTODIFF_ERROR_CATEGORIES,
    AutodiffError,
    AutodiffRequest,
    AutodiffResult,
    DerivativeMetadata,
)


def test_autodiff_request_round_trips():
    request = AutodiffRequest(
        graph={"nodes": []},
        output_value_id="out",
        wrt=["x", "y"],
        seed_value_id="seed",
        tensor_op_contract_version="0.1.0",
        transform_version="0.1.0",
    )

    assert AutodiffRequest.from_dict(request.to_dict()) == request


def test_autodiff_result_and_metadata_round_trip():
    metadata = DerivativeMetadata(
        source_graph_id="graph-1",
        transform_version="0.1.0",
        tensor_op_contract_version="0.1.0",
        wrt_signature=("x", "y"),
        seed_contract="seed matches out",
    )
    result = AutodiffResult(gradients=["dx", "dy"], metadata=metadata)

    assert AutodiffResult.from_dict(result.to_dict()) == result


def test_autodiff_error_categories_cover_spec_codes():
    expected = {
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
        # Traced optimizer/parameter updates authored as ordinary Tensor
        # callables. `details` are carried in the message as the offending
        # callable's signature mismatch or the value the callable returned.
        #   invalid_update_signature: the update callable's signature does not
        #                             accept exactly the declared typed inputs
        #   invalid_update_output:    the update callable did not return a Tensor
        "invalid_update_signature",
        "invalid_update_output",
        # Reusable training-step compile orchestration. `details` are carried
        # in the message as the offending declaration, callable, or expansion
        # pass.
        #   invalid_training_declaration: `inputs` or `parameters` is empty,
        #                                 a parameter name is unknown or
        #                                 repeated, or `parameter(name)` is
        #                                 asked for an undeclared name
        #   invalid_loss_signature:      the loss callable cannot be bound
        #                                 against exactly the declared input
        #                                 names
        #   invalid_loss_output:         the loss callable returned something
        #                                 other than a single Tensor
        #   expansion_contract_violation: an expansion pass returned the
        #                                 wrong type, raised a
        #                                 non-AutodiffError exception, or
        #                                 produced an artifact missing a
        #                                 required semantic value
        "invalid_training_declaration",
        "invalid_loss_signature",
        "invalid_loss_output",
        "expansion_contract_violation",
    }

    assert set(AUTODIFF_ERROR_CATEGORIES) == expected


def test_autodiff_error_round_trip_and_rejects_unknown_category():
    error = AutodiffError("unsupported_operator", "no rule")
    assert AutodiffError.from_dict(error.to_dict()) == error

    with pytest.raises(ValueError, match="unknown autodiff error category"):
        AutodiffError("not_real", "bad")


# --------------------------------------------------------------------------
# The categorized error must survive an ordinary consumer context manager.
#
# `AutodiffError` is a frozen dataclass subclassing `Exception`. Python's
# exception machinery assigns `__traceback__` (and, at Python level,
# `__cause__`/`__context__`/`__notes__`) on the propagating instance, so a
# frozen `__setattr__` that refuses those assignments replaces the categorized
# error with a `FrozenInstanceError` and erases the whole error contract.
# --------------------------------------------------------------------------


def test_error_keeps_its_category_through_a_generator_context_manager():
    """A generator-based `@contextlib.contextmanager` is the ordinary idiom.

    `contextlib._GeneratorContextManager.__exit__` re-raises by assigning
    `value.__traceback__` in Python, which is exactly the assignment a frozen
    dataclass refuses.
    """

    @contextlib.contextmanager
    def consumer_scope():
        yield

    with pytest.raises(AutodiffError) as raised:
        with consumer_scope():
            raise AutodiffError("invalid_selected_output", "nothing was selected")

    assert raised.value.category == "invalid_selected_output"
    assert raised.value.message == "nothing was selected"
    assert str(raised.value) == "invalid_selected_output: nothing was selected"


def test_error_keeps_its_category_through_a_class_based_context_manager():
    """The class-based sibling, pinned so both idioms are proven equivalent."""

    class ConsumerScope:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    with pytest.raises(AutodiffError) as raised:
        with ConsumerScope():
            raise AutodiffError("invalid_selected_output", "nothing was selected")

    assert raised.value.category == "invalid_selected_output"


def test_error_accepts_the_attributes_python_exception_machinery_sets():
    error = AutodiffError("unsupported_operator", "no rule")
    cause = ValueError("root cause")

    error.__traceback__ = None
    error.__cause__ = cause
    error.__context__ = cause
    error.__suppress_context__ = True
    error.add_note("consumer supplied no handler")

    assert error.__cause__ is cause
    assert error.__context__ is cause
    assert error.__suppress_context__ is True
    assert "consumer supplied no handler" in error.__notes__


def test_error_ordinary_fields_stay_immutable():
    error = AutodiffError("unsupported_operator", "no rule")

    with pytest.raises(dataclasses.FrozenInstanceError):
        error.category = "missing_dependency"
    with pytest.raises(dataclasses.FrozenInstanceError):
        error.message = "something else"
    with pytest.raises(dataclasses.FrozenInstanceError):
        error.unrelated_attribute = "anything"

    assert error.category == "unsupported_operator"
    assert error.message == "no rule"


def test_error_keeps_equality_hashing_and_repr():
    error = AutodiffError("unsupported_operator", "no rule")
    same = AutodiffError("unsupported_operator", "no rule")
    other = AutodiffError("unsupported_operator", "a different message")

    assert error == same
    assert error != other
    assert hash(error) == hash(same)
    assert len({error, same, other}) == 2
    assert repr(error) == (
        "AutodiffError(category='unsupported_operator', message='no rule')"
    )


def test_error_still_rejects_an_unknown_category_after_a_traceback_is_set():
    """The category guard must not be weakened by making the type assignable."""
    with pytest.raises(ValueError, match="unknown autodiff error category"):
        AutodiffError("not_real", "bad")
