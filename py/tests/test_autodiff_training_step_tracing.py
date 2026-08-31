"""Unit tests for the single typed trace and source-artifact capture stage.

Pins the contract that a training-step loss is traced exactly once, inside
exactly one `TensorGraphBuilder`, with one typed `Tensor` per declared input
in caller order; that a return value other than a single `Tensor` raises
`invalid_loss_output` naming what it returned; that the builder's typed
`build(outputs=[...])` path performs finalization (never reimplemented here);
and that the resulting source forward graph, loss value id, and
`input_value_ids` are captured faithfully. By the time this stage runs, T-01's
`validate_declaration` has already validated the declaration set, every typed
input spec, the optimizer contract, and the loss signature -- so these tests
never exercise a declaration mistake, only the tracing stage itself.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import tinychain as tc
from tinychain.autodiff.protocol import AutodiffError
from tinychain.autodiff.training_step import TracedLoss, trace_loss


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

X_SPEC = {"dtype": "f32", "shape": (2, 3)}
W_SPEC = {"dtype": "f32", "shape": (3, 4)}
Y_SPEC = {"dtype": "f32", "shape": (2, 4)}

LINEAR_INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": X_SPEC,
    "w": W_SPEC,
    "y": Y_SPEC,
}
LINEAR_INPUT_NAMES = ("x", "w", "y")


def _linear_loss(*, x: object, w: object, y: object) -> object:
    d = x @ w - y
    return (d * d).mean([0, 1])


class _RecordingLoss:
    """A loss callable that records how many times, and how, it was invoked."""

    def __init__(self, compute) -> None:
        self.calls = 0
        self.received_args: tuple[object, ...] | None = None
        self.received_kwargs: dict[str, object] | None = None
        self._compute = compute

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        self.received_args = args
        self.received_kwargs = kwargs
        return self._compute(**kwargs)


def _symbolic_tensor(name: str) -> tc.Tensor:
    return tc.Tensor(tc.state.TCRef(tc.state.IdRef(name)))


# --------------------------------------------------------------------------
# AC: the source forward graph's inputs are the declared names in
# declaration order, and graph.outputs is exactly the one loss value id.
# --------------------------------------------------------------------------


def test_trace_loss_records_declared_inputs_in_declaration_order() -> None:
    with tc.state.scoped_context():
        traced = trace_loss(
            inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=_linear_loss
        )

    assert isinstance(traced, TracedLoss)
    declared_value_ids = [
        traced.input_value_ids[name] for name in LINEAR_INPUT_NAMES
    ]
    graph_input_ids = [value_id for value_id, _typespec in traced.graph.inputs]
    assert graph_input_ids == declared_value_ids


def test_trace_loss_graph_outputs_is_exactly_the_one_loss_value_id() -> None:
    with tc.state.scoped_context():
        traced = trace_loss(
            inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=_linear_loss
        )

    assert traced.graph.outputs == [traced.loss_value_id]


def test_trace_loss_node_set_matches_traced_expression() -> None:
    """`d = x @ w - y`, `d * d`, `mean(...)` traces matmul, sub, mul, mean."""
    with tc.state.scoped_context():
        traced = trace_loss(
            inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=_linear_loss
        )

    route_names = [node.operator.route_name for node in traced.graph.nodes]
    assert route_names == ["matmul", "sub", "mul", "mean"]


# --------------------------------------------------------------------------
# AC: the loss is invoked exactly once, asserted by a counter, and is
# invoked by keyword with one Tensor per declared input and no positional
# arguments.
# --------------------------------------------------------------------------


def test_trace_loss_invokes_loss_exactly_once() -> None:
    recorder = _RecordingLoss(_linear_loss)

    with tc.state.scoped_context():
        trace_loss(inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=recorder)

    assert recorder.calls == 1


def test_trace_loss_invokes_loss_by_keyword_with_no_positional_arguments() -> None:
    recorder = _RecordingLoss(_linear_loss)

    with tc.state.scoped_context():
        trace_loss(inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=recorder)

    assert recorder.received_args == ()
    assert set(recorder.received_kwargs) == set(LINEAR_INPUT_NAMES)


def test_trace_loss_invokes_loss_with_one_tensor_per_declared_input() -> None:
    recorder = _RecordingLoss(_linear_loss)

    with tc.state.scoped_context():
        trace_loss(inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=recorder)

    for name in LINEAR_INPUT_NAMES:
        assert isinstance(recorder.received_kwargs[name], tc.Tensor)


# --------------------------------------------------------------------------
# AC: a loss returning None, a tuple of two Tensors, a list, or a plain
# float raises `invalid_loss_output` naming what it returned.
# --------------------------------------------------------------------------


def _returns_none(*, x: object, w: object, y: object) -> object:
    x @ w - y
    return None


def _returns_two_tensors(*, x: object, w: object, y: object) -> object:
    d = x @ w - y
    return (d, d)


def _returns_a_list(*, x: object, w: object, y: object) -> object:
    d = x @ w - y
    return [d]


def _returns_a_float(*, x: object, w: object, y: object) -> object:
    x @ w - y
    return 3.5


def _returns_a_non_tensor_object(*, x: object, w: object, y: object) -> object:
    x @ w - y
    return object()


@pytest.mark.parametrize(
    ("loss", "expected_type_name"),
    [
        (_returns_none, "NoneType"),
        (_returns_two_tensors, "tuple"),
        (_returns_a_list, "list"),
        (_returns_a_float, "float"),
        (_returns_a_non_tensor_object, "object"),
    ],
    ids=["none", "two_tensor_tuple", "list", "float", "non_tensor_object"],
)
def test_trace_loss_rejects_non_tensor_return_naming_what_it_returned(
    loss, expected_type_name: str
) -> None:
    with tc.state.scoped_context():
        with pytest.raises(AutodiffError) as error:
            trace_loss(inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=loss)

    assert error.value.category == "invalid_loss_output"
    assert expected_type_name in error.value.message


# --------------------------------------------------------------------------
# AC: `input_value_ids` covers every declared name exactly once and each
# value maps to that input's value id in the graph.
# --------------------------------------------------------------------------


def test_trace_loss_input_value_ids_covers_every_declared_name_exactly_once() -> None:
    with tc.state.scoped_context():
        traced = trace_loss(
            inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=_linear_loss
        )

    assert set(traced.input_value_ids) == set(LINEAR_INPUT_NAMES)
    assert len(traced.input_value_ids) == len(LINEAR_INPUT_NAMES)


def test_trace_loss_input_value_ids_match_the_graph_input_value_ids() -> None:
    with tc.state.scoped_context():
        traced = trace_loss(
            inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=_linear_loss
        )

    graph_input_ids = {value_id for value_id, _typespec in traced.graph.inputs}
    for name in LINEAR_INPUT_NAMES:
        assert traced.input_value_ids[name] in graph_input_ids


# --------------------------------------------------------------------------
# AC: a reachable value with incomplete dtype or shape metadata is rejected
# by typed finalization with that path's own category, not a new one.
#
# Only the dtype-missing branch is exercised here. `missing_shape_metadata`
# is not independently reachable from this stage's boundary: every declared
# input arrives with a complete typed spec (T-01 already validated it), and
# the recorder (`tracing.record_operation`) sets output metadata atomically
# -- `dtype` and `shape` together via `_boundary_typespec`, or neither at
# all when an operand is untyped (see `tinychain/autodiff/tracing.py`). A
# free-standing untraced Tensor therefore always presents as *no* metadata
# to `finalize_typed_graph._require_complete_typespec`, which checks dtype
# first -- so it is `missing_dtype_metadata`, never a dtype-present/
# shape-absent split, without fabricating internal builder state that this
# subtask's off-limits list places out of reach (`TensorGraph`,
# `TensorNodeRecord`, `TensorGraphBuilder` internals).
# --------------------------------------------------------------------------


def test_trace_loss_rejects_reachable_untyped_value_with_finalizations_own_category() -> None:
    """A loss that mixes a declared input with a free-standing untyped Tensor.

    The free-standing Tensor was never declared, so it reaches typed
    finalization -- the builder's `build(outputs=[...])` path -- as a
    reachable input with no recorded metadata. Rejecting it is
    finalization's own `missing_dtype_metadata`, not a category this stage
    invents.
    """

    def loss_with_untyped_operand(*, x: object, w: object, y: object) -> object:
        untyped = _symbolic_tensor("untyped")
        d = (x @ w - y) + untyped
        return (d * d).mean([0, 1])

    with tc.state.scoped_context():
        with pytest.raises(AutodiffError) as error:
            trace_loss(
                inputs=LINEAR_INPUTS,
                input_names=LINEAR_INPUT_NAMES,
                loss=loss_with_untyped_operand,
            )

    assert error.value.category == "missing_dtype_metadata"


# --------------------------------------------------------------------------
# AC: an exception raised inside the loss body -- an AutodiffError and a
# custom exception -- propagates unchanged and uncategorized;
# KeyboardInterrupt propagates unchanged.
# --------------------------------------------------------------------------


class _CustomLossError(Exception):
    pass


def _raises_autodiff_error(*, x: object, w: object, y: object) -> object:
    x @ w - y
    raise AutodiffError("unsupported_operator", "deliberate failure from the loss body")


def _raises_custom_exception(*, x: object, w: object, y: object) -> object:
    x @ w - y
    raise _CustomLossError("deliberate failure from the loss body")


def _raises_keyboard_interrupt(*, x: object, w: object, y: object) -> object:
    x @ w - y
    raise KeyboardInterrupt()


def test_trace_loss_propagates_autodiff_error_from_loss_body_unchanged() -> None:
    with tc.state.scoped_context():
        with pytest.raises(AutodiffError) as error:
            trace_loss(
                inputs=LINEAR_INPUTS,
                input_names=LINEAR_INPUT_NAMES,
                loss=_raises_autodiff_error,
            )

    assert error.value.category == "unsupported_operator"
    assert "deliberate failure from the loss body" in error.value.message


def test_trace_loss_propagates_custom_exception_from_loss_body_unchanged() -> None:
    with tc.state.scoped_context():
        with pytest.raises(_CustomLossError, match="deliberate failure from the loss body"):
            trace_loss(
                inputs=LINEAR_INPUTS,
                input_names=LINEAR_INPUT_NAMES,
                loss=_raises_custom_exception,
            )


def test_trace_loss_propagates_keyboard_interrupt_from_loss_body_unchanged() -> None:
    with tc.state.scoped_context():
        with pytest.raises(KeyboardInterrupt):
            trace_loss(
                inputs=LINEAR_INPUTS,
                input_names=LINEAR_INPUT_NAMES,
                loss=_raises_keyboard_interrupt,
            )


# --------------------------------------------------------------------------
# AC: tracing the same declarations twice produces graphs with equal value
# ids and equal node order.
# --------------------------------------------------------------------------


def test_trace_loss_tracing_same_declarations_twice_produces_equal_graphs() -> None:
    with tc.state.scoped_context():
        first = trace_loss(
            inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=_linear_loss
        )
    with tc.state.scoped_context():
        second = trace_loss(
            inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=_linear_loss
        )

    assert first.graph.inputs == second.graph.inputs
    assert first.graph.outputs == second.graph.outputs
    assert [node.node_id for node in first.graph.nodes] == [
        node.node_id for node in second.graph.nodes
    ]
    assert [node.output_value_id for node in first.graph.nodes] == [
        node.output_value_id for node in second.graph.nodes
    ]
    assert first.loss_value_id == second.loss_value_id
    assert first.input_value_ids == second.input_value_ids


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


def test_trace_loss_with_a_single_declared_input() -> None:
    def square(*, x: object) -> object:
        return (x * x).mean([0, 1])

    with tc.state.scoped_context():
        traced = trace_loss(
            inputs={"x": X_SPEC}, input_names=("x",), loss=square
        )

    assert traced.graph.outputs == [traced.loss_value_id]
    assert set(traced.input_value_ids) == {"x"}


def test_trace_loss_with_a_declared_input_the_loss_ignores() -> None:
    def loss_ignoring_y(*, x: object, w: object, y: object) -> object:
        return (x @ w).mean([0, 1])

    with tc.state.scoped_context():
        traced = trace_loss(
            inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=loss_ignoring_y
        )

    # `y` is declared but not reachable from the selected output; it must
    # still be reported in `input_value_ids`, covering every declared name.
    assert set(traced.input_value_ids) == set(LINEAR_INPUT_NAMES)
    # `y`'s value id is not necessarily among the graph's reachable inputs,
    # since the builder's typed build path filters to what is reachable.
    reachable_input_ids = {value_id for value_id, _typespec in traced.graph.inputs}
    assert traced.input_value_ids["x"] in reachable_input_ids
    assert traced.input_value_ids["w"] in reachable_input_ids


def test_trace_loss_returning_a_declared_input_unchanged() -> None:
    def identity_loss(*, x: object, w: object, y: object) -> object:
        return x

    with tc.state.scoped_context():
        traced = trace_loss(
            inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=identity_loss
        )

    assert traced.loss_value_id == traced.input_value_ids["x"]
    assert traced.graph.outputs == [traced.input_value_ids["x"]]


def test_trace_loss_with_symbolic_shape_in_declared_input() -> None:
    """A declared input carrying a symbolic dimension traces normally.

    T-01 already accepts a symbol as a shape dimension (`missing_shape_metadata`
    is raised only for an absent/malformed shape, not a symbolic one), so this
    stage must not reject or otherwise choke on a symbol flowing through
    `builder.input(...)` and typed finalization.
    """

    def square(*, x: object) -> object:
        return (x * x).mean([0, 1])

    with tc.state.scoped_context():
        traced = trace_loss(
            inputs={"x": {"dtype": "f32", "shape": ("N", 3)}},
            input_names=("x",),
            loss=square,
        )

    assert traced.graph.outputs == [traced.loss_value_id]
    assert set(traced.input_value_ids) == {"x"}


def test_trace_loss_returning_an_intermediate_also_consumed_elsewhere() -> None:
    """The selected output is itself an intermediate with other consumers.

    `d` is not a leaf: it feeds both `mul` (`scaled = d * d`) and `add`
    (`also_uses_d = scaled + d`) before being returned unchanged. The single
    selected output must still resolve to `d`'s own value id, and `d` must
    still appear as an operand of the nodes that consume it -- reuse of one
    traced value by multiple downstream nodes must not corrupt value
    identity or the output selection.
    """

    def loss(*, x: object, w: object, y: object) -> object:
        d = x @ w - y
        scaled = d * d
        _also_uses_d = scaled + d
        return d

    with tc.state.scoped_context():
        traced = trace_loss(
            inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=loss
        )

    assert traced.graph.outputs == [traced.loss_value_id]
    d_node = next(
        node for node in traced.graph.nodes if node.operator.route_name == "sub"
    )
    assert traced.loss_value_id == d_node.output_value_id
    consumers = [
        node
        for node in traced.graph.nodes
        if traced.loss_value_id in node.input_value_ids
    ]
    assert len(consumers) == 2


def test_trace_loss_opening_a_nested_builder_raises_without_a_second_builder() -> None:
    """Inv-4: this stage never opens a second builder.

    A loss that itself opens a nested `TensorGraphBuilder` hits the
    builder's own single-active-context guard, which raises a bare
    `RuntimeError` -- this stage neither catches nor wraps it, and no
    second trace this stage owns ever succeeds.
    """
    from tinychain.autodiff import TensorGraphBuilder

    def loss_opening_nested_builder(*, x: object, w: object, y: object) -> object:
        with TensorGraphBuilder():
            pass
        return x @ w - y

    with tc.state.scoped_context():
        with pytest.raises(RuntimeError, match="Nested TensorGraphBuilder"):
            trace_loss(
                inputs=LINEAR_INPUTS,
                input_names=LINEAR_INPUT_NAMES,
                loss=loss_opening_nested_builder,
            )
