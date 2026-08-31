"""End-to-end proof of the compiled training step against the shared reference consumer.

This module is the §17.5 proof: the loss of §17.3.1 (`d = x @ w - y`, `d * d`,
`mean(...)`) is compiled by `compile_training_step`, its three lowered programs
are executed over concrete dense arrays through the one shared execution
semantics, and the updated parameter is compared against a directly computed
`w - learning_rate * dL/dw`.

`compile_training_step` is reached through `tinychain.autodiff.training_step`
rather than through the package: the export surface is owned by a different
subtask, and this module must not depend on it.

**Two phases, and why there are two.** Lowering is eager -- a handler produces
its target value the moment the framework calls it -- so `compile_training_step`
binds every free dependency of all three programs through the single
`bind_input` it was given, in one pass, before any caller can look at the
forward program. A harness that bound the derivative's forward captures inside
that pass could only get them from somewhere other than the forward program's
outputs, which is exactly what §17.3.5 forbids. So the proof is staged:

1. **Compile** (`compile_step`) with a shape-driven placeholder binding. This
   phase proves *structure*: that every reachable operation of every program was
   lowered exactly once and that every operation carries concrete operator
   instances. Nothing numerical is ever read from this phase -- its arrays are
   ones, and no assertion in this file compares them to anything.
2. **Execute** (`execute_step`) the record's own lowered artifacts through the
   public `lower_graph` / `lower_derivative_program` entry points, in the order
   a real backend would run them: forward first, then the derivative, then one
   update per parameter. Only this phase produces the numbers under test.

**The capture rule of §17.3.5 is enforced structurally, not by discipline.**
`run_forward` returns *only* `dict(zip(program.selected_outputs,
program.output_values))`; it never returns the `LoweredProgram`, so no later
step in this file holds an object with a `.values` mapping to read a capture
from. `LoweredProgram.output_values` is restricted to the selected outputs by
construction, which makes reading an internal forward binding for a capture
impossible here rather than merely avoided. The derivative binder then
dispatches on the framework's own dependency *provenance*: a `forward_capture`
can only ever be resolved out of that selected-output mapping.

**Why the bias in the two-parameter case is not broadcast.** A bias of a
shape narrower than the residual makes the gradient path emit
`BroadcastReduceOperator`, which is outside the nine operator types
`training_step_registry` was measured to need, and widening that shared
registry is not this subtask's to do. The bias therefore has the residual's
own shape -- still different from the weight's, which is all the routing guard
requires: a swapped pairing is detectable.

No production file is modified by this module.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pytest
import tinychain as tc
from tinychain.autodiff import (
    TensorOperator,
    lower_derivative_program,
    lower_graph,
)
from tinychain.autodiff import training_step
from tinychain.autodiff.training import SGD

from tests.autodiff_reference_consumer import (
    recording_registry,
    training_step_registry,
)


# --------------------------------------------------------------------------
# tolerance
#
# Every value in this file is `f64` and every program is a handful of dense
# operations deep, so the compiled result and the directly computed reference
# differ only by the order the same additions and multiplications are
# performed in. 1e-12 is far above that and far below any wiring mistake --
# a swapped operand or a missing scale factor moves a value by whole units,
# not by parts in 1e12.
# --------------------------------------------------------------------------

TOLERANCE = 1e-12


# --------------------------------------------------------------------------
# declarations
#
# `x` is 3x2 and `w` is 2x4: an asymmetric shape, so a transposed matmul
# anywhere on the path gives a shape error or a detectably wrong answer rather
# than a plausible one. The batch dimension is 3, greater than one.
# --------------------------------------------------------------------------

SCALAR_SPEC: Mapping[str, object] = {"dtype": "f64", "shape": []}

ONE_PARAMETER_INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": {"dtype": "f64", "shape": (3, 2)},
    "y": {"dtype": "f64", "shape": (3, 4)},
    "w": {"dtype": "f64", "shape": (2, 4)},
}

TWO_PARAMETER_INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": {"dtype": "f64", "shape": (3, 2)},
    "y": {"dtype": "f64", "shape": (3, 4)},
    "w": {"dtype": "f64", "shape": (2, 4)},
    "b": {"dtype": "f64", "shape": (3, 4)},
}

# Element count of the residual, the divisor the mean's derivative carries.
RESIDUAL_SIZE = 3 * 4


def residual_loss(*, x: object, y: object, w: object) -> object:
    """The §17.3.1 loss: the multiply's VJP reads its own operand."""
    residual = x @ w - y
    return (residual * residual).mean([0, 1])


def biased_residual_loss(*, x: object, y: object, w: object, b: object) -> object:
    """The same loss with an additive per-example bias as a second parameter."""
    residual = x @ w + b - y
    return (residual * residual).mean([0, 1])


# --------------------------------------------------------------------------
# concrete arrays
# --------------------------------------------------------------------------


def concrete_inputs(*, with_bias: bool) -> dict[str, np.ndarray]:
    """Fixed, non-degenerate `f64` arrays for one run."""
    generator = np.random.default_rng(20260831)
    values = {
        "x": generator.normal(size=(3, 2)),
        "y": generator.normal(size=(3, 4)),
        "w": generator.normal(size=(2, 4)),
    }
    if with_bias:
        values["b"] = generator.normal(size=(3, 4))
    return values


def reference_gradients(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`dL/dw` and, when a bias is declared, `dL/db`, computed directly.

    Written straight from the calculus of the loss, with no reference to any
    compiled artifact: `L = mean(d * d)` for `d = x @ w (+ b) - y`, so
    `dL/dd = 2 * d / N`, `dL/dw = x.T @ dL/dd`, and `dL/db = dL/dd`.
    """
    residual = values["x"] @ values["w"] - values["y"]
    if "b" in values:
        residual = residual + values["b"]
    residual_gradient = 2.0 * residual / RESIDUAL_SIZE
    gradients = {"w": values["x"].T @ residual_gradient}
    if "b" in values:
        gradients["b"] = residual_gradient
    return gradients


def reference_loss(values: Mapping[str, np.ndarray]) -> float:
    residual = values["x"] @ values["w"] - values["y"]
    if "b" in values:
        residual = residual + values["b"]
    return float(np.mean(residual * residual))


# --------------------------------------------------------------------------
# phase 1 -- the structural compile
# --------------------------------------------------------------------------


def placeholder_binding(dependency: object) -> np.ndarray:
    """A ones array of the dependency's own declared shape.

    Used only by the compile phase, whose numbers no assertion in this file
    reads. Driving it off the framework's analyzed shape rather than off a
    hand-written table keeps the compile working for whichever free
    dependencies a program actually has.
    """
    shape = tuple(int(dimension) for dimension in (dependency.shape or ()))
    return np.ones(shape, dtype=np.float64)


def compile_step(
    loss: object,
    *,
    inputs: Mapping[str, Mapping[str, object]],
    parameters: tuple[str, ...],
    handlers: object,
    fusion: object = None,
) -> object:
    """Compile one training step with the shared reference registry."""
    with tc.state.scoped_context():
        return training_step.compile_training_step(
            loss,
            inputs=inputs,
            parameters=parameters,
            optimizer=SGD(),
            optimizer_inputs={"learning_rate": SCALAR_SPEC},
            handlers=handlers,
            fusion=fusion,
            bind_input=placeholder_binding,
        )


def compile_one_parameter(**overrides: object) -> object:
    kwargs: dict[str, object] = {
        "inputs": ONE_PARAMETER_INPUTS,
        "parameters": ("w",),
        "handlers": training_step_registry(),
    }
    kwargs.update(overrides)
    return compile_step(residual_loss, **kwargs)


def compile_two_parameters(**overrides: object) -> object:
    kwargs: dict[str, object] = {
        "inputs": TWO_PARAMETER_INPUTS,
        "parameters": ("w", "b"),
        "handlers": training_step_registry(),
    }
    kwargs.update(overrides)
    return compile_step(biased_residual_loss, **kwargs)


def lowered_programs(record: object) -> tuple[object, ...]:
    """The three lowerings of a record, in the order the compiler performs them."""
    return (record.forward, record.derivative) + tuple(
        compiled.update for compiled in record.parameters
    )


def lowered_node_id_sequence(record: object) -> list[str]:
    """Every source node id the three lowerings claimed, in lowering order."""
    return [
        node_id
        for program in lowered_programs(record)
        for operation in program.operations
        for node_id in operation.source_node_ids
    ]


# --------------------------------------------------------------------------
# phase 2 -- execution over the record's own lowered artifacts
# --------------------------------------------------------------------------


def run_forward(
    record: object, values: Mapping[str, np.ndarray]
) -> Mapping[str, np.ndarray]:
    """Execute the lowered forward graph; return **only** its selected outputs.

    The `LoweredProgram` itself is deliberately not returned. Everything the
    rest of the harness can see about the forward run is this mapping, whose
    keys are exactly `selected_outputs` -- so a later step cannot reach an
    internal forward binding for a capture even by mistake (§17.3.5).
    """
    bindings = {
        record.input_value_ids[name]: array for name, array in values.items()
    }
    program = lower_graph(
        record.lowered_forward_graph,
        handlers=training_step_registry(),
        outputs=list(record.forward.selected_outputs),
        bind_input=lambda dependency: bindings[dependency.value_id],
    )
    return MappingProxyType(
        dict(zip(program.selected_outputs, program.output_values, strict=True))
    )


def run_derivative(
    record: object,
    *,
    values: Mapping[str, np.ndarray],
    forward_outputs: Mapping[str, np.ndarray],
    seed: float = 1.0,
) -> Mapping[str, np.ndarray]:
    """Execute the lowered derivative program; return one gradient per parameter.

    The binder dispatches on the framework's own provenance. A
    `forward_capture` is resolved out of `capture_bindings`, which is built
    exclusively from the forward program's selected outputs; there is no other
    source it could come from.
    """
    capture_bindings = {
        value_id: forward_outputs[value_id]
        for value_id in record.forward_capture_value_ids
    }
    declared_bindings = {
        record.input_value_ids[name]: array for name, array in values.items()
    }
    seed_bindings = {
        value_id: np.asarray(seed, dtype=np.float64)
        for value_id in record.seed_value_ids
    }

    def bind(dependency: object) -> np.ndarray:
        if dependency.provenance == "forward_capture":
            return capture_bindings[dependency.value_id]
        if dependency.provenance == "seed_input":
            return seed_bindings[dependency.value_id]
        if dependency.provenance == "declared_input":
            return declared_bindings[dependency.value_id]
        raise AssertionError(
            f"unexpected derivative dependency provenance {dependency.provenance!r}"
        )

    program = lower_derivative_program(
        record.lowered_derivative_program,
        forward_graph=record.lowered_forward_graph,
        seed_value_ids=list(record.seed_value_ids),
        handlers=training_step_registry(),
        outputs=list(record.derivative.selected_outputs),
        bind_input=bind,
    )
    gradients = dict(
        zip(program.selected_outputs, program.output_values, strict=True)
    )
    # Keyed by each parameter's own gradient value id, never by position.
    return MappingProxyType(
        {
            compiled.name: np.asarray(gradients[compiled.gradient_value_id])
            for compiled in record.parameters
        }
    )


def run_update(
    compiled: object,
    *,
    parameter: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    """Execute one parameter's lowered update graph and return its new value."""
    names = compiled.update_input_value_ids
    bindings = {
        names["parameter"]: np.asarray(parameter, dtype=np.float64),
        names["gradient"]: np.asarray(gradient, dtype=np.float64),
        names["learning_rate"]: np.asarray(learning_rate, dtype=np.float64),
    }
    program = lower_graph(
        compiled.lowered_update_graph,
        handlers=training_step_registry(),
        outputs=list(compiled.update.selected_outputs),
        bind_input=lambda dependency: bindings[dependency.value_id],
    )
    outputs = dict(zip(program.selected_outputs, program.output_values, strict=True))
    return np.asarray(outputs[compiled.updated_parameter_value_id])


@dataclasses.dataclass(frozen=True)
class ExecutedStep:
    """Everything one executed training step produced."""

    loss: float
    gradients: Mapping[str, np.ndarray]
    updated: Mapping[str, np.ndarray]


def execute_step(
    record: object,
    values: Mapping[str, np.ndarray],
    *,
    learning_rate: float,
) -> ExecutedStep:
    """Run forward, derivative, and every update, in that order."""
    forward_outputs = run_forward(record, values)
    gradients = run_derivative(
        record, values=values, forward_outputs=forward_outputs
    )
    updated = {
        compiled.name: run_update(
            compiled,
            parameter=values[compiled.name],
            gradient=gradients[compiled.name],
            learning_rate=learning_rate,
        )
        for compiled in record.parameters
    }
    return ExecutedStep(
        loss=float(np.asarray(forward_outputs[record.loss_value_id])),
        gradients=gradients,
        updated=MappingProxyType(updated),
    )


# --------------------------------------------------------------------------
# AC: each of the three lowered programs lowered every reachable operation
# exactly once, recorded through `recording_registry`
# --------------------------------------------------------------------------


def test_each_program_lowers_every_reachable_operation_exactly_once():
    # The reachable operations of a program are derived from the framework's
    # own dependency analysis -- every `local_value` is a value some operation
    # in the region produced -- not from a hand-copied list, so the equality
    # cannot be satisfied by a program that lowered a different region.
    record = compile_one_parameter()

    for program in lowered_programs(record):
        produced = [operation.output_value_id for operation in program.operations]
        assert len(produced) == len(set(produced))
        assert set(produced) == {
            dependency.value_id for dependency in program.dependencies.local_values
        }


def test_recording_registry_sees_every_operation_of_all_three_lowerings_once():
    recording = recording_registry(training_step_registry())

    record = compile_one_parameter(handlers=recording.registry)

    recorded = [invocation.node_id for invocation in recording.invocations]
    assert recorded == lowered_node_id_sequence(record)
    # Handler calls are per lowering, so the same node id may legitimately
    # appear in two different programs; within one program it may not.
    for program in lowered_programs(record):
        claimed = [
            node_id
            for operation in program.operations
            for node_id in operation.source_node_ids
        ]
        assert len(claimed) == len(set(claimed))


def test_recording_registry_sees_every_operation_once_with_two_parameters():
    recording = recording_registry(training_step_registry())

    record = compile_two_parameters(handlers=recording.registry)

    recorded = [invocation.node_id for invocation in recording.invocations]
    assert recorded == lowered_node_id_sequence(record)
    assert len(record.parameters) == 2


def test_every_lowered_operation_carries_concrete_operator_instances():
    record = compile_two_parameters()

    for program in lowered_programs(record):
        assert program.operations
        for operation in program.operations:
            assert operation.source_operators
            for operator in operation.source_operators:
                assert isinstance(operator, TensorOperator)
                # An operator *class* is also `isinstance`-compatible with
                # nothing here, but a bare type would pass a looser check, so
                # rule it out explicitly.
                assert not isinstance(operator, type)
                assert type(operator) is not TensorOperator


# --------------------------------------------------------------------------
# AC: the capture set is non-empty and every capture is a forward output
# --------------------------------------------------------------------------


def test_the_canonical_loss_has_a_non_empty_capture_set():
    record = compile_one_parameter()

    assert record.forward_capture_value_ids
    assert record.forward.selected_outputs == (
        record.loss_value_id,
    ) + record.forward_capture_value_ids


def test_every_capture_is_reachable_from_the_forward_programs_selected_outputs():
    record = compile_one_parameter()
    values = concrete_inputs(with_bias=False)

    forward_outputs = run_forward(record, values)

    assert set(record.forward_capture_value_ids) <= set(forward_outputs)
    for value_id in record.forward_capture_value_ids:
        assert forward_outputs[value_id] is not None


# --------------------------------------------------------------------------
# AC: the updated parameter matches `w - learning_rate * dL/dw`
# --------------------------------------------------------------------------


def test_one_parameter_updated_weight_matches_the_direct_sgd_step():
    record = compile_one_parameter()
    values = concrete_inputs(with_bias=False)
    learning_rate = 0.1

    executed = execute_step(record, values, learning_rate=learning_rate)

    expected = values["w"] - learning_rate * reference_gradients(values)["w"]
    np.testing.assert_allclose(
        executed.updated["w"], expected, rtol=TOLERANCE, atol=TOLERANCE
    )


def test_one_parameter_loss_and_gradient_match_the_direct_computation():
    record = compile_one_parameter()
    values = concrete_inputs(with_bias=False)

    executed = execute_step(record, values, learning_rate=0.1)

    assert executed.loss == pytest.approx(reference_loss(values), abs=TOLERANCE)
    np.testing.assert_allclose(
        executed.gradients["w"],
        reference_gradients(values)["w"],
        rtol=TOLERANCE,
        atol=TOLERANCE,
    )


def test_a_zero_learning_rate_leaves_the_parameter_unchanged():
    # Edge case: the update is executed for real, and returns the parameter.
    record = compile_one_parameter()
    values = concrete_inputs(with_bias=False)

    executed = execute_step(record, values, learning_rate=0.0)

    np.testing.assert_allclose(
        executed.updated["w"], values["w"], rtol=TOLERANCE, atol=TOLERANCE
    )


# --------------------------------------------------------------------------
# AC: two parameters, each gradient landing on its own parameter
# --------------------------------------------------------------------------


def test_two_parameter_gradients_land_on_their_own_parameters():
    record = compile_two_parameters()
    values = concrete_inputs(with_bias=True)
    learning_rate = 0.05

    executed = execute_step(record, values, learning_rate=learning_rate)

    expected_gradients = reference_gradients(values)
    for name in ("w", "b"):
        assert executed.gradients[name].shape == values[name].shape
        np.testing.assert_allclose(
            executed.gradients[name],
            expected_gradients[name],
            rtol=TOLERANCE,
            atol=TOLERANCE,
        )
        np.testing.assert_allclose(
            executed.updated[name],
            values[name] - learning_rate * expected_gradients[name],
            rtol=TOLERANCE,
            atol=TOLERANCE,
        )


def test_the_two_parameter_pairing_assertion_has_teeth():
    record = compile_two_parameters()
    values = concrete_inputs(with_bias=True)
    learning_rate = 0.05
    executed = execute_step(record, values, learning_rate=learning_rate)
    expected_gradients = reference_gradients(values)

    # A swapped comparison must fail: if it did not, the pairing assertion
    # above would hold for a compiler that routed the gradients backwards.
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            executed.gradients["w"],
            expected_gradients["b"],
            rtol=TOLERANCE,
            atol=TOLERANCE,
        )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            executed.gradients["b"],
            expected_gradients["w"],
            rtol=TOLERANCE,
            atol=TOLERANCE,
        )

    # And feeding a parameter's update the *other* parameter's gradient does
    # not quietly produce a plausible number.
    with pytest.raises(Exception):
        run_update(
            record.parameter("w"),
            parameter=values["w"],
            gradient=executed.gradients["b"],
            learning_rate=learning_rate,
        )


def test_the_two_parameters_have_different_shapes():
    # The premise the routing guard rests on: a swap is observable.
    record = compile_two_parameters()

    assert tuple(TWO_PARAMETER_INPUTS["w"]["shape"]) != tuple(
        TWO_PARAMETER_INPUTS["b"]["shape"]
    )
    assert tuple(compiled.name for compiled in record.parameters) == ("w", "b")


# --------------------------------------------------------------------------
# AC: a fusion hook is offered operations from all three lowerings, and the
# record is otherwise unchanged in structure
# --------------------------------------------------------------------------


class DecliningFusionHook:
    """A hook that claims nothing and records every operation it was offered."""

    lookahead = 3

    def __init__(self) -> None:
        self.offered: list[str] = []

    def fuse(self, context: object) -> None:
        self.offered.append(context.candidates[0].node_id)
        return None


def structural_digest(record: object) -> tuple[object, ...]:
    """Every framework-owned structural fact about a record, as plain data.

    Field presence, ordering, and the selection tuples -- deliberately not the
    consumer target values, which are the consumer's and are compared nowhere.
    """
    return (
        record.provenance,
        record.loss_value_id,
        record.forward_capture_value_ids,
        record.seed_value_ids,
        dict(record.input_value_ids),
        record.forward.selected_outputs,
        record.derivative.selected_outputs,
        tuple(
            (
                compiled.name,
                compiled.value_id,
                compiled.gradient_value_id,
                dict(compiled.update_input_value_ids),
                compiled.updated_parameter_value_id,
                compiled.update.selected_outputs,
            )
            for compiled in record.parameters
        ),
        tuple(
            tuple(
                (
                    operation.output_value_id,
                    operation.source_node_ids,
                    tuple(
                        type(operator).__name__
                        for operator in operation.source_operators
                    ),
                    operation.is_fused,
                )
                for operation in program.operations
            )
            for program in lowered_programs(record)
        ),
    )


def test_a_fusion_hook_is_offered_operations_from_all_three_lowerings():
    hook = DecliningFusionHook()

    record = compile_one_parameter(fusion=hook)

    # The compiler lowers forward, then derivative, then each update; a hook
    # that claims nothing is offered every operation of every lowering, once,
    # in that order.
    assert hook.offered == lowered_node_id_sequence(record)
    forward_ids = [
        node_id
        for operation in record.forward.operations
        for node_id in operation.source_node_ids
    ]
    derivative_ids = [
        node_id
        for operation in record.derivative.operations
        for node_id in operation.source_node_ids
    ]
    update_ids = [
        node_id
        for operation in record.parameters[0].update.operations
        for node_id in operation.source_node_ids
    ]
    assert forward_ids and derivative_ids and update_ids
    assert len(hook.offered) == len(forward_ids) + len(derivative_ids) + len(
        update_ids
    )


def test_a_declining_fusion_hook_leaves_the_record_structurally_unchanged():
    without_fusion = compile_one_parameter()
    with_fusion = compile_one_parameter(fusion=DecliningFusionHook())

    assert structural_digest(with_fusion) == structural_digest(without_fusion)
    for program in lowered_programs(with_fusion):
        assert not any(operation.is_fused for operation in program.operations)


def test_the_fusion_record_still_executes_to_the_same_numbers():
    record = compile_one_parameter(fusion=DecliningFusionHook())
    values = concrete_inputs(with_bias=False)
    learning_rate = 0.1

    executed = execute_step(record, values, learning_rate=learning_rate)

    expected = values["w"] - learning_rate * reference_gradients(values)["w"]
    np.testing.assert_allclose(
        executed.updated["w"], expected, rtol=TOLERANCE, atol=TOLERANCE
    )
