"""The §17.7 cross-spec composition test: both of issue #128's real passes, composed.

This module is normative in two specifications at once -- §17.7 here and
§17.6.4 of issue #128 -- and lands with whichever feature completes second.
It proves one thing, from six angles, over a single compiled record:

    supplying issue #128's **real** expansion passes to
    `compile_training_step` lets a training step lower end to end against a
    registry that resolves **no** reduction, broadcast, or division handler.

Why the control is the proof
----------------------------
"It lowered" says nothing on its own -- a registry might have happened to
cover everything the loss produced. The proof is the pair: the *same*
declarations against the *same* registry with **empty** expansion sequences
must fail with `unsupported_operator` naming the `MeanOperator`. The only
difference between the two compiles is the expansion sequences, so the
sequences are what enabled the lowering. `test_the_control_without_expansions_fails_unsupported`
is that control, and `test_the_step_lowers_end_to_end_without_reduction_handlers`
is the assertion it gives meaning to.

Which assertion has teeth if a compiler ignored the sequences
-------------------------------------------------------------
Every other fixture in this suite supplies empty expansion sequences, so
nothing in it could detect a compiler that accepted a well-formed sequence
and silently discarded it. Here, discarding the sequences makes the compile
*identical to the control* -- so
`test_the_step_lowers_end_to_end_without_reduction_handlers` fails with
`unsupported_operator` the moment the sequences stop being applied. That is
the assertion that closes the inert-expansion gap;
`test_the_lowered_artifacts_are_not_the_source_artifacts` and
`test_expansion_removed_the_reduction_broadcast_and_division_nodes` close it
structurally as well.

Why an identity assertion is not vacuous here
---------------------------------------------
`record.lowered_forward_graph is record.source_forward_graph` is *true* under
a no-passes fixture (Inv-10: with no passes the lowered artifacts **are** the
source artifacts), so asserting object identity would prove nothing there.
With real passes in play the artifacts genuinely differ, so this module
asserts they *differ* -- object identity and structure both.

Why the derivative pass is a `functools.partial`
------------------------------------------------
Both passes mint from one reserved namespace starting at zero, so composed at
their shared default each artifact contains `exv0` and the §8.6 preservation
recomputation correctly rejects the pair with `ambiguous_producer`. The passes
therefore accept an optional, keyword-only `reserved_id_prefix`; binding one
with `functools.partial` namespaces the gradient path apart. A `partial`
binding a namespace is still the real function -- it is not a wrapper that
changes behaviour -- and `test_the_supplied_passes_are_the_exported_real_ones`
asserts the underlying callable **is** the exported name, so no stand-in can
be substituted later and still satisfy §17.7.

Ownership
---------
Tests only. No production file is touched, neither #128's passes nor the
shared reference consumer's registry factories are modified, and the
reduction-free registry this module needs is composed here out of handlers
those factories already register -- no handler is defined in this file.
"""

from __future__ import annotations

import functools
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pytest

import tinychain as tc
import tinychain.autodiff as autodiff
from tinychain.autodiff import (
    AutodiffError,
    BroadcastOperator,
    DivOperator,
    FillOperator,
    MatmulOperator,
    MeanOperator,
    OperationHandlerRegistry,
    SumOperator,
    analyze_derivative_dependencies,
    compile_training_step,
    lower_derivative_program,
    lower_graph,
)
from tinychain.autodiff.training import SGD

from tests.autodiff_reference_consumer import (
    limited_operation_registry,
    training_step_registry,
)
from tests.autodiff_training_step_numeric_support import (
    ONE_PARAMETER_INPUTS,
    SCALAR_SPEC,
    TWO_PARAMETER_INPUTS,
    ExecutedStep,
    concrete_inputs as _concrete_inputs,
    placeholder_binding,
    reference_gradients,
    reference_loss,
)


# --------------------------------------------------------------------------
# tolerance
#
# The tolerance issue #128 states for the expanded/non-expanded value axis:
# 1e-6 relative for `f32`, 1e-12 relative for `f64`. Every value here is
# `f64`. Bit-for-bit equality is deliberately not asserted: the expansion
# substitutes `x * (1 / d)` for `x / d`, which IEEE-754 does not require to
# agree exactly. `atol` is pinned to zero so the comparison stays genuinely
# relative rather than being absorbed by numpy's 1e-8 default.
# --------------------------------------------------------------------------

RELATIVE_TOLERANCE = 1e-12
ABSOLUTE_TOLERANCE = 0.0


# --------------------------------------------------------------------------
# the two reserved namespaces
#
# The forward pass keeps the documented default (`exn…`/`exv…`); the
# gradient-path pass is bound to a second namespace so the two lowered
# artifacts cannot mint the same identifier.
# --------------------------------------------------------------------------

FORWARD_PREFIX = "ex"
DERIVATIVE_PREFIX = "exd"

FORWARD_PASS = autodiff.expand_mean_graph
DERIVATIVE_PASS = functools.partial(
    autodiff.expand_mean_derivative_program, reserved_id_prefix=DERIVATIVE_PREFIX
)


# --------------------------------------------------------------------------
# declarations
#
# The input specs are the shared ones imported above: `x` is 3x2 and `w` is
# 2x4, so the residual the mean reduces is 3x4 -- a rank-2, non-square
# operand, and an all-axis mean: exactly the reduction #128's forward pass
# supports. Every shape is asymmetric, so a transposed matmul anywhere on the
# path gives a shape error or a detectably wrong answer rather than a
# plausible one. The losses below stay local because their `keepdims` choice
# is the tier this file is testing.
# --------------------------------------------------------------------------

LEARNING_RATE = 0.1


def residual_loss(*, x: object, y: object, w: object) -> object:
    """The canonical loss, reduced by an all-axis rank-2 mean with `keepdims=True`.

    `keepdims=True` keeps the reduction in #128's rank-preserving tier, whose
    expanded region needs only fill, matmul, and multiply.
    """
    residual = x @ w - y
    return (residual * residual).mean([0, 1], keepdims=True)


def rank_reducing_residual_loss(*, x: object, y: object, w: object) -> object:
    """The same loss reduced to rank zero -- #128's rank-reducing tier."""
    residual = x @ w - y
    return (residual * residual).mean([0, 1], keepdims=False)


def biased_residual_loss(*, x: object, y: object, w: object, b: object) -> object:
    """The same loss with a second trained parameter, the expansion still in play."""
    residual = x @ w + b - y
    return (residual * residual).mean([0, 1], keepdims=True)


# --------------------------------------------------------------------------
# the registry with no reduction, broadcast, or division handler
#
# Composed here out of handlers the shared reference consumer already
# registers: none is defined in this file and neither shared factory is
# widened. `MeanOperator`, `BroadcastOperator`, and `DivOperator` are dropped
# -- that omission is the whole experiment -- and `FillOperator`, which only
# `limited_operation_registry` carries, is added because an expanded region
# emits one.
# --------------------------------------------------------------------------

WITHHELD_OPERATOR_TYPES = (MeanOperator, SumOperator, BroadcastOperator, DivOperator)


def expansion_only_registry() -> OperationHandlerRegistry:
    """A registry resolving what an *expanded* artifact needs and nothing more."""
    composed = OperationHandlerRegistry()
    seen: set[type] = set()
    for source in (training_step_registry(), limited_operation_registry()):
        for operator_type in source.supported_types():
            if operator_type in WITHHELD_OPERATOR_TYPES or operator_type in seen:
                continue
            seen.add(operator_type)
            composed.register(source.lookup(operator_type()))
    return composed


def compile_composed(
    loss: object = residual_loss,
    *,
    inputs: Mapping[str, Mapping[str, object]] = ONE_PARAMETER_INPUTS,
    parameters: tuple[str, ...] = ("w",),
    forward_expansions: object = (FORWARD_PASS,),
    derivative_expansions: object = (DERIVATIVE_PASS,),
) -> object:
    """Compile one training step against the reduction-free registry."""
    with tc.state.scoped_context():
        return compile_training_step(
            loss,
            inputs=inputs,
            parameters=parameters,
            optimizer=SGD(),
            optimizer_inputs={"learning_rate": SCALAR_SPEC},
            handlers=expansion_only_registry(),
            bind_input=placeholder_binding,
            forward_expansions=forward_expansions,
            derivative_expansions=derivative_expansions,
        )


# --------------------------------------------------------------------------
# the single compiled record the assertions share
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record() -> object:
    """The §17.7 subject: both real passes, one compile, no reduction handlers."""
    return compile_composed()


# --------------------------------------------------------------------------
# artifact inspection helpers
# --------------------------------------------------------------------------


def operator_type_names(artifact: object) -> set[str]:
    """Every concrete operator type name appearing in a graph or program."""
    return {type(node.operator).__name__ for node in artifact.nodes}


def artifact_identifiers(artifact: object) -> set[str]:
    """Every node id and produced value id of a graph or program."""
    return {node.node_id for node in artifact.nodes} | {
        node.output_value_id for node in artifact.nodes
    }


def minted_identifiers(record: object) -> set[str]:
    """Every identifier either expansion minted, across both lowered artifacts.

    An expansion's minted identifiers are exactly those spelled under one of
    the two reserved namespaces; nothing the tracer (`v…`/`n…`), the reverse
    transform (`d…`/`dn…`), or the seed minter produces can be spelled that
    way, which is what makes the namespaces reserved.
    """
    reserved = tuple(
        f"{prefix}{kind}"
        for prefix in (FORWARD_PREFIX, DERIVATIVE_PREFIX)
        for kind in ("n", "v")
    )
    lowered = artifact_identifiers(record.lowered_forward_graph) | artifact_identifiers(
        record.lowered_derivative_program
    )
    return {identifier for identifier in lowered if identifier.startswith(reserved)}


def recomputed_analysis(record: object) -> object:
    """The §8.6 recomputation, run over the record's own lowered artifacts."""
    return analyze_derivative_dependencies(
        record.lowered_derivative_program,
        forward_graph=record.lowered_forward_graph,
        seed_value_ids=list(record.seed_value_ids),
        outputs=list(record.derivative.selected_outputs),
    )


def source_analysis(record: object) -> object:
    """The §8.5 source analysis, run over the record's own source artifacts."""
    return analyze_derivative_dependencies(
        record.source_derivative_program,
        forward_graph=record.source_forward_graph,
        seed_value_ids=list(record.seed_value_ids),
        outputs=list(record.derivative.selected_outputs),
    )


# --------------------------------------------------------------------------
# execution over the record's own lowered artifacts
#
# Deliberately re-lowered through `expansion_only_registry()` rather than
# through the reduction-capable shared registry: executing an expanded
# artifact against a registry that could also have run the unexpanded one
# would weaken the numerical claim to "the numbers are right" instead of
# "the numbers are right and no reduction, broadcast, or division handler
# was available to produce them".
# --------------------------------------------------------------------------


RNG_SEED = 20260901


def concrete_inputs(*, with_bias: bool) -> dict[str, np.ndarray]:
    """This file's fixed, non-degenerate `f64` arrays for one run."""
    return _concrete_inputs(with_bias=with_bias, seed=RNG_SEED)


def execute_step(
    record: object,
    values: Mapping[str, np.ndarray],
    *,
    learning_rate: float = LEARNING_RATE,
) -> ExecutedStep:
    """Run the forward graph, the derivative program, and every update.

    Every lowering uses `expansion_only_registry()`, so nothing here could
    have been produced by a reduction, broadcast, or division handler.
    """
    handlers = expansion_only_registry()
    declared_bindings = {
        record.input_value_ids[name]: array for name, array in values.items()
    }

    forward = lower_graph(
        record.lowered_forward_graph,
        handlers=handlers,
        outputs=list(record.forward.selected_outputs),
        bind_input=lambda dependency: declared_bindings[dependency.value_id],
    )
    forward_outputs = MappingProxyType(
        dict(zip(forward.selected_outputs, forward.output_values, strict=True))
    )

    # A capture may only come from the forward program's selected outputs.
    capture_bindings = {
        value_id: forward_outputs[value_id]
        for value_id in record.forward_capture_value_ids
    }
    # dL/dL is ones of the loss's own analyzed shape -- rank 2 for a
    # `keepdims=True` mean, rank 0 for a `keepdims=False` one -- taken from
    # the framework's analysis rather than from a hand-written table.
    seed_shapes = {
        dependency.value_id: tuple(
            int(dimension) for dimension in (dependency.shape or ())
        )
        for dependency in recomputed_analysis(record).seed_inputs
    }

    def bind(dependency: object) -> np.ndarray:
        if dependency.provenance == "forward_capture":
            return capture_bindings[dependency.value_id]
        if dependency.provenance == "seed_input":
            return np.ones(seed_shapes[dependency.value_id], dtype=np.float64)
        if dependency.provenance == "declared_input":
            return declared_bindings[dependency.value_id]
        raise AssertionError(
            f"unexpected derivative dependency provenance {dependency.provenance!r}"
        )

    derivative = lower_derivative_program(
        record.lowered_derivative_program,
        forward_graph=record.lowered_forward_graph,
        seed_value_ids=list(record.seed_value_ids),
        handlers=handlers,
        outputs=list(record.derivative.selected_outputs),
        bind_input=bind,
    )
    raw_gradients = dict(
        zip(derivative.selected_outputs, derivative.output_values, strict=True)
    )
    # Keyed by each parameter's own gradient value id, never by position.
    gradients = {
        compiled.name: np.asarray(raw_gradients[compiled.gradient_value_id])
        for compiled in record.parameters
    }

    updated: dict[str, np.ndarray] = {}
    for compiled in record.parameters:
        names = compiled.update_input_value_ids
        update_bindings = {
            names["parameter"]: np.asarray(values[compiled.name], dtype=np.float64),
            names["gradient"]: np.asarray(gradients[compiled.name], dtype=np.float64),
            names["learning_rate"]: np.asarray(learning_rate, dtype=np.float64),
        }
        update = lower_graph(
            compiled.lowered_update_graph,
            handlers=handlers,
            outputs=list(compiled.update.selected_outputs),
            bind_input=lambda dependency: update_bindings[dependency.value_id],
        )
        outputs = dict(zip(update.selected_outputs, update.output_values, strict=True))
        updated[compiled.name] = np.asarray(outputs[compiled.updated_parameter_value_id])

    return ExecutedStep(
        loss=float(np.asarray(forward_outputs[record.loss_value_id]).reshape(())),
        gradients=MappingProxyType(gradients),
        updated=MappingProxyType(updated),
    )


# --------------------------------------------------------------------------
# AC: the passes supplied are #128's real ones, asserted by identity
# --------------------------------------------------------------------------


def test_the_supplied_passes_are_the_exported_real_ones() -> None:
    """No identity pass, no hand-written stand-in, no behaviour-changing wrapper.

    The forward entry is the exported function itself. The derivative entry is
    a `functools.partial` that binds nothing but a reserved namespace, so its
    `func` must be the exported function and its only bound argument must be
    the prefix -- a `partial` that also bound, replaced, or wrapped behaviour
    would fail the second and third assertions.
    """
    assert FORWARD_PASS is autodiff.expand_mean_graph

    assert DERIVATIVE_PASS.func is autodiff.expand_mean_derivative_program
    assert DERIVATIVE_PASS.args == ()
    assert DERIVATIVE_PASS.keywords == {"reserved_id_prefix": DERIVATIVE_PREFIX}


def test_the_composed_sequences_hold_exactly_those_passes(record: object) -> None:
    """The record was compiled with one forward pass and one derivative pass."""
    assert record.provenance.forward_expansions == ("expand_mean_graph",)
    # A `functools.partial` carries no `__qualname__`, so §9.1's label rule
    # falls back to its concrete type's name.
    assert record.provenance.derivative_expansions == ("partial",)
    assert record.provenance.update_expansions == ()


# --------------------------------------------------------------------------
# AC: the registry genuinely has no reduction, broadcast, or division handler
# --------------------------------------------------------------------------


def test_the_registry_resolves_no_reduction_broadcast_or_division() -> None:
    """The premise of the whole experiment, asserted rather than assumed."""
    supported = set(expansion_only_registry().supported_types())

    assert not supported & set(WITHHELD_OPERATOR_TYPES)
    # What an expanded region does need is present, or the experiment would
    # prove only that the registry was too small.
    assert {FillOperator, MatmulOperator} <= supported


def test_a_withheld_operator_lookup_fails_closed() -> None:
    """Looking a reduction up in this registry is `unsupported_operator`."""
    with pytest.raises(AutodiffError) as failure:
        expansion_only_registry().lookup(MeanOperator())

    assert failure.value.category == "unsupported_operator"


# --------------------------------------------------------------------------
# AC: the compiled step lowers end to end -- and the control that proves the
# expansion is what enabled it
# --------------------------------------------------------------------------


def test_the_step_lowers_end_to_end_without_reduction_handlers(record: object) -> None:
    """§17.7.1. This is the assertion with teeth against an inert compiler.

    A compiler that accepted the expansion sequences and silently discarded
    them would produce exactly the control compile below, which fails with
    `unsupported_operator`. Reaching this assertion at all therefore proves
    the sequences were applied, not merely accepted.
    """
    assert record.forward.operations
    assert record.derivative.operations
    assert record.parameters
    for compiled in record.parameters:
        assert compiled.update.operations

    # Every reachable operation of all three programs was lowered by a
    # handler this registry actually holds.
    lowered_operator_types = {
        type(operator).__name__
        for program in (record.forward, record.derivative)
        + tuple(compiled.update for compiled in record.parameters)
        for operation in program.operations
        for operator in operation.source_operators
    }
    assert lowered_operator_types
    assert not lowered_operator_types & {
        operator_type.__name__ for operator_type in WITHHELD_OPERATOR_TYPES
    }


def test_the_control_without_expansions_fails_unsupported() -> None:
    """§17.7.1's control: the same declarations, the same registry, no passes.

    Without this the success above would be consistent with a registry that
    simply covered everything the loss produced.
    """
    with pytest.raises(AutodiffError) as failure:
        compile_composed(forward_expansions=(), derivative_expansions=())

    assert failure.value.category == "unsupported_operator"
    assert MeanOperator.__name__ in str(failure.value)


def test_the_control_fails_for_the_forward_reduction_specifically() -> None:
    """A derivative-only sequence still cannot lower the source forward mean."""
    with pytest.raises(AutodiffError) as failure:
        compile_composed(forward_expansions=())

    assert failure.value.category == "unsupported_operator"


# --------------------------------------------------------------------------
# AC: expansion actually happened -- the lowered artifacts are not the source
# ones (an assertion that would be vacuous under a no-passes fixture)
# --------------------------------------------------------------------------


def test_the_lowered_artifacts_are_not_the_source_artifacts(record: object) -> None:
    """Inv-10 is about the *empty* case; with real passes the artifacts differ.

    Object identity and content are both asserted: an implementation that
    returned a fresh but unchanged copy would satisfy the first pair alone.
    """
    assert record.lowered_forward_graph is not record.source_forward_graph
    assert record.lowered_derivative_program is not record.source_derivative_program

    assert artifact_identifiers(record.lowered_forward_graph) != artifact_identifiers(
        record.source_forward_graph
    )
    assert artifact_identifiers(
        record.lowered_derivative_program
    ) != artifact_identifiers(record.source_derivative_program)


def test_expansion_removed_the_reduction_broadcast_and_division_nodes(
    record: object,
) -> None:
    """The source artifacts carry what the registry withholds; the lowered ones do not."""
    source_forward = operator_type_names(record.source_forward_graph)
    source_derivative = operator_type_names(record.source_derivative_program)
    lowered_forward = operator_type_names(record.lowered_forward_graph)
    lowered_derivative = operator_type_names(record.lowered_derivative_program)

    # The source really did need what the registry withholds.
    assert MeanOperator.__name__ in source_forward
    assert {BroadcastOperator.__name__, DivOperator.__name__} <= source_derivative

    withheld = {operator_type.__name__ for operator_type in WITHHELD_OPERATOR_TYPES}
    assert not lowered_forward & withheld
    assert not lowered_derivative & withheld

    # And the expanded regions are there in their place.
    assert FillOperator.__name__ in lowered_forward
    assert FillOperator.__name__ in lowered_derivative


# --------------------------------------------------------------------------
# AC: §8.6's preservation check passed, and the capture set is unchanged
# --------------------------------------------------------------------------


def test_the_recomputed_capture_and_seed_sets_equal_the_source_analysis(
    record: object,
) -> None:
    """§17.7.2. The §8.6 recomputation, re-run here over the record's artifacts.

    Compared dependency for dependency -- value id, provenance, dtype, and
    shape -- and in analysis order, not as an unordered set of identifiers.
    """
    source = source_analysis(record)
    recomputed = recomputed_analysis(record)

    assert recomputed.forward_captures == source.forward_captures
    assert recomputed.seed_inputs == source.seed_inputs


def test_the_capture_set_is_non_empty_and_matches_the_record(record: object) -> None:
    """A vacuously empty capture set would make the equality above meaningless."""
    recomputed = recomputed_analysis(record)
    recomputed_capture_ids = tuple(
        dependency.value_id for dependency in recomputed.forward_captures
    )

    assert record.forward_capture_value_ids
    assert recomputed_capture_ids == record.forward_capture_value_ids
    # Every capture survived expansion under its own identity: a capture is a
    # forward value the derivative reads, so a renamed one would break the
    # equality above rather than being silently re-bound.
    assert set(record.forward_capture_value_ids) <= artifact_identifiers(
        record.lowered_forward_graph
    )


def test_the_forward_selection_still_carries_the_loss_and_every_capture(
    record: object,
) -> None:
    """Inv-7 holds after expansion, not only before it."""
    assert record.forward.selected_outputs == (
        record.loss_value_id,
    ) + record.forward_capture_value_ids


# --------------------------------------------------------------------------
# AC: the minted seed is a required free input of the expanded derivative
# program and is distinct from every identifier the expansion minted
# --------------------------------------------------------------------------


def test_the_seed_is_a_required_free_input_of_the_expanded_program(
    record: object,
) -> None:
    """§17.7.3, first half. Inv-11: the seed survives expansion, same identity."""
    recomputed = recomputed_analysis(record)
    seed_input_ids = tuple(
        dependency.value_id for dependency in recomputed.seed_inputs
    )

    assert record.seed_value_ids
    assert seed_input_ids == record.seed_value_ids
    # Free, not produced: no node in the expanded program may produce it.
    produced = {node.output_value_id for node in record.lowered_derivative_program.nodes}
    assert not set(record.seed_value_ids) & produced


def test_the_seed_is_distinct_from_every_identifier_expansion_minted(
    record: object,
) -> None:
    """§17.7.3, second half -- and the test would be vacuous without minted ids."""
    minted = minted_identifiers(record)

    assert minted, "the expansion minted nothing, so distinctness proves nothing"
    assert not set(record.seed_value_ids) & minted


def test_the_two_passes_minted_into_disjoint_namespaces(record: object) -> None:
    """The reason the derivative pass carries a bound prefix at all.

    At one shared namespace both artifacts contain `exv0` and the §8.6
    recomputation rejects the pair as an ambiguous producer.
    """
    forward_minted = {
        identifier
        for identifier in artifact_identifiers(record.lowered_forward_graph)
        if identifier.startswith((f"{FORWARD_PREFIX}n", f"{FORWARD_PREFIX}v"))
    }
    derivative_minted = {
        identifier
        for identifier in artifact_identifiers(record.lowered_derivative_program)
        if identifier.startswith((f"{DERIVATIVE_PREFIX}n", f"{DERIVATIVE_PREFIX}v"))
    }

    assert forward_minted
    assert derivative_minted
    assert not forward_minted & derivative_minted


def test_the_shared_default_namespace_is_what_forced_the_prefix() -> None:
    """The control for the namespacing: both passes at their default collide."""
    with pytest.raises(AutodiffError) as failure:
        compile_composed(derivative_expansions=(autodiff.expand_mean_derivative_program,))

    assert failure.value.category == "ambiguous_producer"


# --------------------------------------------------------------------------
# AC: the executed result matches a directly computed update within #128's
# stated tolerance
# --------------------------------------------------------------------------


def test_the_executed_update_matches_the_direct_computation(record: object) -> None:
    """§17.7.4. Loss, gradient, and updated parameter, all from expanded artifacts."""
    values = concrete_inputs(with_bias=False)
    executed = execute_step(record, values)
    expected_gradient = reference_gradients(values)["w"]

    np.testing.assert_allclose(
        executed.loss,
        reference_loss(values),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_allclose(
        executed.gradients["w"],
        expected_gradient,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_allclose(
        executed.updated["w"],
        values["w"] - LEARNING_RATE * expected_gradient,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )


def test_the_numerical_comparison_has_teeth(record: object) -> None:
    """The tolerance is tight enough to reject a wrong update.

    Guards the comparison above against passing for a degenerate reason: a
    learning rate applied at the wrong scale must not be absorbed.
    """
    values = concrete_inputs(with_bias=False)
    executed = execute_step(record, values)
    wrong = values["w"] - (2.0 * LEARNING_RATE) * reference_gradients(values)["w"]

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            executed.updated["w"],
            wrong,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )


# --------------------------------------------------------------------------
# edge cases §17.7's test plan names
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loss",
    [residual_loss, rank_reducing_residual_loss],
    ids=["keepdims_true", "keepdims_false"],
)
def test_both_keepdims_tiers_compose_and_execute(loss: object) -> None:
    """#128's rank-preserving and rank-reducing tiers both compose here.

    The rank-reducing tier additionally needs a reshape handler, on the
    gradient path as well as the forward one -- the omission #128's module
    documentation calls the most likely for a backend author.
    """
    compiled = compile_composed(loss)
    values = concrete_inputs(with_bias=False)
    executed = execute_step(compiled, values)

    np.testing.assert_allclose(
        executed.gradients["w"],
        reference_gradients(values)["w"],
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )


@pytest.mark.parametrize(
    "loss",
    [residual_loss, rank_reducing_residual_loss],
    ids=["keepdims_true", "keepdims_false"],
)
def test_both_keepdims_tiers_fail_the_control(loss: object) -> None:
    """Neither tier lowers against this registry without the passes."""
    with pytest.raises(AutodiffError) as failure:
        compile_composed(loss, forward_expansions=(), derivative_expansions=())

    assert failure.value.category == "unsupported_operator"


def test_two_parameters_compose_with_the_expansion_in_play() -> None:
    """Both gradients land on their own parameter, expanded, with no reduction handler."""
    compiled = compile_composed(
        biased_residual_loss,
        inputs=TWO_PARAMETER_INPUTS,
        parameters=("w", "b"),
    )
    values = concrete_inputs(with_bias=True)
    executed = execute_step(compiled, values)
    expected = reference_gradients(values)

    assert tuple(parameter.name for parameter in compiled.parameters) == ("w", "b")
    # The two parameters have different shapes, so a swapped pairing is a
    # shape error rather than a plausible number.
    assert executed.gradients["w"].shape != executed.gradients["b"].shape
    for name in ("w", "b"):
        np.testing.assert_allclose(
            executed.gradients[name],
            expected[name],
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )
        np.testing.assert_allclose(
            executed.updated[name],
            values[name] - LEARNING_RATE * expected[name],
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )


@pytest.mark.parametrize(
    ("forward_expansions", "derivative_expansions"),
    [
        ((), (FORWARD_PASS,)),
        ((DERIVATIVE_PASS,), ()),
    ],
    ids=["forward_pass_in_derivative_slot", "derivative_pass_in_forward_slot"],
)
def test_a_pass_in_the_wrong_sequence_slot_fails_rather_than_doing_nothing(
    forward_expansions: object, derivative_expansions: object
) -> None:
    """A pass handed the wrong artifact type must be reported, never ignored.

    Silently declining would make a misconfigured sequence indistinguishable
    from an applied one -- the same inertness this module exists to rule out.
    """
    with pytest.raises(AutodiffError) as failure:
        compile_composed(
            forward_expansions=forward_expansions,
            derivative_expansions=derivative_expansions,
        )

    assert failure.value.category == "expansion_contract_violation"


def test_the_error_category_vocabulary_is_unchanged() -> None:
    """This module observes existing categories only; it introduces none."""
    for category in (
        "unsupported_operator",
        "ambiguous_producer",
        "expansion_contract_violation",
    ):
        assert category in autodiff.AUTODIFF_ERROR_CATEGORIES
