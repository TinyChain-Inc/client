"""Unit tests for the source capture analysis and forward selection stage.

Pins the contract of FR-129-005, FR-129-006, FR-129-007, FR-129-013, and
Inv-7: the ordered forward-capture set is read from
`analyze_derivative_dependencies` applied to the **source** artifacts through
the documented `forward_capture` provenance, in analysis order, and the
forward selection is the loss value id followed by that set, with the loss
removed from the capture portion if it is also captured, duplicate-free.

Three rules govern how these tests are written:

* **Order is asserted as a sequence, never as a set.** Inv-7 states one
  equality between tuples. An assertion phrased over `set(...)` would hold
  for a permuted implementation and would therefore pin nothing that matters
  here, so it does not appear in this file.
* **The expected captures are the analysis's own, computed independently by
  the test.** That is the contract of FR-129-005 verbatim. To keep the
  equality from being satisfied by an implementation that rediscovered the
  captures by scanning the forward graph, the fixtures are chosen so that a
  scan gives a different answer: the residual fixture has three forward
  values the derivative never reads, the two-parameter fixture's analysis
  order agrees with neither the graph's declaration order nor sorted order,
  and the hand-built fixtures carry a forward node no derivative node
  consumes.
* **The loss-is-also-a-capture case is constructed, not assumed.** No VJP
  rule in the default registry reads the value its own node produced, so with
  really traced artifacts the loss -- the forward graph's final output --
  can never come back as a forward capture. The two cases that exercise the
  deduplication rule therefore build a forward graph and a derivative program
  by hand, and hand them to the real analysis unchanged: what is faked is the
  shape of the artifacts, never the analysis that reads them.

The Test Plan's edge case "two captures produced by the same node" is
structurally impossible -- a `TensorNodeRecord` has exactly one output value
id -- and its real form, one forward value read twice by the derivative, is
covered by the residual fixture, whose `dn3`/`dn4` multiply nodes both read
the residual.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import pytest
import tinychain as tc
from tinychain.autodiff import training_step
from tinychain.autodiff.dependencies import (
    DependencyAnalysis,
    ValueDependency,
    analyze_derivative_dependencies,
)
from tinychain.autodiff.graph import (
    AddOperator,
    MatmulOperator,
    MulOperator,
    SubOperator,
    TensorGraph,
    TensorNodeRecord,
)
from tinychain.autodiff.protocol import DerivativeMetadata
from tinychain.autodiff.reverse import DerivativeProgram
from tinychain.autodiff.training_step import (
    SourceDerivative,
    TracedLoss,
    differentiate_loss,
    trace_loss,
)


# --------------------------------------------------------------------------
# the stage under test, resolved at call time
#
# Resolved through the module rather than imported by name so each case fails
# on its own line with the missing attribute it needs, instead of every case
# in the file collapsing into one collection-time ImportError. After the
# stage exists this is an ordinary call.
# --------------------------------------------------------------------------


def _analyze_source_captures(**kwargs: object) -> object:
    return training_step.analyze_source_captures(**kwargs)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


SPEC: Mapping[str, object] = {"dtype": "f32", "shape": [2, 3]}


@dataclasses.dataclass(frozen=True)
class _SourceArtifacts:
    """One source trace, its derivative, and the analysis of the two.

    ``analysis`` is computed by the test itself, straight from the public
    `analyze_derivative_dependencies`, so every expectation below is stated
    against the analysis the requirement names rather than against a
    hand-copied list of identifiers.
    """

    traced: TracedLoss
    derivative: SourceDerivative
    analysis: DependencyAnalysis

    @property
    def analysis_capture_value_ids(self) -> tuple[str, ...]:
        return tuple(dependency.value_id for dependency in self.analysis.forward_captures)


def _analyze(traced: TracedLoss, derivative: SourceDerivative) -> DependencyAnalysis:
    return analyze_derivative_dependencies(
        derivative.program,
        forward_graph=traced.graph,
        seed_value_ids=[derivative.seed_value_id],
        outputs=list(derivative.program.output_gradients),
    )


def _compile_source(
    *,
    inputs: Mapping[str, Mapping[str, object]],
    input_names: Sequence[str],
    loss: object,
    parameters: Sequence[str],
) -> _SourceArtifacts:
    """Trace, differentiate, and analyze one loss through the real stages."""
    with tc.state.scoped_context():
        traced = trace_loss(inputs=inputs, input_names=tuple(input_names), loss=loss)
    derivative = differentiate_loss(traced=traced, parameters=tuple(parameters))
    return _SourceArtifacts(
        traced=traced, derivative=derivative, analysis=_analyze(traced, derivative)
    )


RESIDUAL_INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": {"dtype": "f32", "shape": (2, 3)},
    "w": {"dtype": "f32", "shape": (3, 4)},
    "y": {"dtype": "f32", "shape": (2, 4)},
}


def _residual_loss(*, x: object, w: object, y: object) -> object:
    """The §17.3.1 loss: the multiply's VJP reads its own operand."""
    residual = x @ w - y
    return (residual * residual).mean([0, 1])


def _two_residual_loss(*, x: object, w: object, y: object) -> object:
    """The same loss with the residual written twice, giving two captures."""
    return ((x @ w - y) * (x @ w - y)).mean([0, 1])


SHARED_INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": {"dtype": "f32", "shape": (2, 3)},
    "w": {"dtype": "f32", "shape": (3, 4)},
    "b": {"dtype": "f32", "shape": (2, 4)},
    "y": {"dtype": "f32", "shape": (2, 4)},
}


def _shared_capture_loss(*, x: object, w: object, b: object, y: object) -> object:
    """Both parameters' gradient paths read the same residual."""
    residual = x @ w - y
    return ((residual * residual) + (residual * b)).mean([0, 1])


ADDITIVE_INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": {"dtype": "f32", "shape": (2, 3)},
    "w": {"dtype": "f32", "shape": (2, 3)},
}


def _additive_loss(*, x: object, w: object) -> object:
    """No VJP on this path reads a forward value, so nothing is captured."""
    return (x + w).mean([0, 1])


def _identity_loss(*, w: object) -> object:
    """A loss that is a declared input, returned unchanged."""
    return w


def _residual_artifacts() -> _SourceArtifacts:
    return _compile_source(
        inputs=RESIDUAL_INPUTS,
        input_names=("x", "w", "y"),
        loss=_residual_loss,
        parameters=("w",),
    )


def _two_capture_artifacts() -> _SourceArtifacts:
    return _compile_source(
        inputs=RESIDUAL_INPUTS,
        input_names=("x", "w", "y"),
        loss=_two_residual_loss,
        parameters=("x", "w"),
    )


def _shared_capture_artifacts() -> _SourceArtifacts:
    return _compile_source(
        inputs=SHARED_INPUTS,
        input_names=("x", "w", "b", "y"),
        loss=_shared_capture_loss,
        parameters=("w", "b"),
    )


def _additive_artifacts() -> _SourceArtifacts:
    return _compile_source(
        inputs=ADDITIVE_INPUTS,
        input_names=("x", "w"),
        loss=_additive_loss,
        parameters=("w",),
    )


def _identity_artifacts() -> _SourceArtifacts:
    return _compile_source(
        inputs={"w": {"dtype": "f32", "shape": (2, 3)}},
        input_names=("w",),
        loss=_identity_loss,
        parameters=("w",),
    )


# --------------------------------------------------------------------------
# hand-built artifacts: the loss is itself captured
#
# Built rather than traced because no default VJP rule reads the value its
# own node produced, so a really traced loss -- always the forward graph's
# final output -- is never a forward capture. The analysis that reads these
# artifacts is the real one; only their shape is chosen by the test.
# --------------------------------------------------------------------------


def _node(
    node_id: str, output_value_id: str, operator: object, input_value_ids: Sequence[str]
) -> TensorNodeRecord:
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=operator,
        op_params={},
        input_value_ids=list(input_value_ids),
        output_typespec=dict(SPEC),
    )


def _derivative_program(
    nodes: Sequence[TensorNodeRecord],
    *,
    gradient_value_id: str,
    parameter_value_id: str,
    extra_typespecs: Sequence[str] = (),
) -> DerivativeProgram:
    typespecs = {value_id: dict(SPEC) for value_id in extra_typespecs}
    for node in nodes:
        typespecs[node.output_value_id] = dict(SPEC)
    return DerivativeProgram(
        nodes=list(nodes),
        gradients={parameter_value_id: gradient_value_id},
        output_gradients=[gradient_value_id],
        metadata=DerivativeMetadata(
            source_graph_id="source-graph",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=(parameter_value_id,),
            seed_contract="ones_like(output)",
        ),
        value_typespecs=typespecs,
    )


def _captured_loss_last_artifacts() -> _SourceArtifacts:
    """The loss is captured, and is the **last** capture in analysis order.

    Forward: ``zulu = param + param``, ``mike = zulu * param``, then the loss
    ``alpha = mike * zulu``, plus ``omega`` which no derivative node reads.
    The derivative reads ``zulu``, ``mike``, and ``alpha``. The identifiers
    are chosen so that analysis order -- ``zulu``, ``mike``, ``alpha`` -- is
    not sorted order, so the capture portion left after the loss is removed
    is still order-sensitive.
    """
    forward = TensorGraph(
        nodes=[
            _node("f0", "zulu", AddOperator(), ["param", "param"]),
            _node("f1", "mike", MulOperator(), ["zulu", "param"]),
            _node("f2", "alpha", MulOperator(), ["mike", "zulu"]),
            _node("f3", "omega", MulOperator(), ["zulu", "param"]),
        ],
        inputs=[("param", dict(SPEC))],
        outputs=["alpha"],
    )
    program = _derivative_program(
        [
            _node("dn0", "grad_partial", MulOperator(), ["seed", "zulu"]),
            _node("dn1", "grad_scaled", MulOperator(), ["grad_partial", "mike"]),
            _node("dn2", "grad_param", MulOperator(), ["grad_scaled", "alpha"]),
        ],
        gradient_value_id="grad_param",
        parameter_value_id="param",
        extra_typespecs=("seed",),
    )
    traced = TracedLoss(
        graph=forward, loss_value_id="alpha", input_value_ids={"param": "param"}
    )
    derivative = SourceDerivative(
        program=program, seed_value_id="seed", seed_label="seed"
    )
    return _SourceArtifacts(
        traced=traced, derivative=derivative, analysis=_analyze(traced, derivative)
    )


def _captured_loss_first_artifacts() -> _SourceArtifacts:
    """The loss is captured, and is the **first** capture in analysis order.

    Forward: the loss ``alpha = param + param`` is produced first and is the
    graph's selected output, and ``zulu = alpha * param`` is produced after
    it. Both are read by the derivative, so analysis order is ``alpha`` then
    ``zulu`` and the loss must be removed from the front of the capture
    portion rather than from its end.
    """
    forward = TensorGraph(
        nodes=[
            _node("f0", "alpha", AddOperator(), ["param", "param"]),
            _node("f1", "zulu", MulOperator(), ["alpha", "param"]),
        ],
        inputs=[("param", dict(SPEC))],
        outputs=["alpha"],
    )
    program = _derivative_program(
        [
            _node("dn0", "grad_partial", MulOperator(), ["seed", "alpha"]),
            _node("dn1", "grad_param", MulOperator(), ["grad_partial", "zulu"]),
        ],
        gradient_value_id="grad_param",
        parameter_value_id="param",
        extra_typespecs=("seed",),
    )
    traced = TracedLoss(
        graph=forward, loss_value_id="alpha", input_value_ids={"param": "param"}
    )
    derivative = SourceDerivative(
        program=program, seed_value_id="seed", seed_label="seed"
    )
    return _SourceArtifacts(
        traced=traced, derivative=derivative, analysis=_analyze(traced, derivative)
    )


ALL_FIXTURES = {
    "residual": _residual_artifacts,
    "two-capture": _two_capture_artifacts,
    "shared-capture": _shared_capture_artifacts,
    "additive": _additive_artifacts,
    "identity": _identity_artifacts,
    "captured-loss-last": _captured_loss_last_artifacts,
    "captured-loss-first": _captured_loss_first_artifacts,
}


def _forward_value_id(artifacts: _SourceArtifacts, operator_type: type) -> str:
    """The sole forward output value produced by *operator_type*.

    Used to name a value structurally instead of hard-coding an identifier
    the tracer chose.
    """
    matches = [
        node.output_value_id
        for node in artifacts.traced.graph.nodes
        if isinstance(node.operator, operator_type)
    ]
    assert len(matches) == 1, (
        f"fixture must contain exactly one {operator_type.__name__} node, got {matches}"
    )
    return matches[0]


# --------------------------------------------------------------------------
# AC-1 -- the non-degenerate loss captures the residual
# --------------------------------------------------------------------------


def test_residual_loss_captures_are_non_empty_and_name_the_residual_value():
    # Arrange
    artifacts = _residual_artifacts()
    residual_value_id = _forward_value_id(artifacts, SubOperator)

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert -- exactly the residual, and nothing else
    assert captures.forward_capture_value_ids == (residual_value_id,)


def test_residual_loss_captures_exclude_forward_values_the_derivative_never_reads():
    # Arrange -- the matmul output, the squared residual, and the loss itself
    # are all forward values; none of them is read by the derivative program.
    artifacts = _residual_artifacts()
    matmul_value_id = _forward_value_id(artifacts, MatmulOperator)
    squared_value_id = _forward_value_id(artifacts, MulOperator)
    loss_value_id = artifacts.traced.loss_value_id

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    assert matmul_value_id not in captures.forward_capture_value_ids
    assert squared_value_id not in captures.forward_capture_value_ids
    assert loss_value_id not in captures.forward_capture_value_ids


def test_a_forward_value_the_derivative_reads_twice_is_captured_once():
    # Arrange -- the squared residual's VJP emits two multiplies, each of
    # which reads the residual, so the same forward value is read twice.
    artifacts = _residual_artifacts()
    residual_value_id = _forward_value_id(artifacts, SubOperator)
    readers = [
        node
        for node in artifacts.derivative.program.nodes
        if residual_value_id in node.input_value_ids
    ]
    assert len(readers) == 2, (
        "fixture must read the residual twice for this case to discriminate, "
        f"got {len(readers)} reader(s)"
    )

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    assert list(captures.forward_capture_value_ids).count(residual_value_id) == 1


def test_a_capture_shared_by_two_parameters_is_reported_once():
    # Arrange -- both `w` and `b` need the same residual.
    artifacts = _shared_capture_artifacts()
    residual_value_id = _forward_value_id(artifacts, SubOperator)

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    assert captures.forward_capture_value_ids == (residual_value_id,)


# --------------------------------------------------------------------------
# AC-2 -- the forward selection and its order
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", sorted(ALL_FIXTURES))
def test_forward_selection_is_the_loss_followed_by_the_reported_captures(
    fixture_name: str,
):
    # Arrange
    artifacts = ALL_FIXTURES[fixture_name]()

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert -- Inv-7, as a sequence equality
    assert captures.forward_selected_outputs == (
        captures.loss_value_id,
    ) + captures.forward_capture_value_ids


@pytest.mark.parametrize("fixture_name", sorted(ALL_FIXTURES))
def test_forward_selection_is_duplicate_free(fixture_name: str):
    # Arrange
    artifacts = ALL_FIXTURES[fixture_name]()

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    selection = captures.forward_selected_outputs
    assert len(set(selection)) == len(selection)


@pytest.mark.parametrize("fixture_name", sorted(ALL_FIXTURES))
def test_reported_loss_value_id_is_the_traced_loss_value(fixture_name: str):
    # Arrange
    artifacts = ALL_FIXTURES[fixture_name]()

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    assert captures.loss_value_id == artifacts.traced.loss_value_id


def test_capture_order_follows_analysis_order_not_forward_declaration_order():
    # Arrange -- two residuals: the analysis reports them in an order that is
    # neither the graph's node order nor sorted order, so the equality below
    # fails for an implementation that scanned the graph or sorted the ids.
    artifacts = _two_capture_artifacts()
    expected = artifacts.analysis_capture_value_ids
    declaration_order = tuple(
        node.output_value_id
        for node in artifacts.traced.graph.nodes
        if node.output_value_id in set(expected)
    )
    assert len(expected) == 2, (
        f"fixture must produce two captures to discriminate order, got {expected}"
    )
    assert expected != declaration_order, (
        "fixture no longer discriminates: analysis order equals forward "
        "declaration order"
    )
    assert expected != tuple(sorted(expected)), (
        "fixture no longer discriminates: analysis order equals sorted order"
    )

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    assert captures.forward_capture_value_ids == expected


def test_analysis_selects_the_gradient_value_ids_in_parameter_order():
    # Arrange -- `wrt` order is `parameters` order (Inv-6), so the gradients
    # the capture analysis is taken against must be exactly that sequence.
    artifacts = _two_capture_artifacts()

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    assert captures.analysis.selected_outputs == tuple(
        artifacts.derivative.program.output_gradients
    )


# --------------------------------------------------------------------------
# AC-3 -- a loss that is itself captured
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name", ["captured-loss-first", "captured-loss-last"]
)
def test_a_captured_loss_appears_exactly_once_and_at_the_front(fixture_name: str):
    # Arrange
    artifacts = ALL_FIXTURES[fixture_name]()
    loss_value_id = artifacts.traced.loss_value_id
    assert loss_value_id in artifacts.analysis_capture_value_ids, (
        "fixture must genuinely capture the loss for this case to discriminate"
    )

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    selection = captures.forward_selected_outputs
    assert selection[0] == loss_value_id
    assert list(selection).count(loss_value_id) == 1


@pytest.mark.parametrize(
    "fixture_name", ["captured-loss-first", "captured-loss-last"]
)
def test_a_captured_loss_is_absent_from_the_capture_portion(fixture_name: str):
    # Arrange
    artifacts = ALL_FIXTURES[fixture_name]()
    loss_value_id = artifacts.traced.loss_value_id
    expected_captures = tuple(
        value_id
        for value_id in artifacts.analysis_capture_value_ids
        if value_id != loss_value_id
    )
    assert expected_captures, (
        "fixture must keep at least one capture besides the loss, so removing "
        "the loss is distinguishable from emptying the tuple"
    )

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert -- the remaining captures keep their analysis order
    assert captures.forward_capture_value_ids == expected_captures


# --------------------------------------------------------------------------
# AC-4 -- the loss and the captures are reported separately
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", sorted(ALL_FIXTURES))
def test_the_loss_value_id_is_never_a_member_of_the_reported_captures(
    fixture_name: str,
):
    # Arrange
    artifacts = ALL_FIXTURES[fixture_name]()

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    assert captures.loss_value_id not in captures.forward_capture_value_ids
    for value_id in captures.forward_capture_value_ids:
        assert value_id != captures.loss_value_id


def test_declared_inputs_the_derivative_reads_are_not_reported_as_captures():
    # Arrange -- differentiating with respect to `x` makes the derivative read
    # the declared inputs `x` and `w`; those are the consumer's own bindings,
    # not values the forward phase must retain.
    artifacts = _two_capture_artifacts()
    declared_value_ids = tuple(artifacts.traced.input_value_ids.values())
    read_declared = [
        value_id
        for value_id in declared_value_ids
        if any(
            value_id in node.input_value_ids
            for node in artifacts.derivative.program.nodes
        )
    ]
    assert read_declared, (
        "fixture must have the derivative read at least one declared input"
    )

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    for value_id in declared_value_ids:
        assert value_id not in captures.forward_capture_value_ids


def test_the_minted_seed_is_not_reported_as_a_capture():
    # Arrange
    artifacts = _residual_artifacts()

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    assert artifacts.derivative.seed_value_id not in captures.forward_capture_value_ids
    assert artifacts.derivative.seed_value_id not in captures.forward_selected_outputs


# --------------------------------------------------------------------------
# Test Plan -- a derivative that needs no capture
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["additive", "identity"])
def test_a_loss_whose_derivative_needs_no_capture_selects_the_loss_alone(
    fixture_name: str,
):
    # Arrange
    artifacts = ALL_FIXTURES[fixture_name]()
    assert artifacts.analysis_capture_value_ids == (), (
        "fixture must genuinely capture nothing for this case to discriminate"
    )

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    assert captures.forward_capture_value_ids == ()
    assert captures.forward_selected_outputs == (artifacts.traced.loss_value_id,)


# --------------------------------------------------------------------------
# AC-5 -- stability across two analyses of equal artifacts
# --------------------------------------------------------------------------


def test_capture_order_is_equal_across_two_compilations_of_the_same_loss():
    # Arrange -- two independent traces of the same declaration
    first = _two_capture_artifacts()
    second = _two_capture_artifacts()

    # Act
    first_captures = _analyze_source_captures(
        traced=first.traced, derivative=first.derivative
    )
    second_captures = _analyze_source_captures(
        traced=second.traced, derivative=second.derivative
    )

    # Assert
    assert first_captures.forward_capture_value_ids == (
        second_captures.forward_capture_value_ids
    )
    assert first_captures.forward_selected_outputs == (
        second_captures.forward_selected_outputs
    )


def test_repeated_analysis_of_the_same_artifacts_returns_equal_results():
    # Arrange
    artifacts = _shared_capture_artifacts()

    # Act
    first_captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )
    second_captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    assert first_captures.forward_capture_value_ids == (
        second_captures.forward_capture_value_ids
    )
    assert first_captures.forward_selected_outputs == (
        second_captures.forward_selected_outputs
    )
    assert first_captures.loss_value_id == second_captures.loss_value_id


# --------------------------------------------------------------------------
# AC-6 -- the stage restates no per-value provenance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", sorted(ALL_FIXTURES))
def test_the_stage_reports_only_value_identifiers_and_the_analysis_object(
    fixture_name: str,
):
    # Arrange
    artifacts = ALL_FIXTURES[fixture_name]()

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert -- every reported field is either the analysis itself, a value
    # id, or an ordered tuple of value ids. A field holding `ValueDependency`
    # objects, a provenance string, or a per-value mapping would be this
    # stage restating what `LoweredProgram.dependencies` already owns
    # (FR-129-013).
    analysis_fields = 0
    for field in dataclasses.fields(captures):
        value = getattr(captures, field.name)
        if isinstance(value, DependencyAnalysis):
            analysis_fields += 1
            continue
        if isinstance(value, str):
            continue
        assert isinstance(value, tuple), (
            f"field {field.name!r} is neither the analysis, a value id, nor a "
            f"tuple of value ids: {value!r}"
        )
        for item in value:
            assert isinstance(item, str), (
                f"field {field.name!r} carries {item!r}, which is not a value id"
            )
    assert analysis_fields == 1


@pytest.mark.parametrize("fixture_name", sorted(ALL_FIXTURES))
def test_dependency_provenance_is_reachable_unmodified_through_the_analysis(
    fixture_name: str,
):
    # Arrange
    artifacts = ALL_FIXTURES[fixture_name]()

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert -- the analysis the stage carries equals the one the requirement
    # names, dependency for dependency, provenance included.
    assert captures.analysis == artifacts.analysis
    assert all(
        isinstance(dependency, ValueDependency)
        for dependency in captures.analysis.dependencies
    )


@pytest.mark.parametrize(
    "fixture_name", ["captured-loss-first", "captured-loss-last"]
)
def test_a_captured_loss_is_still_visible_as_a_capture_in_the_analysis(
    fixture_name: str,
):
    # Arrange -- removing the loss from the reported capture portion is a
    # selection rule, not a rewrite of the analysis: the analysis must still
    # report the loss under `forward_capture` provenance.
    artifacts = ALL_FIXTURES[fixture_name]()
    loss_value_id = artifacts.traced.loss_value_id

    # Act
    captures = _analyze_source_captures(
        traced=artifacts.traced, derivative=artifacts.derivative
    )

    # Assert
    analysis_captures = tuple(
        dependency.value_id for dependency in captures.analysis.forward_captures
    )
    assert loss_value_id in analysis_captures
    assert loss_value_id not in captures.forward_capture_value_ids
