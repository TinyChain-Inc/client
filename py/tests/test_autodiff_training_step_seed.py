"""Unit tests for the minted-seed and source-differentiation stage.

Pins the contract of FR-129-003, FR-129-004, FR-129-019, Inv-1, Inv-6, and
Inv-11: the framework *mints* the seed value id, requests the VJP from the
**source** forward graph with `wrt` equal to the declared parameters' value
ids in caller order, and -- immediately after `generate` returns, before
dependency analysis and before any expansion pass -- checks that seed against
every node id and node output value id the source derivative program
produces.

Two rules govern how these tests are written, and both come straight from
§8.3 and §17.6.3:

* **The seed's spelling is not the contract.** Not one assertion here names,
  matches, or pattern-checks a namespace, prefix, or literal identifier for
  the minted seed. Where a test needs a graph that occupies the minter's
  natural first candidates, it *discovers* those candidates by minting
  against a graph that does not contain them, then builds the next graph out
  of what it observed. A test that hard-coded `"s0"` would pass for an
  implementation whose search never advances, and would fail the moment the
  spelling changed -- neither of which is the property under test.
* **Ordering is observed, not assumed.** "Before dependency analysis" is
  proved by a recording seam that both counts calls and raises a sentinel if
  it is ever entered, so an implementation that analysed first would surface
  the sentinel instead of `ambiguous_producer`.

`generate` itself is off-limits and is consumed unchanged. It performs no
seed-collision check of its own -- that absence is the entire reason
FR-129-019's post-generation check exists -- so the collision cases inject a
controlled `DerivativeProgram` through the module-level `generate` name,
which is the only seam by which a collision can be exercised at all.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Optional

import pytest
import tinychain as tc
from tinychain.autodiff import training_step
from tinychain.autodiff.generate import generate as _real_generate
from tinychain.autodiff.graph import MulOperator, TensorGraph, TensorNodeRecord
from tinychain.autodiff.protocol import AutodiffError
from tinychain.autodiff.reverse import DerivativeProgram
from tinychain.autodiff.training_step import TracedLoss, trace_loss

from tests.autodiff_training_step_seed_support import (
    PARAMETER_NAMES,
    SPEC,
    AnalysisEntered as _AnalysisEntered,
    FixedGenerate as _FixedGenerate,
    fake_program as _fake_program,
    forbid_dependency_analysis as _forbid_dependency_analysis,
    mint_seed_against as _mint_seed_against,
    mul_chain as _mul_chain,
    occupied_value_ids as _occupied_value_ids,
    traced_chain as _traced_chain,
)


# --------------------------------------------------------------------------
# the stage under test, resolved at call time
#
# Resolved through the module rather than imported by name so each case fails
# on its own line with the missing attribute it needs, instead of every case
# in the file collapsing into one collection-time ImportError. After the
# stage exists this is an ordinary call.
# --------------------------------------------------------------------------


def _differentiate_loss(**kwargs: object) -> object:
    return training_step.differentiate_loss(**kwargs)


# --------------------------------------------------------------------------
# fixtures
#
# The hand-built source graphs, the `generate` seam, and the analysis seam are
# shared with `test_autodiff_training_step_contract` and imported above; only
# the specs and losses this file alone uses are declared here. The end-to-end
# cases below use the real tracer.
# --------------------------------------------------------------------------

SYMBOLIC_SPEC: Mapping[str, object] = {"dtype": "f32", "shape": ["N", 3]}
INTEGER_SPEC: Mapping[str, object] = {"dtype": "i32", "shape": [2, 3]}

LINEAR_INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": {"dtype": "f32", "shape": (2, 3)},
    "w": {"dtype": "f32", "shape": (3, 4)},
    "y": {"dtype": "f32", "shape": (2, 4)},
}
LINEAR_INPUT_NAMES = ("x", "w", "y")


def _linear_loss(*, x: object, w: object, y: object) -> object:
    d = x @ w - y
    return (d * d).mean([0, 1])


def _trace_linear() -> TracedLoss:
    with tc.state.scoped_context():
        return trace_loss(
            inputs=LINEAR_INPUTS, input_names=LINEAR_INPUT_NAMES, loss=_linear_loss
        )


# --------------------------------------------------------------------------
# seams
# --------------------------------------------------------------------------


class _RecordingGenerate:
    """Records every argument `generate` was called with, then delegates.

    Arguments are normalized through `generate`'s own signature, so the
    recording is independent of whether the stage passes them positionally or
    by keyword.
    """

    def __init__(self, delegate=_real_generate) -> None:
        self._delegate = delegate
        self._signature = inspect.signature(_real_generate)
        self.calls: list[dict[str, object]] = []
        self.graphs_at_call: list[TensorGraph] = []
        self.returned: list[object] = []
        self.snapshots_at_call: list[object] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        self.calls.append(dict(bound.arguments))
        graph = bound.arguments["graph"]
        self.graphs_at_call.append(graph)
        self.snapshots_at_call.append(_graph_snapshot(graph))
        program = self._delegate(*args, **kwargs)
        self.returned.append(program)
        return program

    @property
    def only_call(self) -> dict[str, object]:
        assert len(self.calls) == 1, f"expected exactly one generate call, got {len(self.calls)}"
        return self.calls[0]


def _derivative_node(*, node_id: str, output_value_id: str) -> TensorNodeRecord:
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=MulOperator(),
        op_params={},
        input_value_ids=["pa", "pb"],
        output_typespec=dict(SPEC),
    )


# --------------------------------------------------------------------------
# snapshots
# --------------------------------------------------------------------------


def _typespec_snapshot(typespec: Optional[Mapping[str, object]]) -> object:
    if typespec is None:
        return None
    return tuple(sorted((key, repr(value)) for key, value in typespec.items()))


def _node_snapshot(node: TensorNodeRecord) -> tuple:
    return (
        node.node_id,
        node.output_value_id,
        type(node.operator).__name__,
        node.operator.route_name,
        tuple(sorted((key, repr(value)) for key, value in node.op_params.items())),
        tuple(node.input_value_ids),
        _typespec_snapshot(node.output_typespec),
    )


def _graph_snapshot(graph: TensorGraph) -> tuple:
    return (
        tuple(_node_snapshot(node) for node in graph.nodes),
        tuple((value_id, _typespec_snapshot(spec)) for value_id, spec in graph.inputs),
        tuple(graph.outputs),
    )


def _program_snapshot(program: DerivativeProgram) -> tuple:
    metadata = program.metadata
    return (
        tuple(_node_snapshot(node) for node in program.nodes),
        tuple(sorted(program.gradients.items())),
        tuple(program.output_gradients),
        (
            metadata.source_graph_id,
            metadata.transform_version,
            metadata.tensor_op_contract_version,
            tuple(metadata.wrt_signature),
            metadata.seed_contract,
        ),
        tuple(
            (value_id, _typespec_snapshot(spec))
            for value_id, spec in sorted(program.value_typespecs.items())
        ),
    )


def _graph_identifiers(graph: TensorGraph) -> set[str]:
    identifiers = {value_id for value_id, _ in graph.inputs}
    identifiers.update(graph.outputs)
    for node in graph.nodes:
        identifiers.add(node.node_id)
        identifiers.add(node.output_value_id)
        identifiers.update(node.input_value_ids)
    return identifiers


def _program_identifiers(program: DerivativeProgram) -> set[str]:
    identifiers: set[str] = set()
    for node in program.nodes:
        identifiers.add(node.node_id)
        identifiers.add(node.output_value_id)
        identifiers.update(node.input_value_ids)
    identifiers.update(program.gradients)
    identifiers.update(program.gradients.values())
    identifiers.update(
        value_id for value_id in program.output_gradients if value_id is not None
    )
    identifiers.update(program.value_typespecs)
    identifiers.update(program.metadata.wrt_signature)
    return identifiers


# --------------------------------------------------------------------------
# AC-1: the minted seed is absent from the source graph's declared input
# value ids and from every node output value id, including for a graph built
# to occupy the minter's natural first candidates.
# --------------------------------------------------------------------------


def test_minted_seed_walks_past_every_occupied_candidate_it_is_offered() -> None:
    """Occupy the minter's own candidates, one at a time, and keep going.

    The candidates are never named by this test: each round mints against a
    graph that does not contain the previously discovered ones, observes what
    the minter chose, and builds the next graph so that choice is occupied
    too. An implementation whose search does not advance fails in round two;
    one that advances only once fails in round three; one that returns a
    value colliding with a declared input fails immediately.
    """
    discovered: list[str] = []
    for round_index in range(12):
        # Round 0 has no discovered candidates yet, so it needs one node to
        # be a well-formed graph; the loss output id is deliberately unlike
        # anything a minter would produce.
        output_value_ids = discovered or ["loss_output"]
        traced = _traced_chain(output_value_ids)
        seed = _mint_seed_against(traced)

        occupied = _occupied_value_ids(traced.graph)
        assert seed not in occupied, (
            f"round {round_index}: minted seed {seed!r} is already occupied by "
            f"the source forward graph, whose occupied ids are {sorted(occupied)}"
        )
        assert seed not in discovered, (
            f"round {round_index}: minted seed {seed!r} repeats a candidate the "
            "minter had already been shown to be occupied"
        )
        discovered.append(seed)

    assert len(set(discovered)) == 12


def test_minted_seed_avoids_a_candidate_occupying_a_parameters_own_value_id() -> None:
    """The first candidate is occupied by a *parameter*, not by an intermediate.

    A minter that only scanned node outputs -- the ids it is most likely to
    look at -- would hand back the parameter's own value id here.
    """
    first_candidate = _mint_seed_against(_traced_chain(["loss_output"]))

    traced = _traced_chain(
        ["loss_output"], input_value_ids=(first_candidate, "pb")
    )
    result = _differentiate_loss(traced=traced, parameters=[PARAMETER_NAMES[0]])

    assert result.seed_value_id != first_candidate
    assert result.seed_value_id not in _occupied_value_ids(traced.graph)
    assert result.seed_value_id != traced.input_value_ids[PARAMETER_NAMES[0]]


def test_minted_seed_is_absent_from_a_really_traced_graphs_identifiers() -> None:
    traced = _trace_linear()

    result = _differentiate_loss(traced=traced, parameters=["w"])

    assert result.seed_value_id not in _occupied_value_ids(traced.graph)
    assert result.seed_value_id not in _graph_identifiers(traced.graph)


# --------------------------------------------------------------------------
# AC-2: `wrt` equals the parameter value ids in `parameters` order, and
# `metadata.wrt_signature` equals that same order (Inv-6).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parameters", [["w", "b"], ["b", "w"]], ids=["declaration-order", "reversed"]
)
def test_wrt_is_the_parameter_value_ids_in_parameters_order(
    monkeypatch: pytest.MonkeyPatch, parameters: list[str]
) -> None:
    """Both orders are exercised, so agreeing by luck is not possible.

    The graph's value ids (`pa`, `pb`) differ from the parameter names, so a
    `wrt` built from names rather than value ids fails, and the reversed case
    fails any implementation that sorts, uses declaration order, or uses the
    graph's input order instead of `parameters` order.
    """
    recorder = _RecordingGenerate()
    monkeypatch.setattr(training_step, "generate", recorder)
    traced = _traced_chain(["loss_output"])
    expected_wrt = [traced.input_value_ids[name] for name in parameters]

    result = _differentiate_loss(traced=traced, parameters=parameters)

    assert list(recorder.only_call["wrt"]) == expected_wrt
    assert tuple(result.program.metadata.wrt_signature) == tuple(expected_wrt)
    assert list(result.program.output_gradients) == [
        result.program.gradients[value_id] for value_id in expected_wrt
    ]


def test_wrt_for_a_single_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingGenerate()
    monkeypatch.setattr(training_step, "generate", recorder)
    traced = _traced_chain(["loss_output"])

    result = _differentiate_loss(traced=traced, parameters=["b"])

    assert list(recorder.only_call["wrt"]) == [traced.input_value_ids["b"]]
    assert tuple(result.program.metadata.wrt_signature) == (
        traced.input_value_ids["b"],
    )


def test_generate_is_asked_for_the_traced_loss_value_and_the_minted_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _RecordingGenerate()
    monkeypatch.setattr(training_step, "generate", recorder)
    traced = _traced_chain(["loss_output"])

    result = _differentiate_loss(traced=traced, parameters=["w"])

    call = recorder.only_call
    output_value_id = call["output_value_id"]
    assert (
        output_value_id == traced.loss_value_id
        or output_value_id == [traced.loss_value_id]
    )
    seed = call["seed"]
    assert seed == result.seed_value_id or seed == [result.seed_value_id]


# --------------------------------------------------------------------------
# AC (contract step 3): the seed typespec is the loss output's own typespec.
# --------------------------------------------------------------------------


def _recorded_seed_typespec(call: Mapping[str, object]) -> object:
    seed_typespec = call["seed_typespec"]
    if isinstance(seed_typespec, list):
        assert len(seed_typespec) == 1
        return seed_typespec[0]
    return seed_typespec


def test_seed_typespec_is_the_loss_nodes_own_output_typespec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A distinctive symbolic shape, so a fabricated default cannot match.

    An implementation passing `None`, or a plausible-looking
    `{"dtype": "f32", "shape": []}`, fails here rather than silently letting
    reverse traversal fall back to the output's typespec on its own.
    """
    recorder = _RecordingGenerate()
    monkeypatch.setattr(training_step, "generate", recorder)
    traced = _traced_chain(["loss_output"], spec=SYMBOLIC_SPEC)

    _differentiate_loss(traced=traced, parameters=["w"])

    assert _recorded_seed_typespec(recorder.only_call) == dict(SYMBOLIC_SPEC)


def test_seed_typespec_is_read_from_the_graph_inputs_when_the_loss_is_an_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loss value is a declared input, so its typespec is not on a node.

    An implementation that only scanned node output typespecs finds nothing
    here.
    """
    recorder = _RecordingGenerate()
    monkeypatch.setattr(training_step, "generate", recorder)
    graph = TensorGraph(
        nodes=_mul_chain(["intermediate"]).nodes,
        inputs=[("pa", dict(SYMBOLIC_SPEC)), ("pb", dict(SPEC))],
        outputs=["pa"],
    )
    traced = TracedLoss(
        graph=graph,
        loss_value_id="pa",
        input_value_ids={"w": "pa", "b": "pb"},
    )

    _differentiate_loss(traced=traced, parameters=["w"])

    assert _recorded_seed_typespec(recorder.only_call) == dict(SYMBOLIC_SPEC)


# --------------------------------------------------------------------------
# AC-3: `generate` receives the source forward graph object, and both source
# artifacts are unmutated afterwards (Inv-1).
# --------------------------------------------------------------------------


def test_generate_receives_the_source_forward_graph_object_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Object identity, not equality: a defensive copy is still a rewrite risk."""
    recorder = _RecordingGenerate()
    monkeypatch.setattr(training_step, "generate", recorder)
    traced = _traced_chain(["loss_output"])

    _differentiate_loss(traced=traced, parameters=["w"])

    assert recorder.graphs_at_call[0] is traced.graph


def test_the_source_forward_graph_is_unmutated_by_the_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compared at three points: before, at the `generate` call, and after.

    The middle comparison is what catches a stage that appends the seed to
    `graph.inputs` before differentiating -- a mutation that would leave the
    graph correct-looking only if it were also undone afterwards.
    """
    recorder = _RecordingGenerate()
    monkeypatch.setattr(training_step, "generate", recorder)
    traced = _traced_chain(["loss_output"])
    before = _graph_snapshot(traced.graph)

    result = _differentiate_loss(traced=traced, parameters=["w", "b"])

    assert recorder.snapshots_at_call[0] == before
    assert _graph_snapshot(traced.graph) == before
    assert result.seed_value_id not in _occupied_value_ids(traced.graph)


def test_the_returned_program_is_generates_own_object_and_is_unmutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _RecordingGenerate()
    monkeypatch.setattr(training_step, "generate", recorder)
    traced = _traced_chain(["loss_output"])

    result = _differentiate_loss(traced=traced, parameters=["w", "b"])

    generated = recorder.returned[0]
    assert result.program is generated
    assert _program_snapshot(result.program) == _program_snapshot(generated)


def test_the_source_derivative_program_is_unmutated_after_the_stage_returns() -> None:
    """Snapshot-compare a really generated program across the stage boundary.

    Taken without any seam in place, so the comparison covers whatever the
    real `generate` produced rather than a fixture's idea of it: the
    post-generation check must read the program, never edit it.
    """
    traced = _trace_linear()

    result = _differentiate_loss(traced=traced, parameters=["w"])
    after_stage = _program_snapshot(result.program)

    reference = _real_generate(
        traced.graph,
        traced.loss_value_id,
        [traced.input_value_ids["w"]],
        result.seed_value_id,
        seed_typespec=dict(
            next(
                node.output_typespec
                for node in traced.graph.nodes
                if node.output_value_id == traced.loss_value_id
            )
        ),
    )
    assert after_stage == _program_snapshot(reference)


# --------------------------------------------------------------------------
# AC-4: a derivative program colliding with the minted seed raises
# `ambiguous_producer` at the post-generation check, before dependency
# analysis (FR-129-019, §17.6.3).
# --------------------------------------------------------------------------


def _colliding_program_builder(*, collide_on: str):
    def build(seed_value_id: str) -> DerivativeProgram:
        node_id = seed_value_id if collide_on == "node_id" else "dn0"
        output_value_id = seed_value_id if collide_on == "output_value_id" else "d0"
        return _fake_program(
            nodes=[_derivative_node(node_id=node_id, output_value_id=output_value_id)],
            wrt=["pa"],
            gradient_value_id="d0",
            seed_value_id=seed_value_id,
        )

    return build


@pytest.mark.parametrize("collide_on", ["output_value_id", "node_id"])
def test_a_derivative_program_colliding_with_the_seed_fails_before_analysis(
    monkeypatch: pytest.MonkeyPatch, collide_on: str
) -> None:
    """The collision is built out of the seed the stage actually minted.

    The analysis seam both counts calls and raises `_AnalysisEntered`. An
    implementation that analysed the program before checking the seed would
    surface that sentinel -- not `ambiguous_producer` -- so "before analysis"
    is observed here rather than assumed from the fact that an error was
    raised at all.
    """
    analysis_calls = _forbid_dependency_analysis(monkeypatch)
    fake = _FixedGenerate(_colliding_program_builder(collide_on=collide_on))
    monkeypatch.setattr(training_step, "generate", fake)
    traced = _traced_chain(["loss_output"])

    with pytest.raises(AutodiffError) as excinfo:
        _differentiate_loss(traced=traced, parameters=["w"])

    assert excinfo.value.category == "ambiguous_producer"
    seed_value_id = fake.seeds[0]
    message = str(excinfo.value)
    assert seed_value_id in message
    assert analysis_calls == []


def test_a_derivative_program_without_a_collision_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control that keeps the check from degenerating into "always raise".

    The seed is present in `value_typespecs` here, exactly as reverse
    traversal records it for every program it ever produces. A check that
    looked there instead of at the produced node ids and output value ids
    would reject this program, and would therefore reject everything.
    """
    analysis_calls = _forbid_dependency_analysis(monkeypatch)
    fake = _FixedGenerate(
        lambda seed_value_id: _fake_program(
            nodes=[_derivative_node(node_id="dn0", output_value_id="d0")],
            wrt=["pa"],
            gradient_value_id="d0",
            seed_value_id=seed_value_id,
        )
    )
    monkeypatch.setattr(training_step, "generate", fake)
    traced = _traced_chain(["loss_output"])

    result = _differentiate_loss(traced=traced, parameters=["w"])

    assert result.seed_value_id == fake.seeds[0]
    assert result.program.output_gradients == ["d0"]
    assert analysis_calls == []


def test_a_collision_on_a_later_derivative_node_is_still_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every produced node is checked, not only the first one."""
    _forbid_dependency_analysis(monkeypatch)
    fake = _FixedGenerate(
        lambda seed_value_id: _fake_program(
            nodes=[
                _derivative_node(node_id="dn0", output_value_id="d0"),
                _derivative_node(node_id="dn1", output_value_id="d1"),
                _derivative_node(node_id="dn2", output_value_id=seed_value_id),
            ],
            wrt=["pa"],
            gradient_value_id="d0",
            seed_value_id=seed_value_id,
        )
    )
    monkeypatch.setattr(training_step, "generate", fake)

    with pytest.raises(AutodiffError) as excinfo:
        _differentiate_loss(traced=_traced_chain(["loss_output"]), parameters=["w"])

    assert excinfo.value.category == "ambiguous_producer"
    assert fake.seeds[0] in str(excinfo.value)


# --------------------------------------------------------------------------
# AC-5: `seed_label` is provenance only and never an identifier.
# --------------------------------------------------------------------------


def test_seed_label_has_no_effect_on_any_identifier() -> None:
    """Labels are compared against each other, not against a spelling.

    One of the labels is a value id the minter itself would otherwise have
    produced -- the second candidate, discovered by minting twice -- so an
    implementation that used the label as the seed id produces a *different,
    plausible* identifier and is caught. Labels spelled like existing graph
    ids are included because "the label never appears among the identifiers"
    is not the property: `pa` is a legitimate identifier here whatever the
    label says. The property is that the label changes nothing.
    """
    first_candidate = _mint_seed_against(_traced_chain(["loss_output"]))
    second_candidate = _mint_seed_against(_traced_chain([first_candidate]))

    traced = _traced_chain(["loss_output"])
    baseline = _differentiate_loss(traced=traced, parameters=["w", "b"])
    baseline_identifiers = _graph_identifiers(traced.graph) | _program_identifiers(
        baseline.program
    )

    for label in ("seed", "pa", "loss_output", second_candidate, "d0", "dn0"):
        labelled_traced = _traced_chain(["loss_output"])
        result = _differentiate_loss(
            traced=labelled_traced, parameters=["w", "b"], seed_label=label
        )

        assert result.seed_value_id == baseline.seed_value_id, (
            f"seed_label={label!r} changed the minted seed; the label carries "
            "no identity"
        )
        assert (
            _graph_identifiers(labelled_traced.graph)
            | _program_identifiers(result.program)
        ) == baseline_identifiers, f"seed_label={label!r} changed an identifier"
        assert result.seed_label == label


def test_seed_label_defaults_to_a_display_name_and_is_not_the_seed() -> None:
    traced = _traced_chain(["loss_output"])

    result = _differentiate_loss(traced=traced, parameters=["w"])

    assert result.seed_label == "seed"
    assert result.seed_value_id != result.seed_label


# --------------------------------------------------------------------------
# AC-6: `missing_derivative_behavior` from `generate` propagates unchanged.
# --------------------------------------------------------------------------


def test_missing_derivative_behavior_propagates_unchanged() -> None:
    """Compared against the error `generate` itself raises, not a literal.

    The parameter `b` is declared but the loss body never reads it, so
    reverse traversal computes no gradient for it. Asserting equality with
    `generate`'s own category *and message* is what makes "unchanged" a real
    claim: a stage that re-categorized or reworded it fails.
    """
    graph = TensorGraph(
        nodes=[
            TensorNodeRecord(
                node_id="m0",
                output_value_id="loss_output",
                operator=MulOperator(),
                op_params={},
                input_value_ids=["pa", "pa"],
                output_typespec=dict(SPEC),
            )
        ],
        inputs=[("pa", dict(SPEC)), ("pb", dict(SPEC))],
        outputs=["loss_output"],
    )
    traced = TracedLoss(
        graph=graph,
        loss_value_id="loss_output",
        input_value_ids={"w": "pa", "b": "pb"},
    )

    with pytest.raises(AutodiffError) as excinfo:
        _differentiate_loss(traced=traced, parameters=["w", "b"])

    with pytest.raises(AutodiffError) as reference:
        _real_generate(
            graph, "loss_output", ["pa", "pb"], "unused_seed", seed_typespec=dict(SPEC)
        )

    assert excinfo.value.category == reference.value.category == "missing_derivative_behavior"
    assert str(excinfo.value) == str(reference.value)


def test_a_non_differentiable_loss_dtype_propagates_generates_own_error() -> None:
    """Edge case: a loss output typespec with a non-floating dtype.

    Minting happens first and succeeds; the failure belongs to `generate` and
    must arrive with `generate`'s own category, unwrapped.
    """
    traced = _traced_chain(["loss_output"], spec=INTEGER_SPEC)

    with pytest.raises(AutodiffError) as excinfo:
        _differentiate_loss(traced=traced, parameters=["w"])

    with pytest.raises(AutodiffError) as reference:
        _real_generate(
            traced.graph,
            "loss_output",
            ["pa"],
            "unused_seed",
            seed_typespec=dict(INTEGER_SPEC),
        )

    assert excinfo.value.category == reference.value.category
    assert str(excinfo.value) == str(reference.value)


# --------------------------------------------------------------------------
# AC-7: two compilations of equal declarations mint the same seed (Inv-13).
# --------------------------------------------------------------------------


def test_two_stages_over_equal_declarations_mint_the_same_seed() -> None:
    first = _differentiate_loss(
        traced=_traced_chain(["loss_output"]), parameters=["w", "b"]
    )
    second = _differentiate_loss(
        traced=_traced_chain(["loss_output"]), parameters=["w", "b"]
    )

    assert first.seed_value_id == second.seed_value_id
    assert _program_snapshot(first.program) == _program_snapshot(second.program)


def test_two_really_traced_compilations_mint_the_same_seed() -> None:
    first_traced = _trace_linear()
    second_traced = _trace_linear()

    first = _differentiate_loss(traced=first_traced, parameters=["w"])
    second = _differentiate_loss(traced=second_traced, parameters=["w"])

    assert first.seed_value_id == second.seed_value_id
    assert _program_snapshot(first.program) == _program_snapshot(second.program)


def test_the_stage_holds_no_state_between_calls_that_shifts_the_seed() -> None:
    """Inv-12 read narrowly: minting is a function of the graph, not of history.

    A minter memoizing "already handed out" ids across calls would advance
    the seed on the second identical call and fail here.
    """
    traced = _traced_chain(["loss_output"])
    seeds = {
        _differentiate_loss(traced=traced, parameters=["w"]).seed_value_id
        for _ in range(3)
    }

    assert len(seeds) == 1


# --------------------------------------------------------------------------
# Edge cases from the Test Plan.
# --------------------------------------------------------------------------


def test_a_graph_whose_only_node_is_the_loss() -> None:
    traced = _traced_chain(["loss_output"])

    result = _differentiate_loss(traced=traced, parameters=["w"])

    assert result.seed_value_id not in _occupied_value_ids(traced.graph)
    assert tuple(result.program.metadata.wrt_signature) == ("pa",)


def test_a_parameter_that_is_also_the_loss_value() -> None:
    """The gradient of the loss with respect to itself is the seed.

    The derivative program produces no nodes at all here, so the
    post-generation check has an empty produced set -- it must accept that,
    not stumble on it, even though the seed *is* the reported gradient.
    """
    graph = TensorGraph(
        nodes=_mul_chain(["intermediate"]).nodes,
        inputs=[("pa", dict(SPEC)), ("pb", dict(SPEC))],
        outputs=["pa"],
    )
    traced = TracedLoss(
        graph=graph, loss_value_id="pa", input_value_ids={"w": "pa", "b": "pb"}
    )

    result = _differentiate_loss(traced=traced, parameters=["w"])

    assert result.program.nodes == []
    assert result.program.gradients == {"pa": result.seed_value_id}
    assert result.seed_value_id not in _occupied_value_ids(graph)


def test_a_symbolic_shaped_loss_output() -> None:
    traced = _traced_chain(["loss_output"], spec=SYMBOLIC_SPEC)

    result = _differentiate_loss(traced=traced, parameters=["w"])

    assert result.seed_value_id not in _occupied_value_ids(traced.graph)
    assert result.program.value_typespecs[result.seed_value_id] == dict(SYMBOLIC_SPEC)
