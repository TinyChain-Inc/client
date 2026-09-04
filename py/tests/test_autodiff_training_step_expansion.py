"""Unit tests for the expansion stages and the preservation check.

Pins the contract of FR-129-009, FR-129-020, §8.6, Inv-10, and Inv-11: the
caller's three expansion sequences are applied strictly after differentiation,
every pass result is validated before the next pass runs, and the derivative
dependency analysis is recomputed against the lowered artifacts and required to
equal the source analysis exactly.

Five rules govern how these tests are written:

* **Inv-10 is an identity claim.** With an empty sequence the lowered artifact
  must *be* the source artifact, so every inert-case assertion is written with
  `is`, never `==`. An implementation that copied an untouched artifact would
  satisfy equality and violate the invariant.
* **The per-pass check and the recomputation are tested apart.** A pass that
  breaks an artifact structurally must be caught before the next pass sees it;
  a pass that stays structurally valid while changing *which* forward values
  the derivative reads can only be caught by the recomputation. The
  capture-rename and dead-seed passes below are structurally valid by
  construction -- they mint node ids absent from their input and leave every
  required semantic value in place -- so they fail only if the recomputation
  exists.
* **Identifiers are read from the artifacts, never hard-coded.** Every
  expectation names the loss, the capture, the seed, or the gradient through
  the source records that report them, so these tests pin the rule rather than
  one tracer's numbering.
* **"Later pass uninvoked" is asserted with a counter**, not by reading the
  message. A message naming the second pass is consistent with a third pass
  having run and its result discarded.
* **The source artifacts are compared before and after every case.** The
  `source` fixture deep-copies both source artifacts on entry and re-compares
  them on exit, so FR-129-004 is checked in every case in this file, the
  failing ones included, rather than in one dedicated test.

Two facts about the fixture are load-bearing and were read from the real
stages rather than assumed: the residual loss captures exactly one forward
value (the residual), and its derivative program reads that capture, one
declared input, and the minted seed. Both are asserted in
`_require_fixture_shape`, which the fixture builder calls, so a change in the
tracer that invalidates the rest of the file fails in one obvious place.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Optional

import pytest
import tinychain as tc
from tinychain.autodiff.graph import (
    MulOperator,
    TensorGraph,
    TensorNodeRecord,
)
from tinychain.autodiff.protocol import AutodiffError
from tinychain.autodiff.reverse import DerivativeProgram
from tinychain.autodiff import training_step
from tinychain.autodiff.training_step import (
    SourceCaptures,
    SourceDerivative,
    TracedLoss,
    analyze_source_captures,
    differentiate_loss,
    trace_loss,
)


# --------------------------------------------------------------------------
# the stages under test, resolved at call time
#
# Resolved through the module rather than imported by name so each case fails
# on its own line with the missing attribute it needs, instead of every case
# in the file collapsing into one collection-time ImportError.
# --------------------------------------------------------------------------


def _expand_source_artifacts(**kwargs: object) -> object:
    return training_step.expand_source_artifacts(**kwargs)


def _expand_update_graph(**kwargs: object) -> object:
    return training_step.expand_update_graph(**kwargs)


# --------------------------------------------------------------------------
# source fixtures, built through the real stages
# --------------------------------------------------------------------------


RESIDUAL_INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": {"dtype": "f32", "shape": (2, 3)},
    "w": {"dtype": "f32", "shape": (3, 4)},
    "y": {"dtype": "f32", "shape": (2, 4)},
}


def _residual_loss(*, x: object, w: object, y: object) -> object:
    """The §17.3.1 loss: the multiply's VJP reads its own operand."""
    residual = x @ w - y
    return (residual * residual).mean([0, 1])


@dataclasses.dataclass(frozen=True)
class _Source:
    """One traced loss, its source derivative, and the source capture set."""

    traced: TracedLoss
    derivative: SourceDerivative
    captures: SourceCaptures

    @property
    def graph(self) -> TensorGraph:
        return self.traced.graph

    @property
    def program(self) -> DerivativeProgram:
        return self.derivative.program

    @property
    def seed_value_id(self) -> str:
        return self.derivative.seed_value_id

    @property
    def loss_value_id(self) -> str:
        return self.traced.loss_value_id

    @property
    def capture_value_id(self) -> str:
        return self.captures.forward_capture_value_ids[0]

    @property
    def gradient_value_id(self) -> str:
        return self.program.output_gradients[0]


def _build_source(parameters: Sequence[str]) -> _Source:
    with tc.state.scoped_context():
        traced = trace_loss(
            inputs=RESIDUAL_INPUTS, input_names=("x", "w", "y"), loss=_residual_loss
        )
    derivative = differentiate_loss(traced=traced, parameters=tuple(parameters))
    captures = analyze_source_captures(traced=traced, derivative=derivative)
    source = _Source(traced=traced, derivative=derivative, captures=captures)
    _require_fixture_shape(source)
    return source


def _require_fixture_shape(source: _Source) -> None:
    """The upstream facts the rest of this file is written against.

    Checked in the fixture rather than in a test of its own: these assertions
    hold against the already-delivered stages, so as a test they would pass
    before this subtask's stage exists and would not belong to the RED set.
    Checked once per fixture build, a tracer change that invalidates the file
    still fails loudly and in one place.
    """
    assert len(source.captures.forward_capture_value_ids) == 1, (
        "fixture defect: the residual loss must capture exactly one forward value"
    )
    assert source.loss_value_id not in source.captures.forward_capture_value_ids
    assert [
        dependency.value_id for dependency in source.captures.analysis.seed_inputs
    ] == [source.seed_value_id], "fixture defect: the seed must be a required input"
    assert any(
        source.capture_value_id in node.input_value_ids for node in source.program.nodes
    ), "fixture defect: the derivative must read the capture"
    assert any(
        source.seed_value_id in node.input_value_ids for node in source.program.nodes
    ), "fixture defect: the derivative must read the seed"


@pytest.fixture
def source() -> _Source:
    """A one-parameter source triple, guarded against mutation (FR-129-004).

    The guard runs for every case in this file, including the ones that fail
    inside a pass: a pass handed the framework's own source artifact could
    corrupt it in place and no per-test assertion would notice.
    """
    artifacts = _build_source(("w",))
    graph_before = copy.deepcopy(artifacts.graph)
    program_before = copy.deepcopy(artifacts.program)
    yield artifacts
    assert artifacts.graph == graph_before, "the source forward graph was mutated"
    assert artifacts.program == program_before, (
        "the source derivative program was mutated"
    )


@pytest.fixture
def two_parameter_source() -> _Source:
    """A two-parameter source triple, for the gradient-order rule."""
    artifacts = _build_source(("x", "w"))
    graph_before = copy.deepcopy(artifacts.graph)
    program_before = copy.deepcopy(artifacts.program)
    yield artifacts
    assert artifacts.graph == graph_before, "the source forward graph was mutated"
    assert artifacts.program == program_before, (
        "the source derivative program was mutated"
    )


# --------------------------------------------------------------------------
# artifact edit helpers
#
# Every helper returns a new artifact; none mutates its argument. A pass that
# deliberately mutates in place is written out explicitly below so that the
# one case doing it is impossible to miss.
# --------------------------------------------------------------------------


def _node_producing(nodes: Sequence[TensorNodeRecord], value_id: str) -> TensorNodeRecord:
    for node in nodes:
        if node.output_value_id == value_id:
            return node
    raise AssertionError(f"fixture defect: no node produces {value_id!r}")


def _without_nodes(
    nodes: Sequence[TensorNodeRecord], node_ids: Sequence[str]
) -> list[TensorNodeRecord]:
    excluded = set(node_ids)
    return [node for node in nodes if node.node_id not in excluded]


def _new_node(
    *,
    node_id: str,
    output_value_id: str,
    input_value_ids: Sequence[str],
    output_typespec: Optional[dict],
) -> TensorNodeRecord:
    """A structurally well-formed multiply node the fixtures can append."""
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=MulOperator(),
        op_params={},
        input_value_ids=list(input_value_ids),
        output_typespec=None if output_typespec is None else dict(output_typespec),
    )


def _typespec_of(nodes: Sequence[TensorNodeRecord], value_id: str) -> Optional[dict]:
    return _node_producing(nodes, value_id).output_typespec


def _non_capture_forward_value(source: _Source) -> str:
    """A forward value that is neither the loss nor a source capture.

    Chosen with the same typespec as the capture so the rewrite that swaps one
    for the other stays a plausible artifact rather than an obviously broken
    one.
    """
    capture_typespec = _typespec_of(source.graph.nodes, source.capture_value_id)
    for node in source.graph.nodes:
        if node.output_value_id in (source.capture_value_id, source.loss_value_id):
            continue
        if node.output_typespec == capture_typespec:
            return node.output_value_id
    raise AssertionError("fixture defect: no comparable non-capture forward value")


# --------------------------------------------------------------------------
# passes: labels
#
# Defined at module level so every `__qualname__` is stable and predictable.
# --------------------------------------------------------------------------


def module_level_pass(artifact: object) -> object:
    """A plain function pass: its own `__qualname__` is the label."""
    return artifact


def returns_none(artifact: object) -> object:
    return None


def returns_a_string(artifact: object) -> object:
    return "not an artifact"


def returns_a_graph(program: DerivativeProgram) -> object:
    return TensorGraph(nodes=[], inputs=[], outputs=[])


def returns_a_program(graph: TensorGraph) -> object:
    return DerivativeProgram(
        nodes=[],
        gradients={},
        output_gradients=[],
        metadata=None,
        value_typespecs={},
    )


def raises_runtime_error(artifact: object) -> object:
    raise RuntimeError("the pass exploded")


def raises_autodiff_error(artifact: object) -> object:
    raise AutodiffError("unsupported_operator", "the pass declined this artifact")


def raises_keyboard_interrupt(artifact: object) -> object:
    raise KeyboardInterrupt()


def raises_system_exit(artifact: object) -> object:
    raise SystemExit(3)


class _CountingPass:
    """Records how often it ran and applies an optional rewrite.

    An instance carries no `__qualname__` of its own, so it exercises §9.1's
    type-name fallback everywhere it is used.
    """

    def __init__(self, rewrite: object = None) -> None:
        self.calls = 0
        self._rewrite = rewrite

    def __call__(self, artifact: object) -> object:
        self.calls += 1
        if self._rewrite is None:
            return artifact
        return self._rewrite(artifact)


class _EmptyQualnamePass:
    """A pass whose `__qualname__` is present but empty (§9.1 fallback)."""

    def __init__(self) -> None:
        self.__qualname__ = ""

    def __call__(self, artifact: object) -> object:
        return artifact


class _NonStringQualnamePass:
    """A pass whose `__qualname__` is not a string (§9.1 fallback)."""

    def __init__(self) -> None:
        self.__qualname__ = 42

    def __call__(self, artifact: object) -> object:
        return artifact


class _RelabellingPass:
    """A pass that rewrites its own label while running, then fails.

    §9.1 requires the label to be looked up **before** invocation, so the
    reported label must be the type-name fallback this instance had on entry,
    never the one it assigns to itself mid-call.
    """

    def __call__(self, artifact: object) -> object:
        self.__qualname__ = "label_assigned_during_the_call"
        raise RuntimeError("the pass exploded after relabelling itself")


class _FailingLabelledPass:
    """A named pass that fails, used twice to prove position disambiguates."""

    def __call__(self, artifact: object) -> object:
        raise RuntimeError("the pass exploded")


class _PassingLabelledPass:
    """A named pass that succeeds, used twice at two positions."""

    def __call__(self, artifact: object) -> object:
        return artifact


# --------------------------------------------------------------------------
# passes: forward rewrites
# --------------------------------------------------------------------------


def _drop_first_declared_input(graph: TensorGraph) -> TensorGraph:
    return dataclasses.replace(graph, inputs=list(graph.inputs[1:]))


def _drop_loss_producer(graph: TensorGraph, loss_value_id: str) -> TensorGraph:
    producer = _node_producing(graph.nodes, loss_value_id)
    return dataclasses.replace(graph, nodes=_without_nodes(graph.nodes, [producer.node_id]))


def _drop_capture_producer(graph: TensorGraph, capture_value_id: str) -> TensorGraph:
    producer = _node_producing(graph.nodes, capture_value_id)
    return dataclasses.replace(graph, nodes=_without_nodes(graph.nodes, [producer.node_id]))


def _empty_graph(graph: TensorGraph) -> TensorGraph:
    return TensorGraph(nodes=[], inputs=[], outputs=[])


def _duplicate_node_id(graph: TensorGraph) -> TensorGraph:
    first = graph.nodes[0]
    duplicate = _new_node(
        node_id=first.node_id,
        output_value_id="exv_new",
        input_value_ids=list(first.input_value_ids),
        output_typespec=first.output_typespec,
    )
    return dataclasses.replace(graph, nodes=[*graph.nodes, duplicate])


def _mint_one_node_id_twice(graph: TensorGraph) -> TensorGraph:
    """Mint the same new node id twice, for two different computations.

    Neither node id occurs in the pass input, so the semantic-identity rule
    has nothing to compare them against, and the two outputs are distinct
    values, so the value-uniqueness rule is satisfied too. Only the node-id
    uniqueness rule can reject this.
    """
    first = graph.nodes[0]
    return dataclasses.replace(
        graph,
        nodes=[
            *graph.nodes,
            _new_node(
                node_id="exn_twice",
                output_value_id="exv_first",
                input_value_ids=[first.output_value_id],
                output_typespec=first.output_typespec,
            ),
            _new_node(
                node_id="exn_twice",
                output_value_id="exv_second",
                input_value_ids=list(first.input_value_ids),
                output_typespec=first.output_typespec,
            ),
        ],
    )


def appends_a_non_node(artifact: object) -> object:
    """Return an artifact of the right type whose node list holds a non-node."""
    return dataclasses.replace(artifact, nodes=[*artifact.nodes, "junk"])


def appends_a_malformed_input(graph: TensorGraph) -> TensorGraph:
    """Return a graph whose declared inputs hold something that is not a pair."""
    return dataclasses.replace(graph, inputs=[*graph.inputs, "notatuple"])


def returns_a_non_sequence_gradient_list(program: DerivativeProgram) -> DerivativeProgram:
    """A program whose gradient list is not a sequence at all."""
    return dataclasses.replace(program, output_gradients=5)


def returns_a_non_string_gradient(program: DerivativeProgram) -> DerivativeProgram:
    """A program whose gradient list holds something that is not an identifier."""
    return dataclasses.replace(program, output_gradients=[123])


def returns_a_non_mapping_typespec_table(
    program: DerivativeProgram,
) -> DerivativeProgram:
    """A program whose value typespec table is not a mapping."""
    return dataclasses.replace(program, value_typespecs=5)


def returns_a_sequence_typespec_table(program: DerivativeProgram) -> DerivativeProgram:
    """The same defect in the shape that reads most like a plausible mistake."""
    return dataclasses.replace(program, value_typespecs=["ghost_value"])


def _duplicate_value_id(graph: TensorGraph) -> TensorGraph:
    first = graph.nodes[0]
    duplicate = _new_node(
        node_id="exn_new",
        output_value_id=first.output_value_id,
        input_value_ids=list(first.input_value_ids),
        output_typespec=first.output_typespec,
    )
    return dataclasses.replace(graph, nodes=[*graph.nodes, duplicate])


def _produce_a_declared_input(graph: TensorGraph, input_value_id: str) -> TensorGraph:
    minted = _new_node(
        node_id="exn_new",
        output_value_id=input_value_id,
        input_value_ids=[graph.nodes[0].output_value_id],
        output_typespec=graph.nodes[0].output_typespec,
    )
    return dataclasses.replace(graph, nodes=[*graph.nodes, minted])


def _reassign_a_node_id(graph: TensorGraph) -> TensorGraph:
    first = graph.nodes[0]
    reassigned = _new_node(
        node_id=first.node_id,
        output_value_id=first.output_value_id,
        input_value_ids=list(reversed(first.input_value_ids)),
        output_typespec=first.output_typespec,
    )
    return dataclasses.replace(graph, nodes=[reassigned, *graph.nodes[1:]])


def _turn_a_produced_value_into_a_declared_input(graph: TensorGraph) -> TensorGraph:
    """Drop the first node and declare its output as a free input instead.

    Every semantic value the forward artifact must still provide is left
    available, so only the value-identity rule rejects this: a value the graph
    computed has become a value the caller must bind.
    """
    first = graph.nodes[0]
    return dataclasses.replace(
        graph,
        nodes=list(graph.nodes[1:]),
        inputs=[*graph.inputs, (first.output_value_id, first.output_typespec)],
    )


def _reproduce_the_loss_under_a_new_node_id(
    graph: TensorGraph, loss_value_id: str
) -> TensorGraph:
    """Carry the loss terminal id forward under a freshly minted node id.

    This is the shape a real expansion pass has -- issue #128's mean expansion
    replaces the mean node with a region whose terminal node carries the mean's
    own output value id -- so it must be accepted.
    """
    producer = _node_producing(graph.nodes, loss_value_id)
    replacement = dataclasses.replace(producer, node_id="exn_terminal")
    return dataclasses.replace(
        graph,
        nodes=[
            replacement if node.node_id == producer.node_id else node
            for node in graph.nodes
        ],
    )


def _append_a_new_node(graph: TensorGraph) -> TensorGraph:
    first = graph.nodes[0]
    appended = _new_node(
        node_id="exn_new",
        output_value_id="exv_new",
        input_value_ids=[first.output_value_id],
        output_typespec=first.output_typespec,
    )
    return dataclasses.replace(graph, nodes=[*graph.nodes, appended])


def _mutate_in_place_and_return(graph: TensorGraph) -> TensorGraph:
    """Append to the artifact's own node list and hand the same object back."""
    first = graph.nodes[0]
    graph.nodes.append(
        _new_node(
            node_id="exn_in_place",
            output_value_id="exv_in_place",
            input_value_ids=[first.output_value_id],
            output_typespec=first.output_typespec,
        )
    )
    return graph


# --------------------------------------------------------------------------
# passes: derivative rewrites
# --------------------------------------------------------------------------


def _drop_gradient_producer(
    program: DerivativeProgram, gradient_value_id: str
) -> DerivativeProgram:
    producer = _node_producing(program.nodes, gradient_value_id)
    return dataclasses.replace(
        program, nodes=_without_nodes(program.nodes, [producer.node_id])
    )


def _reverse_output_gradients(program: DerivativeProgram) -> DerivativeProgram:
    return dataclasses.replace(
        program, output_gradients=list(reversed(program.output_gradients))
    )


def _drop_the_seed(program: DerivativeProgram, seed_value_id: str) -> DerivativeProgram:
    """Replace every node reading the seed with a node that reads nothing.

    The replacements carry freshly minted node ids and reproduce the same
    output value ids, so the artifact stays structurally sound in every respect
    except the one under test: the seed is no longer a required free input.
    """
    rewritten: list[TensorNodeRecord] = []
    for index, node in enumerate(program.nodes):
        if seed_value_id in node.input_value_ids:
            rewritten.append(
                _new_node(
                    node_id=f"exn_seedless{index}",
                    output_value_id=node.output_value_id,
                    input_value_ids=[
                        value_id
                        for value_id in node.input_value_ids
                        if value_id != seed_value_id
                    ],
                    output_typespec=node.output_typespec,
                )
            )
        else:
            rewritten.append(node)
    return dataclasses.replace(program, nodes=rewritten)


def _produce_the_seed(
    program: DerivativeProgram, seed_value_id: str
) -> DerivativeProgram:
    minted = _new_node(
        node_id="exn_new",
        output_value_id=seed_value_id,
        input_value_ids=[program.nodes[0].output_value_id],
        output_typespec=program.nodes[0].output_typespec,
    )
    return dataclasses.replace(program, nodes=[*program.nodes, minted])


def _use_the_seed_as_a_node_id(
    program: DerivativeProgram, seed_value_id: str
) -> DerivativeProgram:
    minted = _new_node(
        node_id=seed_value_id,
        output_value_id="exv_new",
        input_value_ids=[program.nodes[0].output_value_id],
        output_typespec=program.nodes[0].output_typespec,
    )
    return dataclasses.replace(program, nodes=[*program.nodes, minted])


def _read_a_different_forward_value(
    program: DerivativeProgram, capture_value_id: str, replacement_value_id: str
) -> DerivativeProgram:
    """Read another forward value in place of the capture, structurally validly.

    Every rewritten node is emitted under a freshly minted node id and
    reproduces the value id it replaced, so no occupied id is reassigned, no id
    is duplicated, every gradient is still produced in order, and the seed is
    untouched. The only thing that changed is *which* forward value the
    derivative needs -- which is exactly what the recomputation exists to
    catch.
    """
    rewritten: list[TensorNodeRecord] = []
    for index, node in enumerate(program.nodes):
        if capture_value_id in node.input_value_ids:
            rewritten.append(
                _new_node(
                    node_id=f"exn_renamed{index}",
                    output_value_id=node.output_value_id,
                    input_value_ids=[
                        replacement_value_id if value_id == capture_value_id else value_id
                        for value_id in node.input_value_ids
                    ],
                    output_typespec=node.output_typespec,
                )
            )
        else:
            rewritten.append(node)
    return dataclasses.replace(program, nodes=rewritten)


def _make_the_seed_unreachable(
    program: DerivativeProgram, seed_value_id: str, capture_value_id: str
) -> DerivativeProgram:
    """Keep the seed mentioned, but only by a node nothing selects.

    Structurally the seed is still a free input the program mentions and never
    produces. Semantically it no longer reaches any gradient, so the recomputed
    seed-input set is empty while the source's was not.
    """
    rewritten: list[TensorNodeRecord] = []
    for index, node in enumerate(program.nodes):
        if seed_value_id in node.input_value_ids:
            rewritten.append(
                _new_node(
                    node_id=f"exn_detached{index}",
                    output_value_id=node.output_value_id,
                    input_value_ids=[
                        capture_value_id if value_id == seed_value_id else value_id
                        for value_id in node.input_value_ids
                    ],
                    output_typespec=node.output_typespec,
                )
            )
        else:
            rewritten.append(node)
    rewritten.append(
        _new_node(
            node_id="exn_dead",
            output_value_id="exv_dead",
            input_value_ids=[seed_value_id],
            output_typespec=program.nodes[0].output_typespec,
        )
    )
    return dataclasses.replace(program, nodes=rewritten)


# --------------------------------------------------------------------------
# the update artifact
#
# Built by hand: the per-parameter update trace that feeds this stage belongs
# to the next subtask, and this stage's contract is stated over the artifact,
# not over how it was produced.
# --------------------------------------------------------------------------


UPDATE_SPEC: Mapping[str, object] = {"dtype": "f32", "shape": [3, 4]}


def _update_graph() -> TensorGraph:
    return TensorGraph(
        nodes=[
            _new_node(
                node_id="un0",
                output_value_id="u_scaled",
                input_value_ids=["u_gradient", "u_rate"],
                output_typespec=dict(UPDATE_SPEC),
            ),
            _new_node(
                node_id="un1",
                output_value_id="u_next",
                input_value_ids=["u_parameter", "u_scaled"],
                output_typespec=dict(UPDATE_SPEC),
            ),
        ],
        inputs=[
            ("u_parameter", dict(UPDATE_SPEC)),
            ("u_gradient", dict(UPDATE_SPEC)),
            ("u_rate", dict(UPDATE_SPEC)),
        ],
        outputs=["u_next"],
    )


UPDATED_PARAMETER_VALUE_ID = "u_next"


def _expand_update(expansions: Sequence[object], graph: Optional[TensorGraph] = None) -> object:
    return _expand_update_graph(
        graph=_update_graph() if graph is None else graph,
        updated_parameter_value_id=UPDATED_PARAMETER_VALUE_ID,
        expansions=tuple(expansions),
    )


# --------------------------------------------------------------------------
# assertion helpers
# --------------------------------------------------------------------------


def _expand(
    source: _Source,
    *,
    forward_expansions: Sequence[object] = (),
    derivative_expansions: Sequence[object] = (),
) -> object:
    return _expand_source_artifacts(
        traced=source.traced,
        derivative=source.derivative,
        captures=source.captures,
        forward_expansions=tuple(forward_expansions),
        derivative_expansions=tuple(derivative_expansions),
    )


def _assert_violation(
    error: AutodiffError, *, label: str, position: int, mentions: Sequence[str] = ()
) -> None:
    assert error.category == "expansion_contract_violation"
    message = str(error)
    assert label in message, f"the message does not name the pass: {message}"
    assert f"position {position}" in message, (
        f"the message does not report the zero-based position: {message}"
    )
    for fragment in mentions:
        assert fragment in message, (
            f"the message does not name {fragment!r}: {message}"
        )


# --------------------------------------------------------------------------
# AC-1 -- inert by default (Inv-10)
# --------------------------------------------------------------------------


def test_expansion_with_no_passes_returns_the_source_forward_graph_itself(
    source: _Source,
) -> None:
    result = _expand(source)

    assert result.lowered_forward_graph is source.graph


def test_expansion_with_no_passes_returns_the_source_derivative_program_itself(
    source: _Source,
) -> None:
    result = _expand(source)

    assert result.lowered_derivative_program is source.program


def test_expansion_with_no_passes_records_no_pass_labels(source: _Source) -> None:
    result = _expand(source)

    assert result.forward_pass_labels == ()
    assert result.derivative_pass_labels == ()


def test_expansion_with_no_passes_recomputes_the_source_capture_and_seed_sets(
    source: _Source,
) -> None:
    result = _expand(source)

    assert result.analysis.forward_captures == source.captures.analysis.forward_captures
    assert result.analysis.seed_inputs == source.captures.analysis.seed_inputs


def test_update_expansion_with_no_passes_returns_the_source_graph_itself() -> None:
    graph = _update_graph()

    result = _expand_update_graph(
        graph=graph,
        updated_parameter_value_id=UPDATED_PARAMETER_VALUE_ID,
        expansions=(),
    )

    assert result.lowered_graph is graph
    assert result.pass_labels == ()


# --------------------------------------------------------------------------
# AC-2 -- §13.2 for a pass: wrong type, wrapped, propagated, never wrapped
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expansion, label",
    [
        (returns_none, "returns_none"),
        (returns_a_string, "returns_a_string"),
        (returns_a_program, "returns_a_program"),
    ],
)
def test_forward_pass_returning_the_wrong_type_fails_naming_the_pass_and_position(
    source: _Source, expansion: object, label: str
) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[module_level_pass, expansion])

    _assert_violation(excinfo.value, label=label, position=1)


@pytest.mark.parametrize(
    "expansion, label",
    [
        (returns_none, "returns_none"),
        (returns_a_graph, "returns_a_graph"),
    ],
)
def test_derivative_pass_returning_the_wrong_type_fails_naming_the_pass_and_position(
    source: _Source, expansion: object, label: str
) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[expansion])

    _assert_violation(excinfo.value, label=label, position=0)


def test_update_pass_returning_the_wrong_type_fails_naming_the_pass_and_position() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand_update([module_level_pass, module_level_pass, returns_none])

    _assert_violation(excinfo.value, label="returns_none", position=2)


def test_forward_pass_raising_a_plain_exception_is_wrapped_naming_the_pass_and_position(
    source: _Source,
) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[raises_runtime_error])

    _assert_violation(excinfo.value, label="raises_runtime_error", position=0)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_derivative_pass_raising_a_plain_exception_is_wrapped_naming_the_pass_and_position(
    source: _Source,
) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(
            source,
            derivative_expansions=[module_level_pass, raises_runtime_error],
        )

    _assert_violation(excinfo.value, label="raises_runtime_error", position=1)


def test_forward_pass_raising_an_autodiff_error_propagates_it_unchanged(
    source: _Source,
) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[raises_autodiff_error])

    assert excinfo.value.category == "unsupported_operator"
    assert excinfo.value.message == "the pass declined this artifact"


def test_derivative_pass_raising_an_autodiff_error_propagates_it_unchanged(
    source: _Source,
) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[raises_autodiff_error])

    assert excinfo.value.category == "unsupported_operator"


def test_pass_raising_keyboard_interrupt_is_never_wrapped(source: _Source) -> None:
    with pytest.raises(KeyboardInterrupt):
        _expand(source, forward_expansions=[raises_keyboard_interrupt])


def test_pass_raising_system_exit_is_never_wrapped(source: _Source) -> None:
    with pytest.raises(SystemExit):
        _expand(source, derivative_expansions=[raises_system_exit])


# --------------------------------------------------------------------------
# AC-3 -- a dropped semantic value fails immediately, naming the value
# --------------------------------------------------------------------------


def test_forward_pass_dropping_a_declared_input_fails_naming_the_input(
    source: _Source,
) -> None:
    dropped_value_id = source.graph.inputs[0][0]

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[_drop_first_declared_input])

    _assert_violation(
        excinfo.value,
        label="_drop_first_declared_input",
        position=0,
        mentions=[dropped_value_id],
    )


def test_forward_pass_dropping_the_loss_fails_naming_the_loss(source: _Source) -> None:
    def drop_the_loss(graph: TensorGraph) -> TensorGraph:
        return _drop_loss_producer(graph, source.loss_value_id)

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[drop_the_loss])

    assert excinfo.value.category == "expansion_contract_violation"
    assert source.loss_value_id in str(excinfo.value)
    assert "position 0" in str(excinfo.value)


def test_forward_pass_dropping_a_capture_fails_naming_the_capture(
    source: _Source,
) -> None:
    def drop_the_capture(graph: TensorGraph) -> TensorGraph:
        return _drop_capture_producer(graph, source.capture_value_id)

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[drop_the_capture])

    assert excinfo.value.category == "expansion_contract_violation"
    assert source.capture_value_id in str(excinfo.value)


def test_forward_pass_returning_an_empty_graph_fails_naming_a_required_value(
    source: _Source,
) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[_empty_graph])

    message = str(excinfo.value)
    assert excinfo.value.category == "expansion_contract_violation"
    assert any(
        required in message
        for required in (
            source.loss_value_id,
            source.capture_value_id,
            source.graph.inputs[0][0],
        )
    ), message


def test_derivative_pass_dropping_a_gradient_fails_naming_the_gradient(
    source: _Source,
) -> None:
    def drop_the_gradient(program: DerivativeProgram) -> DerivativeProgram:
        return _drop_gradient_producer(program, source.gradient_value_id)

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[drop_the_gradient])

    assert excinfo.value.category == "expansion_contract_violation"
    assert source.gradient_value_id in str(excinfo.value)


def test_derivative_pass_reordering_the_gradients_fails_naming_a_gradient(
    two_parameter_source: _Source,
) -> None:
    source = two_parameter_source
    assert len(source.program.output_gradients) == 2, "fixture defect"

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[_reverse_output_gradients])

    _assert_violation(
        excinfo.value,
        label="_reverse_output_gradients",
        position=0,
        mentions=[str(source.program.output_gradients[0])],
    )


def test_derivative_pass_dropping_the_seed_fails_naming_the_seed(
    source: _Source,
) -> None:
    def drop_the_seed(program: DerivativeProgram) -> DerivativeProgram:
        return _drop_the_seed(program, source.seed_value_id)

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[drop_the_seed])

    assert excinfo.value.category == "expansion_contract_violation"
    assert source.seed_value_id in str(excinfo.value)


def test_derivative_pass_dropping_the_seed_leaves_the_next_pass_uninvoked(
    source: _Source,
) -> None:
    """The per-pass seed rule, isolated from the recomputation that follows it.

    A dropped seed is also visible to the final recomputation, so the failure
    alone does not prove the per-pass check ran. What only the per-pass check
    can do is stop the sequence: the pass after the offender must never see the
    broken artifact.
    """

    def drop_the_seed(program: DerivativeProgram) -> DerivativeProgram:
        return _drop_the_seed(program, source.seed_value_id)

    later = _CountingPass()

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[drop_the_seed, later])

    assert excinfo.value.category == "expansion_contract_violation"
    assert source.seed_value_id in str(excinfo.value)
    assert later.calls == 0


def test_update_pass_dropping_a_declared_input_fails_naming_the_input() -> None:
    def drop_the_first_input(graph: TensorGraph) -> TensorGraph:
        return _drop_first_declared_input(graph)

    with pytest.raises(AutodiffError) as excinfo:
        _expand_update([drop_the_first_input])

    assert excinfo.value.category == "expansion_contract_violation"
    assert "u_parameter" in str(excinfo.value)


def test_update_pass_dropping_the_updated_parameter_fails_naming_the_value() -> None:
    def drop_the_updated_parameter(graph: TensorGraph) -> TensorGraph:
        return dataclasses.replace(graph, nodes=_without_nodes(graph.nodes, ["un1"]))

    with pytest.raises(AutodiffError) as excinfo:
        _expand_update([drop_the_updated_parameter])

    assert excinfo.value.category == "expansion_contract_violation"
    assert UPDATED_PARAMETER_VALUE_ID in str(excinfo.value)


# --------------------------------------------------------------------------
# AC-4 -- id uniqueness and semantic identity
# --------------------------------------------------------------------------


def test_forward_pass_duplicating_a_node_id_fails_naming_the_id(
    source: _Source,
) -> None:
    duplicated = source.graph.nodes[0].node_id

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[_duplicate_node_id])

    _assert_violation(
        excinfo.value,
        label="_duplicate_node_id",
        position=0,
        mentions=[duplicated],
    )


def test_forward_pass_duplicating_a_value_id_fails_naming_the_id(
    source: _Source,
) -> None:
    duplicated = source.graph.nodes[0].output_value_id

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[_duplicate_value_id])

    _assert_violation(
        excinfo.value,
        label="_duplicate_value_id",
        position=0,
        mentions=[duplicated],
    )


def test_forward_pass_minting_an_occupied_value_id_fails_naming_the_id(
    source: _Source,
) -> None:
    occupied = source.graph.inputs[0][0]

    def produce_a_declared_input(graph: TensorGraph) -> TensorGraph:
        return _produce_a_declared_input(graph, occupied)

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[produce_a_declared_input])

    assert excinfo.value.category == "expansion_contract_violation"
    assert occupied in str(excinfo.value)


def test_forward_pass_reassigning_a_node_id_fails_naming_the_id(
    source: _Source,
) -> None:
    reassigned = source.graph.nodes[0].node_id

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[_reassign_a_node_id])

    _assert_violation(
        excinfo.value,
        label="_reassign_a_node_id",
        position=0,
        mentions=[reassigned],
    )


def test_forward_pass_turning_a_produced_value_into_a_declared_input_fails_naming_the_id(
    source: _Source,
) -> None:
    reassigned = source.graph.nodes[0].output_value_id

    with pytest.raises(AutodiffError) as excinfo:
        _expand(
            source, forward_expansions=[_turn_a_produced_value_into_a_declared_input]
        )

    _assert_violation(
        excinfo.value,
        label="_turn_a_produced_value_into_a_declared_input",
        position=0,
        mentions=[reassigned],
    )


def test_derivative_pass_producing_the_seed_fails_naming_the_seed(
    source: _Source,
) -> None:
    def produce_the_seed(program: DerivativeProgram) -> DerivativeProgram:
        return _produce_the_seed(program, source.seed_value_id)

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[produce_the_seed])

    assert excinfo.value.category == "expansion_contract_violation"
    assert source.seed_value_id in str(excinfo.value)


def test_derivative_pass_using_the_seed_as_a_node_id_fails_naming_the_seed(
    source: _Source,
) -> None:
    def use_the_seed_as_a_node_id(program: DerivativeProgram) -> DerivativeProgram:
        return _use_the_seed_as_a_node_id(program, source.seed_value_id)

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[use_the_seed_as_a_node_id])

    assert excinfo.value.category == "expansion_contract_violation"
    assert source.seed_value_id in str(excinfo.value)


def test_second_pass_reassigning_an_id_the_first_pass_minted_fails_at_its_own_position(
    source: _Source,
) -> None:
    """Identity is judged against each pass's own input, not against the source.

    The id under test does not exist in the source artifact at all: it was
    minted by the pass before. A validator that compared every result with the
    source graph would accept the second pass's rewrite of it.
    """

    def reassign_the_minted_node(graph: TensorGraph) -> TensorGraph:
        minted = _node_producing(graph.nodes, "exv_new")
        replacement = _new_node(
            node_id=minted.node_id,
            output_value_id=minted.output_value_id,
            input_value_ids=list(reversed(graph.nodes[0].input_value_ids)),
            output_typespec=minted.output_typespec,
        )
        return dataclasses.replace(
            graph,
            nodes=[
                replacement if node.node_id == minted.node_id else node
                for node in graph.nodes
            ],
        )

    with pytest.raises(AutodiffError) as excinfo:
        _expand(
            source,
            forward_expansions=[_append_a_new_node, reassign_the_minted_node],
        )

    _assert_violation(
        excinfo.value,
        label="reassign_the_minted_node",
        position=1,
        mentions=["exn_new"],
    )


def test_forward_pass_carrying_a_terminal_id_forward_is_accepted(
    source: _Source,
) -> None:
    """A new node id producing the same terminal value is the legitimate case."""

    def reproduce_the_loss(graph: TensorGraph) -> TensorGraph:
        return _reproduce_the_loss_under_a_new_node_id(graph, source.loss_value_id)

    result = _expand(source, forward_expansions=[reproduce_the_loss])

    assert result.lowered_forward_graph is not source.graph
    assert _node_producing(
        result.lowered_forward_graph.nodes, source.loss_value_id
    ).node_id == "exn_terminal"
    assert result.analysis.forward_captures == source.captures.analysis.forward_captures


def test_forward_pass_appending_a_new_node_is_accepted(source: _Source) -> None:
    result = _expand(source, forward_expansions=[_append_a_new_node])

    assert "exv_new" in [
        node.output_value_id for node in result.lowered_forward_graph.nodes
    ]
    assert result.forward_pass_labels == ("_append_a_new_node",)


def test_pass_returning_its_input_unchanged_is_accepted(source: _Source) -> None:
    result = _expand(
        source,
        forward_expansions=[module_level_pass],
        derivative_expansions=[module_level_pass],
    )

    assert result.lowered_forward_graph == source.graph
    assert result.lowered_derivative_program == source.program


def test_forward_pass_minting_one_node_id_twice_fails_naming_the_id(
    source: _Source,
) -> None:
    """Only the uniqueness rule can reject this one.

    The duplicated id is minted by the pass, so it is absent from the pass
    input and the semantic-identity rule never looks at it, and the two nodes
    produce different values, so value uniqueness is satisfied.
    """
    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[_mint_one_node_id_twice])

    _assert_violation(
        excinfo.value,
        label="_mint_one_node_id_twice",
        position=0,
        mentions=["exn_twice"],
    )


# --------------------------------------------------------------------------
# a malformed artifact of the right type (§8.6, FR-129-020)
#
# The type check accepts these -- they really are a `TensorGraph` and a
# `DerivativeProgram` -- and every rule after it reads their contents. An
# artifact whose node list or input list does not hold what those rules read
# must be reported as the pass's contract breach, with the same care §9.1
# takes over a malformed label, rather than escaping as a raw `AttributeError`
# or `ValueError` from inside a validator.
# --------------------------------------------------------------------------


def test_forward_pass_returning_a_graph_with_a_non_node_fails_naming_the_defect(
    source: _Source,
) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[appends_a_non_node])

    _assert_violation(
        excinfo.value, label="appends_a_non_node", position=0, mentions=["junk"]
    )


def test_forward_pass_returning_a_malformed_declared_input_fails_naming_the_defect(
    source: _Source,
) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(
            source, forward_expansions=[module_level_pass, appends_a_malformed_input]
        )

    _assert_violation(
        excinfo.value,
        label="appends_a_malformed_input",
        position=1,
        mentions=["notatuple"],
    )


def test_derivative_pass_returning_a_program_with_a_non_node_fails_naming_the_defect(
    source: _Source,
) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[appends_a_non_node])

    _assert_violation(
        excinfo.value, label="appends_a_non_node", position=0, mentions=["junk"]
    )


def test_derivative_pass_returning_a_non_sequence_gradient_list_fails_naming_the_defect(
    source: _Source,
) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[returns_a_non_sequence_gradient_list])

    _assert_violation(
        excinfo.value,
        label="returns_a_non_sequence_gradient_list",
        position=0,
        mentions=["int"],
    )


def test_derivative_pass_returning_a_non_string_gradient_fails_naming_the_entry(
    source: _Source,
) -> None:
    """Reported as the shape defect it is, naming the entry and its index.

    The gradient-order rule already rejects this artifact, because a non-string
    entry cannot equal any source gradient -- but it reports the whole expected
    gradient list rather than the offending entry, which is the wrong end of
    the diagnostic. The shape guard runs first and names what is actually
    wrong.
    """
    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[returns_a_non_string_gradient])

    _assert_violation(
        excinfo.value,
        label="returns_a_non_string_gradient",
        position=0,
        mentions=["123", "index 0"],
    )


@pytest.mark.parametrize(
    "expansion, label",
    [
        (returns_a_non_mapping_typespec_table, "returns_a_non_mapping_typespec_table"),
        (returns_a_sequence_typespec_table, "returns_a_sequence_typespec_table"),
    ],
)
def test_derivative_pass_returning_a_non_mapping_typespec_table_fails_naming_the_defect(
    source: _Source, expansion: object, label: str
) -> None:
    """The recomputation reads this table; a malformed one must not reach it.

    `analyze_derivative_dependencies` resolves metadata through
    `value_typespecs`, so a pass that replaces the table with something that is
    not a mapping fails inside a framework collaborator with a raw
    `AttributeError` unless this stage rejects the artifact first.
    """
    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[expansion])

    assert excinfo.value.category == "expansion_contract_violation"
    assert label in str(excinfo.value)
    assert "position 0" in str(excinfo.value)


def test_update_pass_returning_a_graph_with_a_non_node_fails_naming_the_defect() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand_update([appends_a_non_node])

    _assert_violation(
        excinfo.value, label="appends_a_non_node", position=0, mentions=["junk"]
    )


def test_update_pass_returning_a_malformed_declared_input_fails_naming_the_defect() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand_update([appends_a_malformed_input])

    _assert_violation(
        excinfo.value,
        label="appends_a_malformed_input",
        position=0,
        mentions=["notatuple"],
    )


# --------------------------------------------------------------------------
# AC-5 -- the recomputation catches a structurally valid rename
# --------------------------------------------------------------------------


def test_derivative_pass_reading_a_different_forward_value_fails_the_recomputation(
    source: _Source,
) -> None:
    """Structurally valid, semantically wrong: only the recomputation sees it."""
    replacement = _non_capture_forward_value(source)

    def read_a_different_forward_value(program: DerivativeProgram) -> DerivativeProgram:
        return _read_a_different_forward_value(
            program, source.capture_value_id, replacement
        )

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[read_a_different_forward_value])

    _assert_violation(
        excinfo.value,
        label="read_a_different_forward_value",
        position=0,
        mentions=[source.capture_value_id, replacement],
    )


def test_analysis_failure_during_the_recomputation_propagates_with_its_own_category(
    source: _Source,
) -> None:
    """The analysis owns its categories (§13.3); only an inequality is this stage's.

    The pass leaves an artifact this stage's structural rules accept -- unique
    ids, no reassignment, every gradient produced in order, the seed untouched
    -- but reads a value nothing produces. That is the dependency analysis
    reporting on the artifact, not a pass breaking the expansion contract, so
    the failure keeps `missing_dependency` rather than being re-categorized.
    """

    def read_an_unproduced_value(program: DerivativeProgram) -> DerivativeProgram:
        return _read_a_different_forward_value(
            program, source.capture_value_id, "ghost_value"
        )

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[read_an_unproduced_value])

    assert excinfo.value.category == "missing_dependency"
    assert "ghost_value" in str(excinfo.value)


def test_derivative_pass_detaching_the_seed_fails_the_recomputation(
    source: _Source,
) -> None:
    """The seed stays a free input the program mentions, but reaches no gradient."""

    def detach_the_seed(program: DerivativeProgram) -> DerivativeProgram:
        return _make_the_seed_unreachable(
            program, source.seed_value_id, source.capture_value_id
        )

    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, derivative_expansions=[detach_the_seed])

    _assert_violation(
        excinfo.value,
        label="detach_the_seed",
        position=0,
        mentions=[source.seed_value_id],
    )


# --------------------------------------------------------------------------
# AC-6 -- a failure stops the sequence
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failing_rewrite",
    [raises_runtime_error, returns_none, _duplicate_node_id, appends_a_non_node],
    ids=["raises", "wrong-type", "invalid-result", "malformed-artifact"],
)
def test_forward_pass_failing_at_position_one_leaves_position_two_uninvoked(
    source: _Source, failing_rewrite: object
) -> None:
    first = _CountingPass()
    second = _CountingPass(rewrite=failing_rewrite)
    third = _CountingPass()

    with pytest.raises(AutodiffError):
        _expand(source, forward_expansions=[first, second, third])

    assert first.calls == 1
    assert second.calls == 1
    assert third.calls == 0


def test_forward_failure_leaves_every_derivative_pass_uninvoked(
    source: _Source,
) -> None:
    derivative_pass = _CountingPass()

    with pytest.raises(AutodiffError):
        _expand(
            source,
            forward_expansions=[raises_runtime_error],
            derivative_expansions=[derivative_pass],
        )

    assert derivative_pass.calls == 0


def test_derivative_pass_failing_at_position_one_leaves_position_two_uninvoked(
    source: _Source,
) -> None:
    first = _CountingPass()
    second = _CountingPass(rewrite=returns_none)
    third = _CountingPass()

    with pytest.raises(AutodiffError):
        _expand(source, derivative_expansions=[first, second, third])

    assert first.calls == 1
    assert second.calls == 1
    assert third.calls == 0


def test_a_sequence_of_five_passes_runs_every_pass_once_in_order(
    source: _Source,
) -> None:
    applied: list[int] = []

    class _OrderedPass:
        def __init__(self, index: int) -> None:
            self.index = index

        def __call__(self, graph: TensorGraph) -> TensorGraph:
            applied.append(self.index)
            return graph

    passes = [_OrderedPass(index) for index in range(5)]

    result = _expand(source, forward_expansions=passes)

    assert applied == [0, 1, 2, 3, 4]
    assert len(result.forward_pass_labels) == 5


# --------------------------------------------------------------------------
# AC-7 -- the source artifacts survive a hostile pass
#
# The `source` fixture already re-compares both artifacts after every case in
# this file. This case states the strongest form directly: a pass that mutates
# the artifact it was handed must not be able to reach the source.
# --------------------------------------------------------------------------


def test_forward_pass_mutating_its_input_in_place_cannot_reach_the_source_graph(
    source: _Source,
) -> None:
    graph_before = copy.deepcopy(source.graph)

    result = _expand(source, forward_expansions=[_mutate_in_place_and_return])

    assert source.graph == graph_before
    assert "exv_in_place" in [
        node.output_value_id for node in result.lowered_forward_graph.nodes
    ]


def test_a_forward_pass_is_never_handed_the_source_graph_object(
    source: _Source,
) -> None:
    """FR-129-004 stated as the caller can observe it: the pass gets a copy."""
    received: list[TensorGraph] = []

    def record_the_argument(graph: TensorGraph) -> TensorGraph:
        received.append(graph)
        return graph

    _expand(source, forward_expansions=[record_the_argument])

    assert received[0] is not source.graph
    assert received[0] == source.graph


def test_a_derivative_pass_is_never_handed_the_source_program_object(
    source: _Source,
) -> None:
    received: list[DerivativeProgram] = []

    def record_the_argument(program: DerivativeProgram) -> DerivativeProgram:
        received.append(program)
        return program

    _expand(source, derivative_expansions=[record_the_argument])

    assert received[0] is not source.program
    assert received[0] == source.program


def test_source_artifacts_are_unmutated_after_a_failing_pass(source: _Source) -> None:
    graph_before = copy.deepcopy(source.graph)
    program_before = copy.deepcopy(source.program)

    with pytest.raises(AutodiffError):
        _expand(
            source,
            forward_expansions=[_append_a_new_node],
            derivative_expansions=[returns_none],
        )

    assert source.graph == graph_before
    assert source.program == program_before


# --------------------------------------------------------------------------
# AC-8 -- pass labels resolve exactly per §9.1
# --------------------------------------------------------------------------


def test_pass_label_for_a_plain_function_is_its_qualname(source: _Source) -> None:
    result = _expand(source, forward_expansions=[module_level_pass])

    assert result.forward_pass_labels == ("module_level_pass",)


def test_pass_label_for_a_callable_instance_falls_back_to_the_type_qualname(
    source: _Source,
) -> None:
    result = _expand(source, forward_expansions=[_CountingPass()])

    assert result.forward_pass_labels == ("_CountingPass",)


def test_pass_label_for_an_empty_qualname_falls_back_to_the_type_qualname(
    source: _Source,
) -> None:
    result = _expand(source, derivative_expansions=[_EmptyQualnamePass()])

    assert result.derivative_pass_labels == ("_EmptyQualnamePass",)


def test_pass_label_for_a_non_string_qualname_falls_back_to_the_type_qualname(
    source: _Source,
) -> None:
    result = _expand(source, forward_expansions=[_NonStringQualnamePass()])

    assert result.forward_pass_labels == ("_NonStringQualnamePass",)


def test_pass_label_is_resolved_before_the_pass_is_invoked(source: _Source) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(source, forward_expansions=[_RelabellingPass()])

    message = str(excinfo.value)
    assert "_RelabellingPass" in message
    assert "label_assigned_during_the_call" not in message


def test_a_failing_pass_after_a_successful_one_reports_its_own_position(
    source: _Source,
) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(
            source,
            forward_expansions=[_PassingLabelledPass(), _FailingLabelledPass()],
        )

    _assert_violation(excinfo.value, label="_FailingLabelledPass", position=1)


def test_the_first_of_two_same_labelled_passes_reports_position_zero(
    source: _Source,
) -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _expand(
            source,
            forward_expansions=[_FailingLabelledPass(), _FailingLabelledPass()],
        )

    _assert_violation(excinfo.value, label="_FailingLabelledPass", position=0)


def test_same_labelled_passes_are_both_recorded_in_application_order(
    source: _Source,
) -> None:
    result = _expand(
        source,
        forward_expansions=[_PassingLabelledPass(), module_level_pass],
        derivative_expansions=[_PassingLabelledPass(), _PassingLabelledPass()],
    )

    assert result.forward_pass_labels == ("_PassingLabelledPass", "module_level_pass")
    assert result.derivative_pass_labels == (
        "_PassingLabelledPass",
        "_PassingLabelledPass",
    )


def test_update_pass_labels_are_recorded_in_application_order() -> None:
    result = _expand_update([module_level_pass, _CountingPass()])

    assert result.pass_labels == ("module_level_pass", "_CountingPass")
