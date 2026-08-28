"""Unit tests for the mean-expansion sidecar provenance surface.

The two rewrites already documented -- `expand_mean_graph` and
`expand_mean_derivative_program` -- return only the rewritten artifact. This
module pins the detailed forms that additionally report, beside the artifact,
one `MeanExpansionRegion` per rewritten region, and pins that the composable
forms are *defined* as the detailed form followed by selecting the artifact
field, so the two entry points cannot disagree.

These tests pin four things:

* the two pass-name constants and the exact fields and tier strings of the
  three frozen sidecar dataclasses;
* `expand_mean_graph_detailed`'s `graph` field, and
  `expand_mean_derivative_program_detailed`'s `program` field, equal what the
  composable pass of the same name returns for the same input -- proving the
  composable form is the detailed form plus field selection, not a second
  rewrite path;
* `source_node_ids`, `emitted_node_ids`, and `terminal_value_id` match the
  artifact in the order the region actually appears, and no `op_params`
  mapping either pass writes carries a bookkeeping key beyond that operator's
  own parameters;
* the passes remain deterministic and pure: an artifact with nothing to
  rewrite comes back with empty `regions` and an unchanged artifact, and equal
  inputs produce equal outputs including every region record.

Nothing here asserts anything about the fail-closed matrix, the equivalence
tests, or documentation; those are separate work.
"""

from __future__ import annotations

import pytest
import tinychain as tc
from tinychain.autodiff import (
    DerivativeProgram,
    FillOperator,
    MatmulOperator,
    MulOperator,
    ReshapeOperator,
    TensorGraph,
    TensorGraphBuilder,
    generate,
)

# --------------------------------------------------------------------------
# lazily resolved surface under test
#
# The detailed passes and the sidecar constants are resolved inside the test
# body rather than at import time, so a missing name fails one test that has
# already built its input rather than aborting collection of the whole module.
# --------------------------------------------------------------------------


def _expand_graph(graph: TensorGraph) -> TensorGraph:
    from tinychain.autodiff import expand_mean_graph

    return expand_mean_graph(graph)


def _expand_graph_detailed(graph: TensorGraph):
    from tinychain.autodiff import expand_mean_graph_detailed

    return expand_mean_graph_detailed(graph)


def _expand_program(program: DerivativeProgram) -> DerivativeProgram:
    from tinychain.autodiff import expand_mean_derivative_program

    return expand_mean_derivative_program(program)


def _expand_program_detailed(program: DerivativeProgram):
    from tinychain.autodiff import expand_mean_derivative_program_detailed

    return expand_mean_derivative_program_detailed(program)


def _forward_pass_name() -> str:
    from tinychain.autodiff.expansion import MEAN_EXPANSION_FORWARD

    return MEAN_EXPANSION_FORWARD


def _gradient_pass_name() -> str:
    from tinychain.autodiff.expansion import BROADCAST_SCALE_EXPANSION

    return BROADCAST_SCALE_EXPANSION


def _region_type() -> type:
    from tinychain.autodiff.expansion import MeanExpansionRegion

    return MeanExpansionRegion


def _graph_result_type() -> type:
    from tinychain.autodiff.expansion import MeanGraphExpansionResult

    return MeanGraphExpansionResult


def _program_result_type() -> type:
    from tinychain.autodiff.expansion import MeanDerivativeExpansionResult

    return MeanDerivativeExpansionResult


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _traced_mean_graph(
    *, shape: tuple[int, int] = (3, 5), dtype: str = "f64", keepdims: bool
) -> TensorGraph:
    """Trace `value.mean([0, 1], keepdims=...)` and return the finalized graph."""
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype=dtype, shape=shape)
        output = value.mean([0, 1], keepdims=keepdims)
    return trace.build(outputs=output)


def _traced_two_mean_graph(
    *, shape: tuple[int, int] = (2, 3), dtype: str = "f64"
) -> TensorGraph:
    """Trace two independent all-axis means over two distinct inputs."""
    with TensorGraphBuilder() as trace:
        first = trace.input("first", dtype=dtype, shape=shape)
        second = trace.input("second", dtype=dtype, shape=shape)
        first_mean = first.mean([0, 1], keepdims=True)
        second_mean = second.mean([0, 1], keepdims=True)
    return trace.build(outputs=(first_mean, second_mean))


def _traced_mean_program(
    *, shape: tuple[int, int] = (3, 5), dtype: str = "f64", keepdims: bool
) -> DerivativeProgram:
    """Trace `value.mean([0, 1], keepdims=...)` and generate its derivative."""
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype=dtype, shape=shape)
        output = value.mean([0, 1], keepdims=keepdims)
    graph = trace.build(outputs=output)
    return generate(graph, graph.outputs[0], [graph.inputs[0][0]], "seed")


def _graph_with_no_mean(*, shape: tuple[int, int] = (2, 2)) -> TensorGraph:
    """Trace an artifact with no `MeanOperator` node, hence no rewritable region.

    The forward pass fails closed on an unsupported mean rather than leaving
    it alone (unlike the gradient-path pass), so "nothing to rewrite" for this
    pass means no candidate node at all, not a candidate that fails a condition.
    """
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype="f64", shape=shape)
        output = value + value
    return trace.build(outputs=output)


# --------------------------------------------------------------------------
# constants and field shape
# --------------------------------------------------------------------------


def test_mean_expansion_forward_constant_has_its_exact_string_value() -> None:
    assert _forward_pass_name() == "mean_expansion_forward"


def test_broadcast_scale_expansion_constant_has_its_exact_string_value() -> None:
    assert _gradient_pass_name() == "broadcast_scale_expansion"


def test_mean_expansion_region_declares_exactly_the_specified_fields() -> None:
    import dataclasses

    region_type = _region_type()
    field_names = {field.name for field in dataclasses.fields(region_type)}
    assert field_names == {
        "pass_name",
        "source_node_ids",
        "emitted_node_ids",
        "terminal_value_id",
        "tier",
    }


def test_mean_graph_expansion_result_declares_graph_and_regions() -> None:
    import dataclasses

    result_type = _graph_result_type()
    field_names = {field.name for field in dataclasses.fields(result_type)}
    assert field_names == {"graph", "regions"}


def test_mean_derivative_expansion_result_declares_program_and_regions() -> None:
    import dataclasses

    result_type = _program_result_type()
    field_names = {field.name for field in dataclasses.fields(result_type)}
    assert field_names == {"program", "regions"}


# --------------------------------------------------------------------------
# forward detailed pass
# --------------------------------------------------------------------------


def test_rank_preserving_forward_detailed_result_matches_the_composable_pass() -> None:
    graph = _traced_mean_graph(keepdims=True)

    detailed = _expand_graph_detailed(graph)
    composable = _expand_graph(graph)

    assert isinstance(detailed, _graph_result_type())
    assert detailed.graph == composable
    assert len(detailed.regions) == 1

    region = detailed.regions[0]
    assert isinstance(region, _region_type())
    assert region.pass_name == _forward_pass_name()
    assert region.tier == "rank_preserving"


def test_rank_reducing_forward_detailed_result_reports_six_emitted_nodes() -> None:
    graph = _traced_mean_graph(keepdims=False)

    detailed = _expand_graph_detailed(graph)
    composable = _expand_graph(graph)

    assert detailed.graph == composable
    assert len(detailed.regions) == 1

    region = detailed.regions[0]
    assert region.pass_name == _forward_pass_name()
    assert region.tier == "rank_reducing"
    assert len(region.emitted_node_ids) == 6


def test_forward_region_source_emitted_and_terminal_ids_match_the_artifact() -> None:
    graph = _traced_mean_graph(shape=(3, 5), keepdims=True)
    mean_node = next(
        node for node in graph.nodes if type(node.operator).__name__ == "MeanOperator"
    )

    detailed = _expand_graph_detailed(graph)
    region = detailed.regions[0]

    assert region.source_node_ids == (mean_node.node_id,)
    assert region.terminal_value_id == mean_node.output_value_id

    rewritten_graph = detailed.graph
    emitted_ids_in_graph = [
        node.node_id
        for node in rewritten_graph.nodes
        if node.node_id in set(region.emitted_node_ids)
    ]
    assert emitted_ids_in_graph == list(region.emitted_node_ids)

    terminal_node = next(
        node for node in rewritten_graph.nodes if node.node_id == region.emitted_node_ids[-1]
    )
    assert terminal_node.output_value_id == region.terminal_value_id


def test_forward_emitted_op_params_carry_no_key_beyond_the_operators_own() -> None:
    graph = _traced_mean_graph(keepdims=False)
    detailed = _expand_graph_detailed(graph)
    region = detailed.regions[0]

    emitted_nodes = {
        node.node_id: node
        for node in detailed.graph.nodes
        if node.node_id in set(region.emitted_node_ids)
    }
    for node_id in region.emitted_node_ids:
        node = emitted_nodes[node_id]
        if isinstance(node.operator, FillOperator):
            assert set(node.op_params) == {"fill", "dtype", "shape"}
        elif isinstance(node.operator, MatmulOperator):
            assert set(node.op_params) == set()
        elif isinstance(node.operator, MulOperator):
            assert set(node.op_params) == {"right_literal"}
        elif isinstance(node.operator, ReshapeOperator):
            assert set(node.op_params) == {"shape"}
        else:
            pytest.fail(f"unexpected emitted operator type: {type(node.operator).__name__}")


def test_two_means_yield_two_regions_in_artifact_order() -> None:
    graph = _traced_two_mean_graph()
    mean_node_ids_in_order = [
        node.node_id for node in graph.nodes if type(node.operator).__name__ == "MeanOperator"
    ]
    assert len(mean_node_ids_in_order) == 2

    detailed = _expand_graph_detailed(graph)

    assert len(detailed.regions) == 2
    assert [region.source_node_ids[0] for region in detailed.regions] == mean_node_ids_in_order
    for region in detailed.regions:
        assert region.pass_name == _forward_pass_name()


def test_an_artifact_with_no_rewritable_region_yields_empty_regions_and_is_unchanged() -> None:
    graph = _graph_with_no_mean()

    detailed = _expand_graph_detailed(graph)

    assert detailed.regions == ()
    assert detailed.graph == graph


def test_equal_graphs_expand_to_equal_detailed_results(keepdims: bool = True) -> None:
    first = _traced_mean_graph(keepdims=True)
    second = _traced_mean_graph(keepdims=True)

    first_detailed = _expand_graph_detailed(first)
    second_detailed = _expand_graph_detailed(second)

    assert first_detailed == second_detailed


# --------------------------------------------------------------------------
# gradient-path detailed pass
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keepdims", [True, False])
def test_gradient_path_detailed_result_matches_the_composable_pass(keepdims: bool) -> None:
    program = _traced_mean_program(keepdims=keepdims)

    detailed = _expand_program_detailed(program)
    composable = _expand_program(program)

    assert isinstance(detailed, _program_result_type())
    assert detailed.program == composable
    assert len(detailed.regions) == 1

    region = detailed.regions[0]
    assert isinstance(region, _region_type())
    assert region.pass_name == _gradient_pass_name()
    assert len(region.emitted_node_ids) == 5


@pytest.mark.parametrize("keepdims", [True, False])
def test_gradient_path_region_is_recorded_rank_preserving(keepdims: bool) -> None:
    """The gradient-path region performs no rank change for either forward tier."""
    program = _traced_mean_program(keepdims=keepdims)

    detailed = _expand_program_detailed(program)

    assert detailed.regions[0].tier == "rank_preserving"


def test_gradient_region_source_emitted_and_terminal_ids_match_the_artifact() -> None:
    program = _traced_mean_program(keepdims=True)
    broadcast_node = next(
        node for node in program.nodes if type(node.operator).__name__ == "BroadcastOperator"
    )
    div_node = next(
        node for node in program.nodes if type(node.operator).__name__ == "DivOperator"
    )

    detailed = _expand_program_detailed(program)
    region = detailed.regions[0]

    # The matched chain that gets replaced is both nodes: the broadcast, which
    # is removed outright, and the division, whose position the region
    # occupies -- named here in the order they must appear in the artifact for
    # the broadcast's output to be a valid input to the division.
    assert region.source_node_ids == (broadcast_node.node_id, div_node.node_id)
    assert region.terminal_value_id == div_node.output_value_id

    rewritten_program = detailed.program
    emitted_ids_in_program = [
        node.node_id
        for node in rewritten_program.nodes
        if node.node_id in set(region.emitted_node_ids)
    ]
    assert emitted_ids_in_program == list(region.emitted_node_ids)

    terminal_node = next(
        node
        for node in rewritten_program.nodes
        if node.node_id == region.emitted_node_ids[-1]
    )
    assert terminal_node.output_value_id == region.terminal_value_id


def test_gradient_emitted_op_params_carry_no_key_beyond_the_operators_own() -> None:
    program = _traced_mean_program(keepdims=True)
    detailed = _expand_program_detailed(program)
    region = detailed.regions[0]

    emitted_nodes = {
        node.node_id: node
        for node in detailed.program.nodes
        if node.node_id in set(region.emitted_node_ids)
    }
    for node_id in region.emitted_node_ids:
        node = emitted_nodes[node_id]
        if isinstance(node.operator, FillOperator):
            assert set(node.op_params) == {"fill", "dtype", "shape"}
        elif isinstance(node.operator, MatmulOperator):
            assert set(node.op_params) == set()
        elif isinstance(node.operator, MulOperator):
            assert set(node.op_params) == {"right_literal"}
        else:
            pytest.fail(f"unexpected emitted operator type: {type(node.operator).__name__}")


def test_a_program_with_no_matching_region_yields_empty_regions_and_is_unchanged() -> None:
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype="f64", shape=(2, 2))
        output = value.mean([0], keepdims=True)  # partial-axis mean has no matching VJP chain
    graph = trace.build(outputs=output)
    program = generate(graph, graph.outputs[0], [graph.inputs[0][0]], "seed")

    detailed = _expand_program_detailed(program)

    assert detailed.regions == ()
    assert detailed.program == program


def test_equal_programs_expand_to_equal_detailed_results() -> None:
    first = _traced_mean_program(keepdims=True)
    second = _traced_mean_program(keepdims=True)

    first_detailed = _expand_program_detailed(first)
    second_detailed = _expand_program_detailed(second)

    assert first_detailed == second_detailed


# --------------------------------------------------------------------------
# signature: the composable passes remain single-positional-artifact
# --------------------------------------------------------------------------


def test_composable_forward_pass_rejects_a_missing_positional_argument() -> None:
    from tinychain.autodiff import expand_mean_graph

    with pytest.raises(TypeError):
        expand_mean_graph()  # type: ignore[call-arg]


def test_composable_gradient_pass_rejects_a_missing_positional_argument() -> None:
    from tinychain.autodiff import expand_mean_derivative_program

    with pytest.raises(TypeError):
        expand_mean_derivative_program()  # type: ignore[call-arg]


def test_detailed_forward_pass_rejects_a_missing_positional_argument() -> None:
    from tinychain.autodiff import expand_mean_graph_detailed

    with pytest.raises(TypeError):
        expand_mean_graph_detailed()  # type: ignore[call-arg]


def test_detailed_gradient_pass_rejects_a_missing_positional_argument() -> None:
    from tinychain.autodiff import expand_mean_derivative_program_detailed

    with pytest.raises(TypeError):
        expand_mean_derivative_program_detailed()  # type: ignore[call-arg]


# --------------------------------------------------------------------------
# export surface
# --------------------------------------------------------------------------


def test_new_provenance_names_are_exported_from_the_autodiff_package() -> None:
    for name in (
        "MEAN_EXPANSION_FORWARD",
        "BROADCAST_SCALE_EXPANSION",
        "MeanExpansionRegion",
        "MeanGraphExpansionResult",
        "MeanDerivativeExpansionResult",
        "expand_mean_graph_detailed",
        "expand_mean_derivative_program_detailed",
    ):
        assert hasattr(tc.autodiff, name), f"{name} is not reachable from tinychain.autodiff"
        assert name in tc.autodiff.__all__, f"{name} is absent from tinychain.autodiff.__all__"
