"""Repeat-expansion coverage for both mean-expansion passes.

The other five focused modules beside this one --
`test_autodiff_fill_contract.py`, `test_autodiff_mean_expansion.py`,
`test_autodiff_mean_gradient_expansion.py`,
`test_autodiff_expansion_provenance.py`, and
`test_autodiff_expansion_lowering_equivalence.py` -- already exercise every
supported-mean validation branch, every error category, every gradient-path
near miss (including wrong divisors), the positive predicate for both tiers and
a non-square shape, and determinism of both the composable and detailed
passes. This module does not restate any of that; it covers the one scenario
none of them exercises directly: applying a pass to an artifact that already
went through it.

Minted identifiers can collide when an artifact is expanded twice. The
`_IdentifierMinter` collision check already guards that case and is asserted
elsewhere for a hand-seeded
squatter node. What is missing is the literal repeat-application scenario:
feeding a pass's own output back into itself, and feeding it an artifact that
mixes already-expanded content with a fresh candidate the second call must
mint reserved identifiers for. In the first case every candidate the pass
looks for is gone, so the only faithful outcome is a clean no-op. In the
second, the fresh candidate's minted identifiers necessarily start again from
index zero and collide with the reserved identifiers the first expansion left
behind, so the only faithful outcome is `malformed_derivative_ir` naming the
colliding identifier -- never a silently corrupted artifact.
"""

from __future__ import annotations

import pytest
from tinychain.autodiff import (
    AutodiffError,
    BroadcastOperator,
    DerivativeProgram,
    DivOperator,
    MeanOperator,
    TensorGraph,
    TensorGraphBuilder,
    TensorNodeRecord,
    generate,
)


# --------------------------------------------------------------------------
# lazily resolved surface under test
#
# Resolved inside the test body rather than at import time, so a missing name
# fails one test that has already built its input rather than aborting
# collection of the whole module.
# --------------------------------------------------------------------------


def _expand_graph(graph: TensorGraph) -> TensorGraph:
    from tinychain.autodiff import expand_mean_graph

    return expand_mean_graph(graph)


def _expand_program(program: DerivativeProgram) -> DerivativeProgram:
    from tinychain.autodiff import expand_mean_derivative_program

    return expand_mean_derivative_program(program)


def _reserved_node_id(index: int = 0) -> str:
    from tinychain.autodiff.expansion import EXPANSION_NODE_ID_PREFIX

    return f"{EXPANSION_NODE_ID_PREFIX}{index}"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _typespec(dtype: str, shape: object) -> dict[str, object]:
    return {"dtype": dtype, "shape": list(shape)}


def _traced_mean_graph(
    *, shape: tuple[int, int] = (3, 5), dtype: str = "f64", keepdims: bool = True
) -> TensorGraph:
    """Trace `value.mean([0, 1], keepdims=...)` and return the finalized graph."""
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype=dtype, shape=shape)
        output = value.mean([0, 1], keepdims=keepdims)
    return trace.build(outputs=output)


def _traced_mean_program(
    *, shape: tuple[int, int] = (3, 5), dtype: str = "f64", keepdims: bool = True
) -> DerivativeProgram:
    """Trace `value.mean([0, 1], keepdims=...)` and generate its derivative."""
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype=dtype, shape=shape)
        output = value.mean([0, 1], keepdims=keepdims)
    graph = trace.build(outputs=output)
    return generate(graph, graph.outputs[0], [graph.inputs[0][0]], "seed")


# --------------------------------------------------------------------------
# forward pass: reapplication to its own output
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keepdims", [True, False])
def test_forward_pass_reapplied_to_its_own_output_is_a_clean_noop(keepdims: bool) -> None:
    """No `MeanOperator` survives one expansion, so a second call finds nothing
    to rewrite and the artifact comes back exactly as it went in."""
    graph = _traced_mean_graph(shape=(3, 5), keepdims=keepdims)

    once = _expand_graph(graph)
    twice = _expand_graph(once)

    assert twice == once
    assert not any(isinstance(node.operator, MeanOperator) for node in twice.nodes)


def test_forward_pass_reapplied_with_a_fresh_candidate_collides_on_the_reserved_namespace() -> None:
    """An artifact mixing already-expanded content with one fresh, otherwise
    supported mean is what a caller who expands, then adds more work, then
    expands again would produce. The fresh candidate's minted identifiers
    start again from index zero and collide with the reserved identifiers the
    first expansion already placed in the same artifact -- `malformed_derivative_ir`,
    naming the colliding identifier, not a corrupted result."""
    source = _traced_mean_graph(shape=(2, 2), keepdims=True)
    once = _expand_graph(source)

    fresh_mean = TensorNodeRecord(
        node_id="n_fresh",
        output_value_id="v_fresh_out",
        operator=MeanOperator(),
        op_params={"axes": [0, 1], "keepdims": True},
        input_value_ids=["v_fresh_in"],
        output_typespec=_typespec("f64", (1, 1)),
    )
    combined = TensorGraph(
        nodes=[*once.nodes, fresh_mean],
        inputs=[*once.inputs, ("v_fresh_in", _typespec("f64", (2, 2)))],
        outputs=[*once.outputs, "v_fresh_out"],
    )
    before = TensorGraph(nodes=list(combined.nodes), inputs=list(combined.inputs), outputs=list(combined.outputs))

    with pytest.raises(AutodiffError) as raised:
        _expand_graph(combined)

    assert raised.value.category == "malformed_derivative_ir"
    assert _reserved_node_id(0) in raised.value.message
    # Nothing about the input changed; a raise never leaves a partial result.
    assert combined == before


# --------------------------------------------------------------------------
# gradient-path pass: reapplication to its own output
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keepdims", [True, False])
def test_gradient_pass_reapplied_to_its_own_output_is_a_clean_noop(keepdims: bool) -> None:
    """No matching broadcast-and-scale chain survives one expansion, so a
    second call finds nothing to rewrite and the program comes back exactly
    as it went in."""
    program = _traced_mean_program(shape=(3, 5), keepdims=keepdims)

    once = _expand_program(program)
    twice = _expand_program(once)

    assert twice == once
    assert not any(isinstance(node.operator, BroadcastOperator) for node in twice.nodes)
    assert not any(isinstance(node.operator, DivOperator) for node in twice.nodes)


def test_gradient_pass_reapplied_with_a_fresh_candidate_collides_on_the_reserved_namespace() -> None:
    """The gradient-path analogue of the forward-pass collision case above: an
    already-expanded program plus one fresh, otherwise-matching
    broadcast-and-scale chain. The fresh region's minted identifiers start
    again from index zero and collide with the reserved identifiers already
    in the program -- `malformed_derivative_ir`, never silent corruption."""
    source = _traced_mean_program(shape=(2, 2), keepdims=True)
    once = _expand_program(source)

    broadcast_extra = TensorNodeRecord(
        node_id="dn_extra0",
        output_value_id="d_extra0",
        operator=BroadcastOperator(),
        op_params={"shape": [2, 2]},
        input_value_ids=["g_extra"],
        output_typespec=_typespec("f64", (2, 2)),
    )
    div_extra = TensorNodeRecord(
        node_id="dn_extra1",
        output_value_id="d_extra1",
        operator=DivOperator(),
        op_params={"right_literal": 4.0},
        input_value_ids=["d_extra0"],
        output_typespec=_typespec("f64", (2, 2)),
    )
    combined = DerivativeProgram(
        nodes=[*once.nodes, broadcast_extra, div_extra],
        gradients={**once.gradients, "v_extra": "d_extra1"},
        output_gradients=[*once.output_gradients, "d_extra1"],
        metadata=once.metadata,
        value_typespecs={
            **once.value_typespecs,
            "g_extra": _typespec("f64", (1, 1)),
            "d_extra0": _typespec("f64", (2, 2)),
            "d_extra1": _typespec("f64", (2, 2)),
        },
    )

    with pytest.raises(AutodiffError) as raised:
        _expand_program(combined)

    assert raised.value.category == "malformed_derivative_ir"
    assert _reserved_node_id(0) in raised.value.message
