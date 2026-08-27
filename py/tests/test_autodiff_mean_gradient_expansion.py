"""Unit tests for the gradient-path broadcast-and-scale expansion pass.

`expand_mean_derivative_program` rewrites every region of a `DerivativeProgram`
that satisfies the complete seven-clause semantic predicate -- a `DivOperator`
by a literal count whose operand is a `BroadcastOperator` of a `[1, 1]` value --
into the matmul-based five-node region, so a backend needs neither a broadcast
nor a division handler to run the gradient path.

These tests pin the three things the rewrite turns on:

* the predicate is **semantic**: it is justified by an identity that holds for
  any `[1, 1]` value, so nothing here asserts anything about which rule emitted
  a chain, and the hand-built chains below carry no origin marker at all;
* a **near miss is left alone and raises nothing** -- `Broadcast` and `Div` are
  general operators, and declining is the correct response to a chain that fails
  any clause, in deliberate contrast to the forward pass's behaviour on an
  unsupported mean;
* every emitted node declares the dtype and shape its operation *actually*
  produces, and the terminal node carries the replaced `Div` node's value id and
  typespec verbatim.

Nothing here asserts anything about the forward pass, provenance records, the
detailed passes, or lowering; those are separate work.
"""

from __future__ import annotations

import copy

import pytest
from tinychain.autodiff import (
    BroadcastOperator,
    DerivativeMetadata,
    DerivativeProgram,
    DivOperator,
    MatmulOperator,
    MulOperator,
    ReshapeOperator,
    TensorGraphBuilder,
    TensorNodeRecord,
    generate,
)


# --------------------------------------------------------------------------
# lazily resolved surface under test
#
# Resolved inside the test body rather than at import time, so a missing name
# fails a test that has already built its input rather than aborting collection
# of the whole module.
# --------------------------------------------------------------------------


def _expand(program: DerivativeProgram) -> DerivativeProgram:
    from tinychain.autodiff import expand_mean_derivative_program

    return expand_mean_derivative_program(program)


def _fill_operator_type() -> type:
    from tinychain.autodiff import FillOperator

    return FillOperator


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_DERIVED = object()


def _typespec(dtype: str, shape: object) -> dict[str, object]:
    return {"dtype": dtype, "shape": list(shape)}


def _metadata() -> DerivativeMetadata:
    return DerivativeMetadata(
        source_graph_id="graph",
        transform_version="0.1.0",
        tensor_op_contract_version="0.1.0",
        wrt_signature=("v0",),
        seed_contract="seed matches output",
    )


def _traced_mean_program(
    *, shape: tuple[int, int] = (3, 5), dtype: str = "f64", keepdims: bool = True
) -> DerivativeProgram:
    """Trace `value.mean([0, 1], keepdims=...)` and generate its derivative."""
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype=dtype, shape=shape)
        output = value.mean([0, 1], keepdims=keepdims)
    graph = trace.build(outputs=output)
    return generate(graph, graph.outputs[0], [graph.inputs[0][0]], "seed")


def _chain_program(
    *,
    rows: int = 3,
    columns: int = 5,
    dtype: str = "f64",
    divisor: object = _DERIVED,
    broadcast_shape: object = _DERIVED,
    broadcast_typespec: object = _DERIVED,
    source_typespec: object = _DERIVED,
    div_typespec: object = _DERIVED,
    div_input_value_ids: object = _DERIVED,
    second_consumer: bool = False,
    broadcast_in_output_gradients: bool = False,
    broadcast_in_gradients: bool = False,
) -> DerivativeProgram:
    """Build a broadcast-and-scale chain directly, with every clause overridable.

    Hand construction is what lets the near-miss table reach shapes, dtypes,
    divisors, and consumer counts the reverse transform would never emit, and it
    carries no marker of where the chain came from -- which is the point: the
    predicate must decide from the artifact alone.
    """
    target_shape = [rows, columns] if broadcast_shape is _DERIVED else list(broadcast_shape)  # type: ignore[arg-type]
    resolved_source = (
        _typespec(dtype, (1, 1)) if source_typespec is _DERIVED else source_typespec
    )
    resolved_broadcast = (
        _typespec(dtype, target_shape) if broadcast_typespec is _DERIVED else broadcast_typespec
    )
    resolved_div = (
        _typespec(dtype, target_shape) if div_typespec is _DERIVED else div_typespec
    )
    resolved_divisor: object = float(rows * columns) if divisor is _DERIVED else divisor

    broadcast_node = TensorNodeRecord(
        node_id="dn0",
        output_value_id="d0",
        operator=BroadcastOperator(),
        op_params={"shape": target_shape},
        input_value_ids=["seed"],
        output_typespec=resolved_broadcast,  # type: ignore[arg-type]
    )
    div_node = TensorNodeRecord(
        node_id="dn1",
        output_value_id="d1",
        operator=DivOperator(),
        op_params={"right_literal": resolved_divisor},
        input_value_ids=(
            ["d0"] if div_input_value_ids is _DERIVED else list(div_input_value_ids)  # type: ignore[arg-type]
        ),
        output_typespec=resolved_div,  # type: ignore[arg-type]
    )
    nodes = [broadcast_node, div_node]
    value_typespecs: dict[str, dict[str, object]] = {
        "seed": resolved_source,  # type: ignore[dict-item]
        "d0": resolved_broadcast,  # type: ignore[dict-item]
        "d1": resolved_div,  # type: ignore[dict-item]
    }

    if second_consumer:
        extra = TensorNodeRecord(
            node_id="dn2",
            output_value_id="d2",
            operator=MulOperator(),
            op_params={"right_literal": 2.0},
            input_value_ids=["d0"],
            output_typespec=resolved_broadcast,  # type: ignore[arg-type]
        )
        nodes.append(extra)
        value_typespecs["d2"] = resolved_broadcast  # type: ignore[assignment]

    gradients = {"v0": "d1"}
    if broadcast_in_gradients:
        gradients["v1"] = "d0"
    output_gradients: list[str | None] = ["d1"]
    if broadcast_in_output_gradients:
        output_gradients.append("d0")

    return DerivativeProgram(
        nodes=nodes,
        gradients=gradients,
        output_gradients=output_gradients,
        metadata=_metadata(),
        value_typespecs=value_typespecs,
    )


def _operator_types(nodes: list[TensorNodeRecord]) -> list[type]:
    return [type(node.operator) for node in nodes]


def _value_ids(program: DerivativeProgram) -> set[str]:
    ids: set[str] = set(program.value_typespecs)
    for node in program.nodes:
        ids.add(node.output_value_id)
        ids.update(node.input_value_ids)
    return ids


def _assert_region(
    nodes: list[TensorNodeRecord],
    *,
    dtype: str,
    rows: int,
    columns: int,
    source_value_id: str,
    terminal_value_id: str,
    terminal_typespec: dict[str, object],
) -> None:
    """Assert *nodes* is exactly the five-node region of the specification.

    Every expected shape is written out here from `rows` and `columns` rather
    than read back from the node under test, so a node that declares a shape its
    operation cannot produce fails rather than agreeing with itself.
    """
    fill_operator = _fill_operator_type()
    assert _operator_types(nodes) == [
        fill_operator,
        MatmulOperator,
        fill_operator,
        MatmulOperator,
        MulOperator,
    ]
    e1, e2, e3, e4, e5 = nodes

    assert e1.input_value_ids == []
    assert e1.op_params == {"fill": 1.0, "dtype": dtype, "shape": [rows, 1]}
    assert e1.output_typespec == _typespec(dtype, (rows, 1))

    assert e2.input_value_ids == [e1.output_value_id, source_value_id]
    assert e2.op_params == {}
    assert e2.output_typespec == _typespec(dtype, (rows, 1))

    assert e3.input_value_ids == []
    assert e3.op_params == {"fill": 1.0, "dtype": dtype, "shape": [1, columns]}
    assert e3.output_typespec == _typespec(dtype, (1, columns))

    assert e4.input_value_ids == [e2.output_value_id, e3.output_value_id]
    assert e4.op_params == {}
    assert e4.output_typespec == _typespec(dtype, (rows, columns))

    assert e5.input_value_ids == [e4.output_value_id]
    assert e5.op_params == {"right_literal": 1.0 / float(rows * columns)}
    assert e5.output_value_id == terminal_value_id
    assert e5.output_typespec == terminal_typespec


# --------------------------------------------------------------------------
# positive: the emitted region
# --------------------------------------------------------------------------


def test_traced_derivative_expands_the_matched_region_into_five_nodes():
    program = _traced_mean_program(shape=(4, 4), keepdims=True)
    (broadcast_node, div_node) = program.nodes

    expanded = _expand(program)

    assert len(expanded.nodes) == 5
    _assert_region(
        expanded.nodes,
        dtype="f64",
        rows=4,
        columns=4,
        source_value_id=broadcast_node.input_value_ids[0],
        terminal_value_id=div_node.output_value_id,
        terminal_typespec=div_node.output_typespec,
    )


def test_non_square_region_emits_untransposed_fill_shapes():
    program = _traced_mean_program(shape=(3, 5), keepdims=True)

    expanded = _expand(program)

    e1, _, e3, _, _ = expanded.nodes
    assert e1.op_params["shape"] == [3, 1]
    assert e3.op_params["shape"] == [1, 5]


def test_leading_reshape_is_carried_through_and_never_merged_into_the_region():
    program = _traced_mean_program(shape=(3, 5), keepdims=False)
    reshape_node, broadcast_node, div_node = program.nodes
    assert isinstance(reshape_node.operator, ReshapeOperator)

    expanded = _expand(program)

    assert expanded.nodes[0] == reshape_node
    assert len(expanded.nodes) == 6
    _assert_region(
        expanded.nodes[1:],
        dtype="f64",
        rows=3,
        columns=5,
        source_value_id=reshape_node.output_value_id,
        terminal_value_id=div_node.output_value_id,
        terminal_typespec=div_node.output_typespec,
    )
    assert broadcast_node not in expanded.nodes


def test_hand_built_chain_with_no_origin_is_expanded_the_same_way():
    program = _chain_program(rows=2, columns=3, dtype="f32")

    expanded = _expand(program)

    _assert_region(
        expanded.nodes,
        dtype="f32",
        rows=2,
        columns=3,
        source_value_id="seed",
        terminal_value_id="d1",
        terminal_typespec=_typespec("f32", (2, 3)),
    )


# --------------------------------------------------------------------------
# preservation, purity, determinism
# --------------------------------------------------------------------------


def test_gradients_output_gradients_and_metadata_are_carried_through_unchanged():
    program = _traced_mean_program(shape=(3, 5), keepdims=True)

    expanded = _expand(program)

    assert expanded.gradients == program.gradients
    assert expanded.output_gradients == program.output_gradients
    assert expanded.metadata == program.metadata
    assert expanded.value_typespecs == program.value_typespecs


def test_input_program_is_not_mutated():
    program = _traced_mean_program(shape=(3, 5), keepdims=True)
    before = copy.deepcopy(program)

    _expand(program)

    assert program == before


def test_equal_programs_expand_to_equal_programs():
    first = _expand(_traced_mean_program(shape=(3, 5), keepdims=True))
    second = _expand(_traced_mean_program(shape=(3, 5), keepdims=True))

    assert first == second
    assert [node.node_id for node in first.nodes] == [node.node_id for node in second.nodes]


def test_only_the_broadcast_intermediate_value_id_is_removed():
    program = _traced_mean_program(shape=(3, 5), keepdims=True)
    broadcast_node = program.nodes[0]
    before = _value_ids(program)

    expanded = _expand(program)

    surviving = _value_ids(expanded)
    assert before - surviving == {broadcast_node.output_value_id}


# --------------------------------------------------------------------------
# near misses: returned unchanged, raising nothing
#
# Each case fails exactly one clause of the predicate. `Broadcast` and `Div` are
# general operators, so the pass declines rather than rejecting.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "overrides"),
    [
        ("divisor off by one", {"divisor": 16.0}),
        ("divisor equal to r + c", {"divisor": 8.0}),
        ("non-numeric divisor", {"divisor": "15"}),
        ("boolean divisor", {"divisor": True}),
        ("broadcast target of rank 1", {"broadcast_shape": [15]}),
        ("broadcast target of rank 3", {"broadcast_shape": [1, 3, 5]}),
        ("zero dimension in the broadcast target", {"broadcast_shape": [0, 5]}),
        ("broadcast source of shape [1, c]", {"source_typespec": {"dtype": "f64", "shape": [1, 5]}}),
        ("broadcast source of rank 0", {"source_typespec": {"dtype": "f64", "shape": []}}),
        ("broadcast source without a dtype", {"source_typespec": {"shape": [1, 1]}}),
        ("dtype mismatch between source and target", {"source_typespec": {"dtype": "f32", "shape": [1, 1]}}),
        ("broadcast output typespec without a dtype", {"broadcast_typespec": {"shape": [3, 5]}}),
        ("div output disagreeing with the broadcast target", {"div_typespec": {"dtype": "f64", "shape": [5, 3]}}),
        ("div output typespec without a dtype", {"div_typespec": {"shape": [3, 5]}}),
        ("div with two operands", {"div_input_value_ids": ["d0", "seed"]}),
        ("broadcast output with a second consumer", {"second_consumer": True}),
        ("broadcast output named in output_gradients", {"broadcast_in_output_gradients": True}),
        ("broadcast output named in gradients", {"broadcast_in_gradients": True}),
    ],
)
def test_near_miss_is_returned_unchanged_and_raises_nothing(case, overrides):
    program = _chain_program(**overrides)
    before = copy.deepcopy(program)

    expanded = _expand(program)

    assert expanded == before, case
    assert program == before, case


def test_a_program_with_no_matching_chain_is_returned_unchanged():
    program = _traced_mean_program(shape=(3, 5), keepdims=False)
    program = DerivativeProgram(
        nodes=[program.nodes[0]],
        gradients={"v0": program.nodes[0].output_value_id},
        output_gradients=[program.nodes[0].output_value_id],
        metadata=program.metadata,
        value_typespecs=dict(program.value_typespecs),
    )
    before = copy.deepcopy(program)

    assert _expand(program) == before
