"""Unit tests for the forward all-axis mean expansion pass.

`expand_mean_graph` rewrites every supported all-axis rank-2 `MeanOperator` in a
`TensorGraph` into a matmul-based region in both tiers:
five nodes when the mean declared `keepdims=True`, and six -- the sixth a real
`ReshapeOperator` -- when it declared `keepdims=False`.

These tests pin three things the rewrite turns on:

* every emitted node declares the dtype and shape its operation *actually*
  produces, recomputed here from the operands rather than read back from the
  node under test, so a node can never declare a shape its operation cannot
  produce;
* every candidate mean is validated before any node is emitted, so a rejected
  artifact never comes back partially rewritten;
* each supported-mean validation failure raises its own category with a message
  naming the offending node and explaining the failed condition.

Nothing here asserts anything about the gradient-path *rewrite*, provenance
records, or the detailed passes; those are separate work. The one exception is
the identifier-collision section at the end: one helper indexes the value ids an
artifact mentions for both passes, so the cases proving a minted identifier can
never alias an existing value are written against both passes together, where
the shared contract can be seen as one thing.
"""

from __future__ import annotations

import copy

import pytest
import tinychain as tc
from tinychain.autodiff import (
    AutodiffError,
    MatmulOperator,
    MeanOperator,
    MulOperator,
    ReshapeOperator,
    SumOperator,
    TensorGraph,
    TensorGraphBuilder,
    TensorNodeRecord,
)


# --------------------------------------------------------------------------
# lazily resolved surface under test
#
# The pass and its reserved-namespace constants are resolved inside the test
# body rather than at import time, so a missing name fails one test that has
# already built its input rather than aborting collection of the whole module.
# --------------------------------------------------------------------------


def _expand(graph: TensorGraph) -> TensorGraph:
    from tinychain.autodiff import expand_mean_graph

    return expand_mean_graph(graph)


def _reserved_node_id(index: int = 0) -> str:
    from tinychain.autodiff.expansion import EXPANSION_NODE_ID_PREFIX

    return f"{EXPANSION_NODE_ID_PREFIX}{index}"


def _reserved_value_id(index: int = 0) -> str:
    from tinychain.autodiff.expansion import EXPANSION_VALUE_ID_PREFIX

    return f"{EXPANSION_VALUE_ID_PREFIX}{index}"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_DERIVED = object()


def _typespec(dtype: str, shape: object) -> dict[str, object]:
    return {"dtype": dtype, "shape": list(shape)}


def _traced_mean_graph(
    *, shape: tuple[int, ...] = (3, 5), dtype: str = "f64", keepdims: bool = True
) -> TensorGraph:
    """Trace `value.mean([0, 1], keepdims=...)` and return the finalized graph."""
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype=dtype, shape=shape)
        output = value.mean([0, 1], keepdims=keepdims)
    return trace.build(outputs=output)


def _mean_graph(
    *,
    operand_typespec: object = _DERIVED,
    op_params: object = _DERIVED,
    output_typespec: object = _DERIVED,
    node_id: str = "n0",
    output_value_id: str = "v1",
    input_value_ids: list[str] | None = None,
    operand_dtype: str = "f64",
    operand_shape: object = (3, 5),
) -> TensorGraph:
    """Build a one-node mean graph directly, with every field overridable.

    Hand construction is what lets the failure table reach malformed shapes,
    dtypes, axes, and identifiers that the tracer would reject long before the
    pass ever saw them.
    """
    params = (
        {"axes": [0, 1], "keepdims": True} if op_params is _DERIVED else dict(op_params)  # type: ignore[arg-type]
    )
    resolved_operand_typespec = (
        _typespec(operand_dtype, operand_shape)
        if operand_typespec is _DERIVED
        else operand_typespec
    )
    if output_typespec is _DERIVED:
        reduced_shape: tuple[int, ...] = (1, 1) if params.get("keepdims") else ()
        resolved_output_typespec: object = _typespec(operand_dtype, reduced_shape)
    else:
        resolved_output_typespec = output_typespec

    node = TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=MeanOperator(),
        op_params=params,
        input_value_ids=["v0"] if input_value_ids is None else input_value_ids,
        output_typespec=resolved_output_typespec,  # type: ignore[arg-type]
    )
    return TensorGraph(
        nodes=[node],
        inputs=[("v0", resolved_operand_typespec)],  # type: ignore[list-item]
        outputs=[output_value_id],
    )


def _value_typespecs(graph: TensorGraph) -> dict[str, object]:
    """Index every value id in *graph* to the typespec its producer declares."""
    typespecs: dict[str, object] = {
        value_id: typespec for value_id, typespec in graph.inputs
    }
    for node in graph.nodes:
        typespecs[node.output_value_id] = node.output_typespec
    return typespecs


def _value_ids(graph: TensorGraph) -> set[str]:
    ids = {value_id for value_id, _ in graph.inputs}
    for node in graph.nodes:
        ids.add(node.output_value_id)
        ids.update(node.input_value_ids)
    return ids


def _independently_computed_typespec(
    node: TensorNodeRecord, typespecs: dict[str, object]
) -> dict[str, object]:
    """Recompute what *node* truly produces, from its operands and its own rule.

    Deliberately written out here instead of calling the framework's shape
    helpers: the point of the audit is that a declared shape agrees with an
    independent computation, not that the pass agrees with itself.
    """
    operand_typespecs = [typespecs[value_id] for value_id in node.input_value_ids]

    if isinstance(node.operator, MatmulOperator):
        assert len(operand_typespecs) == 2, "a matmul is a two-operand operation"
        left, right = operand_typespecs
        left_shape = list(left["shape"])  # type: ignore[index]
        right_shape = list(right["shape"])  # type: ignore[index]
        assert len(left_shape) == 2 and len(right_shape) == 2
        assert left_shape[1] == right_shape[0], "matmul inner dimensions must agree"
        assert left["dtype"] == right["dtype"]  # type: ignore[index]
        return _typespec(str(left["dtype"]), (left_shape[0], right_shape[1]))  # type: ignore[index]

    if isinstance(node.operator, MulOperator):
        assert len(operand_typespecs) == 1, "an emitted mul scales by a literal"
        operand = operand_typespecs[0]
        return _typespec(str(operand["dtype"]), list(operand["shape"]))  # type: ignore[index]

    if isinstance(node.operator, ReshapeOperator):
        assert len(operand_typespecs) == 1
        operand = operand_typespecs[0]
        source_elements = 1
        for dimension in operand["shape"]:  # type: ignore[index]
            source_elements *= int(dimension)
        target_shape = list(node.op_params["shape"])
        target_elements = 1
        for dimension in target_shape:
            target_elements *= int(dimension)
        assert source_elements == target_elements, "a reshape preserves element count"
        return _typespec(str(operand["dtype"]), target_shape)  # type: ignore[index]

    # A fill node produces exactly what its descriptor declares.
    from tinychain.autodiff import fill_descriptor

    descriptor = fill_descriptor(node)
    assert not operand_typespecs, "a fill node has no operand"
    return _typespec(descriptor.dtype, descriptor.shape)


# --------------------------------------------------------------------------
# The rank-preserving tier emits exactly five nodes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("rows", "columns"), [(3, 5), (1, 1), (5, 3)])
def test_rank_preserving_expansion_emits_the_five_specified_nodes(
    rows: int, columns: int
) -> None:
    from tinychain.autodiff import FillOperator

    graph = _traced_mean_graph(shape=(rows, columns), keepdims=True)
    mean_node = graph.nodes[0]
    operand_value_id = mean_node.input_value_ids[0]

    expanded = _expand(graph)

    assert len(expanded.nodes) == 5
    first_fill, row_sum, second_fill, total_sum, scale = expanded.nodes

    assert isinstance(first_fill.operator, FillOperator)
    assert first_fill.input_value_ids == []
    assert first_fill.op_params == {"fill": 1.0, "dtype": "f64", "shape": [columns, 1]}
    assert first_fill.output_typespec == _typespec("f64", (columns, 1))

    assert isinstance(row_sum.operator, MatmulOperator)
    assert row_sum.input_value_ids == [operand_value_id, first_fill.output_value_id]
    assert row_sum.op_params == {}
    assert row_sum.output_typespec == _typespec("f64", (rows, 1))

    assert isinstance(second_fill.operator, FillOperator)
    assert second_fill.input_value_ids == []
    assert second_fill.op_params == {"fill": 1.0, "dtype": "f64", "shape": [1, rows]}
    assert second_fill.output_typespec == _typespec("f64", (1, rows))

    assert isinstance(total_sum.operator, MatmulOperator)
    assert total_sum.input_value_ids == [
        second_fill.output_value_id,
        row_sum.output_value_id,
    ]
    assert total_sum.op_params == {}
    assert total_sum.output_typespec == _typespec("f64", (1, 1))

    assert isinstance(scale.operator, MulOperator)
    assert scale.input_value_ids == [total_sum.output_value_id]
    assert scale.op_params == {"right_literal": 1.0 / (rows * columns)}
    assert scale.output_value_id == mean_node.output_value_id
    assert scale.output_typespec == mean_node.output_typespec

    assert not any(isinstance(node.operator, MeanOperator) for node in expanded.nodes)


# --------------------------------------------------------------------------
# The rank-reducing tier appends a real reshape to rank zero
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("rows", "columns"), [(3, 5), (1, 1)])
def test_rank_reducing_expansion_appends_a_real_reshape_to_rank_zero(
    rows: int, columns: int
) -> None:
    graph = _traced_mean_graph(shape=(rows, columns), keepdims=False)
    mean_node = graph.nodes[0]
    assert mean_node.output_typespec == _typespec("f64", ())

    expanded = _expand(graph)

    assert len(expanded.nodes) == 6
    scale = expanded.nodes[4]
    reshape = expanded.nodes[5]

    assert isinstance(scale.operator, MulOperator)
    assert scale.output_typespec == _typespec("f64", (1, 1))
    assert scale.output_value_id != mean_node.output_value_id

    assert isinstance(reshape.operator, ReshapeOperator)
    assert reshape.input_value_ids == [scale.output_value_id]
    assert reshape.op_params == {"shape": []}
    assert reshape.output_value_id == mean_node.output_value_id
    assert reshape.output_typespec == mean_node.output_typespec
    assert reshape.output_typespec == _typespec("f64", ())


# --------------------------------------------------------------------------
# Truthful-shape audit over both tiers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize(("rows", "columns"), [(3, 5), (1, 1), (5, 3)])
def test_every_emitted_node_declares_the_shape_its_operation_produces(
    keepdims: bool, rows: int, columns: int
) -> None:
    graph = _traced_mean_graph(shape=(rows, columns), keepdims=keepdims)

    expanded = _expand(graph)
    typespecs = _value_typespecs(expanded)

    for node in expanded.nodes:
        assert node.output_typespec is not None, f"node {node.node_id!r} declares no typespec"
        assert node.output_typespec == _independently_computed_typespec(node, typespecs), (
            f"node {node.node_id!r} declares a shape its operation does not produce"
        )


# --------------------------------------------------------------------------
# Permitted operator set and the two-operand matmul contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keepdims", [True, False])
def test_every_emitted_node_is_in_the_permitted_operator_set(keepdims: bool) -> None:
    from tinychain.autodiff import FillOperator

    graph = _traced_mean_graph(keepdims=keepdims)

    expanded = _expand(graph)

    for node in expanded.nodes:
        assert isinstance(
            node.operator, (FillOperator, MatmulOperator, MulOperator, ReshapeOperator)
        ), f"node {node.node_id!r} emits {type(node.operator).__name__}"
        if isinstance(node.operator, ReshapeOperator):
            assert not keepdims, "a reshape is emitted only in the rank-reducing tier"
        if isinstance(node.operator, MatmulOperator):
            assert len(node.input_value_ids) == 2


# --------------------------------------------------------------------------
# Boundaries and value identifiers are preserved
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keepdims", [True, False])
def test_expansion_preserves_inputs_outputs_and_every_pre_existing_value_id(
    keepdims: bool,
) -> None:
    graph = _traced_mean_graph(keepdims=keepdims)
    value_ids_before = _value_ids(graph)

    expanded = _expand(graph)

    assert expanded.inputs == graph.inputs
    assert expanded.outputs == graph.outputs
    assert value_ids_before <= _value_ids(expanded)


# --------------------------------------------------------------------------
# Purity and determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keepdims", [True, False])
def test_expansion_does_not_mutate_the_input_graph(keepdims: bool) -> None:
    graph = _traced_mean_graph(keepdims=keepdims)
    before = copy.deepcopy(graph)

    _expand(graph)

    assert graph == before


@pytest.mark.parametrize("keepdims", [True, False])
def test_expansion_of_equal_graphs_returns_equal_graphs(keepdims: bool) -> None:
    first = _traced_mean_graph(keepdims=keepdims)
    second = _traced_mean_graph(keepdims=keepdims)
    assert first == second

    assert _expand(first) == _expand(second)
    assert _expand(first) == _expand(copy.deepcopy(first))


# --------------------------------------------------------------------------
# Supported-mean validation and categorized failures
# --------------------------------------------------------------------------

_FAILURE_CASES: list[tuple[str, dict[str, object], str, str]] = [
    (
        "two_operands",
        {"input_value_ids": ["v0", "v0"]},
        "unsupported_reduction",
        "exactly one operand",
    ),
    (
        "operand_typespec_absent",
        {"operand_typespec": None},
        "missing_shape_metadata",
        "operand declares no ranked shape",
    ),
    (
        "operand_typespec_without_shape",
        {"operand_typespec": {"dtype": "f64"}},
        "missing_shape_metadata",
        "operand declares no ranked shape",
    ),
    (
        "operand_typespec_without_dtype",
        {"operand_typespec": {"shape": [3, 5]}},
        "missing_dtype_metadata",
        "operand declares no dtype",
    ),
    (
        "operand_rank_one",
        {
            "operand_typespec": {"dtype": "f64", "shape": [5]},
            "op_params": {"axes": [0], "keepdims": True},
            "output_typespec": {"dtype": "f64", "shape": [1]},
        },
        "unsupported_reduction",
        "has rank",
    ),
    (
        "operand_rank_three",
        {
            "operand_typespec": {"dtype": "f64", "shape": [2, 3, 5]},
            "op_params": {"axes": [0, 1, 2], "keepdims": True},
            "output_typespec": {"dtype": "f64", "shape": [1, 1, 1]},
        },
        "unsupported_reduction",
        "has rank",
    ),
    (
        "symbolic_reduced_dimension",
        {"operand_typespec": {"dtype": "f64", "shape": ["rows", 5]}},
        "unresolved_symbolic_shape",
        "is symbolic",
    ),
    (
        "zero_reduced_dimension",
        {"operand_typespec": {"dtype": "f64", "shape": [0, 5]}},
        "unsupported_reduction",
        "is not positive",
    ),
    (
        "partial_axes",
        {
            "op_params": {"axes": [0], "keepdims": True},
            "output_typespec": {"dtype": "f64", "shape": [1, 5]},
        },
        "unsupported_reduction",
        "partial reduction",
    ),
    (
        "duplicated_axes",
        {"op_params": {"axes": [0, 0], "keepdims": True}},
        "reduction_shape_mismatch",
        "declared axes are malformed",
    ),
    (
        "out_of_range_axis",
        {"op_params": {"axes": [0, 2], "keepdims": True}},
        "reduction_shape_mismatch",
        "declared axes are malformed",
    ),
    (
        "malformed_axes",
        {"op_params": {"axes": "both", "keepdims": True}},
        "reduction_shape_mismatch",
        "declared axes are malformed",
    ),
    (
        "missing_axes",
        {"op_params": {"keepdims": True}},
        "reduction_shape_mismatch",
        "declared axes are malformed",
    ),
    (
        "keepdims_not_a_bool",
        {"op_params": {"axes": [0, 1], "keepdims": 1}},
        "unsupported_reduction",
        "'keepdims' is not a bool",
    ),
    (
        "missing_keepdims",
        {"op_params": {"axes": [0, 1]}},
        "unsupported_reduction",
        "'keepdims' is not a bool",
    ),
    (
        "non_floating_dtype",
        {
            "operand_dtype": "i32",
            "operand_typespec": {"dtype": "i32", "shape": [3, 5]},
            "output_typespec": {"dtype": "i32", "shape": [1, 1]},
        },
        "dtype_not_differentiable",
        "operand dtype is not differentiable",
    ),
    (
        "output_typespec_absent",
        {"output_typespec": None},
        "missing_shape_metadata",
        "output declares no ranked shape",
    ),
    (
        "output_typespec_without_dtype",
        {"output_typespec": {"shape": [1, 1]}},
        "missing_dtype_metadata",
        "output declares no dtype",
    ),
    (
        "output_shape_disagrees_with_the_reduction_rule",
        {
            "op_params": {"axes": [0, 1], "keepdims": False},
            "output_typespec": {"dtype": "f64", "shape": [1, 1]},
        },
        "reduction_shape_mismatch",
        "output declares shape",
    ),
    (
        "output_dtype_disagrees_with_the_operand",
        {"output_typespec": {"dtype": "f32", "shape": [1, 1]}},
        "reduction_shape_mismatch",
        "output declares dtype",
    ),
    (
        "descriptor_key_in_op_params",
        {"op_params": {"axes": [0, 1], "keepdims": True, "fill": 1.0}},
        "unsupported_reduction",
        "fill descriptor key",
    ),
]


@pytest.mark.parametrize(
    ("overrides", "expected_category", "expected_detail"),
    [case[1:] for case in _FAILURE_CASES],
    ids=[case[0] for case in _FAILURE_CASES],
)
def test_an_unsupported_mean_raises_its_category_naming_the_node_and_reason(
    overrides: dict[str, object], expected_category: str, expected_detail: str
) -> None:
    graph = _mean_graph(node_id="n_offender", **overrides)  # type: ignore[arg-type]

    with pytest.raises(AutodiffError) as raised:
        _expand(graph)

    assert raised.value.category == expected_category
    assert "n_offender" in raised.value.message
    assert expected_detail in raised.value.message


def test_a_mean_whose_node_id_is_reserved_names_the_namespace() -> None:
    reserved = _reserved_node_id()
    graph = _mean_graph(node_id=reserved)

    with pytest.raises(AutodiffError) as raised:
        _expand(graph)

    assert raised.value.category == "unsupported_reduction"
    assert reserved in raised.value.message
    assert "reserved expansion namespace" in raised.value.message


def test_a_mean_whose_output_value_id_is_reserved_names_the_namespace() -> None:
    graph = _mean_graph(node_id="n_offender", output_value_id=_reserved_value_id())

    with pytest.raises(AutodiffError) as raised:
        _expand(graph)

    assert raised.value.category == "unsupported_reduction"
    assert "n_offender" in raised.value.message
    assert "reserved expansion namespace" in raised.value.message


def test_a_rejected_mean_leaves_no_partially_rewritten_graph() -> None:
    """Validation runs over every candidate before a single node is emitted."""
    supported = TensorNodeRecord(
        node_id="n0",
        output_value_id="v1",
        operator=MeanOperator(),
        op_params={"axes": [0, 1], "keepdims": True},
        input_value_ids=["v0"],
        output_typespec=_typespec("f64", (1, 1)),
    )
    unsupported = TensorNodeRecord(
        node_id="n_offender",
        output_value_id="v2",
        operator=MeanOperator(),
        op_params={"axes": [0], "keepdims": True},
        input_value_ids=["v0"],
        output_typespec=_typespec("f64", (1, 5)),
    )
    graph = TensorGraph(
        nodes=[supported, unsupported],
        inputs=[("v0", _typespec("f64", (3, 5)))],
        outputs=["v1", "v2"],
    )
    before = copy.deepcopy(graph)

    with pytest.raises(AutodiffError) as raised:
        _expand(graph)

    assert raised.value.category == "unsupported_reduction"
    assert "n_offender" in raised.value.message
    assert graph == before


# --------------------------------------------------------------------------
# Unrelated nodes are carried through identically and in order
# --------------------------------------------------------------------------


def _mixed_graph() -> TensorGraph:
    """A graph whose mean is surrounded by nodes the pass must not touch."""
    column_sum = TensorNodeRecord(
        node_id="n0",
        output_value_id="v1",
        operator=SumOperator(),
        op_params={"axes": [0], "keepdims": True},
        input_value_ids=["v0"],
        output_typespec=_typespec("f64", (1, 5)),
    )
    mean = TensorNodeRecord(
        node_id="n1",
        output_value_id="v2",
        operator=MeanOperator(),
        op_params={"axes": [0, 1], "keepdims": True},
        input_value_ids=["v0"],
        output_typespec=_typespec("f64", (1, 1)),
    )
    reshape = TensorNodeRecord(
        node_id="n2",
        output_value_id="v3",
        operator=ReshapeOperator(),
        op_params={"shape": [5, 1]},
        input_value_ids=["v1"],
        output_typespec=_typespec("f64", (5, 1)),
    )
    return TensorGraph(
        nodes=[column_sum, mean, reshape],
        inputs=[("v0", _typespec("f64", (3, 5)))],
        outputs=["v1", "v2", "v3"],
    )


def test_unmatched_nodes_are_carried_through_identically_and_in_order() -> None:
    graph = _mixed_graph()
    column_sum, mean, reshape = graph.nodes

    expanded = _expand(graph)

    assert len(expanded.nodes) == 7
    assert expanded.nodes[0] == column_sum
    assert expanded.nodes[6] == reshape
    assert [node.node_id for node in expanded.nodes].index("n0") < [
        node.node_id for node in expanded.nodes
    ].index("n2")
    assert not any(isinstance(node.operator, MeanOperator) for node in expanded.nodes)
    assert mean.output_value_id == expanded.nodes[5].output_value_id


# --------------------------------------------------------------------------
# Reserved namespace, collisions, and duplicate identifiers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keepdims", [True, False])
def test_every_minted_identifier_comes_from_the_reserved_namespace(keepdims: bool) -> None:
    from tinychain.autodiff.expansion import (
        EXPANSION_NODE_ID_PREFIX,
        EXPANSION_VALUE_ID_PREFIX,
    )

    graph = _traced_mean_graph(keepdims=keepdims)
    mean_node = graph.nodes[0]

    expanded = _expand(graph)

    node_ids = [node.node_id for node in expanded.nodes]
    assert len(set(node_ids)) == len(node_ids)
    assert all(node_id.startswith(EXPANSION_NODE_ID_PREFIX) for node_id in node_ids)

    minted_value_ids = [
        node.output_value_id
        for node in expanded.nodes
        if node.output_value_id != mean_node.output_value_id
    ]
    assert len(set(minted_value_ids)) == len(minted_value_ids)
    assert all(
        value_id.startswith(EXPANSION_VALUE_ID_PREFIX) for value_id in minted_value_ids
    )
    # Disjoint from the tracer's `v…`/`n…` and the reverse transform's `d…`/`dn…`.
    for identifier in node_ids + minted_value_ids:
        assert not identifier.startswith(("v", "n", "d"))


def test_a_minted_node_id_colliding_with_an_existing_one_is_rejected() -> None:
    reserved = _reserved_node_id()
    graph = _traced_mean_graph(keepdims=True)
    squatter = TensorNodeRecord(
        node_id=reserved,
        output_value_id="v9",
        operator=SumOperator(),
        op_params={"axes": [0], "keepdims": True},
        input_value_ids=["v0"],
        output_typespec=_typespec("f64", (1, 5)),
    )
    seeded = TensorGraph(
        nodes=[squatter, *graph.nodes],
        inputs=list(graph.inputs),
        outputs=[*graph.outputs, "v9"],
    )

    with pytest.raises(AutodiffError) as raised:
        _expand(seeded)

    assert raised.value.category == "malformed_derivative_ir"
    assert reserved in raised.value.message


def test_a_minted_value_id_colliding_with_an_existing_one_is_rejected() -> None:
    reserved = _reserved_value_id()
    graph = _traced_mean_graph(keepdims=True)
    squatter = TensorNodeRecord(
        node_id="n9",
        output_value_id=reserved,
        operator=SumOperator(),
        op_params={"axes": [0], "keepdims": True},
        input_value_ids=["v0"],
        output_typespec=_typespec("f64", (1, 5)),
    )
    seeded = TensorGraph(
        nodes=[squatter, *graph.nodes],
        inputs=list(graph.inputs),
        outputs=[*graph.outputs, reserved],
    )

    with pytest.raises(AutodiffError) as raised:
        _expand(seeded)

    assert raised.value.category == "malformed_derivative_ir"
    assert reserved in raised.value.message


def test_an_input_graph_with_duplicate_node_ids_is_rejected() -> None:
    graph = _traced_mean_graph(keepdims=True)
    duplicate = TensorNodeRecord(
        node_id=graph.nodes[0].node_id,
        output_value_id="v9",
        operator=SumOperator(),
        op_params={"axes": [0], "keepdims": True},
        input_value_ids=["v0"],
        output_typespec=_typespec("f64", (1, 5)),
    )
    seeded = TensorGraph(
        nodes=[*graph.nodes, duplicate],
        inputs=list(graph.inputs),
        outputs=[*graph.outputs, "v9"],
    )

    with pytest.raises(AutodiffError) as raised:
        _expand(seeded)

    assert raised.value.category == "malformed_derivative_ir"
    assert graph.nodes[0].node_id in raised.value.message


def test_an_input_graph_with_duplicate_value_ids_is_rejected() -> None:
    graph = _traced_mean_graph(keepdims=True)
    duplicate = TensorNodeRecord(
        node_id="n9",
        output_value_id=graph.nodes[0].output_value_id,
        operator=SumOperator(),
        op_params={"axes": [0], "keepdims": True},
        input_value_ids=["v0"],
        output_typespec=_typespec("f64", (1, 5)),
    )
    seeded = TensorGraph(
        nodes=[*graph.nodes, duplicate],
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
    )

    with pytest.raises(AutodiffError) as raised:
        _expand(seeded)

    assert raised.value.category == "malformed_derivative_ir"
    assert graph.nodes[0].output_value_id in raised.value.message


# --------------------------------------------------------------------------
# export surface
# --------------------------------------------------------------------------


def test_expand_mean_graph_is_exported_from_the_autodiff_package() -> None:
    from tinychain import autodiff

    assert "expand_mean_graph" in autodiff.__all__
    assert callable(autodiff.expand_mean_graph)
    assert not hasattr(tc, "expand_mean_graph")


# --------------------------------------------------------------------------
# an operand whose declared element count cannot be represented
#
# The reciprocal `1 / (rows * columns)` is the one inexact substitution the
# region makes. A declared shape can name an element count no float can hold,
# and such a mean passes every clause of the predicate -- so without a guard
# the conversion escapes as a bare `OverflowError`, which NFR-128-004 forbids.
# --------------------------------------------------------------------------


def test_a_mean_whose_element_count_cannot_be_converted_is_rejected() -> None:
    rows = columns = 10**200
    graph = _mean_graph(node_id="huge", operand_shape=(rows, columns))

    with pytest.raises(AutodiffError) as raised:
        _expand(graph)

    assert raised.value.category == "unsupported_reduction"
    assert "huge" in raised.value.message
    assert str(rows * columns) in raised.value.message


def test_a_mean_whose_element_count_cannot_be_converted_raises_no_bare_builtin() -> None:
    graph = _mean_graph(node_id="huge", operand_shape=(10**200, 10**200))

    try:
        _expand(graph)
    except AutodiffError:
        pass
    except (OverflowError, KeyError, IndexError, TypeError, ValueError) as exc:
        pytest.fail(f"bare {type(exc).__name__} escaped expand_mean_graph: {exc}")


def test_a_large_but_convertible_element_count_still_expands() -> None:
    """The guard rejects only what cannot be converted; a merely huge count expands."""
    rows = columns = 10**150
    graph = _mean_graph(operand_shape=(rows, columns))

    expanded = _expand(graph)

    scale = expanded.nodes[4]
    assert isinstance(scale.operator, MulOperator)
    assert scale.op_params["right_literal"] == 1.0 / float(rows * columns)


# --------------------------------------------------------------------------
# a minted identifier can never alias a value the artifact merely mentions
#
# Both passes mint into the reserved namespace and both must fail closed on a
# collision (Inv-5). Indexing only *produced* values is not enough: a minted id
# equal to one a node merely reads, or one named only among the artifact's
# declared outputs, would silently rewire that consumer to the emitted node --
# a wrong result with no error at all.
# --------------------------------------------------------------------------


def _graph_with_a_mean_and(
    *,
    extra_nodes: list[TensorNodeRecord] | None = None,
    extra_outputs: list[str] | None = None,
) -> TensorGraph:
    """A traced supported mean, plus whatever extra nodes or outputs a case needs."""
    graph = _traced_mean_graph(keepdims=True)
    return TensorGraph(
        nodes=[*graph.nodes, *(extra_nodes or [])],
        inputs=list(graph.inputs),
        outputs=[*graph.outputs, *(extra_outputs or [])],
    )


def _reading_node(value_id: str) -> TensorNodeRecord:
    """A node that reads *value_id* and nothing else produces it."""
    return TensorNodeRecord(
        node_id="n9",
        output_value_id="v9",
        operator=SumOperator(),
        op_params={"axes": [0], "keepdims": True},
        input_value_ids=[value_id],
        output_typespec=_typespec("f64", (1, 1)),
    )


def _broadcast_scale_program(*, extra_output_gradient: str | None = None):
    """One rewritable broadcast-and-scale chain, minimal and hand-built.

    Built here rather than imported so this module keeps its own inputs; the
    chain is the smallest one satisfying the gradient-path predicate, because
    the pass must match a region before it mints anything at all.
    """
    from tinychain.autodiff import (
        BroadcastOperator,
        DerivativeMetadata,
        DerivativeProgram,
        DivOperator,
    )

    rows, columns = 3, 5
    source = _typespec("f64", (1, 1))
    broadcast_typespec = _typespec("f64", (rows, columns))
    broadcast = TensorNodeRecord(
        node_id="dn0",
        output_value_id="d0",
        operator=BroadcastOperator(),
        op_params={"shape": [rows, columns]},
        input_value_ids=["seed"],
        output_typespec=broadcast_typespec,
    )
    division = TensorNodeRecord(
        node_id="dn1",
        output_value_id="d1",
        operator=DivOperator(),
        op_params={"right_literal": float(rows * columns)},
        input_value_ids=["d0"],
        output_typespec=broadcast_typespec,
    )
    output_gradients: list[str | None] = ["d1"]
    if extra_output_gradient is not None:
        output_gradients.append(extra_output_gradient)
    return DerivativeProgram(
        nodes=[broadcast, division],
        gradients={"v0": "d1"},
        output_gradients=output_gradients,
        metadata=DerivativeMetadata(
            source_graph_id="graph",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=("v0",),
            seed_contract="seed matches output",
        ),
        value_typespecs={"seed": source, "d0": broadcast_typespec, "d1": broadcast_typespec},
    )


def _expand_program(program):
    from tinychain.autodiff import expand_mean_derivative_program

    return expand_mean_derivative_program(program)


def test_the_forward_pass_rejects_a_minted_value_id_a_node_merely_reads() -> None:
    """The consumer must not be silently rewired to the emitted constant."""
    reserved = _reserved_value_id()
    seeded = _graph_with_a_mean_and(
        extra_nodes=[_reading_node(reserved)], extra_outputs=["v9"]
    )

    with pytest.raises(AutodiffError) as raised:
        _expand(seeded)

    assert raised.value.category == "malformed_derivative_ir"
    assert reserved in raised.value.message


def test_the_forward_pass_rejects_a_minted_value_id_named_only_in_the_outputs() -> None:
    reserved = _reserved_value_id()
    seeded = _graph_with_a_mean_and(extra_outputs=[reserved])

    with pytest.raises(AutodiffError) as raised:
        _expand(seeded)

    assert raised.value.category == "malformed_derivative_ir"
    assert reserved in raised.value.message


def test_the_forward_pass_never_rewires_a_consumer_to_an_emitted_node() -> None:
    """The fail-closed half stated as the property it protects: whatever value a
    pre-existing node read before the pass, it still reads afterwards."""
    graph = _traced_mean_graph(keepdims=True)
    reads_before = {node.node_id: list(node.input_value_ids) for node in graph.nodes}

    expanded = _expand(graph)

    for node in expanded.nodes:
        if node.node_id in reads_before:
            assert list(node.input_value_ids) == reads_before[node.node_id]


def test_the_gradient_pass_rejects_a_minted_value_id_a_node_merely_reads() -> None:
    """The reserved id is read by a bystander node, so the chain still matches.

    Feeding it to the chain itself would only prove the predicate declines an
    unmatched region -- the pass would mint nothing at all, and the collision
    this case is about would never arise.
    """
    reserved = _reserved_value_id()
    program = _broadcast_scale_program()
    program.nodes.append(
        TensorNodeRecord(
            node_id="dn9",
            output_value_id="d9",
            operator=MulOperator(),
            op_params={"right_literal": 2.0},
            input_value_ids=[reserved],
            output_typespec=_typespec("f64", (3, 5)),
        )
    )

    with pytest.raises(AutodiffError) as raised:
        _expand_program(program)

    assert raised.value.category == "malformed_derivative_ir"
    assert reserved in raised.value.message


def test_the_gradient_pass_rejects_a_minted_id_named_only_in_the_output_gradients() -> None:
    reserved = _reserved_value_id()
    program = _broadcast_scale_program(extra_output_gradient=reserved)

    with pytest.raises(AutodiffError) as raised:
        _expand_program(program)

    assert raised.value.category == "malformed_derivative_ir"
    assert reserved in raised.value.message
