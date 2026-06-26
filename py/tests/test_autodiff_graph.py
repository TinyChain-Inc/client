import pytest
import tinychain as tc
from tinychain.autodiff import (
    AddOperator,
    BroadcastReduceOperator,
    MatmulOperator,
    TensorGraph,
    TransposeOperator,
    TensorGraphBuilder,
    TensorNodeRecord,
    get_active_builder,
)
from tinychain.autodiff.graph import operator_for_route


def _make_tensor(name: str) -> tc.Tensor:
    return tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef(name)))


def _json(value):
    return tc.state.form_of(value).to_json()


@pytest.fixture
def x():
    return _make_tensor("x")


@pytest.fixture
def y():
    return _make_tensor("y")


@pytest.fixture
def w():
    return _make_tensor("w")


@pytest.fixture
def lhs():
    return _make_tensor("lhs")


@pytest.fixture
def rhs():
    return _make_tensor("rhs")


@pytest.mark.parametrize(("operator_type", "expected"), [
    (AddOperator, "add"),
    (BroadcastReduceOperator, "broadcast_reduce"),
    (MatmulOperator, "matmul"),
    (TransposeOperator, "transpose"),
])
def test_operator_route_name(operator_type, expected):
    assert operator_type().route_name == expected


@pytest.mark.parametrize(("route_name", "operator_type"), [
    ("add", AddOperator),
    ("broadcast_reduce", BroadcastReduceOperator),
    ("matmul", MatmulOperator),
    ("transpose", TransposeOperator),
])
def test_operator_for_route_creates_fresh_instance(route_name, operator_type):
    first = operator_for_route(route_name)
    second = operator_for_route(route_name)

    assert isinstance(first, operator_type)
    assert first is not second


class TestBuilderContext:
    def test_no_active_builder_outside_context(self):
        assert get_active_builder() is None

    def test_active_builder_inside_context(self):
        with TensorGraphBuilder() as builder:
            assert get_active_builder() is builder

    def test_builder_deactivated_after_context_exit(self):
        with TensorGraphBuilder() as builder:
            pass
        assert get_active_builder() is None

    def test_nested_contexts_restore_outer_on_exit(self):
        with TensorGraphBuilder() as outer:
            with TensorGraphBuilder() as inner:
                assert get_active_builder() is inner
            assert get_active_builder() is outer
        assert get_active_builder() is None


class TestAddRecording:
    def test_add_records_single_node(self, x, y):
        with TensorGraphBuilder() as builder:
            x + y

        graph = builder.build()
        assert len(graph.nodes) == 1
        node = graph.nodes[0]
        assert isinstance(node.operator, AddOperator)
        assert node.op_params == {}
        assert len(node.input_value_ids) == 2

    def test_add_assigns_distinct_value_ids(self, x, y):
        with TensorGraphBuilder() as builder:
            x + y

        node = builder.build().nodes[0]
        lhs_vid, rhs_vid = node.input_value_ids
        assert lhs_vid != rhs_vid
        assert node.output_value_id not in (lhs_vid, rhs_vid)

    def test_add_inputs_not_in_node_outputs(self, x, y):
        with TensorGraphBuilder() as builder:
            x + y

        graph = builder.build()
        input_vids = {vid for vid, _ in graph.inputs}
        assert set(graph.nodes[0].input_value_ids) == input_vids

    def test_add_output_value_id_matches_graph_outputs(self, x, y):
        with TensorGraphBuilder() as builder:
            x + y

        graph = builder.build()
        assert graph.outputs == [graph.nodes[0].output_value_id]

    def test_add_fallthrough_outside_context(self, x, y):
        assert _json(x + y) == {"$x/add": {"r": {"$y": []}}}


class TestMatmulRecording:
    def test_matmul_records_single_node(self, lhs, rhs):
        with TensorGraphBuilder() as builder:
            lhs @ rhs

        graph = builder.build()
        assert len(graph.nodes) == 1
        node = graph.nodes[0]
        assert isinstance(node.operator, MatmulOperator)
        assert len(node.input_value_ids) == 2

    def test_matmul_assigns_distinct_value_ids(self, lhs, rhs):
        with TensorGraphBuilder() as builder:
            lhs @ rhs

        node = builder.build().nodes[0]
        lhs_vid, rhs_vid = node.input_value_ids
        assert lhs_vid != rhs_vid
        assert node.output_value_id not in (lhs_vid, rhs_vid)

    def test_matmul_fallthrough_outside_context(self, lhs, rhs):
        assert _json(lhs @ rhs) == {"$lhs/matmul": {"r": {"$rhs": []}}}


class TestTransposeRecording:
    def test_transpose_records_single_node(self, x):
        with TensorGraphBuilder() as builder:
            x.transpose([1, 0])

        graph = builder.build()
        assert len(graph.nodes) == 1
        node = graph.nodes[0]
        assert isinstance(node.operator, TransposeOperator)
        assert node.op_params == {"perm": [1, 0]}
        assert len(node.input_value_ids) == 1

    def test_transpose_assigns_distinct_value_ids(self, x):
        with TensorGraphBuilder() as builder:
            x.transpose([1, 0])

        node = builder.build().nodes[0]
        assert node.output_value_id != node.input_value_ids[0]

    def test_transpose_fallthrough_outside_context(self, x):
        assert _json(x.transpose([1, 0])) == {"$x/transpose": [[1, 0]]}


class TestMultiNodeGraph:
    def test_two_ops_produce_two_nodes(self, x, y, w):
        with TensorGraphBuilder() as builder:
            x + y
            x @ w

        graph = builder.build()
        assert len(graph.nodes) == 2
        assert isinstance(graph.nodes[0].operator, AddOperator)
        assert isinstance(graph.nodes[1].operator, MatmulOperator)

    def test_chained_matmul_then_transpose_records_data_flow(self, lhs, rhs):
        with tc.state.scoped_context():
            with TensorGraphBuilder() as builder:
                result = (lhs @ rhs).transpose([1, 0])

        graph = builder.build()
        matmul, transpose = graph.nodes
        assert isinstance(matmul.operator, MatmulOperator)
        assert isinstance(transpose.operator, TransposeOperator)
        assert transpose.input_value_ids == [matmul.output_value_id]
        assert graph.outputs == [transpose.output_value_id]
        assert result is not None

    def test_shared_input_registered_once(self, x, y, w):
        with TensorGraphBuilder() as builder:
            x + y
            x @ w

        graph = builder.build()
        assert graph.nodes[0].input_value_ids[0] == graph.nodes[1].input_value_ids[0]

    def test_build_returns_tensor_graph_type(self, x, y):
        with TensorGraphBuilder() as builder:
            x + y
        assert isinstance(builder.build(), TensorGraph)


@pytest.mark.parametrize("op_fn,expected_json", [
    (lambda x, y: x + y,        {"$x/add":    {"r": {"$y": []}}}),
    (lambda x, y: x @ y,        {"$x/matmul": {"r": {"$y": []}}}),
    (lambda x, _: x.transpose([1, 0]), {"$x/transpose": [[1, 0]]}),
])
def test_symbolic_form_unchanged_outside_builder(op_fn, expected_json):
    x = _make_tensor("x")
    y = _make_tensor("y")
    assert _json(op_fn(x, y)) == expected_json
