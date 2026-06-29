from __future__ import annotations

from tinychain.autodiff.graph import AddOperator, TensorGraph, TensorNodeRecord


def test_tensor_node_record_op_params_copy() -> None:
    op_params = {"alpha": 1.0}
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v1",
        operator=AddOperator(),
        op_params=op_params,
        input_value_ids=["v0"],
    )
    op_params["alpha"] = 99.0
    assert node.op_params["alpha"] == 1.0


def test_tensor_node_record_input_value_ids_copy() -> None:
    input_ids = ["v0", "v1"]
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=AddOperator(),
        op_params={},
        input_value_ids=input_ids,
    )
    input_ids.append("v99")
    assert node.input_value_ids == ["v0", "v1"]


def test_tensor_graph_nodes_copy() -> None:
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v1",
        operator=AddOperator(),
        op_params={},
        input_value_ids=["v0"],
    )
    nodes = [node]
    graph = TensorGraph(nodes=nodes, inputs=[("v0", None)], outputs=["v1"])
    nodes.clear()
    assert len(graph.nodes) == 1


def test_tensor_graph_inputs_copy() -> None:
    inputs = [("v0", None)]
    graph = TensorGraph(nodes=[], inputs=inputs, outputs=[])
    inputs.append(("v1", None))
    assert graph.inputs == [("v0", None)]


def test_tensor_graph_outputs_copy() -> None:
    outputs = ["v1"]
    graph = TensorGraph(nodes=[], inputs=[], outputs=outputs)
    outputs.append("v99")
    assert graph.outputs == ["v1"]
