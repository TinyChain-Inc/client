from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tinychain.autodiff import (
    AddOperator,
    MatmulOperator,
    TensorGraph,
    TensorNodeRecord,
    TransposeOperator,
    generate,
)
from tinychain.autodiff.reverse import _graph_content_hash


PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _add_graph(out_typespec=None):
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=AddOperator(),
        op_params={},
        input_value_ids=["v0", "v1"],
        output_typespec=out_typespec if out_typespec is not None else _typespec([2, 2]),
    )
    return TensorGraph(
        nodes=[node],
        inputs=[("v0", _typespec([2, 2])), ("v1", _typespec([2, 2]))],
        outputs=["v2"],
    )


def test_graph_content_hash_same_structure():
    graph1 = _add_graph()
    graph2 = _add_graph()
    assert _graph_content_hash(graph1) == _graph_content_hash(graph2)


def test_graph_content_hash_different_operator():
    add_graph = _add_graph()
    matmul_node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=MatmulOperator(),
        op_params={},
        input_value_ids=["v0", "v1"],
        output_typespec=None,
    )
    matmul_graph = TensorGraph(
        nodes=[matmul_node],
        inputs=[("v0", None), ("v1", None)],
        outputs=["v2"],
    )
    assert _graph_content_hash(add_graph) != _graph_content_hash(matmul_graph)


def test_graph_content_hash_different_output_typespec():
    graph1 = _add_graph(_typespec([2, 2], "f32"))
    graph2 = _add_graph(_typespec([2, 2], "f64"))
    assert _graph_content_hash(graph1) != _graph_content_hash(graph2)


def test_graph_content_hash_different_input_value_ids():
    graph1 = _add_graph()
    node2 = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=AddOperator(),
        op_params={},
        input_value_ids=["v0", "v2"],
        output_typespec=None,
    )
    graph2 = TensorGraph(
        nodes=[node2],
        inputs=[("v0", None), ("v2", None)],
        outputs=["v2"],
    )
    assert _graph_content_hash(graph1) != _graph_content_hash(graph2)


def test_graph_content_hash_different_inputs():
    graph1 = _add_graph()
    node2 = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=AddOperator(),
        op_params={},
        input_value_ids=["v0", "v1", "v3"],
        output_typespec=None,
    )
    graph2 = TensorGraph(
        nodes=[node2],
        inputs=[("v0", None), ("v1", None), ("v3", None)],
        outputs=["v2"],
    )
    assert _graph_content_hash(graph1) != _graph_content_hash(graph2)


def test_graph_content_hash_includes_graph_outputs():
    graph1 = _add_graph()
    graph2 = TensorGraph(
        nodes=[_add_graph().nodes[0]],
        inputs=[("v0", None), ("v1", None)],
        outputs=["v2", "v3"],
    )
    assert _graph_content_hash(graph1) != _graph_content_hash(graph2)


def test_graph_content_hash_sorted_node_order():
    node_a = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=AddOperator(),
        op_params={},
        input_value_ids=["v0", "v1"],
        output_typespec=None,
    )
    node_b = TensorNodeRecord(
        node_id="n1",
        output_value_id="v3",
        operator=TransposeOperator(),
        op_params={"perm": [1, 0]},
        input_value_ids=["v2"],
        output_typespec=None,
    )
    graph1 = TensorGraph(
        nodes=[node_a, node_b],
        inputs=[("v0", None), ("v1", None)],
        outputs=["v3"],
    )
    graph2 = TensorGraph(
        nodes=[node_b, node_a],
        inputs=[("v0", None), ("v1", None)],
        outputs=["v3"],
    )
    assert _graph_content_hash(graph1) == _graph_content_hash(graph2)


def test_generate_source_graph_id_is_stable_hash():
    graph1 = _add_graph()
    graph2 = _add_graph()
    program1 = generate(graph1, "v2", ["v0", "v1"], "seed")
    program2 = generate(graph2, "v2", ["v0", "v1"], "seed")
    assert program1.metadata.source_graph_id == program2.metadata.source_graph_id
    assert program1.metadata.source_graph_id == _graph_content_hash(graph1)


def test_generate_source_graph_id_not_process_local():
    graph = _add_graph()
    program = generate(graph, "v2", ["v0", "v1"], "seed")
    assert program.metadata.source_graph_id != str(id(graph))


def test_generate_graph_id_override():
    graph = _add_graph()
    program = generate(graph, "v2", ["v0", "v1"], "seed", graph_id="custom_id")
    assert program.metadata.source_graph_id == "custom_id"


def test_generate_empty_graph_id_override():
    graph = _add_graph()
    program = generate(graph, "v2", ["v0", "v1"], "seed", graph_id="")
    assert program.metadata.source_graph_id == ""


def test_existing_callers_without_graph_id_continue_to_work():
    graph = _add_graph(_typespec([2, 2]))
    program = generate(graph, "v2", ["v0", "v1"], "seed")
    assert isinstance(program.metadata.source_graph_id, str)
    assert len(program.metadata.source_graph_id) == 16


def test_source_graph_id_subprocess_determinism():
    graph = _add_graph(_typespec([2, 2]))
    program = generate(graph, "v2", ["v0", "v1"], "seed")
    expected_id = program.metadata.source_graph_id

    script = """
from tinychain.autodiff import AddOperator, TensorGraph, TensorNodeRecord, generate

node = TensorNodeRecord(
    node_id="n0",
    output_value_id="v2",
    operator=AddOperator(),
    op_params={},
    input_value_ids=["v0", "v1"],
    output_typespec={"shape": [2, 2], "dtype": "f32"},
)
graph = TensorGraph(
    nodes=[node],
    inputs=[
        ("v0", {"shape": [2, 2], "dtype": "f32"}),
        ("v1", {"shape": [2, 2], "dtype": "f32"}),
    ],
    outputs=["v2"],
)
program = generate(graph, "v2", ["v0", "v1"], "seed")
print(program.metadata.source_graph_id)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = PY_DIR + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    actual_id = result.stdout.strip()
    assert actual_id == expected_id
