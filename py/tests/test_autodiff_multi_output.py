from __future__ import annotations

import numpy as np
import pytest

from tinychain.autodiff import (
    AddOperator,
    AutodiffError,
    ExecutionScheduler,
    MulOperator,
    TensorGraph,
    TensorGraphBuilder,
    TensorNodeRecord,
    generate,
)
from tests.autodiff_execution import NumpyAutodiffDispatcher


def _typespec(shape, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype}


def _multi_output_graph():
    nodes = [
        TensorNodeRecord(
            node_id="n0",
            output_value_id="v2",
            operator=AddOperator(),
            op_params={},
            input_value_ids=["v0", "v1"],
            output_typespec=_typespec((2, 2)),
        ),
        TensorNodeRecord(
            node_id="n1",
            output_value_id="v3",
            operator=MulOperator(),
            op_params={},
            input_value_ids=["v2", "v1"],
            output_typespec=_typespec((2, 2)),
        ),
    ]
    return TensorGraph(
        nodes=nodes,
        inputs=[("v0", _typespec((2, 2))), ("v1", _typespec((2, 2)))],
        outputs=["v2", "v3"],
    )


def test_builder_preserves_single_output_default():
    builder = TensorGraphBuilder()
    builder.record(
        TensorNodeRecord(
            node_id="n0",
            output_value_id="v2",
            operator=AddOperator(),
            op_params={},
            input_value_ids=["v0", "v1"],
        )
    )
    builder.record(
        TensorNodeRecord(
            node_id="n1",
            output_value_id="v3",
            operator=MulOperator(),
            op_params={},
            input_value_ids=["v2", "v1"],
        )
    )

    graph = builder.build()

    assert graph.outputs == ["v3"]


def test_builder_marks_explicit_outputs_in_order_without_duplicates():
    builder = TensorGraphBuilder()
    builder.record(
        TensorNodeRecord(
            node_id="n0",
            output_value_id="v2",
            operator=AddOperator(),
            op_params={},
            input_value_ids=["v0", "v1"],
        )
    )
    builder.record(
        TensorNodeRecord(
            node_id="n1",
            output_value_id="v3",
            operator=MulOperator(),
            op_params={},
            input_value_ids=["v2", "v1"],
        )
    )

    builder.mark_output_value("v2")
    builder.mark_output_value("v3")
    builder.mark_output_value("v2")

    graph = builder.build()

    assert graph.outputs == ["v2", "v3"]


def test_multi_output_reverse_traversal_sums_shared_upstream_paths():
    graph = _multi_output_graph()
    program = generate(graph, ["v2", "v3"], ["v0", "v1"], ["seed_v2", "seed_v3"])

    lhs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rhs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    add_output = lhs + rhs
    seed_v2 = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    seed_v3 = np.array([[2.0, 2.5], [3.0, 3.5]], dtype=np.float32)

    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(
        program,
        values={
            "v0": lhs,
            "v1": rhs,
            "v2": add_output,
            "seed_v2": seed_v2,
            "seed_v3": seed_v3,
        },
    )

    d_lhs, d_rhs = result.gradients
    np.testing.assert_allclose(d_lhs, seed_v2 + seed_v3 * rhs, rtol=1e-5)
    np.testing.assert_allclose(d_rhs, seed_v2 + seed_v3 * rhs + seed_v3 * add_output, rtol=1e-5)
    assert program.metadata.seed_contract == "seed_v2 matches v2; seed_v3 matches v3"


def test_multi_output_ordered_wrt_uses_caller_order():
    graph = _multi_output_graph()
    program = generate(graph, ["v2", "v3"], ["v1", "v0"], ["seed_v2", "seed_v3"])

    assert program.output_gradients == [program.gradients["v1"], program.gradients["v0"]]


def test_multi_output_disconnected_wrt_raises_missing_derivative_behavior():
    graph = _multi_output_graph()

    with pytest.raises(AutodiffError) as exc:
        generate(graph, ["v2", "v3"], ["v9"], ["seed_v2", "seed_v3"])

    assert exc.value.category == "missing_derivative_behavior"


def test_multi_output_requires_one_seed_per_output():
    graph = _multi_output_graph()

    with pytest.raises(TypeError, match="one seed value id per output"):
        generate(graph, ["v2", "v3"], ["v0"], "seed")


def test_autodiff_request_round_trips_multi_output_seed_lists():
    from tinychain.autodiff import AutodiffRequest

    request = AutodiffRequest(
        graph={"nodes": []},
        output_value_id=["v2", "v3"],
        wrt=["v0"],
        seed_value_id=["seed_v2", "seed_v3"],
        tensor_op_contract_version="0.1.0",
        transform_version="0.1.0",
    )

    assert AutodiffRequest.from_dict(request.to_dict()) == request
