from __future__ import annotations

import json

import numpy as np
import pytest

from tinychain.autodiff import OP_ADD, OP_BROADCAST_REDUCE, TensorNodeRecord, TensorOperator
from tinychain.autodiff.http_dispatcher import TcServerDispatcher, TensorLiteral
from tinychain.autodiff.protocol import AutodiffError


def _response(payload):
    class Response:
        status_code = 200
        text = ""

        def json(self):
            return payload

    return Response()


def test_dispatcher_calls_installed_opdef_route(monkeypatch):
    calls = []

    def post(url, *, data, headers):
        calls.append((url, json.loads(data), headers))
        return _response({"/state/collection/tensor": [["f32", [2]], [4.0, 6.0]]})

    monkeypatch.setattr("tinychain.autodiff.http_dispatcher.requests.post", post)

    dispatcher = TcServerDispatcher(
        "http://localhost:8702",
        route_root="/lib/example-devco/autodiff/0.1.0",
    )
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=OP_ADD,
        op_params={},
        input_value_ids=["v0", "v1"],
    )
    result = dispatcher(
        node,
        [
            TensorLiteral.from_numpy(np.array([1.0, 2.0], dtype=np.float32)),
            TensorLiteral.from_numpy(np.array([3.0, 4.0], dtype=np.float32)),
        ],
    )

    assert calls[0][0] == "http://localhost:8702/lib/example-devco/autodiff/0.1.0/add"
    assert "/state/scalar/op/post" not in calls[0][0]
    assert set(calls[0][1]) == {"x", "y"}
    assert calls[0][2]["content-type"] == "application/json"
    np.testing.assert_array_equal(result, np.array([4.0, 6.0], dtype=np.float32))


def test_dispatcher_calls_installed_broadcast_reduce_route(monkeypatch):
    calls = []

    def post(url, *, data, headers):
        calls.append((url, json.loads(data), headers))
        return _response({"/state/collection/tensor": [["f32", [1, 3]], [5.0, 7.0, 9.0]]})

    monkeypatch.setattr("tinychain.autodiff.http_dispatcher.requests.post", post)

    dispatcher = TcServerDispatcher("http://localhost:8702")
    node = TensorNodeRecord(
        node_id="dn0",
        output_value_id="dv0",
        operator=OP_BROADCAST_REDUCE,
        op_params={"target_shape": [1, 3]},
        input_value_ids=["seed"],
    )
    result = dispatcher(
        node,
        [TensorLiteral.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))],
    )

    assert calls[0][0] == "http://localhost:8702/lib/std/autodiff/0.1.0/broadcast_reduce"
    assert calls[0][1]["target_shape"] == [1, 3]
    assert set(calls[0][1]) == {"x", "target_shape"}
    np.testing.assert_array_equal(result, np.array([[5.0, 7.0, 9.0]], dtype=np.float32))


def test_dispatcher_rejects_unknown_operator_without_if_chain():
    dispatcher = TcServerDispatcher("http://localhost:8702")
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v1",
        operator=TensorOperator("mystery"),
        op_params={},
        input_value_ids=["v0"],
    )

    with pytest.raises(AutodiffError) as exc:
        dispatcher(node, [TensorLiteral.from_numpy(np.array([1.0], dtype=np.float32))])

    assert exc.value.category == "unsupported_operator"


def test_dispatcher_requires_preencoded_tensor_literals():
    dispatcher = TcServerDispatcher("http://localhost:8702")
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v2",
        operator=OP_ADD,
        op_params={},
        input_value_ids=["v0", "v1"],
    )

    with pytest.raises(TypeError, match="TensorLiteral-compatible"):
        dispatcher(node, [np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)])
