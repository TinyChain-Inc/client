from __future__ import annotations

import json

import pytest
import tinychain as tc


class _Value:
    def __init__(self, payload: object):
        self._payload = payload

    def to_json(self) -> str:
        return json.dumps(self._payload)


class _Body:
    def __init__(self, payload: object):
        self._payload = payload

    def value(self):
        return _Value(self._payload)


class _Response:
    def __init__(self, payload: object, status: int = 200):
        self.status = status
        self.body = _Body(payload)


def test_execute_decodes_string_ref_to_python_str(monkeypatch):
    monkeypatch.setattr(tc, "_dispatch_execute", lambda _op: _Response("hello"))

    ref = tc.String(tc.opref.get("/lib/example-devco/a/0.1.0/hello"))
    result = tc.execute(ref)

    assert isinstance(result, str)
    assert result == "hello"


def test_execute_decodes_number_to_python_int(monkeypatch):
    monkeypatch.setattr(tc, "_dispatch_execute", lambda _op: _Response(7))

    result = tc.execute(tc.opref.get("/lib/example-devco/a/0.1.0/count"))
    assert isinstance(result, int)
    assert result == 7


def test_execute_keeps_json_map_payload_structured(monkeypatch):
    payload = {
        "status": "ok",
        "counts": {"accepted": 2, "rejected": 1},
        "items": ["a", "b"],
    }
    monkeypatch.setattr(tc, "_dispatch_execute", lambda _op: _Response(payload))

    result = tc.execute(tc.opref.get("/lib/example-devco/a/0.1.0/summary"))
    assert isinstance(result, dict)
    assert result == payload
    assert result["counts"]["accepted"] == 2


def test_execute_regression_http_passthrough_without_status(monkeypatch):
    decoded = {"remote": "ok", "latency_ms": 3}
    monkeypatch.setattr(tc, "_dispatch_execute", lambda _op: decoded)

    result = tc.execute(tc.opref.get("https://api.example.test/lib/example-devco/a/0.1.0/ping"))
    assert result is decoded


def test_execute_regression_local_style_response_decode(monkeypatch):
    payload = {"version": "0.1.0", "healthy": True}
    monkeypatch.setattr(tc, "_dispatch_execute", lambda _op: _Response(payload, status=200))

    result = tc.execute(tc.opref.get("/healthz"))
    assert result == payload


def test_execute_decodes_canonical_scalar_array(monkeypatch):
    payload = [1, "x"]
    monkeypatch.setattr(tc, "_dispatch_execute", lambda _op: _Response(payload, status=200))

    result = tc.execute(tc.opref.get("/state/example"))
    assert isinstance(result, list)
    assert result == [1, "x"]


def test_execute_decodes_typed_tensor_when_available(monkeypatch):
    try:
        has_tensor = hasattr(tc, "LocalTensor") and hasattr(tc.LocalTensor, "dense_u64")
    except ImportError:
        has_tensor = False

    if not has_tensor:
        pytest.skip("local Tensor type unavailable")

    payload = {
        tc.uri("state", "collection", "tensor").path: [
            [tc.uri("state", "scalar", "value", "number", "uint", "64").path, [2]],
            [10, 11],
        ]
    }
    monkeypatch.setattr(tc, "_dispatch_execute", lambda _op: _Response(payload, status=200))

    result = tc.execute(tc.opref.get("/state/tensor"))
    assert isinstance(result, tc.LocalTensor)
    assert result.dtype() == "u64"
    assert result.shape() == [2]
    assert result.values() == [10, 11]


def test_execute_decodes_opdef_map_to_python_opdef(monkeypatch):
    payload = tc.state.PostOpDef([("result", 1)]).to_json()
    monkeypatch.setattr(tc, "_dispatch_execute", lambda _op: _Response(payload, status=200))

    result = tc.execute(tc.opref.get("/state/opdef"))
    assert isinstance(result, tc.state.OpDef)
    assert result.method == "POST"
    assert result.form[-1][0] == "result"
