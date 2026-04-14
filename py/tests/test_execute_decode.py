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


def test_execute_decodes_typed_scalar_tuple(monkeypatch):
    payload = {
        tc.uri("state", "scalar", "tuple").path: [
            {tc.uri("state", "scalar", "value", "number").path: 1},
            {tc.uri("state", "scalar", "value", "string").path: "x"},
        ]
    }
    monkeypatch.setattr(tc, "_dispatch_execute", lambda _op: _Response(payload, status=200))

    result = tc.execute(tc.opref.get("/state/example"))
    assert isinstance(result, tuple)
    assert result == (1, "x")


def test_execute_decodes_typed_tensor_when_available(monkeypatch):
    if not hasattr(tc, "Tensor") or not hasattr(tc.Tensor, "dense_u64"):
        pytest.skip("local Tensor type unavailable")

    payload = {
        tc.uri("state", "collection", "tensor").path: [
            [tc.uri("state", "scalar", "value", "number", "uint", "64").path, [2]],
            [10, 11],
        ]
    }
    monkeypatch.setattr(tc, "_dispatch_execute", lambda _op: _Response(payload, status=200))

    result = tc.execute(tc.opref.get("/state/tensor"))
    assert isinstance(result, tc.Tensor)
    assert result.dtype() == "u64"
    assert result.shape() == [2]
    assert result.values() == [10, 11]
