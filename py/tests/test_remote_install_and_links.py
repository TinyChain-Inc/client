from __future__ import annotations

import json

import pytest
import tinychain as tc


class Greeter(tc.Library):
    publisher = "example-devco"
    version = "0.1.0"

    @tc.get
    def hello(self, name: str) -> tc.String:
        return tc.String("Hello, {{name}}!").render(name=name)


class _Response:
    status_code = 204
    text = ""

    def json(self):
        return None


class _EmptyCreatedResponse:
    status_code = 201
    text = ""

    def json(self):
        raise ValueError("empty")


def test_host_carries_default_auth_and_builds_route_url(monkeypatch):
    calls = []

    def fake_request(method, url, *, data=None, headers=None):
        calls.append((method, url, data, headers))
        return _Response()

    monkeypatch.setattr("tinychain.host.requests.request", fake_request)

    token = tc.auth.SignedBearerToken(
        host="https://tokens.example",
        actor_id="demo",
        public_key_b64="pub",
        secret_key_b64="secret",
        bearer_token="token-123",
    )
    host = tc.Host("https://testnet.example", token=token)
    greeter = Greeter()

    assert (
        host.url(greeter, "hello", name="Ada Lovelace")
        == "https://testnet.example/lib/example-devco/greeter/0.1.0/hello?key=%7B%22name%22%3A%22Ada+Lovelace%22%7D"
    )

    host.request("GET", tc.uri(greeter, "hello"))
    assert calls == [
        (
            "GET",
            "https://testnet.example/lib/example-devco/greeter/0.1.0/hello",
            None,
            {
                "authorization": "Bearer token-123",
                "accept": "application/json",
            },
        )
    ]


def test_install_python_library_to_remote_uses_canonical_payload_and_auth(monkeypatch):
    calls = []

    def fake_request(method, url, *, data=None, headers=None):
        calls.append((method, url, data, headers))
        return _Response()

    monkeypatch.setattr("tinychain.host.requests.request", fake_request)

    token = tc.auth.SignedBearerToken(
        host="https://tokens.example",
        actor_id="demo",
        public_key_b64="pub",
        secret_key_b64="secret",
        bearer_token="token-123",
    )
    host = tc.Host("https://testnet.example", token=token)

    assert tc.install(Greeter, remote=host) is None
    assert len(calls) == 1
    method, url, data, headers = calls[0]
    assert (method, url) == ("PUT", "https://testnet.example/lib")
    assert headers["authorization"] == "Bearer token-123"
    assert headers["content-type"] == "application/json"
    payload = json.loads(data.decode("utf-8"))
    assert list(payload) == ["/lib/example-devco/greeter/0.1.0"]
    assert list(payload["/lib/example-devco/greeter/0.1.0"]) == ["hello"]
    encoded_ir = json.dumps(payload)
    assert "$name" in encoded_ir
    assert "/render" in encoded_ir
    assert "schema" not in payload
    assert "artifacts" not in payload


def test_host_accepts_successful_empty_response(monkeypatch):
    monkeypatch.setattr(
        "tinychain.host.requests.request",
        lambda method, url, *, data=None, headers=None: _EmptyCreatedResponse(),
    )

    assert tc.Host("https://testnet.example").request("PUT", "/lib", body={}) is None


def test_string_value_render_is_the_only_value_render_surface():
    rendered = tc.String("Hello, {{name}}!").render(name="Ada")
    assert isinstance(rendered, tc.String)
    assert rendered.to_json() == "Hello, Ada!"
    assert hasattr(tc.state.Value.string("plain"), "render")
    assert not hasattr(tc.state.Value.number(1), "render")


def test_host_rejects_client_transaction_controls():
    host = tc.Host("https://testnet.example")
    greeter = Greeter()

    with pytest.raises(ValueError, match="transaction controls"):
        host.url(greeter, "hello", txn_id="caller-owned")

    with pytest.raises(ValueError, match="transaction controls"):
        host.request("POST", "/lib?txn_id=caller-owned")
