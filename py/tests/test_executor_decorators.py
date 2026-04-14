from __future__ import annotations

import json

import pytest
import tinychain as tc
import tinychain.executor as tc_executor


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


class _Kernel:
    def __init__(self):
        self.dispatched: list[tuple[str, str, object, object | None]] = []
        self.resolved: list[tuple[str, str, object | None, str | None]] = []

    def dispatch(self, request):
        self.dispatched.append((request.method, request.path, request.headers, request.body))
        return _Response("ok")

    def resolve_get(self, path: str, body=None, bearer_token=None):
        self.resolved.append(("GET", path, body, bearer_token))
        return _Response("ok")


class _Request:
    def __init__(self, method: str, path: str, headers, body):
        self.method = method
        self.path = path
        self.headers = headers
        self.body = body


class _Remote:
    def __init__(self):
        self.calls: list[tuple[str, str, object | None, list[tuple[str, str]] | None]] = []

    def request(self, method: str, path: object, *, body=None, headers=None, bearer_token=None):
        assert bearer_token is None
        normalized_headers = list(headers) if headers is not None else None
        self.calls.append((method, str(path), body, normalized_headers))
        return {"remote": "ok"}


def test_stub_route_dispatch(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)

    kernel = _Kernel()

    class A(tc.Library):
        publisher = "example-devco"
        name = "a"
        version = "0.1.0"

        @tc.define.get
        def hello(self):
            ...

    a = A()

    op = a.hello()
    assert isinstance(op, tc.OpRef)
    assert op.method == "GET"
    expected = tc.uri(a, "hello").path
    assert op.path == expected

    with tc.backend(kernel):
        assert tc.execute(a.hello()) == "ok"

    assert kernel.resolved == []
    assert len(kernel.dispatched) == 1
    method, path, _headers, _body = kernel.dispatched[0]
    assert (method, path) == ("GET", expected)


def test_stub_route_resolve(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)

    kernel = _Kernel()

    class B(tc.Library):
        publisher = "example-devco"
        name = "b"
        version = "0.1.0"

        @tc.define.get
        def hello(self):
            ...

    b = B()

    with tc.backend(kernel, bearer_token="t"):
        assert tc.execute(b.hello()) == "ok"

    expected = tc.uri(b, "hello").path
    assert kernel.resolved == []
    assert len(kernel.dispatched) == 1
    method, path, headers, _body = kernel.dispatched[0]
    assert (method, path) == ("GET", expected)
    assert ("authorization", "Bearer t") in list(headers)


def test_stub_route_uses_v1_style_return_type(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)

    class C(tc.Library):
        publisher = "example-devco"
        name = "c"
        version = "0.1.0"

        @tc.define.get
        def hello(self) -> tc.String:
            ...

    c = C()
    ref = c.hello()
    assert isinstance(ref, tc.String)
    assert isinstance(ref.op, tc.OpRef)


def test_stub_route_accepts_body_and_dispatches(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)

    kernel = _Kernel()

    class D(tc.Library):
        publisher = "example-devco"
        name = "d"
        version = "0.1.0"

        @tc.define.get
        def hello(self) -> tc.String:
            ...

    d = D()
    ref = d.hello("World")
    assert isinstance(ref, tc.String)
    assert ref.op.body == "World"

    with tc.backend(kernel):
        assert tc.execute(d.hello("World")) == "ok"

    assert kernel.resolved == []
    expected = tc.uri(d, "hello").path
    assert len(kernel.dispatched) == 1
    method, path, _headers, body = kernel.dispatched[0]
    assert (method, path) == ("GET", expected)
    assert body is not None


def test_backend_auto_executes_stub_calls_by_default(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)

    kernel = _Kernel()

    class E(tc.Library):
        publisher = "example-devco"
        name = "e"
        version = "0.1.0"

        @tc.define.get
        def hello(self) -> tc.String:
            ...

    e = E()
    with tc.backend(kernel, bearer_token="token-auto"):
        assert e.hello() == "ok"

    expected = tc.uri(e, "hello").path
    assert kernel.resolved == []
    assert len(kernel.dispatched) == 1
    method, path, headers, _body = kernel.dispatched[0]
    assert (method, path) == ("GET", expected)
    assert ("authorization", "Bearer token-auto") in list(headers)


def test_backend_can_disable_auto_execute(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)

    kernel = _Kernel()

    class F(tc.Library):
        publisher = "example-devco"
        name = "f"
        version = "0.1.0"

        @tc.define.get
        def hello(self) -> tc.String:
            ...

    f = F()
    with tc.backend(kernel, auto_execute=False):
        deferred = f.hello()
        assert isinstance(deferred, tc.String)
        assert tc.execute(deferred) == "ok"

    expected = tc.uri(f, "hello").path
    assert kernel.resolved == []
    assert len(kernel.dispatched) == 1
    method, path, _headers, _body = kernel.dispatched[0]
    assert (method, path) == ("GET", expected)


def test_backend_routes_remote_by_path_prefix(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)

    kernel = _Kernel()
    remote = _Remote()

    class Local(tc.Library):
        publisher = "example-devco"
        name = "local"
        version = "0.1.0"

        @tc.define.get
        def ping(self) -> tc.String:
            ...

    class Remote(tc.Library):
        publisher = "example-devco"
        name = "remote"
        version = "0.1.0"

        @tc.define.get
        def ping(self, name: str) -> tc.String:
            ...

    local = Local()
    remote_lib = Remote()
    remote_prefix = tc.uri(remote_lib).path
    remote_path = tc.uri(remote_lib, "ping").path
    local_path = tc.uri(local, "ping").path

    with tc.backend(
        kernel,
        bearer_token="token-123",
        remotes={remote_prefix: remote},
    ):
        assert remote_lib.ping("World") == {"remote": "ok"}
        assert local.ping() == "ok"

    assert remote.calls == [
        (
            "GET",
            remote_path,
            "World",
            [],
        )
    ]
    assert kernel.resolved == []
    assert len(kernel.dispatched) == 1
    method, path, headers, _body = kernel.dispatched[0]
    assert (method, path) == ("GET", local_path)
    assert ("authorization", "Bearer token-123") in list(headers)


def test_backend_prefers_authority_remote_target(monkeypatch):
    remote_primary = _Remote()
    remote_fallback = _Remote()
    monkeypatch.setattr(
        tc_executor,
        "_as_request_target",
        lambda value: value if isinstance(value, _Remote) else remote_fallback,
    )

    op = tc.opref.get("https://api.example.test/lib/example-devco/remote/0.1.0/ping")

    with tc.backend(
        bearer_token="token-xyz",
        remotes={
            "https://api.example.test": remote_primary,
        },
    ):
        assert tc.execute(op) == {"remote": "ok"}

    assert remote_primary.calls == [
        (
            "GET",
            "https://api.example.test/lib/example-devco/remote/0.1.0/ping",
            None,
            [],
        )
    ]
    assert remote_fallback.calls == []


def test_backend_remote_only_mode():
    remote = _Remote()
    op = tc.opref.post("/lib/example-devco/remote/0.1.0/ping", body={"name": "World"})

    with tc.backend(remote=remote, bearer_token="token-xyz"):
        assert tc.execute(op) == {"remote": "ok"}

    assert remote.calls == [
        (
            "POST",
            "/lib/example-devco/remote/0.1.0/ping",
            {"name": "World"},
            [],
        )
    ]


def test_backend_requires_local_or_remote():
    op = tc.opref.get("/lib/example-devco/remote/0.1.0/ping")

    with tc.backend():
        try:
            tc.execute(op)
            raise AssertionError("expected tc.execute to fail without a local kernel or remote route")
        except RuntimeError as err:
            assert "no local kernel configured and no remote route matched" in str(err)


def test_execute_without_backend_requires_executor():
    op = tc.opref.post(
        "http://example.test/lib/example-devco/remote/0.1.0/ping",
        body={"name": "World"},
        headers=[("x-trace-id", "abc123")],
    )

    with pytest.raises(RuntimeError, match="no active TinyChain executor"):
        tc.execute(op)


def test_stub_route_emits_authority_qualified_path():
    class Remote(tc.Library):
        publisher = "example-devco"
        name = "remote"
        version = "0.1.0"
        authority = tc.URI.parse("https://api.example.test:443")

        @tc.define.get
        def ping(self) -> tc.String:
            ...

    remote = Remote()
    ref = remote.ping()
    assert isinstance(ref, tc.String)
    assert ref.op.path == "https://api.example.test:443/lib/example-devco/remote/0.1.0/ping"


def test_execute_without_backend_rejects_authority_qualified_stub():
    class Remote(tc.Library):
        publisher = "example-devco"
        name = "remote"
        version = "0.1.0"
        authority = tc.URI.parse("https://api.example.test")

        @tc.define.get
        def ping(self, name: str) -> tc.String:
            ...

    remote = Remote()
    with pytest.raises(RuntimeError, match="no active TinyChain executor"):
        tc.execute(remote.ping("World"))


def test_backend_remote_op_header_overrides_forwarded_bearer_token():
    remote = _Remote()
    op = tc.opref.post(
        "/lib/example-devco/remote/0.1.0/ping",
        body={"name": "World"},
        headers=[("authorization", "Bearer op-token")],
    )

    with tc.backend(
        remote=remote,
        bearer_token="token-xyz",
    ):
        assert tc.execute(op) == {"remote": "ok"}

    assert remote.calls == [
        (
            "POST",
            "/lib/example-devco/remote/0.1.0/ping",
            {"name": "World"},
            [("authorization", "Bearer op-token")],
        )
    ]
