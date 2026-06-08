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

    def dispatch(self, request):
        self.dispatched.append((request.method, request.path, request.headers, request.body))
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
        version = "0.1.0"

        @tc.get
        def hello(self):
            ...

    a = A()

    with tc.backend(mode="deferred"):
        op = a.hello()
        assert isinstance(op, tc.OpRef)
        assert op.method == "GET"
        expected = tc.uri(a, "hello").path
        assert op.path == expected

    with tc.backend(kernel):
        assert a.hello() == "ok"
    assert len(kernel.dispatched) == 1
    method, path, _headers, _body = kernel.dispatched[0]
    assert (method, path) == ("GET", expected)


def test_stub_route_dispatch_forwards_bearer(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)

    kernel = _Kernel()

    class B(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self):
            ...

    b = B()

    with tc.backend(kernel, bearer_token="t"):
        assert b.hello() == "ok"

    expected = tc.uri(b, "hello").path
    assert len(kernel.dispatched) == 1
    method, path, headers, _body = kernel.dispatched[0]
    assert (method, path) == ("GET", expected)
    assert ("authorization", "Bearer t") in list(headers)


def test_stub_route_uses_annotated_return_type(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)

    class C(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

    c = C()
    with tc.backend(mode="deferred"):
        ref = c.hello()
    assert isinstance(ref, tc.String)
    assert isinstance(ref.op, tc.OpRef)


def test_stub_route_accepts_body_and_dispatches(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)

    kernel = _Kernel()

    class D(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

    d = D()
    with tc.backend(mode="deferred"):
        ref = d.hello("World")
    assert isinstance(ref, tc.String)
    assert ref.op.body == "World"

    with tc.backend(kernel):
        assert d.hello("World") == "ok"
    expected = tc.uri(d, "hello").path
    assert len(kernel.dispatched) == 1
    method, path, _headers, body = kernel.dispatched[0]
    assert (method, path) == ("GET", expected)
    assert body is not None


def test_backend_eager_mode_executes_stub_calls_by_default(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)

    kernel = _Kernel()

    class E(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

    e = E()
    with tc.backend(kernel, bearer_token="token-auto"):
        assert e.hello() == "ok"

    expected = tc.uri(e, "hello").path
    assert len(kernel.dispatched) == 1
    method, path, headers, _body = kernel.dispatched[0]
    assert (method, path) == ("GET", expected)
    assert ("authorization", "Bearer token-auto") in list(headers)


def test_backend_mode_deferred_returns_plan(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)

    kernel = _Kernel()

    class F(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

    f = F()
    with tc.backend(kernel, mode="deferred"):
        deferred = f.hello()
        assert isinstance(deferred, tc.String)
        assert tc.execute(deferred) == "ok"

    expected = tc.uri(f, "hello").path
    assert len(kernel.dispatched) == 1
    method, path, _headers, _body = kernel.dispatched[0]
    assert (method, path) == ("GET", expected)


def test_backend_mode_can_be_switched_by_nested_backend_context(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)

    kernel = _Kernel()

    class G(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

    g = G()
    with tc.backend(kernel, bearer_token="token-auto"):
        eager = g.hello()
        assert eager == "ok"

        with tc.backend(kernel, bearer_token="token-auto", mode="deferred"):
            deferred = g.hello()
            assert isinstance(deferred, tc.String)

        assert tc.execute(deferred) == "ok"
        eager_again = g.hello()
        assert eager_again == "ok"

    expected = tc.uri(g, "hello").path
    assert len(kernel.dispatched) == 3
    for method, path, headers, _body in kernel.dispatched:
        assert (method, path) == ("GET", expected)
        assert ("authorization", "Bearer token-auto") in list(headers)


def test_backend_mode_deferred_remains_deferred_when_nested(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)
    kernel = _Kernel()

    class H(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

    h = H()
    with tc.backend(kernel, mode="deferred"):
        plain = h.hello()
        assert isinstance(plain, tc.String)
        with tc.backend(kernel, mode="deferred"):
            nested = h.hello()
            assert isinstance(nested, tc.String)
        assert tc.execute(plain) == "ok"
        assert tc.execute(nested) == "ok"


def test_backend_mode_deferred_preserves_cross_library_dependency_paths(monkeypatch):
    monkeypatch.setattr(tc, "KernelRequest", _Request)
    kernel = _Kernel()
    remote = _Remote()
    monkeypatch.setattr(
        tc_executor,
        "_as_request_target",
        lambda value: value if isinstance(value, _Remote) else remote,
    )

    class Local(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

    class Remote(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        authority = tc.URI.parse("https://api.example.test")

        @tc.get
        def ping(self, name: str) -> tc.String:
            ...

    local = Local()
    remote_lib = Remote()

    with tc.backend(
        kernel=kernel,
        bearer_token="token-123",
        mode="deferred",
    ):
        local_plan = local.hello()
        remote_plan = remote_lib.ping("World")
        assert isinstance(local_plan, tc.String)
        assert isinstance(remote_plan, tc.String)

        assert tc.execute(local_plan) == "ok"
        assert tc.execute(remote_plan) == {"remote": "ok"}

    local_path = tc.uri(local, "hello").path
    assert any(path == local_path for _, path, _, _ in kernel.dispatched)
    assert any(path.startswith("https://api.example.test/") for _, path, _, _ in remote.calls)


def test_backend_routes_authority_qualified_paths(monkeypatch):
    remote = _Remote()
    monkeypatch.setattr(tc_executor, "_as_request_target", lambda _value: remote)

    op = tc.opref.get("https://api.example.test/lib/example-devco/remote/0.1.0/ping")

    with tc.backend(bearer_token="token-xyz"):
        assert tc.execute(op) == {"remote": "ok"}

    assert remote.calls == [
        (
            "GET",
            "https://api.example.test/lib/example-devco/remote/0.1.0/ping",
            None,
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


def test_execute_path_only_without_backend_requires_default_local_host(monkeypatch):
    monkeypatch.setattr(tc, "local", None, raising=False)
    monkeypatch.setattr(tc, "KernelHandle", object(), raising=False)
    op = tc.opref.get("/lib/example-devco/local/0.1.0/ping")

    with pytest.raises(RuntimeError, match="no default local TinyChain host"):
        tc.execute(op)


def test_execute_rejects_already_resolved_values():
    with pytest.raises(TypeError, match="expected OpRef or Ref"):
        tc.execute("ok")


def test_execute_without_backend_runs_authority_qualified_op(monkeypatch):
    remote = _Remote()
    monkeypatch.setattr(tc_executor, "_as_request_target", lambda _value: remote)
    op = tc.opref.post(
        "http://example.test/lib/example-devco/remote/0.1.0/ping",
        body={"name": "World"},
        headers=[("x-trace-id", "abc123")],
    )

    assert tc.execute(op) == {"remote": "ok"}
    assert remote.calls == [
        (
            "POST",
            "http://example.test/lib/example-devco/remote/0.1.0/ping",
            {"name": "World"},
            [("x-trace-id", "abc123")],
        )
    ]


def test_stub_route_emits_authority_qualified_path():
    class Remote(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        authority = tc.URI.parse("https://api.example.test:443")

        @tc.get
        def ping(self) -> tc.String:
            ...

    remote = Remote()
    with tc.backend(mode="deferred"):
        ref = remote.ping()
    assert isinstance(ref, tc.String)
    assert ref.op.path == "https://api.example.test:443/lib/example-devco/remote/0.1.0/ping"


def test_execute_without_backend_runs_authority_qualified_stub(monkeypatch):
    remote_target = _Remote()
    monkeypatch.setattr(tc_executor, "_as_request_target", lambda _value: remote_target)

    class Remote(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        authority = tc.URI.parse("https://api.example.test")

        @tc.get
        def ping(self, name: str) -> tc.String:
            ...

    remote = Remote()
    assert remote.ping("World") == {"remote": "ok"}
    assert remote_target.calls == [
        (
            "GET",
            "https://api.example.test/lib/example-devco/remote/0.1.0/ping",
            "World",
            [],
        )
    ]


def test_backend_authority_op_header_overrides_forwarded_bearer_token(monkeypatch):
    remote = _Remote()
    monkeypatch.setattr(tc_executor, "_as_request_target", lambda _value: remote)
    op = tc.opref.post(
        "https://api.example.test/lib/example-devco/remote/0.1.0/ping",
        body={"name": "World"},
        headers=[("authorization", "Bearer op-token")],
    )

    with tc.backend(bearer_token="token-xyz"):
        assert tc.execute(op) == {"remote": "ok"}

    assert remote.calls == [
        (
            "POST",
            "https://api.example.test/lib/example-devco/remote/0.1.0/ping",
            {"name": "World"},
            [("authorization", "Bearer op-token")],
        )
    ]
