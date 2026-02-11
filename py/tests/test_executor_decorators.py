from __future__ import annotations

import json

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


class _Kernel:
    def __init__(self):
        self.dispatched: list[tuple[str, str]] = []
        self.resolved: list[tuple[str, str, object | None, str | None]] = []

    def dispatch(self, request):
        self.dispatched.append((request.method, request.path))
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

    assert kernel.resolved == [("GET", expected, None, None)]


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
    assert kernel.resolved == [("GET", expected, None, "t")]


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

    assert kernel.dispatched == []
    assert len(kernel.resolved) == 1
    method, path, body, token = kernel.resolved[0]
    expected = tc.uri(d, "hello").path
    assert (method, path, token) == ("GET", expected, None)
    assert body is not None
