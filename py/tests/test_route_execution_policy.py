from __future__ import annotations

import pytest
import tinychain as tc


def test_route_stub_executes_outside_backend(monkeypatch):
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

    monkeypatch.setattr("tinychain.execute", lambda _value: "executed")

    assert A().hello() == "executed"


def test_route_stub_deferred_mode_returns_symbolic_plan(monkeypatch):
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

    def fail_execute(_value):
        raise AssertionError("route call unexpectedly executed")

    monkeypatch.setattr("tinychain.execute", fail_execute)

    with tc.backend(mode="deferred"):
        result = A().hello()

    assert isinstance(result, tc.String)


def test_route_stub_eager_mode_executes_literal_post_body(monkeypatch):
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def echo(self, x: tc.Number) -> tc.Number:
            return x

    monkeypatch.setattr("tinychain.execute", lambda _value: 7)

    with tc.backend(mode="eager"):
        result = A().echo(1)

    assert result == 7
