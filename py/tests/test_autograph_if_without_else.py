from __future__ import annotations

import pytest
import tinychain as tc
from tinychain.library import compile_ir


def test_if_without_else_allows_rebinding_existing_name() -> None:
    class A(tc.Library):
        publisher = "example-devco"
        resource_name = "a"
        version = "0.1.0"

        @tc.post
        def route(self, x: tc.Number):
            out = x
            if x.eq(1):
                out = x + 1
            return out

    ir = compile_ir(A)
    route = next(route for route in ir["routes"] if route["path"] == "/route")
    assert "opdef" in route


def test_if_without_else_rejects_new_branch_local() -> None:
    class A(tc.Library):
        publisher = "example-devco"
        resource_name = "a"
        version = "0.1.0"

        @tc.post
        def route(self, x: tc.Number):
            if x.eq(1):
                branch_only = x
            return x

    with pytest.raises(Exception, match="if without else may only assign previously bound names"):
        compile_ir(A)


def test_if_return_with_immediate_fallback_return_compiles() -> None:
    class A(tc.Library):
        publisher = "example-devco"
        resource_name = "a"
        version = "0.1.0"

        @tc.post
        def route(self, x: tc.Number):
            if x.eq(1):
                return "one"
            return "other"

    ir = compile_ir(A)
    route = next(route for route in ir["routes"] if route["path"] == "/route")
    assert "opdef" in route
