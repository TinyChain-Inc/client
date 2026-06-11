from __future__ import annotations

import pytest
import tinychain as tc
from tinychain.library import compile_ir


def test_autograph_rejects_non_tinychain_global_call() -> None:
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def route(self, x: tc.Number):
            return urllib.request.urlopen("https://example.com")

    with pytest.raises(Exception, match="unsupported name urllib in autograph expression"):
        compile_ir(A)


def test_autograph_rejects_tensorflow_style_global_symbol() -> None:
    tf = object()

    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def route(self, x: tc.Number):
            return tf.add(x, x)

    with pytest.raises(Exception, match="unsupported name tf in autograph expression"):
        compile_ir(A)


def test_autograph_rejects_jax_style_global_symbol() -> None:
    jax = object()

    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def route(self, x: tc.Number):
            return jax.numpy.sin(x)

    with pytest.raises(Exception, match="unsupported name jax in autograph expression"):
        compile_ir(A)


def test_autograph_allows_tinychain_expression_graphs() -> None:
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def route(self, x: tc.Number):
            y = x + 1
            return tc.state.cond(x.eq(1), y, x)

    ir = compile_ir(A)
    route = next(route for route in ir["routes"] if route["path"] == "/route")
    assert "opdef" in route
