from __future__ import annotations

import tinychain as tc


def test_after_preserves_wrapped_type() -> None:
    with tc.state.scoped_context() as cxt:
        dep = tc.state.id("left").eq(1)
        value = tc.String(tc.state.form_of(tc.state.id("right")._string_render({"x": 1})))

        result = tc.state.after(dep, value)

        assert isinstance(result, tc.state.Scalar)
        form = list(cxt.form())
        assert len(form) == 1
        assert form[0][0].startswith("_after")


def test_state_after_returns_scalar_and_binds_dependency() -> None:
    with tc.state.scoped_context() as cxt:
        dep = tc.state.id("a").eq(0)
        then = tc.state.id("b")

        result = tc.state.after(dep, then)

        assert isinstance(result, tc.state.Scalar)
        form = list(cxt.form())
        assert len(form) == 1
        assert form[0][0].startswith("_after")
