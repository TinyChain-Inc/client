from __future__ import annotations

import pathlib

import tinychain as tc
import tinychain.testing as tc_testing

from .support import rjwt_install_token, require_cargo


class A(tc.Library):
    publisher = "example-devco"
    version = "0.1.0"

    @tc.post
    def double(self, x: tc.Number) -> tc.Number:
        return x + x

    @tc.post
    def triple_chain(self, x: tc.Number) -> tc.Number:
        first = x + x
        return first + x

    @tc.post
    def fanout_quad(self, x: tc.Number) -> tc.Number:
        twice = self.double(x)
        return self.double(twice)


class C(tc.Library):
    publisher = "example-devco"
    version = "0.1.0"

    @tc.post
    def reflected_form_size(self, op: tc.state.OpDef) -> tc.Number:
        return len(op.reflect_form())

    @tc.post
    def autodiff_linear_demo(self, op: tc.state.OpDef, x: tc.Number) -> tc.Number:
        # Minimal transform demo: inspect runtime OpDef, then emit a new derivative OpDef.
        # For the `double(x)` shape, the derivative graph is the constant 2.
        op_form = op.reflect_form()
        reflected_count = len(op_form)
        grad_op = tc.state.PostOpDef([
            ("_reflected_count", reflected_count),
            ("result", 2),
        ])

        return tc.state.autobox([x]).reduce(op=grad_op, value={})

    @tc.post
    def autodiff_chain_demo(self, op: tc.state.OpDef, x: tc.Number) -> tc.Number:
        # Runtime OpDef -> OpDef transform for an add-only chain shape (x + x + x).
        op_form = op.reflect_form()
        reflected_count = len(op_form)
        grad_op = tc.state.PostOpDef([
            ("_reflected_count", reflected_count),
            ("result", 3),
        ])

        return tc.state.autobox([x]).reduce(op=grad_op, value={})

    @tc.post
    def autodiff_fanout_demo(self, op: tc.state.OpDef, x: tc.Number) -> tc.Number:
        # Fan-out-like derivative transform: emit 4x using an explicit add tree.
        op_form = op.reflect_form()
        reflected_count = len(op_form)
        grad_op = tc.state.PostOpDef([
            ("_reflected_count", reflected_count),
            ("x2", tc.state.id("x") + tc.state.id("x")),
            ("result", tc.state.id("x2") + tc.state.id("x2")),
        ])

        return tc.state.autobox([x]).reduce(op=grad_op, value={})

def test_op_transform_autodiff_demo(tmp_path: pathlib.Path) -> None:
    require_cargo()

    a = A()
    c = C()

    token = rjwt_install_token(A.class_id().path, C.class_id().path)
    kernel = tc.kernel.with_library(
        c,
        data_dir=tmp_path,
        token=tc.auth.SignedBearerToken(**token),
    )

    for library in (A, C):
        resp = tc.install(
            library,
            kernel=kernel,
            data_dir=tmp_path,
            token=tc.auth.SignedBearerToken(**token),
        )
        assert resp.status == 204

    with tc.backend(kernel):
        double_nodes = tc_testing.run_with_timeout(20, lambda: c.reflected_form_size(a.double))
        fanout_nodes = tc_testing.run_with_timeout(20, lambda: c.reflected_form_size(a.fanout_quad))
        assert fanout_nodes > double_nodes

        primal = tc_testing.run_with_timeout(20, lambda: a.double(7))
        assert primal == 14

        grad = tc_testing.run_with_timeout(20, lambda: c.autodiff_linear_demo(a.double, 7))
        assert grad == 2

        chain_primal = tc_testing.run_with_timeout(20, lambda: a.triple_chain(7))
        assert chain_primal == 21

        chain_grad = tc_testing.run_with_timeout(20, lambda: c.autodiff_chain_demo(a.triple_chain, 7))
        assert chain_grad == 3

        fanout_grad = tc_testing.run_with_timeout(20, lambda: c.autodiff_fanout_demo(a.fanout_quad, 7))
        assert fanout_grad == 28
