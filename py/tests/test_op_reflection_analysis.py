from __future__ import annotations

import pathlib

import tinychain as tc

from .support import rjwt_install_token, require_cargo


def test_op_reflection_analysis(tmp_path: pathlib.Path) -> None:
    require_cargo()

    class A(tc.define.Library):
        publisher = "example-devco"
        name = "a"
        version = "0.1.0"

        @tc.get
        def leaf(self, key: tc.String) -> tc.String:
            return key

    class B(tc.define.Library):
        publisher = "example-devco"
        name = "b"
        version = "0.1.0"

        _a: A = None  # type: ignore[assignment]

        @tc.get
        def branch(self, key: tc.String) -> tc.String:
            return tc.cond(
                tc.state.autobox(key).eq("x"),
                tc.cond(tc.state.autobox(key).eq("y"), "y", "z"),
                "w",
            )

    class C(tc.define.Library):
        publisher = "example-devco"
        name = "c"
        version = "0.1.0"

        @tc.post
        def update(self, cxt: tc.state.Context, item: tc.state.Scalar, max: tc.state.Scalar):
            # TODO: Remove explicit `cxt.*` assignments once autograph AST rewriting lands.
            cxt.parts = item.if_parts()
            is_branch = cxt.parts.len().gt(0)
            return {
                "max": tc.cond(is_branch, 2, max),
            }

        @tc.post
        def seed(self, item: tc.state.Scalar, state: tc.state.Scalar) -> tc.state.Scalar:
            pair = [item[1], 0]
            return state.concat([pair])

        @tc.post
        def cond(self, cxt: tc.state.Context, state: tc.state.Scalar) -> tc.state.Scalar:
            cxt.todo = state[0]
            cxt.todo_len = cxt.todo.len()
            return cxt.todo_len.gt(0)

        @tc.post
        def step(self, cxt: tc.state.Context, state: tc.state.Scalar) -> tc.state.Scalar:
            cxt.todo = state[0]
            cxt.max_depth = state[1]
            cxt.head = cxt.todo.head()
            cxt.tail = cxt.todo.tail()
            cxt.node = cxt.head[0]
            cxt.depth = cxt.head[1]

            cxt.parts = cxt.node.if_parts()
            cxt.is_branch = cxt.parts.len().gt(0)
            cxt.inc_depth = cxt.depth.add(1)
            cxt.branch_children = tc.cond(
                cxt.is_branch,
                [
                    [cxt.parts[1], cxt.inc_depth],
                    [cxt.parts[2], cxt.inc_depth],
                ],
                [],
            )
            cxt.next_todo = cxt.tail.concat(cxt.branch_children)
            cxt.next_max = tc.cond(
                cxt.is_branch,
                tc.cond(cxt.inc_depth.gt(cxt.max_depth), cxt.inc_depth, cxt.max_depth),
                cxt.max_depth,
            )
            return [cxt.next_todo, cxt.next_max]

        @tc.post
        def cyclotomic_depth(self, cxt: tc.state.Context, op: tc.state.OpDef) -> tc.Json:
            cxt.op_form = op.reflect_form()
            cxt.seeded = cxt.op_form.reduce(
                item_name="item",
                op=tc.opdef(self.seed),
                value=[],
            )
            cxt.state = [cxt.seeded, 1]
            cxt.final = tc.state.while_loop(
                tc.opdef(self.cond),
                tc.opdef(self.step),
                cxt.state,
            )
            return {"max": cxt.final[1]}

    a = A()
    B._a = a
    b = B()
    c = C()

    token = rjwt_install_token(a.id().path, b.id().path, c.id().path)
    kernel = tc.KernelHandle.with_library_schema_rjwt(
        c.schema_json(),
        token["host"],
        token["actor_id"],
        token["public_key_b64"],
        data_dir=str(tmp_path),
    )

    for library in (a, b, c):
        resp = tc.define.install(
            library,
            kernel=kernel,
            data_dir=tmp_path,
            bearer_token=token["bearer_token"],
        )
        assert resp.status == 204

    with tc.backend(kernel):
        assert tc.execute(c.cyclotomic_depth(a.leaf))["max"] == 1
        assert tc.execute(c.cyclotomic_depth(b.branch))["max"] == 2
