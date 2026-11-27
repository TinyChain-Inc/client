from __future__ import annotations

import pathlib

import tinychain as tc
import tinychain.testing as tc_testing
from .support import install_token, require_cargo


def test_op_reflection_analysis(tmp_path: pathlib.Path) -> None:
    require_cargo()

    def _run():
        class A(tc.Library):
            publisher = "example-devco"
            resource_name = "a"
            version = "0.1.0"

            @tc.get
            def leaf(self, key: tc.String) -> tc.String:
                return key

        class B(tc.Library):
            publisher = "example-devco"
            resource_name = "b"
            version = "0.1.0"

            _a: A = A()

            @tc.get
            def branch(self, key: tc.String) -> tc.String:
                if key.eq("x"):
                    if key.eq("y"):
                        out = "y"
                    else:
                        out = "z"
                else:
                    out = "w"
                return out

        class C(tc.Library):
            publisher = "example-devco"
            resource_name = "c"
            version = "0.1.0"

            @tc.post
            def update(self, item: tc.state.Scalar, max: tc.state.Scalar):
                parts = item.ref_parts()
                is_branch = len(parts) > 0
                if is_branch:
                    max_out = 2
                else:
                    max_out = max
                return {"max": max_out}

            @tc.post
            def reflection_push(self, walk: tc.Map) -> tc.Ref:
                node = walk["items"][walk["idx"]][1]
                todo = walk["todo"] + [[node, 0]]
                return {"items": walk["items"], "idx": walk["idx"] + 1, "todo": todo, "max": walk["max"]}

            @tc.post
            def reflection_branch(self, walk: tc.Map) -> tc.Ref:
                node = walk["todo"][0][0]
                depth = walk["todo"][0][1]
                parts = node.ref_parts()
                tail = walk["todo"][1:]
                todo = tail + [[parts[1], depth + 1], [parts[2], depth + 1]]
                maximum = tc.state.cond(depth + 1 > walk["max"], depth + 1, walk["max"])
                return {"items": walk["items"], "idx": walk["idx"], "todo": todo, "max": maximum}

            @tc.post
            def reflection_leaf(self, walk: tc.Map) -> tc.Ref:
                node = walk["todo"][0][0]
                depth = walk["todo"][0][1]
                children = node.reflect_scalars()
                tail = walk["todo"][1:]
                candidate = tc.state.cond(depth + 1 > walk["max"], depth + 1, walk["max"])
                maximum = tc.state.cond(len(children) > 0, candidate, walk["max"])
                return {"items": walk["items"], "idx": walk["idx"], "todo": tail, "max": maximum}

            @tc.post
            def reflection_pop(self, cxt, walk: tc.Map) -> tc.Ref:
                cxt.node = walk["todo"][0][0]
                cxt.parts = cxt.node.ref_parts()
                return tc.state.cond(
                    cxt.parts.len() > 0,
                    tc.state.PostOpRef(tc.URI(self, "reflection_branch").path, {"walk": walk}),
                    tc.state.PostOpRef(tc.URI(self, "reflection_leaf").path, {"walk": walk}),
                )

            @tc.post
            def reflection_step(self, cxt, walk: tc.Map) -> tc.Ref:
                return tc.state.cond(
                    walk["items"].len() > walk["idx"],
                    tc.state.PostOpRef(tc.URI(self, "reflection_push").path, {"walk": walk}),
                    tc.state.PostOpRef(tc.URI(self, "reflection_pop").path, {"walk": walk}),
                )

            @tc.post
            def cyclomatic_depth(self, cxt, op: tc.state.OpDef) -> tc.Ref:
                # Keep each ordinary operation within the shared structural bound.
                @tc.post
                def condition(state: tc.Map):
                    return (state["items"].len() > state["idx"]).logical_or(state["todo"].len() > 0)

                @tc.post
                def step(state: tc.Map):
                    return self.reflection_step(walk=state)

                cxt.walk = tc.state.while_loop(
                    condition, step, {"items": op.reflect_form(), "idx": 0, "todo": [], "max": 1}
                )
                return {"max": cxt.walk["max"]}

            @tc.post
            def nested_if_step(self, walk: tc.Map) -> tc.Ref:
                if walk["items"][0] == 0:
                    if walk["count"] > 0:
                        next_count = walk["count"] + 1
                    else:
                        next_count = 1
                else:
                    next_count = walk["count"]
                return {"items": walk["items"][1:], "count": next_count}

            @tc.post
            def nested_if_count(self, cxt, items: tc.state.Scalar) -> tc.Ref:
                @tc.post
                def condition(state: tc.Map):
                    return state["items"].len() > 0

                @tc.post
                def step(state: tc.Map):
                    return self.nested_if_step(walk=state)

                cxt.counted = tc.state.while_loop(condition, step, {"items": items, "count": 0})
                return {"count": cxt.counted["count"]}

            @tc.post
            def tuple_loop_supported(self, items: tc.Tuple) -> tc.Number:
                for item in items:
                    observed = item
                return 1

            @tc.post
            def map_loop_supported(self, items: tc.Map) -> tc.String:
                for key in items:
                    observed = key
                return "ok"

        a = A()
        b = B()
        c = C()
        leaf_op = tc.state.PostOpDef([
            ("result", "leaf"),
        ])
        branch_op = tc.state.PostOpDef([
            ("a", 1),
            ("b", 2),
            ("result", 3),
        ])

        token = install_token(A.class_id().path, B.class_id().path, C.class_id().path)
        with tc.kernel.with_library(
            c,
            data_dir=tmp_path,
            token=token,
        ) as kernel:

            for library in (A, B, C):
                resp = tc.install(
                    library,
                    kernel=kernel,
                    data_dir=tmp_path,
                    token=token,
                )
                assert resp.status == 204

            with tc.backend(kernel):
                depth_leaf = tc_testing.run_with_timeout(20, lambda: c.cyclomatic_depth(op=leaf_op))
                assert depth_leaf["max"] == 1
                depth_branch = tc_testing.run_with_timeout(20, lambda: c.cyclomatic_depth(op=branch_op))
                assert depth_branch["max"] >= depth_leaf["max"]
                nested = tc_testing.run_with_timeout(20, lambda: c.nested_if_count(items=[0, 1, 0, 0]))
                assert nested["count"] == 3
                tuple_loop = tc_testing.run_with_timeout(20, lambda: c.tuple_loop_supported(items=[1, 2, 7]))
                assert tuple_loop == 1
                map_loop = tc_testing.run_with_timeout(20, lambda: c.map_loop_supported(items={"b": 1, "a": 2}))
                assert map_loop == "ok"

    tc_testing.run_with_timeout(45, _run)
