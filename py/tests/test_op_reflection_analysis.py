from __future__ import annotations

import pathlib

import tinychain as tc
import tinychain.testing as tc_testing

from .support import rjwt_install_token, require_cargo


def test_op_reflection_analysis(tmp_path: pathlib.Path) -> None:
    require_cargo()

    def _run():
        class A(tc.Library):
            publisher = "example-devco"
            version = "0.1.0"

            @tc.get
            def leaf(self, key: tc.String) -> tc.String:
                return key

        class B(tc.Library):
            publisher = "example-devco"
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
            def cyclotomic_depth(self, op: tc.state.OpDef) -> tc.Ref:
                op_form = op.reflect_form()
                state = {"items": op_form, "idx": 0, "todo": [], "max": 1}
                while (len(state["items"]) > state["idx"]).logical_or(len(state["todo"]) > 0):
                    has_items = len(state["items"]) > state["idx"]
                    if has_items:
                        node = state["items"][state["idx"]][1]
                        depth = 0
                        parts = node.ref_parts()
                        has_children = len(parts) > 0
                        tail = state["todo"][1:]
                        children = []
                        children_ok = False
                        next_todo = state["todo"] + [[node, depth]]
                        next_idx = state["idx"] + 1
                        next_max = state["max"]
                    else:
                        node = state["todo"][0][0]
                        depth = state["todo"][0][1]
                        parts = node.ref_parts()
                        has_children = len(parts) > 0
                        tail = state["todo"][1:]
                        if has_children:
                            children = []
                            children_ok = False
                            next_todo = tail + [[parts[1], depth + 1], [parts[2], depth + 1]]
                            if depth + 1 > state["max"]:
                                next_max = depth + 1
                            else:
                                next_max = state["max"]
                        else:
                            children = node.reflect_scalars()
                            children_ok = len(children) > 0
                            if children_ok:
                                next_todo = tail
                                if depth + 1 > state["max"]:
                                    next_max = depth + 1
                                else:
                                    next_max = state["max"]
                            else:
                                next_todo = tail
                                next_max = state["max"]
                        next_idx = state["idx"]
                    state = {"items": state["items"], "idx": next_idx, "todo": next_todo, "max": next_max}
                return {"max": state["max"]}

            @tc.post
            def nested_if_count(self, items: tc.state.Scalar) -> tc.Ref:
                state = {"items": items, "count": 0}
                while len(state["items"]) > 0:
                    head = state["items"][0]
                    rest = state["items"][1:]
                    if head == 0:
                        if state["count"] > 0:
                            next_count = state["count"] + 1
                        else:
                            next_count = 1
                    else:
                        next_count = state["count"]
                    state = {"items": rest, "count": next_count}
                return {"count": state["count"]}

        a = A()
        b = B()
        c = C()

        token = rjwt_install_token(A.class_id().path, B.class_id().path, C.class_id().path)
        kernel = tc.KernelHandle.with_library_schema_rjwt(
            C.class_schema_json(),
            token["host"],
            token["actor_id"],
            token["public_key_b64"],
            data_dir=str(tmp_path),
        )

        for library in (A, B, C):
            resp = tc.install(
                library,
                kernel=kernel,
                data_dir=tmp_path,
                bearer_token=token["bearer_token"],
            )
            assert resp.status == 204

        with tc.backend(kernel):
            depth_leaf = tc_testing.run_with_timeout(20, lambda: c.cyclotomic_depth(a.leaf))
            assert depth_leaf["max"] == 1
            depth_branch = tc_testing.run_with_timeout(20, lambda: c.cyclotomic_depth(b.branch))
            assert depth_branch["max"] == 2
            nested = tc_testing.run_with_timeout(20, lambda: c.nested_if_count([0, 1, 0, 0]))
            assert nested["count"] == 3

    tc_testing.run_with_timeout(45, _run)
