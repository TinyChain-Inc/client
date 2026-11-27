from __future__ import annotations

import pytest

import tinychain as tc
from tinychain.library import compile_ir
from .support import install_token


def test_branch_execution_preserves_controls_and_captured_locals(tmp_path) -> None:
    class Branches(tc.Library):
        publisher = "example-devco"
        resource_name = "branches"
        version = "0.1.0"

        @tc.post
        def map_result(self, x: tc.Number):
            result_map = tc.state.cond(x.eq(1), {"value": 7}, {"value": 9})
            return result_map["value"]

        @tc.post
        def tuple_result(self, x: tc.Number):
            result_tuple = tc.state.cond(x.eq(1), [7], [9])
            return result_tuple[0]

        @tc.post
        def captured_assignment(self, x: tc.Number):
            previous = x + 2
            if previous > 3:
                selected = previous + 1
            else:
                selected = previous + 10
            return selected

        @tc.post
        def captured_return(self, x: tc.Number):
            previous = x + 2
            if previous > 3:
                return previous + 1
            return previous + 10

        @tc.post
        def unchanged(self, x: tc.Number):
            out = x
            if x.eq(1):
                out = out + 1
            return out

    library = Branches()
    token = install_token(Branches.class_id().path)
    with tc.kernel.with_library(library, data_dir=tmp_path, token=token) as kernel:
        with tc.backend(kernel):
            for x in (1, 2):
                expected = 7 if x == 1 else 9
                assert library.map_result(x=x) == expected
                assert library.tuple_result(x=x) == expected
                assert library.captured_assignment(x=x) == (13 if x == 1 else 5)
                assert library.captured_return(x=x) == (13 if x == 1 else 5)
                assert library.unchanged(x=x) == 2


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
    route = ir[A.class_id().path]["route"]
    assert tc.URI("state", "scalar", "op", "post").path in route


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
    route = ir[A.class_id().path]["route"]
    assert tc.URI("state", "scalar", "op", "post").path in route
