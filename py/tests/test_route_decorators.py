from __future__ import annotations

import tinychain as tc


def test_route_decorators_return_typed_refs():
    class A(tc.Library):
        publisher = "example-devco"
        resource_name = "a"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

    a = A()
    with tc.backend(mode="deferred"):
        ref = a.hello()
    assert isinstance(ref, tc.String)
    assert ref.op.method == "GET"


def test_standalone_decorator_lowers_runtime_ref_to_canonical_ir():
    @tc.get
    def auth_context() -> tc.Ref:
        return tc.auth.context()

    assert auth_context.to_json() == {
        "/state/scalar/op/get": [
            "key",
            [["result", {"/host/auth/context": [None]}]],
        ]
    }
