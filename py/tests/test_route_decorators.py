from __future__ import annotations

import tinychain as tc


def test_route_decorators_return_typed_refs():
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

    a = A()
    with tc.backend(mode="deferred"):
        ref = a.hello()
    assert isinstance(ref, tc.String)
    assert ref.op.method == "GET"
