from __future__ import annotations

import tinychain as tc


def test_define_decorator_aliases():
    class A(tc.Library):
        publisher = "example-devco"
        name = "a"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

    a = A()
    ref = a.hello()
    assert isinstance(ref, tc.String)
    assert ref.op.method == "GET"
