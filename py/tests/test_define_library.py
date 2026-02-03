from __future__ import annotations

import tinychain as tc


def test_define_library_returns_typed_refs():
    class A(tc.define.Library):
        @tc.define.get
        def hello(self) -> tc.String:
            ...

        @tc.define.get
        def raw(self):
            ...

    a = A(publisher="example-devco", name="a", version="0.1.0")

    hello = a.hello()
    assert isinstance(hello, tc.String)
    assert hello.op.method == "GET"
    assert hello.op.path == "/lib/example-devco/a/0.1.0/hello"

    raw = a.raw()
    assert isinstance(raw, tc.OpRef)
    assert raw.method == "GET"
    assert raw.path == "/lib/example-devco/a/0.1.0/raw"

