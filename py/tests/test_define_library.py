from __future__ import annotations

import pytest
import tinychain as tc


def test_define_library_returns_typed_refs():
    class A(tc.define.Library):
        publisher = "example-devco"
        name = "a"
        version = "0.1.0"

        @tc.define.get
        def hello(self) -> tc.String:
            ...

        @tc.define.get
        def raw(self):
            ...

    a = A()

    hello = a.hello()
    assert isinstance(hello, tc.String)
    assert hello.op.method == "GET"
    assert hello.op.path == tc.uri(a, "hello").path

    raw = a.raw()
    assert isinstance(raw, tc.OpRef)
    assert raw.method == "GET"
    assert raw.path == tc.uri(a, "raw").path


def test_define_library_compiles_opdef_routes():
    class A(tc.define.Library):
        publisher = "example-devco"
        name = "a"
        version = "0.1.0"

        @tc.define.post
        def echo(self, x):
            return x

    a = A()
    ir = tc.define.compile_ir(a)

    route = next(route for route in ir["routes"] if route["path"] == "/echo")
    assert "opdef" in route


def test_define_library_allows_local_opref_subjects():
    class A(tc.define.Library):
        publisher = "example-devco"
        name = "a"
        version = "0.1.0"

        @tc.define.post
        def bad(self):
            subject = "$foo"
            opref = tc.state.OpRef.get(subject)
            return tc.state.Scalar(ref=tc.state.TCRef(op=opref))

    a = A()
    ir = tc.define.compile_ir(a)
    assert "routes" in ir


def test_route_decorators_do_not_accept_name_override():
    with pytest.raises(TypeError, match="unexpected keyword argument 'name'"):

        class A(tc.define.Library):
            publisher = "example-devco"
            name = "a"
            version = "0.1.0"

            @tc.define.get(name="hello")
            def hello(self):
                ...
