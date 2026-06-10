from __future__ import annotations

import inspect

import pytest
import tinychain as tc
from tinychain.library import compile_ir


def test_library_routes_return_typed_refs():
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

        @tc.get
        def raw(self):
            ...

    a = A()

    with tc.backend(mode="deferred"):
        hello = a.hello()
        assert isinstance(hello, tc.String)
        assert hello.op.method == "GET"
        assert hello.op.path == tc.uri(a, "hello").path

        raw = a.raw()
        assert isinstance(raw, tc.OpRef)
        assert raw.method == "GET"
        assert raw.path == tc.uri(a, "raw").path


def test_route_type_hints_resolve_to_runtime_value_types():
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self, name: str) -> str:
            ...

        @tc.post
        def mixed(self, text: str, count: int) -> str | int:
            ...

        @tc.post
        def typed(self, n: tc.Number, b: tc.Bool, m: tc.Map, t: tc.Tuple) -> tc.Tuple:
            ...

    hello_sig = inspect.signature(A().hello)
    assert hello_sig.parameters["name"].annotation is tc.String
    assert hello_sig.return_annotation is tc.String
    assert A().hello.__annotations__ == {"name": tc.String, "return": tc.String}

    mixed_sig = inspect.signature(A().mixed)
    assert mixed_sig.parameters["text"].annotation is tc.String
    assert mixed_sig.parameters["count"].annotation is tc.Number
    assert mixed_sig.return_annotation is tc.state.Value

    typed_sig = inspect.signature(A().typed)
    assert typed_sig.parameters["n"].annotation is tc.Number
    assert typed_sig.parameters["b"].annotation is tc.Bool
    assert typed_sig.parameters["m"].annotation is tc.Map
    assert typed_sig.parameters["t"].annotation is tc.Tuple
    assert typed_sig.return_annotation is tc.Tuple


def test_library_routes_return_typed_value_refs():
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def number(self) -> tc.Number:
            ...

        @tc.get
        def flag(self) -> tc.Bool:
            ...

        @tc.get
        def obj(self) -> tc.Map:
            ...

        @tc.get
        def seq(self) -> tc.Tuple:
            ...

    a = A()
    with tc.backend(mode="deferred"):
        assert isinstance(a.number(), tc.Number)
        assert isinstance(a.flag(), tc.Bool)
        assert isinstance(a.obj(), tc.Map)
        assert isinstance(a.seq(), tc.Tuple)


def test_library_routes_compile_opdef_routes():
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def echo(self, x):
            return x

    a = A()
    ir = compile_ir(a)

    route = next(route for route in ir["routes"] if route["path"] == "/echo")
    assert "opdef" in route


def test_library_routes_use_decorator_time_source_capture(monkeypatch):
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def echo(self, x):
            return x

    def missing_source(_form):
        raise OSError("could not get source code")

    monkeypatch.setattr("tinychain._autograph.inspect.getsource", missing_source)

    ir = compile_ir(A)
    route = next(route for route in ir["routes"] if route["path"] == "/echo")
    assert "opdef" in route


def test_library_routes_allow_local_opref_subjects():
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def bad(self):
            subject = "$foo"
            opref = tc.state.OpRef.get(subject)
            return tc.state.Scalar(ref=tc.state.TCRef(op=opref))

    a = A()
    ir = compile_ir(a)
    assert "routes" in ir


def test_route_decorators_do_not_accept_name_override():
    with pytest.raises(TypeError, match="unexpected keyword argument 'name'"):

        class A(tc.Library):
            publisher = "example-devco"
            version = "0.1.0"

            @tc.get(name="hello")
            def hello(self):
                ...


def test_library_subclasses_do_not_accept_name_override():
    class A(tc.Library):
        publisher = "example-devco"
        name = "custom"
        version = "0.1.0"

    with pytest.raises(TypeError, match="name overrides are not supported"):
        A()


def test_library_instances_do_not_accept_dependency_overrides():
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        dependencies = (tc.uri("lib", "example-devco", "b", "0.1.0"),)

    with pytest.raises(TypeError, match="dependencies"):
        A(dependencies=())
