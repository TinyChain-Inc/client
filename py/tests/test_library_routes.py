from __future__ import annotations

import inspect

import pytest
import tinychain as tc
from tinychain.autodiff import AutodiffError
from tinychain.library import compile_ir, library_definition
from tinychain.state.scalar import OPDEF_POST


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


def test_library_routes_preserve_all_dict_return_keys():
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def stats(self, x: tc.Number):
            return {"min": x, "max": x}

    ir = compile_ir(A)
    route = next(route for route in ir["routes"] if route["path"] == "/stats")
    opdef = route["opdef"][OPDEF_POST]

    assert [name for name, _ in opdef] == ["min", "max"]


def test_library_routes_ref_typed_mapping_return_is_result_value():
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def stats(self, x: tc.Number) -> tc.Ref:
            stats_map = {"max": x}
            return stats_map

    ir = compile_ir(A)
    route = next(route for route in ir["routes"] if route["path"] == "/stats")
    opdef = route["opdef"][OPDEF_POST]

    names = [name for name, _ in opdef]
    assert "result" in names
    assert "max" not in names


def test_library_route_symbolic_post_body_skips_eager_execute(monkeypatch):
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def echo(self, x: tc.Number) -> tc.Number:
            return x

    def fail_execute(_):
        raise AssertionError("route call unexpectedly executed")

    monkeypatch.setattr("tinychain.execute", fail_execute)

    with tc.state.scoped_context():
        symbolic = tc.state.id("x")
    with tc.backend(mode="eager"):
        result = A().echo(symbolic)

    assert isinstance(result, tc.Number)


def test_grad_is_call_site_transform_stub_not_route_decorator():
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def identity(self, x: tc.Number) -> tc.Number:
            return x

    routes = {route["path"]: route for route in compile_ir(A)["routes"]}

    assert "grad" not in routes["/identity"]
    with pytest.raises(AutodiffError) as exc:
        tc.grad(A().identity, wrt=("v0",))

    assert exc.value.category == "non_differentiable_route"


def test_grad_cannot_be_used_as_route_metadata_decorator():
    class A(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def identity(self, x: tc.Number) -> tc.Number:
            return x

    with pytest.raises(TypeError, match="call-site transform"):

        class B(tc.Library):
            publisher = "example-devco"
            version = "0.1.0"

            @tc.post
            @tc.grad
            def bad(self, x: tc.Number) -> tc.Number:
                return x

    definition = library_definition(A)
    route = definition[A.class_id().path]["identity"]
    assert OPDEF_POST in route


def test_grad_tensor_target_fails_until_route_tracing_is_implemented():
    class NativeTensor:
        def __init__(self, shape):
            self.shape = shape
            self.dtype = "f32"
            self.values = []

        def transpose(self, permutation):
            return NativeTensor([self.shape[i] for i in permutation])

    target = tc.Tensor(native=NativeTensor([2, 3])).transpose([1, 0])

    with pytest.raises(AutodiffError) as exc:
        tc.grad(target, wrt=("v0",))

    assert exc.value.category == "autodiff_not_implemented"


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
            opref = tc.state.GetOpRef(subject)
            return tc.state.Scalar(ref=tc.state.TCRef(opref))

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
