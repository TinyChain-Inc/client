from __future__ import annotations

import inspect

import pytest
import tinychain as tc
from tinychain.autodiff import AutodiffError
from tinychain.library import compile_ir, library_definition

def test_library_routes_return_typed_refs():
    class A(tc.Library):
        publisher = "example-devco"
        resource_name = "a"
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
        assert hello.op.path == tc.URI(a, "hello").path

        raw = a.raw()
        assert isinstance(raw, tc.OpRef)
        assert raw.method == "GET"
        assert raw.path == tc.URI(a, "raw").path


def test_route_type_hints_resolve_to_runtime_value_types():
    class A(tc.Library):
        publisher = "example-devco"
        resource_name = "a"
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
        resource_name = "a"
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
        resource_name = "a"
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
        resource_name = "a"
        version = "0.1.0"

        @tc.post
        def stats(self, x: tc.Number):
            return {"min": x, "max": x}

    ir = compile_ir(A)
    route = next(route for route in ir["routes"] if route["path"] == "/stats")
    opdef = route["opdef"][tc.URI("state", "scalar", "op", "post").path]

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
    opdef = route["opdef"][tc.URI("state", "scalar", "op", "post").path]

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
        resource_name = "a"
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
        resource_name = "a"
        version = "0.1.0"

        @tc.post
        def identity(self, x: tc.Number) -> tc.Number:
            return x

    with pytest.raises(TypeError, match="call-site transform"):

        class B(tc.Library):
            publisher = "example-devco"
            resource_name = "b"
            version = "0.1.0"

            @tc.post
            @tc.grad
            def bad(self, x: tc.Number) -> tc.Number:
                return x

    definition = library_definition(A)
    route = definition[A.class_id().path]["identity"]
    assert tc.URI("state", "scalar", "op", "post").path in route


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
        resource_name = "a"
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


def test_library_routes_accept_uri_subjects_for_oprefs():
    class A(tc.Library):
        publisher = "example-devco"
        resource_name = "a"
        version = "0.1.0"

        @tc.post
        def bad(self):
            subject = tc.URI("state", "scalar", "value")
            opref = tc.state.GetOpRef(subject)
            return tc.state.Scalar(ref=tc.state.TCRef(opref))

    a = A()
    ir = compile_ir(a)
    assert any(route["path"] == "/bad" for route in ir["routes"])


def test_route_decorators_do_not_accept_name_override():
    with pytest.raises(TypeError, match="unexpected keyword argument 'name'"):

        class A(tc.Library):
            publisher = "example-devco"
            resource_name = "a"
            version = "0.1.0"

            @tc.get(name="hello")
            def hello(self):
                ...


def test_explicit_resource_name_produces_canonical_id():
    class ExampleClient(tc.Library):
        publisher = "example-devco"
        resource_name = "example-client"
        version = "1.2.3"

    assert ExampleClient.class_id().path == "/lib/example-devco/example-client/1.2.3"
    assert ExampleClient().id().path == "/lib/example-devco/example-client/1.2.3"


def test_class_name_does_not_influence_resource_identity():
    class OrdinaryClient(tc.Library):
        publisher = "example-devco"
        resource_name = "custom-resource"
        version = "0.1.0"

    assert OrdinaryClient.class_id().path == "/lib/example-devco/custom-resource/0.1.0"
    assert OrdinaryClient().id().path == "/lib/example-devco/custom-resource/0.1.0"


def test_missing_resource_name_fails_construction_and_class_id():
    class MissingResourceName(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

    with pytest.raises(TypeError, match="resource_name"):
        MissingResourceName.class_id()

    with pytest.raises(TypeError, match="resource_name"):
        MissingResourceName()


def test_raw_name_is_not_library_identity():
    class OnlyName(tc.Library):
        publisher = "example-devco"
        name = "custom"
        version = "0.1.0"

    with pytest.raises(TypeError, match="resource_name"):
        OnlyName.class_id()

    with pytest.raises(TypeError, match="resource_name"):
        OnlyName()


@pytest.mark.parametrize("raw_name", ["custom", None, 123, object()])
def test_raw_name_is_rejected_regardless_of_value_or_type(raw_name):
    Library = type(
        "NameAndResourceName",
        (tc.Library,),
        {
            "publisher": "example-devco",
            "name": raw_name,
            "resource_name": "valid-resource",
            "version": "0.1.0",
        },
    )

    with pytest.raises(TypeError, match="'name' field is not supported"):
        Library.class_id()

    with pytest.raises(TypeError, match="'name' field is not supported"):
        Library()


def test_inherited_raw_name_cannot_bypass_validation():
    class Base(tc.Library):
        publisher = "example-devco"
        name = 123
        resource_name = "base-resource"
        version = "0.1.0"

    class Child(Base):
        resource_name = "child-resource"

    with pytest.raises(TypeError, match="'name' field is not supported"):
        Child.class_id()


def test_route_method_named_name_remains_valid():
    class RouteName(tc.Library):
        publisher = "example-devco"
        resource_name = "route-name"
        version = "0.1.0"

        @tc.get
        def name(self) -> tc.String:
            ...

    assert RouteName.class_id().path == "/lib/example-devco/route-name/0.1.0"
    assert RouteName().id().path == "/lib/example-devco/route-name/0.1.0"

    ir_paths = [route["path"] for route in compile_ir(RouteName)["routes"]]
    assert "/name" in ir_paths


@pytest.mark.parametrize("field", ("publisher", "resource_name", "name", "version"))
def test_library_instances_reject_constructor_metadata_overrides(field):
    class A(tc.Library):
        publisher = "example-devco"
        resource_name = "a"
        version = "0.1.0"

    with pytest.raises(
        TypeError, match="manifest metadata must be declared on the class"
    ):
        A(**{field: "override"})


def test_remote_install_definition_uses_resource_name(monkeypatch):
    from tinychain.library import install

    class RemoteLib(tc.Library):
        publisher = "applied-physics"
        resource_name = "remote-lib"
        version = "0.1.0"

        @tc.get
        def ping(self) -> tc.String:
            ...

    captured = {}

    class FakeHost:
        def __init__(self, *_args, **_kwargs):
            ...

        def request(self, method, path, *, body):
            captured["method"] = method
            captured["path"] = path
            captured["body"] = body
            return "ok"

    monkeypatch.setattr("tinychain.host.Host", FakeHost)

    result = install(RemoteLib, remote="https://api.example.test", token="t")

    assert result == "ok"
    assert list(captured["body"].keys()) == [
        "/lib/applied-physics/remote-lib/0.1.0"
    ]


def test_library_instances_do_not_accept_dependency_overrides():
    class A(tc.Library):
        publisher = "example-devco"
        resource_name = "a"
        version = "0.1.0"
        dependencies = (tc.URI(path=tc.URI("lib", "example-devco", "b", "0.1.0")),)

    with pytest.raises(TypeError, match="dependencies"):
        A(dependencies=())


@pytest.mark.parametrize(
    "value",
    ["ilc", "ilc-client", "ordinary_client", "v2", "library2-client", "a", "a1b2"],
)
def test_resource_name_grammar_accepts_canonical_values(value: str):
    from tinychain.uri import validate_resource_name

    assert validate_resource_name(value) == value

    Library = type(
        "GrammarLibrary",
        (tc.Library,),
        {"publisher": "example-devco", "resource_name": value, "version": "0.1.0"},
    )
    assert Library.class_id().path == f"/lib/example-devco/{value}/0.1.0"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", ValueError),
        (None, ValueError),
        (123, ValueError),
        ("Bad Name", ValueError),
        ("UPPER", ValueError),
        ("bad/name", ValueError),
        ("-leading", ValueError),
        ("trailing-", ValueError),
        ("_leading", ValueError),
        ("trailing_", ValueError),
        ("double--dash", ValueError),
        ("double__underscore", ValueError),
        (".", ValueError),
        ("..", ValueError),
    ],
)
def test_resource_name_grammar_rejects_invalid_values(value: object, expected):
    from tinychain.uri import validate_resource_name

    with pytest.raises(expected, match="resource_name"):
        validate_resource_name(value)


@pytest.mark.parametrize("field", ("publisher", "resource_name", "version"))
def test_identity_fields_are_read_only_on_instances(field):
    class ReadOnly(tc.Library):
        publisher = "applied-physics"
        resource_name = "read-only"
        version = "0.1.0"

    instance = ReadOnly()

    # Reads resolve to the class-level canonical metadata.
    assert instance.publisher == "applied-physics"
    assert instance.resource_name == "read-only"
    assert instance.version == "0.1.0"

    # Assigning identity metadata on an instance is rejected rather than
    # silently creating a misleading shadow value.
    with pytest.raises(AttributeError, match="read-only on instances"):
        setattr(instance, field, "drifted")

    # A non-identity instance attribute (deployment state) remains assignable.
    instance.authority = None


def test_identity_consumers_are_class_authoritative():
    from tinychain.autodiff.routes import extract_route_identity
    from tinychain.library import _class_schema, _library_schema

    class Canonical(tc.Library):
        publisher = "applied-physics"
        resource_name = "canonical"
        version = "0.1.0"

        @tc.get
        def ping(self) -> tc.String:
            ...

    expected = "/lib/applied-physics/canonical/0.1.0"
    instance = Canonical()

    assert instance.id().path == expected
    assert instance.link().path == expected
    assert Canonical.class_id().path == expected
    assert _class_schema(Canonical)["id"] == expected
    assert _library_schema(instance)["id"] == expected
    assert list(library_definition(instance).keys()) == [expected]

    ir_paths = [route["path"] for route in compile_ir(instance)["routes"]]
    assert "/ping" in ir_paths

    with tc.backend(mode="deferred"):
        op = instance.ping()
        assert op.op.path == expected + "/ping"

    identity = extract_route_identity(instance.ping)
    assert identity.library_name == "canonical"
    assert identity.library_path == expected
    assert identity.library_uri == expected
    assert identity.route_uri == expected + "/path/ping"
