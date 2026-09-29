import json
from pathlib import Path

import pytest
import tinychain as tc
from tinychain.library import library_definition


class Example(tc.Service):
    publisher = "example-devco"
    resource_name = "btree"
    version = "1.0.0"
    label = "native"
    data = tc.chain.Sync(tc.collection.BTree([["key", "/state/scalar/value/number"]]))

    @tc.get
    def count(self, cxt, key):
        return self.data.count()

    @tc.put
    def insert(self, cxt, key, value):
        return self.data._put(value, "insert", key=key)

    @tc.post
    def append(self, cxt, key, value):
        return self.data._put(value, "insert", key=key)


def test_service_definition_matches_server_fixture():
    fixture = Path(__file__).with_name("fixtures") / "service.json"
    assert library_definition(Example) == json.loads(fixture.read_text())
    instance = Example()
    instance.label = "not the definition"
    assert library_definition(instance) == library_definition(Example)
    assert tc.state.form_of(instance.data.count()).subject == str(instance.id().child("data", "count"))


def test_service_inheritance_and_delete():
    class Child(Example):
        label = "child"

        @tc.delete
        def remove(self, cxt, key):
            return self.data.delete(key)

    body = library_definition(Child)[str(Child.class_id())]
    assert body["label"] == "child"
    assert {"count", "insert", "append", "remove", "data"} <= body.keys()
    assert "$self/data" in str(body["remove"])
    with tc.backend(mode="deferred"):
        call = Child().count(key=None)
    assert tc.state.form_of(call).path.endswith("/count")


@pytest.mark.parametrize("subject", [tc.Tensor.dense("u64", [1], [1]), tc.collection.BTree(tc.state.IdRef("tree")), tc.collection.BTree([["key", "/state/scalar/value/number"]], [[1]])])
def test_sync_rejects_unsupported_or_populated_subjects(subject):
    with pytest.raises((TypeError, ValueError)):
        tc.chain.Sync(subject)


def test_service_rejects_bare_collection_and_constructor():
    class Bare(Example):
        data = tc.collection.BTree([["key", "/state/scalar/value/number"]])

    class Constructed(Example):
        def __init__(self):
            super().__init__()

    for cls in (Bare, Constructed):
        with pytest.raises(TypeError):
            library_definition(cls)


def test_service_dependencies_and_scalar_self_binding():
    class Dependency(tc.Library):
        publisher = "example-devco"
        resource_name = "dependency"
        version = "1.0.0"

        @tc.get
        def echo(self, cxt, key):
            return key

    class Dependent(Example):
        dependency = Dependency()

        @tc.get
        def echo(self, cxt, key):
            return self.dependency.echo(key=key)

        @tc.get
        def name_value(self, cxt, key):
            return self.label

        @tc.get
        def nested_call(self, key):
            return self.count(key=key)

    definition = library_definition(Dependent)[str(Dependent.class_id())]
    assert str(Dependency.class_id()) in str(definition["echo"])
    assert "$self/label" in str(definition["name_value"])
    assert "$self/count" in str(definition["nested_call"])
    assert Dependency.class_id() in Dependent().dependencies
    class Inherited(Dependent):
        pass

    assert Dependency.class_id() in Inherited().dependencies


def test_service_rejects_wasm_before_io():
    with pytest.raises(TypeError, match="only for Library"):
        tc.install(Example, wasm=Path("absent.wasm"))


def test_compilation_discovers_once_and_binds_only_symbolic_members(monkeypatch):
    instance = Example()
    instance.data = "instance mutation"
    original_members = Example._members.__func__
    original_bind = tc.chain.Sync._bind
    discoveries = []
    paths = []

    def members(cls):
        discoveries.append(cls)
        return original_members(cls)

    def bind(sync, path):
        paths.append(path)
        return original_bind(sync, path)

    monkeypatch.setattr(Example, "_members", classmethod(members))
    monkeypatch.setattr(tc.chain.Sync, "_bind", bind)
    expected = json.loads((Path(__file__).with_name("fixtures") / "service.json").read_text())
    assert library_definition(instance) == expected
    assert discoveries == [Example]
    assert paths == ["$self/data"]

    discoveries.clear()
    paths.clear()
    assert Example.count.opdef(instance).to_json() == expected[str(Example.class_id())]["count"]
    assert discoveries == [Example]
    assert paths == ["$self/data"]

    class Invalid(Example):
        publisher = ""

    discoveries.clear()
    for action in (Invalid, lambda: library_definition(Invalid), lambda: Invalid.count.opdef(Invalid)):
        with pytest.raises(TypeError, match="requires class"):
            action()
    assert discoveries == []


def test_library_literal_results_and_construction_validation_are_unchanged(monkeypatch):
    marker = object()
    subjects = []

    class Plain(tc.Library):
        publisher = "example-devco"
        resource_name = "plain"
        version = "1.0.0"
        literal_value = marker

        @tc.get
        def literal(self):
            subjects.append(self)
            return self.literal_value

        @tc.get
        def echo(self, cxt, key):
            return key

    class Child(Plain):
        pass

    discoveries = []
    original_members = Plain._members.__func__

    def members(cls):
        discoveries.append(cls)
        return original_members(cls)

    monkeypatch.setattr(Plain, "_members", classmethod(members))
    for _ in range(2):
        assert library_definition(Plain)[str(Plain.class_id())]["literal"] is marker
    assert discoveries == [Plain, Plain]
    assert subjects[0] is not subjects[1]
    discoveries.clear()
    Plain.echo.opdef(Plain())
    assert discoveries == []
    instance = Plain()
    instance.literal_value = object()
    assert library_definition(instance)[str(Plain.class_id())]["literal"] is instance.literal_value
    assert library_definition(Child)[str(Child.class_id())] == {}

    class Invalid(Plain):
        classes = (object,)

    Invalid()  # Class declarations are validated when compiling, not constructing.
    with pytest.raises(TypeError, match="Class subclasses"):
        library_definition(Invalid)
