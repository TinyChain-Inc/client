from __future__ import annotations

import json
from pathlib import Path

import pytest
import tinychain as tc


class Point(tc.Class, tc.Map):
    publisher = "example-devco"
    resource_name = "point"
    version = "1.0.0"
    dimensions = 2

    @tc.get
    def label(self) -> tc.String:
        return "point"

    @tc.post
    def translate(self, dx: tc.Number, dy: tc.Number) -> tc.Map:
        return {"dx": dx, "dy": dy}


class NamedPoint(Point):
    publisher = "example-devco"
    resource_name = "named-point"
    version = "1.0.0"
    name = "origin"

    @tc.get
    def label(self) -> tc.String:
        return "named point"


def test_class_definition_encodes_native_and_user_parents_canonically():
    base = tc.class_definition(Point)
    derived = tc.class_definition(NamedPoint)
    assert base["id"] == "/class/example-devco/point/1.0.0"
    assert base["parent"] == str(tc.Map.__uri__)
    assert "extends" not in Point.__dict__
    assert set(base["prototype"]) == {"dimensions", "label", "translate"}
    assert derived["parent"] == base["id"]
    assert set(derived["prototype"]) == {"name", "label"}
    assert tc.validate_class_definition(derived) == derived


def test_class_construction_and_bound_methods_are_one_deferred_plan():
    point = NamedPoint(name="home")
    assert point.to_json() == {NamedPoint.class_id().path: {"name": "home"}}
    with tc.backend(mode="deferred"):
        translated = point.translate(dx=1, dy=2)
    assert isinstance(translated, tc.Map)
    assert translated.op.path == f"{NamedPoint.class_id().path}/translate"
    assert translated.op.body == {"dx": 1, "dy": 2}
    with tc.backend(mode="deferred"):
        assert isinstance(point.label(), tc.String)


def test_class_inherited_data_and_missing_member_are_typed():
    point = NamedPoint()
    assert point.dimensions.to_json() == {
        f"{NamedPoint.class_id().path}/dimensions": [None]
    }
    with pytest.raises(tc.MissingClassMember, match="unknown"):
        point.unknown


def test_class_rejects_invalid_parent_and_method_data_kind_override():
    with pytest.raises(tc.InvalidClassParent):
        class MissingParent(tc.Class):
            publisher = "example-devco"
            resource_name = "missing-parent"
            version = "1.0.0"

    with pytest.raises(tc.UnsupportedClassOverride, match="label"):
        class InvalidOverride(Point):
            publisher = "example-devco"
            resource_name = "invalid-override"
            version = "1.0.0"
            label = "not a method"


def test_library_manifest_owns_class_installation_lifecycle():
    from tinychain.library import compile_ir

    class Geometry(tc.Library):
        publisher = "example-devco"
        resource_name = "geometry"
        version = "1.0.0"
        classes = (Point, NamedPoint)

    manifest = compile_ir(Geometry)
    assert [definition["id"] for definition in manifest["classes"]] == [
        Point.class_id().path,
        NamedPoint.class_id().path,
    ]


def test_class_definitions_match_language_neutral_golden_fixture():
    fixture = Path(__file__).with_name("fixtures") / "class_definitions.json"
    definitions = json.loads(fixture.read_text())
    assert definitions == [tc.class_definition(Point), tc.class_definition(NamedPoint)]
    assert [tc.validate_class_definition(item) for item in definitions] == definitions
