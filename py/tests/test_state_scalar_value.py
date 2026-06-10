import tinychain as tc
import pytest


def test_value_roundtrip_typed_maps():
    v = tc.state.Value.number(7)
    assert tc.state.Value.from_json(v.to_json()) == v

    s = tc.state.Value.string("x")
    assert tc.state.Value.from_json(s.to_json()) == s

    n = tc.state.Value.none()
    assert tc.state.Value.from_json(n.to_json()) == n


def test_value_bool_is_distinct_from_number():
    b = tc.state.Value.bool(True)
    assert tc.state.Value.from_json(b.to_json()) == b
    assert tc.state.Value.from_json(True).kind == "bool"
    assert tc.state.Value.from_json(1).kind == "number"

    with pytest.raises(TypeError, match="bool is not a number"):
        tc.state.Value.number(True)


def test_value_map_and_tuple_roundtrip():
    value = tc.state.Value.map_of(
        {
            "dtype": "f32",
            "encoding": {"signed": True, "bits": 16},
            "shape": ["N", "D"],
        }
    )

    decoded = tc.state.Value.from_json(value.to_json())
    assert decoded == value


def test_value_subtype_constructors_and_from_json_types():
    assert isinstance(tc.Number(3), tc.Number)
    assert isinstance(tc.Bool(False), tc.Bool)
    assert isinstance(tc.Map({"x": 1}), tc.Map)
    assert isinstance(tc.Tuple([1, 2]), tc.Tuple)

    assert isinstance(tc.state.Value.from_json(3), tc.Number)
    assert isinstance(tc.state.Value.from_json(True), tc.Bool)
    assert isinstance(tc.state.Value.from_json({"x": 1}), tc.Map)
    assert isinstance(tc.state.Value.from_json([1, 2]), tc.Tuple)


def test_map_and_tuple_literal_iteration_helpers():
    m = tc.Map({"a": 1, "b": 2})
    assert list(m) == ["a", "b"]
    assert [k for k, _ in m.items()] == ["a", "b"]

    t = tc.Tuple([1, "x", True])
    assert len(t) == 3
    assert [v.kind for v in t] == ["number", "string", "bool"]


def test_scalar_roundtrip_nested_map_and_tuple():
    scalar = tc.state.Scalar.from_json({
        "dtype": "f32",
        "encoding": {"fixed_point": {"signed": True, "bits": 16, "scale_pow2": -8}},
        "shape": ["N", "D"],
    })
    decoded = tc.state.Scalar.from_json(scalar.to_json())
    assert decoded == scalar


def test_scalar_opref_encoding_get_put_post_delete():
    subject = tc.uri("lib", "acme", "foo", "1.0.0").path
    get = tc.state.Get(subject)(None).to_json()
    assert get == {subject: [None]}

    put = tc.state.Put(subject)("k", 3).to_json()
    assert put == {
        subject: [
            "k",
            3,
        ]
    }

    matmul = tc.uri("class", "tinychain", "numeric", "0.1.0", "matmul").path
    post = tc.state.Post(matmul)(transpose_a=False).to_json()
    assert post == {
        matmul: {"transpose_a": False}
    }

    delete = tc.state.Delete(subject)("k").to_json()
    assert delete == {
        tc.uri("state", "scalar", "ref", "op", "delete").path: [
            subject,
            "k",
        ]
    }


def test_scalar_walk_and_opdef_roundtrip():
    scalar = tc.state.Scalar.from_json({"a": 1, "b": {"c": 2}})
    refs = list(scalar.walk_tcref())
    assert refs == []

    op = tc.state.OpDef.post([("x", 3)])
    encoded = op.to_json()
    decoded = tc.state.OpDef.from_json(encoded)
    assert decoded == op

    scalar_op = tc.state.Scalar.from_json(encoded)
    walked = [s for s in scalar_op.op.walk_scalars()]
    assert len(walked) == 1


def test_scalar_tcref_id_roundtrip():
    scalar = tc.state.Scalar(ref=tc.state.TCRef.for_id("foo"))
    encoded = scalar.to_json()
    assert encoded == {"$foo": []}

    decoded = tc.state.Scalar.from_json(encoded)
    assert decoded == scalar
