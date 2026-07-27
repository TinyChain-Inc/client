import tinychain as tc
import pytest


def test_value_roundtrip_typed_maps():
    v = tc.Number(7)
    assert tc.state.Value.from_json(v.to_json()).to_json() == v.to_json()

    s = tc.String("x")
    assert tc.state.Value.from_json(s.to_json()).to_json() == s.to_json()

    n = tc.state.Null()
    assert tc.state.Value.from_json(n.to_json()).to_json() == n.to_json()
    assert isinstance(n, tc.state.Null)
    assert isinstance(tc.state.Value.from_json(None), tc.state.Null)


def test_value_bool_decodes_as_number():
    b = tc.Number(True)
    assert tc.state.Value.from_json(b.to_json()).to_json() == b.to_json()
    assert isinstance(tc.state.Value.from_json(True), tc.Number)
    assert isinstance(tc.state.Value.from_json(1), tc.Number)


def test_value_map_and_tuple_roundtrip():
    value = tc.Map(
        {
            "dtype": "f32",
            "encoding": {"signed": True, "bits": 16},
            "shape": ["N", "D"],
        }
    )

    decoded = tc.state.Value.from_json(value.to_json())
    assert decoded.to_json() == value.to_json()


def test_value_subtype_constructors_and_from_json_types():
    assert isinstance(tc.Number(3), tc.Number)
    assert isinstance(tc.Number(False), tc.Number)
    assert isinstance(tc.Map({"x": 1}), tc.Map)
    assert isinstance(tc.Tuple([1, 2]), tc.Tuple)

    assert isinstance(tc.state.Value.from_json(3), tc.Number)
    assert isinstance(tc.state.Value.from_json(True), tc.Number)
    assert isinstance(tc.state.Value.from_json({"x": 1}), tc.Map)
    assert isinstance(tc.state.Value.from_json([1, 2]), tc.Tuple)


def test_map_and_tuple_literal_iteration_helpers():
    m = tc.Map({"a": 1, "b": 2})
    assert list(m) == ["a", "b"]
    assert [k for k, _ in m.items()] == ["a", "b"]

    t = tc.Tuple([1, "x", True])
    assert len(t) == 3
    assert [type(v).__name__ for v in t] == ["Number", "String", "Number"]


def test_scalar_roundtrip_nested_map_and_tuple():
    scalar = tc.state.Scalar.from_json({
        "dtype": "f32",
        "encoding": {"fixed_point": {"signed": True, "bits": 16, "scale_pow2": -8}},
        "shape": ["N", "D"],
    })
    decoded = tc.state.Scalar.from_json(scalar.to_json())
    assert decoded.to_json() == scalar.to_json()


def test_scalar_opref_encoding_get_put_post_delete():
    subject = tc.URI.of("lib", "acme", "foo", "1.0.0")
    get = tc.state.Get(subject)(tc.state.Null()).to_json()
    assert get == {subject: [None]}

    put = tc.state.Put(subject)(tc.String("k"), tc.Number(3)).to_json()
    assert put == {
        subject: [
            "k",
            3,
        ]
    }

    matmul = tc.URI.of("class", "tinychain", "numeric", "0.1.0", "matmul")
    post = tc.state.Post(matmul)({"transpose_a": tc.Number(False)}).to_json()
    assert post == {
        matmul: {"transpose_a": False}
    }

    delete = tc.state.Delete(subject)(tc.String("k")).to_json()
    assert delete == {
        tc.URI.of("state", "scalar", "ref", "op", "delete"): [
            subject,
            "k",
        ]
    }


def test_scalar_method_handles_have_distinct_hashes_and_typed_args():
    subject = tc.URI.of("lib", "acme", "foo", "1.0.0")

    assert hash(tc.state.Get(subject)) != hash(tc.state.Put(subject))
    assert hash(tc.state.Post(subject)) != hash(tc.state.Delete(subject))

    with pytest.raises(TypeError, match="Get expects key to be a Value"):
        tc.state.Get(subject)(tc.state.id("k"))

    with pytest.raises(TypeError, match="Delete expects key to be a Value"):
        tc.state.Delete(subject)(tc.state.id("k"))

    with pytest.raises(TypeError, match="Put expects key to be State"):
        tc.state.Put(subject)("k", tc.Number(1))

    with pytest.raises(TypeError, match="Post expects params to be a map of State values"):
        tc.state.Post(subject)({"x": 1})


def test_scalar_walk_and_opdef_roundtrip():
    scalar = tc.state.Scalar.from_json({"a": 1, "b": {"c": 2}})
    refs = [
        tc.state.form_of(node)
        for node in scalar_op_walk(scalar)
        if isinstance(tc.state.form_of(node), tc.state.TCRef)
    ]
    assert refs == []

    op = tc.state.PostOpDef([("x", 3)])
    encoded = op.to_json()
    decoded = tc.state.OpDef.from_json(encoded)
    assert decoded == op

    scalar_op = tc.state.Scalar.from_json(encoded)
    scalar_op_form = tc.state.form_of(scalar_op)
    assert isinstance(scalar_op_form, tc.state.OpDef)
    walked = [s for s in scalar_op_form.walk_scalars()]
    assert len(walked) == 1


def scalar_op_walk(root: tc.state.Scalar):
    stack = [root]
    while stack:
        node = stack.pop()
        node_form = tc.state.form_of(node)
        if isinstance(node_form, dict):
            for value in reversed(list(node_form.values())):
                stack.append(value)
        elif isinstance(node_form, (list, tuple)):
            for value in reversed(list(node_form)):
                stack.append(value)
        yield node


def test_scalar_tcref_id_roundtrip():
    scalar = tc.state.Scalar(ref=tc.state.TCRef.from_json({"$foo": []}))
    encoded = scalar.to_json()
    assert encoded == {"$foo": []}

    decoded = tc.state.Scalar.from_json(encoded)
    assert tc.state.form_of(decoded) == tc.state.form_of(scalar)


def test_number_literal_arithmetic_methods():
    assert tc.state.form_of(tc.Number(8) + 2) == 10
    assert tc.state.form_of(tc.Number(8) - 3) == 5
    assert tc.state.form_of(tc.Number(8) * 2) == 16
    assert tc.state.form_of(tc.Number(8) / 4) == 2

    assert tc.state.form_of(2 + tc.Number(8)) == 10
    assert tc.state.form_of(20 - tc.Number(8)) == 12
    assert tc.state.form_of(3 * tc.Number(8)) == 24
    assert tc.state.form_of(64 / tc.Number(8)) == 8


def test_number_dtype_subclasses_construct_and_validate():
    assert isinstance(tc.I64(3), tc.Integer)
    assert isinstance(tc.U64(3), tc.Integer)
    assert isinstance(tc.F32(3.5), tc.Float)
    assert isinstance(tc.F64(2), tc.Float)

    with pytest.raises(ValueError, match="u64 cannot be negative"):
        tc.U64(-1)

    with pytest.raises(TypeError, match="expected integer value"):
        tc.I64(3.5)


def test_complex_methods_are_subclass_only():
    n = tc.Number(4)
    assert not hasattr(n, "conjugate")
    assert not hasattr(n, "exp")
    assert not hasattr(n, "log")

    z = tc.Complex(2 + 3j)
    assert tc.state.form_of(z.conjugate()) == (2 - 3j)

    expz = tc.state.form_of(z.exp())
    logz = tc.state.form_of(z.log())
    assert isinstance(expz, complex)
    assert isinstance(logz, complex)


def test_value_type_uri_hierarchy_is_parent_appended():
    assert tc.state.Value.__uri__.path == tc.URI.of("state", "scalar", "value")
    assert tc.state.Number.__uri__.path == tc.URI.of(tc.state.Value, "number")
    assert tc.state.Float.__uri__.path == tc.URI.of(tc.state.Number, "float")
    assert tc.state.F32.__uri__.path == tc.URI.of(tc.state.Float, "32")
    assert tc.state.Complex.__uri__.path == tc.URI.of(tc.state.Number, "complex")
    assert tc.state.C128.__uri__.path == tc.URI.of(tc.state.Complex, "128")


def test_value_from_json_delegates_by_uri_to_concrete_subclass():
    f32 = tc.state.Value.from_json({tc.state.F32.__uri__.path: 3.5})
    assert isinstance(f32, tc.state.F32)
    assert tc.state.form_of(f32) == 3.5

    c128 = tc.state.Value.from_json({tc.state.C128.__uri__.path: 2 + 1j})
    assert isinstance(c128, tc.state.C128)
    assert tc.state.form_of(c128) == (2 + 1j)

    u64 = tc.state.Value.from_json({tc.state.U64.__uri__.path: 7})
    assert isinstance(u64, tc.state.U64)
    assert tc.state.form_of(u64) == 7


def test_number_deferred_arithmetic_builds_oprefs():
    with tc.state.scoped_context() as cxt:
        x = tc.state.id("x")
        deferred_form = tc.state.form_of(x.add(1))
        assert isinstance(deferred_form, tc.state.TCRef)
        deferred_ref_form = tc.state.form_of(deferred_form)
        assert isinstance(deferred_ref_form, tc.state.OpRef)
        deferred = tc.Number(deferred_ref_form)
        deferred._ctx = cxt

        add = deferred + 2
        sub = deferred - 2
        mul = deferred * 2
        div = deferred / 2

        cxt.bind("add", add)
        cxt.bind("sub", sub)
        cxt.bind("mul", mul)
        cxt.bind("div", div)

        assert isinstance(add.op, tc.state.OpRef)
        assert isinstance(sub.op, tc.state.OpRef)
        assert isinstance(mul.op, tc.state.OpRef)
        assert isinstance(div.op, tc.state.OpRef)

        assert add.op.subject.endswith("/add")
        assert sub.op.subject.endswith("/sub")
        assert mul.op.subject.endswith("/mul")
        assert div.op.subject.endswith("/div")
        assert add.op.args == {"r": 2}
        assert sub.op.args == {"r": 2}
        assert mul.op.args == {"r": 2}
        assert div.op.args == {"r": 2}


def test_reduce_infers_item_binding_name_from_reducer_inputs():
    op = tc.state.PostOpDef([
        ("x2", tc.state.id("x") + tc.state.id("x")),
        ("result", tc.state.id("x2") + tc.state.id("x2")),
    ])

    with tc.state.scoped_context() as cxt:
        items = tc.state.autobox([1])
        cxt.bind(items, "items")
        reduced = items.reduce(op=op, value={})
    payload = reduced.to_json()
    (subject, params), = payload.items()

    assert subject.endswith("/reduce")
    assert params["item_name"] == "x"


def test_reduce_rejects_ambiguous_item_binding():
    op = tc.state.PostOpDef([
        ("xa", tc.state.id("x").add(1)),
        ("ya", tc.state.id("y").add(1)),
        ("result", tc.state.id("xa") + tc.state.id("ya")),
    ])

    with tc.state.scoped_context() as cxt:
        items = tc.state.autobox([1])
        cxt.bind(items, "items")

        with pytest.raises(TypeError, match="ambiguous"):
            items.reduce(op=op, value={})


def test_autobox_preserves_non_scalar_state_instances():
    collection = tc.state.Collection({"k": tc.Number(1)})
    boxed = tc.state.autobox(collection)

    assert isinstance(boxed, tc.state.Collection)
    assert boxed is collection


def test_autobox_numpy_array_to_tensor_state():
    import numpy as np

    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    boxed = tc.state.autobox(matrix)

    assert isinstance(boxed, tc.Tensor)
    assert boxed.to_json() == {
        tc.URI.of(tc.Tensor): [[matrix.dtype, matrix.shape], [1.0, 2.0, 3.0, 4.0]]
    }
