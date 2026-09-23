import tinychain as tc
import pytest


def test_btree_is_collection_not_scalar():
    btree = tc.collection.BTree(tc.state.IdRef("btree"))

    assert isinstance(btree, tc.state.Collection)
    assert not isinstance(btree, tc.state.Scalar)


def test_btree_does_not_encode_a_runtime_get_as_canonical_state():
    runtime_get = tc.opref.get("$btree/count").with_body(None)
    btree = tc.collection.BTree(runtime_get)

    with pytest.raises(TypeError, match="cannot encode form"):
        btree.to_json()


def test_btree_does_not_encode_a_runtime_delete_as_canonical_state():
    runtime_delete = tc.opref.delete("$btree", body=["a"])
    btree = tc.collection.BTree(runtime_delete)

    with pytest.raises(TypeError, match="cannot encode form"):
        btree.to_json()


def test_btree_stages_anonymous_subject_in_active_context():
    with tc.scoped_context() as ctx:
        btree = tc.collection.BTree([
            ("key", "/state/scalar/value/string"),
        ])

        assert btree.to_json() == {
            "/state/collection/btree": [
                [["key", "/state/scalar/value/string"]],
                [],
            ]
        }

        contains = btree.contains(["a"])
        contains_json = contains.to_json()
        assert len(contains_json) == 1
        (subject, key_payload), = contains_json.items()
        assert subject.endswith("/contains")
        assert subject.startswith("$_")
        assert key_payload == [["a"]]
        assert len(list(ctx.form())) == 1


def test_btree_slice_emits_symbolic_post_opref():
    btree = tc.collection.BTree(tc.state.IdRef("btree"))

    sliced = btree.slice("a", "d")
    assert isinstance(sliced, tc.collection.BTree)
    assert sliced.to_json() == {
        "$btree/slice": [{
            "end": "d",
            "reverse": False,
            "start": "a",
        }]
    }

    sliced_form = tc.state.form_of(sliced)
    assert isinstance(sliced_form, tc.state.TCRef)
    assert isinstance(tc.state.form_of(sliced_form), tc.state.GetOpRef)


def test_btree_slice_with_reverse_emits_symbolic_post_opref():
    btree = tc.collection.BTree(tc.state.IdRef("btree"))

    sliced = btree.slice("b", "d", reverse=True)
    assert isinstance(sliced, tc.collection.BTree)
    assert sliced.to_json() == {
        "$btree/slice": [{
            "end": "d",
            "reverse": True,
            "start": "b",
        }]
    }

    sliced_form = tc.state.form_of(sliced)
    assert isinstance(sliced_form, tc.state.TCRef)
    assert isinstance(tc.state.form_of(sliced_form), tc.state.GetOpRef)


def test_btree_collection_payload_round_trip():
    payload = {
        "/state/collection/btree": [
            [["key", "/state/scalar/value/string"]],
            ["c", "a", "b"],
        ]
    }
    btree = tc.collection.BTree.from_json(payload)
    assert btree.to_json() == payload


def test_btree_collection_module_exports_btree():
    btree = tc.collection.BTree(tc.state.IdRef("btree"))

    sliced = btree.slice("a", "d")
    assert isinstance(sliced, tc.collection.BTree)
    assert sliced.to_json() == {
        "$btree/slice": [{
            "end": "d",
            "reverse": False,
            "start": "a",
        }]
    }


def test_btree_contains_emits_symbolic_get_opref():
    btree = tc.collection.BTree(tc.state.IdRef("btree"))

    contains = btree.contains(["a"])
    assert isinstance(contains, tc.state.Scalar)
    assert contains.to_json() == {
        "$btree/contains": [["a"]]
    }

    contains_form = tc.state.form_of(contains)
    assert isinstance(contains_form, tc.state.TCRef)
    assert isinstance(tc.state.form_of(contains_form), tc.state.GetOpRef)


def test_btree_count_and_is_empty_emit_symbolic_get_oprefs():
    btree = tc.collection.BTree(tc.state.IdRef("btree"))

    count = btree.count()
    assert isinstance(count, tc.state.Scalar)
    assert count.to_json() == {
        "$btree/count": [None]
    }

    count_form = tc.state.form_of(count)
    assert isinstance(count_form, tc.state.TCRef)
    assert isinstance(tc.state.form_of(count_form), tc.state.GetOpRef)

    empty = btree.is_empty()
    assert isinstance(empty, tc.state.Scalar)
    assert empty.to_json() == {
        "$btree/is_empty": [None]
    }

    empty_form = tc.state.form_of(empty)
    assert isinstance(empty_form, tc.state.TCRef)
    assert isinstance(tc.state.form_of(empty_form), tc.state.GetOpRef)


def test_btree_insert_and_delete_emit_symbolic_write_oprefs():
    btree = tc.collection.BTree(tc.state.IdRef("btree"))

    insert = btree.insert(["a"])
    assert isinstance(insert, tc.state.Scalar)
    assert insert.to_json() == {
        "$btree/insert": [None, ["a"]]
    }

    insert_form = tc.state.form_of(insert)
    assert isinstance(insert_form, tc.state.TCRef)
    assert isinstance(tc.state.form_of(insert_form), tc.state.PutOpRef)

    delete = btree.delete(["a"])
    assert isinstance(delete, tc.state.Scalar)
    assert delete.to_json() == {
        "/state/scalar/ref/op/delete": ["$btree", ["a"]]
    }

    delete_form = tc.state.form_of(delete)
    assert isinstance(delete_form, tc.state.TCRef)
    assert isinstance(tc.state.form_of(delete_form), tc.state.DeleteOpRef)


def test_btree_legacy_payload_is_rejected_without_explicit_schema():
    with pytest.raises(TypeError, match=r"BTree payload must be \[schema, rows\]"):
        tc.collection.BTree.from_json({"/state/collection/btree": ["c", "a", "b"]})
