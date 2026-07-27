from __future__ import annotations

import pathlib

import tinychain as tc
import pytest
import tinychain.state.collection.btree as btree_module

from .support import install_token, require_tinychain_local


# BTree is a native state type. This Library exists only as a test harness so we
# can validate how BTree values flow through route compilation and the local
# kernel execution surface used by Python clients.
class BTreeRouteHarness(tc.Library):
    publisher = "example-devco"
    version = "0.1.0"

    @tc.post
    def echo(self, tree: tc.state.collection.BTree) -> tc.state.collection.BTree:
        return tree

    @tc.post
    def slice_tree(self, tree: tc.state.collection.BTree) -> tc.state.collection.BTree:
        return tree.slice("a", "d")

    @tc.post
    def contains_a(self, tree: tc.state.collection.BTree) -> tc.Bool:
        return tree.contains(["a"])

    @tc.post
    def count_rows(self, tree: tc.state.collection.BTree) -> tc.Number:
        return tree.count()

    @tc.post
    def tree_is_empty(self, tree: tc.state.collection.BTree) -> tc.Bool:
        return tree.is_empty()

    @tc.post
    def insert_then_contains_z(self, tree: tc.state.collection.BTree) -> tc.Bool:
        return tc.after(tree.insert(["z"]), tree.contains(["z"]))

    @tc.post
    def delete_then_contains_a(self, tree: tc.state.collection.BTree) -> tc.Bool:
        return tc.after(tree.delete(["a"]), tree.contains(["a"]))


def _sample_btree_payload() -> dict[str, object]:
    return {
        "/state/collection/btree": [
            [["key", "/state/scalar/value/string"]],
            ["a", "b", "c", "d"],
        ]
    }


def _sample_btree_state() -> tc.state.collection.BTree:
    return tc.state.collection.BTree.from_json(_sample_btree_payload())


def test_btree_library_installs_via_local_python_client(tmp_path: pathlib.Path):
    require_tinychain_local()

    library = BTreeRouteHarness()
    token = install_token(BTreeRouteHarness.class_id().path)
    kernel = tc.kernel.with_library(library, data_dir=tmp_path, token=token)
    install = tc.install(BTreeRouteHarness, kernel=kernel, data_dir=tmp_path, token=token)
    assert install.status == 204

    route = library.slice_tree
    assert callable(route)

    with tc.backend(kernel, mode="deferred"):
        symbolic_tree = tc.state.collection.BTree(ref=tc.state.TCRef(tc.state.IdRef("tree")))
        result = route(tree=symbolic_tree)
        assert isinstance(result, tc.state.collection.BTree)
        assert result.to_json() == {
            "/lib/example-devco/b_tree_route_harness/0.1.0/slice_tree": {
                "tree": {
                    "$tree": []
                }
            }
        }


def test_btree_symbolic_methods_execute_via_local_python_client(tmp_path: pathlib.Path):
    require_tinychain_local()

    library = BTreeRouteHarness()
    token = install_token(BTreeRouteHarness.class_id().path)
    kernel = tc.kernel.with_library(library, data_dir=tmp_path, token=token)
    install = tc.install(BTreeRouteHarness, kernel=kernel, data_dir=tmp_path, token=token)
    assert install.status == 204

    with tc.backend(kernel):
        tree = _sample_btree_state()

        assert library.contains_a(tree) is True
        assert library.count_rows(tree) == 4
        assert library.tree_is_empty(tree) is False
        assert library.insert_then_contains_z(tree) is False
        assert library.delete_then_contains_a(tree) is True


def test_btree_imperative_create_update_read_delete_via_public_interface():
    require_tinychain_local()

    btree = tc.state.collection.BTree([
        ("key", "/state/scalar/value/string"),
    ])

    btree.insert(["b"])
    btree.insert(["a"])
    btree.insert(["c"])

    assert btree.contains(["a"])
    assert not btree.contains(["z"])
    assert btree.count() == 3
    assert btree.slice("a", "b").count() == 1

    btree.delete(["b"])

    assert not btree.contains(["b"])
    assert btree.count() == 2
    assert btree.contains(["a"])
    assert btree.contains(["c"])

    multi = tc.state.collection.BTree([
        ("k1", "/state/scalar/value/string"),
        ("k2", "/state/scalar/value/string"),
    ])

    multi.insert(["b", "2"])
    multi.insert(["a", "9"])
    multi.insert(["a", "1"])

    assert multi.contains(["a", "1"])
    assert not multi.contains(["a", "2"])
    assert multi.count() == 3
    assert multi.slice(["a", "1"], ["a", "2"]).count() == 1
    assert multi.slice(["a", "9"], ["b", "0"]).count() == 1
    assert multi.slice(["b", "2"], ["b", "3"]).count() == 1


def test_state_collection_btree_constructor_is_native_in_imperative_mode():
    require_tinychain_local()

    btree = tc.state.collection.BTree([
        ("key", "/state/scalar/value/string"),
    ])

    assert btree.insert(["b"]) is None
    assert btree.insert(["a"]) is None
    assert btree.contains(["a"]) is True
    assert btree.count() == 2
    assert btree.is_empty() is False

    sliced = btree.slice("a", "b")
    assert isinstance(sliced, tc.state.collection.BTree)
    assert sliced.count() == 1


def test_state_collection_btree_constructor_is_symbolic_in_deferred_mode():
    require_tinychain_local()

    with tc.backend(mode="deferred"):
        btree = tc.state.collection.BTree([
            ("key", "/state/scalar/value/string"),
        ])
        assert btree.to_json() == {
            "/state/collection/btree": [
                [["key", "/state/scalar/value/string"]],
                [],
            ]
        }

        with tc.state.scoped_context() as cxt:
            contains = btree.contains(["a"])
            contains_json = contains.to_json()
            assert len(contains_json) == 1
            (subject, key_payload), = contains_json.items()
            assert subject.endswith("/contains")
            assert subject.startswith("$_")
            assert key_payload == [["a"]]
            assert len(list(cxt.form())) == 1


def test_state_collection_btree_constructor_fails_fast_without_local_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        btree_module._local,
        "backend",
        lambda: (_ for _ in ()).throw(ImportError("tinychain-local missing")),
    )

    with pytest.raises(ImportError, match="requires `tinychain-local`"):
        tc.state.collection.BTree([
            ("key", "/state/scalar/value/string"),
        ])


def test_state_collection_btree_constructor_stays_symbolic_in_deferred_mode_without_local_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        btree_module._local,
        "backend",
        lambda: (_ for _ in ()).throw(ImportError("tinychain-local missing")),
    )

    with tc.backend(mode="deferred"):
        btree = tc.state.collection.BTree([
            ("key", "/state/scalar/value/string"),
        ])

    assert btree.to_json() == {
        "/state/collection/btree": [
            [["key", "/state/scalar/value/string"]],
            [],
        ]
    }


def test_local_btree_duplicate_insert_and_delete_missing_are_idempotent():
    require_tinychain_local()

    btree = tc.state.collection.BTree([
        ("key", "/state/scalar/value/string"),
    ])

    btree.insert(["a"])
    btree.insert(["a"])
    assert btree.count() == 1
    assert btree.contains(["a"])

    btree.delete(["missing"])
    assert btree.count() == 1
    assert btree.contains(["a"])


def test_btree_rejects_invalid_schema_column_shape():
    require_tinychain_local()

    with pytest.raises(TypeError, match="must have 2 or 3 elements"):
        tc.state.collection.BTree([
            ("key",),
        ])


def test_local_btree_rejects_insert_row_arity_mismatch():
    require_tinychain_local()

    btree = tc.state.collection.BTree([
        ("k1", "/state/scalar/value/string"),
        ("k2", "/state/scalar/value/string"),
    ])

    with pytest.raises(ValueError, match="row arity 1 does not match schema arity 2"):
        btree.insert(["a"])
