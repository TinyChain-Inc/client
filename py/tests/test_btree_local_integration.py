from __future__ import annotations

import pathlib

import tinychain as tc
from .support import install_token, require_tinychain_local


# BTree is a native state type. This Library exists only as a test harness so we
# can validate how BTree values flow through route compilation and the local
# kernel execution surface used by Python clients.
class BTreeRouteHarness(tc.Library):
    publisher = "example-devco"
    resource_name = "b_tree_route_harness"
    version = "0.1.0"

    @tc.post
    def echo(self, tree: tc.collection.BTree) -> tc.collection.BTree:
        return tree

    @tc.post
    def slice_tree(self, tree: tc.collection.BTree) -> tc.collection.BTree:
        return tree.slice("a", "d")

    @tc.post
    def contains_a(self, tree: tc.collection.BTree) -> tc.Bool:
        return tree.contains(["a"])

    @tc.post
    def count_rows(self, tree: tc.collection.BTree) -> tc.Number:
        return tree.count()

    @tc.post
    def tree_is_empty(self, tree: tc.collection.BTree) -> tc.Bool:
        return tree.is_empty()

    @tc.post
    def insert_then_contains_z(self, tree: tc.collection.BTree) -> tc.Bool:
        return tc.after(tree.insert(["z"]), tree.contains(["z"]))

    @tc.post
    def delete_then_contains_a(self, tree: tc.collection.BTree) -> tc.Bool:
        return tc.after(tree.delete(["a"]), tree.contains(["a"]))


def _sample_btree_payload() -> dict[str, object]:
    return {
        "/state/collection/btree": [
            [["key", "/state/scalar/value/string"]],
            ["a", "b", "c", "d"],
        ]
    }


def _sample_btree_state() -> tc.collection.BTree:
    return tc.collection.BTree.from_json(_sample_btree_payload())


def test_btree_library_installs_via_local_python_client(tmp_path: pathlib.Path):
    require_tinychain_local()

    library = BTreeRouteHarness()
    token = install_token(BTreeRouteHarness.class_id().path)
    kernel = tc.kernel.with_library(
        library, data_dir=tmp_path, workspace=tmp_path / "workspace", token=token
    )
    try:
        install = tc.install(BTreeRouteHarness, kernel=kernel, data_dir=tmp_path, token=token)
        assert install.status == 204

        route = library.slice_tree
        assert callable(route)

        with tc.backend(kernel, mode="deferred"):
            symbolic_tree = tc.collection.BTree(tc.state.IdRef("tree"))
            result = route(tree=symbolic_tree)
            assert isinstance(result, tc.collection.BTree)
            op = tc.state.form_of(result)
            assert op.path == "/lib/example-devco/b_tree_route_harness/0.1.0/slice_tree"
            assert tc.state.form_of(op.body["tree"]) == tc.state.IdRef("tree")
    finally:
        kernel.shutdown()


def test_btree_symbolic_methods_execute_via_local_python_client(tmp_path: pathlib.Path):
    require_tinychain_local()

    library = BTreeRouteHarness()
    token = install_token(BTreeRouteHarness.class_id().path)
    kernel = tc.kernel.with_library(
        library, data_dir=tmp_path, workspace=tmp_path / "workspace", token=token
    )
    try:
        install = tc.install(BTreeRouteHarness, kernel=kernel, data_dir=tmp_path, token=token)
        assert install.status == 204

        with tc.backend(kernel):
            sliced = library.slice_tree(tree=_sample_btree_state())
            assert sliced.to_json() == {
                "/state/collection/btree": [
                    [["key", "/state/scalar/value/string"]],
                    ["a", "b", "c"],
                ]
            }
            assert library.contains_a(tree=_sample_btree_state()) is True
            assert library.count_rows(tree=_sample_btree_state()) == 4
            assert library.tree_is_empty(tree=_sample_btree_state()) is False
            assert library.insert_then_contains_z(tree=_sample_btree_state()) is True
            assert library.delete_then_contains_a(tree=_sample_btree_state()) is False
    finally:
        kernel.shutdown()
