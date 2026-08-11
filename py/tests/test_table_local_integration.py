from __future__ import annotations

import pathlib

import tinychain as tc

from .support import install_token, require_tinychain_local


class TableRouteHarness(tc.Library):
    publisher = "example-devco"
    resource_name = "table_route_harness"
    version = "0.1.0"

    @tc.post
    def echo(self, table: tc.collection.Table) -> tc.collection.Table:
        return table

    @tc.post
    def row(self, table: tc.collection.Table) -> tc.state.scalar.Tuple:
        return table[[1]]

    @tc.post
    def range(self, table: tc.collection.Table) -> tc.collection.Table:
        return table.where(id=slice(1, 3))

    @tc.post
    def select(self, table: tc.collection.Table) -> tc.collection.Table:
        return table.select(["id", "name"])

    @tc.post
    def order(self, table: tc.collection.Table) -> tc.collection.Table:
        return table.order_by(["name"], reverse=True)

    @tc.post
    def limit(self, table: tc.collection.Table) -> tc.collection.Table:
        return table.limit(1)

    @tc.post
    def insert_then_contains(self, table: tc.collection.Table) -> tc.Bool:
        return tc.after(table.insert([4], ["four"]), table.contains([4]))

    @tc.post
    def upsert_then_read(self, table: tc.collection.Table) -> tc.state.scalar.Tuple:
        return tc.after(table.upsert([1], ["updated"]), table[[1]])

    @tc.post
    def update_then_read(self, table: tc.collection.Table) -> tc.state.scalar.Tuple:
        return tc.after(table.update(name="updated"), table[[2]])

    @tc.post
    def update_range_then_read(self, table: tc.collection.Table) -> tc.state.scalar.Tuple:
        selected = table.where(id=slice(1, 2))
        return tc.after(selected.update(name="selected"), table[[1]])

    @tc.post
    def delete_then_contains(self, table: tc.collection.Table) -> tc.Bool:
        return tc.after(table.delete([1]), table.contains([1]))

    @tc.post
    def truncate_then_empty(self, table: tc.collection.Table) -> tc.Bool:
        return tc.after(table.truncate(), table.is_empty())


def sample_table() -> tc.collection.Table:
    schema = tc.collection.table.Schema(
        [tc.collection.table.Column("id", tc.U64)],
        [tc.collection.table.Column("name", tc.String)],
    ).create_index("by_name", ["name"])
    return tc.collection.Table(schema, [[1, "one"], [2, "two"], [3, "three"]])


def test_table_routes_compile_in_deferred_mode():
    library = TableRouteHarness()
    with tc.backend(mode="deferred"):
        table = tc.collection.Table(tc.state.IdRef("table"))
        result = library.update_range_then_read(table=table)
        assert isinstance(result, tc.state.scalar.Tuple)
        assert result.to_json() == {
            "/lib/example-devco/table_route_harness/0.1.0/update_range_then_read": {
                "table": {"$table": []}
            }
        }


def test_table_routes_execute_natively(tmp_path: pathlib.Path):
    require_tinychain_local()

    library = TableRouteHarness()
    token = install_token(TableRouteHarness.class_id().path)
    kernel = tc.kernel.with_library(
        library,
        data_dir=tmp_path,
        workspace=tmp_path / "workspace",
        token=token,
    )
    assert tc.install(TableRouteHarness, kernel=kernel, data_dir=tmp_path, token=token).status == 204

    with tc.backend(kernel):
        assert library.echo(table=sample_table()).to_json() == sample_table().to_json()
        assert library.row(table=sample_table()) == [1, "one"]
        assert library.insert_then_contains(table=sample_table()) is True
        assert library.upsert_then_read(table=sample_table()) == [1, "updated"]
        assert library.update_then_read(table=sample_table()) == [2, "updated"]
        assert library.update_range_then_read(table=sample_table()) == [1, "selected"]
        assert library.delete_then_contains(table=sample_table()) is False
        assert library.truncate_then_empty(table=sample_table()) is True

        ranged = library.range(table=sample_table())
        assert isinstance(ranged, tc.collection.Table)
        assert ranged.to_json()["/state/collection/table"][1] == [[1, "one"], [2, "two"]]

        ordered = library.order(table=sample_table())
        assert ordered.to_json()["/state/collection/table"][1] == [
            [2, "two"],
            [3, "three"],
            [1, "one"],
        ]

        limited = library.limit(table=sample_table())
        assert limited.to_json()["/state/collection/table"][1] == [[1, "one"]]

        selected = library.select(table=sample_table())
        assert selected.to_json()["/state/collection/table"][1] == [
            [1, "one"],
            [2, "two"],
            [3, "three"],
        ]
