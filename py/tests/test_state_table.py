import pytest

import tinychain as tc
from tinychain.codec import decode_payload


def schema():
    return tc.collection.table.Schema(
        [tc.collection.table.Column("id", tc.U64)],
        [tc.collection.table.Column("name", tc.String)],
    ).create_index("by_name", ["name"])


def payload():
    return {
        "/state/collection/table": [
            [
                [
                    [["id", "/state/scalar/value/number"]],
                    [["name", "/state/scalar/value/string"]],
                ],
                [["by_name", ["name"]]],
            ],
            [[1, "one"], [2, "two"]],
        ]
    }


def test_table_literal_uses_canonical_payload():
    table = tc.collection.Table(schema(), [[1, "one"], [2, "two"]])
    assert table.to_json() == payload()


def test_table_payload_round_trip_and_collection_owned_decode():
    table = tc.collection.Table.from_json(payload())
    assert table.to_json() == payload()
    decoded = decode_payload(payload())
    assert isinstance(decoded, tc.collection.Table)
    assert decoded.to_json() == payload()


def test_table_is_collection_only_export():
    assert issubclass(tc.collection.Table, tc.state.Collection)
    assert not hasattr(tc, "Table")
    assert tc.collection.table.Column is not None
    assert tc.collection.table.Schema is not None


def test_schema_requires_a_key():
    with pytest.raises(ValueError, match="at least one key"):
        tc.collection.table.Schema([])


def test_table_rejects_invalid_payload_shapes():
    with pytest.raises(TypeError, match="each Table row"):
        tc.collection.Table(schema(), [1])
    with pytest.raises(TypeError, match=r"\[schema, rows\]"):
        tc.collection.Table.from_json({"/state/collection/table": []})


def test_table_symbolic_routes_and_typed_results():
    table = tc.collection.Table(tc.state.IdRef("table"))

    assert isinstance(table[[1]], tc.state.scalar.Tuple)
    assert table[[1]].to_json() == {"$table": [[1]]}
    assert table.contains([1]).to_json() == {"$table/contains": [[1]]}
    assert table.columns().to_json() == {"$table/columns": [None]}
    assert table.count().to_json() == {"$table/count": [None]}
    assert table.is_empty().to_json() == {"$table/is_empty": [None]}
    assert table.key_columns().to_json() == {"$table/key_columns": [None]}
    assert table.key_names().to_json() == {"$table/key_names": [None]}
    assert isinstance(table.limit(1), tc.collection.Table)
    assert table.limit(1).to_json() == {"$table/limit": [1]}
    assert table.order_by(["name"], reverse=True).to_json() == {
        "$table/order": [[["name"], True]]
    }
    assert table.select(["name"]).to_json() == {"$table/select": [["name"]]}


def test_table_slice_bounds_are_tagged_and_half_open():
    table = tc.collection.Table(tc.state.IdRef("table"))
    assert table.where(id=slice(1, 3)).to_json() == {
        "$table": {"id": [["in", 1], ["ex", 3]]}
    }
    assert table.where(id=slice(None, 3)).to_json() == {
        "$table": {"id": [None, ["ex", 3]]}
    }
    assert table.where() is table
    with pytest.raises(ValueError, match="unit step"):
        table.where(id=slice(1, 3, 2))


def test_table_mutations_use_native_verb_refs():
    table = tc.collection.Table(tc.state.IdRef("table"))
    assert table.insert([1], ["one"]).to_json() == {
        "$table/insert": {"key": [1], "values": ["one"]}
    }
    assert table.upsert([1], ["one"]).to_json() == {"$table": [[1], ["one"]]}
    assert table.update(name="updated").to_json() == {
        "$table": [None, {"name": "updated"}]
    }
    assert table.delete([1]).to_json() == {
        "/state/scalar/ref/op/delete": ["$table", [1]]
    }
    assert table.truncate().to_json() == {
        "/state/scalar/ref/op/delete": ["$table", None]
    }
