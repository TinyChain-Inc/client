from __future__ import annotations

import tinychain as tc
from tinychain.state.tensor._wire import (
    decode_storage_layout,
    decode_storage_schema,
    decode_view_schema,
    encode_storage_layout,
    encode_storage_schema,
    encode_view_schema,
)
from tinychain.state.tensor.schema import TensorStorageLayout


def test_storage_schema_wire_roundtrip():
    x = tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))

    schema = x.to_storage_schema(base_shape=[2, 3], layout="sparse", sparse_axis=1)
    wire = encode_storage_schema(schema)

    assert wire == ("f32", [2, 3], (1, 1))
    assert decode_storage_schema(wire) == schema


def test_storage_layout_wire_roundtrip():
    dense = TensorStorageLayout(kind="dense")
    sparse = TensorStorageLayout(kind="sparse", sparse_axis=2)

    assert encode_storage_layout(dense) == (0, None)
    assert encode_storage_layout(sparse) == (1, 2)
    assert decode_storage_layout((0, None)) == dense
    assert decode_storage_layout((1, 2)) == sparse


def test_view_schema_wire_contract_shape():
    x = tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))
    wire = encode_view_schema(
        x.transpose([1, 0]).broadcast([3, 2]).to_view_schema(base_shape=[1, 3])
    )

    assert wire == (
        2,
        [
            (1, (0, [])),
            (0, (1, [0, 0])),
        ],
        [None, None],
    )
    assert encode_view_schema(decode_view_schema(wire)) == wire
