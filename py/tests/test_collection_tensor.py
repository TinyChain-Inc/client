import inspect

import pytest

import tinychain as tc


def _json(value):
    return tc.state.form_of(value).to_json()


def test_tensor_reflection_preserves_tensor_hint():
    class Math(tc.Library):
        publisher = "example-devco"
        resource_name = "math"
        version = "0.1.0"

        @tc.post
        def mm(self, left: tc.Tensor, right: tc.Tensor) -> tc.Tensor:
            return left @ right

    sig = inspect.signature(Math().mm)
    assert sig.parameters["left"].annotation is tc.Tensor
    assert sig.parameters["right"].annotation is tc.Tensor
    assert sig.return_annotation is tc.Tensor


def test_tensor_basic_method_shapes_are_canonical_refs():
    x = tc.Tensor(tc.state.TCRef(tc.state.IdRef("x")))
    y = tc.Tensor(tc.state.TCRef(tc.state.IdRef("y")))

    assert _json(x @ y) == {"$x/matmul": {"r": {"$y": []}}}
    assert _json(x.as_dense()) == {"$x/as_dense": [None]}
    assert _json(x.exp()) == {"$x/exp": {}}
    assert _json(x.reshape([2, 3])) == {"$x/reshape": [[2, 3]]}
    assert _json(x.broadcast([4, 2, 3])) == {"$x/broadcast": [[4, 2, 3]]}
    assert _json(x.transpose([1, 0])) == {"$x/transpose": [[1, 0]]}
    assert _json(x.sum(axes=1, keepdims=True)) == {"$x/sum": {"axes": [1], "keepdims": True}}


def test_tensor_selected_writes_address_the_original_owner():
    tensor = tc.Tensor(tc.state.IdRef("tensor"))
    selected = tensor.slice([1, (1, 3)])
    assert _json(selected) == {"$tensor": [[1, [1, 3]]]}
    assert _json(selected.write(9)) == {"$tensor": [[1, [1, 3]], 9]}
    assert _json(tensor.slice(None).write(9)) == {"$tensor": [None, 9]}

    with tc.scoped_context() as context:
        bound = context.bind("selected", selected)
        assert _json(bound.write(9)) == {"$tensor": [[1, [1, 3]], 9]}

    for transform, target, key in (
        (lambda view: view.slice([(0, 1)]), "$_tmp0", [[0, 1]]),
        (lambda view: view.transpose([0]), "$_tmp1", None),
        (lambda view: view.reshape([2]), "$_tmp1", None),
        (lambda view: view.broadcast([2]), "$_tmp1", None),
        (lambda view: view.as_dense(), "$_tmp1", None),
        (lambda view: view + 0, "$_tmp1", None),
    ):
        with tc.scoped_context():
            result = transform(selected).write(9)
        assert _json(result) == {target: [key, 9]}


def test_tensor_v1_surface_helpers_are_available():
    x = tc.Tensor(tc.state.TCRef(tc.state.IdRef("x")))
    y = tc.Tensor(tc.state.TCRef(tc.state.IdRef("y")))

    assert _json(tc.einsum("ij,jk->ik", [x, y])) == {
        str(tc.URI(tc.state.OpRef, "post")): [
            str(tc.URI("state", "collection", "tensor", "einsum")),
            {"format": "ij,jk->ik", "tensors": [{"$x": []}, {"$y": []}]},
        ]
    }
    assert _json(tc.concatenate([x, y], axis=0)) == {
        str(tc.URI(tc.state.OpRef, "post")): [
            str(tc.URI("state", "collection", "tensor", "concatenate")),
            {"axis": 0, "tensors": [{"$x": []}, {"$y": []}]},
        ]
    }
    assert _json(tc.tile(x, [2, 1])) == {
        str(tc.URI(tc.state.OpRef, "post")): [
            str(tc.URI("state", "collection", "tensor", "tile")),
            {"multiples": [2, 1], "tensor": {"$x": []}},
        ]
    }


def test_tensor_reverse_add_and_mul_use_tensor_subject():
    x = tc.Tensor(tc.state.TCRef(tc.state.IdRef("x")))

    assert _json(1 + x) == {"$x/add": {"r": 1}}
    assert _json(2 * x) == {"$x/mul": {"r": 2}}


def test_tensor_reverse_sub_and_div_require_tensor_lhs():
    x = tc.Tensor(tc.state.TCRef(tc.state.IdRef("x")))

    with pytest.raises(TypeError, match="reverse subtraction"):
        _ = 1 - x

    with pytest.raises(TypeError, match="reverse division"):
        _ = 1 / x


def test_tensor_binary_ops_emit_minimal_payload_when_known():
    x = tc.Tensor(tc.state.TCRef(tc.state.IdRef("x")))
    y = tc.Tensor(tc.state.TCRef(tc.state.IdRef("y")))

    assert _json(x + y) == {
        "$x/add": {
            "r": {"$y": []},
        }
    }


def test_tensor_matmul_emits_minimal_payload_when_known():
    lhs = tc.Tensor(tc.state.TCRef(tc.state.IdRef("lhs")))
    rhs = tc.Tensor(tc.state.TCRef(tc.state.IdRef("rhs")))

    assert _json(lhs @ rhs) == {
        "$lhs/matmul": {
            "r": {"$rhs": []},
        }
    }


def test_tensor_logical_not_emits_minimal_payload_when_known():
    tensor = tc.Tensor(tc.state.TCRef(tc.state.IdRef("tensor")))

    assert _json(tensor.logical_not()) == {
        "$tensor/not": {
        }
    }


def test_tensor_chained_operations_retain_each_operand():
    x = tc.Tensor(tc.state.IdRef("x"))
    with tc.scoped_context() as context:
        result = x.transpose([1, 0]).reshape([6]).broadcast([2, 6]).slice([1, (0, 6, 2)])
        bound = context.bind("result", result)
    assert _json(bound) == {"$result": []}
    operations = [tc.state.form_of(value).to_json() for _, value in context.form()]
    assert operations == [
        {"$x/transpose": [[1, 0]]},
        {"$_tmp0/reshape": [[6]]},
        {"$_tmp1/broadcast": [[2, 6]]},
        {"$_tmp2": [[1, [0, 6, 2]]]},
    ]


def test_tensor_unbound_intermediate_reports_authoring_error():
    x = tc.Tensor(tc.state.IdRef("x"))
    with pytest.raises(TypeError, match="lexical Context"):
        x.transpose([1, 0]).slice([0, (0, 2)])


@pytest.mark.parametrize("dtype", ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64", "c32", "c64"])
@pytest.mark.parametrize("sparse", [False, True])
def test_materialized_tensor_boundary_preserves_dtype_and_layout(dtype, sparse):
    value = 2 + 3j if dtype.startswith("c") else 7
    tensor = tc.Tensor.sparse(dtype, [2, 3], [([1, 2], value)]) if sparse else tc.Tensor.dense(dtype, [2], [0, value])
    from tinychain.state.collection import Collection

    decoded = Collection.from_json(tensor.to_json())
    assert isinstance(decoded, tc.Tensor)
    assert decoded.dtype == dtype
    assert decoded.shape == tensor.shape
    assert decoded.layout == ("sparse" if sparse else "dense")
    assert decoded.values == tensor.values
    assert decoded.to_json() == tensor.to_json()
    if sparse:
        assert decoded.entries == [([1, 2], value)]


def test_large_sparse_tensor_boundary_retains_only_occupied_values():
    tensor = tc.Tensor.sparse("f64", [10**6, 10**6], [([999999, 1], 2.5)])
    decoded = tc.Tensor.from_json(tensor.to_json())
    assert decoded.shape == [10**6, 10**6]
    assert decoded.entries == [([999999, 1], 2.5)]
    assert len(decoded.values) == 1


def test_dense_and_empty_sparse_tensors_share_a_schema():
    dense = tc.Tensor.dense("f64", [2, 3], [0] * 6)
    sparse = tc.Tensor.sparse("f64", [2, 3], [])
    dense_schema, _ = next(iter(dense.to_json().values()))
    sparse_schema, contents = next(iter(sparse.to_json().values()))
    assert sparse_schema == dense_schema
    assert contents == []
    assert tc.Tensor.from_json(sparse.to_json()).entries == []
    for invalid in [{"entries": []}, {}, None]:
        payload = sparse.to_json()
        next(iter(payload.values()))[1] = invalid
        with pytest.raises(TypeError):
            tc.Tensor.from_json(payload)


@pytest.mark.parametrize("entries", [
    [([0], 0)], [([2], 1)], [([0, 0], 1)], [([0], 1), ([0], 2)], [([1], 1), ([0], 2)],
])
def test_sparse_tensor_boundary_rejects_malformed_entries(entries):
    with pytest.raises(ValueError):
        tc.Tensor.sparse("f64", [2], entries)


def test_tensor_boundary_rejects_unknown_layout_and_dtype():
    payload = tc.Tensor.dense("f64", [1], [1]).to_json()
    metadata = next(iter(payload.values()))[0]
    metadata.append("unknown")
    with pytest.raises(TypeError):
        tc.Tensor.from_json(payload)
    metadata.pop()
    metadata[0] = "f64"
    with pytest.raises(TypeError, match="dtype"):
        tc.Tensor.from_json(payload)


def test_materialized_tensor_uses_the_canonical_state_form_in_operations():
    tensor = tc.Tensor.dense("f64", [2, 3], list(range(6)))
    assert tc.state.form_of(tensor) == tensor.to_json()
    with tc.scoped_context() as context:
        result = tensor.transpose([1, 0]).slice([1, (0, 2)])
    assert context.value("_tmp0").to_json() == tensor.to_json()
    assert _json(context.value("_tmp1")) == {"$_tmp0/transpose": [[1, 0]]}
    assert _json(result) == {"$_tmp1": [[1, [0, 2]]]}
