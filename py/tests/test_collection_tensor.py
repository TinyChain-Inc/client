import inspect

import pytest
import tinychain as tc
import tinychain.collection.tensor as tensor_module


def _json(value):
    return tc.state.form_of(value).to_json()


def test_tensor_reflection_preserves_tensor_hint():
    class Math(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def mm(self, left: tc.Tensor, right: tc.Tensor) -> tc.Tensor:
            return left @ right

    sig = inspect.signature(Math().mm)
    assert sig.parameters["left"].annotation is tc.Tensor
    assert sig.parameters["right"].annotation is tc.Tensor
    assert sig.return_annotation is tc.Tensor


def test_tensor_basic_method_shapes_are_canonical_refs():
    x = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))
    y = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("y")))

    assert _json(x @ y) == {"$x/matmul": {"r": {"$y": []}}}
    assert _json(x.reshape([2, 3])) == {"$x/reshape": [[2, 3]]}
    assert _json(x.broadcast([4, 2, 3])) == {"$x/broadcast": [[4, 2, 3]]}
    assert _json(x.transpose([1, 0])) == {"$x/transpose": [[1, 0]]}
    assert _json(x.sum(axes=1, keepdims=True)) == {"$x/sum": {"axes": [1], "keepdims": True}}


def test_tensor_v1_surface_helpers_are_available():
    x = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))
    y = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("y")))

    assert _json(tc.einsum("ij,jk->ik", [x, y])) == {
        str(tc.uri("state", "collection", "tensor", "einsum")): {
            "format": "ij,jk->ik",
            "tensors": [{"$x": []}, {"$y": []}],
        }
    }
    assert _json(tc.concatenate([x, y], axis=0)) == {
        str(tc.uri("state", "collection", "tensor", "concatenate")): {
            "axis": 0,
            "tensors": [{"$x": []}, {"$y": []}],
        }
    }
    assert _json(tc.tile(x, [2, 1])) == {
        str(tc.uri("state", "collection", "tensor", "tile")): {
            "multiples": [2, 1],
            "tensor": {"$x": []},
        }
    }


def test_tensor_wrapper_uses_canonical_ref_builders():
    source = inspect.getsource(tensor_module)

    assert "TCRef(" not in source
    assert "PostOpRef(" not in source
    assert "GetOpRef(" not in source
    assert "PutOpRef(" not in source
    assert "DeleteOpRef(" not in source
    assert ".path" not in source


def test_tensor_internal_route_helpers_are_not_public_api():
    assert not hasattr(tensor_module, "tensor_route")
    assert not hasattr(tensor_module, "TENSOR_CLASS_URI")


def test_tensor_reverse_add_and_mul_use_tensor_subject():
    x = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))

    assert _json(1 + x) == {"$x/add": {"r": 1}}
    assert _json(2 * x) == {"$x/mul": {"r": 2}}


def test_tensor_reverse_sub_and_div_require_tensor_lhs():
    x = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))

    with pytest.raises(TypeError, match="reverse subtraction"):
        _ = 1 - x

    with pytest.raises(TypeError, match="reverse division"):
        _ = 1 / x


def test_tensor_binary_ops_emit_minimal_payload_when_known():
    x = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))
    y = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("y")))

    assert _json(x + y) == {
        "$x/add": {
            "r": {"$y": []},
        }
    }


def test_tensor_matmul_emits_minimal_payload_when_known():
    lhs = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("lhs")))
    rhs = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("rhs")))

    assert _json(lhs @ rhs) == {
        "$lhs/matmul": {
            "r": {"$rhs": []},
        }
    }


def test_tensor_logical_not_emits_minimal_payload_when_known():
    tensor = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("tensor")))

    assert _json(tensor.logical_not()) == {
        "$tensor/not": {
        }
    }


def test_tensor_records_view_ops_for_symbolic_transforms():
    x = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))

    y = x.transpose([1, 0]).broadcast([3, 2, 4]).reshape([24]).slice([0, 10])

    assert [op.kind for op in y.view_ops] == ["transpose", "broadcast", "reshape", "slice"]
    assert y.view_ops[0].permutation == (1, 0)
    assert y.view_ops[1].shape == (3, 2, 4)
    assert y.view_ops[2].shape == (24,)


def test_tensor_native_transform_path_uses_native_backend():
    class NativeTensor:
        def __init__(self, shape):
            self.shape = shape

        def transpose(self, permutation):
            return NativeTensor([self.shape[i] for i in permutation])

        def broadcast(self, shape):
            return NativeTensor(list(shape))

        def reshape(self, shape):
            return NativeTensor(list(shape))

    x = tc.Tensor(native=NativeTensor([2, 3]))

    y = x.transpose([1, 0]).broadcast([4, 3, 2]).reshape([24])

    assert isinstance(y.native, NativeTensor)
    assert y.native.shape == [24]
    assert [op.kind for op in y.view_ops] == ["transpose", "broadcast", "reshape"]
    assert y.view_ops[1].broadcast_axes == (0,)


def test_tensor_view_spec_is_canonicalized_from_transform_chain():
    x = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))

    y = x.transpose([1, 0]).reshape([6]).broadcast([2, 6])
    spec = y.view_spec()

    assert [op.kind for op in spec.ops] == ["transpose", "reshape", "broadcast"]
    assert spec.ops[0].permutation == (1, 0)
    assert spec.ops[1].shape == (6,)
    assert spec.ops[2].shape == (2, 6)


def test_tensor_materialize_view_spec_uses_backend_adapter_if_supported():
    class Adapter:
        def __init__(self, shape):
            self.shape = shape

        def apply_view_op(self, op):
            if op.kind == "transpose":
                return Adapter([self.shape[i] for i in op.permutation])
            if op.kind == "reshape":
                return Adapter(list(op.shape))
            return Adapter(self.shape)

        def apply_view_spec(self, spec):
            current = self
            for op in spec.ops:
                current = current.apply_view_op(op)
            return current

    x = tc.Tensor(native=Adapter([2, 3]))
    y = x.transpose([1, 0])

    z = y.materialize_view_spec()
    assert isinstance(z.native, Adapter)
    assert z.native.shape == [3, 2]


def test_tensor_materialize_view_spec_prefers_wire_backend_hook():
    class Adapter:
        def __init__(self, shape):
            self.shape = shape
            self.wire = None

        def apply_view_op(self, op):
            return self

        def apply_view_spec(self, spec):
            raise AssertionError("legacy view_spec path should not be used")

        def apply_view_wire(self, wire):
            self.wire = wire
            return Adapter([6])

    x = tc.Tensor(native=Adapter([2, 3]))
    y = x.transpose([1, 0])
    y._view_ops_materialized = False

    z = y.materialize_view_spec()
    assert isinstance(z.native, Adapter)
    assert z.native.shape == [6]


def test_view_spec_compiles_transpose_and_broadcast_to_view_schema():
    x = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))
    spec = x.transpose([1, 0]).broadcast([3, 2]).view_spec()

    schema = spec.to_view_schema(base_shape=[1, 3])
    schema_json = schema.to_json()

    assert schema.base_rank == 2
    assert list(schema.base_fixed) == [None, None]
    assert schema_json["axes"] == [
        {"base_axis": 1, "map": {"identity": True}},
        {"base_axis": 0, "map": {"affine": {"start": 0, "step": 0}}},
    ]

def test_tensor_to_view_schema_requires_base_shape_for_symbolic_tensors():
    x = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x"))).transpose([1, 0])

    with pytest.raises(TypeError, match="base_shape"):
        x.to_view_schema()


def test_tensor_to_view_schema_uses_native_shape_when_available():
    class NativeTensor:
        def __init__(self, shape):
            self.shape = shape

        def transpose(self, permutation):
            return NativeTensor([self.shape[i] for i in permutation])

    x = tc.Tensor(native=NativeTensor([2, 3])).transpose([1, 0])
    schema = x.to_view_schema()
    schema_json = schema.to_json()

    assert schema.base_rank == 2
    assert schema_json["axes"] == [
        {"base_axis": 1, "map": {"identity": True}},
        {"base_axis": 0, "map": {"identity": True}},
    ]


def test_view_spec_compiles_slice_at_and_in_to_view_schema():
    x = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))
    spec = x.slice([1, (0, 4, 2)]).view_spec()

    schema = spec.to_view_schema(base_shape=[3, 4])
    schema_json = schema.to_json()

    assert schema.base_rank == 2
    assert list(schema.base_fixed) == [1, None]
    assert schema_json["axes"] == [
        {"base_axis": 1, "map": {"affine": {"start": 0, "step": 2}}},
    ]


def test_view_spec_slice_rank_mismatch_raises():
    x = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))
    spec = x.slice([0]).view_spec()

    with pytest.raises(ValueError, match="rank"):
        spec.to_view_schema(base_shape=[2, 3])


def test_tensor_storage_schema_compiles_concrete_shape_and_layout():
    x = tc.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))

    schema = x.to_storage_schema(base_shape=[2, 3], layout="sparse", sparse_axis=1)

    assert schema.dtype == "f32"
    assert schema.shape == (2, 3)
    assert schema.layout.kind == "sparse"
    assert schema.layout.sparse_axis == 1
