import inspect

import pytest
import tinychain as tc
import tinychain.state.tensor as tensor_module


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
    x = tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))
    y = tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef("y")))

    assert _json(x @ y) == {"$x/matmul": {"r": {"$y": []}}}
    assert _json(x.reshape([2, 3])) == {"$x/reshape": [[2, 3]]}
    assert _json(x.broadcast([4, 2, 3])) == {"$x/broadcast": [[4, 2, 3]]}
    assert _json(x.transpose([1, 0])) == {"$x/transpose": [[1, 0]]}
    assert _json(x.sum(axes=1, keepdims=True)) == {"$x/sum": {"axes": [1], "keepdims": True}}


def test_tensor_v1_surface_helpers_are_available():
    x = tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))
    y = tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef("y")))

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


def test_tensor_reverse_add_and_mul_use_tensor_subject():
    x = tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))

    assert _json(1 + x) == {"$x/add": {"r": 1}}
    assert _json(2 * x) == {"$x/mul": {"r": 2}}


def test_tensor_reverse_sub_and_div_require_tensor_lhs():
    x = tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))

    with pytest.raises(TypeError, match="reverse subtraction"):
        _ = 1 - x

    with pytest.raises(TypeError, match="reverse division"):
        _ = 1 / x


def test_tensor_binary_ops_emit_minimal_payload_when_known():
    x = tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef("x")))
    y = tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef("y")))

    assert _json(x + y) == {
        "$x/add": {
            "r": {"$y": []},
        }
    }


def test_tensor_matmul_emits_minimal_payload_when_known():
    lhs = tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef("lhs")))
    rhs = tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef("rhs")))

    assert _json(lhs @ rhs) == {
        "$lhs/matmul": {
            "r": {"$rhs": []},
        }
    }


def test_tensor_logical_not_emits_minimal_payload_when_known():
    tensor = tc.state.Tensor(ref=tc.state.TCRef(tc.state.IdRef("tensor")))

    assert _json(tensor.logical_not()) == {
        "$tensor/not": {
        }
    }
