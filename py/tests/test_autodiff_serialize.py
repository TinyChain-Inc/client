"""Tests for autodiff serialization utility and to_dict() implementations."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tinychain.autodiff.graph import (
    AddOperator,
    MatmulOperator,
    TensorNodeRecord,
    TransposeOperator,
)
from tinychain.autodiff.protocol import (
    AutodiffError,
    AutodiffRequest,
    AutodiffResult,
    DerivativeMetadata,
)
from tinychain.autodiff.reverse import DerivativeProgram, ReverseTraversal
from tinychain.serialize import register_serializer, serialize


class TestSerializeUtility:
    """Test the serialize() utility function."""

    def test_serialize_tensor_operator(self) -> None:
        """TensorOperator instances serialize with type and route_name."""
        op = AddOperator()
        result = serialize(op)
        assert result == {"type": "AddOperator", "route_name": "add"}

    def test_serialize_dataclass(self) -> None:
        """Dataclasses serialize field-by-field."""
        metadata = DerivativeMetadata(
            source_graph_id="graph_123",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=("x", "y"),
            seed_contract="seed_v0",
        )
        result = serialize(metadata)
        assert result == {
            "source_graph_id": "graph_123",
            "transform_version": "0.1.0",
            "tensor_op_contract_version": "0.1.0",
            "wrt_signature": ["x", "y"],
            "seed_contract": "seed_v0",
        }

    def test_serialize_list(self) -> None:
        """Lists serialize element-by-element."""
        lst = [1, "two", 3.0, None]
        result = serialize(lst)
        assert result == [1, "two", 3.0, None]

    def test_serialize_dict(self) -> None:
        """Dicts serialize key-value pairs."""
        d = {"a": 1, "b": "two", "c": None}
        result = serialize(d)
        assert result == {"a": 1, "b": "two", "c": None}

    def test_serialize_scalar_passthrough(self) -> None:
        """Scalars pass through unchanged."""
        assert serialize(42) == 42
        assert serialize("hello") == "hello"
        assert serialize(3.14) == 3.14
        assert serialize(True) is True
        assert serialize(None) is None

    def test_serialize_nested_structure(self) -> None:
        """Nested structures serialize recursively."""
        op = MatmulOperator()
        node = TensorNodeRecord(
            node_id="n0",
            output_value_id="v2",
            operator=op,
            op_params={"axis": 0},
            input_value_ids=["v0", "v1"],
            output_typespec={"dtype": "float32", "shape": [3, 4]},
        )
        result = serialize(node)
        assert result["node_id"] == "n0"
        assert result["output_value_id"] == "v2"
        assert result["operator"] == {"type": "MatmulOperator", "route_name": "matmul"}
        assert result["op_params"] == {"axis": 0}
        assert result["input_value_ids"] == ["v0", "v1"]
        assert result["output_typespec"] == {"dtype": "float32", "shape": [3, 4]}

    def test_register_serializer_exact_type(self) -> None:
        """register_serializer dispatches on exact type match."""
        class CustomType:
            def __init__(self, value: str) -> None:
                self.value = value

        def handler(obj: CustomType) -> dict:
            return {"custom": obj.value}

        register_serializer(CustomType, handler)
        obj = CustomType("test")
        result = serialize(obj)
        assert result == {"custom": "test"}

    def test_register_serializer_isinstance_fallback(self) -> None:
        """register_serializer falls back to isinstance checks for base classes."""
        class Base:
            def __init__(self, value: str) -> None:
                self.value = value

        class Derived(Base):
            pass

        def handler(obj: Base) -> dict:
            return {"base": obj.value}

        register_serializer(Base, handler)
        obj = Derived("test")
        result = serialize(obj)
        assert result == {"base": "test"}


class TestTensorNodeRecordToDict:
    """Test TensorNodeRecord.to_dict() implementation."""

    def test_to_dict_basic(self) -> None:
        """TensorNodeRecord.to_dict() serializes all fields."""
        op = AddOperator()
        node = TensorNodeRecord(
            node_id="n0",
            output_value_id="v2",
            operator=op,
            op_params={},
            input_value_ids=["v0", "v1"],
        )
        result = node.to_dict()
        assert result["node_id"] == "n0"
        assert result["output_value_id"] == "v2"
        assert result["operator"] == {"type": "AddOperator", "route_name": "add"}
        assert result["op_params"] == {}
        assert result["input_value_ids"] == ["v0", "v1"]
        assert result["output_typespec"] is None

    def test_to_dict_with_typespec(self) -> None:
        """TensorNodeRecord.to_dict() includes output_typespec when present."""
        op = TransposeOperator()
        node = TensorNodeRecord(
            node_id="n1",
            output_value_id="v3",
            operator=op,
            op_params={"axes": [1, 0]},
            input_value_ids=["v2"],
            output_typespec={"dtype": "float32", "shape": [4, 3]},
        )
        result = node.to_dict()
        assert result["output_typespec"] == {"dtype": "float32", "shape": [4, 3]}

    def test_to_dict_json_serializable(self) -> None:
        """TensorNodeRecord.to_dict() output is JSON-serializable."""
        op = MatmulOperator()
        node = TensorNodeRecord(
            node_id="n2",
            output_value_id="v4",
            operator=op,
            op_params={"transpose_a": False},
            input_value_ids=["v2", "v3"],
            output_typespec={"dtype": "float32", "shape": [3, 3]},
        )
        result = node.to_dict()
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        assert len(json_str) > 0


class TestDerivativeProgramToDict:
    """Test DerivativeProgram.to_dict() implementation."""

    def test_to_dict_basic(self) -> None:
        """DerivativeProgram.to_dict() serializes all fields."""
        op = AddOperator()
        node = TensorNodeRecord(
            node_id="n0",
            output_value_id="v2",
            operator=op,
            op_params={},
            input_value_ids=["v0", "v1"],
        )
        metadata = DerivativeMetadata(
            source_graph_id="graph_123",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=("x",),
            seed_contract="seed_v0",
        )
        program = DerivativeProgram(
            nodes=[node],
            gradients={"v0": "v_grad_0"},
            output_gradients=["v_grad_0", None],
            metadata=metadata,
        )
        result = program.to_dict()
        assert isinstance(result["nodes"], list)
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["node_id"] == "n0"
        assert result["gradients"] == {"v0": "v_grad_0"}
        assert result["output_gradients"] == ["v_grad_0", None]
        assert result["metadata"]["source_graph_id"] == "graph_123"

    def test_to_dict_json_serializable(self) -> None:
        """DerivativeProgram.to_dict() output is JSON-serializable."""
        op = TransposeOperator()
        node = TensorNodeRecord(
            node_id="n0",
            output_value_id="v1",
            operator=op,
            op_params={"axes": [1, 0]},
            input_value_ids=["v0"],
            output_typespec={"dtype": "float32", "shape": [4, 3]},
        )
        metadata = DerivativeMetadata(
            source_graph_id="graph_456",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=("x",),
            seed_contract="seed_v0",
        )
        program = DerivativeProgram(
            nodes=[node],
            gradients={"v0": "v_grad_0"},
            output_gradients=["v_grad_0"],
            metadata=metadata,
        )
        result = program.to_dict()
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        assert len(json_str) > 0


class TestPreRefactorSnapshots:
    """Capture pre-refactor snapshots of existing to_dict() methods.
    
    These fixtures document the current behavior before refactoring
    AutodiffError, AutodiffResult, and DerivativeMetadata to use _serialize().
    """

    def test_autodiff_error_to_dict_snapshot(self) -> None:
        """Snapshot: AutodiffError.to_dict() output before refactoring."""
        error = AutodiffError(
            category="unsupported_operator",
            message="operator not supported",
        )
        result = error.to_dict()
        # Current implementation: hand-rolled dict with category and message
        assert result == {
            "category": "unsupported_operator",
            "message": "operator not supported",
        }
        # Verify it's JSON-serializable
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_derivative_metadata_to_dict_snapshot(self) -> None:
        """Snapshot: DerivativeMetadata.to_dict() output before refactoring."""
        metadata = DerivativeMetadata(
            source_graph_id="graph_789",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=("x", "y"),
            seed_contract="seed_v0",
        )
        result = metadata.to_dict()
        # Current implementation: hand-rolled dict with all fields
        assert result == {
            "source_graph_id": "graph_789",
            "transform_version": "0.1.0",
            "tensor_op_contract_version": "0.1.0",
            "wrt_signature": ["x", "y"],
            "seed_contract": "seed_v0",
        }
        # Verify it's JSON-serializable
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_autodiff_result_to_dict_snapshot(self) -> None:
        """Snapshot: AutodiffResult.to_dict() output before refactoring."""
        metadata = DerivativeMetadata(
            source_graph_id="graph_999",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=("x",),
            seed_contract="seed_v0",
        )
        result_obj = AutodiffResult(
            gradients=[1.0, 2.0, 3.0],
            metadata=metadata,
        )
        result = result_obj.to_dict()
        # Current implementation: hand-rolled dict with gradients list and nested metadata dict
        assert result == {
            "gradients": [1.0, 2.0, 3.0],
            "metadata": {
                "source_graph_id": "graph_999",
                "transform_version": "0.1.0",
                "tensor_op_contract_version": "0.1.0",
                "wrt_signature": ["x"],
                "seed_contract": "seed_v0",
            },
        }
        # Verify it's JSON-serializable
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_autodiff_request_to_dict_snapshot(self) -> None:
        """Snapshot: AutodiffRequest.to_dict() output before refactoring.
        
        Note: self.graph is object-typed and can be a non-JSON-primitive.
        In this test, we use a dict (JSON-serializable), but in practice
        it could be any object. The snapshot captures the current behavior.
        """
        request = AutodiffRequest(
            graph={"nodes": []},
            output_value_id="out",
            wrt=["x", "y"],
            seed_value_id="seed",
            tensor_op_contract_version="0.1.0",
            transform_version="0.1.0",
        )
        result = request.to_dict()
        # Current implementation: hand-rolled dict with all fields
        assert result == {
            "graph": {"nodes": []},
            "output_value_id": "out",
            "wrt": ["x", "y"],
            "seed_value_id": "seed",
            "tensor_op_contract_version": "0.1.0",
            "transform_version": "0.1.0",
        }
        # Verify it's JSON-serializable
        json_str = json.dumps(result)
        assert isinstance(json_str, str)


class TestToDictSnapshotCompatibility:
    """Test that refactored to_dict() methods match pre-refactor snapshots.
    
    After refactoring AutodiffError, AutodiffResult, and DerivativeMetadata
    to use _serialize(), verify that output matches the captured snapshots.
    Any deviations are documented and justified here.
    """

    def test_autodiff_error_snapshot_compatibility(self) -> None:
        """AutodiffError.to_dict() output matches pre-refactor snapshot."""
        error = AutodiffError(
            category="missing_derivative_behavior",
            message="no VJP rule for operator",
        )
        result = error.to_dict()
        # AutodiffError.to_dict() delegates to _serialize(). _serialize() only
        # adds a "type" key for TensorOperator instances; AutodiffError is a
        # plain dataclass, so it serializes field-by-field with no "type" key,
        # matching the pre-refactor hand-rolled output exactly.
        assert result == {
            "category": "missing_derivative_behavior",
            "message": "no VJP rule for operator",
        }

    def test_derivative_metadata_snapshot_compatibility(self) -> None:
        """DerivativeMetadata.to_dict() output matches pre-refactor snapshot."""
        metadata = DerivativeMetadata(
            source_graph_id="graph_abc",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=("a", "b", "c"),
            seed_contract="seed_v0",
        )
        result = metadata.to_dict()
        # After refactoring to use _serialize(), output should match snapshot.
        # _serialize() will serialize all fields recursively.
        assert result == {
            "source_graph_id": "graph_abc",
            "transform_version": "0.1.0",
            "tensor_op_contract_version": "0.1.0",
            "wrt_signature": ["a", "b", "c"],
            "seed_contract": "seed_v0",
        }

    def test_autodiff_result_snapshot_compatibility(self) -> None:
        """AutodiffResult.to_dict() output matches pre-refactor snapshot."""
        metadata = DerivativeMetadata(
            source_graph_id="graph_def",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=("x", "y"),
            seed_contract="seed_v0",
        )
        result_obj = AutodiffResult(
            gradients=[1.5, 2.5],
            metadata=metadata,
        )
        result = result_obj.to_dict()
        # After refactoring to use _serialize(), output should match snapshot.
        assert result == {
            "gradients": [1.5, 2.5],
            "metadata": {
                "source_graph_id": "graph_def",
                "transform_version": "0.1.0",
                "tensor_op_contract_version": "0.1.0",
                "wrt_signature": ["x", "y"],
                "seed_contract": "seed_v0",
            },
        }

    def test_autodiff_request_snapshot_compatibility(self) -> None:
        """AutodiffRequest.to_dict() output matches pre-refactor snapshot.
        
        _serialize() on a plain dataclass with no special-cased type name
        produces the same field-by-field dict, so this should pass with
        no behavior change.
        """
        request = AutodiffRequest(
            graph={"nodes": []},
            output_value_id="out",
            wrt=["x", "y"],
            seed_value_id="seed",
            tensor_op_contract_version="0.1.0",
            transform_version="0.1.0",
        )
        result = request.to_dict()
        # After refactoring to use _serialize(), output should match snapshot.
        assert result == {
            "graph": {"nodes": []},
            "output_value_id": "out",
            "wrt": ["x", "y"],
            "seed_value_id": "seed",
            "tensor_op_contract_version": "0.1.0",
            "transform_version": "0.1.0",
        }
