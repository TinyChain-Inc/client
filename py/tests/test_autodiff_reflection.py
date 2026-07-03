"""Unit tests for autodiff reflection bridge module."""
from __future__ import annotations

import pytest

from tinychain.autodiff import (
    DerivativeProgram,
    TensorNodeRecord,
    reflect_derivative_program,
    tensor_typespec_to_type_spec,
)
from tinychain.autodiff.graph import AddOperator
from tinychain.autodiff.protocol import DerivativeMetadata
from tinychain.graph_reflection import ReflectionError


def test_tensor_typespec_to_type_spec_valid() -> None:
    """tensor_typespec_to_type_spec converts a valid dict to TypeSpec."""
    typespec_dict = {"dtype": "float32", "shape": [3, 4]}
    result = tensor_typespec_to_type_spec(typespec_dict)
    assert result.class_uri == "/state/collection/tensor"
    assert result.params == typespec_dict


def test_tensor_typespec_to_type_spec_empty_params() -> None:
    """tensor_typespec_to_type_spec handles empty params dict."""
    result = tensor_typespec_to_type_spec({})
    assert result.class_uri == "/state/collection/tensor"
    assert result.params == {}


def test_tensor_typespec_to_type_spec_invalid_type() -> None:
    """tensor_typespec_to_type_spec raises ReflectionError for non-dict input."""
    with pytest.raises(ReflectionError) as exc_info:
        tensor_typespec_to_type_spec("not a dict")
    assert exc_info.value.category == "invalid_type_spec"
    assert "must be a dict" in str(exc_info.value)


def test_reflect_derivative_program_single_node() -> None:
    """reflect_derivative_program returns one TypedValueRef per node."""
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v0",
        operator=AddOperator(),
        op_params={},
        input_value_ids=[],
        output_typespec={"dtype": "float32", "shape": [2, 2]},
    )
    program = DerivativeProgram(
        nodes=[node],
        gradients={},
        output_gradients=[],
        metadata=DerivativeMetadata(
            source_graph_id="test_graph",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=(),
            seed_contract="",
        ),
    )
    refs = reflect_derivative_program(program)
    assert len(refs) == 1
    assert refs[0].namespace == "test_graph"
    assert refs[0].value == "v0"
    assert refs[0].output == "n0"
    assert refs[0].value_type.class_uri == "/state/collection/tensor"
    assert refs[0].value_type.params == {"dtype": "float32", "shape": [2, 2]}


def test_reflect_derivative_program_multiple_nodes() -> None:
    """reflect_derivative_program returns correct TypedValueRef for each node."""
    node1 = TensorNodeRecord(
        node_id="n0",
        output_value_id="v0",
        operator=AddOperator(),
        op_params={},
        input_value_ids=[],
        output_typespec={"dtype": "float32", "shape": [2, 2]},
    )
    node2 = TensorNodeRecord(
        node_id="n1",
        output_value_id="v1",
        operator=AddOperator(),
        op_params={},
        input_value_ids=["v0"],
        output_typespec={"dtype": "float32", "shape": [2, 2]},
    )
    program = DerivativeProgram(
        nodes=[node1, node2],
        gradients={},
        output_gradients=[],
        metadata=DerivativeMetadata(
            source_graph_id="test_graph",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=(),
            seed_contract="",
        ),
    )
    refs = reflect_derivative_program(program)
    assert len(refs) == 2
    assert refs[0].value == "v0"
    assert refs[0].output == "n0"
    assert refs[1].value == "v1"
    assert refs[1].output == "n1"
    for ref in refs:
        assert ref.namespace == "test_graph"
        assert ref.value_type.class_uri == "/state/collection/tensor"


def test_reflect_derivative_program_none_typespec() -> None:
    """Nodes with output_typespec=None get an empty-params TypeSpec."""
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v0",
        operator=AddOperator(),
        op_params={},
        input_value_ids=[],
        output_typespec=None,
    )
    program = DerivativeProgram(
        nodes=[node],
        gradients={},
        output_gradients=[],
        metadata=DerivativeMetadata(
            source_graph_id="test_graph",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=(),
            seed_contract="",
        ),
    )
    refs = reflect_derivative_program(program)
    assert len(refs) == 1
    assert refs[0].value_type.class_uri == "/state/collection/tensor"
    assert refs[0].value_type.params == {}


def test_reflect_derivative_program_namespace_from_metadata() -> None:
    """TypedValueRef.namespace equals program.metadata.source_graph_id."""
    node = TensorNodeRecord(
        node_id="n0",
        output_value_id="v0",
        operator=AddOperator(),
        op_params={},
        input_value_ids=[],
        output_typespec={},
    )
    program = DerivativeProgram(
        nodes=[node],
        gradients={},
        output_gradients=[],
        metadata=DerivativeMetadata(
            source_graph_id="custom_namespace",
            transform_version="0.1.0",
            tensor_op_contract_version="0.1.0",
            wrt_signature=(),
            seed_contract="",
        ),
    )
    refs = reflect_derivative_program(program)
    assert refs[0].namespace == "custom_namespace"
