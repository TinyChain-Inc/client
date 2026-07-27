"""Bridge module converting autodiff types to generic reflection primitives.

This module owns the adapter logic between autodiff-specific types
(DerivativeProgram, TensorNodeRecord) and domain-neutral reflection primitives
(TypeSpec, TypedValueRef) from tinychain.graph_reflection. The bridge imports
from both graph_reflection and autodiff, but graph_reflection.py never imports
from autodiff — preserving domain neutrality.
"""
from __future__ import annotations

from ..graph_reflection import ReflectionError, TypeSpec, TypedValueRef
from ..state.base import State
from ..uri import path, uri
from .reverse import DerivativeProgram


def tensor_typespec_to_type_spec(typespec_dict: dict) -> TypeSpec:
    """Convert a TensorNodeRecord output_typespec dict to a TypeSpec.

    The canonical class URI for all tensor types is "/state/collection/tensor".
    The params dict is passed through as-is (it may contain dtype, shape, etc.).

    Args:
        typespec_dict: A dict with tensor type metadata (e.g., {"dtype": "float32", "shape": [3, 4]}).

    Returns:
        A TypeSpec with class_uri="/state/collection/tensor" and the provided params.

    Raises:
        ReflectionError: If typespec_dict is not a dict or is missing required keys.
    """
    if not isinstance(typespec_dict, dict):
        raise ReflectionError(
            "invalid_type_spec",
            f"tensor_typespec must be a dict, got {type(typespec_dict).__name__}",
        )
    return TypeSpec(class_uri=path(uri(State, "collection", "tensor")), params=dict(typespec_dict))


def reflect_derivative_program(program: DerivativeProgram) -> list[TypedValueRef]:
    """Convert a DerivativeProgram to a list of TypedValueRef for reflection.

    Each node in the derivative program becomes one TypedValueRef, mapping
    the node's output value and node ID to the reflection schema. The namespace
    is the source graph ID from the program metadata.

    Args:
        program: A DerivativeProgram from autodiff.reverse.

    Returns:
        A list of TypedValueRef objects, one per node in program.nodes.
        Each TypedValueRef has:
        - namespace: program.metadata.source_graph_id
        - value: node.output_value_id
        - output: node.node_id
        - value_type: TypeSpec("/state/collection/tensor", ...) from node.output_typespec,
          or an empty-params TypeSpec when output_typespec is None.
    """
    namespace = program.metadata.source_graph_id
    refs: list[TypedValueRef] = []
    for node in program.nodes:
        if node.output_typespec is not None:
            value_type = tensor_typespec_to_type_spec(node.output_typespec)
        else:
            value_type = TypeSpec(class_uri=path(uri(State, "collection", "tensor")), params={})
        ref = TypedValueRef(
            namespace=namespace,
            value=node.output_value_id,
            output=node.node_id,
            value_type=value_type,
        )
        refs.append(ref)
    return refs
