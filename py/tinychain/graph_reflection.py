"""Python-owned generic reflection primitives for consumer-owned graphs.

This module provides small metadata helpers that consumers can use to describe
value types, typed value references, operation contracts, and local output type
inference. It intentionally does not own graph topology, scheduling, or remote
execution. Topology, scheduling, and execution remain consumer-owned; in other
words, topology and execution remain consumer-owned.

The first intended client-side consumer is the existing TensorGraphBuilder /
TensorNodeRecord shape from Phase 1 autodiff. Method URIs are metadata and local
resolver lookup keys only; they do not replace concrete TensorOperator VJP
dispatch and they do not imply server-side derivative execution.

Tensor-like consumer example::

    tensor_spec = TypeSpec(
        class_uri="/state/collection/tensor",
        params={"dtype": "float32", "shape": [3, 4]},
    )
    output_ref = TypedValueRef(
        namespace="my_graph",
        value="node_0_out",
        output=None,
        value_type=tensor_spec,
    )

    def infer_tensor_identity(inputs, params):
        return [tensor_spec]

    contract = OperationContract(
        method_uri="/tensor/identity/v1",
        params_schema={},
        output_type_rule=infer_tensor_identity,
    )
    inferred_outputs = contract.infer_outputs([output_ref.value_type], {})

Non-tensor consumer example::

    cost_vector_spec = TypeSpec(
        class_uri="/planner/cost_vector",
        params={"dimension": 16},
    )
    cost_ref = TypedValueRef(
        namespace="query_planner",
        value="join_order_cost",
        output="cost",
        value_type=cost_vector_spec,
    )

    # This type class has no tensor dtype or shape fields. A query planner can
    # still attach domain-specific metadata and validate its own graph topology.

Missing-inference failure example::

    contract = OperationContract(method_uri="/unsupported/op", params_schema={})
    contract.infer_outputs([], {})
    # Raises OperationContractError("missing_output_type_rule", ...).
    # Topology validation belongs to the consumer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


class ReflectionError(Exception):
    """Raised for invalid type spec or JSON-serialization failures.

    First positional argument is a machine-readable category string.
    """

    def __init__(self, category: str, *args: object) -> None:
        super().__init__(category, *args)
        self.category = category


class OperationContractError(Exception):
    """Raised for invalid operation contract construction.

    First positional argument is a machine-readable category string.
    """

    def __init__(self, category: str, *args: object) -> None:
        super().__init__(category, *args)
        self.category = category


def _validate_json_serializable(params: dict, label: str) -> None:
    try:
        json.dumps(params)
    except (TypeError, ValueError) as exc:
        raise ReflectionError(
            "invalid_type_spec",
            f"{label}: params not JSON-serializable: {exc}",
        ) from exc


@dataclass(frozen=True)
class TypeSpec:
    """Descriptor for a TinyChain value type with optional construction params."""

    class_uri: str
    params: dict[str, object]

    def __post_init__(self) -> None:
        if not self.class_uri:
            raise ReflectionError("invalid_type_spec", "class_uri must be non-empty")
        object.__setattr__(self, "params", dict(self.params))
        _validate_json_serializable(self.params, "TypeSpec")

    def to_dict(self) -> dict:
        return {"class_uri": self.class_uri, "params": dict(sorted(self.params.items()))}

    @classmethod
    def from_dict(cls, data: dict) -> TypeSpec:
        try:
            return cls(class_uri=data["class_uri"], params=data.get("params", {}))
        except KeyError as exc:
            raise ReflectionError("invalid_type_spec", f"missing required key: {exc}") from exc


@dataclass(frozen=True)
class TypedValueRef:
    """Reference to a named value in a computation graph with its type annotation."""

    namespace: str | None
    value: str
    output: str | None
    value_type: TypeSpec

    def __post_init__(self) -> None:
        if not self.value:
            raise ReflectionError("invalid_typed_value_ref", "value must be non-empty")

    def to_dict(self) -> dict:
        return {
            "namespace": self.namespace,
            "output": self.output,
            "value": self.value,
            "value_type": self.value_type.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TypedValueRef:
        try:
            return cls(
                namespace=data.get("namespace"),
                value=data["value"],
                output=data.get("output"),
                value_type=TypeSpec.from_dict(data["value_type"]),
            )
        except KeyError as exc:
            raise ReflectionError("invalid_typed_value_ref", f"missing required key: {exc}") from exc


@dataclass(frozen=True)
class OperationContract:
    """Contract describing an operation's method URI, parameter schema, and output type rule."""

    method_uri: str
    params_schema: dict[str, object]
    # Equality/hashing on this field is identity-based for callables (two
    # behaviorally-identical lambdas compare unequal); nothing depends on
    # contract equality/hashing today.
    output_type_rule: Callable[[Sequence[TypeSpec], Mapping[str, object]], list[TypeSpec]] | None = None

    def __post_init__(self) -> None:
        if not self.method_uri:
            raise OperationContractError("invalid_method_uri", "method_uri must be non-empty")
        object.__setattr__(self, "params_schema", dict(self.params_schema))

    def infer_outputs(
        self,
        inputs: Sequence[TypeSpec],
        params: Mapping[str, object],
    ) -> list[TypeSpec]:
        if self.output_type_rule is None:
            raise OperationContractError(
                "missing_output_type_rule",
                f"{self.method_uri!r} has no output_type_rule for inference",
            )
        return self.output_type_rule(inputs, params)
