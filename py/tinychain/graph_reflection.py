"""Python-owned generic reflection primitives. See .ai/specs/client-issue-29-opdef-reflection-primitives.md and .ai/adr/ADR-001."""
from __future__ import annotations

import json
from dataclasses import dataclass


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
    output_type_rule: str | None = None

    def __post_init__(self) -> None:
        if not self.method_uri:
            raise OperationContractError("invalid_method_uri", "method_uri must be non-empty")
        object.__setattr__(self, "params_schema", dict(self.params_schema))
