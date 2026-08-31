from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..serialize import serialize
from ._exception_state import allow_exception_state


AUTODIFF_ERROR_CATEGORIES: tuple[str, ...] = (
    "unsupported_operator",
    "missing_derivative_behavior",
    "missing_derivative_ir",
    "non_differentiable_route",
    "missing_shape_metadata",
    "missing_dtype_metadata",
    "dtype_not_differentiable",
    "shape_mismatch",
    "unresolved_symbolic_shape",
    "symbolic_shape_mismatch",
    "broadcast_shape_mismatch",
    "matmul_shape_mismatch",
    "invalid_permutation",
    "reduction_shape_mismatch",
    "unsupported_reduction",
    "seed_shape_mismatch",
    "malformed_derivative_ir",
    "side_effecting_route_unsupported",
    "autodiff_not_implemented",
    "dtype_mismatch",
    # Structured dependency analysis of a selected forward or derivative output.
    # `details` for each is carried in the message as the offending value ids.
    #   missing_dependency:       a reachable value has no producer and no provenance
    #   ambiguous_producer:       one value has two producers, or a declared seed
    #                             collides with a forward graph value
    #   invalid_selected_output:  the selection is empty or names an unknown value
    "missing_dependency",
    "ambiguous_producer",
    "invalid_selected_output",
    # Framework-owned program lowering. `details` are carried in the message as
    # the offending node ids, value ids, or operator route names.
    #   handler_contract_violation: a consumer handler, handler registration, or
    #                               fusion hook broke the lowering seam contract
    #                               (no target value emitted, an uncategorized
    #                               failure, two handlers for one operator type,
    #                               or a fusion claiming operations it was not
    #                               offered, claiming none of them, claiming one
    #                               twice, or discarding a value still needed)
    "handler_contract_violation",
    # Traced optimizer/parameter updates authored as ordinary Tensor
    # callables. `details` are carried in the message as the offending
    # callable's signature mismatch or the value the callable returned.
    #   invalid_update_signature: the update callable's signature does not
    #                             accept exactly the declared typed inputs
    #   invalid_update_output:    the update callable did not return a Tensor
    "invalid_update_signature",
    "invalid_update_output",
    # Reusable training-step compile orchestration. `details` are carried in the
    # message as the offending declaration, callable, or expansion pass.
    #   invalid_training_declaration: `inputs` or `parameters` is empty, a
    #                                 parameter name is unknown or repeated, or
    #                                 `parameter(name)` is asked for an
    #                                 undeclared name
    #   invalid_loss_signature:      the loss callable cannot be bound against
    #                                 exactly the declared input names
    #   invalid_loss_output:         the loss callable returned something other
    #                                 than a single Tensor
    #   expansion_contract_violation: an expansion pass returned the wrong
    #                                 type, raised a non-AutodiffError
    #                                 exception, or produced an artifact
    #                                 missing a required semantic value
    "invalid_training_declaration",
    "invalid_loss_signature",
    "invalid_loss_output",
    "expansion_contract_violation",
)


@allow_exception_state
@dataclass(frozen=True)
class AutodiffError(Exception):
    category: str
    message: str

    allowed_categories: ClassVar[tuple[str, ...]] = AUTODIFF_ERROR_CATEGORIES

    def __post_init__(self) -> None:
        if self.category not in self.allowed_categories:
            raise ValueError(f"unknown autodiff error category: {self.category}")
        Exception.__init__(self, f"{self.category}: {self.message}")

    def to_dict(self) -> dict[str, str]:
        return serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AutodiffError:
        return cls(category=str(data["category"]), message=str(data["message"]))


def _string_or_strings(value: object) -> str | list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class AutodiffRequest:
    graph: object
    output_value_id: str | list[str]
    wrt: list[str]
    seed_value_id: str | list[str]
    tensor_op_contract_version: str
    transform_version: str

    def to_dict(self) -> dict[str, object]:
        return serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AutodiffRequest:
        return cls(
            graph=data["graph"],
            output_value_id=_string_or_strings(data["output_value_id"]),
            wrt=[str(item) for item in data["wrt"]],
            seed_value_id=_string_or_strings(data["seed_value_id"]),
            tensor_op_contract_version=str(data["tensor_op_contract_version"]),
            transform_version=str(data["transform_version"]),
        )


@dataclass(frozen=True)
class DerivativeMetadata:
    source_graph_id: str
    transform_version: str
    tensor_op_contract_version: str
    wrt_signature: tuple[str, ...]
    seed_contract: str

    def to_dict(self) -> dict[str, object]:
        return serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> DerivativeMetadata:
        return cls(
            source_graph_id=str(data["source_graph_id"]),
            transform_version=str(data["transform_version"]),
            tensor_op_contract_version=str(data["tensor_op_contract_version"]),
            wrt_signature=tuple(str(item) for item in data["wrt_signature"]),
            seed_contract=str(data["seed_contract"]),
        )


@dataclass(frozen=True)
class AutodiffResult:
    gradients: list[object]
    metadata: DerivativeMetadata

    def to_dict(self) -> dict[str, object]:
        return serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AutodiffResult:
        return cls(
            gradients=list(data["gradients"]),
            metadata=DerivativeMetadata.from_dict(data["metadata"]),
        )
