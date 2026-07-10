from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..serialize import serialize


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
)


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


@dataclass(frozen=True)
class AutodiffRequest:
    graph: object
    output_value_id: str
    wrt: list[str]
    seed_value_id: str
    tensor_op_contract_version: str
    transform_version: str

    def to_dict(self) -> dict[str, object]:
        return serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AutodiffRequest:
        return cls(
            graph=data["graph"],
            output_value_id=str(data["output_value_id"]),
            wrt=[str(item) for item in data["wrt"]],
            seed_value_id=str(data["seed_value_id"]),
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
