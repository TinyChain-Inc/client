from __future__ import annotations

from importlib import import_module

from .accumulate import GradientAccumulator
from .executor import DerivativeExecutionDispatcher, ExecutionScheduler
from .graph import (
    AddOperator,
    BroadcastOperator,
    BroadcastReduceOperator,
    DivOperator,
    MatmulOperator,
    MaxOperator,
    MeanOperator,
    MinOperator,
    MulOperator,
    ProductOperator,
    ReshapeOperator,
    SubOperator,
    SumOperator,
    TensorOperator,
    TransposeOperator,
    TensorGraph,
    TensorGraphBuilder,
    TensorNodeRecord,
    get_active_builder,
)
from .protocol import (
    AUTODIFF_ERROR_CATEGORIES,
    AutodiffError,
    AutodiffRequest,
    AutodiffResult,
    DerivativeMetadata,
)
from .reflection import reflect_derivative_program, tensor_typespec_to_type_spec
from .reverse import DerivativeProgram, ReverseTraversal
from .seed import SeedValidator
from .generate import generate
from .tracing import captured_operator_types, captured_route_operators
from .vjp import (
    AddVjpRule,
    BroadcastReductionPlanner,
    DivVjpRule,
    MatmulVjpRule,
    MaxVjpRule,
    MeanVjpRule,
    MinVjpRule,
    MulVjpRule,
    ProductVjpRule,
    ReshapeVjpRule,
    SubVjpRule,
    SumVjpRule,
    TransposeVjpRule,
    VjpRegistry,
)

_ARTIFACT_EXPORTS = frozenset(
    {
        "ARTIFACT_ERROR_CATEGORIES",
        "ArtifactComparisonResult",
        "ArtifactError",
        "ArtifactPayload",
        "ArtifactPublicIdentity",
        "DerivativeArtifactManifest",
        "artifact_digest_input",
        "artifact_manifest_from_program",
        "artifact_payload",
        "artifact_source_dependencies",
        "attach_artifact_digest",
        "build_derivative_artifact_library",
        "build_derivative_execution_library",
        "canonical_artifact_json",
        "compare_artifact_identity",
        "compute_artifact_digest",
        "public_artifact_identity",
        "source_library_dependency_uri",
        "validate_artifact_source_metadata",
    }
)

_ROUTE_EXPORTS = frozenset(
    {
        "ROUTE_DERIVATIVE_COMPATIBILITY_COMPATIBLE",
        "ROUTE_DERIVATIVE_COMPATIBILITY_INCOMPATIBLE",
        "ROUTE_DERIVATIVE_COMPATIBILITY_NOT_VALIDATED",
        "ROUTE_DERIVATIVE_COMPATIBILITY_STATUSES",
        "ROUTE_DERIVATIVE_COMPATIBILITY_UNSUPPORTED",
        "ROUTE_DERIVATIVE_SOURCE_ARTIFACT",
        "ROUTE_DERIVATIVE_SOURCE_KINDS",
        "ROUTE_DERIVATIVE_SOURCE_NON_DIFFERENTIABLE",
        "ROUTE_DERIVATIVE_SOURCE_UNSUPPORTED",
        "RouteDerivativeIdentity",
        "RouteDerivativeMetadata",
        "RouteDerivativePlan",
        "discover_route_derivative",
        "extract_route_identity",
    }
)

_CALLSITE_EXPORTS = frozenset({"grad"})

_COMPILE_EXPORTS = frozenset({"CompiledDerivativeProgram", "compile_derivative_program"})

_DEPENDENCY_EXPORTS = frozenset(
    {
        "DEPENDENCY_PROVENANCE_DECLARED_INPUT",
        "DEPENDENCY_PROVENANCE_SEED_INPUT",
        "DEPENDENCY_PROVENANCE_FORWARD_CAPTURE",
        "DEPENDENCY_PROVENANCE_LOCAL_VALUE",
        "DEPENDENCY_PROVENANCE_ORDER",
        "DependencyAnalysis",
        "ValueDependency",
        "analyze_derivative_dependencies",
        "analyze_graph_dependencies",
    }
)

_EXPANSION_EXPORTS = frozenset(
    {
        "BROADCAST_SCALE_EXPANSION",
        "FillDescriptor",
        "FillOperator",
        "MEAN_EXPANSION_FORWARD",
        "MeanDerivativeExpansionResult",
        "MeanExpansionRegion",
        "MeanGraphExpansionResult",
        "expand_mean_derivative_program",
        "expand_mean_derivative_program_detailed",
        "expand_mean_graph",
        "expand_mean_graph_detailed",
        "fill_descriptor",
    }
)

_LOWERING_EXPORTS = frozenset(
    {
        "LOWERING_CLAIM_HANDLER",
        "LOWERING_CLAIM_FUSION",
        "FusionContext",
        "FusionHook",
        "FusionResult",
        "LoweredOperation",
        "LoweredProgram",
        "OperationContext",
        "OperationHandler",
        "OperationHandlerRegistry",
        "lower_derivative_program",
        "lower_graph",
    }
)

_TRAINING_EXPORTS = frozenset(
    {
        "SGD",
        "Optimizer",
        "TracedUpdate",
        "sgd_update",
        "trace_parameter_update",
    }
)


def __getattr__(name: str) -> object:
    if name in _ARTIFACT_EXPORTS:
        artifact_module = import_module(".artifact", __name__)
        value = getattr(artifact_module, name)
        globals()[name] = value
        return value
    if name in _ROUTE_EXPORTS:
        routes_module = import_module(".routes", __name__)
        value = getattr(routes_module, name)
        globals()[name] = value
        return value
    if name in _CALLSITE_EXPORTS:
        callsite_module = import_module(".callsite", __name__)
        value = getattr(callsite_module, name)
        globals()[name] = value
        return value
    if name in _COMPILE_EXPORTS:
        compile_module = import_module(".compile", __name__)
        value = getattr(compile_module, name)
        globals()[name] = value
        return value
    if name in _DEPENDENCY_EXPORTS:
        dependencies_module = import_module(".dependencies", __name__)
        value = getattr(dependencies_module, name)
        globals()[name] = value
        return value
    if name in _EXPANSION_EXPORTS:
        expansion_module = import_module(".expansion", __name__)
        value = getattr(expansion_module, name)
        globals()[name] = value
        return value
    if name in _LOWERING_EXPORTS:
        lowering_module = import_module(".lowering", __name__)
        value = getattr(lowering_module, name)
        globals()[name] = value
        return value
    if name in _TRAINING_EXPORTS:
        training_module = import_module(".training", __name__)
        value = getattr(training_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ARTIFACT_ERROR_CATEGORIES",
    "ROUTE_DERIVATIVE_COMPATIBILITY_COMPATIBLE",
    "ROUTE_DERIVATIVE_COMPATIBILITY_INCOMPATIBLE",
    "ROUTE_DERIVATIVE_COMPATIBILITY_NOT_VALIDATED",
    "ROUTE_DERIVATIVE_COMPATIBILITY_STATUSES",
    "ROUTE_DERIVATIVE_COMPATIBILITY_UNSUPPORTED",
    "ROUTE_DERIVATIVE_SOURCE_ARTIFACT",
    "ROUTE_DERIVATIVE_SOURCE_KINDS",
    "ROUTE_DERIVATIVE_SOURCE_NON_DIFFERENTIABLE",
    "ROUTE_DERIVATIVE_SOURCE_UNSUPPORTED",
    "RouteDerivativeIdentity",
    "RouteDerivativeMetadata",
    "RouteDerivativePlan",
    "discover_route_derivative",
    "extract_route_identity",
    "ArtifactComparisonResult",
    "ArtifactError",
    "ArtifactPayload",
    "ArtifactPublicIdentity",
    "DerivativeArtifactManifest",
    "artifact_digest_input",
    "artifact_manifest_from_program",
    "artifact_payload",
    "artifact_source_dependencies",
    "attach_artifact_digest",
    "build_derivative_artifact_library",
    "build_derivative_execution_library",
    "canonical_artifact_json",
    "compile_derivative_program",
    "compare_artifact_identity",
    "compute_artifact_digest",
    "public_artifact_identity",
    "source_library_dependency_uri",
    "validate_artifact_source_metadata",
    "AUTODIFF_ERROR_CATEGORIES",
    "AddVjpRule",
    "AutodiffError",
    "CompiledDerivativeProgram",
    "AutodiffRequest",
    "AutodiffResult",
    "AddOperator",
    "BroadcastOperator",
    "BroadcastReduceOperator",
    "BroadcastReductionPlanner",
    "DerivativeMetadata",
    "DerivativeProgram",
    "DerivativeExecutionDispatcher",
    "SubOperator",
    "MulOperator",
    "DivOperator",
    "SumOperator",
    "MeanOperator",
    "MaxOperator",
    "MinOperator",
    "ProductOperator",
    "ReshapeOperator",
    "ExecutionScheduler",
    "GradientAccumulator",
    "MatmulOperator",
    "MatmulVjpRule",
    "SubVjpRule",
    "MulVjpRule",
    "DivVjpRule",
    "SumVjpRule",
    "MeanVjpRule",
    "MaxVjpRule",
    "MinVjpRule",
    "ProductVjpRule",
    "ReshapeVjpRule",
    "ReverseTraversal",
    "SeedValidator",
    "TensorGraph",
    "TensorOperator",
    "TransposeOperator",
    "TransposeVjpRule",
    "TensorGraphBuilder",
    "TensorNodeRecord",
    "VjpRegistry",
    "captured_operator_types",
    "captured_route_operators",
    "generate",
    "grad",
    "get_active_builder",
    "reflect_derivative_program",
    "tensor_typespec_to_type_spec",
    "DEPENDENCY_PROVENANCE_DECLARED_INPUT",
    "DEPENDENCY_PROVENANCE_SEED_INPUT",
    "DEPENDENCY_PROVENANCE_FORWARD_CAPTURE",
    "DEPENDENCY_PROVENANCE_LOCAL_VALUE",
    "DEPENDENCY_PROVENANCE_ORDER",
    "DependencyAnalysis",
    "ValueDependency",
    "analyze_derivative_dependencies",
    "analyze_graph_dependencies",
    "FillDescriptor",
    "FillOperator",
    "BROADCAST_SCALE_EXPANSION",
    "MEAN_EXPANSION_FORWARD",
    "MeanDerivativeExpansionResult",
    "MeanExpansionRegion",
    "MeanGraphExpansionResult",
    "expand_mean_derivative_program",
    "expand_mean_derivative_program_detailed",
    "expand_mean_graph",
    "expand_mean_graph_detailed",
    "fill_descriptor",
    "LOWERING_CLAIM_HANDLER",
    "LOWERING_CLAIM_FUSION",
    "FusionContext",
    "FusionHook",
    "FusionResult",
    "LoweredOperation",
    "LoweredProgram",
    "OperationContext",
    "OperationHandler",
    "OperationHandlerRegistry",
    "lower_derivative_program",
    "lower_graph",
    "Optimizer",
    "SGD",
    "TracedUpdate",
    "sgd_update",
    "trace_parameter_update",
]
