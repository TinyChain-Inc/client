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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate(
    graph: TensorGraph,
    output_value_id: str | list[str],
    wrt: list[str],
    seed: str | list[str],
    *,
    seed_typespec: dict[str, object] | list[dict[str, object] | None] | None = None,
    graph_id: str | None = None,
) -> DerivativeProgram:
    """Experimentally build a structured Python derivative program.

    The returned ``DerivativeProgram`` contains derivative ``TensorNodeRecord``
    objects, ordered output gradient value ids, and metadata. It is a pure data
    structure for inspection or later execution by ``ExecutionScheduler``; this
    function does not execute server routes and does not return Python callbacks.

    ``seed`` is the value id of the upstream cotangent for ``output_value_id``:
    the initial dL/d(output) tensor used to start reverse traversal. During
    execution, callers must provide a concrete value for this id in the
    scheduler environment. When ``seed_typespec`` is supplied, it is validated
    against the selected output typespec and must have the same shape and a
    differentiable floating dtype (f32/f64).

    ``graph_id`` is an optional explicit identifier for the source graph. When
    supplied, it is used verbatim as ``source_graph_id`` in the returned
    metadata; otherwise a stable SHA-256 content hash of the graph structure is
    computed automatically.
    """
    output_value_ids = output_value_id if isinstance(output_value_id, list) else [output_value_id]
    seed_value_ids = seed if isinstance(seed, list) else [seed]
    seed_typespecs = seed_typespec if isinstance(seed_typespec, list) else None
    single_seed_typespec = None if isinstance(seed_typespec, list) else seed_typespec
    if len(output_value_ids) != len(seed_value_ids):
        raise TypeError("generate requires one seed value id per output value id")
    return ReverseTraversal().build(
        graph=graph,
        output_value_id=output_value_ids[0],
        output_value_ids=output_value_ids,
        wrt=wrt,
        seed_value_id=seed_value_ids[0],
        seed_value_ids=seed_value_ids,
        seed_typespec=single_seed_typespec,
        seed_typespecs=seed_typespecs,
        source_graph_id=graph_id,
    )


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
    "generate",
    "grad",
    "get_active_builder",
    "reflect_derivative_program",
    "tensor_typespec_to_type_spec",
]
