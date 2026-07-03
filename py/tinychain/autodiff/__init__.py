from __future__ import annotations

from .accumulate import GradientAccumulator
from .executor import ExecutionScheduler
from .graph import (
    AddOperator,
    BroadcastReduceOperator,
    MatmulOperator,
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
from .vjp import AddVjpRule, BroadcastReductionPlanner, MatmulVjpRule, TransposeVjpRule, VjpRegistry


def generate(
    graph: TensorGraph,
    output_value_id: str,
    wrt: list[str],
    seed: str,
    *,
    seed_typespec: dict[str, object] | None = None,
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
    return ReverseTraversal().build(
        graph=graph,
        output_value_id=output_value_id,
        wrt=wrt,
        seed_value_id=seed,
        seed_typespec=seed_typespec,
        source_graph_id=graph_id,
    )


__all__ = [
    "AUTODIFF_ERROR_CATEGORIES",
    "AddVjpRule",
    "AutodiffError",
    "AutodiffRequest",
    "AutodiffResult",
    "AddOperator",
    "BroadcastReduceOperator",
    "BroadcastReductionPlanner",
    "DerivativeMetadata",
    "DerivativeProgram",
    "ExecutionScheduler",
    "GradientAccumulator",
    "MatmulOperator",
    "MatmulVjpRule",
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
    "get_active_builder",
    "reflect_derivative_program",
    "tensor_typespec_to_type_spec",
]
