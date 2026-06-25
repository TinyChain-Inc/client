from __future__ import annotations

from .accumulate import GradientAccumulator
from .executor import ExecutionScheduler
from .http_dispatcher import TcServerDispatcher, TensorLiteral
from .graph import (
    AddOperator,
    BroadcastReduceOperator,
    MatmulOperator,
    OP_ADD,
    OP_BROADCAST_REDUCE,
    OP_MATMUL,
    OP_TRANSPOSE,
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
from .reverse import DerivativeProgram, ReverseTraversal
from .seed import SeedValidator
from .vjp import AddVjpRule, BroadcastReductionPlanner, MatmulVjpRule, VjpRegistry


def generate(
    graph: TensorGraph,
    output_value_id: str,
    wrt: list[str],
    seed: str,
    *,
    seed_typespec: dict[str, object] | None = None,
) -> DerivativeProgram:
    """Build a reverse-mode derivative program from a recorded tensor graph.

    ``seed`` is the value id of the upstream cotangent for ``output_value_id``:
    the initial dL/d(output) tensor used to start reverse traversal. During
    execution, callers must provide a concrete value for this id in the
    scheduler environment. When ``seed_typespec`` is supplied, it is validated
    against the selected output typespec and must have the same shape and a
    differentiable floating dtype (f32/f64).
    """
    return ReverseTraversal().build(
        graph=graph,
        output_value_id=output_value_id,
        wrt=wrt,
        seed_value_id=seed,
        seed_typespec=seed_typespec,
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
    "OP_ADD",
    "OP_BROADCAST_REDUCE",
    "OP_MATMUL",
    "OP_TRANSPOSE",
    "ReverseTraversal",
    "SeedValidator",
    "TcServerDispatcher",
    "TensorLiteral",
    "TensorGraph",
    "TensorOperator",
    "TransposeOperator",
    "TensorGraphBuilder",
    "TensorNodeRecord",
    "VjpRegistry",
    "generate",
    "get_active_builder",
]
