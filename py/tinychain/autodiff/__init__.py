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
from .vjp import AddVjpRule, BroadcastReductionPlanner, VjpRegistry


def generate(
    graph: TensorGraph,
    output_value_id: str,
    wrt: list[str],
    seed: str,
    *,
    seed_typespec: dict[str, object] | None = None,
) -> DerivativeProgram:
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
