"""Central active-builder recorder for public typed Tensor tracing.

This internal module is the single choke point through which ordinary ``Tensor``
operations record autodiff graph nodes while a :class:`TensorGraphBuilder` is the
active trace context (client ADR-004; spec §8.1;
https://github.com/TinyChain-Inc/client/issues/95).

Centralizing capture here removes the duplicated per-operation hooks that
previously lived in ``collection/tensor/core.py`` and drifted out of sync with
the VJP rule set (spec §2.2, §9.1). ``Tensor`` remains responsible for building
the ordinary symbolic/eager result and then calls :func:`record_operation`; the
recorder owns the route→concrete-operator allowlist, value-id resolution, forward
dtype/shape inference (via the pure ``shape`` helpers), and single-record
construction. It is the sole extension point for future supported operations.

The capture allowlist is explicit and reviewable (spec §9.1, FR-013):
Add, Sub, Mul, Matmul, Mean, and Transpose are captured; Div, Sum, and Reshape
are deliberately NOT captured in this issue even though they have VJP rules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Optional

from .finalize import finalize_typed_graph
from .graph import (
    AddOperator,
    MatmulOperator,
    MeanOperator,
    MulOperator,
    SubOperator,
    TensorNodeRecord,
    TensorOperator,
    TransposeOperator,
    get_active_builder,
)
from .protocol import AutodiffError
from .shape import (
    _normalize_mean_axes,
    check_compatible_operand_dtypes,
    check_differentiable_dtype,
    elementwise_broadcast_shape,
    matmul_output_shape,
    mean_output_shape,
    transpose_output_shape,
)

# Normalized builder-side metadata for a single value, or ``None`` when the
# value has no declared/inferred metadata yet.
_ValueMetadata = Optional[dict[str, object]]

# An inference function maps (operand metadata, normalized params) to the
# record's stored ``op_params`` and its boundary ``output_typespec`` (``None``
# when inference is impossible because an operand is untyped).
_InferenceResult = tuple[dict[str, object], Optional[dict[str, object]]]
_InferenceFn = Callable[
    [list[_ValueMetadata], Mapping[str, object], dict[str, int]], _InferenceResult
]


def _boundary_typespec(dtype: str, shape: Sequence[object]) -> dict[str, object]:
    """Convert an inferred (dtype, shape) into the graph-boundary typespec form."""
    return {"dtype": dtype, "shape": list(shape)}


def _infer_elementwise(
    operand_metadata: list[_ValueMetadata],
    params: Mapping[str, object],
    symbol_bindings: dict[str, int],
) -> _InferenceResult:
    """Infer output metadata for a captured Add/Sub/Mul node (spec §10.1-§10.2)."""
    lhs_metadata, rhs_metadata = operand_metadata
    if lhs_metadata is None or rhs_metadata is None:
        return {}, None
    dtype = check_compatible_operand_dtypes(
        str(lhs_metadata["dtype"]), str(rhs_metadata["dtype"])
    )
    shape = elementwise_broadcast_shape(
        tuple(lhs_metadata["shape"]),
        tuple(rhs_metadata["shape"]),
        bindings=symbol_bindings,
    )
    return {}, _boundary_typespec(dtype, shape)


def _infer_matmul(
    operand_metadata: list[_ValueMetadata],
    params: Mapping[str, object],
    symbol_bindings: dict[str, int],
) -> _InferenceResult:
    """Infer output metadata for a captured Matmul node (spec §10.1, §10.3)."""
    lhs_metadata, rhs_metadata = operand_metadata
    if lhs_metadata is None or rhs_metadata is None:
        return {}, None
    dtype = check_compatible_operand_dtypes(
        str(lhs_metadata["dtype"]), str(rhs_metadata["dtype"])
    )
    shape = matmul_output_shape(
        tuple(lhs_metadata["shape"]),
        tuple(rhs_metadata["shape"]),
        bindings=symbol_bindings,
    )
    return {}, _boundary_typespec(dtype, shape)


def _infer_mean(
    operand_metadata: list[_ValueMetadata],
    params: Mapping[str, object],
    _symbol_bindings: dict[str, int],
) -> _InferenceResult:
    """Infer output metadata for a captured Mean node (spec §10.1, §10.4).

    Explicit axes are required for a differentiable traced Mean; ``axes=None``
    fails with ``unsupported_reduction`` at record time (spec §10.4.9, §13.2).
    """
    axes = params.get("axes")
    keepdims = bool(params.get("keepdims", False))
    if axes is None:
        raise AutodiffError(
            "unsupported_reduction",
            "typed traced mean requires explicit axes; axes=None is unsupported",
        )
    metadata = operand_metadata[0]
    if metadata is None:
        return {"axes": axes, "keepdims": keepdims}, None
    shape = tuple(metadata["shape"])
    normalized_axes = _normalize_mean_axes(axes, len(shape))
    dtype = check_differentiable_dtype(str(metadata["dtype"]))
    output_shape = mean_output_shape(shape, axes, keepdims=keepdims)
    op_params = {"axes": list(normalized_axes), "keepdims": keepdims}
    return op_params, _boundary_typespec(dtype, output_shape)


def _infer_transpose(
    operand_metadata: list[_ValueMetadata],
    params: Mapping[str, object],
    _symbol_bindings: dict[str, int],
) -> _InferenceResult:
    """Infer output metadata for a captured Transpose node (spec §10.5).

    A ``None`` permutation reverses all axes, matching the ordinary symbolic
    transpose default.
    """
    permutation = params.get("permutation")
    metadata = operand_metadata[0]
    if metadata is None:
        stored = list(permutation) if permutation is not None else []
        return {"perm": stored}, None
    shape = tuple(metadata["shape"])
    rank = len(shape)
    perm = permutation if permutation is not None else tuple(reversed(range(rank)))
    dtype = check_differentiable_dtype(str(metadata["dtype"]))
    output_shape = transpose_output_shape(shape, perm)
    return {"perm": list(perm)}, _boundary_typespec(dtype, output_shape)


# Explicit, reviewable route→concrete-operator allowlist (spec §9.1, FR-013).
# Div/Sum/Reshape are deliberately absent (deferred subset, spec §5.2).
_CAPTURED_OPERATORS: dict[str, type[TensorOperator]] = {
    "add": AddOperator,
    "sub": SubOperator,
    "mul": MulOperator,
    "matmul": MatmulOperator,
    "mean": MeanOperator,
    "transpose": TransposeOperator,
}

_INFERENCE: dict[str, _InferenceFn] = {
    "add": _infer_elementwise,
    "sub": _infer_elementwise,
    "mul": _infer_elementwise,
    "matmul": _infer_matmul,
    "mean": _infer_mean,
    "transpose": _infer_transpose,
}


def record_operation(
    operation: str,
    operands: Sequence[object],
    result: object,
    params: Optional[Mapping[str, object]] = None,
) -> None:
    """Record one captured tensor operation on the active trace, if any.

    ``Tensor`` calls this after constructing the ordinary symbolic/eager result
    (spec §8.1). Behavior:

    1. Returns immediately when no builder is active (Invariant 3) or when
       ``operation`` is not in the capture allowlist (Div/Sum/Reshape/logical
       ops are not captured; an uncaptured intermediate on a selected path is
       caught later by fail-closed finalization, spec §5.2, FR-008).
    2. Resolves/registers operand and result value IDs on the active builder,
       which also retains strong references so ``id()`` reuse cannot corrupt
       dataflow identity (Invariant 6).
    3. Reads operand metadata from the builder side table and, when every
       operand is typed, infers the output dtype/shape via the pure ``shape``
       helpers (record-time validation, spec §13.3).
    4. Stores the inferred result metadata on the builder and constructs exactly
       one concrete :class:`TensorNodeRecord` with the concrete operator,
       normalized ``op_params``, ordered input value IDs, and ``output_typespec``
       (Invariants 5-6), then appends it.
    """
    builder = get_active_builder()
    if builder is None:
        return
    operator_class = _CAPTURED_OPERATORS.get(operation)
    if operator_class is None:
        return

    normalized_params = dict(params or {})
    input_value_ids = [builder.register_value(operand) for operand in operands]
    output_value_id = builder.register_value(result)
    operand_metadata = [builder._get_value_metadata(value_id) for value_id in input_value_ids]

    symbol_bindings = builder._copy_symbol_bindings()
    op_params, output_typespec = _INFERENCE[operation](
        operand_metadata, normalized_params, symbol_bindings
    )
    builder._replace_symbol_bindings(symbol_bindings)

    if output_typespec is not None:
        builder._set_value_metadata(
            output_value_id,
            dtype=str(output_typespec["dtype"]),
            shape=tuple(output_typespec["shape"]),
        )

    builder.record(
        TensorNodeRecord(
            node_id=builder._next_node_id(),
            output_value_id=output_value_id,
            operator=operator_class(),
            op_params=op_params,
            input_value_ids=input_value_ids,
            output_typespec=output_typespec,
        )
    )


def captured_operator_types() -> frozenset[type[TensorOperator]]:
    """Return the concrete operator types the recorder captures (FR-013 parity)."""
    return frozenset(_CAPTURED_OPERATORS.values())


def captured_route_operators() -> dict[str, type[TensorOperator]]:
    """Return a copy of the route-name→concrete-operator capture allowlist."""
    return dict(_CAPTURED_OPERATORS)
