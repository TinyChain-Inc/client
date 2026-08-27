"""Generated tensor-valued constants for artifact expansion passes.

An expansion pass that rewrites a reduction into a matmul needs to introduce an
operand no traced graph ever produced -- a dense tensor whose every element is
the same number. This module owns the single public spelling of that operand and
the reader every backend uses to interpret one. Later work in this feature adds
the passes themselves to this module; the constant contract is what they all
emit and what every consumer of an expanded artifact reads.

Why a concrete operator type
----------------------------
Handler dispatch in :mod:`tinychain.autodiff.lowering` keys on the concrete
:class:`~tinychain.autodiff.graph.TensorOperator` type and on nothing else. That
makes a new operator type the only construct for which the framework can
*guarantee* that a backend which does not understand the constant fails before
returning a target value: the pre-flight supported-operation check rejects the
artifact naming the offending node, before any handler runs.

The alternative -- a one-operand ``MatmulOperator`` carrying the constant in
``op_params`` -- reaches an unaware two-operand handler, which reads a second
operand that is not there and reports the wrong cause, or, if it is written
defensively, silently succeeds on garbage. ``MatmulOperator`` therefore keeps its
two-operand form untouched and no existing handler changes behavior.

The descriptor schema
---------------------
A fill node is exactly this and nothing more::

    FillOperator            route_name "fill"
      input_value_ids       ()                    -- zero operands
      op_params             {"fill": float, "dtype": str, "shape": [int, ...]}
      output_typespec       {"dtype": <same>, "shape": <same>}   -- always complete

``op_params`` carries the operator's own parameters and no provenance, no pass
name, and no consumer configuration. The ``output_typespec`` is never partial: a
generated constant has no untyped operand to inherit an unknown from, so there is
no case in which its dtype or shape can honestly be missing.

The shared reader
-----------------
:func:`fill_descriptor` validates one fill node -- as a
:class:`~tinychain.autodiff.graph.TensorNodeRecord` or as the
:class:`~tinychain.autodiff.lowering.OperationContext` a handler is handed -- and
returns the three values as a typed record, so a backend's fill handler is a
two-line materialization rather than a hand-written parser. Using it is not
required; a handler that parses ``op_params`` itself observes the same schema
documented above.

Failures
--------
Every rejection is a categorized
:class:`~tinychain.autodiff.protocol.AutodiffError` drawn from the existing
``AUTODIFF_ERROR_CATEGORIES`` -- this module adds none -- and every message names
the offending node id:

* ``malformed_derivative_ir`` -- the operator is not a :class:`FillOperator`, the
  node carries an operand, ``op_params`` holds a key outside the schema, or
  ``fill`` is absent or not a real number. A ``bool`` is not a real number here:
  ``True`` reaching a numeric field is a construction defect, not the value one.
* ``missing_shape_metadata`` -- ``shape`` is absent, is not a sequence of
  integers, or holds a negative or symbolic dimension.
* ``missing_dtype_metadata`` -- ``dtype`` is absent or is not a non-empty string.

Unknown keys are checked before the individual fields, and ``fill``, ``shape``,
and ``dtype`` are then checked in that order, so an absent ``shape`` reports
``missing_shape_metadata`` rather than being swallowed by the key-set check.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .graph import TensorNodeRecord, TensorOperator
from .lowering import OperationContext
from .protocol import AutodiffError

# The route name a fill node reports. It is not a dispatch key -- handlers are
# selected by concrete operator type -- and exists for serialization parity with
# every other operator.
FILL_ROUTE_NAME = "fill"

# The complete `op_params` key set of a fill node (§8.2). Exact, not minimal: a
# key outside this set is a structural defect, because a pass that wrote one
# would be smuggling configuration through an operator's parameters.
_FILL_PARAM_KEYS = frozenset({"fill", "dtype", "shape"})


@dataclass(frozen=True)
class FillOperator(TensorOperator):
    """A dense tensor of one repeated value, produced from no operand."""

    def __init__(self) -> None:
        object.__setattr__(self, "route_name", FILL_ROUTE_NAME)


@dataclass(frozen=True)
class FillDescriptor:
    """The validated contents of one fill node's ``op_params``."""

    fill: float
    dtype: str
    shape: tuple[int, ...]


def fill_descriptor(operation: TensorNodeRecord | OperationContext) -> FillDescriptor:
    """Return the validated descriptor of *operation*, a fill node.

    Accepts either a recorded :class:`TensorNodeRecord` or the
    :class:`OperationContext` a lowering handler receives for the same node.
    Raises a categorized :class:`AutodiffError` naming the node for any node that
    is not a well-formed fill.
    """
    if not isinstance(operation, (TensorNodeRecord, OperationContext)):
        raise AutodiffError(
            "malformed_derivative_ir",
            "fill_descriptor expects a TensorNodeRecord or an OperationContext, "
            f"got {type(operation).__name__}",
        )

    node_id = operation.node_id
    _check_operator(node_id, operation.operator)
    _check_no_operands(node_id, operation.input_value_ids)

    op_params = operation.op_params
    _check_param_keys(node_id, op_params)

    return FillDescriptor(
        fill=_read_fill(node_id, op_params),
        dtype=_read_dtype(node_id, op_params),
        shape=_read_shape(node_id, op_params),
    )


def _check_operator(node_id: str, operator: TensorOperator) -> None:
    """Reject a node whose concrete operator type is not :class:`FillOperator`."""
    if not isinstance(operator, FillOperator):
        raise AutodiffError(
            "malformed_derivative_ir",
            f"node {node_id!r} is not a fill node: expected a FillOperator, "
            f"got {type(operator).__name__}",
        )


def _check_no_operands(node_id: str, input_value_ids: Sequence[str]) -> None:
    """Reject a fill node that consumes a value; a generated constant has no input."""
    if len(input_value_ids) != 0:
        raise AutodiffError(
            "malformed_derivative_ir",
            f"fill node {node_id!r} must declare zero operands, got "
            f"{list(input_value_ids)!r}",
        )


def _check_param_keys(node_id: str, op_params: Mapping[str, object]) -> None:
    """Reject any ``op_params`` key outside the fill schema (§8.2)."""
    unknown_keys = sorted(str(key) for key in op_params if key not in _FILL_PARAM_KEYS)
    if unknown_keys:
        raise AutodiffError(
            "malformed_derivative_ir",
            f"fill node {node_id!r} declares op_params outside the fill schema: "
            f"{unknown_keys!r}; expected exactly {sorted(_FILL_PARAM_KEYS)!r}",
        )


def _read_fill(node_id: str, op_params: Mapping[str, object]) -> float:
    """Read ``fill`` as a real number, rejecting a boolean and any non-number."""
    if "fill" not in op_params:
        raise AutodiffError(
            "malformed_derivative_ir",
            f"fill node {node_id!r} declares no 'fill' value",
        )
    fill_value = op_params["fill"]
    if isinstance(fill_value, bool) or not isinstance(fill_value, (int, float)):
        raise AutodiffError(
            "malformed_derivative_ir",
            f"fill node {node_id!r} declares a 'fill' that is not a real number: "
            f"{fill_value!r}",
        )
    return float(fill_value)


def _read_dtype(node_id: str, op_params: Mapping[str, object]) -> str:
    """Read ``dtype`` as a non-empty string."""
    dtype_value = op_params.get("dtype")
    if not isinstance(dtype_value, str) or not dtype_value:
        raise AutodiffError(
            "missing_dtype_metadata",
            f"fill node {node_id!r} declares no valid 'dtype': {dtype_value!r}",
        )
    return dtype_value


def _read_shape(node_id: str, op_params: Mapping[str, object]) -> tuple[int, ...]:
    """Read ``shape`` as a ranked tuple of non-negative concrete dimensions."""
    shape_value = op_params.get("shape")
    if isinstance(shape_value, (str, bytes)) or not isinstance(shape_value, Sequence):
        raise AutodiffError(
            "missing_shape_metadata",
            f"fill node {node_id!r} declares no valid 'shape': {shape_value!r}",
        )
    dimensions: list[int] = []
    for dimension in shape_value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise AutodiffError(
                "missing_shape_metadata",
                f"fill node {node_id!r} declares a non-integer dimension in 'shape': "
                f"{dimension!r}",
            )
        if dimension < 0:
            raise AutodiffError(
                "missing_shape_metadata",
                f"fill node {node_id!r} declares a negative dimension in 'shape': "
                f"{dimension!r}",
            )
        dimensions.append(dimension)
    return tuple(dimensions)
