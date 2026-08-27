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

The forward mean expansion
--------------------------
:func:`expand_mean_graph` rewrites every *supported* all-axis rank-2
``MeanOperator`` in a :class:`~tinychain.autodiff.graph.TensorGraph` into a
region built from a generated constant, two matmuls, and one scale -- so a
backend needs no reduction operation at all. For an operand of shape ``(r, c)``
and dtype ``D``::

    f1  FillOperator   --              -> D, [c, 1]     ones
    f2  MatmulOperator [operand, f1]   -> D, [r, 1]     row sums
    f3  FillOperator   --              -> D, [1, r]     ones
    f4  MatmulOperator [f3, f2]        -> D, [1, 1]     total sum
    f5  MulOperator    [f4]            -> D, [1, 1]     mean, rank 2

Two tiers follow from the mean's own ``keepdims``, with no parameter. For
``keepdims=True`` the mean is already ``[1, 1]``, so ``f5`` carries the mean's
value id and typespec and the region is five nodes. For ``keepdims=False`` the
mean is rank zero, so a sixth node -- a real ``ReshapeOperator`` with
``shape = []`` -- performs the genuine rank change and carries the mean's value
id. A rank change is never expressed by an elementwise node declaring a
different shape: a scalar and a ``[1, 1]`` value are not interchangeable, and
every emitted node's ``output_typespec`` is computed from its operands rather
than copied from the node it replaces.

An unsupported mean stops the pass rather than being left in place -- a mean is
unambiguously inside the declared domain of a pass named for mean expansion.
Every candidate is validated before any node is emitted, so a rejected graph
never comes back partially rewritten, and every failure names the offending node
and the clause of FR-128-006 it failed. Operators other than ``MeanOperator``
are never rewritten and are carried through identical.

The reserved identifier namespace
---------------------------------
Nodes and values a pass creates are named ``exn0, exn1, …`` and
``exv0, exv1, …`` -- see :data:`EXPANSION_NODE_ID_PREFIX` and
:data:`EXPANSION_VALUE_ID_PREFIX`. The namespace is reserved for expansion and
is disjoint from the tracer's ``v…``/``n…`` and the reverse transform's
``d…``/``dn…``, so a minted identifier can never be mistaken for a traced or
generated one. Minting is deterministic -- indices run from zero on every call,
so equal graphs expand to equal graphs including identifiers and order -- and
every minted identifier is checked against those already in the artifact,
raising ``malformed_derivative_ir`` naming the identifier rather than silently
shadowing an existing value. An artifact that already contains a duplicate node
or value id is rejected the same way, before anything is minted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .graph import (
    MatmulOperator,
    MeanOperator,
    MulOperator,
    ReshapeOperator,
    TensorGraph,
    TensorNodeRecord,
    TensorOperator,
)
from .lowering import OperationContext
from .protocol import AutodiffError
from .shape import (
    Shape,
    _normalize_mean_axes,
    check_compatible_operand_dtypes,
    check_differentiable_dtype,
    matmul_output_shape,
    mean_output_shape,
    shape_rank,
    typespec_ranked_shape,
)

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


# --------------------------------------------------------------------------
# The reserved identifier namespace (§9.2)
# --------------------------------------------------------------------------

# Every node and value an expansion pass creates is named from this reserved
# namespace. It is disjoint from the tracer's `v…`/`n…` and from the reverse
# transform's `d…`/`dn…`, so a minted identifier can never be confused with a
# traced or generated one; minting is deterministic (indices run from zero on
# every call, so equal artifacts expand to equal artifacts); and every minted
# identifier is checked against the identifiers already in the artifact, which
# fails closed rather than silently shadowing an existing value.
EXPANSION_NODE_ID_PREFIX = "exn"
EXPANSION_VALUE_ID_PREFIX = "exv"


class _IdentifierMinter:
    """Deterministic source of reserved node and value identifiers.

    Constructed once per pass invocation over the identifiers the input
    artifact already uses. Both mint methods take no argument, so every caller
    -- this pass and the gradient-path pass that shares this module -- mints
    from one place and cannot disagree about the namespace or the ordering.
    """

    def __init__(
        self, *, existing_node_ids: frozenset[str], existing_value_ids: frozenset[str]
    ) -> None:
        self._existing_node_ids = existing_node_ids
        self._existing_value_ids = existing_value_ids
        self._node_index = 0
        self._value_index = 0

    def mint_node_id(self) -> str:
        """Return the next reserved node id, rejecting a collision."""
        node_id = f"{EXPANSION_NODE_ID_PREFIX}{self._node_index}"
        self._node_index += 1
        if node_id in self._existing_node_ids:
            raise AutodiffError(
                "malformed_derivative_ir",
                f"minted node id {node_id!r} collides with a node id already present in "
                f"the artifact; {EXPANSION_NODE_ID_PREFIX!r} is a reserved expansion namespace",
            )
        return node_id

    def mint_value_id(self) -> str:
        """Return the next reserved value id, rejecting a collision."""
        value_id = f"{EXPANSION_VALUE_ID_PREFIX}{self._value_index}"
        self._value_index += 1
        if value_id in self._existing_value_ids:
            raise AutodiffError(
                "malformed_derivative_ir",
                f"minted value id {value_id!r} collides with a value id already present in "
                f"the artifact; {EXPANSION_VALUE_ID_PREFIX!r} is a reserved expansion namespace",
            )
        return value_id


def _is_reserved_identifier(identifier: object) -> bool:
    """Report whether *identifier* is spelled inside the reserved namespace."""
    return isinstance(identifier, str) and identifier.startswith(
        (EXPANSION_NODE_ID_PREFIX, EXPANSION_VALUE_ID_PREFIX)
    )


# --------------------------------------------------------------------------
# Artifact identifier indexing
# --------------------------------------------------------------------------


def _indexed_node_ids(nodes: Sequence[TensorNodeRecord]) -> frozenset[str]:
    """Return every node id in *nodes*, rejecting a duplicate."""
    seen: set[str] = set()
    for node in nodes:
        if node.node_id in seen:
            raise AutodiffError(
                "malformed_derivative_ir",
                f"artifact declares duplicate node id {node.node_id!r}",
            )
        seen.add(node.node_id)
    return frozenset(seen)


def _indexed_value_ids(
    nodes: Sequence[TensorNodeRecord], inputs: Sequence[tuple[str, object]]
) -> frozenset[str]:
    """Return every declared value id, rejecting a duplicate producer."""
    seen: set[str] = set()
    for value_id, _ in inputs:
        if value_id in seen:
            raise AutodiffError(
                "malformed_derivative_ir",
                f"artifact declares duplicate value id {value_id!r}",
            )
        seen.add(value_id)
    for node in nodes:
        if node.output_value_id in seen:
            raise AutodiffError(
                "malformed_derivative_ir",
                f"artifact declares duplicate value id {node.output_value_id!r}",
            )
        seen.add(node.output_value_id)
    return frozenset(seen)


def _declared_typespecs(
    nodes: Sequence[TensorNodeRecord], inputs: Sequence[tuple[str, object]]
) -> dict[str, object]:
    """Index every value id to the typespec its producer declares for it."""
    typespecs: dict[str, object] = {value_id: typespec for value_id, typespec in inputs}
    for node in nodes:
        typespecs[node.output_value_id] = node.output_typespec
    return typespecs


# --------------------------------------------------------------------------
# The supported-mean predicate (FR-128-006)
# --------------------------------------------------------------------------

# The rank the forward expansion is defined for. A matmul pair reduces exactly
# two axes, so a mean over any other rank is out of the pass's declared domain
# rather than a metadata defect.
_SUPPORTED_MEAN_RANK = 2


@dataclass(frozen=True)
class _SupportedMean:
    """One `MeanOperator` node proven expandable, with everything the emitter needs."""

    node_id: str
    operand_value_id: str
    operand_typespec: dict[str, object]
    dtype: str
    rows: int
    columns: int
    keepdims: bool
    output_value_id: str
    output_typespec: dict[str, object]


def _mean_failure(node_id: str, clause: int, category: str, detail: str) -> AutodiffError:
    """Build the categorized failure for one clause of FR-128-006.

    Every message names both the offending node and the clause that failed, so
    a caller reading only the message can find the node and the rule.
    """
    return AutodiffError(
        category,
        f"mean node {node_id!r} fails clause {clause} of FR-128-006: {detail}",
    )


def _recategorized(
    node_id: str, clause: int, detail: str, exc: AutodiffError
) -> AutodiffError:
    """Re-raise a helper's own categorized failure with node and clause context."""
    return _mean_failure(node_id, clause, exc.category, f"{detail}: {exc.message}")


def _mean_operand_shape_and_dtype(
    node: TensorNodeRecord, typespecs: Mapping[str, object]
) -> tuple[Shape, str]:
    """Clause 2 -- the operand carries a complete typespec of rank two."""
    operand_typespec = typespecs.get(node.input_value_ids[0])
    try:
        shape = typespec_ranked_shape(operand_typespec)
    except AutodiffError as exc:
        raise _recategorized(
            node.node_id, 2, "operand declares no ranked shape", exc
        ) from exc

    dtype = None if operand_typespec is None else operand_typespec.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise _mean_failure(
            node.node_id,
            2,
            "missing_dtype_metadata",
            f"operand declares no dtype: {dtype!r}",
        )

    if shape_rank(shape) != _SUPPORTED_MEAN_RANK:
        raise _mean_failure(
            node.node_id,
            2,
            "unsupported_reduction",
            f"operand shape {list(shape)!r} has rank {shape_rank(shape)}, and this pass "
            f"expands a mean over a rank-{_SUPPORTED_MEAN_RANK} operand only",
        )
    return shape, dtype


def _mean_operand_dimensions(node_id: str, shape: Shape) -> tuple[int, int]:
    """Clause 3 -- both operand dimensions are positive integers."""
    for axis, dimension in enumerate(shape):
        if isinstance(dimension, str):
            raise _mean_failure(
                node_id,
                3,
                "unresolved_symbolic_shape",
                f"reduced dimension {dimension!r} at axis {axis} is symbolic, not an integer",
            )
        if dimension <= 0:
            raise _mean_failure(
                node_id,
                3,
                "unsupported_reduction",
                f"reduced dimension {dimension!r} at axis {axis} is not positive",
            )
    rows, columns = shape
    return int(rows), int(columns)


def _mean_axes(node: TensorNodeRecord) -> tuple[int, ...]:
    """Clause 4 -- the declared axes normalize to every axis of the operand."""
    axes = node.op_params.get("axes")
    try:
        normalized = _normalize_mean_axes(axes, _SUPPORTED_MEAN_RANK)
    except AutodiffError as exc:
        raise _recategorized(node.node_id, 4, "declared axes are malformed", exc) from exc
    if set(normalized) != set(range(_SUPPORTED_MEAN_RANK)):
        raise _mean_failure(
            node.node_id,
            4,
            "unsupported_reduction",
            f"declared axes {list(normalized)!r} are a partial reduction, and this pass "
            "expands an all-axis mean only",
        )
    return normalized


def _mean_keepdims(node: TensorNodeRecord) -> bool:
    """Clause 5 -- ``keepdims`` is a real boolean; both values select a tier."""
    keepdims = node.op_params.get("keepdims")
    if not isinstance(keepdims, bool):
        raise _mean_failure(
            node.node_id,
            5,
            "unsupported_reduction",
            f"declared 'keepdims' is not a bool: {keepdims!r}",
        )
    return keepdims


def _mean_dtype(node_id: str, dtype: str) -> str:
    """Clause 6 -- the operand dtype is differentiable."""
    try:
        return check_differentiable_dtype(dtype)
    except AutodiffError as exc:
        raise _recategorized(node_id, 6, "operand dtype is not differentiable", exc) from exc


def _mean_output_typespec(
    node: TensorNodeRecord,
    operand_shape: Shape,
    operand_dtype: str,
    axes: tuple[int, ...],
    keepdims: bool,
) -> dict[str, object]:
    """Clause 7 -- the declared output typespec is complete and is the true one."""
    declared = node.output_typespec
    try:
        declared_shape = typespec_ranked_shape(declared)
    except AutodiffError as exc:
        raise _recategorized(
            node.node_id, 7, "output declares no ranked shape", exc
        ) from exc

    declared_dtype = None if declared is None else declared.get("dtype")
    if not isinstance(declared_dtype, str) or not declared_dtype:
        raise _mean_failure(
            node.node_id,
            7,
            "missing_dtype_metadata",
            f"output declares no dtype: {declared_dtype!r}",
        )

    expected_shape = mean_output_shape(operand_shape, list(axes), keepdims=keepdims)
    if tuple(declared_shape) != tuple(expected_shape):
        raise _mean_failure(
            node.node_id,
            7,
            "reduction_shape_mismatch",
            f"output declares shape {list(declared_shape)!r}, but reducing "
            f"{list(operand_shape)!r} over {list(axes)!r} with keepdims={keepdims} "
            f"produces {list(expected_shape)!r}",
        )
    if declared_dtype != operand_dtype:
        raise _mean_failure(
            node.node_id,
            7,
            "reduction_shape_mismatch",
            f"output declares dtype {declared_dtype!r}, but the operand dtype is "
            f"{operand_dtype!r} and a mean performs no promotion",
        )
    return {"dtype": declared_dtype, "shape": list(declared_shape)}


def _check_mean_carries_no_reserved_construct(node: TensorNodeRecord) -> None:
    """Clause 8 -- the node carries no reserved identifier and no fill descriptor key."""
    for label, identifier in (
        ("node id", node.node_id),
        ("output value id", node.output_value_id),
    ):
        if _is_reserved_identifier(identifier):
            raise _mean_failure(
                node.node_id,
                8,
                "unsupported_reduction",
                f"{label} {identifier!r} is spelled inside the reserved expansion namespace "
                f"({EXPANSION_NODE_ID_PREFIX!r}/{EXPANSION_VALUE_ID_PREFIX!r})",
            )
    descriptor_keys = sorted(str(key) for key in node.op_params if key in _FILL_PARAM_KEYS)
    if descriptor_keys:
        raise _mean_failure(
            node.node_id,
            8,
            "unsupported_reduction",
            f"op_params carries fill descriptor key(s) {descriptor_keys!r}",
        )


def _supported_mean(
    node: TensorNodeRecord, typespecs: Mapping[str, object]
) -> _SupportedMean:
    """Prove one `MeanOperator` node expandable, or raise its categorized failure.

    The eight clauses of FR-128-006 are checked in their stated order, so the
    reported clause is always the first one the node fails.
    """
    if len(node.input_value_ids) != 1:
        raise _mean_failure(
            node.node_id,
            1,
            "unsupported_reduction",
            f"a mean takes exactly one operand, got {list(node.input_value_ids)!r}",
        )

    operand_shape, operand_dtype = _mean_operand_shape_and_dtype(node, typespecs)
    rows, columns = _mean_operand_dimensions(node.node_id, operand_shape)
    axes = _mean_axes(node)
    keepdims = _mean_keepdims(node)
    dtype = _mean_dtype(node.node_id, operand_dtype)
    output_typespec = _mean_output_typespec(
        node, operand_shape, operand_dtype, axes, keepdims
    )
    _check_mean_carries_no_reserved_construct(node)

    return _SupportedMean(
        node_id=node.node_id,
        operand_value_id=node.input_value_ids[0],
        operand_typespec={"dtype": dtype, "shape": [rows, columns]},
        dtype=dtype,
        rows=rows,
        columns=columns,
        keepdims=keepdims,
        output_value_id=node.output_value_id,
        output_typespec=output_typespec,
    )


# --------------------------------------------------------------------------
# The §8.3 region emitter
# --------------------------------------------------------------------------

# The value every generated ones-tensor holds. A row of ones is what turns a
# matmul into a sum along one axis, which is the whole mechanism of the region.
_ONES_FILL = 1.0


def _boundary_typespec(dtype: str, shape: Sequence[object]) -> dict[str, object]:
    """Build the graph-boundary typespec form, `{"dtype": str, "shape": list}`."""
    return {"dtype": dtype, "shape": list(shape)}


def _emit_fill(
    minter: _IdentifierMinter, *, dtype: str, shape: Sequence[int]
) -> TensorNodeRecord:
    """Emit one ones-tensor of *shape*; its true result is its own descriptor."""
    descriptor_shape = [int(dimension) for dimension in shape]
    return TensorNodeRecord(
        node_id=minter.mint_node_id(),
        output_value_id=minter.mint_value_id(),
        operator=FillOperator(),
        op_params={"fill": _ONES_FILL, "dtype": dtype, "shape": descriptor_shape},
        input_value_ids=[],
        output_typespec=_boundary_typespec(dtype, descriptor_shape),
    )


def _emit_matmul(
    minter: _IdentifierMinter,
    *,
    left: TensorNodeRecord | tuple[str, Mapping[str, object]],
    right: TensorNodeRecord | tuple[str, Mapping[str, object]],
) -> TensorNodeRecord:
    """Emit one two-operand matmul, deriving its declared type from its operands."""
    left_value_id, left_typespec = _operand(left)
    right_value_id, right_typespec = _operand(right)
    dtype = check_compatible_operand_dtypes(
        str(left_typespec["dtype"]), str(right_typespec["dtype"])
    )
    shape = matmul_output_shape(
        tuple(left_typespec["shape"]),
        tuple(right_typespec["shape"]),
    )
    return TensorNodeRecord(
        node_id=minter.mint_node_id(),
        output_value_id=minter.mint_value_id(),
        operator=MatmulOperator(),
        op_params={},
        input_value_ids=[left_value_id, right_value_id],
        output_typespec=_boundary_typespec(dtype, shape),
    )


def _emit_scale(
    minter: _IdentifierMinter,
    *,
    operand: TensorNodeRecord,
    right_literal: float,
    output_value_id: str | None = None,
) -> TensorNodeRecord:
    """Emit one scale-by-literal; an elementwise scale keeps the operand's type."""
    operand_value_id, operand_typespec = _operand(operand)
    return TensorNodeRecord(
        node_id=minter.mint_node_id(),
        output_value_id=(
            minter.mint_value_id() if output_value_id is None else output_value_id
        ),
        operator=MulOperator(),
        op_params={"right_literal": float(right_literal)},
        input_value_ids=[operand_value_id],
        output_typespec=_boundary_typespec(
            str(operand_typespec["dtype"]), list(operand_typespec["shape"])
        ),
    )


def _emit_reshape(
    minter: _IdentifierMinter,
    *,
    operand: TensorNodeRecord,
    shape: Sequence[int],
    output_value_id: str,
) -> TensorNodeRecord:
    """Emit one real reshape; a reshape keeps the dtype and takes the target shape."""
    operand_value_id, operand_typespec = _operand(operand)
    target_shape = [int(dimension) for dimension in shape]
    return TensorNodeRecord(
        node_id=minter.mint_node_id(),
        output_value_id=output_value_id,
        operator=ReshapeOperator(),
        op_params={"shape": target_shape},
        input_value_ids=[operand_value_id],
        output_typespec=_boundary_typespec(str(operand_typespec["dtype"]), target_shape),
    )


def _operand(
    source: TensorNodeRecord | tuple[str, Mapping[str, object]],
) -> tuple[str, Mapping[str, object]]:
    """Normalize an emitted node or an existing `(value id, typespec)` pair."""
    if isinstance(source, TensorNodeRecord):
        return source.output_value_id, source.output_typespec
    return source


def _emit_mean_region(
    supported: _SupportedMean, minter: _IdentifierMinter
) -> list[TensorNodeRecord]:
    """Emit the region of §8.3 replacing one proven-supported mean.

    Every declared shape below is computed from the operands the node actually
    reads -- the shape helpers derive it -- rather than copied from the mean the
    region replaces, so no node can declare a shape its operation cannot
    produce.
    """
    rows, columns = supported.rows, supported.columns

    column_ones = _emit_fill(minter, dtype=supported.dtype, shape=(columns, 1))
    row_sums = _emit_matmul(
        minter,
        left=(supported.operand_value_id, supported.operand_typespec),
        right=column_ones,
    )
    row_ones = _emit_fill(minter, dtype=supported.dtype, shape=(1, rows))
    total_sum = _emit_matmul(minter, left=row_ones, right=row_sums)

    reciprocal_count = 1.0 / float(rows * columns)
    if supported.keepdims:
        # The rank-preserving tier: the scale already has the mean's rank-2
        # shape, so it carries the mean's own value id and terminates the region.
        scale = _emit_scale(
            minter,
            operand=total_sum,
            right_literal=reciprocal_count,
            output_value_id=supported.output_value_id,
        )
        return [column_ones, row_sums, row_ones, total_sum, scale]

    # The rank-reducing tier: the scale is `[1, 1]` and the mean is rank zero,
    # so a real reshape performs the rank change and carries the mean's value id.
    scale = _emit_scale(minter, operand=total_sum, right_literal=reciprocal_count)
    rank_change = _emit_reshape(
        minter,
        operand=scale,
        shape=(),
        output_value_id=supported.output_value_id,
    )
    return [column_ones, row_sums, row_ones, total_sum, scale, rank_change]


# --------------------------------------------------------------------------
# The public forward pass
# --------------------------------------------------------------------------


def expand_mean_graph(graph: TensorGraph) -> TensorGraph:
    """Return *graph* with every supported all-axis rank-2 mean expanded.

    Each supported `MeanOperator` (FR-128-006) is replaced in place by the
    matmul-based region of §8.3 -- five nodes when the mean declared
    ``keepdims=True``, six when it declared ``keepdims=False``, the sixth a real
    `ReshapeOperator` performing the rank change. Every other node, and
    ``inputs`` and ``outputs``, are carried through unchanged; the input graph is
    never mutated, and equal graphs expand to equal graphs.

    Every candidate mean is validated before a single node is emitted, so an
    unsupported mean raises its categorized failure (§13.1) naming the node and
    the clause of FR-128-006 it failed, and never yields a partially rewritten
    graph.
    """
    nodes = graph.nodes
    existing_node_ids = _indexed_node_ids(nodes)
    existing_value_ids = _indexed_value_ids(nodes, graph.inputs)
    typespecs = _declared_typespecs(nodes, graph.inputs)

    # Validation first, over every candidate, before any emission (§13.2).
    supported_means = {
        node.node_id: _supported_mean(node, typespecs)
        for node in nodes
        if isinstance(node.operator, MeanOperator)
    }

    minter = _IdentifierMinter(
        existing_node_ids=existing_node_ids, existing_value_ids=existing_value_ids
    )
    expanded_nodes: list[TensorNodeRecord] = []
    for node in nodes:
        supported = supported_means.get(node.node_id)
        if supported is None:
            expanded_nodes.append(node)
            continue
        expanded_nodes.extend(_emit_mean_region(supported, minter))

    return TensorGraph(
        nodes=expanded_nodes,
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
    )
