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

The gradient-path broadcast-and-scale expansion
----------------------------------------------
:func:`expand_mean_derivative_program` rewrites every region of a
:class:`~tinychain.autodiff.reverse.DerivativeProgram` that broadcasts a
``[1, 1]`` value to ``[r, c]`` and then divides it by a literal ``d`` into the
same matmul-based shape -- so a backend needs neither a broadcast nor a division
operation to run the gradient path. For a source value ``g`` of dtype ``D`` and
shape ``[1, 1]``::

    e1  FillOperator   --              -> D, [r, 1]     ones
    e2  MatmulOperator [e1, g]         -> D, [r, 1]     g repeated down a column
    e3  FillOperator   --              -> D, [1, c]     ones
    e4  MatmulOperator [e2, e3]        -> D, [r, c]     g repeated across
    e5  MulOperator    [e4]            -> D, [r, c]     scaled by 1/d

``e5`` carries the division node's own ``output_value_id``, and the broadcast
intermediate -- which the predicate below requires to have exactly one consumer
and to be no declared gradient output -- is the only value the pass removes.

Why this is sound for any value, and what it is *not* claiming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The rewrite rests on an algebraic identity, not on a claim about which rule
emitted the region. No artifact carries an origin marker -- a
`DerivativeProgram` records a transform version and a free-text seed contract,
and no node records the rule that produced it -- so such a claim would be
unprovable. What *is* provable from the artifact alone is that for any ``g`` of
shape ``[1, 1]``, ``broadcast(g, [r, c])[i, j] == g[0, 0]``, and equally
``((ones[r,1] @ g) @ ones[1,c])[i, j] == g[0, 0]``. The two matmuls therefore
compute the broadcast exactly, whatever produced ``g``.

The predicate
~~~~~~~~~~~~~
A `DivOperator` node ``q`` is rewritten exactly when all seven of the following
hold; each is decidable from the artifact alone:

1. ``q`` has exactly one operand, and its ``op_params`` is exactly
   ``{"right_literal": d}`` with ``d`` a real non-boolean number.
2. ``q``'s operand is produced by a `BroadcastOperator` node ``b`` in the same
   artifact.
3. ``b`` has exactly one operand; ``b.op_params["shape"]`` is a rank-2 shape
   ``[r, c]`` of positive integers; and ``b.output_typespec`` is complete and
   equals ``D, [r, c]``.
4. ``b``'s operand ``g`` has a complete *recorded* typespec of ``D, [1, 1]`` --
   the same dtype, rank 2, both dimensions exactly 1.
5. ``d`` equals the integer ``r * c`` exactly.
6. ``q.output_typespec`` is complete and equals ``D, [r, c]``.
7. ``b.output_value_id`` has exactly one consumer in the artifact -- ``q`` --
   and is named in neither ``output_gradients`` nor ``gradients``.

**Clause 5 is a scope guard, not a correctness guard.** The rewrite preserves
the computed value for *any* divisor, so nothing about the identity above needs
``d`` to be the element count, and this module does not pretend otherwise. The
clause earns its place for two other reasons: it confines the pass to the
reduction case it exists for, keeping the rewrite reviewable; and it confines
the one inexact step -- substituting ``* (1/d)`` for ``/ d``, which IEEE-754
does not require to agree bit for bit -- to a divisor that is an exact integer
element count rather than to an arbitrary application-chosen divisor whose
numerics a caller may depend on.

A near miss is left alone and raises nothing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A region failing any clause is carried through unchanged, and no error is
raised. `BroadcastOperator`, `DivOperator`, and `ReshapeOperator` are general
operators with many legitimate uses, so declining is the correct response to a
near miss; a backend that then cannot lower one fails closed at lowering with
the existing ``unsupported_operator``. This is deliberately the opposite of the
forward pass's behaviour on an unsupported `MeanOperator`: a mean is
unambiguously inside the declared domain of a pass named for mean expansion, a
broadcast is not. In particular a `ReshapeOperator` preceding a matched region
-- which is what performs the genuine rank change when the reduction declared
``keepdims=False`` -- is already truthful, is exactly the rank change clause 4
requires to have happened, and is never merged into the emitted region.

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
    BroadcastOperator,
    DivOperator,
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
from .reverse import DerivativeProgram
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
# Sidecar provenance (§9.4)
#
# Expansion bookkeeping is never placed in `op_params` -- a handler receives
# `op_params` as operator configuration and cannot be expected to distinguish
# configuration from bookkeeping (Inv-10). Provenance instead travels beside
# the rewritten artifact, in a `MeanExpansionRegion` per rewritten region.
#
# The two detailed passes below are the *only* place either rewrite is
# implemented. `expand_mean_graph` and `expand_mean_derivative_program` are
# defined, further down this module, as the corresponding detailed pass
# followed by selecting its artifact field -- never as a second, independent
# walk -- so the composable and detailed forms cannot disagree.
#
# Why a gradient-path region is always `"rank_preserving"`: §9.4 admits only
# `"rank_preserving"` and `"rank_reducing"` as tier strings, and the §8.4
# region -- two matmuls and one scale -- performs no rank change regardless of
# the forward mean's own `keepdims`, which only ever affects the *forward*
# region's tier. A broadcast-and-scale region is therefore recorded
# `"rank_preserving"` unconditionally; this is a property of the region's
# shape arithmetic, not an oversight of the field's two-value domain.
# --------------------------------------------------------------------------

# §9.1 -- the exact `pass_name` a forward-expansion region reports.
MEAN_EXPANSION_FORWARD = "mean_expansion_forward"

# §9.1 -- the exact `pass_name` a gradient-path region reports.
BROADCAST_SCALE_EXPANSION = "broadcast_scale_expansion"


@dataclass(frozen=True)
class MeanExpansionRegion:
    """Provenance for one region a pass rewrote, exactly the §9.4 fields."""

    pass_name: str
    source_node_ids: tuple[str, ...]
    emitted_node_ids: tuple[str, ...]
    terminal_value_id: str
    tier: str


@dataclass(frozen=True)
class MeanGraphExpansionResult:
    """The forward pass's rewritten graph, plus one region per rewrite."""

    graph: TensorGraph
    regions: tuple[MeanExpansionRegion, ...]


@dataclass(frozen=True)
class MeanDerivativeExpansionResult:
    """The gradient-path pass's rewritten program, plus one region per rewrite."""

    program: DerivativeProgram
    regions: tuple[MeanExpansionRegion, ...]


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


def expand_mean_graph_detailed(graph: TensorGraph) -> MeanGraphExpansionResult:
    """Return *graph* with every supported all-axis rank-2 mean expanded,
    together with one `MeanExpansionRegion` per rewritten region (§9.4).

    Each supported `MeanOperator` (FR-128-006) is replaced in place by the
    matmul-based region of §8.3 -- five nodes when the mean declared
    ``keepdims=True``, six when it declared ``keepdims=False``, the sixth a real
    `ReshapeOperator` performing the rank change. Every other node, and
    ``inputs`` and ``outputs``, are carried through unchanged; the input graph is
    never mutated, and equal graphs expand to equal results including every
    region record.

    Every candidate mean is validated before a single node is emitted, so an
    unsupported mean raises its categorized failure (§13.1) naming the node and
    the clause of FR-128-006 it failed, and never yields a partially rewritten
    graph. `expand_mean_graph` is defined below as this pass followed by
    selecting the `graph` field -- there is no second implementation of the
    rewrite for the two forms to disagree about.
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
    regions: list[MeanExpansionRegion] = []
    for node in nodes:
        supported = supported_means.get(node.node_id)
        if supported is None:
            expanded_nodes.append(node)
            continue
        emitted_region = _emit_mean_region(supported, minter)
        expanded_nodes.extend(emitted_region)
        regions.append(
            MeanExpansionRegion(
                pass_name=MEAN_EXPANSION_FORWARD,
                source_node_ids=(node.node_id,),
                emitted_node_ids=tuple(
                    emitted_node.node_id for emitted_node in emitted_region
                ),
                terminal_value_id=emitted_region[-1].output_value_id,
                tier="rank_preserving" if supported.keepdims else "rank_reducing",
            )
        )

    return MeanGraphExpansionResult(
        graph=TensorGraph(
            nodes=expanded_nodes,
            inputs=list(graph.inputs),
            outputs=list(graph.outputs),
        ),
        regions=tuple(regions),
    )


def expand_mean_graph(graph: TensorGraph) -> TensorGraph:
    """Return *graph* with every supported all-axis rank-2 mean expanded.

    Defined as `expand_mean_graph_detailed` followed by selecting its `graph`
    field, so this composable form and the detailed form share one rewrite and
    cannot disagree. The single positional parameter is unchanged: it is what
    lets this pass be used directly as an expansion hook (§9.2).
    """
    return expand_mean_graph_detailed(graph).graph


# --------------------------------------------------------------------------
# The §8.4 broadcast-and-scale predicate (FR-128-009 .. FR-128-012)
# --------------------------------------------------------------------------

# The rank the rewrite is defined for. A pair of matmuls expands exactly two
# axes, so a broadcast to any other rank is outside the pass's domain.
_BROADCAST_SCALE_RANK = 2

# The shape the broadcast source must have. The identity the rewrite rests on
# holds only when every element of the broadcast result is the same element of
# the source, which is what `[1, 1]` guarantees.
_BROADCAST_SCALE_SOURCE_SHAPE = (1, 1)

# The complete `op_params` key set of a rewritable division. Exact, not minimal:
# a division carrying any further key is describing an operation this pass has
# not proven anything about, so it is declined rather than interpreted.
_SCALE_PARAM_KEYS = frozenset({"right_literal"})


@dataclass(frozen=True)
class _BroadcastScaleRegion:
    """One broadcast-and-scale region proven rewritable by the seven clauses."""

    broadcast_node_id: str
    div_node_id: str
    source_value_id: str
    source_typespec: dict[str, object]
    dtype: str
    rows: int
    columns: int
    divisor: int
    output_value_id: str


def _concrete_typespec(typespec: object) -> tuple[str, tuple[int, ...]] | None:
    """Return *typespec* as ``(dtype, shape)``, or ``None`` if it is not complete.

    "Complete" is the whole content of clauses 3, 4, and 6: a non-empty dtype
    and a ranked shape of concrete integers. This returns ``None`` rather than
    raising, because an incomplete typespec on a general operator is a chain the
    pass declines, not a defect it reports.
    """
    if not isinstance(typespec, Mapping):
        return None
    dtype = typespec.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        return None
    try:
        shape = typespec_ranked_shape(dict(typespec))
    except AutodiffError:
        return None
    dimensions: list[int] = []
    for dimension in shape:
        # A symbolic dimension is a string here; `bool` is not a dimension.
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            return None
        dimensions.append(dimension)
    return dtype, tuple(dimensions)


def _rank_two_target_shape(op_params: Mapping[str, object]) -> tuple[int, int] | None:
    """Read a broadcast target as a rank-2 shape of positive integers."""
    shape = op_params.get("shape")
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        return None
    if len(shape) != _BROADCAST_SCALE_RANK:
        return None
    dimensions: list[int] = []
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            return None
        if dimension <= 0:
            return None
        dimensions.append(dimension)
    rows, columns = dimensions
    return rows, columns


def _value_consumer_counts(nodes: Sequence[TensorNodeRecord]) -> dict[str, int]:
    """Count how many operand slots read each value id.

    Built once per invocation and shared by every candidate, so clause 7 costs a
    dictionary lookup rather than a rescan of the artifact (NFR-128-001). A node
    naming the same value twice counts twice, because it reads it twice.
    """
    counts: dict[str, int] = {}
    for node in nodes:
        for value_id in node.input_value_ids:
            counts[value_id] = counts.get(value_id, 0) + 1
    return counts


def _declared_gradient_value_ids(program: DerivativeProgram) -> frozenset[str]:
    """Return every value id the program declares as a gradient result.

    Clause 7 forbids removing one of these: a value a caller is promised is not
    an intermediate the pass may absorb, however it is computed.
    """
    declared = {value_id for value_id in program.output_gradients if value_id is not None}
    declared.update(program.gradients.values())
    return frozenset(declared)


def _broadcast_scale_region(
    node: TensorNodeRecord,
    *,
    producers: Mapping[str, TensorNodeRecord],
    typespecs: Mapping[str, object],
    consumer_counts: Mapping[str, int],
    declared_gradients: frozenset[str],
) -> _BroadcastScaleRegion | None:
    """Prove *node* rewritable, or return ``None`` for any chain that is not.

    The seven clauses are evaluated in their stated order. Every failure returns
    ``None``: a chain outside the predicate is left alone and raises nothing
    (FR-128-011).
    """
    # Clause 1 -- a one-operand division by a real, non-boolean literal.
    if not isinstance(node.operator, DivOperator):
        return None
    if len(node.input_value_ids) != 1:
        return None
    if set(node.op_params) != _SCALE_PARAM_KEYS:
        return None
    divisor = node.op_params["right_literal"]
    if isinstance(divisor, bool) or not isinstance(divisor, (int, float)):
        return None

    # Clause 2 -- its operand is produced by a broadcast in the same artifact.
    broadcast = producers.get(node.input_value_ids[0])
    if broadcast is None or not isinstance(broadcast.operator, BroadcastOperator):
        return None

    # Clause 3 -- a one-operand broadcast to a rank-2 target it declares truthfully.
    if len(broadcast.input_value_ids) != 1:
        return None
    target_shape = _rank_two_target_shape(broadcast.op_params)
    if target_shape is None:
        return None
    rows, columns = target_shape
    broadcast_typespec = _concrete_typespec(broadcast.output_typespec)
    if broadcast_typespec is None:
        return None
    dtype, broadcast_shape = broadcast_typespec
    if broadcast_shape != (rows, columns):
        return None

    # Clause 4 -- the broadcast source is a recorded `[1, 1]` value of the same dtype.
    source_value_id = broadcast.input_value_ids[0]
    source_typespec = _concrete_typespec(typespecs.get(source_value_id))
    if source_typespec is None:
        return None
    source_dtype, source_shape = source_typespec
    if source_dtype != dtype or source_shape != _BROADCAST_SCALE_SOURCE_SHAPE:
        return None

    # Clause 5 -- the scope guard: the divisor is exactly the element count.
    element_count = rows * columns
    if divisor != element_count:
        return None

    # Clause 6 -- the division declares the broadcast's own dtype and shape.
    output_typespec = _concrete_typespec(node.output_typespec)
    if output_typespec != (dtype, (rows, columns)):
        return None

    # Clause 7 -- the broadcast result is this division's private intermediate.
    broadcast_value_id = broadcast.output_value_id
    if consumer_counts.get(broadcast_value_id, 0) != 1:
        return None
    if broadcast_value_id in declared_gradients:
        return None

    return _BroadcastScaleRegion(
        broadcast_node_id=broadcast.node_id,
        div_node_id=node.node_id,
        source_value_id=source_value_id,
        source_typespec=_boundary_typespec(dtype, source_shape),
        dtype=dtype,
        rows=rows,
        columns=columns,
        divisor=element_count,
        output_value_id=node.output_value_id,
    )


# --------------------------------------------------------------------------
# The §8.4 region emitter
# --------------------------------------------------------------------------


def _emit_broadcast_scale_region(
    region: _BroadcastScaleRegion, minter: _IdentifierMinter
) -> list[TensorNodeRecord]:
    """Emit the five nodes replacing one proven-rewritable region.

    Every declared type below is derived from the operands the node actually
    reads -- the shape helpers compute it -- rather than copied from the region
    being replaced, so no node can declare a shape its operation cannot produce.
    The terminal scale's declared type therefore agrees with the replaced
    division's by derivation rather than by assignment, which clause 6 is what
    guarantees.
    """
    column_ones = _emit_fill(minter, dtype=region.dtype, shape=(region.rows, 1))
    down_column = _emit_matmul(
        minter,
        left=column_ones,
        right=(region.source_value_id, region.source_typespec),
    )
    row_ones = _emit_fill(minter, dtype=region.dtype, shape=(1, region.columns))
    across_row = _emit_matmul(minter, left=down_column, right=row_ones)
    scale = _emit_scale(
        minter,
        operand=across_row,
        right_literal=1.0 / float(region.divisor),
        output_value_id=region.output_value_id,
    )
    return [column_ones, down_column, row_ones, across_row, scale]


# --------------------------------------------------------------------------
# The public gradient-path pass
# --------------------------------------------------------------------------


def _existing_value_ids(program: DerivativeProgram) -> frozenset[str]:
    """Return every value id the program names anywhere.

    Wider than the set of produced values on purpose: a minted identifier must
    not collide with a value the program merely reads or merely records a
    typespec for either (Inv-5).
    """
    value_ids = set(_indexed_value_ids(program.nodes, []))
    value_ids.update(program.value_typespecs)
    for node in program.nodes:
        value_ids.update(node.input_value_ids)
    return frozenset(value_ids)


def expand_mean_derivative_program_detailed(
    program: DerivativeProgram,
) -> MeanDerivativeExpansionResult:
    """Return *program* with every rewritable broadcast-and-scale region
    expanded, together with one `MeanExpansionRegion` per rewritten region
    (§9.4).

    A region is rewritten exactly when it satisfies all seven clauses of the
    predicate documented at the top of this module, which is decided from the
    artifact alone and makes no claim about what produced any node. Each
    rewritten region becomes the five nodes documented there, occupying the
    position of the division it replaces; the broadcast intermediate is the only
    value the pass removes, and clause 7 is what proves removing it is safe.
    `source_node_ids` names both nodes the matched chain replaces -- the
    broadcast, which is removed outright, and the division, whose position the
    emitted region occupies -- in the artifact order they must appear in for
    the broadcast's output to be a valid input to the division.

    A chain failing any clause -- and any `ReshapeOperator` preceding a matched
    one -- is carried through unchanged and raises nothing (FR-128-011,
    FR-128-012). ``gradients``, ``output_gradients``, ``metadata``, and
    ``value_typespecs`` are carried through unchanged; the input program is
    never mutated, and equal programs expand to equal results including every
    region record.

    Every candidate is proven before a single node is emitted, so the walk is
    linear in node count with one consumer index and no per-region rescan.
    `expand_mean_derivative_program` is defined below as this pass followed by
    selecting the `program` field -- there is no second implementation of the
    rewrite for the two forms to disagree about.
    """
    nodes = program.nodes
    existing_node_ids = _indexed_node_ids(nodes)
    existing_value_ids = _existing_value_ids(program)
    typespecs = _declared_typespecs(nodes, list(program.value_typespecs.items()))
    producers = {node.output_value_id: node for node in nodes}
    consumer_counts = _value_consumer_counts(nodes)
    declared_gradients = _declared_gradient_value_ids(program)

    # Proof first, over every candidate, before any emission (§13.2).
    matched_regions: dict[str, _BroadcastScaleRegion] = {}
    for node in nodes:
        matched_region = _broadcast_scale_region(
            node,
            producers=producers,
            typespecs=typespecs,
            consumer_counts=consumer_counts,
            declared_gradients=declared_gradients,
        )
        if matched_region is not None:
            matched_regions[node.node_id] = matched_region
    absorbed_node_ids = {
        matched_region.broadcast_node_id for matched_region in matched_regions.values()
    }

    minter = _IdentifierMinter(
        existing_node_ids=existing_node_ids, existing_value_ids=existing_value_ids
    )
    expanded_nodes: list[TensorNodeRecord] = []
    provenance_regions: list[MeanExpansionRegion] = []
    for node in nodes:
        if node.node_id in absorbed_node_ids:
            continue
        matched_region = matched_regions.get(node.node_id)
        if matched_region is None:
            expanded_nodes.append(node)
            continue
        emitted_region = _emit_broadcast_scale_region(matched_region, minter)
        expanded_nodes.extend(emitted_region)
        provenance_regions.append(
            MeanExpansionRegion(
                pass_name=BROADCAST_SCALE_EXPANSION,
                source_node_ids=(
                    matched_region.broadcast_node_id,
                    matched_region.div_node_id,
                ),
                emitted_node_ids=tuple(
                    emitted_node.node_id for emitted_node in emitted_region
                ),
                terminal_value_id=emitted_region[-1].output_value_id,
                # The §8.4 region performs no rank change regardless of the
                # forward mean's own tier -- see the module-level rationale
                # above the sidecar dataclasses.
                tier="rank_preserving",
            )
        )

    return MeanDerivativeExpansionResult(
        program=DerivativeProgram(
            nodes=expanded_nodes,
            gradients=dict(program.gradients),
            output_gradients=list(program.output_gradients),
            metadata=program.metadata,
            value_typespecs=dict(program.value_typespecs),
        ),
        regions=tuple(provenance_regions),
    )


def expand_mean_derivative_program(program: DerivativeProgram) -> DerivativeProgram:
    """Return *program* with every rewritable broadcast-and-scale region expanded.

    Defined as `expand_mean_derivative_program_detailed` followed by selecting
    its `program` field, so this composable form and the detailed form share
    one rewrite and cannot disagree. The single positional parameter is
    unchanged: it is what lets this pass be used directly as an expansion hook
    (§9.2).
    """
    return expand_mean_derivative_program_detailed(program).program
