from __future__ import annotations

from collections.abc import Mapping, Sequence

from .protocol import AutodiffError


# Client-side shape metadata only; runtime route params receive concrete ints.
ShapeDim = int | str
Shape = tuple[ShapeDim, ...]
ConcreteShape = tuple[int, ...]


def typespec_shape(typespec: dict[str, object] | None) -> ConcreteShape:
    shape = typespec_ranked_shape(typespec)
    unresolved = [dim for dim in shape if isinstance(dim, str)]
    if unresolved:
        joined = ", ".join(repr(dim) for dim in unresolved)
        raise AutodiffError(
            "unresolved_symbolic_shape",
            f"tensor shape requires concrete dimensions; unresolved symbol(s): {joined}",
        )
    return tuple(dim for dim in shape if isinstance(dim, int))


def typespec_ranked_shape(typespec: dict[str, object] | None) -> Shape:
    if typespec is None or "shape" not in typespec:
        raise AutodiffError("missing_shape_metadata", "tensor shape metadata is required")
    return parse_shape(typespec["shape"], label="tensor shape metadata")


def parse_shape(value: object, *, label: str) -> Shape:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AutodiffError("missing_shape_metadata", f"{label} must be a sequence")

    dims: list[ShapeDim] = []
    for dim in value:
        dims.append(parse_dim(dim, label=label))
    return tuple(dims)


def parse_dim(value: object, *, label: str) -> ShapeDim:
    if type(value) is int:
        if value < 0:
            raise AutodiffError("missing_shape_metadata", f"{label} dimensions must be non-negative")
        return value
    if isinstance(value, str):
        if not value.isidentifier():
            raise AutodiffError(
                "missing_shape_metadata",
                f"{label} symbolic dimensions must be valid identifiers",
            )
        return value
    raise AutodiffError("missing_shape_metadata", f"{label} dimensions must be integers or symbols")


def shape_rank(shape: Shape) -> int:
    return len(shape)


def same_shape_or_symbolically_compatible(
    lhs: Shape,
    rhs: Shape,
    *,
    category: str,
    message: str,
) -> None:
    if len(lhs) != len(rhs):
        raise AutodiffError(category, message)
    bindings: dict[str, int] = {}
    try:
        for lhs_dim, rhs_dim in zip(lhs, rhs, strict=True):
            bind_compatible_dims(lhs_dim, rhs_dim, bindings=bindings)
    except AutodiffError as exc:
        if exc.category in {"symbolic_shape_mismatch", "unresolved_symbolic_shape"}:
            raise AutodiffError(category, message) from exc
        raise


def bind_compatible_shapes(
    *,
    symbolic_shape: Shape,
    concrete_shape: Shape,
    bindings: dict[str, int],
    label: str,
) -> None:
    if len(symbolic_shape) != len(concrete_shape):
        raise AutodiffError(
            "symbolic_shape_mismatch",
            f"{label} rank {len(concrete_shape)} does not match metadata rank {len(symbolic_shape)}",
        )
    for symbolic_dim, concrete_dim in zip(symbolic_shape, concrete_shape, strict=True):
        bind_compatible_dims(symbolic_dim, concrete_dim, bindings=bindings, label=label)


def bind_compatible_dims(
    left: ShapeDim,
    right: ShapeDim,
    *,
    bindings: dict[str, int],
    label: str = "shape",
) -> None:
    if isinstance(left, int) and isinstance(right, int):
        if left != right:
            raise AutodiffError(
                "symbolic_shape_mismatch",
                f"{label} dimension mismatch: {left} != {right}",
            )
        return
    if isinstance(left, str) and isinstance(right, int):
        bind_symbol(left, right, bindings=bindings, label=label)
        return
    if isinstance(left, int) and isinstance(right, str):
        bind_symbol(right, left, bindings=bindings, label=label)
        return
    if left != right:
        raise AutodiffError(
            "unresolved_symbolic_shape",
            f"{label} symbols {left!r} and {right!r} cannot be proven equal",
        )


def bind_symbol(symbol: str, value: int, *, bindings: dict[str, int], label: str) -> None:
    if value < 0:
        raise AutodiffError("symbolic_shape_mismatch", f"{label} dimension {value} is negative")
    existing = bindings.get(symbol)
    if existing is not None and existing != value:
        raise AutodiffError(
            "symbolic_shape_mismatch",
            f"symbol {symbol!r} resolved to both {existing} and {value}",
        )
    bindings[symbol] = value


def resolve_shape(
    shape: Shape,
    bindings: Mapping[str, int] | None,
    *,
    label: str,
) -> ConcreteShape:
    resolved: list[int] = []
    binding_map = dict(bindings or {})
    for dim in shape:
        if isinstance(dim, int):
            resolved.append(dim)
            continue
        if dim not in binding_map:
            raise AutodiffError(
                "unresolved_symbolic_shape",
                f"{label} dimension symbol {dim!r} is unresolved",
            )
        resolved_dim = binding_map[dim]
        if type(resolved_dim) is not int or resolved_dim < 0:
            raise AutodiffError(
                "symbolic_shape_mismatch",
                f"{label} binding for symbol {dim!r} must be a non-negative integer",
            )
        resolved.append(resolved_dim)
    return tuple(resolved)


def resolve_shape_value(
    value: object,
    bindings: Mapping[str, int] | None,
    *,
    label: str,
) -> list[int]:
    return list(resolve_shape(parse_shape(value, label=label), bindings, label=label))


def shape_from_value(value: object) -> ConcreteShape | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise AutodiffError("symbolic_shape_mismatch", "runtime tensor shape must be concrete") from exc


# Differentiable dtypes accepted for forward-traced operations.
FLOATING_DTYPES: tuple[str, ...] = ("f32", "f64")


def check_differentiable_dtype(dtype: str) -> str:
    """Require a floating dtype for a traced forward operation."""
    if dtype not in FLOATING_DTYPES:
        raise AutodiffError(
            "dtype_not_differentiable",
            f"dtype {dtype!r} is not a differentiable floating dtype; expected one of {FLOATING_DTYPES}",
        )
    return dtype


def check_compatible_operand_dtypes(lhs_dtype: str, rhs_dtype: str) -> str:
    """Require equal, differentiable operand dtypes and return the output dtype.

    No dtype promotion is performed.
    """
    if lhs_dtype != rhs_dtype:
        raise AutodiffError(
            "dtype_mismatch",
            f"operand dtypes must match: {lhs_dtype!r} != {rhs_dtype!r}",
        )
    return check_differentiable_dtype(lhs_dtype)


def _broadcast_dim(lhs_dim: ShapeDim, rhs_dim: ShapeDim, bindings: dict[str, int]) -> ShapeDim:
    if isinstance(lhs_dim, int) and lhs_dim == 1:
        return rhs_dim
    if isinstance(rhs_dim, int) and rhs_dim == 1:
        return lhs_dim
    if isinstance(lhs_dim, int) and isinstance(rhs_dim, int):
        if lhs_dim != rhs_dim:
            raise AutodiffError(
                "broadcast_shape_mismatch",
                f"broadcast dimension mismatch: {lhs_dim} != {rhs_dim}",
            )
        return lhs_dim
    try:
        bind_compatible_dims(lhs_dim, rhs_dim, bindings=bindings, label="broadcast")
    except AutodiffError as exc:
        if exc.category == "symbolic_shape_mismatch":
            raise
        raise AutodiffError(
            "unresolved_symbolic_shape",
            f"broadcast dimensions {lhs_dim!r} and {rhs_dim!r} cannot be proven compatible",
        ) from exc
    if isinstance(lhs_dim, str) and isinstance(rhs_dim, str):
        return lhs_dim
    return lhs_dim if isinstance(lhs_dim, int) else rhs_dim


def elementwise_broadcast_shape(
    lhs: Shape, rhs: Shape, *, bindings: dict[str, int] | None = None
) -> Shape:
    """Compute the proven right-aligned broadcast output shape.

    Equal dimensions are kept, `1` broadcasts to the other dimension, missing
    leading dimensions act as leading `1`s, unequal concrete dimensions raise
    `broadcast_shape_mismatch`, equal symbolic dimensions remain symbolic, and
    unprovable symbolic dimensions raise `unresolved_symbolic_shape`.
    """
    rank = max(shape_rank(lhs), shape_rank(rhs))
    padded_lhs = (1,) * (rank - shape_rank(lhs)) + lhs
    padded_rhs = (1,) * (rank - shape_rank(rhs)) + rhs
    binding_map = {} if bindings is None else bindings
    return tuple(
        _broadcast_dim(lhs_dim, rhs_dim, binding_map)
        for lhs_dim, rhs_dim in zip(padded_lhs, padded_rhs, strict=True)
    )


def matmul_output_shape(
    lhs: Shape, rhs: Shape, *, bindings: dict[str, int] | None = None
) -> Shape:
    """Compute the proven matmul output shape.

    Both operands must have rank at least two; `lhs[-1]` and `rhs[-2]` must be
    equal, bindable, or the same symbol; batch dimensions follow the
    elementwise broadcast rules; the output shape is
    `broadcast(lhs[:-2], rhs[:-2]) + (lhs[-2], rhs[-1])`.
    """
    binding_map = {} if bindings is None else bindings
    lhs_rank = shape_rank(lhs)
    rhs_rank = shape_rank(rhs)
    if lhs_rank < 2 or rhs_rank < 2:
        raise AutodiffError(
            "matmul_shape_mismatch",
            f"matmul requires operands of rank at least 2; got ranks {lhs_rank} and {rhs_rank}",
        )
    lhs_inner, rhs_inner = lhs[-1], rhs[-2]
    if isinstance(lhs_inner, int) and isinstance(rhs_inner, int):
        if lhs_inner != rhs_inner:
            raise AutodiffError(
                "matmul_shape_mismatch",
                f"matmul inner dimensions incompatible: {lhs_inner} != {rhs_inner}",
            )
    else:
        try:
            bind_compatible_dims(
                lhs_inner, rhs_inner, bindings=binding_map, label="matmul inner dimension"
            )
        except AutodiffError as exc:
            if exc.category == "symbolic_shape_mismatch":
                raise
            raise AutodiffError(
                "unresolved_symbolic_shape",
                f"matmul inner dimensions {lhs_inner!r} and {rhs_inner!r} cannot be proven compatible",
            ) from exc
    batch_shape = elementwise_broadcast_shape(lhs[:-2], rhs[:-2], bindings=binding_map)
    return batch_shape + (lhs[-2], rhs[-1])


def _normalize_mean_axes(axes: object, rank: int) -> tuple[int, ...]:
    if isinstance(axes, int) and not isinstance(axes, bool):
        raw_axes: tuple[object, ...] = (axes,)
    elif isinstance(axes, Sequence) and not isinstance(axes, (str, bytes)):
        raw_axes = tuple(axes)
    else:
        raise AutodiffError(
            "reduction_shape_mismatch",
            f"mean axes must be an integer or a sequence of integers; got {axes!r}",
        )

    normalized: list[int] = []
    for axis in raw_axes:
        if not isinstance(axis, int) or isinstance(axis, bool):
            raise AutodiffError(
                "reduction_shape_mismatch",
                f"mean axis {axis!r} must be an integer",
            )
        normalized_axis = axis + rank if axis < 0 else axis
        if normalized_axis < 0 or normalized_axis >= rank:
            raise AutodiffError(
                "reduction_shape_mismatch",
                f"mean axis {axis} is out of range for rank {rank}",
            )
        normalized.append(normalized_axis)

    if len(set(normalized)) != len(normalized):
        raise AutodiffError(
            "reduction_shape_mismatch",
            f"mean axes must be unique after normalization: {normalized}",
        )
    return tuple(normalized)


def mean_output_shape(shape: Shape, axes: object, *, keepdims: bool) -> Shape:
    """Compute the proven mean-reduction output shape.

    Axes may be a single integer or a non-string sequence of integers;
    negative axes normalize against rank; normalized axes must be in range and
    unique (`reduction_shape_mismatch`); `keepdims` replaces reduced
    dimensions with `1` instead of removing them; a fully reduced shape
    yields the empty (rank-zero) shape; reducing a symbolic dimension raises
    `unresolved_symbolic_shape`.
    """
    rank = shape_rank(shape)
    normalized_axes = _normalize_mean_axes(axes, rank)
    for axis in normalized_axes:
        dim = shape[axis]
        if isinstance(dim, str):
            raise AutodiffError(
                "unresolved_symbolic_shape",
                f"mean reduction over symbolic dimension {dim!r} at axis {axis} is unresolved",
            )
    if keepdims:
        return tuple(1 if index in normalized_axes else dim for index, dim in enumerate(shape))
    return tuple(dim for index, dim in enumerate(shape) if index not in normalized_axes)


def normalize_transpose_permutation(perm: object) -> tuple[int, ...]:
    """Require a concrete transpose permutation before static inference."""
    if not isinstance(perm, Sequence) or isinstance(perm, (str, bytes)):
        raise AutodiffError(
            "invalid_permutation",
            f"transpose permutation must be a sequence of integers; got {perm!r}",
        )
    perm_tuple = tuple(perm)
    if any(type(axis) is not int for axis in perm_tuple):
        raise AutodiffError(
            "invalid_permutation",
            f"transpose permutation axes must be integers; got {perm_tuple!r}",
        )
    return perm_tuple


def transpose_output_shape(shape: Shape, perm: object) -> Shape:
    """Compute the proven transpose output shape.

    The permutation length must equal the input rank and each axis must
    appear exactly once (`invalid_permutation`); the output shape follows the
    permutation.
    """
    rank = shape_rank(shape)
    perm_tuple = normalize_transpose_permutation(perm)
    if len(perm_tuple) != rank or sorted(perm_tuple) != list(range(rank)):
        raise AutodiffError(
            "invalid_permutation",
            f"transpose permutation {perm_tuple!r} must be a rearrangement of axes 0..{rank - 1}",
        )
    return tuple(shape[axis] for axis in perm_tuple)
