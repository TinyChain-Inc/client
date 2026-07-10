from __future__ import annotations

from collections.abc import Mapping, Sequence

from .protocol import AutodiffError


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
