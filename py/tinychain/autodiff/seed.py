from __future__ import annotations

from dataclasses import dataclass

from .protocol import AutodiffError
from .shape import (
    bind_compatible_shapes,
    same_shape_or_symbolically_compatible,
    typespec_ranked_shape,
    typespec_shape,
)


FLOAT_DTYPES: tuple[str, ...] = ("f32", "f64")


def typespec_dtype(typespec: dict[str, object] | None) -> str:
    if typespec is None or "dtype" not in typespec:
        raise AutodiffError("missing_dtype_metadata", "tensor dtype metadata is required")
    return str(typespec["dtype"])


@dataclass(frozen=True)
class SeedValidator:
    """Validate the initial reverse-mode cotangent for a selected output.

    The seed is the upstream dL/d(output) tensor that starts reverse traversal.
    It must have the selected output rank/shape and a differentiable floating dtype.
    Symbolic output dimensions may be bound by a concrete seed typespec.
    """

    floating_dtypes: tuple[str, ...] = FLOAT_DTYPES

    def validate(
        self,
        *,
        seed_typespec: dict[str, object] | None,
        output_typespec: dict[str, object] | None,
    ) -> None:
        """Check that the seed can serve as dL/d(output)."""
        seed_dtype = typespec_dtype(seed_typespec)
        output_dtype = typespec_dtype(output_typespec)
        if seed_dtype not in self.floating_dtypes or output_dtype not in self.floating_dtypes:
            raise AutodiffError(
                "dtype_not_differentiable",
                f"autodiff supports only {', '.join(self.floating_dtypes)} tensors",
            )

        seed_shape = typespec_ranked_shape(seed_typespec)
        output_shape = typespec_ranked_shape(output_typespec)
        same_shape_or_symbolically_compatible(
            seed_shape,
            output_shape,
            category="seed_shape_mismatch",
            message=f"seed shape {seed_shape} does not match selected output shape {output_shape}",
        )

    def bind_seed_symbols(
        self,
        *,
        seed_typespec: dict[str, object] | None,
        output_typespec: dict[str, object] | None,
        bindings: dict[str, int],
    ) -> None:
        seed_shape = typespec_ranked_shape(seed_typespec)
        output_shape = typespec_ranked_shape(output_typespec)
        bind_compatible_shapes(
            symbolic_shape=output_shape,
            concrete_shape=seed_shape,
            bindings=bindings,
            label="seed shape",
        )


__all__ = [
    "FLOAT_DTYPES",
    "SeedValidator",
    "typespec_dtype",
    "typespec_ranked_shape",
    "typespec_shape",
]
