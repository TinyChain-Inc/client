from __future__ import annotations

from dataclasses import dataclass

from .protocol import AutodiffError


FLOAT_DTYPES: tuple[str, ...] = ("f32", "f64")


def typespec_dtype(typespec: dict[str, object] | None) -> str:
    if typespec is None or "dtype" not in typespec:
        raise AutodiffError("missing_dtype_metadata", "tensor dtype metadata is required")
    return str(typespec["dtype"])


def typespec_shape(typespec: dict[str, object] | None) -> tuple[int, ...]:
    if typespec is None or "shape" not in typespec:
        raise AutodiffError("missing_shape_metadata", "tensor shape metadata is required")
    try:
        return tuple(int(dim) for dim in typespec["shape"])
    except (TypeError, ValueError) as exc:
        raise AutodiffError("missing_shape_metadata", "tensor shape metadata must be a sequence") from exc


@dataclass(frozen=True)
class SeedValidator:
    floating_dtypes: tuple[str, ...] = FLOAT_DTYPES

    def validate(
        self,
        *,
        seed_typespec: dict[str, object] | None,
        output_typespec: dict[str, object] | None,
    ) -> None:
        seed_dtype = typespec_dtype(seed_typespec)
        output_dtype = typespec_dtype(output_typespec)
        if seed_dtype not in self.floating_dtypes or output_dtype not in self.floating_dtypes:
            raise AutodiffError(
                "dtype_not_differentiable",
                f"autodiff supports only {', '.join(self.floating_dtypes)} tensors",
            )

        seed_shape = typespec_shape(seed_typespec)
        output_shape = typespec_shape(output_typespec)
        if seed_shape != output_shape:
            raise AutodiffError(
                "seed_shape_mismatch",
                f"seed shape {seed_shape} does not match selected output shape {output_shape}",
            )
