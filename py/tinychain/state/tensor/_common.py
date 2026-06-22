from __future__ import annotations

from collections.abc import Iterable as IterableABC

from ..scalar import Scalar, autobox


def params(**kwargs: object) -> dict[str, Scalar]:
    return {key: autobox(value) for key, value in kwargs.items() if value is not None}


def reduce_args(axes: object = None, keepdims: bool = False) -> dict[str, Scalar]:
    params_out: dict[str, Scalar] = {}
    if axes is not None:
        if isinstance(axes, IterableABC) and not isinstance(axes, (str, bytes, bytearray)):
            params_out["axes"] = autobox(list(axes))
        else:
            params_out["axes"] = autobox([axes])
    if keepdims:
        params_out["keepdims"] = autobox(keepdims)
    return params_out


def normalize_shape(shape: object) -> tuple[int, ...] | None:
    if isinstance(shape, IterableABC) and not isinstance(shape, (str, bytes, bytearray)):
        dims: list[int] = []
        for dim in shape:
            try:
                dims.append(int(dim))
            except (TypeError, ValueError):
                return None
        return tuple(dims)

    return None


def normalize_permutation(permutation: object) -> tuple[int, ...] | None:
    if permutation is None:
        return None

    if isinstance(permutation, IterableABC) and not isinstance(permutation, (str, bytes, bytearray)):
        axes: list[int] = []
        for axis in permutation:
            try:
                axes.append(int(axis))
            except (TypeError, ValueError):
                return None
        return tuple(axes)

    return None


def infer_broadcast_axes(source_shape: object, target_shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if not isinstance(source_shape, IterableABC) or isinstance(source_shape, (str, bytes, bytearray)):
        return None

    try:
        source = [int(dim) for dim in source_shape]
    except (TypeError, ValueError):
        return None

    if len(source) > len(target_shape):
        return None

    pad = len(target_shape) - len(source)
    padded = [1] * pad + source
    broadcast_axes: list[int] = []

    for axis, (src, dst) in enumerate(zip(padded, target_shape)):
        if src == 1 and dst > 1:
            broadcast_axes.append(axis)
        elif src != dst:
            return None

    return tuple(broadcast_axes)
