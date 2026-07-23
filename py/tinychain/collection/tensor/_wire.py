from __future__ import annotations

from .schema import (
    TensorStorageLayout,
    TensorStorageSchema,
    TensorViewAxis,
    TensorViewAxisMap,
    TensorViewSchema,
)


def encode_storage_layout(layout: TensorStorageLayout) -> tuple[int, int | None]:
    if layout.kind == "dense":
        return (0, None)

    return (1, layout.sparse_axis)


def decode_storage_layout(payload: object) -> TensorStorageLayout:
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise TypeError("layout wire payload must be a (tag, axis) pair")

    tag, axis = payload
    if tag == 0:
        if axis is not None:
            raise ValueError("dense layout wire payload must not include axis")
        return TensorStorageLayout(kind="dense")

    if tag == 1:
        if axis is not None:
            axis = int(axis)
        return TensorStorageLayout(kind="sparse", sparse_axis=axis)

    raise ValueError(f"unknown layout tag {tag}")


def encode_storage_schema(schema: TensorStorageSchema) -> tuple[str, list[int], tuple[int, int | None]]:
    return (schema.dtype, list(schema.shape), encode_storage_layout(schema.layout))


def decode_storage_schema(payload: object) -> TensorStorageSchema:
    if not isinstance(payload, (tuple, list)) or len(payload) != 3:
        raise TypeError("tensor schema wire payload must be (dtype, shape, layout)")

    dtype, shape, layout = payload
    if not isinstance(dtype, str):
        raise TypeError("tensor schema dtype must be string")
    if not isinstance(shape, (tuple, list)):
        raise TypeError("tensor schema shape must be a list")

    normalized_shape = tuple(int(dim) for dim in shape)
    if not normalized_shape:
        raise ValueError("tensor schema shape must not be empty")
    if any(dim <= 0 for dim in normalized_shape):
        raise ValueError("tensor schema shape dimensions must be positive")

    return TensorStorageSchema(
        dtype=dtype,
        shape=normalized_shape,
        layout=decode_storage_layout(layout),
    )


def encode_view_axis_map(axis_map: TensorViewAxisMap) -> tuple[int, list[int]]:
    if axis_map.kind == "identity":
        return (0, [])

    if axis_map.kind == "affine":
        return (1, [axis_map.start, axis_map.step])

    return (2, list(axis_map.gather))


def decode_view_axis_map(payload: object) -> TensorViewAxisMap:
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise TypeError("axis map wire payload must be (tag, payload)")

    tag, data = payload
    if not isinstance(data, (tuple, list)):
        raise TypeError("axis map payload data must be a list")

    if tag == 0:
        if len(data) != 0:
            raise ValueError("identity axis map payload must be empty")
        return TensorViewAxisMap(kind="identity")

    if tag == 1:
        if len(data) != 2:
            raise ValueError("affine axis map payload must contain [start, step]")
        return TensorViewAxisMap(kind="affine", start=int(data[0]), step=int(data[1]))

    if tag == 2:
        return TensorViewAxisMap(kind="gather", gather=tuple(int(index) for index in data))

    raise ValueError(f"unknown axis map tag {tag}")


def encode_view_axis(axis: TensorViewAxis) -> tuple[int, tuple[int, list[int]]]:
    return (axis.base_axis, encode_view_axis_map(axis.map))


def decode_view_axis(payload: object) -> TensorViewAxis:
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise TypeError("view axis wire payload must be (base_axis, axis_map)")

    base_axis, axis_map = payload
    return TensorViewAxis(base_axis=int(base_axis), map=decode_view_axis_map(axis_map))


def encode_view_schema(
    schema: TensorViewSchema,
) -> tuple[int, list[tuple[int, tuple[int, list[int]]]], list[int | None]]:
    return (
        schema.base_rank,
        [encode_view_axis(axis) for axis in schema.axes],
        list(schema.base_fixed),
    )


def decode_view_schema(payload: object) -> TensorViewSchema:
    if not isinstance(payload, (tuple, list)) or len(payload) != 3:
        raise TypeError("view schema wire payload must be (base_rank, axes, base_fixed)")

    base_rank, axes, base_fixed = payload
    if not isinstance(axes, (tuple, list)):
        raise TypeError("view schema axes must be a list")
    if not isinstance(base_fixed, (tuple, list)):
        raise TypeError("view schema base_fixed must be a list")

    return TensorViewSchema(
        base_rank=int(base_rank),
        axes=tuple(decode_view_axis(axis) for axis in axes),
        base_fixed=tuple(None if item is None else int(item) for item in base_fixed),
    )
