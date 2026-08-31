from __future__ import annotations

import numpy as np

from tinychain.autodiff import (
    AddOperator,
    BroadcastOperator,
    BroadcastReduceOperator,
    DivOperator,
    FillOperator,
    MatmulOperator,
    MeanOperator,
    MulOperator,
    ReshapeOperator,
    SubOperator,
    SumOperator,
    TensorNodeRecord,
    TransposeOperator,
    fill_descriptor,
)

# Explicit, total mapping from a fill descriptor's dtype string to the numpy
# dtype it denotes. Kept total over the differentiable dtypes the framework
# validates elsewhere (`FLOATING_DTYPES`, `seed.FLOAT_DTYPES`); an unmapped
# dtype fails loudly rather than silently widening to a float64 guess, which
# would let a later equivalence assertion pass for the wrong reason.
_FILL_NUMPY_DTYPES: dict[str, np.dtype] = {
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
}


def numpy_dtype_for_fill(dtype: str) -> np.dtype:
    """Return the numpy dtype a fill descriptor's ``dtype`` string denotes.

    Raises ``KeyError`` for any dtype outside the explicit, total mapping --
    this test-only helper never guesses a dtype it was not told about.
    """
    return _FILL_NUMPY_DTYPES[dtype]


class NumpyAutodiffDispatcher:
    """Test-only executor for generated derivative nodes."""

    def __call__(self, node: TensorNodeRecord, args: list[object]) -> np.ndarray:
        if isinstance(node.operator, AddOperator):
            return np.asarray(args[0]) + np.asarray(args[1])
        if isinstance(node.operator, FillOperator):
            descriptor = fill_descriptor(node)
            return np.full(descriptor.shape, descriptor.fill, dtype=numpy_dtype_for_fill(descriptor.dtype))
        if isinstance(node.operator, BroadcastOperator):
            return np.broadcast_to(np.asarray(args[0]), tuple(node.op_params["shape"]))
        if isinstance(node.operator, BroadcastReduceOperator):
            return self._broadcast_reduce(np.asarray(args[0]), node.op_params["target_shape"])
        if isinstance(node.operator, SubOperator):
            return np.asarray(args[0]) - self._right_arg(node, args)
        if isinstance(node.operator, MulOperator):
            return np.asarray(args[0]) * self._right_arg(node, args)
        if isinstance(node.operator, DivOperator):
            return np.asarray(args[0]) / self._right_arg(node, args)
        if isinstance(node.operator, MatmulOperator):
            return np.matmul(np.asarray(args[0]), np.asarray(args[1]))
        if isinstance(node.operator, MeanOperator):
            return np.mean(np.asarray(args[0]), axis=tuple(node.op_params["axes"]), keepdims=node.op_params["keepdims"])
        if isinstance(node.operator, ReshapeOperator):
            return np.reshape(np.asarray(args[0]), tuple(node.op_params["shape"]))
        if isinstance(node.operator, SumOperator):
            return np.sum(np.asarray(args[0]), axis=tuple(node.op_params["axes"]), keepdims=node.op_params["keepdims"])
        if isinstance(node.operator, TransposeOperator):
            return np.transpose(np.asarray(args[0]), axes=node.op_params["perm"])
        raise AssertionError(f"unsupported test operator {node.operator!r}")

    @staticmethod
    def _right_arg(node: TensorNodeRecord, args: list[object]) -> object:
        if "right_literal" in node.op_params:
            return node.op_params["right_literal"]
        return np.asarray(args[1])

    @staticmethod
    def _broadcast_reduce(value: np.ndarray, target_shape: object) -> np.ndarray:
        target = tuple(int(dim) for dim in target_shape)
        if len(target) > value.ndim:
            raise AssertionError("broadcast-reduce target rank exceeds input rank")

        padded_target = (1,) * (value.ndim - len(target)) + target
        axes: list[int] = []
        for axis, (source_dim, target_dim) in enumerate(zip(value.shape, padded_target, strict=True)):
            if source_dim == target_dim:
                continue
            if target_dim == 1:
                axes.append(axis)
                continue
            raise AssertionError("invalid broadcast-reduce target shape")

        reduced = value.sum(axis=tuple(axes), keepdims=True) if axes else value
        return reduced.reshape(target)
