from __future__ import annotations

import numpy as np

from tinychain.autodiff import AddOperator, BroadcastReduceOperator, TensorNodeRecord


class NumpyAutodiffDispatcher:
    """Test-only executor for generated derivative nodes."""

    def __call__(self, node: TensorNodeRecord, args: list[object]) -> np.ndarray:
        if isinstance(node.operator, AddOperator):
            return np.asarray(args[0]) + np.asarray(args[1])
        if isinstance(node.operator, BroadcastReduceOperator):
            return self._broadcast_reduce(np.asarray(args[0]), node.op_params["target_shape"])
        raise AssertionError(f"unsupported test operator {node.operator!r}")

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
