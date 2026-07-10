from __future__ import annotations

import numpy as np

from tinychain.autodiff import (
    AddOperator,
    BroadcastOperator,
    BroadcastReduceOperator,
    DivOperator,
    MatmulOperator,
    MeanOperator,
    MulOperator,
    ReshapeOperator,
    SubOperator,
    SumOperator,
    TensorNodeRecord,
    TransposeOperator,
)


class NumpyAutodiffDispatcher:
    """Test-only executor for generated derivative nodes."""

    def __call__(self, node: TensorNodeRecord, args: list[object]) -> np.ndarray:
        if isinstance(node.operator, AddOperator):
            return np.asarray(args[0]) + np.asarray(args[1])
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
