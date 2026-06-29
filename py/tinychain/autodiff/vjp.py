from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .graph import (
    AddOperator,
    BroadcastReduceOperator,
    MatmulOperator,
    TensorNodeRecord,
    TensorOperator,
    TransposeOperator,
)
from .protocol import AutodiffError
from .seed import typespec_shape


@dataclass(frozen=True)
class VjpContext:
    upstream_value_id: str
    node: TensorNodeRecord
    value_typespecs: dict[str, dict[str, object]]
    needed_input_value_ids: frozenset[str]
    next_value_id: Callable[[], str]
    next_node_id: Callable[[], str]


@dataclass(frozen=True)
class VjpResult:
    gradients: dict[str, str]
    derivative_nodes: list[TensorNodeRecord]


class VjpRule(Protocol):
    operator_type: type[TensorOperator]

    def apply(self, context: VjpContext) -> VjpResult:
        ...


class VjpRegistry:
    """Transform-owned mapping from TensorOperator types to VJP rules.

    TensorOperator instances are operation descriptors. Autodiff rules live in
    the transform layer so different transform versions can choose rule sets,
    unsupported operators fail explicitly, and operators do not become
    callback/backward implementations.
    """

    def __init__(self) -> None:
        self._rules: dict[type[TensorOperator], VjpRule] = {}

    def register(self, rule: VjpRule) -> None:
        self._rules[rule.operator_type] = rule

    def lookup(self, operator: TensorOperator) -> VjpRule:
        try:
            return self._rules[type(operator)]
        except KeyError as exc:
            raise AutodiffError(
                "unsupported_operator",
                f"no VJP rule registered for {operator.route_name}",
            ) from exc


@dataclass(frozen=True)
class BroadcastReductionPlan:
    result_shape: tuple[int, ...]
    operand_shape: tuple[int, ...]
    axes: tuple[int, ...]


class BroadcastReductionPlanner:
    def plan(
        self,
        *,
        result_shape: tuple[int, ...],
        operand_shape: tuple[int, ...],
    ) -> BroadcastReductionPlan:
        if len(operand_shape) > len(result_shape):
            raise AutodiffError(
                "broadcast_shape_mismatch",
                f"operand shape {operand_shape} has higher rank than result shape {result_shape}",
            )

        leading = len(result_shape) - len(operand_shape)
        axes: list[int] = list(range(leading))

        aligned_result = result_shape[leading:]
        for axis, (result_dim, operand_dim) in enumerate(zip(aligned_result, operand_shape, strict=True), start=leading):
            if operand_dim == result_dim:
                continue
            if operand_dim == 1:
                axes.append(axis)
                continue
            raise AutodiffError(
                "broadcast_shape_mismatch",
                f"operand shape {operand_shape} cannot broadcast to result shape {result_shape}",
            )

        return BroadcastReductionPlan(
            result_shape=result_shape,
            operand_shape=operand_shape,
            axes=tuple(axes),
        )


class AddVjpRule:
    operator_type = AddOperator

    def __init__(self, planner: BroadcastReductionPlanner | None = None) -> None:
        self._planner = planner or BroadcastReductionPlanner()

    def apply(self, context: VjpContext) -> VjpResult:
        if len(context.node.input_value_ids) != 2:
            raise AutodiffError("malformed_derivative_ir", "add VJP requires exactly two inputs")

        lhs_id, rhs_id = context.node.input_value_ids
        result_shape = typespec_shape(context.node.output_typespec)
        gradients: dict[str, str] = {}
        derivative_nodes: list[TensorNodeRecord] = []

        for input_id in (lhs_id, rhs_id):
            if input_id not in context.needed_input_value_ids:
                continue

            operand_typespec = context.value_typespecs.get(input_id)
            operand_shape = typespec_shape(operand_typespec)
            plan = self._planner.plan(result_shape=result_shape, operand_shape=operand_shape)
            if plan.axes:
                gradient_id = context.next_value_id()
                derivative_nodes.append(
                    TensorNodeRecord(
                        node_id=context.next_node_id(),
                        output_value_id=gradient_id,
                        operator=BroadcastReduceOperator(),
                        op_params={
                            "target_shape": list(plan.operand_shape),
                        },
                        input_value_ids=[context.upstream_value_id],
                        output_typespec=operand_typespec,
                    )
                )
                gradients[input_id] = gradient_id
            else:
                gradients[input_id] = context.upstream_value_id

        return VjpResult(gradients=gradients, derivative_nodes=derivative_nodes)


def _swap_last_two_dims(shape: tuple[int, ...]) -> tuple[int, ...]:
    return shape[:-2] + (shape[-1], shape[-2])


def _transpose_last_two_perm(rank: int) -> list[int]:
    return list(range(rank - 2)) + [rank - 1, rank - 2]


class MatmulVjpRule:
    """Build requested matmul VJP branches, reducing broadcast batches."""

    operator_type = MatmulOperator

    def __init__(self, planner: BroadcastReductionPlanner | None = None) -> None:
        self._planner = planner or BroadcastReductionPlanner()

    def apply(self, context: VjpContext) -> VjpResult:
        if len(context.node.input_value_ids) != 2:
            raise AutodiffError("malformed_derivative_ir", "matmul VJP requires exactly two inputs")

        lhs_id, rhs_id = context.node.input_value_ids
        lhs_typespec = context.value_typespecs.get(lhs_id)
        rhs_typespec = context.value_typespecs.get(rhs_id)
        lhs_shape = typespec_shape(lhs_typespec)
        rhs_shape = typespec_shape(rhs_typespec)

        if len(lhs_shape) < 2 or len(rhs_shape) < 2:
            raise AutodiffError("matmul_shape_mismatch", "matmul operands must have rank >= 2")
        if lhs_shape[-1] != rhs_shape[-2]:
            raise AutodiffError(
                "matmul_shape_mismatch",
                f"inner dimensions mismatch: A last dim {lhs_shape[-1]}, B second-to-last dim {rhs_shape[-2]}",
            )

        need_lhs = lhs_id in context.needed_input_value_ids
        need_rhs = rhs_id in context.needed_input_value_ids
        if not need_lhs and not need_rhs:
            return VjpResult(gradients={}, derivative_nodes=[])

        m, k, n = lhs_shape[-2], lhs_shape[-1], rhs_shape[-1]
        result_shape = typespec_shape(context.node.output_typespec)
        batch_z = result_shape[:-2]

        dtype = lhs_typespec.get("dtype") if lhs_typespec else None
        dz_id = context.upstream_value_id
        derivative_nodes: list[TensorNodeRecord] = []
        gradients: dict[str, str] = {}

        if need_lhs:
            # dA = matmul(dZ, B^T)
            b_t_shape = _swap_last_two_dims(rhs_shape)
            b_t_id = context.next_value_id()
            b_t_typespec: dict[str, object] = {"shape": list(b_t_shape)}
            if dtype is not None:
                b_t_typespec["dtype"] = dtype
            derivative_nodes.append(TensorNodeRecord(
                node_id=context.next_node_id(),
                output_value_id=b_t_id,
                operator=TransposeOperator(),
                op_params={"perm": _transpose_last_two_perm(len(rhs_shape))},
                input_value_ids=[rhs_id],
                output_typespec=b_t_typespec,
            ))

            da_shape = batch_z + (m, k)
            da_id = context.next_value_id()
            da_typespec: dict[str, object] = {"shape": list(da_shape)}
            if dtype is not None:
                da_typespec["dtype"] = dtype
            derivative_nodes.append(TensorNodeRecord(
                node_id=context.next_node_id(),
                output_value_id=da_id,
                operator=MatmulOperator(),
                op_params={},
                input_value_ids=[dz_id, b_t_id],
                output_typespec=da_typespec,
            ))

            plan_a = self._planner.plan(result_shape=da_shape, operand_shape=lhs_shape)
            if plan_a.axes:
                da_reduced_id = context.next_value_id()
                derivative_nodes.append(TensorNodeRecord(
                    node_id=context.next_node_id(),
                    output_value_id=da_reduced_id,
                    operator=BroadcastReduceOperator(),
                    op_params={"target_shape": list(lhs_shape)},
                    input_value_ids=[da_id],
                    output_typespec=lhs_typespec,
                ))
                gradients[lhs_id] = da_reduced_id
            else:
                gradients[lhs_id] = da_id

        if need_rhs:
            # dB = matmul(A^T, dZ)
            a_t_shape = _swap_last_two_dims(lhs_shape)
            a_t_id = context.next_value_id()
            a_t_typespec: dict[str, object] = {"shape": list(a_t_shape)}
            if dtype is not None:
                a_t_typespec["dtype"] = dtype
            derivative_nodes.append(TensorNodeRecord(
                node_id=context.next_node_id(),
                output_value_id=a_t_id,
                operator=TransposeOperator(),
                op_params={"perm": _transpose_last_two_perm(len(lhs_shape))},
                input_value_ids=[lhs_id],
                output_typespec=a_t_typespec,
            ))

            db_shape = batch_z + (k, n)
            db_id = context.next_value_id()
            db_typespec: dict[str, object] = {"shape": list(db_shape)}
            if dtype is not None:
                db_typespec["dtype"] = dtype
            derivative_nodes.append(TensorNodeRecord(
                node_id=context.next_node_id(),
                output_value_id=db_id,
                operator=MatmulOperator(),
                op_params={},
                input_value_ids=[a_t_id, dz_id],
                output_typespec=db_typespec,
            ))

            plan_b = self._planner.plan(result_shape=db_shape, operand_shape=rhs_shape)
            if plan_b.axes:
                db_reduced_id = context.next_value_id()
                derivative_nodes.append(TensorNodeRecord(
                    node_id=context.next_node_id(),
                    output_value_id=db_reduced_id,
                    operator=BroadcastReduceOperator(),
                    op_params={"target_shape": list(rhs_shape)},
                    input_value_ids=[db_id],
                    output_typespec=rhs_typespec,
                ))
                gradients[rhs_id] = db_reduced_id
            else:
                gradients[rhs_id] = db_id

        return VjpResult(gradients=gradients, derivative_nodes=derivative_nodes)


def default_vjp_registry() -> VjpRegistry:
    registry = VjpRegistry()
    registry.register(AddVjpRule())
    registry.register(MatmulVjpRule())
    return registry
