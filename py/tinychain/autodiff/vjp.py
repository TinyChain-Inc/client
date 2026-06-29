from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .graph import AddOperator, BroadcastReduceOperator, TensorNodeRecord, TensorOperator
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


def default_vjp_registry() -> VjpRegistry:
    registry = VjpRegistry()
    registry.register(AddVjpRule())
    return registry
