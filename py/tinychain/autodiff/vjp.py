from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .graph import (
    AddOperator,
    BroadcastOperator,
    BroadcastReduceOperator,
    DivOperator,
    MatmulOperator,
    MaxOperator,
    MeanOperator,
    MinOperator,
    MulOperator,
    ProductOperator,
    ReshapeOperator,
    SubOperator,
    SumOperator,
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


@runtime_checkable
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

    def has_rule(self, operator: TensorOperator | type[TensorOperator]) -> bool:
        """Check if a VJP rule is registered for the given operator."""
        operator_type = operator if isinstance(operator, type) else type(operator)
        return operator_type in self._rules

    def supported_types(self) -> list[type[TensorOperator]]:
        """Return all registered TensorOperator types."""
        return list(self._rules.keys())

    def rule(self, operator_cls: type[TensorOperator]) -> Callable[[type], type]:
        """Class decorator for auto-registering VJP rules.

        Instantiates the decorated class with no arguments, verifies it is a
        VjpRule instance, registers it, and returns the original class unchanged.

        Args:
            operator_cls: The TensorOperator subclass this rule handles.

        Returns:
            The original decorated class (unchanged).

        Raises:
            TypeError: If the instantiated rule is not a VjpRule instance.
        """
        def decorator(rule_cls: type) -> type:
            rule_instance = rule_cls()
            if not isinstance(rule_instance, VjpRule):
                raise TypeError(
                    f"{rule_cls.__name__} is not a VjpRule instance; "
                    f"it must implement the VjpRule protocol"
                )
            self._rules[operator_cls] = rule_instance
            return rule_cls
        return decorator

    def copy_into(self, other: VjpRegistry) -> None:
        """Copy all registered rules from this registry into another.

        Args:
            other: The target VjpRegistry to copy rules into.
        """
        for operator_type, rule in self._rules.items():
            other._rules[operator_type] = rule


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


def _elementwise_binary_node(
    *,
    context: VjpContext,
    operator: TensorOperator,
    input_value_ids: list[str],
    output_typespec: dict[str, object] | None,
    op_params: dict[str, object] | None = None,
) -> TensorNodeRecord:
    return TensorNodeRecord(
        node_id=context.next_node_id(),
        output_value_id=context.next_value_id(),
        operator=operator,
        op_params=dict(op_params or {}),
        input_value_ids=input_value_ids,
        output_typespec=output_typespec,
    )


class _ElementwiseVjpRule:
    operator_type: type[TensorOperator]

    def __init__(self, planner: BroadcastReductionPlanner | None = None) -> None:
        self._planner = planner or BroadcastReductionPlanner()

    def _validate_binary(self, context: VjpContext, name: str) -> tuple[str, str, tuple[int, ...], dict[str, object] | None]:
        if len(context.node.input_value_ids) != 2:
            raise AutodiffError("malformed_derivative_ir", f"{name} VJP requires exactly two inputs")
        lhs_id, rhs_id = context.node.input_value_ids
        result_shape = typespec_shape(context.node.output_typespec)
        result_typespec = context.node.output_typespec
        return lhs_id, rhs_id, result_shape, result_typespec

    def _reduce_to_operand(
        self,
        *,
        context: VjpContext,
        gradient_id: str,
        operand_id: str,
        result_shape: tuple[int, ...],
        derivative_nodes: list[TensorNodeRecord],
    ) -> str:
        operand_typespec = context.value_typespecs.get(operand_id)
        operand_shape = typespec_shape(operand_typespec)
        plan = self._planner.plan(result_shape=result_shape, operand_shape=operand_shape)
        if not plan.axes:
            return gradient_id

        reduced_id = context.next_value_id()
        derivative_nodes.append(
            TensorNodeRecord(
                node_id=context.next_node_id(),
                output_value_id=reduced_id,
                operator=BroadcastReduceOperator(),
                op_params={"target_shape": list(plan.operand_shape)},
                input_value_ids=[gradient_id],
                output_typespec=operand_typespec,
            )
        )
        return reduced_id


class SubVjpRule(_ElementwiseVjpRule):
    operator_type = SubOperator

    def apply(self, context: VjpContext) -> VjpResult:
        lhs_id, rhs_id, result_shape, result_typespec = self._validate_binary(context, "sub")
        gradients: dict[str, str] = {}
        derivative_nodes: list[TensorNodeRecord] = []

        if lhs_id in context.needed_input_value_ids:
            gradients[lhs_id] = self._reduce_to_operand(
                context=context,
                gradient_id=context.upstream_value_id,
                operand_id=lhs_id,
                result_shape=result_shape,
                derivative_nodes=derivative_nodes,
            )

        if rhs_id in context.needed_input_value_ids:
            negated = _elementwise_binary_node(
                context=context,
                operator=MulOperator(),
                input_value_ids=[context.upstream_value_id],
                output_typespec=result_typespec,
                op_params={"right_literal": -1.0},
            )
            derivative_nodes.append(negated)
            gradients[rhs_id] = self._reduce_to_operand(
                context=context,
                gradient_id=negated.output_value_id,
                operand_id=rhs_id,
                result_shape=result_shape,
                derivative_nodes=derivative_nodes,
            )

        return VjpResult(gradients=gradients, derivative_nodes=derivative_nodes)


class MulVjpRule(_ElementwiseVjpRule):
    operator_type = MulOperator

    def apply(self, context: VjpContext) -> VjpResult:
        lhs_id, rhs_id, result_shape, result_typespec = self._validate_binary(context, "mul")
        gradients: dict[str, str] = {}
        derivative_nodes: list[TensorNodeRecord] = []

        if lhs_id in context.needed_input_value_ids:
            lhs_raw = _elementwise_binary_node(
                context=context,
                operator=MulOperator(),
                input_value_ids=[context.upstream_value_id, rhs_id],
                output_typespec=result_typespec,
            )
            derivative_nodes.append(lhs_raw)
            gradients[lhs_id] = self._reduce_to_operand(
                context=context,
                gradient_id=lhs_raw.output_value_id,
                operand_id=lhs_id,
                result_shape=result_shape,
                derivative_nodes=derivative_nodes,
            )

        if rhs_id in context.needed_input_value_ids:
            rhs_raw = _elementwise_binary_node(
                context=context,
                operator=MulOperator(),
                input_value_ids=[context.upstream_value_id, lhs_id],
                output_typespec=result_typespec,
            )
            derivative_nodes.append(rhs_raw)
            gradients[rhs_id] = self._reduce_to_operand(
                context=context,
                gradient_id=rhs_raw.output_value_id,
                operand_id=rhs_id,
                result_shape=result_shape,
                derivative_nodes=derivative_nodes,
            )

        return VjpResult(gradients=gradients, derivative_nodes=derivative_nodes)


class DivVjpRule(_ElementwiseVjpRule):
    operator_type = DivOperator

    def apply(self, context: VjpContext) -> VjpResult:
        lhs_id, rhs_id, result_shape, result_typespec = self._validate_binary(context, "div")
        gradients: dict[str, str] = {}
        derivative_nodes: list[TensorNodeRecord] = []

        if lhs_id in context.needed_input_value_ids:
            lhs_raw = _elementwise_binary_node(
                context=context,
                operator=DivOperator(),
                input_value_ids=[context.upstream_value_id, rhs_id],
                output_typespec=result_typespec,
            )
            derivative_nodes.append(lhs_raw)
            gradients[lhs_id] = self._reduce_to_operand(
                context=context,
                gradient_id=lhs_raw.output_value_id,
                operand_id=lhs_id,
                result_shape=result_shape,
                derivative_nodes=derivative_nodes,
            )

        if rhs_id in context.needed_input_value_ids:
            numerator = _elementwise_binary_node(
                context=context,
                operator=MulOperator(),
                input_value_ids=[context.upstream_value_id, lhs_id],
                output_typespec=result_typespec,
            )
            denominator = _elementwise_binary_node(
                context=context,
                operator=MulOperator(),
                input_value_ids=[rhs_id, rhs_id],
                output_typespec=result_typespec,
            )
            quotient = _elementwise_binary_node(
                context=context,
                operator=DivOperator(),
                input_value_ids=[numerator.output_value_id, denominator.output_value_id],
                output_typespec=result_typespec,
            )
            negated = _elementwise_binary_node(
                context=context,
                operator=MulOperator(),
                input_value_ids=[quotient.output_value_id],
                output_typespec=result_typespec,
                op_params={"right_literal": -1.0},
            )
            derivative_nodes.extend([numerator, denominator, quotient, negated])
            gradients[rhs_id] = self._reduce_to_operand(
                context=context,
                gradient_id=negated.output_value_id,
                operand_id=rhs_id,
                result_shape=result_shape,
                derivative_nodes=derivative_nodes,
            )

        return VjpResult(gradients=gradients, derivative_nodes=derivative_nodes)



def _same_dtype_typespec(reference: dict[str, object] | None, shape: tuple[int, ...]) -> dict[str, object]:
    typespec: dict[str, object] = {"shape": list(shape)}
    if reference is not None and "dtype" in reference:
        typespec["dtype"] = reference["dtype"]
    return typespec


def _normalize_reduction_axes(axes: object, rank: int, route_name: str) -> tuple[int, ...]:
    if axes is None:
        raise AutodiffError(
            "unsupported_reduction",
            f"{route_name} VJP requires explicit axes; axes=None is not supported yet",
        )
    if type(axes) is int:
        raw_axes = [axes]
    elif isinstance(axes, Sequence) and not isinstance(axes, (str, bytes)):
        raw_axes = list(axes)
    else:
        raise AutodiffError("malformed_derivative_ir", f"{route_name} axes must be an int or sequence")

    normalized: list[int] = []
    for axis in raw_axes:
        if type(axis) is not int:
            raise AutodiffError("malformed_derivative_ir", f"{route_name} axes must be integers")
        normalized_axis = axis + rank if axis < 0 else axis
        if normalized_axis < 0 or normalized_axis >= rank:
            raise AutodiffError("reduction_shape_mismatch", f"{route_name} axis {axis} is out of bounds for rank {rank}")
        if normalized_axis in normalized:
            raise AutodiffError("malformed_derivative_ir", f"{route_name} axes cannot contain duplicates")
        normalized.append(normalized_axis)
    return tuple(normalized)


def _reduced_shape(input_shape: tuple[int, ...], axes: tuple[int, ...], keepdims: bool) -> tuple[int, ...]:
    if keepdims:
        return tuple(1 if axis in axes else dim for axis, dim in enumerate(input_shape))
    return tuple(dim for axis, dim in enumerate(input_shape) if axis not in axes)


class _ReductionVjpRule:
    operator_type: type[TensorOperator]
    route_name: str

    def _validate_unary(self, context: VjpContext) -> tuple[str, dict[str, object] | None, tuple[int, ...], tuple[int, ...], bool]:
        if len(context.node.input_value_ids) != 1:
            raise AutodiffError("malformed_derivative_ir", f"{self.route_name} VJP requires exactly one input")
        input_id = context.node.input_value_ids[0]
        input_typespec = context.value_typespecs.get(input_id)
        input_shape = typespec_shape(input_typespec)
        axes = _normalize_reduction_axes(context.node.op_params.get("axes"), len(input_shape), self.route_name)
        keepdims = bool(context.node.op_params.get("keepdims", False))
        expected_output_shape = _reduced_shape(input_shape, axes, keepdims)
        output_shape = typespec_shape(context.node.output_typespec)
        if output_shape != expected_output_shape:
            raise AutodiffError(
                "reduction_shape_mismatch",
                f"{self.route_name} output shape {output_shape} does not match expected {expected_output_shape}",
            )
        return input_id, input_typespec, input_shape, axes, keepdims

    def _expand_upstream(
        self,
        *,
        context: VjpContext,
        input_typespec: dict[str, object] | None,
        input_shape: tuple[int, ...],
        axes: tuple[int, ...],
        keepdims: bool,
    ) -> tuple[str, list[TensorNodeRecord]]:
        derivative_nodes: list[TensorNodeRecord] = []
        gradient_id = context.upstream_value_id
        if not keepdims:
            singleton_shape = tuple(1 if axis in axes else dim for axis, dim in enumerate(input_shape))
            reshape_node = TensorNodeRecord(
                node_id=context.next_node_id(),
                output_value_id=context.next_value_id(),
                operator=ReshapeOperator(),
                op_params={"shape": list(singleton_shape)},
                input_value_ids=[gradient_id],
                output_typespec=_same_dtype_typespec(input_typespec, singleton_shape),
            )
            derivative_nodes.append(reshape_node)
            gradient_id = reshape_node.output_value_id

        broadcast_node = TensorNodeRecord(
            node_id=context.next_node_id(),
            output_value_id=context.next_value_id(),
            operator=BroadcastOperator(),
            op_params={"shape": list(input_shape)},
            input_value_ids=[gradient_id],
            output_typespec=input_typespec,
        )
        derivative_nodes.append(broadcast_node)
        return broadcast_node.output_value_id, derivative_nodes


@_registry.rule(SumOperator)
class SumVjpRule(_ReductionVjpRule):
    operator_type = SumOperator
    route_name = "sum"

    def apply(self, context: VjpContext) -> VjpResult:
        input_id, input_typespec, input_shape, axes, keepdims = self._validate_unary(context)
        if input_id not in context.needed_input_value_ids:
            return VjpResult(gradients={}, derivative_nodes=[])
        gradient_id, derivative_nodes = self._expand_upstream(
            context=context,
            input_typespec=input_typespec,
            input_shape=input_shape,
            axes=axes,
            keepdims=keepdims,
        )
        return VjpResult(gradients={input_id: gradient_id}, derivative_nodes=derivative_nodes)


@_registry.rule(MeanOperator)
class MeanVjpRule(_ReductionVjpRule):
    operator_type = MeanOperator
    route_name = "mean"

    def apply(self, context: VjpContext) -> VjpResult:
        input_id, input_typespec, input_shape, axes, keepdims = self._validate_unary(context)
        if input_id not in context.needed_input_value_ids:
            return VjpResult(gradients={}, derivative_nodes=[])
        gradient_id, derivative_nodes = self._expand_upstream(
            context=context,
            input_typespec=input_typespec,
            input_shape=input_shape,
            axes=axes,
            keepdims=keepdims,
        )
        factor = 1
        for axis in axes:
            factor *= input_shape[axis]
        scaled = TensorNodeRecord(
            node_id=context.next_node_id(),
            output_value_id=context.next_value_id(),
            operator=DivOperator(),
            op_params={"right_literal": float(factor)},
            input_value_ids=[gradient_id],
            output_typespec=input_typespec,
        )
        derivative_nodes.append(scaled)
        return VjpResult(gradients={input_id: scaled.output_value_id}, derivative_nodes=derivative_nodes)


@_registry.rule(ReshapeOperator)
class ReshapeVjpRule:
    operator_type = ReshapeOperator

    def apply(self, context: VjpContext) -> VjpResult:
        if len(context.node.input_value_ids) != 1:
            raise AutodiffError("malformed_derivative_ir", "reshape VJP requires exactly one input")
        input_id = context.node.input_value_ids[0]
        if input_id not in context.needed_input_value_ids:
            return VjpResult(gradients={}, derivative_nodes=[])
        input_typespec = context.value_typespecs.get(input_id)
        input_shape = typespec_shape(input_typespec)
        gradient_id = context.next_value_id()
        gradient_node = TensorNodeRecord(
            node_id=context.next_node_id(),
            output_value_id=gradient_id,
            operator=ReshapeOperator(),
            op_params={"shape": list(input_shape)},
            input_value_ids=[context.upstream_value_id],
            output_typespec=input_typespec,
        )
        return VjpResult(gradients={input_id: gradient_id}, derivative_nodes=[gradient_node])


class _UnsupportedReductionVjpRule:
    operator_type: type[TensorOperator]
    route_name: str
    reason: str

    def apply(self, context: VjpContext) -> VjpResult:
        raise AutodiffError("unsupported_reduction", f"{self.route_name} VJP is unsupported: {self.reason}")


@_registry.rule(MaxOperator)
class MaxVjpRule(_UnsupportedReductionVjpRule):
    operator_type = MaxOperator
    route_name = "max"
    reason = "exact gradients require equality masks and tie handling not expressible with current routes"


@_registry.rule(MinOperator)
class MinVjpRule(_UnsupportedReductionVjpRule):
    operator_type = MinOperator
    route_name = "min"
    reason = "exact gradients require equality masks and tie handling not expressible with current routes"


@_registry.rule(ProductOperator)
class ProductVjpRule(_UnsupportedReductionVjpRule):
    operator_type = ProductOperator
    route_name = "product"
    reason = "zero-safe product gradients require masking/counting primitives not expressible with current routes"

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


def _validate_permutation(perm: object, rank: int) -> tuple[int, ...]:
    if not isinstance(perm, Sequence) or isinstance(perm, (str, bytes)):
        raise AutodiffError("invalid_permutation", "transpose permutation must be a sequence of axes")

    axes: list[int] = []
    for axis in perm:
        if type(axis) is not int:
            raise AutodiffError("invalid_permutation", "transpose permutation axes must be integers")
        axes.append(axis)

    if len(axes) != rank:
        raise AutodiffError(
            "invalid_permutation",
            f"transpose permutation length {len(axes)} does not match rank {rank}",
        )
    if sorted(axes) != list(range(rank)):
        raise AutodiffError(
            "invalid_permutation",
            f"transpose permutation must contain each axis 0..{rank - 1} exactly once",
        )
    return tuple(axes)


def _inverse_permutation(perm: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(perm)
    for index, axis in enumerate(perm):
        inverse[axis] = index
    return tuple(inverse)


class TransposeVjpRule:
    operator_type = TransposeOperator

    def apply(self, context: VjpContext) -> VjpResult:
        if len(context.node.input_value_ids) != 1:
            raise AutodiffError("malformed_derivative_ir", "transpose VJP requires exactly one input")

        input_id = context.node.input_value_ids[0]
        input_typespec = context.value_typespecs.get(input_id)
        input_shape = typespec_shape(input_typespec)
        perm = _validate_permutation(context.node.op_params.get("perm"), len(input_shape))
        inverse_perm = _inverse_permutation(perm)

        gradient_id = context.next_value_id()
        gradient_node = TensorNodeRecord(
            node_id=context.next_node_id(),
            output_value_id=gradient_id,
            operator=TransposeOperator(),
            op_params={"perm": list(inverse_perm)},
            input_value_ids=[context.upstream_value_id],
            output_typespec=input_typespec,
        )
        return VjpResult(gradients={input_id: gradient_id}, derivative_nodes=[gradient_node])


def default_vjp_registry() -> VjpRegistry:
    """Return the default VJP registry with the built-in transform rules."""
    registry = VjpRegistry()
    for rule in (
        AddVjpRule(),
        SubVjpRule(),
        MulVjpRule(),
        DivVjpRule(),
        MatmulVjpRule(),
        TransposeVjpRule(),
    ):
        registry.register(rule)
    return registry
