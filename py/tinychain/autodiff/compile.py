from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..state import PostOpDef, Scalar, Tensor, id as state_id, tuple_of
from .graph import (
    AddOperator,
    BroadcastReduceOperator,
    DivOperator,
    MatmulOperator,
    MulOperator,
    SubOperator,
    TensorNodeRecord,
    TensorOperator,
    TransposeOperator,
)
from .protocol import AutodiffError
from .reverse import DerivativeProgram


@dataclass(frozen=True)
class CompiledDerivativeProgram:
    """Normal TinyChain route body compiled from a derivative program."""

    params: tuple[str, ...]
    results: tuple[str, ...]
    opdef: PostOpDef


def compile_derivative_program(program: DerivativeProgram) -> CompiledDerivativeProgram:
    """Compile a ``DerivativeProgram`` into deterministic route-shaped IR.

    The generated ``PostOpDef`` uses symbolic route parameters for free input
    values, emits one form entry per derivative node, and returns a tuple whose
    items follow ``program.output_gradients`` order.
    """
    _validate_program(program)

    produced_ids: set[str] = set()
    form: list[tuple[str, Scalar]] = []
    values: dict[str, Scalar] = {}
    free_inputs: list[str] = []

    for node in program.nodes:
        inputs: list[Scalar] = []
        for value_id in node.input_value_ids:
            if value_id not in values:
                values[value_id] = state_id(value_id)
                free_inputs.append(value_id)
            inputs.append(values[value_id])

        output = _compile_node(node, inputs)
        produced_ids.add(node.output_value_id)
        form.append((node.output_value_id, output))
        values[node.output_value_id] = state_id(node.output_value_id)

    result_values: list[Scalar] = []
    result_ids: list[str] = []
    for value_id in program.output_gradients:
        if value_id is None:
            raise _malformed("output gradients cannot contain None")
        result_ids.append(value_id)
        if value_id not in values:
            values[value_id] = state_id(value_id)
            if value_id not in produced_ids:
                free_inputs.append(value_id)
        result_values.append(values[value_id])

    form.append(("result", tuple_of(result_values)))
    return CompiledDerivativeProgram(
        params=tuple(dict.fromkeys(free_inputs)),
        results=tuple(result_ids),
        opdef=PostOpDef(form),
    )


def _compile_node(node: TensorNodeRecord, inputs: list[Scalar]) -> Tensor:
    if isinstance(node.operator, AddOperator):
        _require_arity(node, inputs, 2)
        return Tensor._post_ref(inputs[0]._subject_ref("add"), {"r": inputs[1]})

    if isinstance(node.operator, SubOperator):
        return _compile_binary_tensor_node(node, inputs, "sub")

    if isinstance(node.operator, MulOperator):
        return _compile_binary_tensor_node(node, inputs, "mul")

    if isinstance(node.operator, DivOperator):
        return _compile_binary_tensor_node(node, inputs, "div")

    if isinstance(node.operator, MatmulOperator):
        _require_arity(node, inputs, 2)
        return Tensor._post_ref(inputs[0]._subject_ref("matmul"), {"r": inputs[1]})

    if isinstance(node.operator, TransposeOperator):
        _require_arity(node, inputs, 1)
        permutation = _transpose_permutation(node)
        return Tensor._post_ref(
            inputs[0]._subject_ref("transpose"), {"permutation": permutation}
        )

    if isinstance(node.operator, BroadcastReduceOperator):
        _require_arity(node, inputs, 1)
        target_shape = _required_param(node, "target_shape")
        return Tensor._post_ref(
            inputs[0]._subject_ref("broadcast_reduce"), {"target_shape": target_shape}
        )

    if isinstance(node.operator, TensorOperator):
        raise AutodiffError(
            "malformed_derivative_ir",
            f"unsupported derivative operator {node.operator.route_name!r}",
        )

    raise _malformed("node operator must be a TensorOperator")


def _compile_binary_tensor_node(node: TensorNodeRecord, inputs: list[Scalar], route_name: str) -> Tensor:
    if "right_literal" in node.op_params:
        _require_arity(node, inputs, 1)
        right: object = node.op_params["right_literal"]
    else:
        _require_arity(node, inputs, 2)
        right = inputs[1]
    return Tensor._post_ref(inputs[0]._subject_ref(route_name), {"r": right})


def _validate_program(program: DerivativeProgram) -> None:
    seen_node_ids: set[str] = set()
    seen_outputs: set[str] = set()
    for node in program.nodes:
        _validate_identifier(node.node_id, "node id")
        _validate_identifier(node.output_value_id, "output value id")
        if node.node_id in seen_node_ids:
            raise _malformed(f"duplicate node id {node.node_id!r}")
        if node.output_value_id in seen_outputs:
            raise _malformed(f"duplicate output value id {node.output_value_id!r}")
        seen_node_ids.add(node.node_id)
        seen_outputs.add(node.output_value_id)
        for input_id in node.input_value_ids:
            _validate_identifier(input_id, "input value id")

    if not program.output_gradients:
        raise _malformed("program must declare at least one output gradient")
    for value_id in program.output_gradients:
        if value_id is None:
            raise _malformed("output gradients cannot contain None")
        _validate_identifier(value_id, "output gradient id")


def _validate_identifier(value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise _malformed(f"{label} must be a non-empty string")


def _require_arity(node: TensorNodeRecord, inputs: list[Scalar], expected: int) -> None:
    if len(inputs) != expected:
        raise _malformed(
            f"operator {node.operator.route_name!r} expected {expected} inputs but got {len(inputs)}"
        )


def _required_param(node: TensorNodeRecord, name: str) -> object:
    if not isinstance(node.op_params, Mapping) or name not in node.op_params:
        raise _malformed(f"operator {node.operator.route_name!r} missing param {name!r}")
    return node.op_params[name]


def _transpose_permutation(node: TensorNodeRecord) -> object:
    if not isinstance(node.op_params, Mapping):
        raise _malformed("operator 'transpose' missing param 'perm'")
    if "perm" in node.op_params:
        return node.op_params["perm"]
    if "permutation" in node.op_params:
        return node.op_params["permutation"]
    raise _malformed("operator 'transpose' missing param 'perm'")


def _malformed(message: str) -> AutodiffError:
    return AutodiffError("malformed_derivative_ir", message)
