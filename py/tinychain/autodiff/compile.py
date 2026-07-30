from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..collection.tensor import Tensor
from ..state import PostOpDef, Scalar, id as state_id, tuple_of
from ..state._ops import subject_of
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
from .shape import parse_shape, resolve_shape, resolve_shape_value
from .reverse import DerivativeProgram


@dataclass(frozen=True)
class CompiledDerivativeProgram:
    """Normal TinyChain route body compiled from a derivative program."""

    params: tuple[str, ...]
    results: tuple[str, ...]
    opdef: PostOpDef
    shape_params: dict[str, str] | None = None


def compile_derivative_program(
    program: DerivativeProgram,
    *,
    symbol_bindings: Mapping[str, int] | None = None,
    defer_symbolic_shape_params: bool = False,
) -> CompiledDerivativeProgram:
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
    reserved_params = set(_free_input_ids(program))
    shape_params: list[str] = []
    shape_param_symbols: dict[str, str] = {}
    shape_symbol_params: dict[str, str] = {}

    for node in program.nodes:
        inputs: list[Scalar] = []
        for value_id in node.input_value_ids:
            if value_id not in values:
                values[value_id] = state_id(value_id)
                free_inputs.append(value_id)
            inputs.append(values[value_id])

        output = _compile_node(
            node,
            inputs,
            symbol_bindings=symbol_bindings,
            defer_symbolic_shape_params=defer_symbolic_shape_params,
            shape_params=shape_params,
            shape_param_symbols=shape_param_symbols,
            shape_symbol_params=shape_symbol_params,
            used_params=reserved_params,
        )
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
        params=tuple(dict.fromkeys([*free_inputs, *shape_params])),
        results=tuple(result_ids),
        opdef=PostOpDef(form),
        shape_params=dict(shape_param_symbols) or None,
    )



def _free_input_ids(program: DerivativeProgram) -> tuple[str, ...]:
    produced_ids: set[str] = set()
    values: set[str] = set()
    free_inputs: list[str] = []

    for node in program.nodes:
        for value_id in node.input_value_ids:
            if value_id not in values:
                values.add(value_id)
                free_inputs.append(value_id)
        produced_ids.add(node.output_value_id)
        values.add(node.output_value_id)

    for value_id in program.output_gradients:
        if value_id is None:
            continue
        if value_id not in values:
            values.add(value_id)
            if value_id not in produced_ids:
                free_inputs.append(value_id)

    return tuple(dict.fromkeys(free_inputs))


def _compile_node(
    node: TensorNodeRecord,
    inputs: list[Scalar],
    *,
    symbol_bindings: Mapping[str, int] | None,
    defer_symbolic_shape_params: bool,
    shape_params: list[str],
    shape_param_symbols: dict[str, str],
    shape_symbol_params: dict[str, str],
    used_params: set[str],
) -> Tensor:
    if isinstance(node.operator, AddOperator):
        _require_arity(node, inputs, 2)
        return Tensor._post_ref(f"{subject_of(inputs[0])}/add", {"r": inputs[1]})

    if isinstance(node.operator, SubOperator):
        return _compile_binary_tensor_node(node, inputs, "sub")

    if isinstance(node.operator, MulOperator):
        return _compile_binary_tensor_node(node, inputs, "mul")

    if isinstance(node.operator, DivOperator):
        return _compile_binary_tensor_node(node, inputs, "div")

    if isinstance(node.operator, MatmulOperator):
        _require_arity(node, inputs, 2)
        return Tensor._post_ref(f"{subject_of(inputs[0])}/matmul", {"r": inputs[1]})

    if isinstance(node.operator, SumOperator):
        return _compile_reduction_node(node, inputs, "sum")

    if isinstance(node.operator, MeanOperator):
        return _compile_reduction_node(node, inputs, "mean")

    if isinstance(node.operator, MaxOperator):
        return _compile_reduction_node(node, inputs, "max")

    if isinstance(node.operator, MinOperator):
        return _compile_reduction_node(node, inputs, "min")

    if isinstance(node.operator, ProductOperator):
        return _compile_reduction_node(node, inputs, "product")

    if isinstance(node.operator, ReshapeOperator):
        _require_arity(node, inputs, 1)
        shape = _shape_param(
            node,
            "shape",
            symbol_bindings,
            defer_symbolic_shape_params=defer_symbolic_shape_params,
            shape_params=shape_params,
            shape_param_symbols=shape_param_symbols,
            shape_symbol_params=shape_symbol_params,
            used_params=used_params,
        )
        return Tensor._post_ref(f"{subject_of(inputs[0])}/reshape", {"shape": shape})

    if isinstance(node.operator, BroadcastOperator):
        _require_arity(node, inputs, 1)
        shape = _shape_param(
            node,
            "shape",
            symbol_bindings,
            defer_symbolic_shape_params=defer_symbolic_shape_params,
            shape_params=shape_params,
            shape_param_symbols=shape_param_symbols,
            shape_symbol_params=shape_symbol_params,
            used_params=used_params,
        )
        return Tensor._post_ref(f"{subject_of(inputs[0])}/broadcast", {"shape": shape})

    if isinstance(node.operator, TransposeOperator):
        _require_arity(node, inputs, 1)
        permutation = _transpose_permutation(node)
        return Tensor._post_ref(
            f"{subject_of(inputs[0])}/transpose", {"permutation": permutation}
        )

    if isinstance(node.operator, BroadcastReduceOperator):
        _require_arity(node, inputs, 1)
        target_shape = _shape_param(
            node,
            "target_shape",
            symbol_bindings,
            defer_symbolic_shape_params=defer_symbolic_shape_params,
            shape_params=shape_params,
            shape_param_symbols=shape_param_symbols,
            shape_symbol_params=shape_symbol_params,
            used_params=used_params,
        )
        return Tensor._post_ref(
            f"{subject_of(inputs[0])}/broadcast_reduce", {"target_shape": target_shape}
        )

    if isinstance(node.operator, TensorOperator):
        raise AutodiffError(
            "malformed_derivative_ir",
            f"unsupported derivative operator {node.operator.route_name!r}",
        )

    raise _malformed("node operator must be a TensorOperator")


def _compile_reduction_node(node: TensorNodeRecord, inputs: list[Scalar], route_name: str) -> Tensor:
    _require_arity(node, inputs, 1)
    axes = _required_param(node, "axes")
    keepdims = _required_param(node, "keepdims")
    return Tensor._post_ref(
        f"{subject_of(inputs[0])}/{route_name}",
        {"axes": axes, "keepdims": keepdims},
    )


def _compile_binary_tensor_node(node: TensorNodeRecord, inputs: list[Scalar], route_name: str) -> Tensor:
    if "right_literal" in node.op_params:
        _require_arity(node, inputs, 1)
        right: object = node.op_params["right_literal"]
    else:
        _require_arity(node, inputs, 2)
        right = inputs[1]
    return Tensor._post_ref(f"{subject_of(inputs[0])}/{route_name}", {"r": right})


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


def _shape_param(
    node: TensorNodeRecord,
    name: str,
    symbol_bindings: Mapping[str, int] | None,
    *,
    defer_symbolic_shape_params: bool,
    shape_params: list[str],
    shape_param_symbols: dict[str, str],
    shape_symbol_params: dict[str, str],
    used_params: set[str],
) -> list[object]:
    value = _required_param(node, name)
    label = f"operator {node.operator.route_name!r} param {name!r}"
    if defer_symbolic_shape_params:
        return _deferred_shape_value(
            value,
            symbol_bindings,
            shape_params=shape_params,
            shape_param_symbols=shape_param_symbols,
            shape_symbol_params=shape_symbol_params,
            used_params=used_params,
            label=label,
        )
    return resolve_shape_value(value, symbol_bindings, label=label)


def _deferred_shape_value(
    value: object,
    bindings: Mapping[str, int] | None,
    *,
    shape_params: list[str],
    shape_param_symbols: dict[str, str],
    shape_symbol_params: dict[str, str],
    used_params: set[str],
    label: str,
) -> list[object]:
    shape: list[object] = []
    binding_map = dict(bindings or {})
    for dim in parse_shape(value, label=label):
        if isinstance(dim, int):
            shape.append(dim)
            continue
        if dim in binding_map:
            shape.append(resolve_shape((dim,), binding_map, label=label)[0])
            continue
        param = _shape_symbol_param(
            dim,
            shape_params=shape_params,
            shape_param_symbols=shape_param_symbols,
            shape_symbol_params=shape_symbol_params,
            used_params=used_params,
        )
        shape.append(state_id(param))
    return shape


def _shape_symbol_param(
    symbol: str,
    *,
    shape_params: list[str],
    shape_param_symbols: dict[str, str],
    shape_symbol_params: dict[str, str],
    used_params: set[str],
) -> str:
    existing = shape_symbol_params.get(symbol)
    if existing is not None:
        return existing

    base = f"__tc_shape_{symbol}"
    param = base
    suffix = 1
    unavailable = used_params | set(shape_param_symbols)
    while param in unavailable:
        param = f"{base}_{suffix}"
        suffix += 1

    shape_symbol_params[symbol] = param
    shape_param_symbols[param] = symbol
    shape_params.append(param)
    return param


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
