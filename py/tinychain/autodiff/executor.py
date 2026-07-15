from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from .graph import TensorNodeRecord
from .protocol import AutodiffError, AutodiffResult
from .shape import (
    bind_compatible_shapes,
    resolve_shape_value,
    shape_from_value,
    typespec_ranked_shape,
)
from .reverse import DerivativeProgram


NodeDispatcher = Callable[[TensorNodeRecord, list[object]], object]
RouteExecutor = Callable[[Mapping[str, object]], object]


@dataclass(frozen=True)
class ExecutionScheduler:
    """Execute a generated derivative program through an injected node dispatcher.

    Phase 1 uses this as the transport seam: tests inject a NumPy dispatcher,
    while production tc-server transport is deferred until shared client
    transport and tensor codec support tensor operation calls.
    """

    dispatch: NodeDispatcher

    def execute(
        self,
        program: DerivativeProgram,
        *,
        values: dict[str, object],
        shape_bindings: Mapping[str, int] | None = None,
    ) -> AutodiffResult:
        environment = dict(values)
        bindings = _runtime_shape_bindings(program, environment, shape_bindings)
        for node in program.nodes:
            args = []
            for value_id in node.input_value_ids:
                if value_id not in environment:
                    raise AutodiffError(
                        "missing_derivative_ir",
                        f"missing input value {value_id!r} for node {node.node_id!r}",
                    )
                args.append(environment[value_id])
            resolved_node = _resolve_node_shape_params(node, bindings)
            result = self.dispatch(resolved_node, args)
            environment[node.output_value_id] = result
            _bind_runtime_value_shape(
                node.output_value_id,
                node.output_typespec,
                result,
                bindings=bindings,
            )

        gradients = []
        for gradient_id in program.output_gradients:
            if gradient_id is None:
                gradients.append(None)
            elif gradient_id not in environment:
                raise AutodiffError(
                    "missing_derivative_ir",
                    f"missing output-gradient value {gradient_id!r}",
                )
            else:
                gradients.append(environment[gradient_id])
        return AutodiffResult(gradients=gradients, metadata=program.metadata)


@dataclass(slots=True)
class DerivativeExecutionDispatcher:
    """Execute a derivative program through an injected route executor."""

    route_executor: RouteExecutor
    params: tuple[str, ...]
    shape_params: Mapping[str, str] | None = None

    def execute(
        self,
        program: DerivativeProgram,
        *,
        values: Mapping[str, object],
        shape_bindings: Mapping[str, int] | None = None,
    ) -> AutodiffResult:
        bindings = _validate_program_shapes_resolved(program, values, shape_bindings)
        shape_param_symbols = dict(self.shape_params or {})
        missing = [
            param for param in self.params
            if param not in values
            and not _shape_param_is_bound(param, shape_param_symbols, bindings)
        ]
        if missing:
            joined = ", ".join(repr(param) for param in missing)
            raise AutodiffError(
                "missing_derivative_ir",
                f"missing derivative execution input(s): {joined}",
            )

        call_values = {
            param: _route_param_value(param, values, bindings, shape_param_symbols)
            for param in self.params
        }
        try:
            gradients = self.route_executor(call_values)
        except AutodiffError:
            raise
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            raise AutodiffError("missing_derivative_ir", str(exc)) from exc

        if not isinstance(gradients, list):
            gradients = [gradients]
        return AutodiffResult(gradients=gradients, metadata=program.metadata)



def _shape_param_is_bound(
    param: str,
    shape_param_symbols: Mapping[str, str],
    bindings: Mapping[str, int],
) -> bool:
    symbol = shape_param_symbols.get(param)
    return symbol is not None and symbol in bindings


def _route_param_value(
    param: str,
    values: Mapping[str, object],
    bindings: Mapping[str, int],
    shape_param_symbols: Mapping[str, str],
) -> object:
    symbol = shape_param_symbols.get(param)
    if symbol is not None:
        return bindings[symbol]
    return values[param]


def _runtime_shape_bindings(
    program: DerivativeProgram,
    values: Mapping[str, object],
    shape_bindings: Mapping[str, int] | None,
) -> dict[str, int]:
    bindings = dict(shape_bindings or {})
    value_typespecs = _program_value_typespecs(program)
    for value_id, value in values.items():
        _bind_runtime_value_shape(
            value_id,
            value_typespecs.get(value_id),
            value,
            bindings=bindings,
        )
    return bindings


def _program_value_typespecs(program: DerivativeProgram) -> dict[str, dict[str, object]]:
    value_typespecs = {
        value_id: dict(typespec)
        for value_id, typespec in getattr(program, "value_typespecs", {}).items()
    }
    for node in program.nodes:
        if node.output_typespec is not None:
            value_typespecs[node.output_value_id] = dict(node.output_typespec)
    return value_typespecs


def _bind_runtime_value_shape(
    value_id: str,
    typespec: dict[str, object] | None,
    value: object,
    *,
    bindings: dict[str, int],
) -> None:
    if typespec is None:
        return
    concrete_shape = shape_from_value(value)
    if concrete_shape is None:
        return
    bind_compatible_shapes(
        symbolic_shape=typespec_ranked_shape(typespec),
        concrete_shape=concrete_shape,
        bindings=bindings,
        label=f"runtime value {value_id!r}",
    )


def _resolve_node_shape_params(
    node: TensorNodeRecord,
    bindings: Mapping[str, int],
) -> TensorNodeRecord:
    op_params = dict(node.op_params)
    for name in ("shape", "target_shape"):
        if name in op_params:
            op_params[name] = resolve_shape_value(
                op_params[name],
                bindings,
                label=f"operator {node.operator.route_name!r} param {name!r}",
            )
    if op_params == node.op_params:
        return node
    return replace(node, op_params=op_params)


def _validate_program_shapes_resolved(
    program: DerivativeProgram,
    values: Mapping[str, object],
    shape_bindings: Mapping[str, int] | None,
) -> dict[str, int]:
    bindings = _runtime_shape_bindings(program, values, shape_bindings)
    for node in program.nodes:
        _resolve_node_shape_params(node, bindings)
    return bindings
