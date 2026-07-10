from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

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
    """Execute a derivative program through one installed TinyChain route call."""

    library_cls: type
    kernel: object
    token: object
    data_dir: object | None = None
    route_name: str | None = None
    _is_installed: bool = field(default=False, init=False, repr=False)

    def execute(
        self,
        program: DerivativeProgram,
        *,
        values: Mapping[str, object],
        shape_bindings: Mapping[str, int] | None = None,
    ) -> AutodiffResult:
        route_name = self.route_name or getattr(
            self.library_cls, "__tc_derivative_route_name__", "execute"
        )
        params = tuple(getattr(self.library_cls, "__tc_derivative_params__", ()))
        _validate_program_shapes_resolved(program, values, shape_bindings)
        missing = [param for param in params if param not in values]
        if missing:
            joined = ", ".join(repr(param) for param in missing)
            raise AutodiffError(
                "missing_derivative_ir",
                f"missing derivative execution input(s): {joined}",
            )

        self._install_once()
        library = self.library_cls()
        route = getattr(library, route_name)
        call_values = {param: values[param] for param in params}
        try:
            import tinychain as tc

            with tc.backend(self.kernel):
                gradients = route(**call_values)
        except AutodiffError:
            raise
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            raise AutodiffError("missing_derivative_ir", str(exc)) from exc

        if not isinstance(gradients, list):
            gradients = [gradients]
        return AutodiffResult(gradients=gradients, metadata=program.metadata)

    def _install_once(self) -> None:
        if self._is_installed:
            return

        try:
            import tinychain as tc

            response = tc.install(
                self.library_cls,
                kernel=self.kernel,
                data_dir=self.data_dir,
                token=self.token,
            )
        except AutodiffError:
            raise
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            raise AutodiffError("missing_derivative_ir", str(exc)) from exc

        status = getattr(response, "status", None)
        if status not in (None, 200, 204):
            raise AutodiffError(
                "missing_derivative_ir",
                f"derivative execution library install failed with status {status}",
            )
        self._is_installed = True



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
) -> None:
    bindings = _runtime_shape_bindings(program, values, shape_bindings)
    for node in program.nodes:
        _resolve_node_shape_params(node, bindings)
