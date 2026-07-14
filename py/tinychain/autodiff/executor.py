from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .graph import TensorNodeRecord
from .protocol import AutodiffError, AutodiffResult
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
    ) -> AutodiffResult:
        environment = dict(values)
        for node in program.nodes:
            args = []
            for value_id in node.input_value_ids:
                if value_id not in environment:
                    raise AutodiffError(
                        "missing_derivative_ir",
                        f"missing input value {value_id!r} for node {node.node_id!r}",
                    )
                args.append(environment[value_id])
            environment[node.output_value_id] = self.dispatch(node, args)

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

    def execute(
        self,
        program: DerivativeProgram,
        *,
        values: Mapping[str, object],
    ) -> AutodiffResult:
        missing = [param for param in self.params if param not in values]
        if missing:
            joined = ", ".join(repr(param) for param in missing)
            raise AutodiffError(
                "missing_derivative_ir",
                f"missing derivative execution input(s): {joined}",
            )

        call_values = {param: values[param] for param in self.params}
        try:
            gradients = self.route_executor(call_values)
        except AutodiffError:
            raise
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            raise AutodiffError("missing_derivative_ir", str(exc)) from exc

        if not isinstance(gradients, list):
            gradients = [gradients]
        return AutodiffResult(gradients=gradients, metadata=program.metadata)

