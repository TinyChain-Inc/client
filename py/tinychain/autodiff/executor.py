from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .protocol import AutodiffResult
from .reverse import DerivativeProgram


RouteDispatcher = Callable[[str, dict[str, object], list[object]], object]


@dataclass(frozen=True)
class ExecutionScheduler:
    dispatch: RouteDispatcher

    def execute(
        self,
        program: DerivativeProgram,
        *,
        values: dict[str, object],
    ) -> AutodiffResult:
        environment = dict(values)
        for node in program.nodes:
            args = [environment[value_id] for value_id in node.input_value_ids]
            environment[node.output_value_id] = self.dispatch(
                node.op_kind,
                dict(node.op_params),
                args,
            )

        gradients = [
            environment[gradient_id] if gradient_id is not None else None
            for gradient_id in program.output_gradients
        ]
        return AutodiffResult(gradients=gradients, metadata=program.metadata)
