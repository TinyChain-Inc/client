from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .graph import TensorNodeRecord
from .protocol import AutodiffResult
from .reverse import DerivativeProgram


RouteDispatcher = Callable[[TensorNodeRecord, list[object]], object]


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
            environment[node.output_value_id] = self.dispatch(node, args)

        gradients = [
            environment[gradient_id] if gradient_id is not None else None
            for gradient_id in program.output_gradients
        ]
        return AutodiffResult(gradients=gradients, metadata=program.metadata)
