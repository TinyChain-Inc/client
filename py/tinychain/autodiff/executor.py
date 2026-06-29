from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .graph import TensorNodeRecord
from .protocol import AutodiffError, AutodiffResult
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
