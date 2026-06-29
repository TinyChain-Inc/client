from __future__ import annotations

from dataclasses import dataclass, field

from .graph import AddOperator, BroadcastReduceOperator, TensorNodeRecord
from .protocol import AutodiffError
from .seed import typespec_shape
from .vjp import BroadcastReductionPlanner


@dataclass
class GradientAccumulator:
    value_typespecs: dict[str, dict[str, object]]
    _contributions: dict[str, list[str]] = field(default_factory=dict)
    _planner: BroadcastReductionPlanner = field(default_factory=BroadcastReductionPlanner)

    def add(self, value_id: str, gradient_value_id: str) -> None:
        self._contributions.setdefault(value_id, []).append(gradient_value_id)

    def result_for(self, value_id: str, *, next_value_id=None, next_node_id=None) -> tuple[str | None, list[TensorNodeRecord]]:
        contributions = self._contributions.get(value_id, [])
        if not contributions:
            return None, []

        target_typespec = self.value_typespecs.get(value_id)
        target_shape = typespec_shape(target_typespec)
        normalized: list[str] = []
        nodes: list[TensorNodeRecord] = []

        for contribution in sorted(contributions):
            normalized_id, reduction_nodes = self._normalize_contribution(
                contribution,
                target_typespec=target_typespec,
                target_shape=target_shape,
                next_value_id=next_value_id,
                next_node_id=next_node_id,
            )
            nodes.extend(reduction_nodes)
            normalized.append(normalized_id)

        if len(normalized) == 1:
            return normalized[0], nodes

        if next_value_id is None or next_node_id is None:
            raise AutodiffError(
                "missing_derivative_ir",
                "multiple gradient contributions require derivative node id generators",
            )

        current = normalized[0]
        for contribution in normalized[1:]:
            previous = current
            current = next_value_id()
            self._record_typespec(current, target_typespec)
            nodes.append(
                TensorNodeRecord(
                    node_id=next_node_id(),
                    output_value_id=current,
                    operator=AddOperator(),
                    op_params={},
                    input_value_ids=[previous, contribution],
                    output_typespec=target_typespec,
                )
            )

        return current, nodes

    def _normalize_contribution(
        self,
        contribution: str,
        *,
        target_typespec: dict[str, object] | None,
        target_shape: tuple[int, ...],
        next_value_id,
        next_node_id,
    ) -> tuple[str, list[TensorNodeRecord]]:
        contribution_typespec = self.value_typespecs.get(contribution)
        if contribution_typespec is None:
            return contribution, []

        contribution_shape = typespec_shape(contribution_typespec)
        if contribution_shape == target_shape:
            return contribution, []

        if next_value_id is None or next_node_id is None:
            raise AutodiffError(
                "missing_derivative_ir",
                "broadcast gradient contributions require derivative node id generators",
            )

        self._planner.plan(result_shape=contribution_shape, operand_shape=target_shape)
        reduced_id = next_value_id()
        self._record_typespec(reduced_id, target_typespec)
        return reduced_id, [
            TensorNodeRecord(
                node_id=next_node_id(),
                output_value_id=reduced_id,
                operator=BroadcastReduceOperator(),
                op_params={"target_shape": list(target_shape)},
                input_value_ids=[contribution],
                output_typespec=target_typespec,
            )
        ]

    def _record_typespec(self, value_id: str, typespec: dict[str, object] | None) -> None:
        if typespec is not None:
            self.value_typespecs[value_id] = dict(typespec)
