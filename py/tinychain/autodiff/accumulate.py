from __future__ import annotations

from dataclasses import dataclass, field

from .graph import AddOperator, TensorNodeRecord
from .protocol import AutodiffError
from .seed import typespec_shape


@dataclass
class GradientAccumulator:
    value_typespecs: dict[str, dict[str, object]]
    _contributions: dict[str, list[str]] = field(default_factory=dict)

    def add(self, value_id: str, gradient_value_id: str) -> None:
        self._contributions.setdefault(value_id, []).append(gradient_value_id)

    def result_for(self, value_id: str, *, next_value_id=None, next_node_id=None) -> tuple[str | None, list[TensorNodeRecord]]:
        contributions = self._contributions.get(value_id, [])
        if not contributions:
            return None, []
        if len(contributions) == 1:
            self._validate_shape(value_id)
            return contributions[0], []

        if next_value_id is None or next_node_id is None:
            raise AutodiffError(
                "missing_derivative_ir",
                "multiple gradient contributions require derivative node id generators",
            )

        nodes: list[TensorNodeRecord] = []
        current = contributions[0]
        for contribution in contributions[1:]:
            previous = current
            current = next_value_id()
            nodes.append(
                TensorNodeRecord(
                    node_id=next_node_id(),
                    output_value_id=current,
                    operator=AddOperator(),
                    op_params={},
                    input_value_ids=[previous, contribution],
                    output_typespec=self.value_typespecs.get(value_id),
                )
            )

        self._validate_shape(value_id)
        return current, nodes

    def _validate_shape(self, value_id: str) -> None:
        typespec_shape(self.value_typespecs.get(value_id))
