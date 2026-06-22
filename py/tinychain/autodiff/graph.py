from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Optional

OP_ADD: str = "add"
OP_BROADCAST_REDUCE: str = "broadcast_reduce"
OP_MATMUL: str = "matmul"
OP_TRANSPOSE: str = "transpose"

_active_builder: contextvars.ContextVar[Optional[TensorGraphBuilder]] = contextvars.ContextVar(
    "_active_builder", default=None
)


def get_active_builder() -> Optional[TensorGraphBuilder]:
    return _active_builder.get()


@dataclass(frozen=True)
class TensorNodeRecord:
    """Single recorded computation step captured by TensorGraphBuilder."""

    node_id: str
    output_value_id: str
    op_kind: str
    op_params: dict
    input_value_ids: list[str]
    output_typespec: Optional[dict] = None


@dataclass(frozen=True)
class TensorGraph:
    """Python-native mirror of tc-ir TensorGraph; Phase 1 authoring/transform representation."""

    nodes: list[TensorNodeRecord]
    inputs: list[tuple[str, Optional[dict]]]
    outputs: list[str]


class TensorGraphBuilder:
    """Context manager that records tensor operations as TensorNodeRecord entries.

    Usage::

        with TensorGraphBuilder() as builder:
            z = x + y
        graph = builder.build()
    """

    def __init__(self) -> None:
        self._nodes: list[TensorNodeRecord] = []
        self._value_map: dict[int, str] = {}
        self._token: Optional[contextvars.Token[Optional[TensorGraphBuilder]]] = None

    def _next_value_id(self) -> str:
        return f"v{len(self._value_map)}"

    def _next_node_id(self) -> str:
        return f"n{len(self._nodes)}"

    def register_value(self, obj: object, value_id: Optional[str] = None) -> str:
        """Return the ValueId for *obj*, registering it with a fresh id if not seen before."""
        key = id(obj)
        if key not in self._value_map:
            vid = value_id if value_id is not None else self._next_value_id()
            self._value_map[key] = vid
        return self._value_map[key]

    def record(self, node: TensorNodeRecord) -> None:
        self._nodes.append(node)

    def build(self) -> TensorGraph:
        """Assemble and return the recorded TensorGraph."""
        produced: set[str] = {rec.output_value_id for rec in self._nodes}
        seen: set[str] = set()
        input_value_ids: list[str] = []
        for rec in self._nodes:
            for vid in rec.input_value_ids:
                if vid not in produced and vid not in seen:
                    input_value_ids.append(vid)
                    seen.add(vid)
        inputs = [(vid, None) for vid in input_value_ids]
        outputs = [self._nodes[-1].output_value_id] if self._nodes else []
        return TensorGraph(nodes=list(self._nodes), inputs=inputs, outputs=outputs)

    def __enter__(self) -> TensorGraphBuilder:
        self._token = _active_builder.set(self)
        return self

    def __exit__(self, *args: object) -> None:
        if self._token is not None:
            _active_builder.reset(self._token)
            self._token = None
