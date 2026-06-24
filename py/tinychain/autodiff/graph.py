from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TensorOperator:
    """Base tensor operator descriptor used by Python-owned autodiff graphs."""

    route_name: str


@dataclass(frozen=True)
class AddOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "add")


@dataclass(frozen=True)
class BroadcastReduceOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "broadcast_reduce")


@dataclass(frozen=True)
class MatmulOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "matmul")


@dataclass(frozen=True)
class TransposeOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "transpose")


OP_ADD: AddOperator = AddOperator()
OP_BROADCAST_REDUCE: BroadcastReduceOperator = BroadcastReduceOperator()
OP_MATMUL: MatmulOperator = MatmulOperator()
OP_TRANSPOSE: TransposeOperator = TransposeOperator()
_OPERATORS_BY_ROUTE: dict[str, TensorOperator] = {
    OP_ADD.route_name: OP_ADD,
    OP_BROADCAST_REDUCE.route_name: OP_BROADCAST_REDUCE,
    OP_MATMUL.route_name: OP_MATMUL,
    OP_TRANSPOSE.route_name: OP_TRANSPOSE,
}


def operator_for_route(route_name: str) -> TensorOperator:
    return _OPERATORS_BY_ROUTE[route_name]

_active_builder: contextvars.ContextVar[Optional[TensorGraphBuilder]] = contextvars.ContextVar(
    "_active_builder", default=None
)


def get_active_builder() -> Optional[TensorGraphBuilder]:
    return _active_builder.get()


@dataclass(frozen=True, init=False)
class TensorNodeRecord:
    """Single recorded computation step captured by TensorGraphBuilder."""

    node_id: str
    output_value_id: str
    operator: TensorOperator
    op_params: dict
    input_value_ids: list[str]
    output_typespec: Optional[dict] = None

    def __init__(
        self,
        *,
        node_id: str,
        output_value_id: str,
        operator: TensorOperator | str | None = None,
        op_params: dict,
        input_value_ids: list[str],
        output_typespec: Optional[dict] = None,
        op_kind: TensorOperator | str | None = None,
    ) -> None:
        selected = operator if operator is not None else op_kind
        if selected is None:
            raise TypeError("TensorNodeRecord requires an operator")
        if isinstance(selected, str):
            selected = operator_for_route(selected)

        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "output_value_id", output_value_id)
        object.__setattr__(self, "operator", selected)
        object.__setattr__(self, "op_params", op_params)
        object.__setattr__(self, "input_value_ids", input_value_ids)
        object.__setattr__(self, "output_typespec", output_typespec)

    @property
    def op_kind(self) -> str:
        return self.operator.route_name


@dataclass(frozen=True)
class TensorGraph:
    """Python-owned derivative program representation for Phase 1 autodiff graph construction."""

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
