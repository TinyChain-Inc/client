from __future__ import annotations

import contextvars
import keyword
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from ..serialize import serialize

# Differentiable dtypes accepted by TensorGraphBuilder.input(...)
# (spec §7.3.4; https://github.com/TinyChain-Inc/client/issues/95).
_INPUT_DTYPES: tuple[str, ...] = ("f32", "f64")


@dataclass(frozen=True)
class TensorOperator:
    """Base tensor operator descriptor used by Python-owned autodiff graphs."""

    route_name: str

    def __serialize__(self) -> dict:
        return {"type": type(self).__name__, "route_name": self.route_name}


@dataclass(frozen=True)
class AddOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "add")


@dataclass(frozen=True)
class SubOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "sub")


@dataclass(frozen=True)
class MulOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "mul")


@dataclass(frozen=True)
class DivOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "div")


@dataclass(frozen=True)
class SumOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "sum")


@dataclass(frozen=True)
class MeanOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "mean")


@dataclass(frozen=True)
class MaxOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "max")


@dataclass(frozen=True)
class MinOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "min")


@dataclass(frozen=True)
class ProductOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "product")


@dataclass(frozen=True)
class ReshapeOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "reshape")


@dataclass(frozen=True)
class BroadcastOperator(TensorOperator):
    def __init__(self) -> None:
        object.__setattr__(self, "route_name", "broadcast")


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


# ContextVar-based active builder tracking (Decision D6 from client-issue-13-phase2 spec §5).
# Rationale: Task-scoped by stdlib design, correct for tc.grad's single call-site transform use case.
# Alternative thread-local would be incorrect for async/threaded execution contexts.
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
        operator: TensorOperator,
        op_params: dict,
        input_value_ids: list[str],
        output_typespec: Optional[dict] = None,
    ) -> None:
        if not isinstance(operator, TensorOperator):
            raise TypeError("TensorNodeRecord operator must be a TensorOperator")

        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "output_value_id", output_value_id)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "op_params", dict(op_params))
        object.__setattr__(self, "input_value_ids", list(input_value_ids))
        object.__setattr__(self, "output_typespec", output_typespec)

    def to_dict(self) -> dict:
        return serialize(self)


@dataclass(frozen=True)
class TensorGraph:
    """Python-owned derivative program representation for Phase 1 autodiff graph construction."""

    nodes: list[TensorNodeRecord]
    inputs: list[tuple[str, Optional[dict]]]
    outputs: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", list(self.nodes))
        object.__setattr__(self, "inputs", list(self.inputs))
        object.__setattr__(self, "outputs", list(self.outputs))


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
        self._outputs: list[str] = []
        self._token: Optional[contextvars.Token[Optional[TensorGraphBuilder]]] = None
        # Builder-owned typed-tracing side tables (client ADR-004; issue 95
        # https://github.com/TinyChain-Inc/client/issues/95). These are
        # never exposed to public callers and are separate from `_value_map`
        # (Invariant 10): `_value_map` only resolves `id(obj) -> value_id`,
        # while `_retained_values` keeps the object itself alive so a GC'd
        # intermediate cannot have its `id()` reused and corrupt identity.
        self._value_metadata: dict[str, dict[str, object]] = {}
        self._retained_values: dict[str, object] = {}
        self._input_value_ids: list[str] = []
        self._input_names: set[str] = set()
        self._completed: bool = False

    # TODO(issue-13-followup): introduce scoped opaque NodeId/ValueId namespaces
    # to prevent accidental cross-graph/context id reuse.
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
        resolved = self._value_map[key]
        # Retain a strong reference so identity survives GC/`id()` reuse (Invariant 6).
        self._retained_values[resolved] = obj
        return resolved

    def _describe_value_role(self, value: object) -> str:
        """Best-effort human-readable role for an untraced *value*, for error messages."""
        subject_root = getattr(value, "_subject_root", None)
        if isinstance(subject_root, str) and subject_root.startswith("$"):
            return f" (named {subject_root[1:]!r})"
        return ""

    def value_id(self, value: object) -> str:
        """Resolve a previously traced Python object to its stable ValueId.

        Raises `ValueError` (naming the object's role when determinable) if
        *value* was never traced by this builder.
        """
        resolved = self._value_map.get(id(value))
        if resolved is None:
            raise ValueError(
                f"object was never traced by this TensorGraphBuilder{self._describe_value_role(value)}"
            )
        return resolved

    def _set_value_metadata(self, value_id: str, *, dtype: str, shape: tuple[object, ...]) -> None:
        self._value_metadata[value_id] = {"dtype": dtype, "shape": tuple(shape)}

    def _get_value_metadata(self, value_id: str) -> Optional[dict[str, object]]:
        """Return a defensive copy of the normalized metadata for *value_id*, if present."""
        metadata = self._value_metadata.get(value_id)
        if metadata is None:
            return None
        return {"dtype": metadata["dtype"], "shape": tuple(metadata["shape"])}

    def _value_metadata_to_boundary_dict(self, value_id: str) -> Optional[dict[str, object]]:
        """Convert normalized metadata to the `{"dtype", "shape": list(...)}` graph-boundary form."""
        metadata = self._get_value_metadata(value_id)
        if metadata is None:
            return None
        return {"dtype": metadata["dtype"], "shape": list(metadata["shape"])}

    @staticmethod
    def _normalize_input_shape(shape: object) -> tuple[object, ...]:
        if isinstance(shape, str) or not isinstance(shape, Sequence):
            raise ValueError("TensorGraphBuilder.input(...) shape must be a non-string sequence of dimensions")
        normalized: list[object] = []
        for dim in shape:
            if isinstance(dim, bool):
                raise ValueError(f"TensorGraphBuilder.input(...) shape dimension must not be a bool, got {dim!r}")
            if isinstance(dim, int):
                if dim < 0:
                    raise ValueError(f"TensorGraphBuilder.input(...) shape dimension must be non-negative, got {dim!r}")
                normalized.append(dim)
            elif isinstance(dim, str) and dim.isidentifier():
                normalized.append(dim)
            else:
                raise ValueError(
                    "TensorGraphBuilder.input(...) shape dimension must be a non-negative int "
                    f"or a valid identifier symbol, got {dim!r}"
                )
        return tuple(normalized)

    def input(self, name: str, *, dtype: str, shape: Sequence[object]) -> "Tensor":
        """Declare a named, typed graph input and return an ordinary symbolic `Tensor`.

        Requires this builder to be the active trace context (client ADR-004,
        spec §7.3; https://github.com/TinyChain-Inc/client/issues/95). The
        returned `Tensor` is built through the canonical state-reference
        builders; no raw type-spec dict is ever exposed.
        """
        if get_active_builder() is not self:
            raise RuntimeError("TensorGraphBuilder.input(...) requires this builder to be the active trace context")
        if not isinstance(name, str) or not name:
            raise TypeError("TensorGraphBuilder.input(...) name must be a non-empty string")
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(
                f"TensorGraphBuilder.input(...) name {name!r} must be a valid, non-keyword Python identifier"
            )
        if name in self._input_names:
            raise ValueError(f"TensorGraphBuilder.input(...) duplicate input name {name!r}")
        if dtype not in _INPUT_DTYPES:
            raise ValueError(f"TensorGraphBuilder.input(...) dtype must be one of {_INPUT_DTYPES}, got {dtype!r}")
        normalized_shape = self._normalize_input_shape(shape)

        from ..state.scalar import IdRef
        from ..collection.tensor import Tensor

        value = Tensor(ref=IdRef(name))
        value_id = self.register_value(value)
        self._set_value_metadata(value_id, dtype=dtype, shape=normalized_shape)
        self._input_value_ids.append(value_id)
        self._input_names.add(name)
        return value

    def record(self, node: TensorNodeRecord) -> None:
        self._nodes.append(node)

    def mark_output(self, obj: object) -> str:
        """Mark a registered graph value as an explicit output and return its ValueId."""
        value_id = self.register_value(obj)
        self.mark_output_value(value_id)
        return value_id

    def mark_output_value(self, value_id: str) -> None:
        """Mark *value_id* as an explicit graph output."""
        if not isinstance(value_id, str) or not value_id:
            raise TypeError("TensorGraphBuilder output value ids must be non-empty strings")
        if value_id not in self._outputs:
            self._outputs.append(value_id)

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
        outputs = list(self._outputs) if self._outputs else [self._nodes[-1].output_value_id] if self._nodes else []
        return TensorGraph(nodes=self._nodes, inputs=inputs, outputs=outputs)

    def __enter__(self) -> TensorGraphBuilder:
        if self._completed:
            raise RuntimeError("TensorGraphBuilder instances are single-trace; this builder already completed a trace")
        if _active_builder.get() is not None:
            raise RuntimeError("Nested TensorGraphBuilder contexts are not supported")
        self._token = _active_builder.set(self)
        return self

    def __exit__(self, *args: object) -> None:
        if self._token is not None:
            _active_builder.reset(self._token)
            self._token = None
        # Retention/metadata persist after exit; only discarding the builder
        # instance releases them (spec §8.3, §12.5;
        # https://github.com/TinyChain-Inc/client/issues/95).
        self._completed = True
