from __future__ import annotations

import contextvars
import keyword
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from ..serialize import serialize
from .finalize import finalize_typed_graph
from .protocol import AutodiffError
from .shape import check_differentiable_dtype


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
        self._symbol_bindings: dict[str, int] = {}
        self._retained_values: dict[str, object] = {}
        self._input_value_ids: list[str] = []
        self._input_names: set[str] = set()
        self._completed: bool = False
        self._exited_cleanly: bool = False

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

    def _copy_symbol_bindings(self) -> dict[str, int]:
        """Return a transactional copy of the graph-wide symbolic bindings."""
        return dict(self._symbol_bindings)

    def _replace_symbol_bindings(self, bindings: dict[str, int]) -> None:
        """Commit graph-wide symbolic bindings after successful node inference."""
        self._symbol_bindings = dict(bindings)

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
        check_differentiable_dtype(dtype)
        normalized_shape = self._normalize_input_shape(shape)

        # Importing state/tensor at module scope would initialize Tensor, whose
        # recorder imports this module for concrete operator identities.
        from ..state.scalar import IdRef
        from ..collection.tensor import Tensor

        value = Tensor(IdRef(name))
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

    @staticmethod
    def _as_output_sequence(outputs: object) -> list[object]:
        """Normalize a single traced value or a sequence of them to a list.

        A lone traced ``Tensor`` (which is not a ``Sequence``) is wrapped in a
        one-element list; strings/bytes are never treated as sequences of
        outputs.
        """
        if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
            return [outputs]
        return list(outputs)

    def _resolve_output_value_ids(self, outputs: object) -> list[str]:
        """Resolve traced output values to ValueIds in caller order.

        Later duplicate ValueIds are silently dropped (first occurrence kept);
        an untraced output raises ``ValueError`` naming the role when
        determinable (via :meth:`value_id`).
        """
        output_sequence = self._as_output_sequence(outputs)
        if not output_sequence:
            raise ValueError(
                "TensorGraphBuilder.build(outputs=...) requires at least one explicit output"
            )

        resolved: list[str] = []
        seen: set[str] = set()
        for output in output_sequence:
            value_id = self.value_id(output)
            if value_id not in seen:
                seen.add(value_id)
                resolved.append(value_id)
        return resolved

    def build(self, outputs: object | Sequence[object] | None = None) -> TensorGraph:
        """Assemble and return the recorded TensorGraph.

        With no argument this preserves the low-level experimental behavior:
        graph inputs carry no typespec (``None``) and outputs default to the
        explicitly marked outputs, else the last recorded node. When ``outputs``
        is supplied the typed application path is taken (client ADR-004, spec
        §8.3; https://github.com/TinyChain-Inc/client/issues/95): each traced
        output value is resolved to its ValueId in caller order with later
        duplicates silently dropped (first occurrence kept), untraced outputs
        raise, graph input typespecs are populated from the builder metadata
        table, and the graph is run through typed finalization, which rejects any
        reachable input or node output with incomplete dtype/shape metadata.
        """
        produced: set[str] = {rec.output_value_id for rec in self._nodes}
        seen: set[str] = set()
        input_value_ids: list[str] = []
        for rec in self._nodes:
            for vid in rec.input_value_ids:
                if vid not in produced and vid not in seen:
                    input_value_ids.append(vid)
                    seen.add(vid)

        if outputs is None:
            inputs = [(vid, None) for vid in input_value_ids]
            graph_outputs = (
                list(self._outputs)
                if self._outputs
                else [self._nodes[-1].output_value_id]
                if self._nodes
                else []
            )
            return TensorGraph(nodes=self._nodes, inputs=inputs, outputs=graph_outputs)

        graph_outputs = self._resolve_output_value_ids(outputs)
        produced_by = {record.output_value_id: record for record in self._nodes}
        reachable_inputs: set[str] = set()
        visited_values: set[str] = set()
        pending = list(graph_outputs)
        while pending:
            value_id = pending.pop()
            if value_id in visited_values:
                continue
            visited_values.add(value_id)
            record = produced_by.get(value_id)
            if record is None:
                reachable_inputs.add(value_id)
            else:
                pending.extend(record.input_value_ids)

        declared_inputs = [
            value_id for value_id in self._input_value_ids if value_id in reachable_inputs
        ]
        ordered_inputs = list(declared_inputs)
        ordered_seen = set(ordered_inputs)
        for value_id in [*input_value_ids, *graph_outputs]:
            if value_id in reachable_inputs and value_id not in ordered_seen:
                ordered_inputs.append(value_id)
                ordered_seen.add(value_id)
        input_value_ids = ordered_inputs
        inputs = [(vid, self._value_metadata_to_boundary_dict(vid)) for vid in input_value_ids]
        graph = TensorGraph(nodes=self._nodes, inputs=inputs, outputs=graph_outputs)
        return finalize_typed_graph(graph)

    def vjp(
        self,
        output: object,
        *,
        wrt: Sequence[object],
        seed: str = "seed",
        graph_id: str | None = None,
    ) -> "DerivativeProgram":
        """Generate the vector-Jacobian product for ``output`` with respect to ``wrt``.

        Callable only after this builder's trace context has exited successfully
        (client ADR-004, spec §10.6/§12/§13.1;
        https://github.com/TinyChain-Inc/client/issues/95). It resolves the
        selected ``output`` and each ``wrt`` value to ValueIds (preserving
        ``wrt`` order), runs typed finalization on the selected path, infers the
        seed typespec verbatim from the selected output's dtype and ranked shape
        (no promotion, broadcasting, scalar expansion, or rank normalization),
        and delegates to the existing ``generate(...)``. It performs no second
        reverse traversal and does not mutate the recorded forward graph, so
        repeated calls with different ``wrt`` subsets are deterministic.

        ``seed`` is the ValueId of the upstream cotangent used to start reverse
        traversal; ``graph_id`` is forwarded verbatim as the source graph id.
        """
        if not (self._completed and self._exited_cleanly):
            raise RuntimeError(
                "TensorGraphBuilder.vjp(...) is only callable after the trace context has exited successfully"
            )

        if isinstance(wrt, (str, bytes)) or not isinstance(wrt, Sequence):
            raise TypeError("TensorGraphBuilder.vjp(...) wrt must be a non-string sequence of traced values")
        wrt_objects = list(wrt)
        if not wrt_objects:
            raise ValueError("TensorGraphBuilder.vjp(...) wrt must not be empty")

        output_value_id = self.value_id(output)

        wrt_value_ids: list[str] = []
        seen_wrt: set[str] = set()
        for position, wrt_object in enumerate(wrt_objects):
            wrt_value_id = self.value_id(wrt_object)
            if wrt_value_id in seen_wrt:
                raise ValueError(
                    "TensorGraphBuilder.vjp(...) duplicate wrt value"
                    f"{self._describe_value_role(wrt_object)} at position {position}"
                )
            seen_wrt.add(wrt_value_id)
            wrt_value_ids.append(wrt_value_id)

        # Typed finalization of the selected path (rejects incomplete metadata);
        # rebuilt from `self._nodes` each call, so the forward graph is never mutated.
        graph = self.build(outputs=output)

        seed_typespec = self._value_metadata_to_boundary_dict(output_value_id)
        if seed_typespec is None or not seed_typespec.get("dtype"):
            raise AutodiffError(
                "missing_dtype_metadata",
                f"selected vjp output {output_value_id!r} is missing dtype metadata for seed inference",
            )

        # Generation reaches reverse traversal, which owns VJP rules and imports
        # the graph model. Importing it while this module initializes would form
        # graph -> generate -> reverse -> graph, so this remains a call-time
        # cycle guard. The implementation itself lives in `generate.py`.
        from .generate import generate

        return generate(
            graph,
            output_value_id,
            wrt_value_ids,
            seed,
            seed_typespec=seed_typespec,
            graph_id=graph_id,
        )

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
        # Track whether the context exited without a propagating exception so
        # `vjp(...)` can require a *successful* exit (spec §13.1).
        exc_type = args[0] if args else None
        self._exited_cleanly = exc_type is None
