from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..serialize import serialize
from .accumulate import GradientAccumulator
from .graph import TensorGraph, TensorNodeRecord
from .protocol import AutodiffError, DerivativeMetadata
from .seed import SeedValidator
from .vjp import VjpContext, VjpRegistry, default_vjp_registry


@dataclass(frozen=True)
class DerivativeProgram:
    nodes: list[TensorNodeRecord]
    gradients: dict[str, str]
    output_gradients: list[str | None]
    metadata: DerivativeMetadata

    def to_dict(self) -> dict:
        return serialize(self)


class ReverseTraversal:
    def __init__(
        self,
        *,
        registry: VjpRegistry | None = None,
        seed_validator: SeedValidator | None = None,
        transform_version: str = "0.1.0",
        tensor_op_contract_version: str = "0.1.0",
    ) -> None:
        self._registry = registry or default_vjp_registry()
        self._seed_validator = seed_validator or SeedValidator()
        self._transform_version = transform_version
        self._tensor_op_contract_version = tensor_op_contract_version
        self._next_value_index = 0
        self._next_node_index = 0

    def build(
        self,
        *,
        graph: TensorGraph,
        output_value_id: str,
        wrt: list[str],
        seed_value_id: str,
        seed_typespec: dict[str, object] | None = None,
        source_graph_id: str | None = None,
    ) -> DerivativeProgram:
        value_typespecs = self._value_typespecs(graph)
        output_typespec = value_typespecs.get(output_value_id)
        if seed_typespec is not None:
            self._seed_validator.validate(
                seed_typespec=seed_typespec,
                output_typespec=output_typespec,
            )
            value_typespecs[seed_value_id] = dict(seed_typespec)
        elif output_typespec is not None:
            value_typespecs[seed_value_id] = dict(output_typespec)

        nodes = self._topological_sort(graph)
        targets_by_value = self._targets_by_value(graph, nodes, wrt)
        upstream = GradientAccumulator(value_typespecs=value_typespecs)
        upstream.add(output_value_id, seed_value_id)
        derivative_nodes: list[TensorNodeRecord] = []

        for node in reversed(nodes):
            upstream_id, accumulation_nodes = upstream.result_for(
                node.output_value_id,
                next_value_id=self._next_value_id,
                next_node_id=self._next_node_id,
            )
            derivative_nodes.extend(accumulation_nodes)
            self._record_node_typespecs(value_typespecs, accumulation_nodes)
            if upstream_id is None:
                continue

            needed_inputs = frozenset(
                input_id
                for input_id in node.input_value_ids
                if targets_by_value.get(input_id)
            )
            if not needed_inputs:
                continue

            rule = self._registry.lookup(node.operator)
            result = rule.apply(
                VjpContext(
                    upstream_value_id=upstream_id,
                    node=node,
                    value_typespecs=value_typespecs,
                    needed_input_value_ids=needed_inputs,
                    next_value_id=self._next_value_id,
                    next_node_id=self._next_node_id,
                )
            )
            derivative_nodes.extend(result.derivative_nodes)
            self._record_node_typespecs(value_typespecs, result.derivative_nodes)
            for value_id, gradient_id in result.gradients.items():
                upstream.add(value_id, gradient_id)

        gradients: dict[str, str] = {}
        ordered: list[str | None] = []
        for value_id in wrt:
            gradient_id, accumulation_nodes = upstream.result_for(
                value_id,
                next_value_id=self._next_value_id,
                next_node_id=self._next_node_id,
            )
            derivative_nodes.extend(accumulation_nodes)
            self._record_node_typespecs(value_typespecs, accumulation_nodes)
            if gradient_id is None:
                raise AutodiffError(
                    "missing_derivative_behavior",
                    f"no derivative was computed for wrt value {value_id!r}",
                )
            gradients[value_id] = gradient_id
            ordered.append(gradient_id)

        graph_id = source_graph_id if source_graph_id is not None else _graph_content_hash(graph)
        return DerivativeProgram(
            nodes=derivative_nodes,
            gradients=gradients,
            output_gradients=ordered,
            metadata=DerivativeMetadata(
                source_graph_id=graph_id,
                transform_version=self._transform_version,
                tensor_op_contract_version=self._tensor_op_contract_version,
                wrt_signature=tuple(wrt),
                seed_contract=f"{seed_value_id} matches {output_value_id}",
            ),
        )

    def _value_typespecs(self, graph: TensorGraph) -> dict[str, dict[str, object]]:
        """Build value metadata used by seed checks, VJP shape planning, and accumulation."""
        typespecs: dict[str, dict[str, object]] = {}
        for value_id, typespec in graph.inputs:
            if typespec is not None:
                typespecs[value_id] = dict(typespec)
        for node in graph.nodes:
            if node.output_typespec is not None:
                typespecs[node.output_value_id] = dict(node.output_typespec)
        return typespecs

    def _record_node_typespecs(
        self,
        value_typespecs: dict[str, dict[str, object]],
        nodes: list[TensorNodeRecord],
    ) -> None:
        for node in nodes:
            if node.output_typespec is not None:
                value_typespecs[node.output_value_id] = dict(node.output_typespec)

    def _targets_by_value(
        self,
        graph: TensorGraph,
        nodes: list[TensorNodeRecord],
        wrt: list[str],
    ) -> dict[str, frozenset[str]]:
        requested = set(wrt)
        targets: dict[str, set[str]] = {}

        for value_id, _typespec in graph.inputs:
            targets[value_id] = {value_id} if value_id in requested else set()
        for node in nodes:
            targets.setdefault(
                node.output_value_id,
                {node.output_value_id} if node.output_value_id in requested else set(),
            )
            for input_id in node.input_value_ids:
                targets.setdefault(input_id, {input_id} if input_id in requested else set())

        for node in nodes:
            output_targets = set(targets.get(node.output_value_id, set()))
            for input_id in node.input_value_ids:
                output_targets.update(targets.get(input_id, set()))
            targets[node.output_value_id] = output_targets

        return {value_id: frozenset(value_targets) for value_id, value_targets in targets.items()}

    def _topological_sort(self, graph: TensorGraph) -> list[TensorNodeRecord]:
        produced_by = {node.output_value_id: node for node in graph.nodes}
        visited: set[str] = set()
        visiting: set[str] = set()
        ordered: list[TensorNodeRecord] = []

        def visit(node: TensorNodeRecord) -> None:
            if node.node_id in visited:
                return
            if node.node_id in visiting:
                raise AutodiffError("malformed_derivative_ir", "cycle detected in tensor graph")
            visiting.add(node.node_id)
            for input_id in node.input_value_ids:
                parent = produced_by.get(input_id)
                if parent is not None:
                    visit(parent)
            visiting.remove(node.node_id)
            visited.add(node.node_id)
            ordered.append(node)

        for node in graph.nodes:
            visit(node)
        return ordered

    def _next_value_id(self) -> str:
        value_id = f"d{self._next_value_index}"
        self._next_value_index += 1
        return value_id

    def _next_node_id(self) -> str:
        node_id = f"dn{self._next_node_index}"
        self._next_node_index += 1
        return node_id


def _graph_content_hash(graph: TensorGraph) -> str:
    """Return a stable 16-hex-digit SHA-256 content hash of *graph*.

    The canonical representation includes every node (sorted by node_id) and
    the sorted graph inputs/outputs so that structurally identical graphs
    produce the same hash across processes and runs.
    """
    canonical = [
        [
            node.node_id,
            type(node.operator).__name__,
            sorted(node.input_value_ids),
            node.output_value_id,
            node.output_typespec if node.output_typespec is not None else {},
        ]
        for node in sorted(graph.nodes, key=lambda n: n.node_id)
    ] + [sorted(graph.inputs), sorted(graph.outputs)]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
