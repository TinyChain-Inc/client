"""Fail-closed validation for completed typed Tensor graphs.

This module intentionally depends only on the shape and error contracts. Keeping
it independent of graph construction lets :mod:`graph` invoke finalization
without importing the recorder, whose capture registry depends on graph operator
types.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

from .protocol import AutodiffError
from .shape import typespec_ranked_shape

if TYPE_CHECKING:
    from .graph import TensorGraph


def _require_complete_typespec(typespec: Optional[Mapping[str, object]], *, label: str) -> None:
    """Fail closed unless *typespec* carries both a dtype and a ranked shape."""
    if typespec is None or not typespec.get("dtype"):
        raise AutodiffError("missing_dtype_metadata", f"{label} is missing dtype metadata")
    # Raises `missing_shape_metadata` when the shape is absent or malformed.
    typespec_ranked_shape(dict(typespec))


def finalize_typed_graph(graph: TensorGraph) -> TensorGraph:
    """Validate every reachable typed value and return *graph* unchanged.

    Unsupported or untraced intermediates appear as inputs without metadata and
    therefore fail closed instead of being omitted from derivative generation.
    """
    input_typespecs = dict(graph.inputs)
    produced_by = {node.output_value_id: node for node in graph.nodes}

    reachable_inputs: set[str] = set()
    reachable_nodes: list[object] = []
    seen_nodes: set[str] = set()
    pending: list[str] = list(graph.outputs)
    while pending:
        value_id = pending.pop()
        node = produced_by.get(value_id)
        if node is None:
            reachable_inputs.add(value_id)
            continue
        if node.node_id in seen_nodes:
            continue
        seen_nodes.add(node.node_id)
        reachable_nodes.append(node)
        pending.extend(node.input_value_ids)

    for value_id in sorted(reachable_inputs):
        _require_complete_typespec(input_typespecs.get(value_id), label=f"graph input {value_id!r}")
    for node in reachable_nodes:
        _require_complete_typespec(node.output_typespec, label=f"node {node.node_id!r} output")
    return graph
