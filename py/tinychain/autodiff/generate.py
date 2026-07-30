"""Public derivative-program generation orchestration.

Generation owns the small adaptation from the public ``generate`` arguments to
the reverse traversal. It is separate from package exports so graph builders do
not need to import ``tinychain.autodiff`` during a VJP call.
"""

from __future__ import annotations

from .graph import TensorGraph
from .reverse import DerivativeProgram, ReverseTraversal


def generate(
    graph: TensorGraph,
    output_value_id: str | list[str],
    wrt: list[str],
    seed: str | list[str],
    *,
    seed_typespec: dict[str, object] | list[dict[str, object] | None] | None = None,
    graph_id: str | None = None,
) -> DerivativeProgram:
    """Build a structured Python derivative program without executing routes.

    ``seed`` supplies one upstream cotangent value id for each selected output.
    Optional seed metadata is passed unchanged to reverse traversal, which
    validates it against the selected output metadata.
    """
    output_value_ids = output_value_id if isinstance(output_value_id, list) else [output_value_id]
    seed_value_ids = seed if isinstance(seed, list) else [seed]
    seed_typespecs = seed_typespec if isinstance(seed_typespec, list) else None
    single_seed_typespec = None if isinstance(seed_typespec, list) else seed_typespec
    if len(output_value_ids) != len(seed_value_ids):
        raise TypeError("generate requires one seed value id per output value id")
    return ReverseTraversal().build(
        graph=graph,
        output_value_id=output_value_ids[0],
        output_value_ids=output_value_ids,
        wrt=wrt,
        seed_value_id=seed_value_ids[0],
        seed_value_ids=seed_value_ids,
        seed_typespec=single_seed_typespec,
        seed_typespecs=seed_typespecs,
        source_graph_id=graph_id,
    )
