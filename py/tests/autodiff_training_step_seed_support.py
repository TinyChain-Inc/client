"""Shared harness for the training-step seed and source-differentiation tests.

`test_autodiff_training_step_seed` drives the stage into existence and
`test_autodiff_training_step_contract` certifies the finished surface, but
both need the same fixtures to do it: a hand-built forward graph whose value
ids the test chooses, a `generate` seam that hands back a caller-controlled
`DerivativeProgram` built out of the seed the stage actually minted, and a
dependency-analysis seam that makes entering the analysis impossible to miss.
The two files had a copy each; this module owns the single copy they now
share, so a change to the harness cannot land in one file and not the other.

Two rules of §8.3/§17.6.3 govern everything here, exactly as they govern the
files that import it:

* **The seed's spelling is not the contract.** Nothing in this module names,
  matches, or pattern-checks a namespace, prefix, or literal identifier for
  the minted seed. `FixedGenerate` exists precisely so a collision can be
  built out of the identifier the stage minted rather than out of a guess.
* **Ordering is observed, not assumed.** `forbid_dependency_analysis`
  installs a seam that both records its calls and raises a `BaseException`
  sentinel, so an implementation that analysed before checking would surface
  the sentinel instead of the category under test.

Names are exported without a leading underscore because they are this
module's public surface; the importing files bind them to their own private
names so their case bodies read unchanged.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Optional

import pytest
from tinychain.autodiff import dependencies as _dependencies_module
from tinychain.autodiff import training_step
from tinychain.autodiff.generate import generate as _real_generate
from tinychain.autodiff.graph import MulOperator, TensorGraph, TensorNodeRecord
from tinychain.autodiff.protocol import DerivativeMetadata
from tinychain.autodiff.reverse import DerivativeProgram
from tinychain.autodiff.training_step import TracedLoss


SPEC: Mapping[str, object] = {"dtype": "f32", "shape": [2, 3]}

PARAMETER_NAMES = ("w", "b")


# --------------------------------------------------------------------------
# fixtures: hand-built source graphs
#
# These graphs are built directly rather than traced, because the property
# under test is what the minter does when specific identifiers are already
# occupied -- and a traced graph's identifiers are chosen by the builder, not
# by the test.
# --------------------------------------------------------------------------


def mul_chain(
    output_value_ids: Sequence[str],
    *,
    input_value_ids: tuple[str, str] = ("pa", "pb"),
    spec: Mapping[str, object] = SPEC,
) -> TensorGraph:
    """A chain of elementwise multiplies with caller-chosen identifiers.

    Value ids are deliberately unlike the tracer's own (`pa`/`pb` rather than
    `v0`/`v1`) so a `wrt` built from parameter *names*, or from declaration
    order, cannot accidentally agree with a `wrt` built from the parameters'
    value ids in `parameters` order.
    """
    first, second = input_value_ids
    nodes: list[TensorNodeRecord] = []
    previous = first
    for index, output_value_id in enumerate(output_value_ids):
        nodes.append(
            TensorNodeRecord(
                node_id=f"m{index}",
                output_value_id=output_value_id,
                operator=MulOperator(),
                op_params={},
                input_value_ids=[previous, second],
                output_typespec=dict(spec),
            )
        )
        previous = output_value_id
    return TensorGraph(
        nodes=nodes,
        inputs=[(first, dict(spec)), (second, dict(spec))],
        outputs=[output_value_ids[-1]],
    )


def traced_chain(
    output_value_ids: Sequence[str],
    *,
    input_value_ids: tuple[str, str] = ("pa", "pb"),
    spec: Mapping[str, object] = SPEC,
) -> TracedLoss:
    graph = mul_chain(output_value_ids, input_value_ids=input_value_ids, spec=spec)
    return TracedLoss(
        graph=graph,
        loss_value_id=output_value_ids[-1],
        input_value_ids={
            PARAMETER_NAMES[0]: input_value_ids[0],
            PARAMETER_NAMES[1]: input_value_ids[1],
        },
    )


def occupied_value_ids(graph: TensorGraph) -> set[str]:
    """Exactly the set §8.3 requires the candidate search to avoid."""
    occupied = {value_id for value_id, _ in graph.inputs}
    occupied.update(node.output_value_id for node in graph.nodes)
    return occupied


def mint_seed_against(traced: TracedLoss) -> str:
    """Mint through the stage itself, resolved on the module at call time."""
    return training_step.differentiate_loss(
        traced=traced, parameters=[PARAMETER_NAMES[0]]
    ).seed_value_id


# --------------------------------------------------------------------------
# seams
# --------------------------------------------------------------------------


class AnalysisEntered(BaseException):
    """Raised by the analysis seam so entering it can never go unnoticed.

    Derives from `BaseException`, not `Exception`, so a stage that wrapped
    collaborator failures in a broad `except Exception` could not swallow it
    and re-report a different category.
    """


def forbid_dependency_analysis(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Install a recording seam over both dependency analyses.

    Patched on the owning module *and* on `training_step` itself, so the seam
    holds whether the stage reaches the analyses through the module or
    through a name bound at import time -- this stage does not call them at
    all today, and the tests using this seam must keep failing if a later
    change makes it.
    """
    calls: list[tuple] = []

    def recorder(*args: object, **kwargs: object):
        calls.append((args, kwargs))
        raise AnalysisEntered("dependency analysis was entered")

    for module in (_dependencies_module, training_step):
        for name in ("analyze_derivative_dependencies", "analyze_graph_dependencies"):
            monkeypatch.setattr(module, name, recorder, raising=False)
    return calls


class FixedGenerate:
    """Returns a caller-controlled `DerivativeProgram`, ignoring the graph.

    The seed the stage minted is recorded and handed to *build_program*, so a
    collision can be constructed against an identifier the test never had to
    guess.
    """

    def __init__(self, build_program) -> None:
        self._build_program = build_program
        self.seeds: list[object] = []
        self.calls: list[dict[str, object]] = []

    def __call__(self, *args: object, **kwargs: object) -> DerivativeProgram:
        bound = inspect.signature(_real_generate).bind(*args, **kwargs)
        bound.apply_defaults()
        self.calls.append(dict(bound.arguments))
        seed = bound.arguments["seed"]
        seed_value_id = seed[0] if isinstance(seed, list) else seed
        self.seeds.append(seed_value_id)
        return self._build_program(seed_value_id)


def fake_metadata(wrt: Sequence[str]) -> DerivativeMetadata:
    return DerivativeMetadata(
        source_graph_id="source",
        transform_version="0.1.0",
        tensor_op_contract_version="0.1.0",
        wrt_signature=tuple(wrt),
        seed_contract="",
    )


def fake_program(
    *,
    nodes: Sequence[TensorNodeRecord],
    wrt: Sequence[str],
    gradient_value_id: str,
    seed_value_id: Optional[str] = None,
    spec: Mapping[str, object] = SPEC,
) -> DerivativeProgram:
    value_typespecs = {value_id: dict(spec) for value_id in wrt}
    value_typespecs[gradient_value_id] = dict(spec)
    # The seed is always present in `value_typespecs` -- reverse traversal
    # records it there deliberately. A check that looked there instead of at
    # the produced node ids and output value ids would therefore reject every
    # program ever generated, so the fixture keeps that entry present in the
    # non-colliding control too.
    if seed_value_id is not None:
        value_typespecs[seed_value_id] = dict(spec)
    return DerivativeProgram(
        nodes=list(nodes),
        gradients={wrt[0]: gradient_value_id},
        output_gradients=[gradient_value_id],
        metadata=fake_metadata(wrt),
        value_typespecs=value_typespecs,
    )
