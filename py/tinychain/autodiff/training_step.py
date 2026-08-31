"""Declaration and loss-signature validation for the training-step compiler.

`compile_training_step` (built on top of this module) turns a framework-traced
loss into a set of lowered programs. Before any of that tracing happens,
everything checkable about the caller's declaration is checked, so a malformed
declaration never reaches the caller's loss body: the declaration set itself
(`inputs` and `parameters`), the loss signature bound against the declared
input names, and the optimizer contract in both directions.

The optimizer contract and the typed-input-spec well-formedness of each
declared input are **not** re-implemented here. Both are delegated to the
validators `tinychain.autodiff.training` already owns -- one owner per check --
so a malformed typed spec raises that validator's own
`missing_dtype_metadata`/`missing_shape_metadata`, and an optimizer
disagreement raises its own `invalid_update_signature`, never a new category.
Those validators are module-private in `training.py`; this module imports them
intra-package rather than promoting them, because the package's public export
set is pinned to exactly four names by the surface this module composes into.

Validation order, and why it is a property of the code:

1. Declaration set -- `inputs`/`parameters` well-formedness.
2. Each declared input's typed spec, through `training._typed_input_spec`.
3. The optimizer contract, through `training`'s existing validators, in both
   directions: the declared-optimizer-inputs comparison and the
   `update`-signature binding.
4. The loss callable's signature, bound against exactly the declared input
   names.

Steps 1-3 run to completion before step 4, and step 4 runs to completion
before any builder is entered -- `validate_declaration` returns normally or
raises; it never calls the loss callable, so a rejected declaration can never
have caused a side effect in the caller's loss body.

Once a declaration validates, `trace_loss` performs the single typed trace:
one `TensorGraphBuilder` declares every input, invokes the loss exactly once
by keyword, and finalizes the resulting graph through the builder's typed
`build(outputs=[...])` path -- never reimplemented here. `invalid_loss_output`
is this module's own category for a loss that does not return a single
`Tensor`; every other failure inside the traced call belongs to whichever
collaborator raised it, unchanged.

`differentiate_loss` is the stage after that: it mints the seed value id,
requests the VJP from the **source** forward graph, and checks the minted
seed against the derivative program that comes back. Three properties of it
are deliberate and should survive a later tidy-up:

* The seed is **minted**, never taken from the caller (D-3, Inv-11), and its
  concrete spelling is an implementation detail. Uniqueness comes from the
  candidate search against the source graph plus the post-generation check --
  never from the shape of the identifier -- so the spelling may be changed
  freely, and nothing outside this module may depend on it.
* `generate` is bound here as a module-level name and called bare. That
  binding is what lets a test substitute a controlled `DerivativeProgram`,
  which is the only way the post-generation collision path can be exercised
  at all: `generate` performs no seed-collision check of its own, which is
  precisely why FR-129-019 requires one here. Rewriting the call as
  `generate_module.generate(...)` or importing it inside the function body
  would silently disable that coverage.
* `graph_id` defaults to `None` and is passed through untouched, so
  `DerivativeMetadata.source_graph_id` stays the content hash reverse
  traversal derives. Minting an id here instead would make two identical
  compilations disagree and break Inv-13's conditional determinism.

`analyze_source_captures` closes the source phase. It asks
`analyze_derivative_dependencies` what the source derivative program needs
from the source forward graph, reads the answer through the documented
`forward_capture` provenance, and derives the forward output selection from
it. The capture set and the selection are one rule, not two (Inv-7), which is
why one function produces both: a caller that received only the list would
have to rebuild the selection itself, and that is the rejected alternative
B-2 of §15.2. For the same reason the captures are never rediscovered by
walking forward nodes -- §15.2's B-1 -- and the record restates no per-value
provenance: dtype, shape, and provenance already live on the analysis it
carries and, later, on `LoweredProgram.dependencies` (FR-129-013). What it
adds are identifiers, because selecting them as forward outputs is an
obligation of this compiler rather than a re-export of the analysis.
"""

from __future__ import annotations

import inspect
import keyword
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from .dependencies import DependencyAnalysis, analyze_derivative_dependencies
from .generate import generate
from .graph import TensorGraph, TensorGraphBuilder
from .protocol import AutodiffError
from .reverse import DerivativeProgram
from ..state import Scalar
from .training import (
    Optimizer,
    _resolve_optimizer_inputs,
    _typed_input_spec,
    _update_label,
    _validate_declared_optimizer_inputs,
    _validate_update_signature,
)


def validate_declaration(
    *,
    inputs: Mapping[str, object],
    parameters: Sequence[str],
    loss: Callable[..., object],
    optimizer: "Optimizer | Callable[..., object]",
    optimizer_inputs: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> None:
    """Validate a training-step declaration before any builder is entered.

    Raises `invalid_training_declaration` for a malformed declaration set,
    the training module's own categories for a malformed typed input spec or
    a disagreeing optimizer contract, and `invalid_loss_signature` for a loss
    callable that cannot be bound against exactly the declared input names.
    Returns `None` when the declaration is well-formed. The loss callable's
    body is never invoked, on any path.
    """
    input_names = _validate_declaration_set(inputs, parameters)

    for name in input_names:
        _typed_input_spec(inputs[name], label=name)

    resolved_optimizer_inputs = _resolve_optimizer_inputs(optimizer_inputs)
    _validate_optimizer_contract(optimizer, resolved_optimizer_inputs)

    _validate_loss_signature(loss, input_names)


def _require_identifier(name: str, *, role: str) -> None:
    """Require *name* (already known to be a `str`) to be a usable identifier.

    A name that is not a valid, non-keyword Python identifier cannot be
    bound as a keyword argument. Left unchecked here, it is only caught
    later -- by `_validate_loss_signature` for a loss with exact keyword
    parameters, or not at all for a loss accepting `**kwargs`, in which case
    it reaches `TensorGraphBuilder.input` and fails there with a raw,
    uncategorized `ValueError`/`TypeError`. *role* names which declaration
    the offending entry came from (`"inputs"` or `"parameters"`) so the
    message points at the actual mistake.
    """
    if not name or not name.isidentifier() or keyword.iskeyword(name):
        raise AutodiffError(
            "invalid_training_declaration",
            f"{role} declares a name {name!r}, which is not usable as a "
            "keyword argument: a declared name must be a non-empty, "
            "non-keyword Python identifier",
        )


def _validate_declaration_set(
    inputs: object, parameters: object
) -> tuple[str, ...]:
    """Require `inputs` and `parameters` to be a well-formed declaration set.

    `inputs` must be a non-empty mapping. `parameters` must be a non-empty
    collection of names, free of repeats, every one a key of `inputs`. A bare
    string is rejected rather than iterated character by character, matching
    the same convention `training._required_optimizer_input_names` already
    uses for the same mistake.

    Every key of `inputs` is required to be a `str` before anything iterates,
    sorts, or reports those keys: a non-`str` key is still hashable -- dict
    construction already requires that -- so it survives silently until a
    later message tries to `sorted()` the declared input names for display,
    which raises a bare `TypeError` for a mix of incomparable key types
    rather than reporting the declaration mistake at its source.

    Each entry of `parameters` is required to be a `str` before it is hashed
    or looked up: the duplicate check and the `inputs` membership test both
    hash the entry, so an unhashable entry -- a `list`, `dict`, or `set` given
    where a name was expected -- would otherwise raise a bare `TypeError` from
    those checks themselves rather than being reported as the declaration
    mistake it is.

    Every `inputs` key and `parameters` entry is also required to be a valid,
    non-keyword Python identifier, checked here rather than left to whatever
    happens to notice it later. A loss declaring exact keyword parameters
    cannot bind a name like `"a b"`, so `_validate_loss_signature` rejects it
    first and no builder is ever reached -- but a loss accepting `**kwargs`
    binds any string key at all, so the same declaration would otherwise
    reach `TensorGraphBuilder.input`, which fails with a raw, uncategorized
    `ValueError`/`TypeError`. This mirrors
    `training._validate_optimizer_input_name`, which validates
    `optimizer_inputs` names in their own right for the identical reason.
    Unlike `optimizer_inputs`, which is an independent name namespace,
    `parameters` here is not a second namespace to check for its own sake --
    every `parameters` entry is already required to be a key of `inputs`
    (checked below), so a bad `parameters` name is necessarily also a bad
    `inputs` key. The `parameters` entry is still checked explicitly rather
    than relying on that coupling alone, so the failure is attributed to
    whichever declaration actually names the bad identifier, and to keep the
    two loops symmetric rather than one depending on an invariant enforced
    only by the other.
    """
    if not isinstance(inputs, Mapping) or not inputs:
        raise AutodiffError(
            "invalid_training_declaration",
            "inputs must be a non-empty mapping of input name to typed "
            f"input spec, got {inputs!r}",
        )
    for key in inputs:
        if not isinstance(key, str):
            raise AutodiffError(
                "invalid_training_declaration",
                f"inputs declares a key {key!r} of type "
                f"{type(key).__name__!r}; each declared input name must be "
                "a str",
            )
        _require_identifier(key, role="inputs")
    input_names = tuple(inputs)

    if isinstance(parameters, str) or not isinstance(parameters, Sequence):
        raise AutodiffError(
            "invalid_training_declaration",
            "parameters must be declared as a non-empty collection of "
            f"names, got {type(parameters).__name__!r}",
        )
    if not parameters:
        raise AutodiffError(
            "invalid_training_declaration",
            "parameters must be non-empty: at least one declared input "
            "must be trained",
        )

    seen: set[str] = set()
    for name in parameters:
        if not isinstance(name, str):
            raise AutodiffError(
                "invalid_training_declaration",
                f"parameters declares {name!r} of type "
                f"{type(name).__name__!r}; each parameter name must be a "
                "str",
            )
        _require_identifier(name, role="parameters")
        if name in seen:
            raise AutodiffError(
                "invalid_training_declaration",
                f"parameters declares {name!r} more than once; each "
                "parameter name must be declared exactly once",
            )
        seen.add(name)
        if name not in inputs:
            raise AutodiffError(
                "invalid_training_declaration",
                f"parameters declares {name!r}, which is not a key of "
                f"inputs; declared inputs are {sorted(input_names)}",
            )

    return input_names


def _validate_optimizer_contract(
    optimizer: "Optimizer | Callable[..., object]",
    resolved_optimizer_inputs: Mapping[str, Mapping[str, object]],
) -> None:
    """Validate the optimizer contract in both directions, per §8.1/§13.3.

    Delegates entirely to the training module's existing validators. For an
    `Optimizer`, both the declared-inputs comparison and the `update`-method
    signature binding are applied -- the same two checks
    `trace_parameter_update` applies before tracing. For a plain callable,
    only the signature binding applies, because there is no declared
    `required_optimizer_inputs` to compare against.
    """
    label = _update_label(optimizer)
    if isinstance(optimizer, Optimizer):
        _validate_declared_optimizer_inputs(optimizer, resolved_optimizer_inputs)
        _validate_update_signature(
            optimizer.update,
            parameter=None,
            gradient=None,
            optimizer_inputs=resolved_optimizer_inputs,
            label=label,
        )
    else:
        _validate_update_signature(
            optimizer,
            parameter=None,
            gradient=None,
            optimizer_inputs=resolved_optimizer_inputs,
            label=label,
        )


def _validate_loss_signature(
    loss: Callable[..., object], input_names: tuple[str, ...]
) -> None:
    """Bind *loss* against exactly the declared input names, per FR-129-017.

    Runs last, after the declaration set, every typed input spec, and the
    optimizer contract have all validated cleanly -- so a rejected loss
    signature is the only failure ever reported once the declaration itself
    is known to be sound. `inspect.signature` on a non-callable raises
    `TypeError`, which is caught here exactly like a genuine binding mismatch,
    so a loss that is not callable is reported as an invalid loss signature
    rather than escaping as a bare `TypeError`. Signature *retrieval* can also
    fail on its own with `ValueError` -- for a C-implemented callable such as
    `min` that carries no introspectable signature -- and that is caught here
    too, so both failure modes are reported as an invalid loss signature
    rather than one of them escaping bare.
    """
    try:
        signature = inspect.signature(loss)
        signature.bind(**dict.fromkeys(input_names))
    except (TypeError, ValueError) as exc:
        raise AutodiffError(
            "invalid_loss_signature",
            f"loss callable {loss!r} must accept exactly the declared "
            f"input names {input_names!r} by keyword: {exc}",
        ) from exc


@dataclass(frozen=True)
class TracedLoss:
    """The finalized typed graph produced by tracing a training-step loss.

    ``input_value_ids`` maps every declared input name to its stable value id
    in ``graph``, covering every declared name exactly once regardless of
    whether the loss body actually reads it -- so a consumer binds runtime
    values by name instead of scanning ``graph.inputs`` or inferring order.
    ``loss_value_id`` is the value id of the single `Tensor` the loss
    returned, and is also ``graph.outputs``'s only element.
    """

    graph: TensorGraph
    loss_value_id: str
    input_value_ids: Mapping[str, str]


def trace_loss(
    *, inputs: Mapping[str, object], input_names: Sequence[str], loss: Callable[..., object]
) -> TracedLoss:
    """Trace *loss* exactly once inside exactly one `TensorGraphBuilder`.

    By the time this runs, `validate_declaration` has already validated the
    declaration set, every typed input spec, the optimizer contract, and the
    loss signature -- so this never re-checks any of that, and never reaches
    a loss body that a declaration mistake should have prevented.

    One builder is opened; every entry of *input_names* is declared through
    `builder.input(name, **spec)`, in *input_names* order (already the
    declaration's own mapping-insertion order, per `validate_declaration`);
    *loss* is invoked exactly once, by keyword, with one declared `Tensor`
    per name and no positional arguments. The call is not wrapped: an
    exception raised inside the loss body -- `AutodiffError` or not,
    interpreter control flow included -- propagates unchanged (see §13.2).

    A return value that is not a single `Tensor` raises `invalid_loss_output`
    naming the type actually returned. Otherwise the returned `Tensor` is
    built as the graph's sole selected output through the builder's typed
    `build(outputs=[...])` path, so typed finalization runs and rejects any
    reachable value with incomplete dtype/shape metadata under that path's
    own category -- never reimplemented here.
    """
    with TensorGraphBuilder() as builder:
        input_tensors = {
            name: builder.input(name, **_typed_input_spec(inputs[name], label=name))
            for name in input_names
        }
        result = loss(**input_tensors)

    # `Tensor` is a `Scalar` (see `collection/tensor/core.py`'s MRO), and every
    # reduction -- `mean`, `max`, `min` -- is documented to return `Scalar`
    # even under an active trace, so a fully reduced loss (the normal shape
    # for a scalar loss) is legitimately a `Scalar`, never a `Tensor`.
    # Checking the common base accepts both without accepting anything an
    # ordinary Python collaborator could hand back by mistake: `None`, a
    # tuple, a list, a plain number, and a bare `object()` are not `Scalar`.
    if not isinstance(result, Scalar):
        raise AutodiffError(
            "invalid_loss_output",
            f"loss callable {loss!r} must return a single Tensor, got "
            f"{type(result).__name__!r}",
        )

    # `result` being a `Scalar` only proves its *type* is right -- it does not
    # prove *this* trace produced it. Any Scalar subclass a caller has lying
    # around (a Tensor traced by a different builder, a bare tc.String, a
    # tc.Number) passes the check above unchanged. TensorGraphBuilder.value_id
    # raises a bare ValueError for an object it never traced; that failure is
    # not this stage's to leave uncategorized (NFR-129-004), and
    # TensorGraphBuilder itself is off-limits to change, so it is caught here,
    # at this stage's own call site, and reported as the second half of the
    # same invalid_loss_output validation -- worded to be told apart from the
    # wrong-type case above: this one names that the value was never traced
    # by this trace, not that its type was wrong.
    try:
        builder.value_id(result)
    except ValueError as exc:
        raise AutodiffError(
            "invalid_loss_output",
            f"loss callable {loss!r} returned a {type(result).__name__!r} "
            "scalar this trace never produced: the returned value must be "
            "one this trace itself derived from the declared inputs, not an "
            "untraced or foreign value",
        ) from exc

    graph = builder.build(outputs=[result])

    input_value_ids = {
        name: builder.value_id(tensor) for name, tensor in input_tensors.items()
    }

    return TracedLoss(
        graph=graph,
        loss_value_id=builder.value_id(result),
        input_value_ids=input_value_ids,
    )


# The first candidate the seed minter offers, and the stem every later
# candidate is built from. The value is an implementation detail in the
# strongest sense: §8.3 makes the *search* and the post-generation check the
# source of uniqueness, and explicitly declines to promise any namespace to an
# expansion pass, which is a plain callable and exposes no such protocol. It
# is written as a constant here so the search reads as one rule rather than
# two string literals, not because anything may rely on it. It is chosen to
# sit outside the namespaces the surrounding machinery already mints into --
# the tracer's `v…`/`n…`, reverse traversal's `d…`/`dn…`, and expansion's
# `exn`/`exv` -- which lowers the odds of a collision but proves nothing on
# its own; that is what the search and the check are for.
_SEED_VALUE_ID_STEM = "tsv"


@dataclass(frozen=True)
class SourceDerivative:
    """The source derivative program and the seed it was generated against.

    ``program`` is the object `generate` returned, not a copy of it: the
    source artifacts are never rewritten (Inv-1), so there is nothing to copy
    them for. ``seed_value_id`` is the framework-minted identifier that is
    unique against both source artifacts -- against the forward graph by
    construction, and against this program by the check that ran before this
    record existed. ``seed_label`` is the caller's display name, carried for
    provenance and messages only; it is never an identifier (Inv-11).
    """

    program: DerivativeProgram
    seed_value_id: str
    seed_label: str


def _source_value_typespecs(graph: TensorGraph) -> dict[str, dict[str, object]]:
    """Map every typed value of *graph* to its typespec.

    Declared inputs first, then node outputs, exactly as reverse traversal's
    own `_value_typespecs` composes them -- so the typespec this module reads
    for the loss output is the same one the traversal will read for it. A
    loss value that is itself a declared input (a loss returning a parameter
    unchanged) therefore resolves through `graph.inputs`, not only through
    the node table.
    """
    typespecs: dict[str, dict[str, object]] = {}
    for value_id, typespec in graph.inputs:
        if typespec is not None:
            typespecs[value_id] = dict(typespec)
    for node in graph.nodes:
        if node.output_typespec is not None:
            typespecs[node.output_value_id] = dict(node.output_typespec)
    return typespecs


def _occupied_value_ids(graph: TensorGraph) -> set[str]:
    """The value ids *graph* already uses: declared inputs and node outputs.

    Exactly the set §8.3 names. Node *ids* are deliberately not included:
    they live in a different namespace from value ids, and treating them as
    occupied would narrow the candidate space for no stated reason.
    """
    occupied = {value_id for value_id, _ in graph.inputs}
    occupied.update(node.output_value_id for node in graph.nodes)
    return occupied


def _mint_seed_value_id(graph: TensorGraph) -> str:
    """Mint a value id that *graph* does not already use.

    A deterministic search from `_SEED_VALUE_ID_STEM + "0"` onward: the first
    unoccupied candidate wins, so two calls on equal graphs mint the same
    seed (Inv-13) and the result depends on the occupied set alone, never on
    how many times this function has run. The graph is finite and the
    candidate space is unbounded, so the loop always terminates.

    This establishes uniqueness against the forward graph only. Uniqueness
    against the derivative program cannot be established here -- that program
    does not exist yet -- which is why `differentiate_loss` checks it the
    moment `generate` returns.
    """
    occupied = _occupied_value_ids(graph)
    index = 0
    while True:
        candidate = f"{_SEED_VALUE_ID_STEM}{index}"
        if candidate not in occupied:
            return candidate
        index += 1


def _check_seed_against_derivative_program(
    program: DerivativeProgram, seed_value_id: str
) -> None:
    """Fail if the derivative program produces the minted seed (FR-129-019).

    Runs immediately after `generate` returns and **before** dependency
    analysis or any expansion pass, so a seed that is simultaneously a free
    input and a produced value is reported as the identity defect it is
    rather than surfacing later as an analysis failure on a value the caller
    believed was fresh.

    Every node id and every node output value id the program produces is
    checked. `program.value_typespecs` deliberately is not: reverse traversal
    records the seed's own typespec there for every program it produces, so
    reading that table would reject every derivative ever generated.
    """
    for node in program.nodes:
        for produced_id in (node.node_id, node.output_value_id):
            if produced_id == seed_value_id:
                raise AutodiffError(
                    "ambiguous_producer",
                    f"the minted seed value id {seed_value_id!r} collides with "
                    f"{produced_id!r}, which the source derivative program "
                    "produces: the seed must be a free input of that program, "
                    "never a value it computes",
                )


def differentiate_loss(
    *,
    traced: TracedLoss,
    parameters: Sequence[str],
    seed_label: str = "seed",
    graph_id: Optional[str] = None,
) -> SourceDerivative:
    """Generate the source derivative program for *traced*, per §8.3/§8.4.

    The seed is minted against the source forward graph; the VJP is requested
    from that same graph object, with `wrt` equal to the declared parameters'
    value ids in *parameters* order and the seed typespec taken from the loss
    output's own typespec; and the minted seed is checked against everything
    the returned program produces before this function returns.

    Neither source artifact is rewritten (Inv-1): the graph is handed to
    `generate` as-is and the program comes back out unchanged. *parameters*
    order is the `wrt` order, which becomes `metadata.wrt_signature` and,
    downstream, the gradient and record order -- the single chain of
    equalities Inv-6 rests on.

    `generate`'s own failures propagate unchanged, with their own categories
    and messages: `missing_derivative_behavior` for a declared parameter that
    receives no gradient, and the tracing/typing categories for a loss output
    this transform cannot differentiate (§13.2, §13.3).
    """
    seed_value_id = _mint_seed_value_id(traced.graph)
    seed_typespec = _source_value_typespecs(traced.graph).get(traced.loss_value_id)
    wrt = [traced.input_value_ids[name] for name in parameters]

    program = generate(
        traced.graph,
        traced.loss_value_id,
        wrt,
        seed_value_id,
        seed_typespec=seed_typespec,
        graph_id=graph_id,
    )

    _check_seed_against_derivative_program(program, seed_value_id)

    return SourceDerivative(
        program=program, seed_value_id=seed_value_id, seed_label=seed_label
    )


@dataclass(frozen=True)
class SourceCaptures:
    """The source artifacts' ordered forward captures and the selection they imply.

    ``analysis`` is the `DependencyAnalysis` object itself, unmodified: every
    consumer that needs a capture's dtype, shape, or provenance reads it
    there rather than from a field of this record (FR-129-013). It also still
    reports a captured loss under `forward_capture` provenance -- removing
    the loss is a rule about the forward *selection*, never a rewrite of the
    analysis.

    ``loss_value_id`` and ``forward_capture_value_ids`` are reported
    separately and are always disjoint (FR-129-007): the first is the value
    the step reports to the application, the second the values the forward
    phase must retain for the backward phase. They are two different
    obligations and this record does not blur them.

    ``forward_selected_outputs`` is what the forward lowering selects, and is
    exactly ``(loss_value_id,) + forward_capture_value_ids`` -- the single
    equality Inv-7 states. It is carried rather than left to the call site so
    the capture set and the selection derived from it cannot drift apart.
    """

    analysis: DependencyAnalysis
    loss_value_id: str
    forward_capture_value_ids: tuple[str, ...]
    forward_selected_outputs: tuple[str, ...]


def analyze_source_captures(
    *, traced: TracedLoss, derivative: SourceDerivative
) -> SourceCaptures:
    """Determine the ordered forward captures and the forward selection, per §8.5.

    The analysis runs against the **source** artifacts -- the traced forward
    graph and the program `generate` returned -- with the minted seed
    declared and the gradient value ids selected in `parameters` order, which
    is what `output_gradients` already is (it is `wrt` order, and `wrt` order
    is `parameters` order; Inv-6). Passing the gradients explicitly rather
    than relying on the default states at this call site which selection the
    capture set is a property of.

    Captures are read through the documented `forward_capture` provenance, in
    analysis order, and nothing else: not by walking forward nodes, and not
    through `required_inputs`, which also carries the declared inputs and the
    seed -- values the consumer binds itself and must never be asked to
    retain from a forward run (§15.2 B-1, FR-129-005).
    """
    analysis = analyze_derivative_dependencies(
        derivative.program,
        forward_graph=traced.graph,
        seed_value_ids=[derivative.seed_value_id],
        outputs=list(derivative.program.output_gradients),
    )

    loss_value_id = traced.loss_value_id

    # FR-129-006's deduplication, in the exact shape it states: order-
    # preserving, first occurrence kept, and the loss dropped from the
    # capture portion if it is also captured.
    #
    # Both guards are defensive, and deliberately so -- neither is reachable
    # through the current rule set, and neither may be deleted as dead code:
    #
    # * The loss guard. No VJP rule in the default registry reads the value
    #   its own node produced (`max`/`min`/`product` raise
    #   `unsupported_reduction`; every implemented rule reads operands and
    #   upstream cotangents only), and the loss is by construction the
    #   forward graph's final output, so no traced loss can come back as a
    #   forward capture today. The moment a rule that reads its own output is
    #   registered, this branch becomes live -- and FR-129-006 states it
    #   regardless of which rules happen to exist. The captured-loss cases in
    #   `test_autodiff_training_step_captures.py` are what exercise it: they
    #   hand a hand-built forward graph and derivative program to the real
    #   analysis, with the loss captured first in one and last in the other.
    # * The repeat guard. `analyze_derivative_dependencies` assigns each value
    #   exactly one provenance and emits it once, so a repeat cannot arrive
    #   from it; the guard states this stage's own rule rather than depending
    #   on that property of a collaborator.
    capture_value_ids: list[str] = []
    for dependency in analysis.forward_captures:
        value_id = dependency.value_id
        if value_id == loss_value_id or value_id in capture_value_ids:
            continue
        capture_value_ids.append(value_id)

    forward_capture_value_ids = tuple(capture_value_ids)

    return SourceCaptures(
        analysis=analysis,
        loss_value_id=loss_value_id,
        forward_capture_value_ids=forward_capture_value_ids,
        forward_selected_outputs=(loss_value_id,) + forward_capture_value_ids,
    )
