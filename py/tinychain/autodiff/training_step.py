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

`expand_source_artifacts` is the stage after that, and the first one that runs
caller-supplied rewrites. Its shape follows from three things the specification
insists on together:

* With every sequence empty the stage is **inert**: it returns the source
  artifacts themselves, the same objects, so Inv-10 is observable by identity
  rather than by comparison. With a non-empty sequence the first pass receives
  a deep copy instead, because a pass is caller code that may mutate what it is
  handed and the source artifacts are never rewritten (Inv-1, FR-129-004).
  Those two rules are not in tension: the copy exists precisely because a pass
  does, so there is nothing to copy for when none runs.
* Every result is validated **between** passes, so a failure names the pass and
  its zero-based position and no later pass ever sees a broken artifact. The
  identifier rules are stated in `_validate_occupied_ids`, and the one that
  looks too weak on first reading -- a value id keeps its *role*, not its
  producing node -- is the one that lets this compiler compose with issue
  #128's shipped passes at all. That docstring explains why, with the example;
  read it before tightening anything there.
* The per-pass checks and the final recomputation are two halves of one
  contract, and neither is meaningful alone. The checks catch a structurally
  broken artifact before the next pass compounds it. The recomputation catches
  what no structural rule can: a pass that stayed entirely well-formed while
  changing which forward values the derivative actually reads.

`expand_update_graph` is the same machinery for one parameter's update graph,
with that artifact's own semantic requirement and no recomputation, since an
update graph has no dependency analysis to preserve.

`compile_training_step` is the public entry point, and is nothing more than
the ordered composition of every stage above with the three lowerings, the
per-parameter update cycle, and the record assembly (FR-129-001, §13.4). Two
properties of how it reaches its collaborators are load-bearing:

* `lower_graph`, `lower_derivative_program`, and `trace_parameter_update` are
  bound here as module-level names and called bare, for the same reason
  `generate` is. Inv-9 is an **identity** claim -- `handlers`, `fusion`, and
  `bind_input` reach every lowering as the same objects, never a wrapper, a
  filter, or a substituted default -- and an identity claim can only be
  observed at the seam the call passes through. The same binding is what makes
  "one update traced per parameter, with that parameter's own declared dtype
  and shape for both typed inputs" observable at all. Rewriting a call as
  `lowering.lower_graph(...)` or `training.trace_parameter_update(...)`, or
  importing any of the three inside the function body, would silently disable
  twenty-one test cases without failing one of them.
* The record is assembled **once**, at the end, out of local values only. No
  attribute is populated as the sequence proceeds and no half-filled record
  exists at any point, so a failure at any stage leaves nothing observable
  behind: the function returns a complete `CompiledTrainingStep` or raises
  (§13.4).

The module holds no module-level mutable state -- no cache, no registry, no
accumulator (Inv-12) -- so two compilations of the same declarations are
independent of each other and of their order.

"""

from __future__ import annotations

import copy
import inspect
import keyword
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional

from .dependencies import (
    DependencyAnalysis,
    ValueDependency,
    analyze_derivative_dependencies,
)
from .generate import generate
from .graph import TensorGraph, TensorGraphBuilder, TensorNodeRecord
from .lowering import (
    FusionHook,
    LoweredProgram,
    OperationHandlerRegistry,
    lower_derivative_program,
    lower_graph,
)
from .protocol import AutodiffError
from .reverse import DerivativeProgram
from ..state import Scalar
from .training import (
    Optimizer,
    _required_optimizer_input_names,
    _resolve_optimizer_inputs,
    _typed_input_spec,
    _update_label,
    _validate_declared_optimizer_inputs,
    _validate_update_signature,
    trace_parameter_update,
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


@dataclass(frozen=True)
class ExpandedArtifacts:
    """The artifacts the lowering phase consumes, plus the recomputed analysis.

    ``lowered_forward_graph`` and ``lowered_derivative_program`` are the source
    artifacts themselves when the matching sequence is empty -- the same
    objects, not copies (Inv-10). With a non-empty sequence they are whatever
    the last pass returned, validated.

    ``analysis`` is the recomputation of §8.6 against these two artifacts and
    the minted seed. It is carried rather than recomputed by the caller because
    it is the evidence for the equality this stage just enforced, and because
    lowering needs the same answer the check was made against.

    ``forward_pass_labels`` and ``derivative_pass_labels`` are the §9.1 labels
    of the passes that actually ran, in application order, for provenance. A
    label is not an identifier and nothing may key off it: two passes may
    legitimately carry the same label, which is why the position is what
    distinguishes them in every message.
    """

    lowered_forward_graph: TensorGraph
    lowered_derivative_program: DerivativeProgram
    analysis: DependencyAnalysis
    forward_pass_labels: tuple[str, ...]
    derivative_pass_labels: tuple[str, ...]


@dataclass(frozen=True)
class ExpandedUpdate:
    """One parameter's expanded update graph and the labels that produced it.

    ``lowered_graph`` is the source update graph itself when the sequence is
    empty, for the same reason `ExpandedArtifacts` carries the source
    artifacts: the stage is inert by default (Inv-10).
    """

    lowered_graph: TensorGraph
    pass_labels: tuple[str, ...]


def _pass_label(expansion: object) -> str:
    """Resolve *expansion*'s display label, per §9.1.

    The non-empty string `__qualname__` when the object carries one --
    ordinary functions do -- and otherwise the concrete type's `__qualname__`,
    which is what covers a callable instance, whose type's `__qualname__` is
    not visible on the instance itself. An empty or non-string `__qualname__`
    is treated as absent rather than reported verbatim: it would name nothing.

    `getattr` with a default is what keeps a missing attribute from escaping
    as a raw `AttributeError` (§9.1). Callers resolve the label **before**
    invoking the pass, so a pass that rewrites its own `__qualname__` while
    running cannot change how it is named in the failure it caused.
    """
    qualname = getattr(expansion, "__qualname__", None)
    if isinstance(qualname, str) and qualname:
        return qualname
    return type(expansion).__qualname__


def _expansion_violation(
    *, label: str, position: int, sequence: str, detail: str
) -> AutodiffError:
    """Build the `expansion_contract_violation` every pass failure reports.

    One constructor so the label, the zero-based position, and the sequence
    name appear in every message in the same shape. FR-129-020 requires all
    three: the label alone does not identify a pass, because two passes may
    share one.
    """
    return AutodiffError(
        "expansion_contract_violation",
        f"expansion pass {label!r} at position {position} of the {sequence} "
        f"expansion sequence {detail}",
    )


def _invoke_expansion(
    expansion: Callable[..., object],
    artifact: object,
    *,
    label: str,
    position: int,
    sequence: str,
) -> object:
    """Call one pass, applying §13.2's rule for an expansion pass.

    An `AutodiffError` is the pass declining an artifact in the framework's own
    vocabulary and propagates unchanged. Anything else is an uncategorized
    breach of a declared artifact-to-artifact contract and is wrapped, with the
    original chained so the pass's own traceback survives. `KeyboardInterrupt`
    and `SystemExit` are `BaseException`s and are therefore never caught here
    at all -- interpreter control flow is never wrapped anywhere (§13.2).
    """
    try:
        return expansion(artifact)
    except AutodiffError:
        raise
    except Exception as exc:
        raise _expansion_violation(
            label=label,
            position=position,
            sequence=sequence,
            detail=f"raised {type(exc).__name__}: {exc}",
        ) from exc


def _declared_input_value_ids(artifact: object) -> tuple[str, ...]:
    """The value ids *artifact* declares as free inputs.

    A `DerivativeProgram` declares none: its free values are whatever its nodes
    read and nothing produces, which is why the seed is checked by its own
    rule rather than by looking for a declaration that does not exist.
    """
    if isinstance(artifact, TensorGraph):
        return tuple(value_id for value_id, _ in artifact.inputs)
    return ()


def _produced_value_ids(artifact: object) -> tuple[str, ...]:
    return tuple(node.output_value_id for node in artifact.nodes)


def _available_value_ids(artifact: object) -> set[str]:
    """Every value id *artifact* can supply: a declared input or a node output."""
    return set(_declared_input_value_ids(artifact)) | set(_produced_value_ids(artifact))


def _value_roles(artifact: object) -> dict[str, str]:
    """Map each value id of *artifact* to the role it plays there.

    Two roles only -- a declared free input, or a value some node produces.
    Which role a value plays is the part of its identity a pass may not
    silently change (see `_validate_occupied_ids`).
    """
    roles = {
        value_id: "declared input"
        for value_id in _declared_input_value_ids(artifact)
    }
    for node in artifact.nodes:
        roles[node.output_value_id] = "node output"
    return roles


def _same_node_definition(left: object, right: object) -> bool:
    """Whether two records with one node id describe the same computation."""
    return (
        type(left.operator) is type(right.operator)
        and left.op_params == right.op_params
        and list(left.input_value_ids) == list(right.input_value_ids)
        and left.output_value_id == right.output_value_id
    )


def _validate_artifact_shape(
    result: object, *, label: str, position: int, sequence: str
) -> None:
    """Require a pass result to have the shape every later rule reads.

    The type check that runs before this one proves only that the result is a
    `TensorGraph` or a `DerivativeProgram`. Neither type validates what it is
    constructed from, so an artifact of exactly the right type can still hold a
    string where a node belongs, or a bare string where a `(value id,
    typespec)` pair belongs. Every rule after this point reads
    ``node.node_id``, ``node.output_value_id``, or unpacks a declared input,
    and would fail on such an artifact with a raw `AttributeError` or
    `ValueError` from inside a framework validator -- an uncategorized failure
    for what is a pass breaching its contract (§8.6's "malformed artifact",
    FR-129-020). This is the same care §9.1 already takes over a malformed
    label, applied to the artifact.

    The same applies to the two derivative-program fields something downstream
    reads: ``output_gradients``, which the gradient rule compares as
    identifiers, and ``value_typespecs``, which the recomputation's
    `analyze_derivative_dependencies` resolves metadata through -- a table that
    is not a mapping fails there, inside a collaborator, rather than here. A
    gradient entry is required to be a `str` even though the field is annotated
    `list[str | None]`: reverse traversal never emits `None`, raising
    `missing_derivative_behavior` instead, so no artifact reaching this stage
    can carry one, and the gradient-order rule would reject it regardless.
    ``gradients`` and ``metadata`` are deliberately **not** checked -- nothing
    on this path reads either, and guarding a field no rule consults would be
    inventing a contract rather than protecting one.

    It is deliberately shape-only: what the containers hold, and nothing about
    what the identifiers inside them mean. Whether ids are unique, carried
    forward for the same semantic node or value, or still reachable is owned by
    `_validate_occupied_ids` and the per-artifact semantic validators, and
    restating any of it here would create a second owner for a rule that has
    one.
    """
    nodes = result.nodes
    if isinstance(nodes, str) or not isinstance(nodes, Sequence):
        raise _expansion_violation(
            label=label,
            position=position,
            sequence=sequence,
            detail=(
                f"returned an artifact whose node list is a "
                f"{type(nodes).__name__!r}, not a sequence of nodes"
            ),
        )
    for index, node in enumerate(nodes):
        if not isinstance(node, TensorNodeRecord):
            raise _expansion_violation(
                label=label,
                position=position,
                sequence=sequence,
                detail=(
                    f"returned an artifact whose node at index {index} is "
                    f"{node!r}, a {type(node).__name__!r} rather than a "
                    "TensorNodeRecord"
                ),
            )

    if isinstance(result, DerivativeProgram):
        gradients = result.output_gradients
        if isinstance(gradients, str) or not isinstance(gradients, Sequence):
            raise _expansion_violation(
                label=label,
                position=position,
                sequence=sequence,
                detail=(
                    f"returned a program whose gradient list is a "
                    f"{type(gradients).__name__!r}, not a sequence of value ids"
                ),
            )
        for index, gradient in enumerate(gradients):
            if not isinstance(gradient, str):
                raise _expansion_violation(
                    label=label,
                    position=position,
                    sequence=sequence,
                    detail=(
                        f"returned a program whose gradient at index {index} is "
                        f"{gradient!r}, a {type(gradient).__name__!r} rather "
                        "than a value id"
                    ),
                )

        typespecs = result.value_typespecs
        if not isinstance(typespecs, Mapping):
            raise _expansion_violation(
                label=label,
                position=position,
                sequence=sequence,
                detail=(
                    f"returned a program whose value typespec table is a "
                    f"{type(typespecs).__name__!r}, not a mapping of value id "
                    "to typespec"
                ),
            )
        return

    if not isinstance(result, TensorGraph):
        return

    declared_inputs = result.inputs
    if isinstance(declared_inputs, str) or not isinstance(declared_inputs, Sequence):
        raise _expansion_violation(
            label=label,
            position=position,
            sequence=sequence,
            detail=(
                f"returned a graph whose declared input list is a "
                f"{type(declared_inputs).__name__!r}, not a sequence of "
                "declarations"
            ),
        )
    for index, declaration in enumerate(declared_inputs):
        if (
            isinstance(declaration, str)
            or not isinstance(declaration, Sequence)
            or len(declaration) != 2
            or not isinstance(declaration[0], str)
        ):
            raise _expansion_violation(
                label=label,
                position=position,
                sequence=sequence,
                detail=(
                    f"returned a graph whose declared input at index {index} "
                    f"is {declaration!r}, not a (value id, typespec) pair "
                    "whose value id is a str"
                ),
            )


def _validate_occupied_ids(
    result: object, pass_input: object, *, label: str, position: int, sequence: str
) -> None:
    """Apply §8.6's identifier rules to one pass result.

    Three rules, in the order a reader of a failure would want them:

    1. Node ids and value ids are each unique within the result. A value id is
       occupied by a declared input as much as by a node output, so the two
       are checked against one set.
    2. A **node id** present in both the pass input and the result carries the
       same definition -- operator type, `op_params`, `input_value_ids`, and
       `output_value_id`. Reusing an occupied node id for a different
       computation is the reassignment §8.6 forbids.
    3. A **value id** present in both keeps its role: a declared input stays a
       declared input, a produced value stays produced.

    Rule 3 is deliberately about the value's *role*, not about which node
    produces it, and that is the subtle part. A real expansion pass replaces a
    node with a region and has the region's terminal node carry the replaced
    node's own output value id -- issue #128's `expand_mean_graph` emits the
    loss value `v6` from the freshly minted `exn5` rather than from the `n3` it
    replaced, and its derivative counterpart does the same for `d2`. Requiring
    the producing *node* to be unchanged, or its definition to be unchanged,
    would reject the shipped passes this compiler exists to compose with, so
    the rule states what actually matters: a value the graph computed must not
    quietly become a value the caller has to bind, and vice versa. Do not
    "tighten" this back to producer identity.

    Every comparison is against the **pass input**, not against the source
    artifact: an id one pass minted is occupied for every pass after it.
    """
    node_ids: set[str] = set()
    for node in result.nodes:
        if node.node_id in node_ids:
            raise _expansion_violation(
                label=label,
                position=position,
                sequence=sequence,
                detail=f"declares the node id {node.node_id!r} more than once",
            )
        node_ids.add(node.node_id)

    value_ids: set[str] = set()
    for value_id in _declared_input_value_ids(result):
        if value_id in value_ids:
            raise _expansion_violation(
                label=label,
                position=position,
                sequence=sequence,
                detail=f"declares the value id {value_id!r} more than once",
            )
        value_ids.add(value_id)
    for node in result.nodes:
        if node.output_value_id in value_ids:
            raise _expansion_violation(
                label=label,
                position=position,
                sequence=sequence,
                detail=(
                    f"declares the value id {node.output_value_id!r} more than "
                    "once: an occupied value id may not be minted again"
                ),
            )
        value_ids.add(node.output_value_id)

    occupied_nodes = {node.node_id: node for node in pass_input.nodes}
    for node in result.nodes:
        occupied = occupied_nodes.get(node.node_id)
        if occupied is not None and not _same_node_definition(occupied, node):
            raise _expansion_violation(
                label=label,
                position=position,
                sequence=sequence,
                detail=(
                    f"reassigns the occupied node id {node.node_id!r} to a "
                    "different definition: an existing id may be carried "
                    "forward only for the same semantic node"
                ),
            )

    occupied_roles = _value_roles(pass_input)
    for value_id, role in _value_roles(result).items():
        occupied_role = occupied_roles.get(value_id)
        if occupied_role is not None and occupied_role != role:
            raise _expansion_violation(
                label=label,
                position=position,
                sequence=sequence,
                detail=(
                    f"reassigns the occupied value id {value_id!r} from a "
                    f"{occupied_role} to a {role}: an existing id may be "
                    "carried forward only for the same semantic value"
                ),
            )


def _validate_forward_result(
    result: TensorGraph,
    *,
    traced: TracedLoss,
    captures: SourceCaptures,
    label: str,
    position: int,
) -> None:
    """Require a forward result to still declare and produce what §8.6 names.

    The required set is taken from the **source** artifacts, not from the pass
    input, so the obligation is the same for every pass in the sequence and a
    value dropped by pass 0 is reported against pass 0 rather than surviving
    as an obligation nobody restates.
    """
    declared = set(_declared_input_value_ids(result))
    for value_id in traced.input_value_ids.values():
        if value_id not in declared:
            raise _expansion_violation(
                label=label,
                position=position,
                sequence="forward",
                detail=f"no longer declares the input value {value_id!r}",
            )

    # "Declared or produced" is deliberately the weaker test here: a required
    # value that survived only by turning from a computed value into a free
    # input would pass it. What rejects that is the value-role rule in
    # `_validate_occupied_ids`, which runs first for every pass. The two are
    # one check split across two owners -- do not delete the role rule as
    # redundant on the strength of this one, or the gap opens silently.
    available = _available_value_ids(result)
    for value_id in captures.forward_selected_outputs:
        if value_id not in available:
            raise _expansion_violation(
                label=label,
                position=position,
                sequence="forward",
                detail=(
                    f"no longer produces the value {value_id!r}, which the "
                    "forward phase must retain"
                ),
            )


def _mentions_value(program: DerivativeProgram, value_id: str) -> bool:
    """Whether *program* reads *value_id* anywhere, or reports it as a gradient."""
    if any(value_id in node.input_value_ids for node in program.nodes):
        return True
    return value_id in program.output_gradients


def _validate_derivative_result(
    result: DerivativeProgram,
    *,
    derivative: SourceDerivative,
    label: str,
    position: int,
) -> None:
    """Require a derivative result to keep its gradients and its seed (§8.6).

    Gradient *order* is checked as a whole-sequence equality because order is
    meaning here (Inv-6): `output_gradients` is `wrt` order is `parameters`
    order, and a permutation would silently reroute every gradient the record
    reports. A gradient the source program computed must still be computed;
    one the source did not compute -- a loss that is a declared parameter makes
    the seed itself the gradient -- is not invented as a new obligation.

    The seed is checked in both directions Inv-11 states. It must not be
    produced, as a node id or as a value id, which is the same rule
    `_check_seed_against_derivative_program` applies to the generated program
    -- restated here because every pass is a new opportunity to break it. And
    it must still be *read*, if the source program read it: a program that no
    longer consumes the seed computes something that is no longer the caller's
    derivative, even though nothing about it is structurally malformed.
    """
    if list(result.output_gradients) != list(derivative.program.output_gradients):
        raise _expansion_violation(
            label=label,
            position=position,
            sequence="derivative",
            detail=(
                "no longer reports the gradients "
                f"{list(derivative.program.output_gradients)!r} in the same order"
            ),
        )

    produced = set(_produced_value_ids(result))
    source_produced = set(_produced_value_ids(derivative.program))
    for value_id in derivative.program.output_gradients:
        if value_id in source_produced and value_id not in produced:
            raise _expansion_violation(
                label=label,
                position=position,
                sequence="derivative",
                detail=f"no longer produces the gradient {value_id!r}",
            )

    seed_value_id = derivative.seed_value_id
    for node in result.nodes:
        for produced_id in (node.node_id, node.output_value_id):
            if produced_id == seed_value_id:
                raise _expansion_violation(
                    label=label,
                    position=position,
                    sequence="derivative",
                    detail=(
                        f"produces {seed_value_id!r}, the minted seed, which "
                        "must remain a required free input and never a value "
                        "the program computes"
                    ),
                )

    if _mentions_value(derivative.program, seed_value_id) and not _mentions_value(
        result, seed_value_id
    ):
        raise _expansion_violation(
            label=label,
            position=position,
            sequence="derivative",
            detail=(
                f"no longer reads {seed_value_id!r}, the minted seed the source "
                "program required as a free input"
            ),
        )


def _validate_update_result(
    result: TensorGraph,
    *,
    source_graph: TensorGraph,
    updated_parameter_value_id: str,
    label: str,
    position: int,
) -> None:
    """Require an update result to keep its traced inputs and its output (§8.6)."""
    declared = set(_declared_input_value_ids(result))
    for value_id in _declared_input_value_ids(source_graph):
        if value_id not in declared:
            raise _expansion_violation(
                label=label,
                position=position,
                sequence="update",
                detail=f"no longer declares the input value {value_id!r}",
            )

    if updated_parameter_value_id not in _available_value_ids(result):
        raise _expansion_violation(
            label=label,
            position=position,
            sequence="update",
            detail=(
                f"no longer produces {updated_parameter_value_id!r}, the "
                "updated parameter this update exists to compute"
            ),
        )


def _apply_expansion_sequence(
    expansions: Sequence[Callable[..., object]],
    artifact: object,
    *,
    expected_type: type,
    sequence: str,
    validate: Callable[[object, str, int], None],
) -> tuple[object, tuple[str, ...]]:
    """Apply one sequence in order, validating between passes.

    Two properties of the loop are the contract rather than an implementation
    detail:

    * With no passes the artifact is returned **as it came in** -- the same
      object, never a copy -- which is what makes the whole stage inert by
      default and observable by identity (Inv-10).
    * With at least one pass, the first pass receives a deep copy. A pass is
      caller code and may mutate what it is handed; the source artifacts are
      never rewritten (Inv-1, FR-129-004), and the only way to guarantee that
      against a hostile or careless pass is to keep the source object out of
      its reach entirely. Later passes need no further copy: they already
      receive an artifact that is not the source.

    Validation happens after each pass and before the next one is invoked, so
    a failure names the pass that caused it and no later pass ever sees a
    broken artifact (§8.6).
    """
    if not expansions:
        return artifact, ()

    labels: list[str] = []
    current = copy.deepcopy(artifact)
    for position, expansion in enumerate(expansions):
        label = _pass_label(expansion)
        pass_input = current
        current = _invoke_expansion(
            expansion, pass_input, label=label, position=position, sequence=sequence
        )
        if not isinstance(current, expected_type):
            raise _expansion_violation(
                label=label,
                position=position,
                sequence=sequence,
                detail=(
                    f"returned {type(current).__name__!r}, not the "
                    f"{expected_type.__name__} it was given"
                ),
            )
        _validate_artifact_shape(
            current, label=label, position=position, sequence=sequence
        )
        _validate_occupied_ids(
            current, pass_input, label=label, position=position, sequence=sequence
        )
        validate(current, label, position)
        labels.append(label)

    return current, tuple(labels)


def _applied_passes(labels: Sequence[str]) -> str:
    """Describe an applied sequence for a message that cannot blame one pass."""
    if not labels:
        return "none"
    return "; ".join(
        f"{label!r} at position {position}" for position, label in enumerate(labels)
    )


def _preservation_difference(
    source_analysis: DependencyAnalysis, analysis: DependencyAnalysis
) -> list[str]:
    """The value ids the recomputation and the source analysis disagree about.

    Reported as a sorted set of identifiers rather than two whole dependency
    tuples: the caller needs to know *which* values moved, and the equality
    that failed is asserted over the dependencies themselves, metadata
    included, not over this summary.
    """
    def value_ids(dependency_analysis: DependencyAnalysis) -> set[str]:
        return {
            dependency.value_id
            for dependency in dependency_analysis.forward_captures
            + dependency_analysis.seed_inputs
        }

    return sorted(value_ids(source_analysis).symmetric_difference(value_ids(analysis)))


def expand_source_artifacts(
    *,
    traced: TracedLoss,
    derivative: SourceDerivative,
    captures: SourceCaptures,
    forward_expansions: Sequence[Callable[[TensorGraph], TensorGraph]] = (),
    derivative_expansions: Sequence[
        Callable[[DerivativeProgram], DerivativeProgram]
    ] = (),
) -> ExpandedArtifacts:
    """Apply both source expansion sequences and prove preservation, per §8.6.

    The forward sequence runs first and completely, then the derivative
    sequence, then the recomputation -- the order of §7.2's items 8, 9, and 10.
    A forward failure therefore leaves every derivative pass uninvoked, which
    is the point: a pass is caller code, and the framework does not run more of
    it after the contract is already broken.

    The recomputation is the second half of one preservation contract. The
    per-pass checks catch a structurally broken artifact before the next pass
    compounds it; only this recomputation catches a pass that stayed
    structurally valid while changing *which* forward values the derivative
    reads -- a rewritten region that reads a different capture, or one that
    leaves the seed mentioned but no longer reaching any gradient. The ordered
    capture set and the seed-input set must equal the source analysis of §8.5
    exactly, dependency for dependency, and an inequality is this stage's own
    `expansion_contract_violation`.

    A failure raised by `analyze_derivative_dependencies` itself -- a
    `missing_dependency`, an `ambiguous_producer` -- propagates unchanged. Those
    categories belong to the analysis (§13.3); it is reporting on the artifact
    it was given, not a pass breaching the contract §13.2 wraps, and its
    message already names the offending value more precisely than a re-wording
    could.

    The inequality message names every applied pass with its position rather
    than one culprit. After two sequences have run, responsibility genuinely is
    not attributable to a single pass -- and with exactly one applied pass, the
    common case, the message names exactly that pass and its position.
    """
    lowered_forward_graph, forward_pass_labels = _apply_expansion_sequence(
        forward_expansions,
        traced.graph,
        expected_type=TensorGraph,
        sequence="forward",
        validate=lambda result, label, position: _validate_forward_result(
            result, traced=traced, captures=captures, label=label, position=position
        ),
    )

    lowered_derivative_program, derivative_pass_labels = _apply_expansion_sequence(
        derivative_expansions,
        derivative.program,
        expected_type=DerivativeProgram,
        sequence="derivative",
        validate=lambda result, label, position: _validate_derivative_result(
            result, derivative=derivative, label=label, position=position
        ),
    )

    analysis = analyze_derivative_dependencies(
        lowered_derivative_program,
        forward_graph=lowered_forward_graph,
        seed_value_ids=[derivative.seed_value_id],
        outputs=list(derivative.program.output_gradients),
    )

    source_analysis = captures.analysis
    if (
        analysis.forward_captures != source_analysis.forward_captures
        or analysis.seed_inputs != source_analysis.seed_inputs
    ):
        difference = _preservation_difference(source_analysis, analysis)
        raise AutodiffError(
            "expansion_contract_violation",
            "expansion did not preserve the values the derivative needs: the "
            "recomputed forward-capture and seed-input sets differ from the "
            f"source analysis at {difference!r}"
            f"; applied forward passes: {_applied_passes(forward_pass_labels)}"
            f"; applied derivative passes: {_applied_passes(derivative_pass_labels)}",
        )

    return ExpandedArtifacts(
        lowered_forward_graph=lowered_forward_graph,
        lowered_derivative_program=lowered_derivative_program,
        analysis=analysis,
        forward_pass_labels=forward_pass_labels,
        derivative_pass_labels=derivative_pass_labels,
    )


def expand_update_graph(
    *,
    graph: TensorGraph,
    updated_parameter_value_id: str,
    expansions: Sequence[Callable[[TensorGraph], TensorGraph]] = (),
) -> ExpandedUpdate:
    """Apply one parameter's update expansion sequence, per §8.6 and §8.7.

    The same identifier and collaborator rules as the source sequences, with
    the update artifact's own semantic requirement: every input the update was
    traced with is still declared, and the updated parameter is still produced.
    There is no recomputation here -- an update graph has no derivative
    dependency analysis to preserve -- so the per-pass check is the whole
    contract.

    *graph* is the traced update graph and *updated_parameter_value_id* the
    single value that update computes; how both were produced belongs to the
    tracing stage, and this stage states its contract over the artifact alone.
    """
    lowered_graph, pass_labels = _apply_expansion_sequence(
        expansions,
        graph,
        expected_type=TensorGraph,
        sequence="update",
        validate=lambda result, label, position: _validate_update_result(
            result,
            source_graph=graph,
            updated_parameter_value_id=updated_parameter_value_id,
            label=label,
            position=position,
        ),
    )
    return ExpandedUpdate(lowered_graph=lowered_graph, pass_labels=pass_labels)


@dataclass(frozen=True)
class ParameterCompilation:
    """Everything one declared parameter contributes to a compiled step.

    One of these exists per declared parameter, always, in `parameters` order
    -- one parameter is a tuple of length one and nothing else about the
    surrounding record changes (§9.4, Inv-8). There is deliberately no scalar
    counterpart anywhere in the record.

    ``value_id`` is the parameter's own value id in the forward graph and
    ``gradient_value_id`` the source derivative program's gradient for exactly
    that value -- keyed by the value, never by position, so a mis-wiring
    between two parameters is impossible rather than merely unlikely (Inv-6).

    ``update_input_value_ids`` is the traced update's own per-name mapping,
    carried through unmodified: ``"parameter"``, ``"gradient"``, and one entry
    per declared optimizer input. A consumer binds runtime values by the name
    it chose, never by scanning nodes or parsing an identifier.

    ``source_update_graph`` and ``lowered_update_graph`` are the same object
    when no update expansion pass was supplied, for the reason
    `ExpandedUpdate` states. Per-value dependency provenance is not restated
    here: it is reachable through ``update.dependencies`` (FR-129-013).
    """

    name: str
    value_id: str
    gradient_value_id: str
    source_update_graph: TensorGraph
    lowered_update_graph: TensorGraph
    update: LoweredProgram
    update_input_value_ids: Mapping[str, str]
    updated_parameter_value_id: str


@dataclass(frozen=True)
class TrainingStepProvenance:
    """Enough to identify what produced a record, and to detect a changed one.

    Three sources, and no fourth: the derivative metadata (the source graph id
    and the two contract versions), the caller's own ordered declarations
    (parameters, inputs, seed label, and the optimizer's identity), and the
    labels of the expansion passes that actually ran.

    ``wrt_signature`` is the parameter *value ids* in order and
    ``parameter_names`` the parameter *names* in the same order; together with
    `CompiledTrainingStep.parameters` and `derivative.selected_outputs` they
    are the four carriers Inv-6 requires to agree.

    ``optimizer_input_names`` is read off the optimizer's own declaration
    rather than off the caller's mapping (FR-129-014). Validation has already
    proven the two name *sets* agree, so only their order could differ, and
    the requirement names the optimizer as the source.

    ``update_expansions`` records the update sequence's labels **once**. The
    same sequence is applied once per parameter, and repeating the labels per
    parameter would make the record's shape vary with parameter count, which
    Inv-8 forbids.

    A label is not an identifier: two passes may legitimately carry the same
    one, so nothing may key off these tuples (§9.1).
    """

    source_graph_id: str
    transform_version: str
    tensor_op_contract_version: str
    wrt_signature: tuple[str, ...]
    parameter_names: tuple[str, ...]
    input_names: tuple[str, ...]
    seed_value_ids: tuple[str, ...]
    seed_label: str
    optimizer_label: str
    optimizer_input_names: tuple[str, ...]
    forward_expansions: tuple[str, ...]
    derivative_expansions: tuple[str, ...]
    update_expansions: tuple[str, ...]


@dataclass(frozen=True)
class CompiledTrainingStep:
    """One compiled training step: a framework envelope over opaque values.

    The structure, the field names, the ordering, and every identifier in this
    record are framework-owned and backend-neutral. The `LoweredProgram`
    values inside it are not: they hold target values the consumer's own
    handlers produced, which this module carried without importing,
    inspecting, comparing, hashing, or iterating (Inv-3). The record is
    therefore not portable between backends, not comparable across consumers,
    and not serializable, and nothing here claims otherwise (§9.3, NG-7).

    Four artifacts are named separately so a consumer can tell what was
    traced, what was differentiated, and what was actually lowered, and can
    diff the pair when an expansion is in play. With no passes supplied the
    ``source_*`` and ``lowered_*`` members are the *same objects*, which is
    itself observable (Inv-10).

    ``loss_value_id`` and ``forward_capture_value_ids`` are reported
    separately and never conflated (FR-129-007): the first is what the step
    reports to the application, the second what the forward phase must retain
    for the backward phase. ``forward.selected_outputs`` is exactly
    ``(loss_value_id,) + forward_capture_value_ids`` (Inv-7), and
    ``derivative.selected_outputs`` the gradient value ids in ``parameters``
    order.

    Per-value dependency and capture provenance is deliberately **not**
    restated: it is already carried per program by `LoweredProgram`'s own
    ``dependencies``, on ``forward``, on ``derivative``, and on each
    parameter's ``update`` (FR-129-013). The capture *identifiers* appear
    because selecting them is an output obligation of this compiler, not a
    re-export of the analysis.
    """

    source_forward_graph: TensorGraph
    source_derivative_program: DerivativeProgram
    lowered_forward_graph: TensorGraph
    lowered_derivative_program: DerivativeProgram
    forward: LoweredProgram
    derivative: LoweredProgram
    input_value_ids: Mapping[str, str]
    loss_value_id: str
    forward_capture_value_ids: tuple[str, ...]
    seed_value_ids: tuple[str, ...]
    parameters: tuple[ParameterCompilation, ...]
    provenance: TrainingStepProvenance

    def parameter(self, name: str) -> ParameterCompilation:
        """Return the compilation of the parameter called *name*.

        The by-name accessor §9.4 puts in place of a scalar convenience field.
        A name that was not declared as a parameter -- including a declared
        *input* that is not one -- is a declaration mistake and is reported as
        one, with the names that were declared (FR-129-015).
        """
        for compiled in self.parameters:
            if compiled.name == name:
                return compiled
        raise AutodiffError(
            "invalid_training_declaration",
            f"{name!r} is not a declared parameter of this training step; "
            "the declared parameters are "
            f"{tuple(compiled.name for compiled in self.parameters)!r}",
        )


def compile_training_step(
    loss: Callable[..., object],
    *,
    inputs: Mapping[str, Mapping[str, object]],
    parameters: Sequence[str],
    optimizer: Optimizer,
    optimizer_inputs: Optional[Mapping[str, Mapping[str, object]]] = None,
    handlers: OperationHandlerRegistry,
    fusion: Optional[FusionHook] = None,
    bind_input: Optional[Callable[[ValueDependency], object]] = None,
    seed_label: str = "seed",
    forward_expansions: Sequence[Callable[[TensorGraph], TensorGraph]] = (),
    derivative_expansions: Sequence[
        Callable[[DerivativeProgram], DerivativeProgram]
    ] = (),
    update_expansions: Sequence[Callable[[TensorGraph], TensorGraph]] = (),
) -> CompiledTrainingStep:
    """Compile a framework-traced loss into one lowered training step.

    The ordered composition of every stage this module owns, in the order
    FR-129-001 and §13.4 state and in no other: declaration and optimizer
    validation, loss-signature binding, one typed trace, seed minting, the VJP
    from the source graph, the post-generation collision check, source
    dependency analysis and capture determination, the expansion sequences
    with their preservation check, forward lowering, derivative lowering, the
    per-parameter update cycle, and finally the record.

    Nothing here re-implements a check another owner already performs. Every
    failure a caller can provoke is raised by the stage that owns it, with
    that stage's own category and message: a malformed declaration or optimizer
    contract by the validators of `validate_declaration`, a wrong loss
    signature or return by `trace_loss`, a seed collision by
    `differentiate_loss`, a pass breaching its contract by
    `expand_source_artifacts` or `expand_update_graph`, and a malformed
    `handlers` or `fusion` by the lowering module's own validation as
    `handler_contract_violation` (FR-129-021). An exception raised inside the
    loss body or inside the optimizer's `update` body is application logic and
    propagates entirely unchanged (§13.2).

    `handlers`, `fusion`, and `bind_input` are passed to all three lowerings
    exactly as given -- the same objects, never wrapped, filtered, or replaced
    by a default (Inv-9). The forward lowering selects the loss followed by
    every capture; the derivative lowering receives the *lowered* forward
    graph and the minted seed and selects the gradients in `parameters` order;
    each update lowering selects exactly that update's own updated-parameter
    value (FR-129-008).

    Per parameter, in order, the optimizer update is traced with that
    parameter's **own** declared dtype and shape for both the `parameter` and
    the `gradient` typed input, with the declared optimizer inputs shared
    across parameters (§8.7).

    The record is built once, at the end, from local values: a failure at any
    stage raises and leaves nothing partially assembled.
    """
    validate_declaration(
        inputs=inputs,
        parameters=parameters,
        loss=loss,
        optimizer=optimizer,
        optimizer_inputs=optimizer_inputs,
    )

    # Declaration order for the inputs, caller order for the parameters. The
    # second is the `wrt` order, and therefore the gradient order, the record
    # order, and `wrt_signature`'s order -- the single chain Inv-6 rests on.
    input_names = tuple(inputs)
    parameter_names = tuple(parameters)

    traced = trace_loss(inputs=inputs, input_names=input_names, loss=loss)
    derivative = differentiate_loss(
        traced=traced, parameters=parameter_names, seed_label=seed_label
    )
    captures = analyze_source_captures(traced=traced, derivative=derivative)
    expanded = expand_source_artifacts(
        traced=traced,
        derivative=derivative,
        captures=captures,
        forward_expansions=forward_expansions,
        derivative_expansions=derivative_expansions,
    )

    forward = lower_graph(
        expanded.lowered_forward_graph,
        handlers=handlers,
        outputs=list(captures.forward_selected_outputs),
        fusion=fusion,
        bind_input=bind_input,
    )

    # Keyed by each parameter's own forward value, never by position: this is
    # the one place a two-parameter step could silently swap its gradients.
    gradient_value_ids = [
        derivative.program.gradients[traced.input_value_ids[name]]
        for name in parameter_names
    ]

    lowered_derivative = lower_derivative_program(
        expanded.lowered_derivative_program,
        forward_graph=expanded.lowered_forward_graph,
        seed_value_ids=[derivative.seed_value_id],
        handlers=handlers,
        outputs=gradient_value_ids,
        fusion=fusion,
        bind_input=bind_input,
    )

    compiled_parameters: list[ParameterCompilation] = []
    update_pass_labels: tuple[str, ...] = ()
    for name, gradient_value_id in zip(parameter_names, gradient_value_ids):
        # Both typed inputs take *this* parameter's declared spec. A gradient
        # has the shape of the value it is a gradient of, so reusing the first
        # parameter's spec would mis-type every later one.
        declared_spec = inputs[name]
        traced_update = trace_parameter_update(
            optimizer,
            parameter=declared_spec,
            gradient=declared_spec,
            optimizer_inputs=optimizer_inputs,
        )
        expanded_update = expand_update_graph(
            graph=traced_update.graph,
            updated_parameter_value_id=traced_update.updated_parameter_id,
            expansions=update_expansions,
        )
        update = lower_graph(
            expanded_update.lowered_graph,
            handlers=handlers,
            outputs=[traced_update.updated_parameter_id],
            fusion=fusion,
            bind_input=bind_input,
        )
        # The same sequence runs for every parameter, so every iteration
        # resolves the same labels; provenance records them once (Inv-8).
        update_pass_labels = expanded_update.pass_labels
        compiled_parameters.append(
            ParameterCompilation(
                name=name,
                value_id=traced.input_value_ids[name],
                gradient_value_id=gradient_value_id,
                source_update_graph=traced_update.graph,
                lowered_update_graph=expanded_update.lowered_graph,
                update=update,
                update_input_value_ids=MappingProxyType(
                    dict(traced_update.input_value_ids)
                ),
                updated_parameter_value_id=traced_update.updated_parameter_id,
            )
        )

    metadata = derivative.program.metadata
    provenance = TrainingStepProvenance(
        source_graph_id=metadata.source_graph_id,
        transform_version=metadata.transform_version,
        tensor_op_contract_version=metadata.tensor_op_contract_version,
        wrt_signature=tuple(metadata.wrt_signature),
        parameter_names=parameter_names,
        input_names=input_names,
        seed_value_ids=(derivative.seed_value_id,),
        seed_label=seed_label,
        optimizer_label=type(optimizer).__name__,
        optimizer_input_names=_required_optimizer_input_names(optimizer),
        forward_expansions=expanded.forward_pass_labels,
        derivative_expansions=expanded.derivative_pass_labels,
        update_expansions=update_pass_labels,
    )

    return CompiledTrainingStep(
        source_forward_graph=traced.graph,
        source_derivative_program=derivative.program,
        lowered_forward_graph=expanded.lowered_forward_graph,
        lowered_derivative_program=expanded.lowered_derivative_program,
        forward=forward,
        derivative=lowered_derivative,
        input_value_ids=MappingProxyType(dict(traced.input_value_ids)),
        loss_value_id=traced.loss_value_id,
        forward_capture_value_ids=captures.forward_capture_value_ids,
        seed_value_ids=(derivative.seed_value_id,),
        parameters=tuple(compiled_parameters),
        provenance=provenance,
    )
