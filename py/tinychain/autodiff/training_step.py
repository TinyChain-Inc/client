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
"""

from __future__ import annotations

import inspect
import keyword
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from .graph import TensorGraph, TensorGraphBuilder
from .protocol import AutodiffError
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

    graph = builder.build(outputs=[result])

    input_value_ids = {
        name: builder.value_id(tensor) for name, tensor in input_tensors.items()
    }

    return TracedLoss(
        graph=graph,
        loss_value_id=builder.value_id(result),
        input_value_ids=input_value_ids,
    )
