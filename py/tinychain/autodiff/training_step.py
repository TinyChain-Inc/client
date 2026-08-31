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
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Optional

from .protocol import AutodiffError
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
