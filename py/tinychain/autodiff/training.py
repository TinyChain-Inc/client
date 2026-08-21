"""Composition helper for tracing parameter updates as ordinary Tensor code.

An optimizer update is, structurally, no different from an application loss:
it is a typed Tensor expression over declared inputs. This module supplies the
one piece that is otherwise missing from the public tracing surface -- typed
parameter/gradient/optimizer inputs and a single selected updated-parameter
output -- so an update can be authored and traced entirely with ordinary
`Tensor` operations (spec invariant 6) and then handed to the same structured
dependency analysis and extensible lowering seam as any other traced graph.

This module defines no optimizer catalog, no state lifecycle, and no consumer
policy. `sgd_update` below is the one reference expression the Issue 86 MVP
needs: `parameter - learning_rate * gradient`, written with the `-`/`*`
operators `Tensor` already supports -- nothing here constructs a
`TensorNodeRecord` or a concrete `TensorOperator` directly.

Named invariants and where each is enforced (spec-driven, each in one place):

* **Update-callable well-formedness.** The callable's signature must accept
  exactly the declared keyword inputs. Checked once, before the builder is
  entered, in :func:`_validate_update_signature` -- this is what makes
  "invalid update callables fail before consumer execution" a structural
  property rather than an accident of statement order: the callable's body
  cannot run before its signature has been validated.
* **Traced output validity.** The callable must return a `Tensor`. Checked
  once, immediately after invoking the callable and before `builder.build`.
* **Typed input spec well-formedness.** Each declared spec must be a mapping
  providing a ``dtype`` and a ``shape``. Checked once, in
  :func:`_typed_input_spec`, before the builder is entered. This is *not*
  delegated to `TensorGraphBuilder.input`, and cannot be: the specs used to be
  unpacked into it as ``**dict(spec)``, so a spec that was not a mapping, or
  that named its keys wrongly, failed in the unpack -- before the builder was
  ever reached -- and a keyword-argument `TypeError` carries no category. The
  spec is therefore read by key instead, exactly as the structured dependency
  analysis reads a type spec, and an unrecognized extra key is ignored for the
  same reason it is ignored there: one spec must not be accepted by the
  analysis and rejected by the tracer.
* **Declared optimizer inputs.** ``optimizer_inputs`` declares *which*
  keyword inputs exist, so a container that is not a mapping leaves the
  declared input set unestablished and the update callable's signature
  uncheckable. Checked once, in :func:`_resolve_optimizer_inputs`.
* **Typed input dtype and shape validity.** The dtype *value* is delegated
  entirely to `TensorGraphBuilder.input`'s existing
  `check_differentiable_dtype`. The shape's rank and dimensions are checked
  against `shape.parse_shape`, the same shared parser the dependency analysis
  uses, so the builder's own normalization can no longer be reached with a
  value it would reject with an uncategorized `ValueError`.
* **Shape/dtype compatibility of the traced expression** (for example a
  gradient shape incompatible with the parameter shape). Delegated entirely to
  the existing typed-tracing Sub/Mul shape inference; this module does not
  re-check operand shapes.
* **Inactive-tracing precondition.** Tracing must start with no builder
  already active. Delegated entirely to `TensorGraphBuilder`'s existing
  nested-context guard.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Optional

from .graph import TensorGraph, TensorGraphBuilder
from .protocol import AutodiffError
from .shape import parse_shape


@dataclass(frozen=True)
class TracedUpdate:
    """The finalized typed graph produced by tracing a parameter update.

    ``input_value_ids`` maps each declared input name (``"parameter"``,
    ``"gradient"``, and every ``optimizer_inputs`` key) to its stable value id
    in ``graph``, so a consumer binds runtime values by name instead of
    scanning ``graph.inputs`` or inferring order.
    """

    graph: TensorGraph
    updated_parameter_id: str
    input_value_ids: Mapping[str, str]


def trace_parameter_update(
    update: Callable[..., object],
    *,
    parameter: Mapping[str, object],
    gradient: Mapping[str, object],
    optimizer_inputs: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> TracedUpdate:
    """Trace an ordinary Tensor *update* callable into a finalized typed graph.

    ``parameter``, ``gradient``, and each ``optimizer_inputs`` value are typed
    input specs (``{"dtype": ..., "shape": ...}``), read by key and passed on
    to :meth:`TensorGraphBuilder.input`; an unrecognized extra key is ignored.
    ``update`` is called once, by keyword, with a ``Tensor`` for
    ``parameter``, a ``Tensor`` for ``gradient``, and a ``Tensor`` for each
    declared optimizer input; it must return the single updated-parameter
    ``Tensor``, expressed with ordinary Tensor operations.

    Every declared spec is validated before the builder is entered, so a
    malformed declaration never reaches the consumer's callable body.
    """
    resolved_optimizer_inputs = _resolve_optimizer_inputs(optimizer_inputs)
    _validate_update_signature(
        update, parameter=parameter, gradient=gradient, optimizer_inputs=resolved_optimizer_inputs
    )
    # Ahead of the builder, like the signature check above and for the same
    # reason: failing before any trace begins is then a structural property
    # rather than an accident of statement order inside the trace.
    declared_specs = {
        "parameter": _typed_input_spec(parameter, label="parameter"),
        "gradient": _typed_input_spec(gradient, label="gradient"),
    }
    for name, spec in resolved_optimizer_inputs.items():
        declared_specs[name] = _typed_input_spec(spec, label=name)

    # Deferred import: importing collection.tensor at module scope would
    # initialize Tensor, whose recorder imports this package for concrete
    # operator identities (same rationale as TensorGraphBuilder.input).
    from ..collection.tensor import Tensor

    with TensorGraphBuilder() as builder:
        parameter_tensor = builder.input("parameter", **declared_specs["parameter"])
        gradient_tensor = builder.input("gradient", **declared_specs["gradient"])
        optimizer_tensors = {
            name: builder.input(name, **declared_specs[name])
            for name in resolved_optimizer_inputs
        }
        updated_parameter = update(
            parameter=parameter_tensor,
            gradient=gradient_tensor,
            **optimizer_tensors,
        )

    if not isinstance(updated_parameter, Tensor):
        raise AutodiffError(
            "invalid_update_output",
            "update callable must return a Tensor, got "
            f"{type(updated_parameter).__name__!r}",
        )

    graph = builder.build(outputs=[updated_parameter])

    input_value_ids = {
        "parameter": builder.value_id(parameter_tensor),
        "gradient": builder.value_id(gradient_tensor),
    }
    for name, tensor in optimizer_tensors.items():
        input_value_ids[name] = builder.value_id(tensor)

    return TracedUpdate(
        graph=graph,
        updated_parameter_id=builder.value_id(updated_parameter),
        input_value_ids=input_value_ids,
    )


def _resolve_optimizer_inputs(optimizer_inputs: object) -> Mapping[str, object]:
    """Normalize the declared optimizer inputs, rejecting a malformed container.

    ``optimizer_inputs`` declares *which* keyword inputs exist, so a container
    that is not a mapping leaves the declared input set unestablished and the
    update callable's signature uncheckable -- which is the failure
    ``invalid_update_signature`` owns. The category name points a reader at
    the callable, so the message says plainly that the argument is at fault
    and the callable is not.
    """
    if optimizer_inputs is None:
        return {}
    if not isinstance(optimizer_inputs, Mapping):
        raise AutodiffError(
            "invalid_update_signature",
            "the optimizer_inputs argument is at fault here, not the update "
            "callable: optimizer_inputs must be a mapping of input name to "
            f"typed input spec, got {type(optimizer_inputs).__name__!r}; the "
            "declared input set cannot be established, so the callable's "
            "signature cannot be checked against it",
        )
    return optimizer_inputs


def _typed_input_spec(spec: object, *, label: str) -> dict[str, object]:
    """Read one declared typed input spec, failing closed with a category.

    Mirrors ``dependencies._complete_typespec``: the spec is read by key
    rather than unpacked, an absent or unreadable ``dtype`` is
    ``missing_dtype_metadata``, an absent or malformed ``shape`` is
    ``missing_shape_metadata``, and an extra key is ignored. A consumer then
    meets one category per class of mistake wherever they hit it.

    The dtype *value* is deliberately not judged here. `TensorGraphBuilder`'s
    ``check_differentiable_dtype`` already categorizes it, and re-checking
    would give one mistake two decision sites. The shape *is* checked here,
    because the builder's normalization rejects a malformed dimension with an
    uncategorized `ValueError`; ``parse_shape`` is the shared parser the
    dependency analysis uses and accepts exactly what the builder accepts.
    """
    if not isinstance(spec, Mapping):
        raise AutodiffError(
            "missing_dtype_metadata",
            f"typed input {label!r} must be declared as a mapping of 'dtype' "
            f"and 'shape', got {type(spec).__name__!r}",
        )
    if "dtype" not in spec:
        raise AutodiffError(
            "missing_dtype_metadata",
            f"typed input {label!r} declares no 'dtype'; it has "
            f"{sorted(str(key) for key in spec)}",
        )
    if "shape" not in spec:
        raise AutodiffError(
            "missing_shape_metadata",
            f"typed input {label!r} declares no 'shape'; it has "
            f"{sorted(str(key) for key in spec)}",
        )
    parse_shape(spec["shape"], label=f"typed input {label!r} shape")
    return {"dtype": spec["dtype"], "shape": spec["shape"]}


def _validate_update_signature(
    update: Callable[..., object],
    *,
    parameter: Mapping[str, object],
    gradient: Mapping[str, object],
    optimizer_inputs: Mapping[str, Mapping[str, object]],
) -> None:
    """Require *update* to accept exactly the declared typed inputs by keyword.

    This is the single point where update-callable well-formedness is
    enforced, and it runs before any builder is entered or any typed input is
    declared, so a rejected callable never has its body invoked (AC4).
    """
    required_names = ("parameter", "gradient", *optimizer_inputs.keys())
    try:
        signature = inspect.signature(update)
        signature.bind(**dict.fromkeys(required_names))
    except TypeError as exc:
        raise AutodiffError(
            "invalid_update_signature",
            "update callable must accept exactly the declared typed inputs "
            f"{required_names!r} by keyword: {exc}",
        ) from exc


def sgd_update(*, parameter: object, gradient: object, learning_rate: object) -> object:
    """Reference SGD update expressed with ordinary Tensor operations.

    ``parameter - learning_rate * gradient``, written with the `Tensor`
    `-`/`*` operators -- no `TensorNodeRecord` or `TensorOperator` is
    constructed here.
    """
    return parameter - learning_rate * gradient
