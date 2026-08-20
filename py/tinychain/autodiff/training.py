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
* **Typed input completeness.** Delegated entirely to
  `TensorGraphBuilder.input`'s existing dtype/shape validation -- this module
  does not re-validate a type spec mapping.
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
    input specs (``{"dtype": ..., "shape": ...}``) forwarded verbatim to
    :meth:`TensorGraphBuilder.input`. ``update`` is called once, by keyword,
    with a ``Tensor`` for ``parameter``, a ``Tensor`` for ``gradient``, and a
    ``Tensor`` for each declared optimizer input; it must return the single
    updated-parameter ``Tensor``, expressed with ordinary Tensor operations.
    """
    resolved_optimizer_inputs = optimizer_inputs or {}
    _validate_update_signature(
        update, parameter=parameter, gradient=gradient, optimizer_inputs=resolved_optimizer_inputs
    )

    # Deferred import: importing collection.tensor at module scope would
    # initialize Tensor, whose recorder imports this package for concrete
    # operator identities (same rationale as TensorGraphBuilder.input).
    from ..collection.tensor import Tensor

    with TensorGraphBuilder() as builder:
        parameter_tensor = builder.input("parameter", **dict(parameter))
        gradient_tensor = builder.input("gradient", **dict(gradient))
        optimizer_tensors = {
            name: builder.input(name, **dict(spec))
            for name, spec in resolved_optimizer_inputs.items()
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
