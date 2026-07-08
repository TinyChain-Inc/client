from __future__ import annotations

from . import generate
from .graph import TensorGraph
from .protocol import AutodiffError
from .routes import discover_route_derivative


def _normalize_wrt(wrt: object) -> list[str]:
    if wrt is None:
        raise TypeError("tc.grad requires `wrt` value ids")

    if isinstance(wrt, str):
        names = [wrt]
    elif isinstance(wrt, (list, tuple)):
        names = list(wrt)
    else:
        raise TypeError("tc.grad `wrt` must be a value id string or sequence of value id strings")

    if not names:
        raise TypeError("tc.grad `wrt` must not be empty")

    for name in names:
        if not isinstance(name, str) or not name:
            raise TypeError("tc.grad `wrt` entries must be non-empty value id strings")

    return names


def _is_bound_route_target(target: object) -> bool:
    return (
        callable(target)
        and getattr(target, "__tc_route__", None) is not None
        and getattr(target, "__tc_instance__", None) is not None
    )


def grad(
    target: object,
    *,
    wrt: object = None,
    seed: str = "seed",
    output_value_id: str | None = None,
    seed_typespec: dict[str, object] | None = None,
) -> object:
    """Generate a derivative program or route derivative discovery plan.

    TensorGraph targets use the Python-owned VJP engine. Bound TinyChain route
    targets use local route derivative metadata discovery. Other call-site
    forms fail clearly until route tracing/final API work lands.
    """

    if callable(target) and wrt is None:
        raise TypeError("tc.grad is a call-site transform and cannot be used as a route decorator")

    if isinstance(target, TensorGraph):
        if wrt is None:
            raise TypeError("tc.grad requires wrt for TensorGraph targets")
        selected_output = output_value_id
        if selected_output is None:
            if not target.outputs:
                raise AutodiffError("malformed_derivative_ir", "TensorGraph has no outputs")
            selected_output = target.outputs[0]
        return generate(
            target,
            selected_output,
            list(_normalize_wrt(wrt)),
            seed,
            seed_typespec=seed_typespec,
        )

    if _is_bound_route_target(target):
        return discover_route_derivative(
            target,
            wrt=wrt,
            seed=seed,
            seed_typespec=seed_typespec,
        )

    raise AutodiffError(
        "autodiff_not_implemented",
        "tc.grad currently requires a Python-owned TensorGraph or bound route target",
    )
