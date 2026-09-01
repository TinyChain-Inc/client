"""Shared numeric harness for the training-step end-to-end tests.

`test_autodiff_training_step_end_to_end` proves the compiled step computes the
right numbers, and `test_autodiff_training_step_expansion_composition` proves
it still computes them when both of #128's real expansion passes are in play
and no reduction handler is available. The two ask different questions, but
they answer them against the same declarations and the same hand-written
reference calculus -- and that half was copied between the files. This module
owns the single copy, so a change to the reference cannot land in one file and
leave the other certifying the old arithmetic.

What is deliberately *not* here: each file's loss callables, which differ in
the reduction they use (`keepdims=True` keeps the composition file inside
#128's rank-preserving tier), and each file's `execute_step`, which differs in
the registry it lowers through and in how it binds the seed. The composition
file's execution is re-lowered through a registry with no reduction,
broadcast, or division handler -- that omission is its entire experiment, and
folding the two executions together would weaken exactly the claim it makes.

The reference functions read no compiled artifact: `L = mean(d * d)` for
`d = x @ w (+ b) - y`, written straight from the calculus, so a framework bug
cannot cancel itself out against the expected value.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np


# --------------------------------------------------------------------------
# declarations
#
# `x` is 3x2 and `w` is 2x4, so the residual is 3x4: every shape is
# asymmetric, and a transposed matmul anywhere on the path gives a shape error
# or a detectably wrong answer rather than a plausible one. The batch
# dimension is 3, greater than one.
# --------------------------------------------------------------------------

SCALAR_SPEC: Mapping[str, object] = {"dtype": "f64", "shape": []}

ONE_PARAMETER_INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": {"dtype": "f64", "shape": (3, 2)},
    "y": {"dtype": "f64", "shape": (3, 4)},
    "w": {"dtype": "f64", "shape": (2, 4)},
}

TWO_PARAMETER_INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": {"dtype": "f64", "shape": (3, 2)},
    "y": {"dtype": "f64", "shape": (3, 4)},
    "w": {"dtype": "f64", "shape": (2, 4)},
    "b": {"dtype": "f64", "shape": (3, 4)},
}

# Element count of the residual, the divisor the mean's derivative carries.
RESIDUAL_SIZE = 3 * 4


# --------------------------------------------------------------------------
# concrete arrays and the reference calculus
# --------------------------------------------------------------------------


def concrete_inputs(*, with_bias: bool, seed: int) -> dict[str, np.ndarray]:
    """Fixed, non-degenerate `f64` arrays for one run.

    *seed* is required rather than defaulted: each importing file pins its own
    generator, so the two suites are not silently asserting over one shared
    draw.
    """
    generator = np.random.default_rng(seed)
    values = {
        "x": generator.normal(size=(3, 2)),
        "y": generator.normal(size=(3, 4)),
        "w": generator.normal(size=(2, 4)),
    }
    if with_bias:
        values["b"] = generator.normal(size=(3, 4))
    return values


def reference_gradients(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`dL/dw` and, when a bias is declared, `dL/db`, computed directly.

    Written straight from the calculus of the loss, with no reference to any
    compiled artifact: `L = mean(d * d)` for `d = x @ w (+ b) - y`, so
    `dL/dd = 2 * d / N`, `dL/dw = x.T @ dL/dd`, and `dL/db = dL/dd`.
    """
    residual = values["x"] @ values["w"] - values["y"]
    if "b" in values:
        residual = residual + values["b"]
    residual_gradient = 2.0 * residual / RESIDUAL_SIZE
    gradients = {"w": values["x"].T @ residual_gradient}
    if "b" in values:
        gradients["b"] = residual_gradient
    return gradients


def reference_loss(values: Mapping[str, np.ndarray]) -> float:
    residual = values["x"] @ values["w"] - values["y"]
    if "b" in values:
        residual = residual + values["b"]
    return float(np.mean(residual * residual))


def placeholder_binding(dependency: object) -> np.ndarray:
    """A ones array of the dependency's own declared shape.

    Used only by the compile phase, whose numbers no assertion reads. Driving
    it off the framework's analyzed shape rather than off a hand-written table
    keeps the compile working for whichever free dependencies a program
    actually has.
    """
    shape = tuple(int(dimension) for dimension in (dependency.shape or ()))
    return np.ones(shape, dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class ExecutedStep:
    """Everything one executed training step produced."""

    loss: float
    gradients: Mapping[str, np.ndarray]
    updated: Mapping[str, np.ndarray]
