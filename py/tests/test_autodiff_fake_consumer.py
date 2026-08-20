"""A fake, non-ILC consumer proving the extensible lowering seam is generic.

Spec acceptance criterion 3 requires a fake consumer that lowers at least two
concrete operators and one supported fusion through the framework traversal
seam, using no framework-private access and no ILC concept. This module is
that proof: a consumer with no framework-private access and no target-specific
concept can lower two concrete operators and one supported fusion into its own
representation, reaching the seam through public `tinychain.autodiff` names
alone.

`FakeExpr` below is this test's own throwaway expression tree -- it shares no
type, no naming, and no representation with any encrypted or ILC target IR.
This module demonstrates that the seam is *usable* generically; it is not a
regression guard against the framework *becoming* coupled to a specific
consumer's concept after the fact. A framework change that hard-codes
consumer-specific behavior on the lowering path while leaving the public
names and their signatures alone would not be caught by this module -- this
consumer would keep passing because its own inputs and expectations never
changed, not because the framework stayed generic. Import-level coupling is
what `test_autodiff_no_forbidden_dependencies.py` checks (and only for a
static, name-based import; see that module's own documented limits).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tinychain.autodiff import (
    AddOperator,
    MulOperator,
    SubOperator,
    TensorGraphBuilder,
)
from tinychain.autodiff import (
    FusionContext,
    FusionResult,
    OperationContext,
    OperationHandlerRegistry,
    lower_graph,
)


@dataclass(frozen=True)
class FakeExpr:
    """One node of this consumer's own target expression tree.

    Nothing about this shape -- a tagged n-ary operator string plus operand
    tuple -- has any relationship to a route name, an ILC instruction, or an
    encrypted physical layout. It exists only to prove the seam is generic.
    """

    op: str
    operands: tuple[object, ...]


class _FakeAddHandler:
    """Lowers a concrete Add operation into this consumer's own representation."""

    operator_type = AddOperator

    def lower(self, context: OperationContext) -> FakeExpr:
        return FakeExpr("add", context.inputs)


class _FakeMulHandler:
    """Lowers a concrete Mul operation into this consumer's own representation."""

    operator_type = MulOperator

    def lower(self, context: OperationContext) -> FakeExpr:
        return FakeExpr("mul", context.inputs)


class _FakeSubHandler:
    """Lowers a concrete Sub operation into this consumer's own representation."""

    operator_type = SubOperator

    def lower(self, context: OperationContext) -> FakeExpr:
        return FakeExpr("sub", context.inputs)


class _FuseMulIntoSubHook:
    """Recognizes `a - (b * c)` and folds it into one fused instruction.

    This is a supported local pattern chosen entirely by this consumer: the
    framework never inspects, names, or special-cases it. It fires only when
    the Mul's result feeds exactly the second operand of the following Sub,
    which the fusion legality check (not this hook) is what actually
    guarantees is safe to collapse.
    """

    lookahead = 2

    def fuse(self, context: FusionContext) -> Optional[FusionResult]:
        offered = context.candidates[0]
        if type(offered.operator) is not MulOperator:
            return None
        if len(context.candidates) < 2:
            return None
        following = context.candidates[1]
        if type(following.operator) is not SubOperator:
            return None
        if following.input_value_ids[1] != offered.output_value_id:
            return None

        minuend = context.value_of(following.input_value_ids[0])
        multiplicand = context.value_of(offered.input_value_ids[0])
        multiplier = context.value_of(offered.input_value_ids[1])
        fused_value = FakeExpr("fused_multiply_subtract", (minuend, multiplicand, multiplier))
        return FusionResult(
            value=fused_value,
            consumed_node_ids=(offered.node_id, following.node_id),
        )


def _registry() -> OperationHandlerRegistry:
    registry = OperationHandlerRegistry()
    registry.register(_FakeAddHandler())
    registry.register(_FakeMulHandler())
    registry.register(_FakeSubHandler())
    return registry


def _trace_sgd_shaped_graph():
    """Trace `parameter - learning_rate * gradient` with ordinary Tensor code.

    The Mul feeds only the Sub, so the same graph exercises both an
    unfused two-operator lowering and a fusable Mul-then-Sub pattern,
    depending on whether a fusion hook is supplied.
    """
    with TensorGraphBuilder() as builder:
        parameter = builder.input("parameter", dtype="f32", shape=(2, 3))
        gradient = builder.input("gradient", dtype="f32", shape=(2, 3))
        learning_rate = builder.input("learning_rate", dtype="f32", shape=())
        updated = parameter - learning_rate * gradient

    graph = builder.build(outputs=[updated])
    value_ids = {
        "parameter": builder.value_id(parameter),
        "gradient": builder.value_id(gradient),
        "learning_rate": builder.value_id(learning_rate),
        "updated": builder.value_id(updated),
    }
    return graph, value_ids


def test_fake_consumer_lowers_two_concrete_operators_without_fusion() -> None:
    graph, value_ids = _trace_sgd_shaped_graph()

    lowered = lower_graph(
        graph,
        handlers=_registry(),
        outputs=[value_ids["updated"]],
        bind_input=lambda dependency: dependency.value_id,
    )

    assert len(lowered.operations) == 2
    assert [operation.is_fused for operation in lowered.operations] == [False, False]

    source_operator_types = {
        type(operator)
        for operation in lowered.operations
        for operator in operation.source_operators
    }
    assert source_operator_types == {MulOperator, SubOperator}

    (output_value,) = lowered.output_values
    expected = FakeExpr(
        "sub",
        (
            value_ids["parameter"],
            FakeExpr("mul", (value_ids["learning_rate"], value_ids["gradient"])),
        ),
    )
    assert output_value == expected


def test_fake_consumer_lowers_one_supported_fusion() -> None:
    graph, value_ids = _trace_sgd_shaped_graph()

    lowered = lower_graph(
        graph,
        handlers=_registry(),
        outputs=[value_ids["updated"]],
        fusion=_FuseMulIntoSubHook(),
        bind_input=lambda dependency: dependency.value_id,
    )

    assert len(lowered.operations) == 1
    (operation,) = lowered.operations
    assert operation.is_fused is True
    assert {type(operator) for operator in operation.source_operators} == {
        MulOperator,
        SubOperator,
    }

    (output_value,) = lowered.output_values
    expected = FakeExpr(
        "fused_multiply_subtract",
        (value_ids["parameter"], value_ids["learning_rate"], value_ids["gradient"]),
    )
    assert output_value == expected


def test_fake_consumer_module_imports_no_private_autodiff_submodule() -> None:
    """This consumer reaches the seam entirely through public package names.

    This is the mechanical half of the "generic seam" claim: nothing in this
    file imports `tinychain.autodiff.graph`, `.lowering`, or `.dependencies`
    directly, so a consumer never needs private submodule access to reach the
    extension seam. This checks only this file's own import statements; it
    does not check, and cannot detect, whether the framework's public surface
    or lowering behavior has itself become coupled to a specific consumer's
    concept. `test_autodiff_extension_contract.py` pins that the expected
    names remain exported, which is a narrower and different claim than
    "the seam stayed generic."
    """
    import ast
    import inspect

    import tests.test_autodiff_fake_consumer as this_module

    tree = ast.parse(inspect.getsource(this_module))
    tinychain_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("tinychain")
    ]
    assert tinychain_imports
    for module_name in tinychain_imports:
        assert module_name == "tinychain.autodiff", (
            f"fake consumer must import only the public package, found {module_name!r}"
        )
