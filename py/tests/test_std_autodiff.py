import tinychain as tc
import pytest

from tinychain.autodiff.vjp import (
    VjpRegistry,
    AddOperator,
    MatmulOperator,
    TransposeOperator,
    VjpRule,
    VjpContext,
    VjpResult,
)


def test_autodiff_grad_raises_not_implemented():
    autodiff = tc.std.Autodiff()
    target = tc.state.id("target")
    wrt = tc.state.tuple_of([tc.state.id("x")])
    with pytest.raises(NotImplementedError, match="autodiff_not_implemented"):
        autodiff.grad(target=target, wrt=wrt)


def test_autodiff_vjp_raises_not_implemented():
    autodiff = tc.std.Autodiff()
    target = tc.state.id("target")
    wrt = tc.state.tuple_of([tc.state.id("x")])
    cotangent = tc.state.id("cotangent")
    with pytest.raises(NotImplementedError, match="autodiff_not_implemented"):
        autodiff.vjp(target=target, wrt=wrt, cotangent=cotangent)


def test_autodiff_trace_raises_not_implemented():
    autodiff = tc.std.Autodiff()
    op = tc.state.PostOpDef([])
    with pytest.raises(NotImplementedError, match="autodiff_not_implemented"):
        autodiff.trace(op=op)


def test_autodiff_grad_does_not_return_identity():
    autodiff = tc.std.Autodiff()
    target = tc.state.id("target")
    wrt = tc.state.tuple_of([tc.state.id("x")])
    raised = False
    try:
        result = autodiff.grad(target=target, wrt=wrt)
    except NotImplementedError:
        raised = True
    assert raised, "grad must not silently return an identity result"


def test_autodiff_vjp_does_not_return_cotangent():
    autodiff = tc.std.Autodiff()
    target = tc.state.id("target")
    wrt = tc.state.tuple_of([tc.state.id("x")])
    cotangent = tc.state.id("cotangent")
    raised = False
    try:
        result = autodiff.vjp(target=target, wrt=wrt, cotangent=cotangent)
    except NotImplementedError:
        raised = True
    assert raised, "vjp must not silently return cotangent as identity"


def test_autodiff_trace_does_not_return_op():
    autodiff = tc.std.Autodiff()
    op = tc.state.PostOpDef([])
    raised = False
    try:
        result = autodiff.trace(op=op)
    except NotImplementedError:
        raised = True
    assert raised, "trace must not silently return op unchanged"


def test_autodiff_operator_is_abstract_and_dualoperator_has_typed_fields():
    with pytest.raises(TypeError, match="abstract"):
        tc.std.Operator()

    left = tc.state.id("l")
    right = tc.state.id("r")
    op = tc.std.DualOperator(left=left, right=right)

    assert op.left == left
    assert op.right == right


def test_dualoperator_accepts_tensor_operands():
    left = tc.Tensor(ref=tc.state.id("t1"))
    right = tc.Tensor(ref=tc.state.id("t2"))

    op = tc.std.DualOperator(left=left, right=right)
    assert op.left == left
    assert op.right == right


def test_autodiff_operator_docstring_covers_typed_subclass_usage():
    doc = tc.std.Operator.__doc__
    assert doc is not None
    assert "typed fields" in doc


def test_vjp_registry_decorator_registration():
    """Test that @registry.rule() decorator registers rules at class definition time."""
    registry = VjpRegistry()

    @registry.rule(AddOperator)
    class TestRule:
        operator_type = AddOperator

        def apply(self, context: VjpContext) -> VjpResult:
            return VjpResult(gradients={}, derivative_nodes=[])

    assert registry.has_rule(AddOperator)
    assert registry.has_rule(AddOperator())
    assert not registry.has_rule(MatmulOperator)


def test_vjp_registry_decorator_transparency():
    """Test that the decorator returns the original class unchanged."""
    registry = VjpRegistry()

    @registry.rule(AddOperator)
    class TestRule:
        operator_type = AddOperator

        def apply(self, context: VjpContext) -> VjpResult:
            return VjpResult(gradients={}, derivative_nodes=[])

    assert TestRule.__name__ == "TestRule"
    assert TestRule.operator_type == AddOperator


def test_vjp_registry_has_rule():
    """Test has_rule() returns True for registered operators, False for unregistered."""
    registry = VjpRegistry()

    @registry.rule(AddOperator)
    class TestRule:
        operator_type = AddOperator

        def apply(self, context: VjpContext) -> VjpResult:
            return VjpResult(gradients={}, derivative_nodes=[])

    assert registry.has_rule(AddOperator)
    assert registry.has_rule(AddOperator())
    assert not registry.has_rule(MatmulOperator)
    assert not registry.has_rule(TransposeOperator)


def test_vjp_registry_supported_types():
    """Test supported_types() returns all registered TensorOperator types."""
    registry = VjpRegistry()

    @registry.rule(AddOperator)
    class TestRule1:
        operator_type = AddOperator

        def apply(self, context: VjpContext) -> VjpResult:
            return VjpResult(gradients={}, derivative_nodes=[])

    @registry.rule(MatmulOperator)
    class TestRule2:
        operator_type = MatmulOperator

        def apply(self, context: VjpContext) -> VjpResult:
            return VjpResult(gradients={}, derivative_nodes=[])

    @registry.rule(TransposeOperator)
    class TestRule3:
        operator_type = TransposeOperator

        def apply(self, context: VjpContext) -> VjpResult:
            return VjpResult(gradients={}, derivative_nodes=[])

    types = registry.supported_types()
    assert len(types) == 3
    assert AddOperator in types
    assert MatmulOperator in types
    assert TransposeOperator in types


def test_vjp_registry_decorator_type_error():
    """Test that decorator raises TypeError for non-VjpRule classes."""
    registry = VjpRegistry()

    with pytest.raises(TypeError, match="is not a VjpRule instance"):

        @registry.rule(AddOperator)
        class NotARule:
            pass


def test_vjp_registry_manual_register_backwards_compatibility():
    """Test that manual register() still works for backwards compatibility."""
    registry = VjpRegistry()

    class ManualRule:
        operator_type = AddOperator

        def apply(self, context: VjpContext) -> VjpResult:
            return VjpResult(gradients={}, derivative_nodes=[])

    rule_instance = ManualRule()
    registry.register(rule_instance)

    assert registry.has_rule(AddOperator)
    assert registry.lookup(AddOperator()) is rule_instance


def test_vjp_registry_default_registry_has_all_three_rules():
    """Test that default_vjp_registry() returns all 3 registered rule types."""
    from tinychain.autodiff.vjp import default_vjp_registry

    registry = default_vjp_registry()

    assert registry.has_rule(AddOperator)
    assert registry.has_rule(MatmulOperator)
    assert registry.has_rule(TransposeOperator)

    types = registry.supported_types()
    assert len(types) == 3
    assert AddOperator in types
    assert MatmulOperator in types
    assert TransposeOperator in types


def test_vjp_rule_declaration_order():
    """Test that declaration order does not affect registration correctness."""
    registry1 = VjpRegistry()

    @registry1.rule(AddOperator)
    class Rule1:
        operator_type = AddOperator

        def apply(self, context: VjpContext) -> VjpResult:
            return VjpResult(gradients={}, derivative_nodes=[])

    @registry1.rule(MatmulOperator)
    class Rule2:
        operator_type = MatmulOperator

        def apply(self, context: VjpContext) -> VjpResult:
            return VjpResult(gradients={}, derivative_nodes=[])

    registry2 = VjpRegistry()

    @registry2.rule(MatmulOperator)
    class Rule3:
        operator_type = MatmulOperator

        def apply(self, context: VjpContext) -> VjpResult:
            return VjpResult(gradients={}, derivative_nodes=[])

    @registry2.rule(AddOperator)
    class Rule4:
        operator_type = AddOperator

        def apply(self, context: VjpContext) -> VjpResult:
            return VjpResult(gradients={}, derivative_nodes=[])

    # Both registries should have the same types registered
    types1 = registry1.supported_types()
    types2 = registry2.supported_types()

    assert set(types1) == set(types2)
    assert len(types1) == 2
    assert AddOperator in types1
    assert MatmulOperator in types1

    # Both should pass has_rule checks
    assert registry1.has_rule(AddOperator)
    assert registry1.has_rule(MatmulOperator)
    assert registry2.has_rule(AddOperator)
    assert registry2.has_rule(MatmulOperator)
