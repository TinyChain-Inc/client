import tinychain as tc
import pytest


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
