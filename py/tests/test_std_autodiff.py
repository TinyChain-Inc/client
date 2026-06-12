import tinychain as tc
import pytest


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
