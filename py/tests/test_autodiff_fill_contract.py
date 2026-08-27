"""Unit tests for the generated-constant contract: `FillOperator` and its reader.

A generated tensor-valued constant operand needs exactly one public spelling, so
that a backend which does not understand it is rejected by concrete-type dispatch
before any handler runs, and so that every backend which does understand it reads
the same validated descriptor rather than hand-parsing `op_params`.

These tests pin that contract only — the operator type, the descriptor schema,
the shared reader and its categorized failures, and the package-level export.
They assert nothing about any expansion pass, provenance record, or VJP rule.
"""

from __future__ import annotations

from dataclasses import fields
from types import MappingProxyType

import pytest
import tinychain as tc
from tinychain.autodiff import (
    AUTODIFF_ERROR_CATEGORIES,
    AutodiffError,
    MulOperator,
    OperationContext,
    TensorNodeRecord,
    TensorOperator,
    captured_route_operators,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _well_formed_params() -> dict[str, object]:
    return {"fill": 0.25, "dtype": "f32", "shape": [2, 3]}


def _fill_node(
    *,
    node_id: str = "fill_0",
    op_params: dict[str, object] | None = None,
    input_value_ids: list[str] | None = None,
    operator: TensorOperator | None = None,
    output_typespec: dict[str, object] | None = None,
) -> TensorNodeRecord:
    """Build one fill node record, with every field overridable for the failure table."""
    from tinychain.autodiff import FillOperator

    params = _well_formed_params() if op_params is None else op_params
    if output_typespec is None:
        output_typespec = {"dtype": params.get("dtype"), "shape": list(params.get("shape", []))}
    return TensorNodeRecord(
        node_id=node_id,
        output_value_id=f"{node_id}_out",
        operator=FillOperator() if operator is None else operator,
        op_params=params,
        input_value_ids=[] if input_value_ids is None else input_value_ids,
        output_typespec=output_typespec,
    )


def _operation_context(node: TensorNodeRecord) -> OperationContext:
    """Normalize a node the way `lower_graph` hands one to a consumer handler."""
    typespec = node.output_typespec
    return OperationContext(
        node_id=node.node_id,
        operator=node.operator,
        op_params=MappingProxyType(dict(node.op_params)),
        input_value_ids=tuple(node.input_value_ids),
        inputs=tuple(object() for _ in node.input_value_ids),
        input_provenance=tuple("local_value" for _ in node.input_value_ids),
        output_value_id=node.output_value_id,
        output_typespec=None if typespec is None else MappingProxyType(dict(typespec)),
    )


# --------------------------------------------------------------------------
# AC-1 — the operator type
# --------------------------------------------------------------------------


def test_fill_operator_declares_fill_route_name() -> None:
    from tinychain.autodiff import FillOperator

    operator = FillOperator()

    assert isinstance(operator, TensorOperator)
    assert operator.route_name == "fill"


def test_fill_operator_declares_no_fields_beyond_the_base_operator() -> None:
    from tinychain.autodiff import FillOperator

    assert tuple(field.name for field in fields(FillOperator)) == ("route_name",)


def test_fill_operator_instances_compare_equal() -> None:
    from tinychain.autodiff import FillOperator

    assert FillOperator() == FillOperator()


# --------------------------------------------------------------------------
# AC-2 / AC-3 — the reader on well-formed input
# --------------------------------------------------------------------------


def test_fill_descriptor_returns_declared_values_from_a_node() -> None:
    from tinychain.autodiff import FillDescriptor, fill_descriptor

    descriptor = fill_descriptor(_fill_node())

    assert isinstance(descriptor, FillDescriptor)
    assert descriptor.fill == 0.25
    assert descriptor.dtype == "f32"
    assert descriptor.shape == (2, 3)


def test_fill_descriptor_returns_the_same_record_from_an_operation_context() -> None:
    from tinychain.autodiff import fill_descriptor

    node = _fill_node()

    assert fill_descriptor(_operation_context(node)) == fill_descriptor(node)


def test_fill_descriptor_accepts_an_integer_fill_as_a_real_number() -> None:
    from tinychain.autodiff import fill_descriptor

    descriptor = fill_descriptor(_fill_node(op_params={"fill": 1, "dtype": "f32", "shape": [2]}))

    assert descriptor.fill == 1.0


def test_fill_descriptor_accepts_a_zero_dimensional_shape() -> None:
    from tinychain.autodiff import fill_descriptor

    descriptor = fill_descriptor(_fill_node(op_params={"fill": 0.5, "dtype": "f32", "shape": []}))

    assert descriptor.shape == ()


def test_fill_node_output_typespec_equals_its_descriptor_dtype_and_shape() -> None:
    from tinychain.autodiff import fill_descriptor

    node = _fill_node()
    descriptor = fill_descriptor(node)

    assert node.output_typespec is not None
    assert node.output_typespec["dtype"] == descriptor.dtype
    assert tuple(node.output_typespec["shape"]) == descriptor.shape


# --------------------------------------------------------------------------
# AC-4 / AC-5 — the categorized failure table (§13.1)
# --------------------------------------------------------------------------


_MALFORMED_CASES = [
    ("missing_fill_key", {"dtype": "f32", "shape": [2, 3]}, "malformed_derivative_ir"),
    (
        "extra_key",
        {"fill": 0.25, "dtype": "f32", "shape": [2, 3], "axes": [0]},
        "malformed_derivative_ir",
    ),
    ("fill_is_a_string", {"fill": "0.25", "dtype": "f32", "shape": [2, 3]}, "malformed_derivative_ir"),
    ("fill_is_boolean", {"fill": True, "dtype": "f32", "shape": [2, 3]}, "malformed_derivative_ir"),
    ("fill_is_none", {"fill": None, "dtype": "f32", "shape": [2, 3]}, "malformed_derivative_ir"),
    ("missing_shape_key", {"fill": 0.25, "dtype": "f32"}, "missing_shape_metadata"),
    ("shape_is_none", {"fill": 0.25, "dtype": "f32", "shape": None}, "missing_shape_metadata"),
    ("shape_is_not_a_sequence", {"fill": 0.25, "dtype": "f32", "shape": 3}, "missing_shape_metadata"),
    (
        "shape_dimension_is_symbolic",
        {"fill": 0.25, "dtype": "f32", "shape": ["rows", 3]},
        "missing_shape_metadata",
    ),
    (
        "shape_dimension_is_boolean",
        {"fill": 0.25, "dtype": "f32", "shape": [True, 3]},
        "missing_shape_metadata",
    ),
    (
        "shape_dimension_is_negative",
        {"fill": 0.25, "dtype": "f32", "shape": [-1, 3]},
        "missing_shape_metadata",
    ),
    ("missing_dtype_key", {"fill": 0.25, "shape": [2, 3]}, "missing_dtype_metadata"),
    ("dtype_is_none", {"fill": 0.25, "dtype": None, "shape": [2, 3]}, "missing_dtype_metadata"),
    ("dtype_is_empty", {"fill": 0.25, "dtype": "", "shape": [2, 3]}, "missing_dtype_metadata"),
    ("dtype_is_not_a_string", {"fill": 0.25, "dtype": 32, "shape": [2, 3]}, "missing_dtype_metadata"),
]


@pytest.mark.parametrize(
    ("op_params", "expected_category"),
    [(params, category) for _, params, category in _MALFORMED_CASES],
    ids=[case_id for case_id, _, _ in _MALFORMED_CASES],
)
def test_fill_descriptor_rejects_a_malformed_descriptor(
    op_params: dict[str, object], expected_category: str
) -> None:
    from tinychain.autodiff import fill_descriptor

    node = _fill_node(
        node_id="fill_bad",
        op_params=op_params,
        output_typespec={"dtype": "f32", "shape": [2, 3]},
    )

    with pytest.raises(AutodiffError) as raised:
        fill_descriptor(node)

    assert raised.value.category == expected_category
    assert "fill_bad" in raised.value.message


@pytest.mark.parametrize(
    ("op_params", "expected_category"),
    [(params, category) for _, params, category in _MALFORMED_CASES],
    ids=[case_id for case_id, _, _ in _MALFORMED_CASES],
)
def test_fill_descriptor_rejects_a_malformed_operation_context(
    op_params: dict[str, object], expected_category: str
) -> None:
    from tinychain.autodiff import fill_descriptor

    node = _fill_node(
        node_id="fill_bad",
        op_params=op_params,
        output_typespec={"dtype": "f32", "shape": [2, 3]},
    )

    with pytest.raises(AutodiffError) as raised:
        fill_descriptor(_operation_context(node))

    assert raised.value.category == expected_category
    assert "fill_bad" in raised.value.message


def test_fill_descriptor_rejects_a_fill_node_carrying_an_operand() -> None:
    from tinychain.autodiff import fill_descriptor

    node = _fill_node(node_id="fill_with_operand", input_value_ids=["value_0"])

    with pytest.raises(AutodiffError) as raised:
        fill_descriptor(node)

    assert raised.value.category == "malformed_derivative_ir"
    assert "fill_with_operand" in raised.value.message


def test_fill_descriptor_rejects_a_node_whose_operator_is_not_a_fill_operator() -> None:
    from tinychain.autodiff import fill_descriptor

    node = _fill_node(node_id="not_a_fill", operator=MulOperator(), op_params={})

    with pytest.raises(AutodiffError) as raised:
        fill_descriptor(node)

    assert raised.value.category == "malformed_derivative_ir"
    assert "not_a_fill" in raised.value.message


def test_fill_descriptor_rejects_an_argument_that_is_neither_node_nor_context() -> None:
    from tinychain.autodiff import fill_descriptor

    with pytest.raises(AutodiffError) as raised:
        fill_descriptor(object())

    assert raised.value.category == "malformed_derivative_ir"


@pytest.mark.parametrize(
    "op_params",
    [params for _, params, _ in _MALFORMED_CASES],
    ids=[case_id for case_id, _, _ in _MALFORMED_CASES],
)
def test_fill_descriptor_never_leaks_a_bare_builtin_exception(
    op_params: dict[str, object],
) -> None:
    from tinychain.autodiff import fill_descriptor

    node = _fill_node(node_id="fill_bad", op_params=op_params)

    try:
        fill_descriptor(node)
    except AutodiffError:
        pass
    except (KeyError, IndexError, TypeError, ValueError, AssertionError) as exc:
        pytest.fail(f"bare {type(exc).__name__} escaped fill_descriptor: {exc}")


# --------------------------------------------------------------------------
# AC-6 / AC-7 — export surface and untouched neighbours
# --------------------------------------------------------------------------


_FILL_EXPORTS = frozenset({"FillOperator", "FillDescriptor", "fill_descriptor"})


def test_fill_contract_surface_is_exported_from_the_autodiff_package() -> None:
    import tinychain.autodiff as autodiff
    from tinychain.autodiff import expansion

    assert _FILL_EXPORTS.issubset(set(autodiff.__all__))
    for export_name in _FILL_EXPORTS:
        assert getattr(autodiff, export_name) is getattr(expansion, export_name)
        assert not hasattr(tc, export_name)


def test_fill_operator_is_absent_from_the_tracing_capture_allowlist() -> None:
    from tinychain.autodiff import FillOperator

    allowlist = captured_route_operators()

    assert "fill" not in allowlist
    assert FillOperator not in set(allowlist.values())


def test_expansion_module_adds_no_autodiff_error_category() -> None:
    from tinychain.autodiff import expansion  # noqa: F401 -- import is the subject

    assert AUTODIFF_ERROR_CATEGORIES == (
        "unsupported_operator",
        "missing_derivative_behavior",
        "missing_derivative_ir",
        "non_differentiable_route",
        "missing_shape_metadata",
        "missing_dtype_metadata",
        "dtype_not_differentiable",
        "shape_mismatch",
        "unresolved_symbolic_shape",
        "symbolic_shape_mismatch",
        "broadcast_shape_mismatch",
        "matmul_shape_mismatch",
        "invalid_permutation",
        "reduction_shape_mismatch",
        "unsupported_reduction",
        "seed_shape_mismatch",
        "malformed_derivative_ir",
        "side_effecting_route_unsupported",
        "autodiff_not_implemented",
        "dtype_mismatch",
        "missing_dependency",
        "ambiguous_producer",
        "invalid_selected_output",
        "handler_contract_violation",
        "invalid_update_signature",
        "invalid_update_output",
    )
