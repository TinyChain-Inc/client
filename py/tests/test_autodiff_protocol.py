from __future__ import annotations

import pytest

from tinychain.autodiff import (
    AUTODIFF_ERROR_CATEGORIES,
    AutodiffError,
    AutodiffRequest,
    AutodiffResult,
    DerivativeMetadata,
)


def test_autodiff_request_round_trips():
    request = AutodiffRequest(
        graph={"nodes": []},
        output_value_id="out",
        wrt=["x", "y"],
        seed_value_id="seed",
        tensor_op_contract_version="0.1.0",
        transform_version="0.1.0",
    )

    assert AutodiffRequest.from_dict(request.to_dict()) == request


def test_autodiff_result_and_metadata_round_trip():
    metadata = DerivativeMetadata(
        source_graph_id="graph-1",
        transform_version="0.1.0",
        tensor_op_contract_version="0.1.0",
        wrt_signature=("x", "y"),
        seed_contract="seed matches out",
    )
    result = AutodiffResult(gradients=["dx", "dy"], metadata=metadata)

    assert AutodiffResult.from_dict(result.to_dict()) == result


def test_autodiff_error_categories_cover_spec_codes():
    expected = {
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
    }

    assert set(AUTODIFF_ERROR_CATEGORIES) == expected


def test_autodiff_error_round_trip_and_rejects_unknown_category():
    error = AutodiffError("unsupported_operator", "no rule")
    assert AutodiffError.from_dict(error.to_dict()) == error

    with pytest.raises(ValueError, match="unknown autodiff error category"):
        AutodiffError("not_real", "bad")
