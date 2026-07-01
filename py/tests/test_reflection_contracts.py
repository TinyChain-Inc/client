from __future__ import annotations

import pytest

from tinychain.graph_reflection import OperationContract, OperationContractError, TypeSpec
from tinychain.autodiff.vjp import VjpRegistry, default_vjp_registry


def test_valid_method_uri_accepted():
    contract = OperationContract(method_uri="/tensor/matmul/v1", params_schema={})
    assert contract.method_uri == "/tensor/matmul/v1"


def test_empty_method_uri_raises():
    with pytest.raises(OperationContractError) as exc_info:
        OperationContract(method_uri="", params_schema={})
    assert exc_info.value.category == "invalid_method_uri"


def test_operation_contract_fields_preserved():
    schema = {"a": "float32", "b": [1, 2, 3]}
    contract = OperationContract(
        method_uri="/tensor/add/v2",
        params_schema=schema,
        output_type_rule=lambda inputs, params: list(inputs),
    )
    assert contract.method_uri == "/tensor/add/v2"
    assert contract.params_schema == schema
    spec = TypeSpec(class_uri="/state/tensor", params={"dtype": "float32"})
    assert contract.infer_outputs([spec], {}) == [spec]


def test_operation_contract_is_frozen():
    contract = OperationContract(method_uri="/tensor/matmul/v1", params_schema={})
    with pytest.raises(AttributeError):
        contract.method_uri = "/tensor/other/v1"  # type: ignore[misc]


def test_whitespace_only_method_uri_raises():
    with pytest.raises(OperationContractError) as exc_info:
        OperationContract(method_uri="   ", params_schema={})
    assert exc_info.value.category == "invalid_method_uri"


def test_method_uri_not_used_in_vjp_dispatch():
    registry = default_vjp_registry()
    keys = list(registry._rules.keys())
    assert len(keys) > 0, "expected at least one rule registered"
    for key in keys:
        assert isinstance(key, type), f"expected type key, got {type(key)!r}: {key!r}"
        assert not isinstance(key, str), f"string key found in VjpRegistry: {key!r}"
