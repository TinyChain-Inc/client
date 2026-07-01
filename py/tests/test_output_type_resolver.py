from __future__ import annotations

import pytest

from tinychain.graph_reflection import OperationContract, OperationContractError, TypeSpec


def test_contract_infers_tensor_like_output():
    contract = OperationContract(
        method_uri="/tensor/identity/v1",
        params_schema={},
        output_type_rule=lambda inputs, params: [
            TypeSpec(class_uri=inputs[0].class_uri, params={"dtype": inputs[0].params["dtype"]})
        ],
    )

    input_spec = TypeSpec(class_uri="/tensor/dense", params={"dtype": "float32", "shape": [3, 4]})
    result = contract.infer_outputs([input_spec], {})

    assert len(result) == 1
    assert result[0].class_uri == "/tensor/dense"
    assert result[0].params["dtype"] == "float32"
    assert "shape" not in result[0].params


def test_contract_infers_non_tensor_output():
    contract = OperationContract(
        method_uri="/planner/cost/v1",
        params_schema={},
        output_type_rule=lambda inputs, params: [TypeSpec(class_uri="/planner/cost_vector", params={})],
    )

    result = contract.infer_outputs([], {})

    assert len(result) == 1
    assert result[0].class_uri == "/planner/cost_vector"
    assert result[0].params == {}


def test_missing_output_type_rule_raises():
    contract = OperationContract(method_uri="/unregistered/op/v1", params_schema={})

    with pytest.raises(OperationContractError) as exc_info:
        contract.infer_outputs([], {})
    assert exc_info.value.category == "missing_output_type_rule"


def test_contract_infers_identity_via_plain_function():
    def infer_identity(inputs, params):
        return list(inputs)

    contract = OperationContract(
        method_uri="/test/identity/v1",
        params_schema={},
        output_type_rule=infer_identity,
    )

    spec = TypeSpec(class_uri="/state/collection/tensor", params={"dtype": "float32"})
    result = contract.infer_outputs([spec], {})
    assert result == [spec]
