from __future__ import annotations

import pytest

from tinychain.graph_reflection import (
    OperationContract,
    OperationContractError,
    ReflectionError,
    TypedValueRef,
    TypeSpec,
)


# --- TypeSpec ---


def test_typespec_valid_construction():
    ts = TypeSpec(class_uri="/state/collection/tensor", params={"dtype": "float32", "ndim": 2})
    assert ts.class_uri == "/state/collection/tensor"
    assert ts.params == {"dtype": "float32", "ndim": 2}


def test_typespec_round_trip():
    ts = TypeSpec(class_uri="/state/collection/tensor", params={"dtype": "float32"})
    recovered = TypeSpec.from_dict(ts.to_dict())
    assert recovered == ts


def test_typespec_to_dict_sorted_keys():
    ts = TypeSpec(class_uri="/state/scalar", params={"z": 1, "a": 2})
    d = ts.to_dict()
    assert list(d["params"].keys()) == ["a", "z"]


def test_typespec_empty_params():
    ts = TypeSpec(class_uri="/state/scalar", params={})
    assert ts.params == {}
    recovered = TypeSpec.from_dict(ts.to_dict())
    assert recovered == ts


def test_typespec_empty_class_uri_raises():
    with pytest.raises(ReflectionError) as exc_info:
        TypeSpec(class_uri="", params={})
    assert exc_info.value.category == "invalid_type_spec"


def test_typespec_from_dict_missing_key_raises():
    with pytest.raises(ReflectionError) as exc_info:
        TypeSpec.from_dict({"params": {}})
    assert exc_info.value.category == "invalid_type_spec"


def test_typespec_non_json_serializable_params_raises():
    with pytest.raises(ReflectionError) as exc_info:
        TypeSpec(class_uri="/state/collection/tensor", params={"bad": object()})
    assert exc_info.value.category == "invalid_type_spec"


def test_typespec_defensive_copy_isolation():
    caller_params: dict = {"dtype": "float32"}
    ts = TypeSpec(class_uri="/state/collection/tensor", params=caller_params)
    caller_params["dtype"] = "int8"
    caller_params["extra"] = "injected"
    assert ts.params == {"dtype": "float32"}


# --- TypedValueRef ---


def test_typed_value_ref_valid_construction():
    ts = TypeSpec(class_uri="/state/collection/tensor", params={})
    ref = TypedValueRef(namespace="my_ns", value="x", output=None, value_type=ts)
    assert ref.value == "x"
    assert ref.namespace == "my_ns"
    assert ref.output is None


def test_typed_value_ref_round_trip():
    ts = TypeSpec(class_uri="/state/collection/tensor", params={"ndim": 3})
    ref = TypedValueRef(namespace="ns", value="y", output="out0", value_type=ts)
    recovered = TypedValueRef.from_dict(ref.to_dict())
    assert recovered == ref


def test_typed_value_ref_round_trip_no_namespace():
    ts = TypeSpec(class_uri="/state/scalar", params={})
    ref = TypedValueRef(namespace=None, value="z", output=None, value_type=ts)
    recovered = TypedValueRef.from_dict(ref.to_dict())
    assert recovered == ref


def test_typed_value_ref_empty_value_raises():
    ts = TypeSpec(class_uri="/state/collection/tensor", params={})
    with pytest.raises(ReflectionError) as exc_info:
        TypedValueRef(namespace=None, value="", output=None, value_type=ts)
    assert exc_info.value.category == "invalid_typed_value_ref"


def test_typed_value_ref_from_dict_missing_key_raises():
    with pytest.raises(ReflectionError) as exc_info:
        TypedValueRef.from_dict({"namespace": None, "output": None, "value_type": {"class_uri": "/x", "params": {}}})
    assert exc_info.value.category == "invalid_typed_value_ref"


# --- OperationContract ---


def test_operation_contract_valid_construction():
    oc = OperationContract(method_uri="/lib/math/add", params_schema={"x": "float"})
    assert oc.method_uri == "/lib/math/add"
    assert oc.params_schema == {"x": "float"}
    assert oc.output_type_rule is None


def test_operation_contract_empty_method_uri_raises():
    with pytest.raises(OperationContractError) as exc_info:
        OperationContract(method_uri="", params_schema={})
    assert exc_info.value.category == "invalid_method_uri"


def test_operation_contract_defensive_copy_isolation():
    schema: dict = {"x": "float"}
    oc = OperationContract(method_uri="/lib/math/add", params_schema=schema)
    schema["injected"] = "bad"
    assert "injected" not in oc.params_schema
