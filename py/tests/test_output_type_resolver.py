from tinychain.graph_reflection import ReflectionError, ResolverRegistry, TypeSpec


def test_tensor_like_resolver():
    class TensorIdentityResolver:
        def infer_outputs(self, method_uri, inputs, params):
            return [TypeSpec(class_uri=inputs[0].class_uri, params={"dtype": inputs[0].params["dtype"]})]

    registry = ResolverRegistry()
    registry.register("/tensor/identity/v1", TensorIdentityResolver())

    input_spec = TypeSpec(class_uri="/tensor/dense", params={"dtype": "float32", "shape": [3, 4]})
    result = registry.infer("/tensor/identity/v1", [input_spec], {})

    assert len(result) == 1
    assert result[0].class_uri == "/tensor/dense"
    assert result[0].params["dtype"] == "float32"
    assert "shape" not in result[0].params


def test_non_tensor_resolver():
    class PlannerCostResolver:
        def infer_outputs(self, method_uri, inputs, params):
            return [TypeSpec(class_uri="/planner/cost_vector", params={})]

    registry = ResolverRegistry()
    registry.register("/planner/cost/v1", PlannerCostResolver())

    result = registry.infer("/planner/cost/v1", [], {})

    assert len(result) == 1
    assert result[0].class_uri == "/planner/cost_vector"
    assert result[0].params == {}


def test_missing_method_uri_raises():
    registry = ResolverRegistry()

    try:
        registry.infer("/unregistered/op/v1", [], {})
        assert False, "expected ReflectionError"
    except ReflectionError as exc:
        assert exc.category == "unsupported_method_uri"
