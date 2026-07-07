from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tinychain as tc
from tinychain.autodiff import (
    ROUTE_DERIVATIVE_COMPATIBILITY_NOT_VALIDATED,
    ROUTE_DERIVATIVE_SOURCE_ARTIFACT,
    ROUTE_DERIVATIVE_SOURCE_NON_DIFFERENTIABLE,
    ROUTE_DERIVATIVE_SOURCE_UNSUPPORTED,
    AutodiffError,
    RouteDerivativeMetadata,
    RouteDerivativePlan,
)
from tinychain.autodiff.protocol import DerivativeMetadata
from tinychain.autodiff.routes import RouteDerivativeIdentity, extract_route_identity
from tinychain.graph_reflection import TypeSpec


class RouteIdentityLibrary(tc.Library):
    publisher = "example-devco"
    version = "0.1.0"

    @tc.get
    def fetch(self, key: tc.String) -> tc.String:
        raise AssertionError("route body must not execute")

    @tc.post
    def create(self, value: tc.String) -> tc.String:
        raise AssertionError("route body must not execute")

    @tc.put
    def replace(self, value: tc.String) -> tc.String:
        raise AssertionError("route body must not execute")

    @tc.delete
    def remove(self, key: tc.String) -> tc.String:
        raise AssertionError("route body must not execute")


@pytest.mark.parametrize(
    ("route_name", "expected_method"),
    [
        ("fetch", "GET"),
        ("create", "POST"),
        ("replace", "PUT"),
        ("remove", "DELETE"),
    ],
)
def test_extract_route_identity_for_bound_route_decorators(
    route_name: str,
    expected_method: str,
) -> None:
    library = RouteIdentityLibrary()
    target = getattr(library, route_name)

    identity = extract_route_identity(target)

    assert identity == RouteDerivativeIdentity.from_dict(identity.to_dict())
    assert identity.to_dict() == {
        "publisher": "example-devco",
        "library_name": "route_identity_library",
        "library_version": "0.1.0",
        "library_path": "/lib/example-devco/route_identity_library/0.1.0",
        "library_uri": "/lib/example-devco/route_identity_library/0.1.0",
        "route_name": route_name,
        "route_path": f"/{route_name}",
        "route_uri": f"/lib/example-devco/route_identity_library/0.1.0/path/{route_name}",
        "http_method": expected_method,
    }


def test_extract_route_identity_respects_authority_aware_library_uris() -> None:
    library = RouteIdentityLibrary(authority=tc.URI.parse("https://example.test:8443"))

    identity = extract_route_identity(library.create)

    assert identity.library_path == "/lib/example-devco/route_identity_library/0.1.0"
    assert (
        identity.library_uri
        == "https://example.test:8443/lib/example-devco/route_identity_library/0.1.0"
    )
    assert (
        identity.route_uri
        == "https://example.test:8443/lib/example-devco/route_identity_library/0.1.0/path/create"
    )


def test_extract_route_identity_rejects_unbound_route_descriptor() -> None:
    with pytest.raises(TypeError, match="bound TinyChain route target"):
        extract_route_identity(RouteIdentityLibrary.fetch)


def test_extract_route_identity_rejects_plain_callables() -> None:
    def plain_target() -> None:
        raise AssertionError("plain target must not execute")

    with pytest.raises(TypeError, match="call-site transform"):
        extract_route_identity(plain_target)


def test_extract_route_identity_rejects_malformed_route_metadata() -> None:
    def fake_target() -> None:
        raise AssertionError("fake target must not execute")

    fake_target.__tc_route__ = SimpleNamespace(name="", method="POST")
    fake_target.__tc_instance__ = RouteIdentityLibrary()

    with pytest.raises(AutodiffError) as exc_info:
        extract_route_identity(fake_target)

    assert exc_info.value.category == "non_differentiable_route"
    assert "missing route name" in str(exc_info.value)


def test_extract_route_identity_does_not_call_route_compile_install_or_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"body": 0}

    class SideEffectLibrary(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.post
        def mutate(self, value: tc.String) -> tc.String:
            calls["body"] += 1
            raise AssertionError("route body must not execute")

    def forbidden_behavior(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("compile/install/dispatch behavior must not run")

    monkeypatch.setattr("tinychain.library.compile_ir", forbidden_behavior)
    monkeypatch.setattr("tinychain.library.install", forbidden_behavior)
    monkeypatch.setattr(tc, "execute", forbidden_behavior)

    identity = extract_route_identity(SideEffectLibrary().mutate)

    assert calls == {"body": 0}
    assert identity.route_name == "mutate"
    assert identity.http_method == "POST"


def _string_type_spec() -> TypeSpec:
    return TypeSpec(class_uri="/state/scalar/string", params={"encoding": "utf-8"})


def _route_identity() -> RouteDerivativeIdentity:
    return extract_route_identity(RouteIdentityLibrary().create)


def test_route_derivative_metadata_serializes_stable_json_shape() -> None:
    value_type = _string_type_spec()
    metadata = RouteDerivativeMetadata(
        source_kind=ROUTE_DERIVATIVE_SOURCE_ARTIFACT,
        is_pure=True,
        is_differentiable=True,
        input_signature=(value_type,),
        output_signature=(value_type,),
        supported_wrt=("value",),
        seed_contract="cotangent:output",
        transform_version="route-discovery-v1",
        tensor_op_contract_version="tensor-contract-v1",
    )

    payload = metadata.to_dict()

    assert RouteDerivativeMetadata.from_dict(payload) == metadata
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
    assert payload == {
        "source_kind": "artifact",
        "is_pure": True,
        "is_differentiable": True,
        "input_signature": [
            {"class_uri": "/state/scalar/string", "params": {"encoding": "utf-8"}}
        ],
        "output_signature": [
            {"class_uri": "/state/scalar/string", "params": {"encoding": "utf-8"}}
        ],
        "supported_wrt": ["value"],
        "seed_contract": "cotangent:output",
        "transform_version": "route-discovery-v1",
        "tensor_op_contract_version": "tensor-contract-v1",
    }


def test_route_derivative_plan_serializes_identity_and_artifact_reference_fields() -> None:
    plan = RouteDerivativePlan(
        route_identity=_route_identity(),
        requested_wrt=("value",),
        seed_contract="cotangent:output",
        source_kind=ROUTE_DERIVATIVE_SOURCE_ARTIFACT,
        compatibility_status=ROUTE_DERIVATIVE_COMPATIBILITY_NOT_VALIDATED,
        artifact_uri="/lib/example-devco/create_derivative/0.1.0",
        artifact_digest="sha256:abc123",
        artifact_visibility="public",
    )

    payload = plan.to_dict()

    assert RouteDerivativePlan.from_dict(payload) == plan
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
    assert payload["route_identity"] == _route_identity().to_dict()
    assert payload["requested_wrt"] == ["value"]
    assert payload["compatibility_status"] == "not_validated"
    assert payload["artifact_uri"] == "/lib/example-devco/create_derivative/0.1.0"


@pytest.mark.parametrize(
    "source_kind",
    [ROUTE_DERIVATIVE_SOURCE_UNSUPPORTED, ROUTE_DERIVATIVE_SOURCE_NON_DIFFERENTIABLE],
)
def test_route_derivative_metadata_rejects_differentiable_unsupported_states(
    source_kind: str,
) -> None:
    with pytest.raises(AutodiffError) as exc_info:
        RouteDerivativeMetadata(
            source_kind=source_kind,
            is_pure=True,
            is_differentiable=True,
            input_signature=(_string_type_spec(),),
            output_signature=(_string_type_spec(),),
            supported_wrt=("value",),
            seed_contract="cotangent:output",
            transform_version="route-discovery-v1",
            tensor_op_contract_version="tensor-contract-v1",
        )

    assert exc_info.value.category == "non_differentiable_route"


def test_route_derivative_metadata_rejects_unknown_source_kind() -> None:
    with pytest.raises(AutodiffError) as exc_info:
        RouteDerivativeMetadata(
            source_kind="python_callback",
            is_pure=True,
            is_differentiable=True,
            input_signature=(_string_type_spec(),),
            output_signature=(_string_type_spec(),),
            supported_wrt=("value",),
            seed_contract="cotangent:output",
            transform_version="route-discovery-v1",
            tensor_op_contract_version="tensor-contract-v1",
        )

    assert exc_info.value.category == "non_differentiable_route"
    assert "source_kind" in str(exc_info.value)


def test_route_derivative_plan_rejects_malformed_non_json_fields() -> None:
    with pytest.raises(AutodiffError) as exc_info:
        RouteDerivativePlan(
            route_identity=_route_identity(),
            requested_wrt=(object(),),
            seed_contract="cotangent:output",
            source_kind=ROUTE_DERIVATIVE_SOURCE_ARTIFACT,
            compatibility_status=ROUTE_DERIVATIVE_COMPATIBILITY_NOT_VALIDATED,
        )

    assert exc_info.value.category == "non_differentiable_route"
    assert "requested_wrt" in str(exc_info.value)


def test_route_metadata_is_separate_from_derivative_metadata_and_top_level_exports() -> None:
    route_payload = RouteDerivativeMetadata(
        source_kind=ROUTE_DERIVATIVE_SOURCE_ARTIFACT,
        is_pure=True,
        is_differentiable=True,
        input_signature=(_string_type_spec(),),
        output_signature=(_string_type_spec(),),
        supported_wrt=("value",),
        seed_contract="cotangent:output",
        transform_version="route-discovery-v1",
        tensor_op_contract_version="tensor-contract-v1",
    ).to_dict()
    tensor_payload = DerivativeMetadata(
        source_graph_id="graph-1",
        transform_version="route-discovery-v1",
        tensor_op_contract_version="tensor-contract-v1",
        wrt_signature=("value",),
        seed_contract="cotangent:output",
    ).to_dict()

    assert "source_graph_id" not in route_payload
    assert "source_kind" not in tensor_payload
    assert "artifact_uri" not in route_payload
    assert hasattr(tc.autodiff, "RouteDerivativeMetadata")
    assert not hasattr(tc, "RouteDerivativeMetadata")
