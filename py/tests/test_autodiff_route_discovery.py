from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tinychain as tc
from tinychain.autodiff import (
    ROUTE_DERIVATIVE_COMPATIBILITY_COMPATIBLE,
    ROUTE_DERIVATIVE_COMPATIBILITY_NOT_VALIDATED,
    ROUTE_DERIVATIVE_SOURCE_ARTIFACT,
    ROUTE_DERIVATIVE_SOURCE_NON_DIFFERENTIABLE,
    ROUTE_DERIVATIVE_SOURCE_UNSUPPORTED,
    AutodiffError,
    RouteDerivativeMetadata,
    RouteDerivativePlan,
    discover_route_derivative,
)
from tinychain.autodiff.protocol import DerivativeMetadata
from tinychain.autodiff.routes import (
    RouteDerivativeIdentity,
    extract_route_identity,
    lookup_route_derivative_metadata,
)
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


def _route_metadata(*, seed_contract: str = "cotangent:output") -> RouteDerivativeMetadata:
    value_type = _string_type_spec()
    return RouteDerivativeMetadata(
        source_kind=ROUTE_DERIVATIVE_SOURCE_ARTIFACT,
        is_pure=True,
        is_differentiable=True,
        input_signature=(value_type,),
        output_signature=(value_type,),
        supported_wrt=("value",),
        seed_contract=seed_contract,
        transform_version="route-discovery-v1",
        tensor_op_contract_version="tensor-contract-v1",
        artifact_uri="/lib/example-devco/create_derivative/0.1.0",
        artifact_digest="sha256:abc123",
        artifact_source_library="example-devco/route_identity_library",
        artifact_source_library_version="0.1.0",
        artifact_source_route="/create",
        artifact_visibility="public",
    )


def test_lookup_route_derivative_metadata_by_route_name() -> None:
    metadata = _route_metadata()

    class NameMetadataLibrary(RouteIdentityLibrary):
        derivative_routes = {"create": metadata}

    assert lookup_route_derivative_metadata(NameMetadataLibrary().create) == metadata


def test_lookup_route_derivative_metadata_by_route_path_from_mapping() -> None:
    metadata = _route_metadata()

    class PathMetadataLibrary(RouteIdentityLibrary):
        derivative_routes = {"/create": metadata.to_dict()}

    assert lookup_route_derivative_metadata(PathMetadataLibrary().create) == metadata


def test_lookup_route_derivative_metadata_missing_route_is_distinguishable() -> None:
    class MissingMetadataLibrary(RouteIdentityLibrary):
        derivative_routes = {"fetch": _route_metadata()}

    with pytest.raises(AutodiffError) as exc_info:
        lookup_route_derivative_metadata(MissingMetadataLibrary().create)

    assert exc_info.value.category == "non_differentiable_route"
    assert "missing derivative metadata" in str(exc_info.value)
    assert "'create'" in str(exc_info.value)
    assert "'/create'" in str(exc_info.value)


def test_lookup_route_derivative_metadata_rejects_ambiguous_duplicate_keys() -> None:
    class AmbiguousMetadataLibrary(RouteIdentityLibrary):
        derivative_routes = {
            "create": _route_metadata(seed_contract="cotangent:output"),
            "/create": _route_metadata(seed_contract="cotangent:alternate"),
        }

    with pytest.raises(AutodiffError) as exc_info:
        lookup_route_derivative_metadata(AmbiguousMetadataLibrary().create)

    assert exc_info.value.category == "non_differentiable_route"
    assert "ambiguous derivative metadata" in str(exc_info.value)


def test_lookup_route_derivative_metadata_rejects_malformed_metadata_separately() -> None:
    class MalformedMetadataLibrary(RouteIdentityLibrary):
        derivative_routes = {"create": {"source_kind": ROUTE_DERIVATIVE_SOURCE_ARTIFACT}}

    with pytest.raises(AutodiffError) as exc_info:
        lookup_route_derivative_metadata(MalformedMetadataLibrary().create)

    assert exc_info.value.category == "non_differentiable_route"
    assert "malformed derivative metadata" in str(exc_info.value)
    assert "missing derivative metadata" not in str(exc_info.value)


def _tensor_type_spec(
    *,
    dtype: str | None = "f32",
    shape: list[int] | None = None,
) -> TypeSpec:
    params: dict[str, object] = {}
    if dtype is not None:
        params["dtype"] = dtype
    if shape is not None:
        params["shape"] = shape
    return TypeSpec(class_uri="/state/collection/tensor", params=params)


def _route_tensor_metadata(
    *,
    source_kind: str = ROUTE_DERIVATIVE_SOURCE_ARTIFACT,
    is_pure: bool = True,
    is_differentiable: bool = True,
    input_signature: tuple[TypeSpec | dict[str, object], ...] | None = None,
    output_signature: tuple[TypeSpec | dict[str, object], ...] | None = None,
    supported_wrt: tuple[str, ...] = ("value",),
    artifact_digest: str | None = "sha256:abc123",
    artifact_source_library: str | None = "example-devco/route_identity_library",
    artifact_source_library_version: str | None = "0.1.0",
    artifact_source_route: str | None = "/create",
    artifact_visibility: str | None = "public",
) -> RouteDerivativeMetadata:
    input_types = input_signature or (_tensor_type_spec(shape=[2, 2]),)
    output_types = output_signature or (_tensor_type_spec(shape=[2, 2]),)
    return RouteDerivativeMetadata(
        source_kind=source_kind,
        is_pure=is_pure,
        is_differentiable=is_differentiable,
        input_signature=input_types,
        output_signature=output_types,
        supported_wrt=supported_wrt,
        seed_contract="seed matches output",
        transform_version="route-discovery-v1",
        tensor_op_contract_version="tensor-contract-v1",
        artifact_uri="/lib/example-devco/create_derivative/0.1.0",
        artifact_digest=artifact_digest,
        artifact_source_library=artifact_source_library,
        artifact_source_library_version=artifact_source_library_version,
        artifact_source_route=artifact_source_route,
        artifact_visibility=artifact_visibility,
    )


def test_discover_route_derivative_returns_plan_for_valid_metadata_mapping() -> None:
    class ValidMetadataLibrary(RouteIdentityLibrary):
        pass

    metadata = _route_tensor_metadata(
        input_signature=(_tensor_type_spec(shape=[2, 2]).to_dict(),),
        output_signature=(_tensor_type_spec(shape=[2, 2]).to_dict(),),
        artifact_source_library="example-devco/valid_metadata_library",
    )
    ValidMetadataLibrary.derivative_routes = {"create": metadata.to_dict()}

    plan = discover_route_derivative(
        ValidMetadataLibrary().create,
        wrt="value",
        seed="upstream",
        seed_typespec=_tensor_type_spec(shape=[2, 2]).to_dict(),
    )

    assert isinstance(plan, RouteDerivativePlan)
    assert plan.route_identity.route_name == "create"
    assert plan.requested_wrt == ("value",)
    assert plan.seed_contract == "seed matches output"
    assert plan.source_kind == ROUTE_DERIVATIVE_SOURCE_ARTIFACT
    assert plan.compatibility_status == ROUTE_DERIVATIVE_COMPATIBILITY_COMPATIBLE
    assert plan.artifact_uri == "/lib/example-devco/create_derivative/0.1.0"
    assert plan.artifact_digest == "sha256:abc123"
    assert plan.artifact_visibility == "public"


@pytest.mark.parametrize("shape", [[2], [2, 2], []])
def test_discover_route_derivative_accepts_integer_shape_metadata(shape: list[int]) -> None:
    class ValidShapeLibrary(RouteIdentityLibrary):
        pass

    metadata = _route_tensor_metadata(
        input_signature=(_tensor_type_spec(shape=shape),),
        output_signature=(_tensor_type_spec(shape=shape),),
        artifact_source_library="example-devco/valid_shape_library",
    )
    ValidShapeLibrary.derivative_routes = {"create": metadata}

    plan = discover_route_derivative(
        ValidShapeLibrary().create,
        wrt="value",
        seed_typespec=_tensor_type_spec(shape=shape),
    )

    assert plan.route_identity.route_name == "create"
    assert plan.requested_wrt == ("value",)


def test_discover_route_derivative_missing_metadata_sanitizes_route_message() -> None:
    with pytest.raises(AutodiffError) as exc_info:
        discover_route_derivative(RouteIdentityLibrary().create, wrt=("value",))

    assert exc_info.value.category == "non_differentiable_route"
    assert "/path/create" in str(exc_info.value)
    assert "route body must not execute" not in str(exc_info.value)


def test_tc_grad_route_target_without_metadata_raises_route_specific_error() -> None:
    with pytest.raises(AutodiffError) as exc_info:
        tc.grad(RouteIdentityLibrary().create, wrt=("value",))

    assert exc_info.value.category == "non_differentiable_route"
    assert "missing derivative metadata" in str(exc_info.value)


def test_tc_grad_route_target_returns_plan_for_valid_artifact_metadata() -> None:
    class TcGradMetadataLibrary(RouteIdentityLibrary):
        pass

    metadata = _route_tensor_metadata(
        input_signature=(_tensor_type_spec(shape=[2, 2]),),
        output_signature=(_tensor_type_spec(shape=[2, 2]),),
        artifact_source_library="example-devco/tc_grad_metadata_library",
    )
    TcGradMetadataLibrary.derivative_routes = {"create": metadata}

    plan = tc.grad(
        TcGradMetadataLibrary().create,
        wrt=("value",),
        seed="upstream",
        seed_typespec=_tensor_type_spec(shape=[2, 2]).to_dict(),
    )

    assert isinstance(plan, RouteDerivativePlan)
    assert plan.route_identity.route_name == "create"
    assert plan.requested_wrt == ("value",)
    assert plan.seed_contract == "seed matches output"
    assert plan.source_kind == ROUTE_DERIVATIVE_SOURCE_ARTIFACT
    assert plan.compatibility_status == ROUTE_DERIVATIVE_COMPATIBILITY_COMPATIBLE
    assert plan.artifact_uri == "/lib/example-devco/create_derivative/0.1.0"
    assert plan.artifact_digest == "sha256:abc123"


def test_discover_route_derivative_rejects_side_effecting_metadata() -> None:
    class SideEffectMetadataLibrary(RouteIdentityLibrary):
        derivative_routes = {"create": _route_tensor_metadata(is_pure=False)}

    with pytest.raises(AutodiffError) as exc_info:
        discover_route_derivative(SideEffectMetadataLibrary().create, wrt=("value",))

    assert exc_info.value.category == "side_effecting_route_unsupported"


def test_discover_route_derivative_rejects_unknown_purity_metadata() -> None:
    metadata = _route_tensor_metadata().to_dict()
    metadata["is_pure"] = None

    class UnknownPurityMetadataLibrary(RouteIdentityLibrary):
        derivative_routes = {"create": metadata}

    with pytest.raises(AutodiffError) as exc_info:
        discover_route_derivative(UnknownPurityMetadataLibrary().create, wrt=("value",))

    assert exc_info.value.category == "side_effecting_route_unsupported"


def test_discover_route_derivative_rejects_invalid_wrt() -> None:
    class WrtMetadataLibrary(RouteIdentityLibrary):
        derivative_routes = {"create": _route_tensor_metadata(supported_wrt=("value",))}

    with pytest.raises(AutodiffError) as exc_info:
        discover_route_derivative(WrtMetadataLibrary().create, wrt=("other",))

    assert exc_info.value.category == "non_differentiable_route"
    assert "other" in str(exc_info.value)


def test_discover_route_derivative_rejects_missing_dtype_metadata() -> None:
    class MissingDtypeLibrary(RouteIdentityLibrary):
        derivative_routes = {
            "create": _route_tensor_metadata(input_signature=(_tensor_type_spec(dtype=None, shape=[2]),))
        }

    with pytest.raises(AutodiffError) as exc_info:
        discover_route_derivative(MissingDtypeLibrary().create, wrt=("value",))

    assert exc_info.value.category == "missing_dtype_metadata"


def test_discover_route_derivative_rejects_missing_shape_metadata() -> None:
    class MissingShapeLibrary(RouteIdentityLibrary):
        derivative_routes = {
            "create": _route_tensor_metadata(output_signature=(_tensor_type_spec(shape=None),))
        }

    with pytest.raises(AutodiffError) as exc_info:
        discover_route_derivative(MissingShapeLibrary().create, wrt=("value",))

    assert exc_info.value.category == "missing_shape_metadata"


@pytest.mark.parametrize(
    "shape",
    ["22", 22, ["two"], [2.5], [True]],
)
def test_discover_route_derivative_rejects_malformed_shape_metadata(shape: object) -> None:
    class MalformedShapeLibrary(RouteIdentityLibrary):
        derivative_routes = {
            "create": _route_tensor_metadata(
                output_signature=(_tensor_type_spec(shape=shape),)
            )
        }

    with pytest.raises(AutodiffError) as exc_info:
        discover_route_derivative(MalformedShapeLibrary().create, wrt=("value",))

    assert exc_info.value.category == "missing_shape_metadata"
    assert "route body must not execute" not in str(exc_info.value)


def test_discover_route_derivative_rejects_non_floating_dtype() -> None:
    class IntegerTensorLibrary(RouteIdentityLibrary):
        derivative_routes = {
            "create": _route_tensor_metadata(input_signature=(_tensor_type_spec(dtype="i64", shape=[2]),))
        }

    with pytest.raises(AutodiffError) as exc_info:
        discover_route_derivative(IntegerTensorLibrary().create, wrt=("value",))

    assert exc_info.value.category == "dtype_not_differentiable"


def test_discover_route_derivative_rejects_wrong_artifact_source_library() -> None:
    class WrongSourceLibrary(RouteIdentityLibrary):
        derivative_routes = {
            "create": _route_tensor_metadata(artifact_source_library="other/create")
        }

    with pytest.raises(AutodiffError) as exc_info:
        discover_route_derivative(WrongSourceLibrary().create, wrt=("value",))

    assert exc_info.value.category == "non_differentiable_route"
    assert "source library" in str(exc_info.value)


def test_discover_route_derivative_rejects_wrong_artifact_source_version() -> None:
    class WrongSourceVersion(RouteIdentityLibrary):
        derivative_routes = {
            "create": _route_tensor_metadata(
                artifact_source_library="example-devco/wrong_source_version",
                artifact_source_library_version="9.9.9",
            )
        }

    with pytest.raises(AutodiffError) as exc_info:
        discover_route_derivative(WrongSourceVersion().create, wrt=("value",))

    assert exc_info.value.category == "non_differentiable_route"
    assert "source library version" in str(exc_info.value)


def test_discover_route_derivative_rejects_wrong_artifact_source_route() -> None:
    class WrongSourceRoute(RouteIdentityLibrary):
        derivative_routes = {
            "create": _route_tensor_metadata(
                artifact_source_library="example-devco/wrong_source_route",
                artifact_source_route="fetch",
            )
        }

    with pytest.raises(AutodiffError) as exc_info:
        discover_route_derivative(WrongSourceRoute().create, wrt=("value",))

    assert exc_info.value.category == "non_differentiable_route"
    assert "source route" in str(exc_info.value)


def test_discover_route_derivative_rejects_missing_artifact_digest() -> None:
    metadata = _route_tensor_metadata().to_dict()
    metadata["artifact_digest"] = None

    class MissingArtifactDigest(RouteIdentityLibrary):
        derivative_routes = {"create": metadata}

    with pytest.raises(AutodiffError) as exc_info:
        discover_route_derivative(MissingArtifactDigest().create, wrt=("value",))

    assert exc_info.value.category == "non_differentiable_route"
    assert "artifact_digest" in str(exc_info.value)


def test_discover_route_derivative_rejects_missing_derivative_behavior() -> None:
    class UnsupportedMetadataLibrary(RouteIdentityLibrary):
        derivative_routes = {
            "create": _route_tensor_metadata(
                source_kind=ROUTE_DERIVATIVE_SOURCE_UNSUPPORTED,
                is_differentiable=False,
                supported_wrt=(),
            )
        }

    with pytest.raises(AutodiffError) as exc_info:
        discover_route_derivative(UnsupportedMetadataLibrary().create, wrt=("value",))

    assert exc_info.value.category == "missing_derivative_behavior"


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
        artifact_uri="/lib/example-devco/create_derivative/0.1.0",
        artifact_digest="sha256:abc123",
        artifact_source_library="example-devco/route_identity_library",
        artifact_source_library_version="0.1.0",
        artifact_source_route="/create",
        artifact_visibility="public",
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
        "artifact_uri": "/lib/example-devco/create_derivative/0.1.0",
        "artifact_digest": "sha256:abc123",
        "artifact_source_library": "example-devco/route_identity_library",
        "artifact_source_library_version": "0.1.0",
        "artifact_source_route": "/create",
        "artifact_visibility": "public",
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
        artifact_uri="/lib/example-devco/create_derivative/0.1.0",
        artifact_digest="sha256:abc123",
        artifact_source_library="example-devco/route_identity_library",
        artifact_source_library_version="0.1.0",
        artifact_source_route="/create",
        artifact_visibility="public",
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
    assert route_payload["artifact_uri"] == "/lib/example-devco/create_derivative/0.1.0"
    assert "artifact_uri" not in tensor_payload
    assert hasattr(tc.autodiff, "RouteDerivativeMetadata")
    assert not hasattr(tc, "RouteDerivativeMetadata")
