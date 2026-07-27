from __future__ import annotations

import json

import pytest

import tinychain as tc
from tinychain.library import compile_ir, library_definition
from tinychain.autodiff import (
    ARTIFACT_ERROR_CATEGORIES,
    ArtifactError,
    ArtifactPublicIdentity,
    DerivativeArtifactManifest,
    artifact_digest_input,
    artifact_source_dependencies,
    artifact_manifest_from_program,
    artifact_payload,
    attach_artifact_digest,
    build_derivative_artifact_library,
    canonical_artifact_json,
    compare_artifact_identity,
    compute_artifact_digest,
    public_artifact_identity,
    source_library_dependency_uri,
    validate_artifact_source_metadata,
)
from tinychain.autodiff.graph import AddOperator, TensorNodeRecord
from tinychain.autodiff.protocol import DerivativeMetadata
from tinychain.autodiff.reverse import DerivativeProgram


def _metadata(**overrides: object) -> DerivativeMetadata:
    fields = {
        "source_graph_id": "source-graph-1",
        "transform_version": "0.1.0",
        "tensor_op_contract_version": "0.1.0",
        "wrt_signature": ("x", "y"),
        "seed_contract": "seed matches out",
    }
    fields.update(overrides)
    return DerivativeMetadata(**fields)


def _program(
    metadata: DerivativeMetadata | None = None,
    *,
    node_id: str = "n0",
    output_value_id: str = "out",
    gradients: dict[str, str] | None = None,
    output_gradients: list[str | None] | None = None,
) -> DerivativeProgram:
    gradient_map = gradients or {"x": "dx", "y": "dy"}
    ordered_gradients = output_gradients or ["dx", "dy"]
    node = TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=AddOperator(),
        op_params={},
        input_value_ids=["x", "y"],
    )
    return DerivativeProgram(
        nodes=[node],
        gradients=gradient_map,
        output_gradients=ordered_gradients,
        metadata=metadata or _metadata(),
    )


def _manifest(**overrides: object) -> DerivativeArtifactManifest:
    fields = {
        "artifact_name": "ExampleDerivative",
        "artifact_version": "1.2.3",
        "artifact_publisher": "tester",
        "source_graph_id": "source-graph-1",
        "transform_version": "0.1.0",
        "tensor_op_contract_version": "0.1.0",
        "wrt_signature": ("x", "y"),
        "seed_contract": "seed matches out",
        "visibility": "public",
        "digest_algorithm": "sha256",
        "artifact_digest": "abc123",
        "source_library": "source_pub/source_library",
        "source_library_version": "0.1.0",
        "source_route": "train",
        "source_operator": "add",
    }
    fields.update(overrides)
    return DerivativeArtifactManifest(**fields)


def test_artifact_helpers_are_exported_from_autodiff_package() -> None:
    import tinychain.autodiff as autodiff

    expected_exports = {
        "ARTIFACT_ERROR_CATEGORIES",
        "ArtifactComparisonResult",
        "ArtifactError",
        "ArtifactPayload",
        "ArtifactPublicIdentity",
        "DerivativeArtifactManifest",
        "artifact_digest_input",
        "artifact_source_dependencies",
        "artifact_manifest_from_program",
        "artifact_payload",
        "attach_artifact_digest",
        "build_derivative_artifact_library",
        "canonical_artifact_json",
        "compare_artifact_identity",
        "compute_artifact_digest",
        "public_artifact_identity",
        "source_library_dependency_uri",
        "validate_artifact_source_metadata",
    }

    assert expected_exports.issubset(set(autodiff.__all__))
    for export_name in expected_exports:
        assert hasattr(autodiff, export_name)

    assert not hasattr(tc, "DerivativeArtifactManifest")
    assert not hasattr(tc, "build_derivative_artifact_library")


def test_artifact_error_categories_are_artifact_specific() -> None:
    assert set(ARTIFACT_ERROR_CATEGORIES) == {
        "invalid_manifest",
        "unsupported_visibility",
        "unsupported_digest_algorithm",
        "source_metadata_mismatch",
        "artifact_conflict",
    }


def test_manifest_serializes_to_json_compatible_primitives() -> None:
    manifest = _manifest()

    result = manifest.to_dict()

    assert result == {
        "artifact_name": "ExampleDerivative",
        "artifact_version": "1.2.3",
        "artifact_publisher": "tester",
        "source_graph_id": "source-graph-1",
        "transform_version": "0.1.0",
        "tensor_op_contract_version": "0.1.0",
        "wrt_signature": ["x", "y"],
        "seed_contract": "seed matches out",
        "visibility": "public",
        "digest_algorithm": "sha256",
        "artifact_digest": "abc123",
        "source_library": "source_pub/source_library",
        "source_library_version": "0.1.0",
        "source_route": "train",
        "source_operator": "add",
    }
    assert isinstance(json.dumps(result), str)
    assert DerivativeArtifactManifest.from_dict(result) == manifest


def test_derivative_metadata_to_dict_remains_artifact_free() -> None:
    result = _metadata().to_dict()

    assert result == {
        "source_graph_id": "source-graph-1",
        "transform_version": "0.1.0",
        "tensor_op_contract_version": "0.1.0",
        "wrt_signature": ["x", "y"],
        "seed_contract": "seed matches out",
    }
    assert not any(key.startswith("artifact_") for key in result)
    assert "visibility" not in result


def test_derivative_program_to_dict_remains_artifact_free() -> None:
    result = _program().to_dict()

    assert set(result) == {"nodes", "gradients", "output_gradients", "metadata"}
    assert "manifest" not in result
    assert "artifact" not in result
    assert not any(key.startswith("artifact_") for key in result["metadata"])
    assert "visibility" not in result["metadata"]


def test_manifest_from_program_copies_derivative_metadata_without_mutating_program_payload() -> None:
    program = _program()
    before = program.to_dict()

    manifest = artifact_manifest_from_program(
        program,
        artifact_name="ExampleDerivative",
        artifact_version="1.2.3",
        artifact_publisher="tester",
        visibility="private",
    )

    assert manifest.source_graph_id == program.metadata.source_graph_id
    assert manifest.transform_version == program.metadata.transform_version
    assert manifest.tensor_op_contract_version == program.metadata.tensor_op_contract_version
    assert manifest.wrt_signature == program.metadata.wrt_signature
    assert manifest.seed_contract == program.metadata.seed_contract
    assert manifest.visibility == "private"
    assert program.to_dict() == before
    assert "artifact_name" not in program.to_dict()["metadata"]


@pytest.mark.parametrize(
    ("field_name", "override"),
    [
        ("artifact_name", {"artifact_name": ""}),
        ("artifact_version", {"artifact_version": ""}),
        ("artifact_publisher", {"artifact_publisher": ""}),
        ("source_graph_id", {"source_graph_id": ""}),
        ("transform_version", {"transform_version": ""}),
        ("tensor_op_contract_version", {"tensor_op_contract_version": ""}),
        ("seed_contract", {"seed_contract": ""}),
        ("wrt_signature", {"wrt_signature": ()}),
        ("artifact_digest", {"artifact_digest": ""}),
    ],
)
def test_manifest_rejects_missing_required_identity_or_metadata_fields(
    field_name: str,
    override: dict[str, object],
) -> None:
    with pytest.raises(ArtifactError) as raised:
        _manifest(**override)

    assert raised.value.category == "invalid_manifest"
    assert field_name in raised.value.message


def test_manifest_rejects_unsupported_visibility() -> None:
    with pytest.raises(ArtifactError) as raised:
        _manifest(visibility="hidden")

    assert raised.value.category == "unsupported_visibility"
    assert "hidden" in raised.value.message


def test_manifest_rejects_unsupported_digest_algorithm() -> None:
    with pytest.raises(ArtifactError) as raised:
        _manifest(digest_algorithm="md5")

    assert raised.value.category == "unsupported_digest_algorithm"
    assert "md5" in raised.value.message


def test_manifest_accepts_absent_optional_source_metadata() -> None:
    manifest = _manifest(
        source_library=None,
        source_library_version=None,
        source_route=None,
        source_operator=None,
        artifact_digest=None,
    )

    result = manifest.to_dict()

    assert result["source_library"] is None
    assert result["source_library_version"] is None
    assert result["source_route"] is None
    assert result["source_operator"] is None
    assert result["artifact_digest"] is None


def test_manifest_rejects_empty_optional_source_metadata() -> None:
    with pytest.raises(ArtifactError) as raised:
        _manifest(source_library="")

    assert raised.value.category == "invalid_manifest"
    assert "source_library" in raised.value.message


def test_source_metadata_compatibility_accepts_matching_program() -> None:
    manifest = _manifest()

    validate_artifact_source_metadata(manifest, _program())
    manifest.validate_source_metadata(_program())


@pytest.mark.parametrize(
    ("field_name", "metadata"),
    [
        ("source_graph_id", _metadata(source_graph_id="other-graph")),
        ("transform_version", _metadata(transform_version="0.2.0")),
        ("tensor_op_contract_version", _metadata(tensor_op_contract_version="0.2.0")),
        ("wrt_signature", _metadata(wrt_signature=("x",))),
        ("seed_contract", _metadata(seed_contract="other seed")),
    ],
)
def test_source_metadata_mismatches_raise_categorized_safe_errors(
    field_name: str,
    metadata: DerivativeMetadata,
) -> None:
    program = _program(metadata)

    with pytest.raises(ArtifactError) as raised:
        validate_artifact_source_metadata(_manifest(), program)

    assert raised.value.category == "source_metadata_mismatch"
    assert field_name in raised.value.message
    assert "nodes" not in raised.value.message
    assert "gradients" not in raised.value.message


def test_canonical_artifact_json_uses_sorted_keys_and_compact_separators() -> None:
    payload = {"zeta": [2, 1], "alpha": {"beta": True}}

    assert canonical_artifact_json(payload) == '{"alpha":{"beta":true},"zeta":[2,1]}'


def test_digest_input_includes_manifest_and_program_but_nulls_digest() -> None:
    manifest = _manifest(artifact_digest="existing-digest")
    program = _program()

    result = artifact_digest_input(manifest, program)

    assert result["manifest"]["artifact_digest"] is None
    assert result["manifest"]["artifact_name"] == "ExampleDerivative"
    assert result["manifest"]["source_graph_id"] == "source-graph-1"
    assert result["program"] == program.to_dict()
    assert json.dumps(result)


def test_artifact_digest_is_deterministic_sha256_and_distinct_from_source_graph_id() -> None:
    manifest = _manifest(artifact_digest=None)
    program = _program()

    first_digest = compute_artifact_digest(manifest, program)
    second_digest = compute_artifact_digest(_manifest(artifact_digest="ignored"), _program())

    assert first_digest == second_digest
    assert len(first_digest) == 64
    assert all(character in "0123456789abcdef" for character in first_digest)
    assert first_digest != program.metadata.source_graph_id


@pytest.mark.parametrize(
    "changed_manifest",
    [
        _manifest(artifact_digest=None, artifact_name="OtherDerivative"),
        _manifest(artifact_digest=None, artifact_version="2.0.0"),
        _manifest(artifact_digest=None, artifact_publisher="other"),
        _manifest(artifact_digest=None, transform_version="0.2.0"),
        _manifest(artifact_digest=None, tensor_op_contract_version="0.2.0"),
        _manifest(artifact_digest=None, visibility="private"),
    ],
)
def test_artifact_digest_changes_when_identity_compatibility_or_visibility_changes(
    changed_manifest: DerivativeArtifactManifest,
) -> None:
    program = _program(
        _metadata(
            transform_version=changed_manifest.transform_version,
            tensor_op_contract_version=changed_manifest.tensor_op_contract_version,
        )
    )
    baseline = compute_artifact_digest(_manifest(artifact_digest=None), _program())

    assert compute_artifact_digest(changed_manifest, program) != baseline


def test_artifact_digest_changes_when_program_payload_changes() -> None:
    manifest = _manifest(artifact_digest=None)
    baseline = compute_artifact_digest(manifest, _program())

    assert compute_artifact_digest(manifest, _program(node_id="other-node")) != baseline
    assert compute_artifact_digest(
        manifest,
        _program(gradients={"x": "other-dx", "y": "dy"}),
    ) != baseline


def test_attach_artifact_digest_returns_new_manifest_without_mutating_input() -> None:
    manifest = _manifest(artifact_digest=None)
    program = _program()

    result = attach_artifact_digest(manifest, program)

    assert manifest.artifact_digest is None
    assert result.artifact_digest == compute_artifact_digest(manifest, program)


def test_artifact_payload_contains_computed_manifest_and_program() -> None:
    manifest = _manifest(artifact_digest=None)
    program = _program()

    result = artifact_payload(manifest, program)

    assert result == {
        "manifest": {
            "artifact_name": "ExampleDerivative",
            "artifact_version": "1.2.3",
            "artifact_publisher": "tester",
            "source_graph_id": "source-graph-1",
            "transform_version": "0.1.0",
            "tensor_op_contract_version": "0.1.0",
            "wrt_signature": ["x", "y"],
            "seed_contract": "seed matches out",
            "visibility": "public",
            "digest_algorithm": "sha256",
            "artifact_digest": compute_artifact_digest(manifest, program),
            "source_library": "source_pub/source_library",
            "source_library_version": "0.1.0",
            "source_route": "train",
            "source_operator": "add",
        },
        "program": program.to_dict(),
    }
    assert json.loads(json.dumps(result)) == result


def test_artifact_payload_uses_derivative_program_to_dict_result() -> None:
    program = _program(output_value_id="custom-out", output_gradients=["dx", None])

    result = artifact_payload(_manifest(artifact_digest=None), program)

    assert result["program"] == program.to_dict()
    assert result["program"]["nodes"][0]["output_value_id"] == "custom-out"
    assert result["program"]["output_gradients"] == ["dx", None]


def test_public_artifact_identity_returns_library_uri() -> None:
    identity = public_artifact_identity(_manifest())

    assert identity == ArtifactPublicIdentity(
        publisher="tester",
        name="example_derivative",
        version="1.2.3",
    )
    assert identity.to_uri().path == "/lib/tester/example_derivative/1.2.3"


def test_compare_artifact_identity_accepts_idempotent_repeat() -> None:
    result = compare_artifact_identity(
        _manifest(artifact_name="ExampleDerivative", artifact_digest="same-digest"),
        _manifest(artifact_name="example_derivative", artifact_digest="same-digest"),
    )

    assert result.identity.name == "example_derivative"
    assert result.candidate_identity.name == "example_derivative"
    assert result.is_idempotent is True
    assert result.is_conflict is False


def test_compare_artifact_identity_allows_different_public_identity() -> None:
    result = compare_artifact_identity(
        _manifest(artifact_digest="digest-a"),
        _manifest(artifact_name="OtherDerivative", artifact_digest="digest-b"),
    )

    assert result.is_idempotent is False
    assert result.is_conflict is False
    assert result.candidate_identity.name == "other_derivative"


def test_compare_artifact_identity_raises_safe_conflict_error() -> None:
    with pytest.raises(ArtifactError) as raised:
        compare_artifact_identity(
            _manifest(artifact_digest="digest-a"),
            _manifest(artifact_digest="digest-b"),
        )

    assert raised.value.category == "artifact_conflict"
    assert "/lib/tester/example_derivative/1.2.3" in raised.value.message
    assert "digest-a" in raised.value.message
    assert "digest-b" in raised.value.message
    assert "nodes" not in raised.value.message
    assert "gradients" not in raised.value.message


def test_compare_artifact_identity_conflicts_on_same_derived_public_name() -> None:
    with pytest.raises(ArtifactError) as raised:
        compare_artifact_identity(
            _manifest(artifact_name="ExampleDerivative", artifact_digest="digest-a"),
            _manifest(artifact_name="example_derivative", artifact_digest="digest-b"),
        )

    assert raised.value.category == "artifact_conflict"
    assert "/lib/tester/example_derivative/1.2.3" in raised.value.message


def test_public_artifact_identity_rejects_invalid_derived_resource_name() -> None:
    with pytest.raises(ArtifactError) as raised:
        public_artifact_identity(_manifest(artifact_name="_InvalidDerivative"))

    assert raised.value.category == "invalid_manifest"
    assert "canonical resource name" in raised.value.message


def test_compare_artifact_identity_requires_digests_for_matching_identity() -> None:
    with pytest.raises(ArtifactError) as raised:
        compare_artifact_identity(
            _manifest(artifact_digest=None),
            _manifest(artifact_digest="digest-b"),
        )

    assert raised.value.category == "invalid_manifest"
    assert "artifact_digest" in raised.value.message


def test_graph_only_artifact_has_no_source_dependencies() -> None:
    manifest = _manifest(source_library=None, source_library_version=None)

    assert artifact_source_dependencies(manifest) == ()


def test_source_library_dependency_uri_accepts_publisher_name_and_version() -> None:
    dependency = source_library_dependency_uri("source_pub/source_library", "0.1.0")

    assert dependency is not None
    assert dependency.path == "/lib/source_pub/source_library/0.1.0"
    assert artifact_source_dependencies(_manifest()) == (dependency,)


def test_source_library_dependency_uri_accepts_canonical_library_path() -> None:
    dependency = source_library_dependency_uri(
        "/lib/source_pub/source_library/0.1.0",
        "0.1.0",
    )

    assert dependency is not None
    assert dependency.path == "/lib/source_pub/source_library/0.1.0"


@pytest.mark.parametrize(
    "override",
    [
        {"source_library": "source_pub/source_library", "source_library_version": None},
        {"source_library": None, "source_library_version": "0.1.0"},
        {"source_library": "/not-lib/source_library/0.1.0"},
        {
            "source_library": "/lib/source_pub/source_library/0.2.0",
            "source_library_version": "0.1.0",
        },
    ],
)
def test_source_library_dependency_validation_rejects_invalid_dependency_metadata(
    override: dict[str, object],
) -> None:
    with pytest.raises(ArtifactError) as raised:
        artifact_source_dependencies(_manifest(**override))

    assert raised.value.category == "invalid_manifest"

def test_artifact_error_round_trips_and_rejects_unknown_category() -> None:
    error = ArtifactError("invalid_manifest", "bad manifest")
    assert ArtifactError.from_dict(error.to_dict()) == error

    with pytest.raises(ValueError, match="unknown artifact error category"):
        ArtifactError("not_real", "bad")


def test_build_derivative_artifact_library_returns_normal_library_subclass() -> None:
    artifact_library = build_derivative_artifact_library(
        publisher="tester",
        class_name="ExampleDerivative",
        version="1.2.3",
        program=_program(),
        visibility="internal",
        source_library="source_pub/source_library",
        source_library_version="0.1.0",
        source_route="train",
        source_operator="add",
    )

    assert issubclass(artifact_library, tc.Library)
    assert "__init__" not in artifact_library.__dict__
    assert "name" not in artifact_library.__dict__
    assert artifact_library.publisher == "tester"
    assert artifact_library.version == "1.2.3"
    assert artifact_library.class_id().path == "/lib/tester/example_derivative/1.2.3"
    assert artifact_library().dependencies == (
        tc.URI(path=tc.URI.of("lib", "source_pub", "source_library", "0.1.0")),
    )


def test_build_derivative_artifact_library_compile_ir_emits_static_get_value_route() -> None:
    program = _program()
    artifact_library = build_derivative_artifact_library(
        publisher="tester",
        class_name="ExampleDerivative",
        version="1.2.3",
        program=program,
        source_library="source_pub/source_library",
        source_library_version="0.1.0",
    )
    expected_manifest = artifact_manifest_from_program(
        program,
        artifact_name="ExampleDerivative",
        artifact_version="1.2.3",
        artifact_publisher="tester",
        source_library="source_pub/source_library",
        source_library_version="0.1.0",
    )

    result = compile_ir(artifact_library)

    assert result["schema"] == {
        "id": "/lib/tester/example_derivative/1.2.3",
        "version": "1.2.3",
        "dependencies": ["/lib/source_pub/source_library/0.1.0"],
    }
    assert len(result["routes"]) == 1
    route = result["routes"][0]
    assert route["path"] == "/artifact"
    assert set(route) == {"path", "value"}
    assert route["value"] == artifact_payload(expected_manifest, program)


def test_build_derivative_artifact_library_library_definition_is_v1_style_payload() -> None:
    program = _program()
    artifact_library = build_derivative_artifact_library(
        publisher="tester",
        class_name="ExampleDerivative",
        version="1.2.3",
        program=program,
    )
    expected_manifest = artifact_manifest_from_program(
        program,
        artifact_name="ExampleDerivative",
        artifact_version="1.2.3",
        artifact_publisher="tester",
    )

    result = library_definition(artifact_library)

    assert list(result) == ["/lib/tester/example_derivative/1.2.3"]
    assert result["/lib/tester/example_derivative/1.2.3"] == {
        "artifact": artifact_payload(expected_manifest, program)
    }


def test_build_derivative_artifact_library_omits_dependencies_for_graph_only_artifact() -> None:
    artifact_library = build_derivative_artifact_library(
        publisher="tester",
        class_name="ExampleDerivative",
        version="1.2.3",
        program=_program(),
    )

    assert compile_ir(artifact_library)["schema"]["dependencies"] == []


def test_build_derivative_artifact_library_rejects_invalid_public_identity() -> None:
    with pytest.raises(ArtifactError) as raised:
        build_derivative_artifact_library(
            publisher="tester",
            class_name="Invalid/Derivative",
            version="1.2.3",
            program=_program(),
    )

    assert raised.value.category == "invalid_manifest"
    assert "resource_name must match" in raised.value.message
