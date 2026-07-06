from __future__ import annotations

import json

import pytest

from tinychain.autodiff.artifact import (
    ARTIFACT_ERROR_CATEGORIES,
    ArtifactError,
    DerivativeArtifactManifest,
    artifact_digest_input,
    artifact_manifest_from_program,
    artifact_payload,
    attach_artifact_digest,
    canonical_artifact_json,
    compute_artifact_digest,
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
        "source_library": "source.library",
        "source_route": "train",
        "source_operator": "add",
    }
    fields.update(overrides)
    return DerivativeArtifactManifest(**fields)


def test_artifact_error_categories_are_artifact_specific() -> None:
    assert set(ARTIFACT_ERROR_CATEGORIES) == {
        "invalid_manifest",
        "unsupported_visibility",
        "unsupported_digest_algorithm",
        "source_metadata_mismatch",
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
        "source_library": "source.library",
        "source_route": "train",
        "source_operator": "add",
    }
    assert isinstance(json.dumps(result), str)
    assert DerivativeArtifactManifest.from_dict(result) == manifest


def test_manifest_from_dict_accepts_legacy_digest_key() -> None:
    manifest = DerivativeArtifactManifest.from_dict(_manifest().to_dict() | {"digest": "legacy"})

    assert manifest.artifact_digest == "abc123"


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
        source_route=None,
        source_operator=None,
        artifact_digest=None,
    )

    result = manifest.to_dict()

    assert result["source_library"] is None
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
            "source_library": "source.library",
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


def test_artifact_error_round_trips_and_rejects_unknown_category() -> None:
    error = ArtifactError("invalid_manifest", "bad manifest")
    assert ArtifactError.from_dict(error.to_dict()) == error

    with pytest.raises(ValueError, match="unknown artifact error category"):
        ArtifactError("not_real", "bad")
