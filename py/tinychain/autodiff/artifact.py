from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..serialize import serialize
from .protocol import DerivativeMetadata


ARTIFACT_ERROR_CATEGORIES: tuple[str, ...] = (
    "invalid_manifest",
    "unsupported_visibility",
    "unsupported_digest_algorithm",
    "source_metadata_mismatch",
)

SUPPORTED_ARTIFACT_VISIBILITIES: tuple[str, ...] = ("public", "private", "internal")
SUPPORTED_ARTIFACT_DIGEST_ALGORITHMS: tuple[str, ...] = ("sha256",)


@dataclass(frozen=True)
class ArtifactError(Exception):
    category: str
    message: str

    allowed_categories: ClassVar[tuple[str, ...]] = ARTIFACT_ERROR_CATEGORIES

    def __post_init__(self) -> None:
        if self.category not in self.allowed_categories:
            raise ValueError(f"unknown artifact error category: {self.category}")
        Exception.__init__(self, f"{self.category}: {self.message}")

    def to_dict(self) -> dict[str, str]:
        return serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ArtifactError:
        return cls(category=str(data["category"]), message=str(data["message"]))


@dataclass(frozen=True)
class DerivativeArtifactManifest:
    artifact_name: str
    artifact_version: str
    artifact_publisher: str
    source_graph_id: str
    transform_version: str
    tensor_op_contract_version: str
    wrt_signature: tuple[str, ...]
    seed_contract: str
    visibility: str = "public"
    digest_algorithm: str = "sha256"
    digest: str | None = None
    source_library: str | None = None
    source_route: str | None = None
    source_operator: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty("artifact_name", self.artifact_name)
        _require_non_empty("artifact_version", self.artifact_version)
        _require_non_empty("artifact_publisher", self.artifact_publisher)
        _require_non_empty("source_graph_id", self.source_graph_id)
        _require_non_empty("transform_version", self.transform_version)
        _require_non_empty("tensor_op_contract_version", self.tensor_op_contract_version)
        _require_non_empty("seed_contract", self.seed_contract)
        if not self.wrt_signature:
            raise ArtifactError("invalid_manifest", "wrt_signature must not be empty")
        for value_id in self.wrt_signature:
            _require_non_empty("wrt_signature", value_id)
        if self.visibility not in SUPPORTED_ARTIFACT_VISIBILITIES:
            raise ArtifactError(
                "unsupported_visibility",
                f"unsupported artifact visibility {self.visibility!r}",
            )
        if self.digest_algorithm not in SUPPORTED_ARTIFACT_DIGEST_ALGORITHMS:
            raise ArtifactError(
                "unsupported_digest_algorithm",
                f"unsupported artifact digest algorithm {self.digest_algorithm!r}",
            )
        if self.digest is not None:
            _require_non_empty("digest", self.digest)
        _validate_optional_text("source_library", self.source_library)
        _validate_optional_text("source_route", self.source_route)
        _validate_optional_text("source_operator", self.source_operator)

    def to_dict(self) -> dict[str, object]:
        return serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> DerivativeArtifactManifest:
        return cls(
            artifact_name=str(data["artifact_name"]),
            artifact_version=str(data["artifact_version"]),
            artifact_publisher=str(data["artifact_publisher"]),
            source_graph_id=str(data["source_graph_id"]),
            transform_version=str(data["transform_version"]),
            tensor_op_contract_version=str(data["tensor_op_contract_version"]),
            wrt_signature=tuple(str(item) for item in data["wrt_signature"]),
            seed_contract=str(data["seed_contract"]),
            visibility=str(data.get("visibility", "public")),
            digest_algorithm=str(data.get("digest_algorithm", "sha256")),
            digest=_optional_string(data.get("digest")),
            source_library=_optional_string(data.get("source_library")),
            source_route=_optional_string(data.get("source_route")),
            source_operator=_optional_string(data.get("source_operator")),
        )

    @classmethod
    def from_program(
        cls,
        program: object,
        *,
        artifact_name: str,
        artifact_version: str,
        artifact_publisher: str,
        visibility: str = "public",
        digest_algorithm: str = "sha256",
        digest: str | None = None,
        source_library: str | None = None,
        source_route: str | None = None,
        source_operator: str | None = None,
    ) -> DerivativeArtifactManifest:
        metadata = _program_metadata(program)
        return cls(
            artifact_name=artifact_name,
            artifact_version=artifact_version,
            artifact_publisher=artifact_publisher,
            source_graph_id=metadata.source_graph_id,
            transform_version=metadata.transform_version,
            tensor_op_contract_version=metadata.tensor_op_contract_version,
            wrt_signature=metadata.wrt_signature,
            seed_contract=metadata.seed_contract,
            visibility=visibility,
            digest_algorithm=digest_algorithm,
            digest=digest,
            source_library=source_library,
            source_route=source_route,
            source_operator=source_operator,
        )

    def validate_source_metadata(self, program: object) -> None:
        validate_artifact_source_metadata(self, program)


def artifact_manifest_from_program(
    program: object,
    *,
    artifact_name: str,
    artifact_version: str,
    artifact_publisher: str,
    visibility: str = "public",
    digest_algorithm: str = "sha256",
    digest: str | None = None,
    source_library: str | None = None,
    source_route: str | None = None,
    source_operator: str | None = None,
) -> DerivativeArtifactManifest:
    return DerivativeArtifactManifest.from_program(
        program,
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        artifact_publisher=artifact_publisher,
        visibility=visibility,
        digest_algorithm=digest_algorithm,
        digest=digest,
        source_library=source_library,
        source_route=source_route,
        source_operator=source_operator,
    )


def validate_artifact_source_metadata(
    manifest: DerivativeArtifactManifest,
    program: object,
) -> None:
    metadata = _program_metadata(program)
    expected = {
        "source_graph_id": manifest.source_graph_id,
        "transform_version": manifest.transform_version,
        "tensor_op_contract_version": manifest.tensor_op_contract_version,
        "wrt_signature": manifest.wrt_signature,
        "seed_contract": manifest.seed_contract,
    }
    actual = {
        "source_graph_id": metadata.source_graph_id,
        "transform_version": metadata.transform_version,
        "tensor_op_contract_version": metadata.tensor_op_contract_version,
        "wrt_signature": metadata.wrt_signature,
        "seed_contract": metadata.seed_contract,
    }

    mismatched_fields = [
        field_name
        for field_name, expected_value in expected.items()
        if actual[field_name] != expected_value
    ]
    if mismatched_fields:
        joined_fields = ", ".join(mismatched_fields)
        raise ArtifactError(
            "source_metadata_mismatch",
            f"artifact manifest does not match derivative metadata fields: {joined_fields}",
        )


def _program_metadata(program: object) -> DerivativeMetadata:
    metadata = getattr(program, "metadata", None)
    if not isinstance(metadata, DerivativeMetadata):
        raise ArtifactError(
            "invalid_manifest",
            "program must expose DerivativeMetadata as metadata",
        )
    return metadata


def _require_non_empty(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ArtifactError("invalid_manifest", f"{field_name} must be a non-empty string")


def _validate_optional_text(field_name: str, value: str | None) -> None:
    if value is not None:
        _require_non_empty(field_name, value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
