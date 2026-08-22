from __future__ import annotations

import hashlib
import json
import keyword
from dataclasses import dataclass, replace
from typing import ClassVar

from ..library import Library, Route, get
from ..serialize import serialize
from ..uri import URI, _python_name_to_resource
from ._exception_state import allow_exception_state
from .compile import compile_derivative_program
from .protocol import DerivativeMetadata


ARTIFACT_ERROR_CATEGORIES: tuple[str, ...] = (
    "invalid_manifest",
    "unsupported_visibility",
    "unsupported_digest_algorithm",
    "source_metadata_mismatch",
    "artifact_conflict",
)

SUPPORTED_ARTIFACT_VISIBILITIES: tuple[str, ...] = ("public", "private", "internal")
SUPPORTED_ARTIFACT_DIGEST_ALGORITHMS: tuple[str, ...] = ("sha256",)


@allow_exception_state
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
class ArtifactPublicIdentity:
    publisher: str
    name: str
    version: str

    def __post_init__(self) -> None:
        _require_non_empty("artifact_publisher", self.publisher)
        _require_non_empty("artifact_name", self.name)
        _require_non_empty("artifact_version", self.version)

    def to_uri(self) -> URI:
        try:
            resolved = URI("lib", self.publisher, self.name, self.version)
        except ValueError as exc:
            raise ArtifactError("invalid_manifest", str(exc)) from exc
        if not isinstance(resolved, URI):
            raise ArtifactError("invalid_manifest", "artifact identity must resolve to a URI")
        return resolved


@dataclass(frozen=True)
class ArtifactComparisonResult:
    identity: ArtifactPublicIdentity
    candidate_identity: ArtifactPublicIdentity
    is_idempotent: bool
    is_conflict: bool


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
    artifact_digest: str | None = None
    source_library: str | None = None
    source_library_version: str | None = None
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
        if self.artifact_digest is not None:
            _require_non_empty("artifact_digest", self.artifact_digest)
        _validate_optional_text("source_library", self.source_library)
        _validate_optional_text("source_library_version", self.source_library_version)
        _validate_source_library_dependency(self.source_library, self.source_library_version)
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
            artifact_digest=_optional_string(data.get("artifact_digest")),
            source_library=_optional_string(
                data.get("source_library", data.get("source_library_id"))
            ),
            source_library_version=_optional_string(data.get("source_library_version")),
            source_route=_optional_string(data.get("source_route", data.get("source_route_id"))),
            source_operator=_optional_string(
                data.get("source_operator", data.get("source_operator_id"))
            ),
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
        artifact_digest: str | None = None,
        source_library: str | None = None,
        source_library_version: str | None = None,
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
            artifact_digest=artifact_digest,
            source_library=source_library,
            source_library_version=source_library_version,
            source_route=source_route,
            source_operator=source_operator,
        )

    def validate_source_metadata(self, program: object) -> None:
        validate_artifact_source_metadata(self, program)


ArtifactPayload = dict[str, dict[str, object]]


def artifact_manifest_from_program(
    program: object,
    *,
    artifact_name: str,
    artifact_version: str,
    artifact_publisher: str,
    visibility: str = "public",
    digest_algorithm: str = "sha256",
    artifact_digest: str | None = None,
    source_library: str | None = None,
    source_library_version: str | None = None,
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
        artifact_digest=artifact_digest,
        source_library=source_library,
        source_library_version=source_library_version,
        source_route=source_route,
        source_operator=source_operator,
    )


def canonical_artifact_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def artifact_digest_input(
    manifest: DerivativeArtifactManifest,
    program: object,
) -> ArtifactPayload:
    manifest.validate_source_metadata(program)
    manifest_payload = manifest.to_dict()
    manifest_payload["artifact_digest"] = None
    return {
        "manifest": manifest_payload,
        "program": _program_payload(program),
    }


def compute_artifact_digest(
    manifest: DerivativeArtifactManifest,
    program: object,
) -> str:
    if manifest.digest_algorithm != "sha256":
        raise ArtifactError(
            "unsupported_digest_algorithm",
            f"unsupported artifact digest algorithm {manifest.digest_algorithm!r}",
        )
    canonical_payload = canonical_artifact_json(artifact_digest_input(manifest, program))
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def attach_artifact_digest(
    manifest: DerivativeArtifactManifest,
    program: object,
) -> DerivativeArtifactManifest:
    artifact_digest = compute_artifact_digest(manifest, program)
    return replace(manifest, artifact_digest=artifact_digest)


def artifact_payload(
    manifest: DerivativeArtifactManifest,
    program: object,
) -> ArtifactPayload:
    payload = artifact_digest_input(manifest, program)
    artifact_digest = hashlib.sha256(
        canonical_artifact_json(payload).encode("utf-8")
    ).hexdigest()
    payload["manifest"]["artifact_digest"] = artifact_digest
    json.dumps(payload)
    return payload


def public_artifact_identity(
    manifest: DerivativeArtifactManifest,
) -> ArtifactPublicIdentity:
    return ArtifactPublicIdentity(
        publisher=manifest.artifact_publisher,
        name=_artifact_resource_name(manifest.artifact_name),
        version=manifest.artifact_version,
    )


def compare_artifact_identity(
    existing: DerivativeArtifactManifest,
    candidate: DerivativeArtifactManifest,
) -> ArtifactComparisonResult:
    existing_identity = public_artifact_identity(existing)
    candidate_identity = public_artifact_identity(candidate)
    if existing_identity != candidate_identity:
        return ArtifactComparisonResult(
            identity=existing_identity,
            candidate_identity=candidate_identity,
            is_idempotent=False,
            is_conflict=False,
        )

    existing_digest = _comparison_digest("existing artifact", existing.artifact_digest)
    candidate_digest = _comparison_digest("candidate artifact", candidate.artifact_digest)
    if existing_digest == candidate_digest:
        return ArtifactComparisonResult(
            identity=existing_identity,
            candidate_identity=candidate_identity,
            is_idempotent=True,
            is_conflict=False,
        )

    identity_path = existing_identity.to_uri().path
    raise ArtifactError(
        "artifact_conflict",
        "artifact conflict for "
        f"{identity_path}: existing digest {existing_digest} differs "
        f"from candidate digest {candidate_digest}",
    )


def source_library_dependency_uri(
    source_library: str | None,
    source_library_version: str | None,
) -> URI | None:
    if source_library is None and source_library_version is None:
        return None
    _validate_source_library_dependency(source_library, source_library_version)
    assert source_library is not None
    assert source_library_version is not None

    if source_library.startswith("/") or "://" in source_library:
        try:
            dependency = URI.parse(source_library)
        except ValueError as exc:
            raise ArtifactError("invalid_manifest", str(exc)) from exc
        _validate_library_dependency_path(dependency, source_library_version)
        return dependency

    parts = source_library.split("/")
    if len(parts) != 2:
        raise ArtifactError(
            "invalid_manifest",
            "source_library must be 'publisher/name' or a canonical /lib path",
        )
    publisher, name = parts
    try:
        dependency = URI("lib", publisher, name, source_library_version)
    except ValueError as exc:
        raise ArtifactError("invalid_manifest", str(exc)) from exc
    if not isinstance(dependency, URI):
        raise ArtifactError("invalid_manifest", "source library dependency must resolve to a URI")
    return dependency


def artifact_source_dependencies(
    manifest: DerivativeArtifactManifest,
) -> tuple[URI, ...]:
    dependency = source_library_dependency_uri(
        manifest.source_library,
        manifest.source_library_version,
    )
    if dependency is None:
        return ()
    return (dependency,)


def build_derivative_artifact_library(
    *,
    publisher: str,
    class_name: str,
    version: str,
    program: object,
    visibility: str = "public",
    source_library: str | None = None,
    source_library_version: str | None = None,
    source_route: str | None = None,
    source_operator: str | None = None,
) -> type[Library]:
    manifest = artifact_manifest_from_program(
        program,
        artifact_name=class_name,
        artifact_version=version,
        artifact_publisher=publisher,
        visibility=visibility,
        source_library=source_library,
        source_library_version=source_library_version,
        source_route=source_route,
        source_operator=source_operator,
    )
    payload = artifact_payload(manifest, program)
    dependencies = artifact_source_dependencies(manifest)

    def artifact(self) -> ArtifactPayload:
        return payload

    artifact.__name__ = "artifact"
    artifact.__qualname__ = f"{class_name}.artifact"

    resource_name = public_artifact_identity(manifest).name
    try:
        library_cls = type(
            class_name,
            (Library,),
            {
                "__module__": __name__,
                "publisher": publisher,
                "resource_name": resource_name,
                "version": version,
                "dependencies": dependencies,
                "artifact": get(artifact),
            },
        )
    except TypeError as exc:
        raise ArtifactError("invalid_manifest", str(exc)) from exc

    _validate_artifact_library_identity(library_cls, manifest)
    return library_cls


def build_derivative_execution_library(
    *,
    publisher: str,
    resource_name: str,
    class_name: str,
    version: str,
    program: object,
    route_name: str = "execute",
    artifact_resource_name: str | None = None,
) -> type[Library]:
    """Build a normal installable Library for a compiled derivative program.

    ``resource_name`` is the canonical library path component and is supplied
    explicitly, independently from the Python ``class_name``.
    """
    _validate_route_name(route_name)
    if artifact_resource_name is not None:
        _validate_execution_library_identity(resource_name, artifact_resource_name)

    compiled = compile_derivative_program(program, defer_symbolic_shape_params=True)
    _validate_execution_params(compiled.params)
    opdef = compiled.opdef
    source = _execution_route_source(route_name, compiled.params)
    namespace = {"__opdef": opdef, "tc": __import__("tinychain")}
    try:
        exec(source, namespace)
        route_form = namespace[route_name]
        library_cls = type(
            class_name,
            (Library,),
            {
                "__module__": __name__,
                "publisher": publisher,
                "resource_name": resource_name,
                "version": version,
                "__tc_derivative_route_name__": route_name,
                "__tc_derivative_params__": compiled.params,
                "__tc_derivative_shape_params__": compiled.shape_params or {},
                "__tc_derivative_results__": compiled.results,
                route_name: Route("POST", route_form, source=source),
            },
        )
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ArtifactError("invalid_manifest", str(exc)) from exc

    _validate_execution_library_class(
        library_cls, class_name, resource_name, publisher, version
    )
    return library_cls


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


def _validate_route_name(route_name: str) -> None:
    if not route_name.isidentifier() or keyword.iskeyword(route_name):
        raise ArtifactError(
            "invalid_manifest",
            f"execution route name must be a valid Python identifier, got {route_name!r}",
        )


def _validate_execution_params(params: tuple[str, ...]) -> None:
    if not params:
        raise ArtifactError("invalid_manifest", "execution route must have at least one parameter")
    for param in params:
        if not param.isidentifier() or keyword.iskeyword(param):
            raise ArtifactError(
                "invalid_manifest",
                "derivative execution route parameters must be valid Python identifiers: "
                f"{param!r}",
            )


def _execution_route_source(route_name: str, params: tuple[str, ...]) -> str:
    joined_params = ", ".join(params)
    return (
        f"def {route_name}(self, cxt, {joined_params}):\n"
        "    return __opdef\n"
    )


def _validate_execution_library_identity(
    execution_resource_name: str,
    artifact_resource_name: str,
) -> None:
    if execution_resource_name == artifact_resource_name:
        raise ArtifactError(
            "invalid_manifest",
            "derivative execution library identity must not collide with artifact library identity",
        )


def _validate_execution_library_class(
    library_cls: type[Library],
    class_name: str,
    resource_name: str,
    publisher: str,
    version: str,
) -> None:
    if library_cls.__name__ != class_name:
        raise ArtifactError("invalid_manifest", "execution library class name mismatch")
    if getattr(library_cls, "resource_name", None) != resource_name:
        raise ArtifactError("invalid_manifest", "execution library resource_name mismatch")
    if getattr(library_cls, "publisher", None) != publisher:
        raise ArtifactError("invalid_manifest", "execution library publisher mismatch")
    if getattr(library_cls, "version", None) != version:
        raise ArtifactError("invalid_manifest", "execution library version mismatch")
    try:
        library_cls.class_id()
    except (TypeError, ValueError) as exc:
        raise ArtifactError("invalid_manifest", str(exc)) from exc


def _validate_artifact_library_identity(
    library_cls: type[Library],
    manifest: DerivativeArtifactManifest,
) -> None:
    identity = public_artifact_identity(manifest)
    library_resource_name = getattr(library_cls, "resource_name", None)
    if identity.name != library_resource_name:
        raise ArtifactError(
            "invalid_manifest",
            "artifact manifest public identity must match Library.resource_name",
        )
    try:
        class_path = library_cls.class_id().path
    except (TypeError, ValueError) as exc:
        raise ArtifactError("invalid_manifest", str(exc)) from exc
    if identity.to_uri().path != class_path:
        raise ArtifactError(
            "invalid_manifest",
            "artifact manifest public identity must match Library.class_id().path",
        )


def _program_payload(program: object) -> dict[str, object]:
    to_dict = getattr(program, "to_dict", None)
    if not callable(to_dict):
        raise ArtifactError(
            "invalid_manifest",
            "program must expose to_dict() for artifact payload construction",
        )
    payload = to_dict()
    if not isinstance(payload, dict):
        raise ArtifactError(
            "invalid_manifest",
            "program.to_dict() must return a dict payload",
        )
    return payload


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


def _artifact_resource_name(artifact_name: str) -> str:
    try:
        return _python_name_to_resource(artifact_name)
    except TypeError as exc:
        raise ArtifactError("invalid_manifest", str(exc)) from exc


def _comparison_digest(label: str, value: str | None) -> str:
    if value is None:
        raise ArtifactError(
            "invalid_manifest",
            f"{label} must include artifact_digest for comparison",
        )
    _require_non_empty("artifact_digest", value)
    return value


def _validate_source_library_dependency(
    source_library: str | None,
    source_library_version: str | None,
) -> None:
    if source_library is None and source_library_version is None:
        return
    if source_library is None:
        raise ArtifactError(
            "invalid_manifest",
            "source_library is required when source_library_version is supplied",
        )
    if source_library_version is None:
        raise ArtifactError(
            "invalid_manifest",
            "source_library_version is required when source_library is supplied",
        )


def _validate_library_dependency_path(dependency: URI, expected_version: str) -> None:
    parts = dependency.path.strip("/").split("/")
    if len(parts) != 4 or parts[0] != "lib":
        raise ArtifactError(
            "invalid_manifest",
            "source_library dependency path must have /lib/{publisher}/{name}/{version} form",
        )
    if parts[3] != expected_version:
        raise ArtifactError(
            "invalid_manifest",
            "source_library dependency version must match source_library_version",
        )
