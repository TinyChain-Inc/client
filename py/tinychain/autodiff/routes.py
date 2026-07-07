from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..graph_reflection import TypeSpec
from ..library import Library
from ..uri import uri
from ..serialize import serialize
from .protocol import AutodiffError


ROUTE_DERIVATIVE_SOURCE_ARTIFACT = "artifact"
ROUTE_DERIVATIVE_SOURCE_UNSUPPORTED = "unsupported"
ROUTE_DERIVATIVE_SOURCE_NON_DIFFERENTIABLE = "non_differentiable"
ROUTE_DERIVATIVE_SOURCE_KINDS: tuple[str, ...] = (
    ROUTE_DERIVATIVE_SOURCE_ARTIFACT,
    ROUTE_DERIVATIVE_SOURCE_UNSUPPORTED,
    ROUTE_DERIVATIVE_SOURCE_NON_DIFFERENTIABLE,
)

ROUTE_DERIVATIVE_COMPATIBILITY_NOT_VALIDATED = "not_validated"
ROUTE_DERIVATIVE_COMPATIBILITY_COMPATIBLE = "compatible"
ROUTE_DERIVATIVE_COMPATIBILITY_INCOMPATIBLE = "incompatible"
ROUTE_DERIVATIVE_COMPATIBILITY_UNSUPPORTED = "unsupported"
ROUTE_DERIVATIVE_COMPATIBILITY_STATUSES: tuple[str, ...] = (
    ROUTE_DERIVATIVE_COMPATIBILITY_NOT_VALIDATED,
    ROUTE_DERIVATIVE_COMPATIBILITY_COMPATIBLE,
    ROUTE_DERIVATIVE_COMPATIBILITY_INCOMPATIBLE,
    ROUTE_DERIVATIVE_COMPATIBILITY_UNSUPPORTED,
)


@dataclass(frozen=True, slots=True)
class RouteDerivativeIdentity:
    publisher: str
    library_name: str
    library_version: str
    library_path: str
    library_uri: str
    route_name: str
    route_path: str
    route_uri: str
    http_method: str

    def to_dict(self) -> dict[str, object]:
        return serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RouteDerivativeIdentity:
        return cls(
            publisher=str(data["publisher"]),
            library_name=str(data["library_name"]),
            library_version=str(data["library_version"]),
            library_path=str(data["library_path"]),
            library_uri=str(data["library_uri"]),
            route_name=str(data["route_name"]),
            route_path=str(data["route_path"]),
            route_uri=str(data["route_uri"]),
            http_method=str(data["http_method"]),
        )


@dataclass(frozen=True, slots=True)
class RouteDerivativeMetadata:
    source_kind: str
    is_pure: bool
    is_differentiable: bool
    input_signature: tuple[TypeSpec, ...]
    output_signature: tuple[TypeSpec, ...]
    supported_wrt: tuple[str, ...]
    seed_contract: str
    transform_version: str
    tensor_op_contract_version: str

    def __post_init__(self) -> None:
        _require_allowed("source_kind", self.source_kind, ROUTE_DERIVATIVE_SOURCE_KINDS)
        _require_bool("is_pure", self.is_pure)
        _require_bool("is_differentiable", self.is_differentiable)
        object.__setattr__(
            self, "input_signature", _typespec_tuple("input_signature", self.input_signature)
        )
        object.__setattr__(
            self, "output_signature", _typespec_tuple("output_signature", self.output_signature)
        )
        object.__setattr__(
            self, "supported_wrt", _string_tuple("supported_wrt", self.supported_wrt)
        )
        _require_non_empty("seed_contract", self.seed_contract)
        _require_non_empty("transform_version", self.transform_version)
        _require_non_empty("tensor_op_contract_version", self.tensor_op_contract_version)
        if self.is_differentiable and not self.supported_wrt:
            raise AutodiffError(
                "non_differentiable_route",
                "differentiable route metadata must declare at least one supported wrt value",
            )
        if self.source_kind in (
            ROUTE_DERIVATIVE_SOURCE_UNSUPPORTED,
            ROUTE_DERIVATIVE_SOURCE_NON_DIFFERENTIABLE,
        ) and self.is_differentiable:
            raise AutodiffError(
                "non_differentiable_route",
                f"{self.source_kind!r} route metadata cannot be marked differentiable",
            )
        _validate_json_compatible(self.to_dict(), "RouteDerivativeMetadata")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_kind": self.source_kind,
            "is_pure": self.is_pure,
            "is_differentiable": self.is_differentiable,
            "input_signature": [type_spec.to_dict() for type_spec in self.input_signature],
            "output_signature": [type_spec.to_dict() for type_spec in self.output_signature],
            "supported_wrt": list(self.supported_wrt),
            "seed_contract": self.seed_contract,
            "transform_version": self.transform_version,
            "tensor_op_contract_version": self.tensor_op_contract_version,
        }
        _validate_json_compatible(payload, "RouteDerivativeMetadata")
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RouteDerivativeMetadata:
        return cls(
            source_kind=_required_string(data, "source_kind"),
            is_pure=_required_bool(data, "is_pure"),
            is_differentiable=_required_bool(data, "is_differentiable"),
            input_signature=_typespec_tuple_from_dicts(data, "input_signature"),
            output_signature=_typespec_tuple_from_dicts(data, "output_signature"),
            supported_wrt=_string_tuple_from_mapping(data, "supported_wrt"),
            seed_contract=_required_string(data, "seed_contract"),
            transform_version=_required_string(data, "transform_version"),
            tensor_op_contract_version=_required_string(data, "tensor_op_contract_version"),
        )


@dataclass(frozen=True, slots=True)
class RouteDerivativePlan:
    route_identity: RouteDerivativeIdentity
    requested_wrt: tuple[str, ...]
    seed_contract: str
    source_kind: str
    compatibility_status: str
    artifact_uri: str | None = None
    artifact_digest: str | None = None
    artifact_visibility: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.route_identity, RouteDerivativeIdentity):
            raise AutodiffError(
                "non_differentiable_route",
                "route_identity must be a RouteDerivativeIdentity",
            )
        object.__setattr__(
            self, "requested_wrt", _string_tuple("requested_wrt", self.requested_wrt)
        )
        if not self.requested_wrt:
            raise AutodiffError(
                "non_differentiable_route",
                "route derivative plan must request at least one wrt value",
            )
        _require_non_empty("seed_contract", self.seed_contract)
        _require_allowed("source_kind", self.source_kind, ROUTE_DERIVATIVE_SOURCE_KINDS)
        _require_allowed(
            "compatibility_status",
            self.compatibility_status,
            ROUTE_DERIVATIVE_COMPATIBILITY_STATUSES,
        )
        _require_optional_string("artifact_uri", self.artifact_uri)
        _require_optional_string("artifact_digest", self.artifact_digest)
        _require_optional_string("artifact_visibility", self.artifact_visibility)
        _validate_json_compatible(self.to_dict(), "RouteDerivativePlan")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "route_identity": self.route_identity.to_dict(),
            "requested_wrt": list(self.requested_wrt),
            "seed_contract": self.seed_contract,
            "source_kind": self.source_kind,
            "compatibility_status": self.compatibility_status,
            "artifact_uri": self.artifact_uri,
            "artifact_digest": self.artifact_digest,
            "artifact_visibility": self.artifact_visibility,
        }
        _validate_json_compatible(payload, "RouteDerivativePlan")
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RouteDerivativePlan:
        identity_data = data["route_identity"]
        if not isinstance(identity_data, dict):
            raise AutodiffError(
                "non_differentiable_route",
                "route_identity must be a JSON object",
            )
        return cls(
            route_identity=RouteDerivativeIdentity.from_dict(identity_data),
            requested_wrt=_string_tuple_from_mapping(data, "requested_wrt"),
            seed_contract=_required_string(data, "seed_contract"),
            source_kind=_required_string(data, "source_kind"),
            compatibility_status=_required_string(data, "compatibility_status"),
            artifact_uri=_optional_string(data.get("artifact_uri")),
            artifact_digest=_optional_string(data.get("artifact_digest")),
            artifact_visibility=_optional_string(data.get("artifact_visibility")),
        )


def extract_route_identity(target: object) -> RouteDerivativeIdentity:
    """Return deterministic identity metadata for a bound TinyChain route target.

    This helper inspects descriptor metadata attached by ``Route.__get__``. It
    must not call the target, compile the library, install routes, or dispatch
    any operation.
    """
    if not callable(target):
        raise TypeError("expected a bound TinyChain route target")

    route = getattr(target, "__tc_route__", None)
    route_instance = getattr(target, "__tc_instance__", None)
    if route is None or route_instance is None:
        raise TypeError(
            "tc.grad is a call-site transform and requires a bound TinyChain route target"
        )

    if not isinstance(route_instance, Library):
        raise TypeError("bound TinyChain route target must belong to a Library instance")

    route_name = getattr(route, "name", None)
    http_method = getattr(route, "method", None)
    if not isinstance(route_name, str) or not route_name:
        raise AutodiffError(
            "non_differentiable_route",
            "malformed TinyChain route target: missing route name",
        )
    if not isinstance(http_method, str) or not http_method:
        raise AutodiffError(
            "non_differentiable_route",
            "malformed TinyChain route target: missing HTTP method",
        )

    _validate_library_identity_fields(route_instance)

    library_path = route_instance.id().path
    return RouteDerivativeIdentity(
        publisher=route_instance.publisher,
        library_name=route_instance.name,
        library_version=route_instance.version,
        library_path=library_path,
        library_uri=route_instance.link().absolute(),
        route_name=route_name,
        route_path=f"/{route_name}",
        route_uri=uri(route_instance.link(), "path", route_name).absolute(),
        http_method=http_method.upper(),
    )


def _validate_library_identity_fields(route_instance: Library) -> None:
    for field_name in ("publisher", "name", "version"):
        value: Any = getattr(route_instance, field_name, None)
        if not isinstance(value, str) or not value:
            raise TypeError(
                "bound TinyChain route target has malformed Library identity metadata"
            )


def _require_allowed(label: str, value: object, allowed: tuple[str, ...]) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise AutodiffError(
            "non_differentiable_route",
            f"{label} must be one of {allowed!r}",
        )


def _require_bool(label: str, value: object) -> None:
    if not isinstance(value, bool):
        raise AutodiffError(
            "non_differentiable_route",
            f"{label} must be a bool",
        )


def _require_non_empty(label: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise AutodiffError(
            "non_differentiable_route",
            f"{label} must be a non-empty string",
        )


def _require_optional_string(label: str, value: object | None) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise AutodiffError(
            "non_differentiable_route",
            f"{label} must be None or a non-empty string",
        )


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data[key]
    _require_non_empty(key, value)
    return value


def _required_bool(data: Mapping[str, object], key: str) -> bool:
    value = data[key]
    _require_bool(key, value)
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AutodiffError(
            "non_differentiable_route",
            "optional artifact fields must be None or non-empty strings",
        )
    return value


def _typespec_tuple(label: str, values: Iterable[TypeSpec]) -> tuple[TypeSpec, ...]:
    if isinstance(values, (str, bytes)):
        raise AutodiffError("non_differentiable_route", f"{label} must be a sequence")
    result = tuple(values)
    for type_spec in result:
        if not isinstance(type_spec, TypeSpec):
            raise AutodiffError(
                "non_differentiable_route",
                f"{label} entries must be TypeSpec instances",
            )
    return result


def _typespec_tuple_from_dicts(
    data: Mapping[str, object],
    key: str,
) -> tuple[TypeSpec, ...]:
    values = data[key]
    if not isinstance(values, list):
        raise AutodiffError("non_differentiable_route", f"{key} must be a list")
    result: list[TypeSpec] = []
    for value in values:
        if not isinstance(value, dict):
            raise AutodiffError(
                "non_differentiable_route",
                f"{key} entries must be JSON objects",
            )
        result.append(TypeSpec.from_dict(value))
    return tuple(result)


def _string_tuple(label: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise AutodiffError("non_differentiable_route", f"{label} must be a sequence")
    result = tuple(values)
    for value in result:
        _require_non_empty(label, value)
    return result


def _string_tuple_from_mapping(data: Mapping[str, object], key: str) -> tuple[str, ...]:
    values = data[key]
    if not isinstance(values, list):
        raise AutodiffError("non_differentiable_route", f"{key} must be a list")
    return _string_tuple(key, values)


def _validate_json_compatible(payload: object, label: str) -> None:
    try:
        json.dumps(payload, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise AutodiffError(
            "non_differentiable_route",
            f"{label} contains non-JSON-compatible fields: {exc}",
        ) from exc
