from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..graph_reflection import TypeSpec
from ..library import Library
from ..uri import uri
from ..serialize import serialize
from .protocol import AutodiffError
from .seed import FLOAT_DTYPES, SeedValidator


TENSOR_TYPESPEC_CLASS_URI = "/state/collection/tensor"

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

ROUTE_DERIVATIVE_METADATA_FIELD = "derivative_routes"


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
    artifact_uri: str | None = None
    artifact_digest: str | None = None
    artifact_source_library: str | None = None
    artifact_source_library_version: str | None = None
    artifact_source_route: str | None = None
    artifact_visibility: str | None = None

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
        _require_optional_string("artifact_uri", self.artifact_uri)
        _require_optional_string("artifact_digest", self.artifact_digest)
        _require_optional_string("artifact_source_library", self.artifact_source_library)
        _require_optional_string(
            "artifact_source_library_version", self.artifact_source_library_version
        )
        _require_optional_string("artifact_source_route", self.artifact_source_route)
        _require_optional_string("artifact_visibility", self.artifact_visibility)
        if self.source_kind == ROUTE_DERIVATIVE_SOURCE_ARTIFACT:
            if self.artifact_uri is None:
                raise AutodiffError(
                    "non_differentiable_route",
                    "artifact-backed route metadata must declare artifact_uri",
                )
            if self.artifact_digest is None:
                raise AutodiffError(
                    "non_differentiable_route",
                    "artifact-backed route metadata must declare artifact_digest",
                )
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
            "artifact_uri": self.artifact_uri,
            "artifact_digest": self.artifact_digest,
            "artifact_source_library": self.artifact_source_library,
            "artifact_source_library_version": self.artifact_source_library_version,
            "artifact_source_route": self.artifact_source_route,
            "artifact_visibility": self.artifact_visibility,
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
            artifact_uri=_optional_string(data.get("artifact_uri")),
            artifact_digest=_optional_string(data.get("artifact_digest")),
            artifact_source_library=_optional_string(data.get("artifact_source_library")),
            artifact_source_library_version=_optional_string(
                data.get("artifact_source_library_version")
            ),
            artifact_source_route=_optional_string(data.get("artifact_source_route")),
            artifact_visibility=_optional_string(data.get("artifact_visibility")),
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
    route, route_instance = _extract_bound_route_parts(target)

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


def lookup_route_derivative_metadata(target: object) -> RouteDerivativeMetadata:
    """Return class-level derivative metadata for a bound route target.

    Library subclasses declare metadata as ``derivative_routes`` keyed by route
    name (``"create"``) or route path (``"/create"``). This helper performs
    local metadata inspection only; it never executes or compiles the route.
    """
    _, route_instance = _extract_bound_route_parts(target)
    identity = extract_route_identity(target)
    metadata_by_key = getattr(type(route_instance), ROUTE_DERIVATIVE_METADATA_FIELD, None)
    if metadata_by_key is None:
        raise _missing_route_metadata(identity)
    if not isinstance(metadata_by_key, Mapping):
        raise AutodiffError(
            "non_differentiable_route",
            f"{ROUTE_DERIVATIVE_METADATA_FIELD} must be a mapping",
        )

    candidate_keys = (identity.route_name, identity.route_path)
    matched_entries = [
        (key, metadata_by_key[key]) for key in candidate_keys if key in metadata_by_key
    ]
    if not matched_entries:
        raise _missing_route_metadata(identity)

    normalized_entries = [
        (key, _normalize_route_derivative_metadata(value, key))
        for key, value in matched_entries
    ]
    first_key, first_metadata = normalized_entries[0]
    for duplicate_key, duplicate_metadata in normalized_entries[1:]:
        if duplicate_metadata != first_metadata:
            raise AutodiffError(
                "non_differentiable_route",
                "ambiguous derivative metadata for "
                f"{identity.route_uri}: keys {first_key!r} and {duplicate_key!r} disagree",
            )
    return first_metadata


def discover_route_derivative(
    target: object,
    *,
    wrt: object,
    seed: str = "seed",
    seed_typespec: TypeSpec | dict[str, object] | None = None,
) -> RouteDerivativePlan:
    """Validate route derivative metadata and return a local discovery plan.

    Discovery is metadata-only: it never executes route bodies, installs
    derivative artifacts, fetches remote state, or calls a backend.
    """
    identity = extract_route_identity(target)
    metadata = lookup_route_derivative_metadata(target)
    requested_wrt = _normalize_requested_wrt(wrt)
    _validate_route_derivative_metadata(
        identity=identity,
        metadata=metadata,
        requested_wrt=requested_wrt,
        seed=seed,
        seed_typespec=seed_typespec,
    )
    _validate_artifact_backed_route_metadata(identity, metadata)
    return RouteDerivativePlan(
        route_identity=identity,
        requested_wrt=requested_wrt,
        seed_contract=metadata.seed_contract,
        source_kind=metadata.source_kind,
        compatibility_status=ROUTE_DERIVATIVE_COMPATIBILITY_COMPATIBLE,
        artifact_uri=metadata.artifact_uri,
        artifact_digest=metadata.artifact_digest,
        artifact_visibility=metadata.artifact_visibility,
    )


def _extract_bound_route_parts(target: object) -> tuple[object, Library]:
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
    return route, route_instance


def _normalize_requested_wrt(wrt: object) -> tuple[str, ...]:
    if wrt is None:
        raise TypeError("discover_route_derivative requires `wrt` value ids")
    if isinstance(wrt, str):
        values = (wrt,)
    elif isinstance(wrt, (list, tuple)):
        values = tuple(wrt)
    else:
        raise TypeError(
            "discover_route_derivative `wrt` must be a value id string "
            "or sequence of value id strings"
        )
    if not values:
        raise TypeError("discover_route_derivative `wrt` must not be empty")
    for value in values:
        if not isinstance(value, str) or not value:
            raise TypeError(
                "discover_route_derivative `wrt` entries must be non-empty value id strings"
            )
    return values


def _validate_route_derivative_metadata(
    *,
    identity: RouteDerivativeIdentity,
    metadata: RouteDerivativeMetadata,
    requested_wrt: tuple[str, ...],
    seed: str,
    seed_typespec: TypeSpec | dict[str, object] | None,
) -> None:
    if not metadata.is_pure:
        raise AutodiffError(
            "side_effecting_route_unsupported",
            f"route derivative discovery requires explicit pure metadata for {identity.route_uri}",
        )
    if not metadata.is_differentiable:
        raise AutodiffError(
            "missing_derivative_behavior",
            f"route {identity.route_uri} does not declare differentiable behavior",
        )
    if metadata.source_kind != ROUTE_DERIVATIVE_SOURCE_ARTIFACT:
        raise AutodiffError(
            "missing_derivative_behavior",
            f"route {identity.route_uri} has unsupported derivative source kind {metadata.source_kind!r}",
        )

    supported_wrt = frozenset(metadata.supported_wrt)
    invalid_wrt = [value for value in requested_wrt if value not in supported_wrt]
    if invalid_wrt:
        raise AutodiffError(
            "non_differentiable_route",
            f"route {identity.route_uri} does not support wrt values {invalid_wrt!r}",
        )

    if not isinstance(seed, str) or not seed:
        raise TypeError("discover_route_derivative `seed` must be a non-empty string")

    for type_spec in (*metadata.input_signature, *metadata.output_signature):
        _validate_tensor_type_spec(identity, type_spec)
    if not metadata.output_signature:
        raise AutodiffError(
            "missing_shape_metadata",
            f"route {identity.route_uri} derivative metadata must declare tensor outputs",
        )

    if seed_typespec is not None:
        output_type_spec = metadata.output_signature[0]
        SeedValidator().validate(
            seed_typespec=_typespec_params(_normalize_type_spec(seed_typespec, "seed_typespec")),
            output_typespec=_typespec_params(output_type_spec),
        )


def _validate_artifact_backed_route_metadata(
    identity: RouteDerivativeIdentity,
    metadata: RouteDerivativeMetadata,
) -> None:
    if metadata.source_kind != ROUTE_DERIVATIVE_SOURCE_ARTIFACT:
        return
    artifact_digest = metadata.artifact_digest
    if artifact_digest is None:
        raise AutodiffError(
            "non_differentiable_route",
            f"artifact-backed metadata for {identity.route_uri} must declare artifact_digest",
        )
    artifact_uri = metadata.artifact_uri
    if artifact_uri is None:
        raise AutodiffError(
            "non_differentiable_route",
            f"artifact-backed metadata for {identity.route_uri} must declare artifact_uri",
        )
    source_library = metadata.artifact_source_library
    if source_library is None:
        raise AutodiffError(
            "non_differentiable_route",
            f"artifact-backed metadata for {identity.route_uri} must declare source library",
        )
    accepted_libraries = {
        identity.library_uri,
        identity.library_path,
        f"{identity.publisher}/{identity.library_name}",
    }
    if source_library not in accepted_libraries:
        raise AutodiffError(
            "non_differentiable_route",
            "artifact source library does not match route library "
            f"{identity.library_uri}: {source_library!r}",
        )
    if metadata.artifact_source_library_version != identity.library_version:
        raise AutodiffError(
            "non_differentiable_route",
            "artifact source library version does not match route library version "
            f"{identity.library_version!r}",
        )
    source_route = metadata.artifact_source_route
    if source_route is None:
        raise AutodiffError(
            "non_differentiable_route",
            f"artifact-backed metadata for {identity.route_uri} must declare source route",
        )
    if source_route not in (identity.route_name, identity.route_path, identity.route_uri):
        raise AutodiffError(
            "non_differentiable_route",
            "artifact source route does not match route identity "
            f"{identity.route_uri}: {source_route!r}",
        )


def _validate_tensor_type_spec(
    identity: RouteDerivativeIdentity,
    type_spec: TypeSpec,
) -> None:
    if type_spec.class_uri != TENSOR_TYPESPEC_CLASS_URI:
        raise AutodiffError(
            "non_differentiable_route",
            f"route {identity.route_uri} derivative signatures must use tensor TypeSpec values",
        )
    params = _typespec_params(type_spec)
    if "dtype" not in params:
        raise AutodiffError(
            "missing_dtype_metadata",
            f"route {identity.route_uri} tensor signature is missing dtype metadata",
        )
    if "shape" not in params:
        raise AutodiffError(
            "missing_shape_metadata",
            f"route {identity.route_uri} tensor signature is missing shape metadata",
        )
    dtype = str(params["dtype"])
    if dtype not in FLOAT_DTYPES:
        raise AutodiffError(
            "dtype_not_differentiable",
            f"route {identity.route_uri} supports only {', '.join(FLOAT_DTYPES)} tensors",
        )
    try:
        _normalize_tensor_shape(params["shape"])
    except (TypeError, ValueError) as exc:
        raise AutodiffError(
            "missing_shape_metadata",
            f"route {identity.route_uri} tensor shape metadata must be a sequence",
        ) from exc


def _normalize_tensor_shape(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("tensor shape metadata must be a non-string sequence")
    shape: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("tensor shape dimensions must be integers")
        shape.append(dimension)
    return tuple(shape)


def _typespec_params(type_spec: TypeSpec) -> dict[str, object]:
    return dict(type_spec.params)


def _normalize_route_derivative_metadata(
    value: object,
    metadata_key: str,
) -> RouteDerivativeMetadata:
    if isinstance(value, RouteDerivativeMetadata):
        return value
    if isinstance(value, Mapping):
        try:
            return RouteDerivativeMetadata.from_dict(dict(value))
        except KeyError as exc:
            missing_key = exc.args[0]
            raise AutodiffError(
                "non_differentiable_route",
                f"malformed derivative metadata for {metadata_key!r}: missing {missing_key!r}",
            ) from exc
    raise AutodiffError(
        "non_differentiable_route",
        f"derivative metadata for {metadata_key!r} must be RouteDerivativeMetadata or mapping",
    )


def _missing_route_metadata(identity: RouteDerivativeIdentity) -> AutodiffError:
    return AutodiffError(
        "non_differentiable_route",
        "missing derivative metadata for route "
        f"{identity.route_uri}; declare {ROUTE_DERIVATIVE_METADATA_FIELD}[{identity.route_name!r}] "
        f"or {ROUTE_DERIVATIVE_METADATA_FIELD}[{identity.route_path!r}]",
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
    if key == "is_pure" and not isinstance(value, bool):
        raise AutodiffError(
            "side_effecting_route_unsupported",
            "route derivative discovery requires explicit pure metadata",
        )
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


def _typespec_tuple(
    label: str,
    values: Iterable[TypeSpec | Mapping[str, object]],
) -> tuple[TypeSpec, ...]:
    if isinstance(values, (str, bytes)):
        raise AutodiffError("non_differentiable_route", f"{label} must be a sequence")
    return tuple(_normalize_type_spec(value, label) for value in values)


def _normalize_type_spec(value: object, label: str) -> TypeSpec:
    if isinstance(value, TypeSpec):
        return value
    if isinstance(value, Mapping):
        try:
            return TypeSpec.from_dict(dict(value))
        except KeyError as exc:
            missing_key = exc.args[0]
            raise AutodiffError(
                "non_differentiable_route",
                f"{label} TypeSpec mapping is missing {missing_key!r}",
            ) from exc
    raise AutodiffError(
        "non_differentiable_route",
        f"{label} entries must be TypeSpec instances or TypeSpec mappings",
    )


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
