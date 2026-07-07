from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..library import Library
from ..uri import uri
from ..serialize import serialize
from .protocol import AutodiffError


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
