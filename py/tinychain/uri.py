from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True, slots=True)
class URI:
    """
    A canonical TinyChain path with an optional authority.

    - Canonical identity is always the path (e.g. `/lib/...`).
    - Authority (scheme/host/port) is deployment configuration used for routing remote dependencies.
    """

    path: str = ""
    scheme: str = "http"
    host: Optional[str] = None
    port: Optional[int] = None

    def __post_init__(self) -> None:
        if self.path and not self.path.startswith("/"):
            raise ValueError(f"URI.path must start with '/': {self.path}")
        if self.host is not None and not self.host:
            raise ValueError("URI.host must be non-empty when provided")
        if self.port is not None and (self.port <= 0 or self.port > 65535):
            raise ValueError(f"invalid port: {self.port}")

    @classmethod
    def parse(cls, value: str, *, default_scheme: str = "http") -> "URI":
        value = value.strip()
        if not value:
            raise ValueError("empty URI")

        # Canonical path-only form.
        if value.startswith("/"):
            return cls(path=value)

        scheme = default_scheme
        rest = value
        if "://" in value:
            scheme, rest = value.split("://", 1)

        # Accept either an authority-only string (`host[:port]`) or a full URI (`host[:port]/path`).
        if "/" in rest:
            authority_raw, path = rest.split("/", 1)
            path = f"/{path}" if path else ""
        else:
            authority_raw, path = rest, ""

        if not authority_raw:
            raise ValueError(f"invalid URI: {value}")

        if ":" in authority_raw:
            host, port_str = authority_raw.rsplit(":", 1)
            return cls(path=path, scheme=scheme, host=host, port=int(port_str))

        return cls(path=path, scheme=scheme, host=authority_raw, port=None)

    def canonical(self) -> str:
        return self.path

    def authority(self) -> Optional[str]:
        if self.host is None:
            return None
        return f"{self.host}:{self.port}" if self.port is not None else self.host

    def absolute(self) -> str:
        if self.host is None:
            return self.path
        base = f"{self.scheme}://{self.host}"
        if self.port is not None:
            base = f"{base}:{self.port}"
        return f"{base}{self.path}"

    def __str__(self) -> str:
        return self.absolute()


def healthz() -> str:
    return "/healthz"


def lib_root() -> str:
    return "/lib"


def service_root() -> str:
    return "/service"


def _segment(label: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if "/" in value:
        raise ValueError(f"{label} must not contain '/'")
    if value in (".", ".."):
        raise ValueError(f"{label} must not be '.' or '..'")
    return value


def _path_segments(path: Optional[Iterable[str]]) -> list[str]:
    if path is None:
        return []
    segments: list[str] = []
    for part in path:
        segments.append(_segment("path", part))
    return segments


def library(
    *,
    publisher: str,
    name: str,
    version: str,
    path: Optional[Iterable[str]] = None,
) -> URI:
    segments = [
        "lib",
        _segment("publisher", publisher),
        _segment("name", name),
        _segment("version", version),
        *_path_segments(path),
    ]
    return URI("/" + "/".join(segments))


def library_link(
    *,
    publisher: str,
    name: str,
    version: str,
    authority: Optional[URI] = None,
    path: Optional[Iterable[str]] = None,
) -> URI:
    base = library(publisher=publisher, name=name, version=version, path=path)
    if authority is None:
        return base
    return URI(path=base.path, scheme=authority.scheme, host=authority.host, port=authority.port)


def service(
    *,
    publisher: str,
    namespace: str,
    name: str,
    version: str,
    path: Optional[Iterable[str]] = None,
) -> str:
    segments = [
        "service",
        _segment("publisher", publisher),
        _segment("namespace", namespace),
        _segment("name", name),
        _segment("version", version),
        *_path_segments(path),
    ]
    return "/" + "/".join(segments)


def state(*, namespace: str, path: Optional[Iterable[str]] = None) -> str:
    segments = ["state", _segment("namespace", namespace), *_path_segments(path)]
    return "/" + "/".join(segments)


def media(*, path: Optional[Iterable[str]] = None) -> str:
    segments = ["state", "media", *_path_segments(path)]
    return "/" + "/".join(segments)
