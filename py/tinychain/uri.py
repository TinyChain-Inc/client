from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

@dataclass(frozen=True, slots=True, init=False)
class URI:
    """
    A canonical TinyChain path with an optional authority.

    - Canonical identity is always the path (e.g. `/lib/...`).
    - Authority (scheme/host/port) is deployment configuration used for routing remote dependencies.
    """

    path: str
    scheme: str
    host: Optional[str]
    port: Optional[int]

    def __init__(
        self,
        subject: object | None = None,
        *parts: str,
        path: str | "URI" | None = None,
        scheme: str | None = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        if path is not None and (subject is not None or parts):
            raise TypeError("URI accepts either path=... or subject/parts, not both")

        inherited_scheme: str | None = None
        inherited_host: str | None = None
        inherited_port: int | None = None

        if path is None:
            if subject is None:
                resolved_path = ""
            elif isinstance(subject, URI):
                resolved = subject.child(*parts)
                resolved_path = resolved.path
                inherited_scheme = resolved.scheme
                inherited_host = resolved.host
                inherited_port = resolved.port
            elif isinstance(subject, str) and (subject.startswith("/") or subject.startswith("$") or "://" in subject):
                resolved = URI.parse(subject).child(*parts)
                resolved_path = resolved.path
                inherited_scheme = resolved.scheme
                inherited_host = resolved.host
                inherited_port = resolved.port
            else:
                resolved_path = _path_from_subject(subject, *parts)
        else:
            if isinstance(path, URI):
                resolved_path = path.path
                inherited_scheme = path.scheme
                inherited_host = path.host
                inherited_port = path.port
            elif isinstance(path, str) and "://" in path:
                parsed = URI.parse(path)
                resolved_path = parsed.path
                inherited_scheme = parsed.scheme
                inherited_host = parsed.host
                inherited_port = parsed.port
            else:
                resolved_path = path

        resolved_scheme = inherited_scheme if scheme is None else str(scheme)
        if resolved_scheme is None:
            resolved_scheme = "http"
        resolved_host = inherited_host if host is None else host
        resolved_port = inherited_port if port is None else port

        object.__setattr__(self, "path", resolved_path)
        object.__setattr__(self, "scheme", resolved_scheme)
        object.__setattr__(self, "host", resolved_host)
        object.__setattr__(self, "port", resolved_port)
        self.__post_init__()

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

    def child(self, *parts: str) -> "URI":
        if not parts:
            return self

        segments = _path_segments(parts)
        base_path = self.path.rstrip("/")
        new_path = _join_path([base_path.lstrip("/")] + segments) if base_path else _join_path(segments)
        return URI(path=new_path, scheme=self.scheme, host=self.host, port=self.port)

    def with_authority(self, authority: "URI | str") -> "URI":
        parsed = authority if isinstance(authority, URI) else URI.parse(authority)
        if parsed.host is None:
            raise ValueError("authority URI must include a host")
        return URI(path=self.path, scheme=parsed.scheme, host=parsed.host, port=parsed.port)


def _segment(label: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if "/" in value:
        raise ValueError(f"{label} must not contain '/'")
    if value in (".", ".."):
        raise ValueError(f"{label} must not be '.' or '..'")
    return value


# Canonical grammar for ``Library.resource_name``: lowercase ASCII alphanumeric
# components joined by single ``-`` or ``_`` separators (e.g. ``ilc``,
# ``ilc-client``, ``ordinary_client``, ``v2``, ``library2-client``).
_RESOURCE_NAME_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def validate_resource_name(value: object) -> str:
    """Validate and return a canonical ``Library.resource_name``.

    ``resource_name`` is the sole source of truth for the library name path
    component. This is the single validation path for that field.
    """
    if not isinstance(value, str) or not value:
        raise ValueError("resource_name must be a non-empty string")
    if _RESOURCE_NAME_RE.fullmatch(value) is None:
        raise ValueError(
            "resource_name must match [a-z0-9]+(?:[-_][a-z0-9]+)* "
            "(lowercase alphanumerics with single '-' or '_' separators): "
            f"{value!r}"
        )
    return value


def _python_name_to_resource(name: str) -> str:
    if not name:
        raise TypeError("expected a non-empty class name")
    first = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    resource = re.sub("([a-z0-9])([A-Z])", r"\1_\2", first).replace("-", "_").lower()
    if resource.startswith("_") or resource.endswith("_") or "__" in resource:
        raise TypeError(f"cannot derive a canonical resource name from class name {name!r}")
    return resource


def _path_segments(path: Optional[Iterable[str]]) -> list[str]:
    if path is None:
        return []
    segments: list[str] = []
    for part in path:
        segments.append(_segment("path", part))
    return segments


def _join_path(segments: Iterable[str]) -> str:
    joined = "/".join(segments)
    return f"/{joined}" if joined else "/"


def _path_from_subject(subject: object, *parts: str) -> str:
    if isinstance(subject, str):
        if subject.startswith("/") or subject.startswith("$") or "://" in subject:
            base = URI.parse(subject)
            return base.child(*parts).path if parts else base.path

        segments = [_segment("path", subject), *_path_segments(parts)]
        return _join_path(segments)

    resolved = uri(subject)
    if isinstance(resolved, URI):
        return resolved.child(*parts).path if parts else resolved.path

    if parts:
        raise TypeError("cannot append path segments to a Scalar URI")

    raw: object = resolved
    if hasattr(raw, "to_json"):
        raw = raw.to_json()
    return URI.parse(str(raw)).path


def uri(subject: object) -> URI | "Scalar":
    if isinstance(subject, URI):
        return subject

    if hasattr(subject, "__uri__"):
        base_value = getattr(subject, "__uri__")
        if isinstance(base_value, URI):
            return base_value
        if isinstance(base_value, str):
            return URI.parse(base_value)
        raise TypeError(f"expected __uri__ to be URI or str, got {type(base_value).__name__}")

    if hasattr(subject, "class_") and callable(getattr(subject, "class_")):
        return subject.class_()

    if hasattr(subject, "id") and callable(getattr(subject, "id")):
        base_value = subject.id()
        return base_value if isinstance(base_value, URI) else URI.parse(str(base_value))

    if isinstance(subject, str):
        if subject.startswith("/") or subject.startswith("$") or "://" in subject:
            return URI.parse(subject)
        raise TypeError("uri(...) is an accessor; construct via URI(...)")

    raise TypeError(f"unsupported URI subject: {type(subject).__name__}")


def authority(value: str | URI) -> str:
    if isinstance(value, URI):
        parsed = value
    else:
        text = value.strip()
        if not text or text.startswith("/"):
            raise ValueError(f"expected host[:port] or absolute URI, got: {value}")
        parsed = URI.parse(text)

    out = parsed.authority()
    if out is None:
        raise ValueError(f"expected URI with authority, got: {value}")
    return out


def origin(value: str | URI) -> str:
    if isinstance(value, URI):
        parsed = value
    else:
        text = value.strip()
        if not text or text.startswith("/"):
            raise ValueError(f"expected host[:port] or absolute URI, got: {value}")
        parsed = URI.parse(text)

    out = parsed.authority()
    if out is None:
        raise ValueError(f"expected URI with authority, got: {value}")
    return f"{parsed.scheme}://{out}"
