from __future__ import annotations

import pathlib
from collections.abc import Iterable

from .library import Library
from .uri import URI


def _token_parts(token: object | None) -> tuple[str | None, str | None, str | None]:
    if token is None:
        return None, None, None

    host = getattr(token, "host", None)
    actor = getattr(token, "actor_id", None)
    pub = getattr(token, "public_key_b64", None)
    if host is None and actor is None and pub is None:
        raise TypeError(
            "expected `token` with `host`, `actor_id`, and `public_key_b64` attributes"
        )

    return host, actor, pub


def _as_uri(value: object) -> URI:
    if isinstance(value, URI):
        return value
    if isinstance(value, str):
        return URI.parse(value)
    if hasattr(value, "link") and callable(getattr(value, "link")):
        linked = value.link()
        if isinstance(linked, URI):
            return linked
        return URI.parse(str(linked))
    if hasattr(value, "id") and callable(getattr(value, "id")):
        base = value.id()
        base_uri = base if isinstance(base, URI) else URI.parse(str(base))
        authority = getattr(value, "authority", None)
        if isinstance(authority, URI):
            return URI(
                path=base_uri.path,
                scheme=authority.scheme,
                host=authority.host,
                port=authority.port,
            )
        if isinstance(authority, str):
            authority_uri = URI.parse(authority)
            return URI(
                path=base_uri.path,
                scheme=authority_uri.scheme,
                host=authority_uri.host,
                port=authority_uri.port,
            )
        return base_uri
    raise TypeError(f"unsupported dependency type: {type(value).__name__}")


def _runtime_dependency_bindings(library: Library) -> list[object]:
    bindings: list[object] = []
    seen: set[int] = set()

    def collect(values: Iterable[object]) -> None:
        for value in values:
            marker = id(value)
            if marker in seen:
                continue
            seen.add(marker)
            bindings.append(value)

    library_cls = type(library)
    collect(vars(library_cls).values())

    try:
        collect(vars(library).values())
    except TypeError:
        pass

    return bindings


def _dependency_routes_for_library(library: Library) -> list[tuple[str, str]]:
    declared = list(getattr(library, "dependencies", ()) or ())
    if not declared:
        return []

    authorities_by_path: dict[str, set[str]] = {}
    for candidate in [*declared, *_runtime_dependency_bindings(library)]:
        try:
            parsed = _as_uri(candidate)
        except (TypeError, ValueError):
            continue
        authority = parsed.authority()
        if not parsed.path or authority is None:
            continue
        authorities_by_path.setdefault(parsed.path, set()).add(authority)

    routes: list[tuple[str, str]] = []
    for dep in declared:
        parsed = _as_uri(dep)
        if not parsed.path:
            raise ValueError(f"dependency route requires a canonical path, got: {dep!r}")

        authority = parsed.authority()
        if authority is None:
            candidates = authorities_by_path.get(parsed.path, set())
            if not candidates:
                raise ValueError(
                    "missing dependency authority for "
                    f"{parsed.path}; bind an authority-qualified dependency URI on the library "
                    "or bind a dependency instance with `authority` on the library class/instance"
                )
            if len(candidates) > 1:
                choices = ", ".join(sorted(candidates))
                raise ValueError(
                    "ambiguous dependency authority for "
                    f"{parsed.path}: {choices}; keep exactly one bound authority per dependency path"
                )
            authority = next(iter(candidates))

        route = (parsed.path, authority)
        if route not in routes:
            routes.append(route)

    return routes


def _shared_path_prefix(paths: list[str]) -> str:
    if not paths:
        raise ValueError("expected at least one dependency path")
    if len(paths) == 1:
        return paths[0]

    split = [path.strip("/").split("/") for path in paths]
    shared = split[0]
    for segments in split[1:]:
        limit = min(len(shared), len(segments))
        idx = 0
        while idx < limit and shared[idx] == segments[idx]:
            idx += 1
        shared = shared[:idx]
        if not shared:
            break

    if not shared:
        return "/"

    return "/" + "/".join(shared)


def with_library(
    library: Library,
    *,
    data_dir: pathlib.Path,
    token: object | None = None,
    token_host: str | None = None,
    actor_id: str | None = None,
    public_key_b64: str | None = None,
) -> "object":
    """
    Create a local kernel handle configured to route declared library dependencies by authority.

    Routes are inferred from `library.dependencies` plus any authority-qualified runtime bindings
    on the library instance/class.
    """
    import tinychain as tc

    if not hasattr(tc, "KernelHandle"):
        raise ImportError("`tc.kernel.with_library` requires the optional `tinychain-local` backend")

    routes = _dependency_routes_for_library(library)

    if not routes:
        raise ValueError(
            "expected at least one dependency with an `authority` to configure egress routing"
        )

    token_host_from_token, actor_id_from_token, public_key_b64_from_token = _token_parts(token)

    token_host = token_host or token_host_from_token
    actor_id = actor_id or actor_id_from_token
    public_key_b64 = public_key_b64 or public_key_b64_from_token

    call_kwargs = dict(
        token_host=None,
        actor_id=None,
        public_key_b64=None,
        data_dir=str(data_dir),
    )
    if token_host and actor_id and public_key_b64:
        call_kwargs.update(
            token_host=token_host,
            actor_id=actor_id,
            public_key_b64=public_key_b64,
        )

    multi_route_ctor = getattr(tc.KernelHandle, "local_with_dependency_routes", None)
    if callable(multi_route_ctor) and len(routes) > 1:
        try:
            return multi_route_ctor(routes, **call_kwargs)
        except TypeError:
            return multi_route_ctor(dependency_routes=routes, **call_kwargs)

    route_root, route_authority = routes[0]
    if len(routes) > 1:
        authorities = {authority for _, authority in routes}
        if len(authorities) != 1:
            raise ValueError(
                "multiple dependency authorities require a backend with "
                "`KernelHandle.local_with_dependency_routes` support"
            )

        route_root = _shared_path_prefix([path for path, _ in routes])
        route_authority = next(iter(authorities))

    local_with_route = getattr(tc.KernelHandle, "local_with_dependency_route", None)
    if callable(local_with_route):
        return local_with_route(
            route_root,
            route_authority,
            **call_kwargs,
        )

    raise RuntimeError(
        "tinychain-local backend does not support dependency route configuration; "
        "expected `KernelHandle.local_with_dependency_route` or "
        "`KernelHandle.local_with_dependency_routes`"
    )
