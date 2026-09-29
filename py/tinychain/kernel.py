from __future__ import annotations

import pathlib

from .library import Library, install


def _token_parts(token: object) -> tuple[str, str, str]:
    host = getattr(token, "host", None)
    actor = getattr(token, "actor_id", None)
    pub = getattr(token, "public_key_b64", None)
    if host is None and actor is None and pub is None:
        raise TypeError(
            "expected `token` with `host`, `actor_id`, and `public_key_b64` attributes"
        )

    return host, actor, pub


def with_library(
    library: Library,
    *,
    data_dir: pathlib.Path,
    workspace: pathlib.Path | None = None,
    token: object,
) -> "object":
    """
    Create a local kernel and install a Library or Service definition.

    Dependency authorities are part of the references compiled into that
    definition; no adapter-local routing table is constructed.
    """
    from . import _local

    _token_parts(token)

    if workspace is None:
        workspace = data_dir.with_name(f"{data_dir.name}-workspace")

    kernel = _local.kernel_handle().local(
        data_dir=str(data_dir), workspace=str(workspace), token=token
    )
    install(library, kernel=kernel, token=token)
    return kernel
