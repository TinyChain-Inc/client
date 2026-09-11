from __future__ import annotations

import pathlib

from .library import Library, _LIB_ROOT_URI, _library_class, _submit_local_definition, library_definition


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
    Create a local kernel and install the Library's canonical literal definition.

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
    bearer = getattr(token, "bearer_token")
    from .classdef import _CLASS_ROOT_URI, class_definition

    for cls in (getattr(_library_class(library), "classes", ()) or ()):
        _submit_local_definition(
            kernel,
            _CLASS_ROOT_URI.path,
            class_definition(cls),
            bearer_token=bearer,
        )
    _submit_local_definition(
        kernel,
        _LIB_ROOT_URI.path,
        library_definition(library),
        bearer_token=bearer,
    )
    return kernel
