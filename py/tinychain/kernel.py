from __future__ import annotations

import pathlib
from typing import Optional

from .library import Library
from .uri import URI


def for_library(
    library: Library,
    *,
    data_dir: pathlib.Path,
    dependency: Optional[URI] = None,
) -> "object":
    import tinychain as tc

    if not hasattr(tc, "KernelHandle"):
        raise ImportError("`tc.kernel.for_library` requires the optional `tinychain-local` backend")

    dep = dependency or next((d for d in library.dependencies if d.host is not None), None)

    if dep is None or dep.host is None:
        raise ValueError(
            "expected at least one dependency with an `authority` to configure egress routing"
        )

    return tc.KernelHandle.with_library_schema_and_dependency_route(
        library.schema_json(),
        dep.path,
        dep.authority(),
        data_dir=str(data_dir),
    )
