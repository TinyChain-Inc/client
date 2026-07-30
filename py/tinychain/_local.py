from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any


def backend() -> Any:
    existing = sys.modules.get("tinychain_local")
    if existing is not None:  # pragma: no cover
        return existing

    if importlib.util.find_spec("tinychain_local") is None:  # pragma: no cover
        raise ImportError(
            "install `tinychain-local` to use the in-process TinyChain backend"
        )

    local = importlib.import_module("tinychain_local")  # type: ignore
    return local


def kernel_handle() -> Any:
    return backend().KernelHandle


def kernel_request(*args: object) -> Any:
    return backend().KernelRequest(*args)


def state_handle(value: object) -> Any:
    return backend().StateHandle(value)


def backend_handle(*args: object) -> Any:
    return backend().Backend(*args)


def local_kernel(*, data_dir: str | None = None) -> Any:
    handle = kernel_handle()
    if data_dir is None:
        return handle.local()
    return handle.local(data_dir=data_dir)


def kernel_with_library_definition(
    definition_json: str,
    *,
    routes: object = None,
    token: object = None,
    data_dir: str | None = None,
) -> Any:
    kwargs = {"token": token, "data_dir": data_dir}
    if routes is not None:
        kwargs["routes"] = routes
    return kernel_handle().with_library_definition(definition_json, **kwargs)
