from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any


def is_available() -> bool:
    return (
        sys.modules.get("tinychain_local") is not None
        or importlib.util.find_spec("tinychain_local") is not None
    )


def backend() -> Any:
    existing = sys.modules.get("tinychain_local")
    if existing is not None:  # pragma: no cover
        return existing

    if not is_available():  # pragma: no cover
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


def backend_handle(*args: object, **kwargs: object) -> Any:
    return backend().Backend(*args, **kwargs)


def local_kernel(*, data_dir: str | None = None, token: object | None = None) -> Any:
    handle = kernel_handle()
    if data_dir is None:
        return handle.local(token=token)
    workspace = f"{data_dir}-workspace"
    return handle.local(data_dir=data_dir, workspace=workspace, token=token)
