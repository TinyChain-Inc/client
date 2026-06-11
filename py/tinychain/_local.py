from __future__ import annotations

from typing import Any


def backend() -> Any:
    try:
        import tinychain_local as local  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "install `tinychain-local` to use the in-process TinyChain backend"
        ) from exc

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
