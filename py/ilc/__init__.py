from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib
from typing import Union

import tinychain as tc

ILC_SERVER_PATH = "/lib/applied-physics/ilc"
ILC_CLIENT_PATH = "/lib/applied-physics/ilc-client"
DEFAULT_VERSION = "0.1.0"


def _parse_authority(server: Union[str, tc.URI]) -> tc.URI:
    uri = tc.URI.parse(server) if isinstance(server, str) else server
    if uri.host is None:
        raise ValueError("expected a server authority like 'host:port'")
    return uri


def _dependency_uri(server: tc.URI) -> tc.URI:
    return tc.URI(
        path=ILC_SERVER_PATH,
        scheme=server.scheme,
        host=server.host,
        port=server.port,
    )


@dataclass(frozen=True, slots=True)
class ILCLibrary:
    server: tc.URI
    version: str = DEFAULT_VERSION
    library_id: str = ILC_CLIENT_PATH

    def __init__(
        self,
        *,
        server: Union[str, tc.URI],
        version: str = DEFAULT_VERSION,
        library_id: str = ILC_CLIENT_PATH,
    ) -> None:
        object.__setattr__(self, "server", _parse_authority(server))
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "library_id", library_id)

    @property
    def dependency(self) -> tc.URI:
        return _dependency_uri(self.server)

    @property
    def dependencies(self) -> tuple[tc.URI, ...]:
        return (self.dependency,)

    def schema(self) -> dict:
        return {
            "id": self.library_id,
            "version": self.version,
            "dependencies": [ILC_SERVER_PATH],
        }

    def schema_json(self) -> str:
        return json.dumps(self.schema(), separators=(",", ":"))

    def route(self, *path: str) -> str:
        if not path:
            return self.library_id
        return "/".join((self.library_id.rstrip("/"), *path))

    def opref_add(self, body: object) -> tc.OpRef:
        return tc.OpRef("POST", self.route("cipher", "add"), body=body)

    def opref_mul(self, body: object) -> tc.OpRef:
        return tc.OpRef("POST", self.route("cipher", "mul"), body=body)

    def opref_encrypt(self, body: object) -> tc.OpRef:
        return tc.OpRef("POST", f"{ILC_SERVER_PATH}/crypto/encrypt", body=body)

    def opref_decrypt(self, body: object) -> tc.OpRef:
        return tc.OpRef("POST", f"{ILC_SERVER_PATH}/crypto/decrypt", body=body)


def kernel_for_local(
    data_dir: Union[str, pathlib.Path],
    *,
    server: Union[str, tc.URI],
    version: str = DEFAULT_VERSION,
    library_id: str = ILC_CLIENT_PATH,
) -> object:
    if not hasattr(tc, "KernelHandle"):
        raise ImportError("`ilc.kernel_for_local` requires the optional tinychain-local backend")

    lib = ILCLibrary(server=server, version=version, library_id=library_id)
    data_dir = pathlib.Path(data_dir)
    return tc.KernelHandle.with_library_schema_and_dependency_route(
        lib.schema_json(),
        ILC_SERVER_PATH,
        lib.dependency.authority(),
        data_dir=str(data_dir),
    )


def install_wasm(
    wasm_path: Union[str, pathlib.Path],
    *,
    data_dir: Union[str, pathlib.Path],
    server: Union[str, tc.URI],
    bearer_token: str | None = None,
    version: str = DEFAULT_VERSION,
    library_id: str = ILC_CLIENT_PATH,
) -> object:
    lib = ILCLibrary(server=server, version=version, library_id=library_id)
    kernel = kernel_for_local(data_dir, server=server, version=version, library_id=library_id)
    wasm_path = pathlib.Path(wasm_path)
    return tc.wasm.install(
        lib.schema(),
        wasm_path,
        kernel=kernel,
        bearer_token=bearer_token,
    )


__all__ = [
    "DEFAULT_VERSION",
    "ILC_CLIENT_PATH",
    "ILC_SERVER_PATH",
    "ILCLibrary",
    "install_wasm",
    "kernel_for_local",
]
