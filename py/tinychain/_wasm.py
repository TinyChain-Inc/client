from __future__ import annotations

import base64
import json
import pathlib
from typing import Optional, Union

from ._install import dispatch_install, kernel_for_install, local_backend, token_bearer, token_rjwt_parts
from .uri import uri


Schema = Union[pathlib.Path, dict]


def _read_schema(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_wasm_b64(path: pathlib.Path) -> str:
    data = path.read_bytes()
    if not data:
        raise RuntimeError(f"WASM binary {path} is empty")
    return base64.b64encode(data).decode("ascii")


def install_payload(schema: Schema, wasm_path: pathlib.Path) -> dict:
    schema_value = schema if isinstance(schema, dict) else _read_schema(schema)
    return {
        "schema": schema_value,
        "artifacts": [
            {
                "path": uri("lib", "wasm").path,
                "content_type": "application/wasm",
                "bytes": _read_wasm_b64(wasm_path),
            }
        ],
    }


def install(
    schema: Schema,
    wasm_path: pathlib.Path,
    *,
    kernel: Optional[object] = None,
    data_dir: Optional[pathlib.Path] = None,
    token: object | None = None,
    bearer_token: Optional[str] = None,
    token_host: str | None = None,
    actor_id: str | None = None,
    public_key_b64: str | None = None,
) -> object:
    local = local_backend()

    token_host_from_token, actor_id_from_token, public_key_b64_from_token = token_rjwt_parts(token)
    bearer_token = token_bearer(token, bearer_token)
    if bearer_token is None:
        raise ValueError("expected `bearer_token` for WASM installs")

    schema_value = schema if isinstance(schema, dict) else _read_schema(schema)

    token_host = token_host or token_host_from_token
    actor_id = actor_id or actor_id_from_token
    public_key_b64 = public_key_b64 or public_key_b64_from_token
    kernel = kernel_for_install(
        local,
        kernel=kernel,
        data_dir=data_dir,
        schema=schema_value,
        token=None,
        token_host=token_host,
        actor_id=actor_id,
        public_key_b64=public_key_b64,
    )

    payload = install_payload(schema_value, wasm_path)
    return dispatch_install(local, kernel, payload, bearer_token=bearer_token)
