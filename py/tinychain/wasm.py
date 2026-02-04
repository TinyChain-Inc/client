from __future__ import annotations

import base64
import json
import os
import pathlib
from typing import Optional, Union


Schema = Union[pathlib.Path, dict]


def _read_schema(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_wasm_b64(path: pathlib.Path) -> str:
    data = path.read_bytes()
    if not data:
        raise RuntimeError(f"WASM binary {path} is empty")
    return base64.b64encode(data).decode("ascii")


def install(
    schema: Schema,
    wasm_path: pathlib.Path,
    *,
    kernel: Optional[object] = None,
    data_dir: Optional[pathlib.Path] = None,
    bearer_token: Optional[str] = None,
) -> object:
    try:
        import tinychain_local as local  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "`tc.wasm.install` requires the optional `tinychain-local` backend"
        ) from exc

    schema_value = schema if isinstance(schema, dict) else _read_schema(schema)

    if kernel is None:
        if data_dir is None:
            raise ValueError("expected either `kernel` or `data_dir`")
        token_host = os.environ.get("TC_TOKEN_HOST")
        actor_id = os.environ.get("TC_ACTOR_ID")
        public_key_b64 = os.environ.get("TC_PUBLIC_KEY_B64")
        if token_host and actor_id and public_key_b64:
            kernel = local.KernelHandle.with_library_schema_rjwt(
                json.dumps(schema_value, separators=(",", ":")),
                token_host,
                actor_id,
                public_key_b64,
                data_dir=str(data_dir),
            )
        else:
            kernel = local.KernelHandle.local(data_dir=str(data_dir))
    payload = json.dumps(
        {
            "schema": schema_value,
            "artifacts": [
                {
                    "path": "/lib/wasm",
                    "content_type": "application/wasm",
                    "bytes": _read_wasm_b64(wasm_path),
                }
            ],
        },
        separators=(",", ":"),
    )

    headers = None
    if bearer_token is not None:
        headers = [("authorization", f"Bearer {bearer_token}")]
    request = local.KernelRequest("PUT", "/lib", headers, local.StateHandle(payload))
    return kernel.dispatch(request)
