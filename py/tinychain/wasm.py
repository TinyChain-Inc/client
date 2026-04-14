from __future__ import annotations

import base64
import json
import pathlib
from typing import Optional, Union

from .uri import uri


Schema = Union[pathlib.Path, dict]


def _token_parts(token: object | None) -> tuple[str | None, str | None, str | None, str | None]:
    if token is None:
        return None, None, None, None

    host = getattr(token, "host", None)
    actor = getattr(token, "actor_id", None)
    pub = getattr(token, "public_key_b64", None)
    bearer = getattr(token, "bearer_token", None)
    if host is None and actor is None and pub is None and bearer is None:
        raise TypeError(
            "expected `token` with `host`, `actor_id`, `public_key_b64`, and optionally `bearer_token` attributes"
        )

    return host, actor, pub, bearer


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
    token: object | None = None,
    bearer_token: Optional[str] = None,
    token_host: str | None = None,
    actor_id: str | None = None,
    public_key_b64: str | None = None,
) -> object:
    try:
        import tinychain_local as local  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "`tc.wasm.install` requires the optional `tinychain-local` backend"
        ) from exc

    token_host_from_token, actor_id_from_token, public_key_b64_from_token, bearer_from_token = _token_parts(token)
    bearer_token = bearer_token or bearer_from_token
    if bearer_token is None:
        raise ValueError("expected `bearer_token` for WASM installs")

    schema_value = schema if isinstance(schema, dict) else _read_schema(schema)

    if kernel is None:
        if data_dir is None:
            raise ValueError("expected either `kernel` or `data_dir`")
        token_host = token_host or token_host_from_token
        actor_id = actor_id or actor_id_from_token
        public_key_b64 = public_key_b64 or public_key_b64_from_token
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
                    "path": uri("lib", "wasm").path,
                    "content_type": "application/wasm",
                    "bytes": _read_wasm_b64(wasm_path),
                }
            ],
        },
        separators=(",", ":"),
    )

    headers = [("authorization", f"Bearer {bearer_token}")]
    install_path = uri("lib").path
    request = local.KernelRequest("PUT", install_path, headers, local.StateHandle(payload))
    response = kernel.dispatch(request)
    return _finalize_implicit_txn(local, kernel, response, install_path, bearer_token)


def _header_value(response: object, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    needle = name.lower()
    for key, value in headers:
        if str(key).lower() == needle:
            return str(value)
    return None


def _finalize_implicit_txn(
    local: object,
    kernel: object,
    response: object,
    path: str,
    bearer_token: str | None,
) -> object:
    txn_id = _header_value(response, "x-tc-txn-id")
    if not txn_id:
        return response

    headers = [("authorization", f"Bearer {bearer_token}")] if bearer_token else None
    commit = local.KernelRequest("POST", f"{path}?txn_id={txn_id}", headers, None)
    return kernel.dispatch(commit)
