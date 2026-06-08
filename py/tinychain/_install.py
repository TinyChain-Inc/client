from __future__ import annotations

import json
import pathlib
from typing import Optional


def token_bearer(token: object | None, bearer_token: Optional[str]) -> Optional[str]:
    if bearer_token is not None:
        return bearer_token
    if token is None:
        return None
    bearer = getattr(token, "bearer_token", None)
    return str(bearer) if bearer is not None else None


def token_rjwt_parts(token: object | None) -> tuple[str | None, str | None, str | None]:
    if token is None:
        return None, None, None

    host = getattr(token, "host", None)
    actor = getattr(token, "actor_id", None)
    pub = getattr(token, "public_key_b64", None)
    bearer = getattr(token, "bearer_token", None)
    if host is None and actor is None and pub is None and bearer is not None:
        return None, None, None
    if host is None and actor is None and pub is None:
        raise TypeError("expected `token` with `host`, `actor_id`, and `public_key_b64` attributes")

    return (
        str(host) if host is not None else None,
        str(actor) if actor is not None else None,
        str(pub) if pub is not None else None,
    )


def local_backend():
    try:
        import tinychain_local as local  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("install requires the optional `tinychain-local` backend") from exc
    return local


def kernel_for_install(
    local: object,
    *,
    kernel: object | None,
    data_dir: pathlib.Path | None,
    schema: dict | None = None,
    token: object | None = None,
    token_host: str | None = None,
    actor_id: str | None = None,
    public_key_b64: str | None = None,
) -> object:
    if kernel is not None:
        return kernel

    if data_dir is None:
        raise ValueError("expected either `kernel` or `data_dir`")

    if schema is not None:
        token_host_from_token, actor_id_from_token, public_key_b64_from_token = token_rjwt_parts(token)
        token_host = token_host or token_host_from_token
        actor_id = actor_id or actor_id_from_token
        public_key_b64 = public_key_b64 or public_key_b64_from_token
        if token_host and actor_id and public_key_b64:
            return local.KernelHandle.with_library_schema_rjwt(
                json.dumps(schema, separators=(",", ":")),
                token_host,
                actor_id,
                public_key_b64,
                data_dir=str(data_dir),
            )

    return local.KernelHandle.local(data_dir=str(data_dir))


def dispatch_install(
    local: object,
    kernel: object,
    payload: dict,
    *,
    bearer_token: str,
) -> object:
    import tinychain as tc

    install_path = tc.uri("lib").path
    body = json.dumps(payload, separators=(",", ":"))
    headers = [("authorization", f"Bearer {bearer_token}")]
    request = local.KernelRequest("PUT", install_path, headers, local.StateHandle(body))
    response = kernel.dispatch(request)
    return finalize_implicit_txn(local, kernel, response, install_path, bearer_token)


def header_value(response: object, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    needle = name.lower()
    for key, value in headers:
        if str(key).lower() == needle:
            return str(value)
    return None


def finalize_implicit_txn(
    local: object,
    kernel: object,
    response: object,
    path: str,
    bearer_token: str | None,
) -> object:
    txn_id = header_value(response, "x-tc-txn-id")
    if not txn_id:
        return response

    headers = [("authorization", f"Bearer {bearer_token}")] if bearer_token else None
    commit = local.KernelRequest("POST", f"{path}?txn_id={txn_id}", headers, None)
    return kernel.dispatch(commit)
