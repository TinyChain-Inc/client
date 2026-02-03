from __future__ import annotations

import base64
import json
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
) -> object:
    try:
        import tinychain_local as local  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "`tc.wasm.install` requires the optional `tinychain-local` backend"
        ) from exc

    if kernel is None:
        if data_dir is None:
            raise ValueError("expected either `kernel` or `data_dir`")
        kernel = local.KernelHandle.local(data_dir=str(data_dir))

    schema_value = schema if isinstance(schema, dict) else _read_schema(schema)

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

    request = local.KernelRequest("PUT", "/lib", None, local.StateHandle(payload))
    return kernel.dispatch(request)
