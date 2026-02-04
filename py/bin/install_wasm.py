#!/usr/bin/env python3
"""Install a WASM TinyChain library using the PyO3 tinychain bindings."""

from __future__ import annotations

import argparse
import pathlib
import sys

try:
    import tinychain as tc
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "tinychain module not installed; run scripts/install_tc_server_python.sh first"
    ) from exc


def _body_text(handle: "tc.StateHandle") -> str:
    value = handle.value()
    return value.to_json() if hasattr(value, "to_json") else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a WASM TinyChain library")
    parser.add_argument("schema", type=pathlib.Path, help="Path to schema JSON")
    parser.add_argument("wasm", type=pathlib.Path, help="Path to compiled .wasm")
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        help="Optional data directory to persist the library",
    )
    parser.add_argument(
        "--bearer-token",
        type=str,
        default=None,
        help="Optional bearer token for authorized installs",
    )
    args = parser.parse_args()

    response = install(
        args.schema,
        args.wasm,
        data_dir=args.data_dir,
        bearer_token=args.bearer_token,
    )
    if response.status != 204:
        body = _body_text(response.body) if response.body else ""
        raise RuntimeError(
            f"install failed: status={response.status} body={body}"
        )


def install(
    schema_path: pathlib.Path,
    wasm_path: pathlib.Path,
    *,
    kernel: "tc.KernelHandle | None" = None,
    data_dir: pathlib.Path | None = None,
    bearer_token: str | None = None,
) -> "tc.KernelResponse":
    return tc.wasm.install(
        schema_path,
        wasm_path,
        kernel=kernel,
        data_dir=data_dir,
        bearer_token=bearer_token,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:  # pragma: no cover
        sys.stderr.write(f"error: {err}\n")
        sys.exit(1)
