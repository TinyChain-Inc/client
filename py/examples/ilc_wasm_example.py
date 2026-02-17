#!/usr/bin/env python3
"""Install the ILC client WASM library and call its local cipher routes from Python."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile
from typing import Optional

import ilc
import tinychain as tc


REPO_ROOT = tc.testing.repo_root()
DEFAULT_WASM = (
    REPO_ROOT
    / "target"
    / "wasm32-unknown-unknown"
    / "release"
    / "examples"
    / "cipher_wasm.wasm"
)


def _args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install the ilc-client WASM binary into a local data_dir and call "
            "POST /cipher/add and POST /cipher/mul via the PyO3 backend."
        )
    )
    parser.add_argument(
        "--wasm",
        type=pathlib.Path,
        default=DEFAULT_WASM,
        help="Path to ilc-client/examples/cipher_wasm WASM artifact",
    )
    parser.add_argument(
        "--server",
        default="127.0.0.1:8700",
        help=(
            "Authority used for /lib/applied-physics/ilc dependency routing "
            "(required by schema; add/mul calls run locally)"
        ),
    )
    parser.add_argument(
        "--bearer-token",
        default=os.environ.get("TC_BEARER_TOKEN"),
        help="Bearer token used for /lib install and route execution",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=None,
        help="Optional persistent data_dir (defaults to a temporary directory)",
    )
    return parser.parse_args(argv)


def _require_wasm(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(
            f"WASM artifact not found at {path}. Build it first with: "
            "cargo build --manifest-path ilc-client/Cargo.toml --example cipher_wasm "
            "--target wasm32-unknown-unknown --release"
        )
    return path


def _require_token(token: Optional[str]) -> str:
    if token:
        return token

    raise RuntimeError(
        "missing bearer token. Set TC_BEARER_TOKEN or pass --bearer-token. "
        "For local dev, generate one with: "
        "cargo run -p ilc-server --example issue_token -- "
        "--link /lib/applied-physics/ilc-client --mode 700 --ttl 10m"
    )


def _run(wasm_path: pathlib.Path, server: str, bearer_token: str, data_dir: pathlib.Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    install = ilc.install_wasm(
        wasm_path,
        data_dir=data_dir,
        server=server,
        bearer_token=bearer_token,
    )
    if getattr(install, "status", None) != 204:
        raise RuntimeError(f"WASM install failed: status={getattr(install, 'status', None)}")

    kernel = ilc.kernel_for_local(data_dir=data_dir, server=server)
    lib = ilc.ILCLibrary(server=server)
    payload = {
        "metric": [3, 5],
        "blind": [1.0, 1.0],
        "lhs": [1.0, 2.0],
        "rhs": [3.0, 4.0],
    }

    with tc.backend(kernel, bearer_token=bearer_token):
        add_result = tc.execute(lib.opref_add(payload))
        mul_result = tc.execute(lib.opref_mul(payload))

    print(json.dumps({"add": add_result, "mul": mul_result}, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    args = _args(argv)
    wasm_path = _require_wasm(args.wasm.resolve())
    token = _require_token(args.bearer_token)

    if args.data_dir is not None:
        _run(wasm_path, args.server, token, args.data_dir)
        return 0

    with tempfile.TemporaryDirectory(prefix="ilc-wasm-") as tmp:
        _run(wasm_path, args.server, token, pathlib.Path(tmp) / "tc-data")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
