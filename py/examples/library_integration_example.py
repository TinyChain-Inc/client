#!/usr/bin/env python3
"""
Single-file integration example demonstrating two Python-side Library configurations:

1) A local *stub* `Library` backed by a local WASM binary, invoked through the PyO3 backend.
2) That local WASM route returns an `OpRef` (as JSON) pointing at a remote `/lib/...` dependency,
   which the kernel resolves via HTTP RPC to a running TinyChain server on another host.

This uses the existing Rust "Hello, world!" host + WASM example artifacts.
"""

from __future__ import annotations

import argparse
import pathlib
import tempfile
from typing import Optional
import os

try:
    import tinychain as tc
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "tinychain module not installed; run scripts/install_tc_server_python.sh first"
    ) from exc

REPO_ROOT = tc.testing.repo_root()

BEARER_TOKEN = os.environ.get("TC_BEARER_TOKEN")
TOKEN_HOST = os.environ.get("TC_TOKEN_HOST")
ACTOR_ID = os.environ.get("TC_ACTOR_ID")
PUBLIC_KEY_B64 = os.environ.get("TC_PUBLIC_KEY_B64")


class RemoteB(tc.Library):
    def __init__(self, authority: tc.URI) -> None:
        super().__init__(
            publisher="example-devco",
            name="example",
            version="0.1.0",
            dependencies=(),
            authority=authority,
        )

    @tc.get
    def hello(self, name: str) -> tc.String:
        ...


class LocalWasmA(tc.Library):
    def __init__(self, b: RemoteB) -> None:
        super().__init__(
            publisher="example-devco",
            name="a",
            version="0.1.0",
            dependencies=(b.link(),),
        )

    @tc.get
    def from_b(self, name: str) -> tc.String:
        ...


def _start_remote_host():
    return tc.testing.start_rust_example(
        "http_rpc_native_host",
        args=("--bind=127.0.0.1:0",),
        root=REPO_ROOT,
        prefer_binary=True,
        require_binary=True,
    )


def _ensure_opref_wasm() -> pathlib.Path:
    artifact = (
        REPO_ROOT
        / "tc-wasm"
        / "target"
        / "wasm32-unknown-unknown"
        / "release"
        / "examples"
        / "opref_to_remote.wasm"
    )
    if artifact.exists():
        return artifact

    raise RuntimeError(
        "opref_to_remote.wasm not found; build it first with "
        "`cargo build --manifest-path tc-wasm/Cargo.toml --example opref_to_remote "
        "--target wasm32-unknown-unknown --release`"
    )


def test_local_wasm_resolves_remote_opref(authority: str, wasm_path: pathlib.Path) -> None:
    b = RemoteB(tc.URI.parse(authority))
    with tempfile.TemporaryDirectory(prefix="tinychain-data-") as temp_dir:
        data_dir = pathlib.Path(temp_dir) / "tc-data"
        data_dir.mkdir(parents=True, exist_ok=True)

        a = LocalWasmA(b)
        if not BEARER_TOKEN or not TOKEN_HOST or not ACTOR_ID or not PUBLIC_KEY_B64:
            raise RuntimeError(
                "TC_BEARER_TOKEN must be set to a bearer token with install and txn claims. "
                "Generate one via: "
                "cargo run --example rjwt_install_token -- "
                "--host http://127.0.0.1:8702 --actor example-admin "
                "--lib /lib/example-devco/a/0.1.0"
            )

        install = tc.wasm.install(
            a.schema(),
            wasm_path,
            data_dir=data_dir,
            bearer_token=BEARER_TOKEN,
        )
        assert install.status == 204

        kernel = tc.kernel.for_library(a, data_dir=data_dir)

        with tc.backend(kernel):
            assert tc.execute(b.hello("World")) == "Hello, World!"

            # Local WASM call, where the WASM export returns an OpRef pointing at B; the kernel resolves
            # it via HTTP RPC within the same transaction and returns the resolved body.
            assert tc.execute(a.from_b("World")) == "Hello, World!"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authority",
        default=None,
        help="Optional remote TinyChain host authority (host:port); if omitted, starts the Rust example host",
    )
    args = parser.parse_args(argv)

    proc: Optional[subprocess.Popen[str]] = None
    authority = args.authority
    try:
        if authority is None:
            proc, addr = _start_remote_host()
            authority = addr

        wasm_path = _ensure_opref_wasm()
        test_local_wasm_resolves_remote_opref(authority, wasm_path)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
