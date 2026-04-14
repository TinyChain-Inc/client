#!/usr/bin/env python3
"""
Framework-native mixed-mode example (no custom request/response wrappers).

Shows all three modes together:
1) method definition (`@tc.get`)
2) authority-driven dependency routing from declared library dependencies
3) one idiomatic backend execution context (`with tc.backend(kernel): ...`)

Auth model demonstrated:
- mint minimal-scope RJWT bearer tokens from an Ed25519 private key
- use a short-lived install token (claim: local `/lib/.../a/...`)

Prerequisites:
- tinychain-local installed
- remote Rust host example binary built (`http_rpc_native_host`)
- WASM example built (`opref_to_remote.wasm`)
- `cargo` available, or `rjwt_install_token` binary already built
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import tempfile
from typing import Optional

try:
    import tinychain as tc
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "tinychain module not installed; run scripts/install_tc_server_python.sh first"
    ) from exc

REPO_ROOT = tc.testing.repo_root()
DEFAULT_SECRET_KEY_B64 = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="


class RemoteB(tc.Library):
    publisher = "example-devco"
    name = "example"
    version = "0.1.0"
    dependencies = ()

    @tc.get
    def hello(self, name: str) -> tc.String:
        ...


class LocalWasmA(tc.Library):
    publisher = "example-devco"
    name = "a"
    version = "0.1.0"

    @tc.get
    def from_b(self, name: str) -> tc.String:
        ...

    @tc.get
    def auth_context(self) -> tc.Ref:
        ...


def _require_local_backend() -> None:
    try:
        _ = tc.KernelHandle.local
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "tinychain-local is required for local backend execution. "
            "Install it first (e.g. scripts/install_tc_server_python.sh)."
        ) from exc


def _start_remote_host(*, actor_id: str, secret_key_b64: str):
    return tc.testing.start_rust_example(
        "http_rpc_native_host",
        args=(
            "--bind=127.0.0.1:0",
            f"--actor-id={actor_id}",
            f"--secret-key-b64={secret_key_b64}",
        ),
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
        "opref_to_remote.wasm not found; build it first with:\n"
        "  cargo build --manifest-path tc-wasm/Cargo.toml "
        "--example opref_to_remote --target wasm32-unknown-unknown --release"
    )

def run_demo(
    authority: str,
    wasm_path: pathlib.Path,
    *,
    actor_id: str,
    secret_key_b64: str | None,
    ttl_secs: int,
) -> None:
    b = RemoteB(authority=tc.URI.parse(authority))
    a = LocalWasmA(dependencies=(b.link(),))
    a_root = tc.uri(a).path
    b_root = tc.uri(b).path

    host_link = tc.origin(authority)

    mint_secret = secret_key_b64 or DEFAULT_SECRET_KEY_B64

    install_token = tc.auth.mint_rjwt_token(
        host=host_link,
        actor_id=actor_id,
        libs=[a_root],
        ttl_secs=ttl_secs,
        secret_key_b64=mint_secret,
        repo_root=REPO_ROOT,
    )
    runtime_token = tc.auth.mint_rjwt_token(
        host=host_link,
        actor_id=actor_id,
        libs=[b_root, a_root],
        ttl_secs=ttl_secs,
        secret_key_b64=mint_secret,
        repo_root=REPO_ROOT,
    )

    print("minted install token claims:", [a_root])
    print("minted runtime token claims:", [b_root, a_root])
    print("actor_id:", runtime_token.actor_id)
    print("public_key_b64:", runtime_token.public_key_b64)

    host = tc.Host(tc.origin(authority))

    # Explicit transport call for direct host execution.
    print(
        "explicit RPC via Host:",
        host.execute(b.hello("World")),
    )

    with tempfile.TemporaryDirectory(prefix="tinychain-data-") as temp_dir:
        data_dir = pathlib.Path(temp_dir) / "tc-data"
        data_dir.mkdir(parents=True, exist_ok=True)

        kernel = tc.kernel.with_library(
            a,
            data_dir=data_dir,
            token=runtime_token,
        )

        install = tc.wasm.install(
            a.schema(),
            wasm_path,
            kernel=kernel,
            token=install_token,
        )
        if install.status != 204:
            raise RuntimeError(f"unexpected install status {install.status}")

        # Idiomatic mode: one backend context, route calls auto-execute.
        with tc.backend(
            kernel,
            bearer_token=runtime_token.bearer_token,
        ):
            print("route auth context:", a.auth_context())
            print("auto remote call:", b.hello("World"))
            print("auto local call:", a.from_b("World"))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authority",
        default=None,
        help="Optional remote TinyChain host authority (host:port); if omitted, starts the Rust example host",
    )
    parser.add_argument(
        "--actor-id",
        default="example-admin",
        help="Actor ID used when minting scoped tokens",
    )
    parser.add_argument(
        "--secret-key-b64",
        default=None,
        help="Optional Ed25519 secret key (base64). If omitted, an ephemeral keypair is generated.",
    )
    parser.add_argument(
        "--ttl-secs",
        type=int,
        default=300,
        help="Token TTL in seconds (default: 300)",
    )
    args = parser.parse_args(argv)
    _require_local_backend()

    proc: Optional[subprocess.Popen[str]] = None
    authority = args.authority
    try:
        if authority is None:
            proc, addr = _start_remote_host(
                actor_id=args.actor_id,
                secret_key_b64=args.secret_key_b64 or DEFAULT_SECRET_KEY_B64,
            )
            authority = addr

        wasm_path = _ensure_opref_wasm()
        run_demo(
            authority,
            wasm_path,
            actor_id=args.actor_id,
            secret_key_b64=args.secret_key_b64,
            ttl_secs=args.ttl_secs,
        )
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
