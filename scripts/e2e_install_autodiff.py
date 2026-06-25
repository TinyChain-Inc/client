#!/usr/bin/env python3
"""Start tc-server and install the autodiff OpDef library for e2e tests.

Usage:
    python scripts/e2e_install_autodiff.py [--server-binary PATH]

The script:
  1. Generates a one-time Falcon-512 keypair + bearer token via rjwt_install_token.
  2. Starts tc-server with a trusted-installer policy for that keypair.
  3. Waits for the server to become healthy.
  4. PUTs the autodiff library definition to /lib.
  5. Verifies the installed routes respond.

Leave the process running and then execute:
    python -m pytest py/tests/test_e2e_add_gradient.py -v
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import requests

TC_SERVER_URL = "http://127.0.0.1:8702"
AUTODIFF_ROUTE_ROOT = "/lib/std/autodiff/0.1.0"
ACTOR_ID = "autodiff-admin"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TC_SERVER_DIR = os.path.join(REPO_ROOT, "tc-server")


def _binary(name: str, fallback: str) -> str:
    return fallback if os.path.isfile(fallback) else shutil.which(name) or fallback


def generate_token(token_binary: str) -> tuple[str, str]:
    """Return (public_key_b64, bearer_token) for /lib/std/autodiff/0.1.0."""
    result = subprocess.run(
        [
            token_binary,
            "--host", TC_SERVER_URL,
            "--actor", ACTOR_ID,
            "--lib", AUTODIFF_ROUTE_ROOT,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    public_key_b64 = ""
    bearer_token = ""
    in_bearer = False
    for line in result.stdout.splitlines():
        if line.startswith("public_key_b64:"):
            public_key_b64 = line.split(":", 1)[1].strip()
            in_bearer = False
        elif line.startswith("bearer_token:"):
            bearer_token = line.split(":", 1)[1].strip()
            in_bearer = True
        elif line.startswith("secret_key_b64:"):
            in_bearer = False
        elif in_bearer:
            bearer_token += line.strip()

    if not public_key_b64 or not bearer_token:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError("Failed to parse rjwt_install_token output")

    return public_key_b64, bearer_token


def start_server(
    server_binary: str,
    data_dir: str,
    public_key_b64: str,
) -> subprocess.Popen:
    trusted_installers = json.dumps([
        {
            "host": TC_SERVER_URL,
            "actor_id": ACTOR_ID,
            "public_key_b64": public_key_b64,
            "allowed_lib_prefixes": ["/lib/std"],
        }
    ])

    cmd = [
        server_binary,
        "--no-replicate",
        "--data-dir", data_dir,
        "--trusted-installers-json", trusted_installers,
    ]
    print(f"[e2e] starting: {shlex.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=TC_SERVER_DIR)
    return proc


def wait_healthy(timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = requests.get(f"{TC_SERVER_URL}/healthz", timeout=1)
            if resp.status_code == 200:
                print(f"[e2e] tc-server healthy at {TC_SERVER_URL}")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"tc-server did not become healthy within {timeout}s")


def library_definition() -> dict:
    """Return the OpDef library definition JSON for the autodiff routes.

    add(x, y) -> tensor:
        body: $x.add(r=$y)

    broadcast_reduce(x, target_shape) -> tensor:
        body: $x.broadcast_reduce(target_shape=$target_shape)
    """
    opdef_post = "/state/scalar/op/post"
    return {
        AUTODIFF_ROUTE_ROOT: {
            "add": {
                opdef_post: [
                    ["result", {"$x/add": {"r": {"$y": []}}}],
                ]
            },
            "broadcast_reduce": {
                opdef_post: [
                    ["result", {"$x/broadcast_reduce": {"target_shape": {"$target_shape": []}}}],
                ]
            },
            "matmul": {
                opdef_post: [
                    ["result", {"$x/matmul": {"r": {"$y": []}}}],
                ]
            },
            "transpose": {
                opdef_post: [
                    ["result", {"$x/transpose": [{"$perm": []}]}],
                ]
            },
        }
    }


def install_library(bearer_token: str) -> None:
    definition = library_definition()
    resp = requests.put(
        f"{TC_SERVER_URL}/lib",
        json=definition,
        headers={"Authorization": f"Bearer {bearer_token}"},
        timeout=10,
    )
    if resp.status_code not in (200, 204):
        raise RuntimeError(
            f"Library install failed: HTTP {resp.status_code}\n{resp.text}"
        )
    print(f"[e2e] library installed (HTTP {resp.status_code})")


def verify_routes() -> None:
    for route in ("add", "broadcast_reduce", "matmul", "transpose"):
        url = f"{TC_SERVER_URL}{AUTODIFF_ROUTE_ROOT}/{route}"
        resp = requests.post(url, json={}, timeout=5)
        if resp.status_code == 404:
            raise RuntimeError(f"Route {url!r} returned 404 after install")
        print(f"[e2e] probe {route}: HTTP {resp.status_code} ✓")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-binary",
        default=os.path.join(TC_SERVER_DIR, "target/debug/tc-server"),
        help="Path to tc-server binary",
    )
    parser.add_argument(
        "--token-binary",
        default=os.path.join(TC_SERVER_DIR, "target/debug/examples/rjwt_install_token"),
        help="Path to rjwt_install_token binary",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Data directory for tc-server (default: fresh tmpdir)",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Do not delete the data directory on exit",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or tempfile.mkdtemp(prefix="tc-e2e-")
    os.makedirs(data_dir, exist_ok=True)
    print(f"[e2e] data dir: {data_dir}")

    print("[e2e] generating install token…")
    public_key_b64, bearer_token = generate_token(args.token_binary)
    print(f"[e2e] actor: {ACTOR_ID}, claim: {AUTODIFF_ROUTE_ROOT}")

    proc = start_server(args.server_binary, data_dir, public_key_b64)

    def _cleanup(signum=None, frame=None):
        print("\n[e2e] shutting down tc-server…")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if not args.keep_data:
            shutil.rmtree(data_dir, ignore_errors=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        wait_healthy()
        install_library(bearer_token)
        verify_routes()

        print()
        print("=" * 60)
        print("tc-server is running with autodiff library installed.")
        print("Run in another terminal:")
        print()
        print("  python -m pytest py/tests/test_e2e_add_gradient.py -v")
        print("  python -m pytest py/tests/test_e2e_matmul_gradient.py -v")
        print()
        print("Press Ctrl+C to stop the server.")
        print("=" * 60)

        proc.wait()
    except Exception as exc:
        print(f"[e2e] error: {exc}", file=sys.stderr)
        _cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
