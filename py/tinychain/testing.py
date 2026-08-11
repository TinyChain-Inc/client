from __future__ import annotations

import pathlib
import selectors
import signal
import subprocess
import time
import shutil
import os
from typing import Callable, Iterable, Optional, Tuple, TypeVar

from .codec import decode_payload, decode_response_body

_T = TypeVar("_T")


def decode_json_body(response: "object"):
    return decode_response_body(response)


def response_json(response: "object"):
    return decode_json_body(response)

def cargo_command() -> str | None:
    preferred = os.path.expanduser("~/.cargo/bin/cargo")
    for path in (preferred, shutil.which("cargo"), "/snap/bin/cargo"):
        if path is None:
            continue
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def cargo_available() -> bool:
    cmd = cargo_command()
    if cmd is None:
        return False
    try:
        subprocess.run([cmd, "--version"], check=True, capture_output=True, text=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def repo_root(start: Optional[pathlib.Path] = None) -> pathlib.Path:
    """
    Best-effort repository root discovery for local dev/test harnesses.

    Looks for a directory containing both `tc-server` and `tc-wasm`.
    """

    cursor = (start or pathlib.Path(__file__)).resolve()
    for parent in [cursor, *cursor.parents]:
        if (parent / "tc-server").is_dir() and (parent / "tc-wasm").is_dir():
            return parent
    raise RuntimeError("unable to locate repo root (expected `tc-server/` and `tc-wasm/`)")


def start_rust_example(
    name: str,
    *,
    args: Iterable[str] = (),
    root: Optional[pathlib.Path] = None,
    prefer_binary: bool = True,
    require_binary: bool = False,
    startup_timeout_secs: float = 20.0,
) -> Tuple[subprocess.Popen[str], str]:
    """
    Start a Rust example which prints its bound `host:port` on stdout.

    - If `prefer_binary`, tries `tc-server/target/{release,debug}/examples/<name>` first.
    - Otherwise falls back to `cargo run --example <name> -- <args>`.
    """

    root = root or repo_root()

    def _read_addr_with_timeout(proc: subprocess.Popen[str]) -> str:
        assert proc.stdout is not None
        deadline = time.monotonic() + startup_timeout_secs
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    proc.kill()
                    _, stderr = proc.communicate()
                    raise RuntimeError(
                        f"example {name} did not print a bound address within "
                        f"{startup_timeout_secs:.1f}s (stderr):\n{stderr}"
                    )

                if proc.poll() is not None:
                    stderr = proc.stderr.read() if proc.stderr is not None else ""
                    raise RuntimeError(f"example {name} exited before startup (stderr):\n{stderr}")

                ready = selector.select(timeout=remaining)
                if not ready:
                    continue

                addr = proc.stdout.readline().strip()
                if addr:
                    return addr
        finally:
            selector.close()

    if prefer_binary:
        candidates = [
            root / "target" / "release" / "examples" / name,
            root / "target" / "debug" / "examples" / name,
        ]
        binary = next((p for p in candidates if p.exists()), None)
        if binary is not None:
            proc = subprocess.Popen(
                [str(binary), *list(args)],
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            addr = _read_addr_with_timeout(proc)
            return proc, addr

        if require_binary:
            raise RuntimeError(
                f"{name} binary not found. Build it first with:\n"
                f"  cargo build --manifest-path tc-server/Cargo.toml --example {name}\n"
                "Or run the host in another terminal and pass `--authority host:port`."
            )

    if not cargo_available():
        raise RuntimeError("`cargo` not found; install Rust tooling to run this example")
    cargo = cargo_command()
    if cargo is None:
        raise RuntimeError("`cargo` not found; install Rust tooling to run this example")

    proc = subprocess.Popen(
        [
            cargo,
            "run",
            "--manifest-path",
            str(root / "tc-server" / "Cargo.toml"),
            "--example",
            name,
            "--",
            *list(args),
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    addr = _read_addr_with_timeout(proc)
    return proc, addr


def run_with_timeout(timeout_secs: int, fn: Callable[[], _T]) -> _T:
    if timeout_secs <= 0:
        raise ValueError("timeout_secs must be positive")

    def _on_timeout(_signum, _frame):
        raise TimeoutError(f"operation exceeded {timeout_secs}s timeout")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(timeout_secs)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
