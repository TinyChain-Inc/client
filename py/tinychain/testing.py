from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Iterable, Optional, Tuple


_SCALAR_VALUE_PREFIX = "/state/scalar/value/"


def _unwrap_scalar_value(payload: object) -> object:
    if isinstance(payload, dict) and len(payload) == 1:
        (key, value), = payload.items()
        if isinstance(key, str) and key.startswith(_SCALAR_VALUE_PREFIX):
            return value
    return payload


def _unwrap_state(payload: object) -> object:
    if isinstance(payload, dict) and len(payload) == 1:
        (key, value), = payload.items()
        if isinstance(key, str) and key.startswith(_SCALAR_VALUE_PREFIX):
            return _unwrap_state(value)
        if key == "/state/scalar/map" and isinstance(value, dict):
            return {k: _unwrap_state(v) for k, v in value.items()}
    if isinstance(payload, list):
        return [_unwrap_state(item) for item in payload]
    return payload


def decode_json_body(response: "object"):
    body = getattr(response, "body", None)
    if body is None:
        raise AssertionError("response missing body")

    value = body.value()
    text = value.to_json() if hasattr(value, "to_json") else value
    return _unwrap_state(json.loads(text))


def response_json(response: "object"):
    return decode_json_body(response)


def cargo_available() -> bool:
    try:
        subprocess.run(["cargo", "--version"], check=True, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
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
) -> Tuple[subprocess.Popen[str], str]:
    """
    Start a Rust example which prints its bound `host:port` on stdout.

    - If `prefer_binary`, tries `tc-server/target/{release,debug}/examples/<name>` first.
    - Otherwise falls back to `cargo run --example <name> -- <args>`.
    """

    root = root or repo_root()

    if prefer_binary:
        candidates = [
            root / "tc-server" / "target" / "release" / "examples" / name,
            root / "tc-server" / "target" / "debug" / "examples" / name,
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
            assert proc.stdout is not None
            addr = proc.stdout.readline().strip()
            if not addr:
                stderr = proc.stderr.read() if proc.stderr is not None else ""
                proc.kill()
                raise RuntimeError(f"example {name} failed to start (stderr):\n{stderr}")
            return proc, addr

        if require_binary:
            raise RuntimeError(
                f"{name} binary not found. Build it first with:\n"
                f"  cargo build --manifest-path tc-server/Cargo.toml --example {name}\n"
                "Or run the host in another terminal and pass `--authority host:port`."
            )

    if not cargo_available():
        raise RuntimeError("`cargo` not found; install Rust tooling to run this example")

    proc = subprocess.Popen(
        [
            "cargo",
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

    assert proc.stdout is not None
    addr = proc.stdout.readline().strip()
    if not addr:
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        proc.kill()
        raise RuntimeError(f"example {name} failed to start (stderr):\n{stderr}")

    return proc, addr
