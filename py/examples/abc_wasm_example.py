#!/usr/bin/env python3
"""ABC-style example using the ILC WASM client from Python.

This script installs `ilc-client/examples/cipher_wasm` into a local `data_dir`
and runs an `a + b - c` flow through TinyChain `OpRef`s executed by the
in-process backend.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile
from typing import Any, Optional

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
            "Install the ilc-client WASM binary and run an ABC-style flow "
            "(a + b - c) through local /cipher/add calls."
        )
    )
    parser.add_argument("--a", type=float, default=7.0, help="Operand a")
    parser.add_argument("--b", type=float, default=5.0, help="Operand b")
    parser.add_argument("--c", type=float, default=3.0, help="Operand c")
    parser.add_argument(
        "--metric",
        default="[3,5]",
        help="Metric coefficients as JSON list, e.g. '[3,5]'",
    )
    parser.add_argument(
        "--blind",
        default="[1.0,1.0]",
        help="Blind parameters as JSON list [gain, exponent]",
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
            "(required by schema; this example uses local cipher routes)"
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


def _parse_list(raw: str, label: str) -> list[Any]:
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    return value


def _extract_result_vector(response: Any) -> list[float]:
    if isinstance(response, dict):
        value = response.get("result")
    else:
        value = response

    if not isinstance(value, list):
        raise ValueError(f"unexpected cipher response payload: {response!r}")

    return [float(x) for x in value]


def _run(
    wasm_path: pathlib.Path,
    server: str,
    bearer_token: str,
    data_dir: pathlib.Path,
    a: float,
    b: float,
    c: float,
    metric: list[int],
    blind: list[float],
) -> None:
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

    with tc.backend(kernel, bearer_token=bearer_token):
        add_ab_raw = tc.execute(
            lib.opref_add(
                {
                    "metric": metric,
                    "blind": blind,
                    "lhs": [a, 0.0],
                    "rhs": [b, 0.0],
                }
            )
        )
        mul_ab_raw = tc.execute(
            lib.opref_mul(
                {
                    "metric": metric,
                    "blind": blind,
                    "lhs": [a, 0.0],
                    "rhs": [b, 0.0],
                }
            )
        )

        add_ab = _extract_result_vector(add_ab_raw)
        abc_raw = tc.execute(
            lib.opref_add(
                {
                    "metric": metric,
                    "blind": blind,
                    "lhs": add_ab,
                    "rhs": [-c, 0.0],
                }
            )
        )

    result = {
        "inputs": {"a": a, "b": b, "c": c},
        "cipher_add_ab": _extract_result_vector(add_ab_raw),
        "cipher_mul_ab": _extract_result_vector(mul_ab_raw),
        "cipher_abc": _extract_result_vector(abc_raw),
        "first_component_abc": _extract_result_vector(abc_raw)[0],
    }
    print(json.dumps(result, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    args = _args(argv)
    wasm_path = _require_wasm(args.wasm.resolve())
    token = _require_token(args.bearer_token)

    metric = [int(x) for x in _parse_list(args.metric, "--metric")]
    blind = [float(x) for x in _parse_list(args.blind, "--blind")]
    if len(metric) != 2:
        raise ValueError("--metric currently must contain exactly 2 coefficients")
    if len(blind) != 2:
        raise ValueError("--blind must contain exactly 2 values: [gain, exponent]")

    if args.data_dir is not None:
        _run(wasm_path, args.server, token, args.data_dir, args.a, args.b, args.c, metric, blind)
        return 0

    with tempfile.TemporaryDirectory(prefix="ilc-abc-wasm-") as tmp:
        _run(
            wasm_path,
            args.server,
            token,
            pathlib.Path(tmp) / "tc-data",
            args.a,
            args.b,
            args.c,
            metric,
            blind,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
