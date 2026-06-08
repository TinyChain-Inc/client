from __future__ import annotations

import pathlib
import os
import subprocess
import pytest

import tinychain as tc
import tinychain.testing as tc_testing

REPO_ROOT = tc_testing.repo_root()


def require_cargo() -> None:
    if tc_testing.rjwt_install_token_bin() is not None:
        return
    if not tc_testing.cargo_available():
        pytest.skip("`cargo` not found and rjwt_install_token binary missing")


def wasm_example_artifact(example_name: str) -> pathlib.Path:
    return (
        REPO_ROOT
        / "tc-wasm"
        / "target"
        / "wasm32-unknown-unknown"
        / "release"
        / "examples"
        / f"{example_name}.wasm"
    )


def ensure_wasm_example_built(example_name: str) -> pathlib.Path:
    artifact = wasm_example_artifact(example_name)
    if artifact.exists():
        return artifact

    auto_build = os.environ.get("TC_AUTO_BUILD_WASM", "0") == "1"
    if not auto_build:
        pytest.fail(
            f"{artifact.name} not found. Build it first with "
            f"`cargo build --manifest-path tc-wasm/Cargo.toml --example {example_name} "
            "--target wasm32-unknown-unknown --release` "
            "(set TC_AUTO_BUILD_WASM=1 to let the test run that command automatically)."
        )

    require_cargo()
    try:
        subprocess.run(
            [
                "cargo",
                "build",
                "--manifest-path",
                str(REPO_ROOT / "tc-wasm" / "Cargo.toml"),
                "--example",
                example_name,
                "--target",
                "wasm32-unknown-unknown",
                "--release",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as err:
        pytest.fail(
            f"failed to build tc-wasm example {example_name}. Resolve the error below:\n{err}"
        )

    if not artifact.exists():
        pytest.fail(
            f"tc-wasm build reported success but {artifact.name} is missing; "
            "ensure the wasm target is installed and the build directory is writable."
        )

    return artifact


def rjwt_install_token(*lib_paths: str) -> dict[str, str]:
    require_cargo()
    host = os.environ.get("TC_TOKEN_HOST", "http://127.0.0.1:8702")
    actor_id = os.environ.get("TC_ACTOR_ID", "example-admin")

    binary = os.environ.get("TC_RJWT_INSTALL_TOKEN_BIN")
    if binary:
        args = [
            binary,
            "--host",
            host,
            "--actor",
            actor_id,
        ]
    else:
        bin_path = tc_testing.rjwt_install_token_bin()
        if bin_path is not None:
            args = [
                str(bin_path),
                "--host",
                host,
                "--actor",
                actor_id,
            ]
        else:
            args = [
                "cargo",
                "run",
                "--manifest-path",
                str(REPO_ROOT / "tc-server" / "Cargo.toml"),
                "--example",
                "rjwt_install_token",
                "--",
                "--host",
                host,
                "--actor",
                actor_id,
            ]
    for lib_path in lib_paths:
        args.extend(["--lib", lib_path])

    try:
        result = subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as err:
        pytest.fail(f"failed to mint install token: {err.stderr or err}")

    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key in {"host", "actor_id", "public_key_b64", "bearer_token"}:
            values[key] = value.strip()

    missing = [key for key in ("host", "actor_id", "public_key_b64", "bearer_token") if key not in values]
    if missing:
        pytest.fail(f"token output missing fields: {', '.join(missing)}")

    return values
