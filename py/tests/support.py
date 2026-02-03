from __future__ import annotations

import pathlib
import os
import subprocess
import pytest

import tinychain as tc

REPO_ROOT = tc.testing.repo_root()


def require_cargo() -> None:
    if not tc.testing.cargo_available():
        pytest.skip("`cargo` not found; install Rust tooling to run this test")


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
