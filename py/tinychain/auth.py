from __future__ import annotations

import pathlib
import subprocess
from dataclasses import dataclass
from typing import Sequence

from . import testing


@dataclass(frozen=True, slots=True)
class SignedBearerToken:
    host: str
    actor_id: str
    public_key_b64: str
    bearer_token: str
    secret_key_b64: str = ""


def bearer_token(token: object | None = None) -> str | None:
    if token is None:
        return None
    bearer = getattr(token, "bearer_token", None)
    return str(bearer) if bearer is not None else None


def context():
    """
    Return the framework-authenticated route context for the current request/transaction.

    This is evaluated by the kernel from validated bearer auth, not caller-provided payload.
    """
    import tinychain as tc

    return tc.Ref(tc.OpRef("GET", tc.uri("host", "auth", "context").path))


def _resolve_minter_command(
    *,
    repo_root: pathlib.Path,
    binary: str | pathlib.Path | None,
) -> list[str]:
    if binary is not None:
        return [str(binary)]

    built = testing.rjwt_install_token_bin(repo_root)
    if built is not None:
        return [str(built)]

    if testing.cargo_available():
        return [
            "cargo",
            "run",
            "--manifest-path",
            str(repo_root / "tc-server" / "Cargo.toml"),
            "--example",
            "rjwt_install_token",
            "--",
        ]

    raise RuntimeError(
        "cannot mint token: no `rjwt_install_token` binary found and `cargo` is unavailable"
    )


def mint_rjwt_token(
    *,
    host: str,
    actor_id: str,
    libs: Sequence[str],
    ttl_secs: int = 3600,
    secret_key_b64: str | None = None,
    repo_root: pathlib.Path | None = None,
    binary: str | pathlib.Path | None = None,
) -> SignedBearerToken:
    if not libs:
        raise ValueError("minted token requires at least one `libs` claim")
    if not host.strip():
        raise ValueError("minted token host must be non-empty")
    if not actor_id.strip():
        raise ValueError("minted token actor_id must be non-empty")

    root = repo_root or testing.repo_root()
    cmd = _resolve_minter_command(repo_root=root, binary=binary)
    cmd.extend(["--host", host, "--actor", actor_id])
    if secret_key_b64:
        cmd.extend(["--secret-key-b64", secret_key_b64])
    for lib in libs:
        cmd.extend(["--lib", lib])

    try:
        result = subprocess.run(
            cmd,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as err:
        detail = err.stderr or err.stdout or str(err)
        raise RuntimeError(f"failed to mint token:\n{detail}") from err

    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key in {"host", "actor_id", "public_key_b64", "secret_key_b64", "bearer_token"}:
            values[key] = value.strip()

    required = ("host", "actor_id", "public_key_b64", "secret_key_b64", "bearer_token")
    missing = [field for field in required if field not in values]
    if missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            f"minted token output missing field(s): {missing_csv}\nstdout:\n{result.stdout}"
        )

    return SignedBearerToken(
        host=values["host"],
        actor_id=values["actor_id"],
        public_key_b64=values["public_key_b64"],
        secret_key_b64=values["secret_key_b64"],
        bearer_token=values["bearer_token"],
    )
