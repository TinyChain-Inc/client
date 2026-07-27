from __future__ import annotations

import base64
import importlib
import importlib.util
import sys
import time
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SignedBearerToken:
    host: str
    actor_id: str
    public_key_b64: str
    bearer_token: str
    alg: str = "falcon512"
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

    return tc.Ref(tc.opref.get(tc.uri("host", "auth", "context").path))


def _rjwt():
    existing = sys.modules.get("rjwt")
    if existing is not None:
        return existing

    if importlib.util.find_spec("rjwt") is None:
        raise RuntimeError(
            "rjwt is required to mint TinyChain bearer tokens. Install the PyO3 package with "
            "`maturin develop --manifest-path deps/rjwt/rjwt-py/Cargo.toml`."
        )

    rjwt = importlib.import_module("rjwt")
    return rjwt


def generate_actor_secret(actor_id: str, *, alg: str = "falcon512") -> str:
    if not actor_id.strip():
        raise ValueError("actor_id must be non-empty")

    rjwt = _rjwt()
    actor = _actor(rjwt, actor_id, None, alg.strip().lower())
    if not hasattr(actor, "private_key_bytes"):
        raise RuntimeError("installed rjwt package does not expose private key export")

    return base64.b64encode(actor.private_key_bytes()).decode("ascii")


def _actor(rjwt, actor_id: str, secret_key_b64: str | None, alg: str):
    if secret_key_b64:
        key = base64.b64decode(secret_key_b64)
        return rjwt.Actor.with_keypair(actor_id, key, alg)

    if alg == "falcon512" and hasattr(rjwt.Actor, "new_falcon512"):
        return rjwt.Actor.new_falcon512(actor_id)

    return rjwt.Actor(actor_id)


def mint_rjwt_token(
    *,
    host: str,
    actor_id: str,
    libs: Sequence[str],
    ttl_secs: int = 3600,
    secret_key_b64: str | None = None,
    alg: str = "falcon512",
) -> SignedBearerToken:
    if not libs:
        raise ValueError("minted token requires at least one `libs` claim")
    if not host.strip():
        raise ValueError("minted token host must be non-empty")
    if not actor_id.strip():
        raise ValueError("minted token actor_id must be non-empty")

    rjwt = _rjwt()
    alg = alg.strip().lower()
    actor = _actor(rjwt, actor_id, secret_key_b64, alg)
    now = time.time()
    claims = {lib: 0o200 for lib in libs}
    token = rjwt.Token(host, now, float(ttl_secs), actor_id, claims)
    signed = actor.sign_token(token)
    secret_key_b64 = secret_key_b64 or (
        base64.b64encode(actor.private_key_bytes()).decode("ascii")
        if hasattr(actor, "private_key_bytes")
        else ""
    )

    return SignedBearerToken(
        host=token.issuer() if hasattr(token, "issuer") else host,
        actor_id=actor_id,
        public_key_b64=base64.b64encode(actor.public_key_bytes()).decode("ascii"),
        alg=alg,
        secret_key_b64=secret_key_b64,
        bearer_token=signed.jwt(),
    )
