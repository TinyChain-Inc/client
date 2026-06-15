from __future__ import annotations

import base64
import types

import pytest

import tinychain as tc


class _Signed:
    def jwt(self):
        return "signed.jwt"


class _Actor:
    calls: list[tuple[str, object]] = []

    def __init__(self, actor_id: str):
        self.actor_id = actor_id

    @classmethod
    def new_falcon512(cls, actor_id: str):
        cls.calls.append(("falcon", actor_id))
        return cls(actor_id)

    @classmethod
    def with_keypair(cls, actor_id: str, key: bytes, alg: str):
        cls.calls.append(("keypair", alg, key))
        return cls(actor_id)

    def public_key_bytes(self):
        return b"public"

    def private_key_bytes(self):
        return b"private"

    def sign_token(self, token):
        self.token = token
        return _Signed()


class _Token:
    def __init__(self, host, now, ttl, actor_id, claims):
        self.host = host
        self.ttl = ttl
        self.actor_id = actor_id
        self.claims = claims


def _fake_rjwt():
    _Actor.calls = []
    return types.SimpleNamespace(Actor=_Actor, Token=_Token)


def test_mint_rjwt_token_uses_falcon_by_default(monkeypatch):
    monkeypatch.setattr(tc.auth, "_rjwt", _fake_rjwt)

    token = tc.auth.mint_rjwt_token(
        host="https://api.tctest.net",
        actor_id="example-admin",
        libs=["/lib/example-devco/a/0.1.0"],
        ttl_secs=300,
    )

    assert _Actor.calls == [("falcon", "example-admin")]
    assert token.host == "https://api.tctest.net"
    assert token.actor_id == "example-admin"
    assert token.public_key_b64 == base64.b64encode(b"public").decode("ascii")
    assert token.secret_key_b64 == base64.b64encode(b"private").decode("ascii")
    assert token.alg == "falcon512"
    assert token.bearer_token == "signed.jwt"


def test_mint_rjwt_token_imports_falcon_secret_key(monkeypatch):
    monkeypatch.setattr(tc.auth, "_rjwt", _fake_rjwt)
    secret = base64.b64encode(b"falcon-secret").decode("ascii")

    token = tc.auth.mint_rjwt_token(
        host="http://127.0.0.1:8702",
        actor_id="example-admin",
        libs=["/lib/example-devco/a/0.1.0"],
        secret_key_b64=secret,
    )

    assert _Actor.calls == [("keypair", "falcon512", b"falcon-secret")]
    assert token.secret_key_b64 == secret


def test_mint_rjwt_token_rejects_obsolete_cli_options():
    with pytest.raises(ValueError):
        tc.auth.mint_rjwt_token(
            host="http://127.0.0.1:8702",
            actor_id="a",
            libs=["/lib/example/a/0.1.0"],
            binary="/tmp/rjwt_install_token",
        )


def test_mint_rjwt_token_requires_claims():
    with pytest.raises(ValueError):
        tc.auth.mint_rjwt_token(host="http://127.0.0.1:8702", actor_id="a", libs=[])


def test_auth_context_helper_uses_host_route():
    context_ref = tc.auth.context()
    assert isinstance(context_ref, tc.Ref)
    assert context_ref.op.path == tc.uri("host", "auth", "context").path
