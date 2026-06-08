from __future__ import annotations

import pathlib

import pytest

import tinychain as tc


def test_mint_rjwt_token_parses_output_and_builds_cmd(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _Result:
        stdout = "\n".join(
            [
                "host: http://127.0.0.1:8702",
                "actor_id: example-admin",
                "public_key_b64: pk",
                "secret_key_b64: sk",
                "bearer_token: token",
            ]
        )

    def _fake_run(cmd, cwd, check, capture_output, text):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        assert check is True
        assert capture_output is True
        assert text is True
        return _Result()

    monkeypatch.setattr(tc.auth.testing, "rjwt_install_token_bin", lambda _root=None: pathlib.Path("/tmp/rjwt_install_token"))
    monkeypatch.setattr(tc.auth.subprocess, "run", _fake_run)

    token = tc.auth.mint_rjwt_token(
        host="http://127.0.0.1:8702",
        actor_id="example-admin",
        libs=["/lib/example-devco/a/0.1.0"],
        ttl_secs=300,
        repo_root=tmp_path,
    )

    assert token.host == "http://127.0.0.1:8702"
    assert token.actor_id == "example-admin"
    assert token.public_key_b64 == "pk"
    assert token.secret_key_b64 == "sk"
    assert token.bearer_token == "token"
    assert captured["cwd"] == tmp_path
    assert captured["cmd"] == [
        "/tmp/rjwt_install_token",
        "--host",
        "http://127.0.0.1:8702",
        "--actor",
        "example-admin",
        "--lib",
        "/lib/example-devco/a/0.1.0",
    ]


def test_mint_rjwt_token_accepts_https_host(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _Result:
        stdout = "\n".join(
            [
                "host: https://api.tctest.net/",
                "actor_id: example-admin",
                "public_key_b64: pk",
                "secret_key_b64: sk",
                "bearer_token: token",
            ]
        )

    def _fake_run(cmd, cwd, check, capture_output, text):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(tc.auth.testing, "rjwt_install_token_bin", lambda _root=None: pathlib.Path("/tmp/rjwt_install_token"))
    monkeypatch.setattr(tc.auth.subprocess, "run", _fake_run)

    token = tc.auth.mint_rjwt_token(
        host="https://api.tctest.net",
        actor_id="example-admin",
        libs=["/lib/example-devco/a/0.1.0"],
        repo_root=tmp_path,
    )

    assert token.host == "https://api.tctest.net/"
    assert captured["cmd"][1:3] == ["--host", "https://api.tctest.net"]


def test_mint_rjwt_token_requires_claims():
    with pytest.raises(ValueError):
        tc.auth.mint_rjwt_token(host="http://127.0.0.1:8702", actor_id="a", libs=[])


def test_auth_context_helper_uses_host_route():
    context_ref = tc.auth.context()
    assert isinstance(context_ref, tc.Ref)
    assert context_ref.op.path == tc.uri("host", "auth", "context").path
