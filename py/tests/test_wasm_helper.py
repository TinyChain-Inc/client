from __future__ import annotations

import pathlib
import json

import pytest
import tinychain as tc


def test_wasm_install_accepts_single_token_object(tmp_path, monkeypatch):
    calls: list[tuple[str, str, str, str, str, str]] = []
    installs: list[tuple[dict, str]] = []

    class _Kernel:
        def install_compiled_package(self, package_json, bearer_token):
            installs.append((json.loads(package_json), bearer_token))
            return type("Resp", (), {"status": 204})()

    class _KernelHandle:
        @staticmethod
        def with_library_definition(
            definition_json: str,
            *,
            token=None,
            data_dir: str | None = None,
        ):
            calls.append((definition_json, token, data_dir or "", "rjwt"))
            return _Kernel()

    class _Local:
        KernelHandle = _KernelHandle

        class KernelRequest:
            def __init__(self, method, path, headers, body):
                self.method = method
                self.path = path
                self.headers = headers
                self.body = body

        @staticmethod
        def StateHandle(value):
            return value

    import sys

    monkeypatch.setitem(sys.modules, "tinychain_local", _Local)

    wasm_path = pathlib.Path(tmp_path) / "hello.wasm"
    wasm_path.write_bytes(b"\x00asm")

    token = tc.auth.SignedBearerToken(
        host="http://127.0.0.1:8702",
        actor_id="example-admin",
        public_key_b64="pubkey",
        bearer_token="token",
    )

    class Example(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

    response = tc.install(Example, wasm=wasm_path, data_dir=tmp_path, token=token)

    assert response.status == 204
    assert len(calls) == 1
    assert calls[0][1:3] == (token, str(tmp_path))
    assert len(installs) == 1
    assert installs[0][0]["schema"]["id"] == "/lib/example-devco/example/0.1.0"
    assert installs[0][0]["artifacts"][0]["content_type"] == "application/wasm"
    assert installs[0][1] == "token"


def test_wasm_install_rejects_raw_schema_input(tmp_path):
    wasm_path = pathlib.Path(tmp_path) / "hello.wasm"
    wasm_path.write_bytes(b"\x00asm")

    token = tc.auth.SignedBearerToken(
        host="http://127.0.0.1:8702",
        actor_id="example-admin",
        public_key_b64="pubkey",
        bearer_token="token",
    )

    with pytest.raises(TypeError):
        tc.install(
            {"id": "/lib/example-devco/example/0.1.0", "version": "0.1.0", "dependencies": []},
            wasm=wasm_path,
            data_dir=tmp_path,
            token=token,
        )
