from __future__ import annotations

import pathlib

import tinychain as tc


def test_wasm_install_accepts_single_token_object(tmp_path, monkeypatch):
    calls: list[tuple[str, str, str, str, str, str]] = []
    requests: list[tuple[str, str, list[tuple[str, str]], object]] = []

    class _Kernel:
        def dispatch(self, request):
            requests.append((request.method, request.path, list(request.headers), request.body))
            return type("Resp", (), {"status": 204})()

    class _KernelHandle:
        @staticmethod
        def with_library_schema_rjwt(
            schema_json: str,
            token_host: str,
            actor_id: str,
            public_key_b64: str,
            *,
            data_dir: str | None = None,
        ):
            calls.append((schema_json, token_host, actor_id, public_key_b64, data_dir or "", "rjwt"))
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
        secret_key_b64="secret",
        bearer_token="token",
    )

    response = tc.install(
        {"id": "/lib/example-devco/example/0.1.0", "version": "0.1.0", "dependencies": []},
        wasm=wasm_path,
        data_dir=tmp_path,
        token=token,
    )

    assert response.status == 204
    assert len(calls) == 1
    assert calls[0][1:5] == ("http://127.0.0.1:8702", "example-admin", "pubkey", str(tmp_path))
    assert len(requests) == 1
    assert requests[0][0] == "PUT"
    assert requests[0][1] == tc.uri("lib").path
    assert ("authorization", "Bearer token") in requests[0][2]
