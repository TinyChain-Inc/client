from __future__ import annotations

import pathlib

import pytest

import tinychain as tc


def test_wasm_install_accepts_single_token_object(tmp_path, monkeypatch):
    calls: list[tuple[str, str, str, str]] = []
    installs: list[object] = []
    closed = []

    class _Kernel:
        def shutdown(self):
            closed.append(True)

        def dispatch(self, request):
            installs.append(request)
            return type("Resp", (), {"status": 204})()

    class _KernelHandle:
        @staticmethod
        def local(*, data_dir: str | None = None, workspace: str | None = None, token=None):
            calls.append(("local", "", data_dir or "", "rjwt"))
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
        alg="falcon512",
        public_key_b64="pubkey",
        bearer_token="token",
    )

    class Example(tc.Library):
        publisher = "example-devco"
        resource_name = "example"
        version = "0.1.0"

    response = tc.install(Example, wasm=wasm_path, data_dir=tmp_path, token=token)

    assert response.status == 204
    assert len(calls) == 1
    assert calls[0][2] == str(tmp_path)
    assert len(installs) == 1
    assert closed == [True]
    assert installs[0].body == b"\x00asm"
    assert ("authorization", "Bearer token") in installs[0].headers


def test_wasm_install_rejects_raw_schema_input(tmp_path):
    wasm_path = pathlib.Path(tmp_path) / "hello.wasm"
    wasm_path.write_bytes(b"\x00asm")

    token = tc.auth.SignedBearerToken(
        host="http://127.0.0.1:8702",
        actor_id="example-admin",
        alg="falcon512",
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


def test_wasm_install_executes_through_native_adapter(tmp_path):
    import json

    from .support import install_token

    class Example(tc.Library):
        publisher = "example-devco"
        resource_name = "native-wasm"
        version = "1.0.0"

        @tc.get
        def answer(self) -> tc.Number:
            ...

    identity = Example.class_id().path
    path = pathlib.Path(__file__).parent / "fixtures" / "native_wasm.wasm"
    token = install_token(identity)
    from tinychain import _local

    with _local.kernel_handle().local(
        data_dir=str(tmp_path / "data"), workspace=str(tmp_path / "workspace"), token=token,
    ) as kernel:
        assert tc.install(Example, wasm=path, kernel=kernel, token=token).status == 204
        with tc.backend(kernel, token=token):
            assert Example().answer() == 42

        response = kernel.dispatch(_local.kernel_request("GET", identity, None, None))
        native = response.body
        clone = native.clone_handle()
        projected = native.value()
        assert clone.value() == projected
        assert "/state/scalar/value/bytes" in json.loads(projected)
