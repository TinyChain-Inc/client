from __future__ import annotations

import pathlib
import pytest

import tinychain as tc
import tinychain.testing as tc_testing

from .support import REPO_ROOT, ensure_wasm_example_built, require_tinychain_local


ACTOR_ID = "example-admin"


class Example(tc.Library):
    publisher = "example-devco"
    version = "0.1.0"
    dependencies = ()

    @tc.get
    def hello(self, name: str) -> tc.String:
        ...


class A(tc.Library):
    publisher = "example-devco"
    version = "0.1.0"
    dependencies = (Example.class_id(),)

    @tc.get
    def from_b(self, name: str) -> tc.String:
        ...

    @tc.get
    def auth_context(self) -> tc.Ref:
        ...


def test_framework_auth_context_available_in_local_and_wasm_routes(tmp_path: pathlib.Path):
    if not tc_testing.cargo_available():
        pytest.skip("`cargo` not found; install Rust tooling to run auth context integration")
    _, _ = require_tinychain_local(require_library_definition=True)

    wasm_path = ensure_wasm_example_built("opref_to_remote")
    secret_key_b64 = tc.auth.generate_actor_secret(ACTOR_ID)
    try:
        proc, authority = tc_testing.start_rust_example(
            "http_rpc_native_host",
            args=(
                "--bind=127.0.0.1:0",
                f"--actor-id={ACTOR_ID}",
                "--alg=falcon512",
                f"--secret-key-b64={secret_key_b64}",
            ),
            root=REPO_ROOT,
            prefer_binary=True,
            require_binary=True,
        )
    except RuntimeError as err:
        if "Operation not permitted" in str(err):
            pytest.skip("sandbox does not permit launching local Rust host example")
        raise

    try:
        b = Example(authority=tc.URI.parse(authority))
        a = A()
        a.remote_example = b
        a_root = tc.uri(a).path
        b_root = tc.uri(b).path
        host_link = tc.origin(authority)

        install_token = tc.auth.mint_rjwt_token(
            host=host_link,
            actor_id=ACTOR_ID,
            libs=[a_root],
            ttl_secs=300,
            secret_key_b64=secret_key_b64,
        )
        runtime_token = tc.auth.mint_rjwt_token(
            host=host_link,
            actor_id=ACTOR_ID,
            libs=[b_root, a_root],
            ttl_secs=300,
            secret_key_b64=secret_key_b64,
        )

        data_dir = tmp_path / "tc-data"
        data_dir.mkdir(parents=True, exist_ok=True)

        kernel = tc.kernel.with_library(
            a,
            data_dir=data_dir,
            token=runtime_token,
        )
        install = tc.install(
            a,
            wasm=wasm_path,
            kernel=kernel,
            token=install_token,
        )
        assert install.status == 204

        with tc.backend(kernel, token=runtime_token):
            direct_ctx = tc.execute(tc.auth.context())
            wasm_ctx = a.auth_context()
            assert isinstance(direct_ctx, dict)
            assert isinstance(wasm_ctx, dict)
            assert direct_ctx["principal"].endswith(f"::{ACTOR_ID}")
            assert wasm_ctx["principal"].endswith(f"::{ACTOR_ID}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
