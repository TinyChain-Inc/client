import pytest

import tinychain as tc
import tinychain.testing as tc_testing

from .support import install_token


def test_http_host_materializes_library_runtime_ref(tmp_path, monkeypatch):
    if not tc_testing.cargo_available():
        pytest.skip("`cargo` not found; install Rust tooling to run this test")
    monkeypatch.setenv("TC_DATA_DIR", str(tmp_path / "remote-data"))
    monkeypatch.setenv("TC_WORKSPACE", str(tmp_path / "remote-workspace"))

    actor_id = "example-admin"
    secret_key_b64 = tc.auth.generate_actor_secret(actor_id)

    try:
        proc, addr = tc_testing.start_rust_example(
            "http_rpc_native_host",
            args=(
                "--bind=127.0.0.1:0",
                f"--actor-id={actor_id}",
                f"--secret-key-b64={secret_key_b64}",
            ),
        )
    except RuntimeError as err:
        if "Operation not permitted" in str(err):
            pytest.skip("sandbox does not permit launching local Rust host example")
        raise
    try:
        class AuthContext(tc.Library):
            publisher = "example-devco"
            resource_name = "auth-context"
            version = "0.1.0"

            @tc.get
            def context(self) -> tc.Ref:
                return tc.auth.context()

        token = tc.auth.mint_rjwt_token(
            host=f"http://{addr}",
            actor_id=actor_id,
            libs=[AuthContext.class_id()],
            ttl_secs=300,
            secret_key_b64=secret_key_b64,
        )
        host = tc.Host(f"http://{addr}", token=token)

        assert tc.install(AuthContext, remote=host, token=token) is None
        context = host.execute(tc.opref.get(tc.URI(AuthContext.class_id(), "context")))
        assert isinstance(context, dict)
        assert context["principal"].endswith(f"::{actor_id}")
    finally:
        proc.kill()


def test_kernel_with_library_does_not_read_auth_env(monkeypatch, tmp_path):
    calls: list[tuple[object | None, str | None]] = []

    class _KernelHandle:
        @staticmethod
        def local(*, token=None, data_dir=None, workspace=None):
            calls.append((token, data_dir))
            return _KernelHandle()

        def dispatch(self, _request):
            return None

    import tinychain._local as tc_local
    monkeypatch.setattr(tc_local, "kernel_handle", lambda: _KernelHandle)
    monkeypatch.setattr(tc_local, "state_handle", lambda value: value)
    monkeypatch.setattr(tc_local, "kernel_request", lambda *args: args)
    monkeypatch.setenv("TC_TOKEN_HOST", "https://tokens.example.test")
    monkeypatch.setenv("TC_ACTOR_ID", "example-admin")
    monkeypatch.setenv("TC_PUBLIC_KEY_B64", "pubkey")

    remote_dep = tc.URI.parse("https://api.example.test/lib/example-devco/example/0.1.0")

    class Local(tc.Library):
        publisher = "example-devco"
        resource_name = "local"
        version = "0.1.0"
        dependencies = (remote_dep,)

    local = Local()
    token = install_token(Local.class_id().path)
    kernel = tc.kernel.with_library(local, data_dir=tmp_path, token=token)
    assert isinstance(kernel, _KernelHandle)
    assert calls == [(token, str(tmp_path))]
