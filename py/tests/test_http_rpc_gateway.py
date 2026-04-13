import json

import pytest

import tinychain as tc


def test_pyo3_kernel_resolves_opref_over_http_gateway():
    if not tc.testing.cargo_available():
        pytest.skip("`cargo` not found; install Rust tooling to run this test")
    try:
        _ = tc.KernelHandle.local_with_dependency_route
    except (ImportError, AttributeError):
        pytest.skip("`tinychain-local` not installed; skipping PyO3 kernel gateway test")

    proc, addr = tc.testing.start_rust_example(
        "http_rpc_native_host",
        args=("--bind=127.0.0.1:0",),
        prefer_binary=False,
    )
    try:
        b_root = tc.uri("lib", "example-devco", "example", "0.1.0").path
        b_hello = tc.uri("lib", "example-devco", "example", "0.1.0", "hello").path

        kernel = tc.KernelHandle.local_with_dependency_route(b_root, addr)

        # Control check: the remote route itself is reachable and returns the expected value.
        host = tc.Host(f"http://{addr}")
        assert host.execute(tc.OpRef("GET", b_hello, body="World")) == "Hello, World!"
        assert host.execute(tc.String(tc.OpRef("GET", b_hello, body="World"))) == "Hello, World!"

        # The local kernel should not proxy arbitrary dependency paths directly.
        with pytest.raises(ValueError):
            kernel.resolve_get(
                b_hello,
                tc.StateHandle(json.dumps("World").encode("utf-8")),
            )

        with pytest.raises(ValueError):
            kernel.resolve_get(tc.uri("service").path)
    finally:
        proc.kill()


def test_kernel_with_library_uses_dependency_authority_and_auth_env(monkeypatch, tmp_path):
    calls: list[tuple[str, str, str | None, str | None, str | None, str | None]] = []

    class _KernelHandle:
        @staticmethod
        def local_with_dependency_route(
            dep_path: str,
            dep_authority: str,
            *,
            token_host: str | None = None,
            actor_id: str | None = None,
            public_key_b64: str | None = None,
            data_dir: str | None = None,
        ):
            calls.append(
                (
                    dep_path,
                    dep_authority,
                    token_host,
                    actor_id,
                    public_key_b64,
                    data_dir,
                )
            )
            return "kernel-handle"

    monkeypatch.setattr(tc, "KernelHandle", _KernelHandle)
    monkeypatch.setenv("TC_TOKEN_HOST", "https://tokens.example.test")
    monkeypatch.setenv("TC_ACTOR_ID", "example-admin")
    monkeypatch.setenv("TC_PUBLIC_KEY_B64", "pubkey")

    remote_dep = tc.URI.parse("https://api.example.test/lib/example-devco/example/0.1.0")

    class Local(tc.Library):
        publisher = "example-devco"
        name = "a"
        version = "0.1.0"
        dependencies = (remote_dep,)

    local = Local()
    kernel = tc.kernel.with_library(local, data_dir=tmp_path)
    assert kernel == "kernel-handle"
    assert calls == [
        (
            "/lib/example-devco/example/0.1.0",
            "api.example.test",
            "https://tokens.example.test",
            "example-admin",
            "pubkey",
            str(tmp_path),
        )
    ]
