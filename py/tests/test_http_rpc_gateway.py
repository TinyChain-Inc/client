import json

import pytest

import tinychain as tc
import tinychain.testing as tc_testing


def test_pyo3_kernel_resolves_opref_over_http_gateway(tmp_path):
    if not tc_testing.cargo_available():
        pytest.skip("`cargo` not found; install Rust tooling to run this test")
    try:
        _ = tc.KernelHandle.with_library_definition
    except (ImportError, AttributeError):
        pytest.skip("`tinychain-local` not installed; skipping PyO3 kernel gateway test")

    proc, addr = tc_testing.start_rust_example(
        "http_rpc_native_host",
        args=("--bind=127.0.0.1:0",),
        prefer_binary=False,
    )
    try:
        b_root = tc.uri("lib", "example-devco", "example", "0.1.0").path
        b_hello = tc.uri("lib", "example-devco", "example", "0.1.0", "hello").path

        class Local(tc.Library):
            publisher = "example-devco"
            version = "0.1.0"
            dependencies = (tc.URI.parse(f"http://{addr}{b_root}"),)

        kernel = tc.kernel.with_library(Local(), data_dir=tmp_path)

        # Control check: the remote route itself is reachable and returns the expected value.
        host = tc.Host(f"http://{addr}")
        assert host.execute(tc.opref.get(b_hello, body="World")) == "Hello, World!"
        assert host.execute(tc.String(tc.opref.get(b_hello, body="World"))) == "Hello, World!"

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


def test_kernel_with_library_does_not_read_auth_env(monkeypatch, tmp_path):
    calls: list[tuple[list[tuple[str, str]] | None, object | None, str | None]] = []

    class _KernelHandle:
        @staticmethod
        def with_library_definition(_definition_json, *, routes=None, token=None, data_dir=None):
            calls.append(
                (
                    routes,
                    token,
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
        version = "0.1.0"
        dependencies = (remote_dep,)

    local = Local()
    kernel = tc.kernel.with_library(local, data_dir=tmp_path)
    assert kernel == "kernel-handle"
    assert calls == [
        (
            [("/lib/example-devco/example/0.1.0", "api.example.test")],
            None,
            str(tmp_path),
        )
    ]
