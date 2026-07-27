import json

import pytest

import tinychain as tc
import tinychain.testing as tc_testing

from .support import require_tinychain_local


def test_pyo3_kernel_resolves_opref_over_http_gateway(tmp_path):
    if not tc_testing.cargo_available():
        pytest.skip("`cargo` not found; install Rust tooling to run this test")
    tc_local, _ = require_tinychain_local(require_library_definition=True)

    try:
        proc, addr = tc_testing.start_rust_example(
            "http_rpc_native_host",
            args=("--bind=127.0.0.1:0",),
            prefer_binary=False,
        )
    except RuntimeError as err:
        if "Operation not permitted" in str(err):
            pytest.skip("sandbox does not permit launching local Rust host example")
        raise
    try:
        b_root = tc.URI.of("lib", "example-devco", "example", "0.1.0")
        b_hello = tc.URI.of("lib", "example-devco", "example", "0.1.0", "example-devco", "example", "0.1.0", "hello")

        class Local(tc.Library):
            publisher = "example-devco"
            resource_name = "local"
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
                tc_local.state_handle(json.dumps("World").encode("utf-8")),
            )

        with pytest.raises(ValueError):
            kernel.resolve_get(tc.URI.of("service"))
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

    import tinychain._local as tc_local
    monkeypatch.setattr(tc_local, "kernel_handle", lambda:  _KernelHandle)
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
    kernel = tc.kernel.with_library(local, data_dir=tmp_path)
    assert kernel == "kernel-handle"
    assert calls == [
        (
            [("/lib/example-devco/example/0.1.0", "api.example.test")],
            None,
            str(tmp_path),
        )
    ]
