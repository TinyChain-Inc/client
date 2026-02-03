import json

import pytest

import tinychain as tc


def test_pyo3_kernel_resolves_opref_over_http_gateway():
    if not tc.testing.cargo_available():
        pytest.skip("`cargo` not found; install Rust tooling to run this test")

    proc, addr = tc.testing.start_rust_example(
        "http_rpc_native_host",
        args=("--bind=127.0.0.1:0",),
        prefer_binary=False,
    )
    try:
        b_root = tc.uri.library(publisher="example-devco", name="example", version="0.1.0").path
        b_hello = tc.uri.library(
            publisher="example-devco",
            name="example",
            version="0.1.0",
            path=["hello"],
        ).path

        schema = json.dumps(
            {"id": tc.uri.library(publisher="example-devco", name="local", version="0.1.0").path, "version": "0.1.0", "dependencies": [b_root]},
            separators=(",", ":"),
        )
        kernel = tc.KernelHandle.with_library_schema_and_dependency_route(
            schema, b_root, addr
        )

        response = kernel.resolve_get(
            b_hello,
            tc.StateHandle(json.dumps("World")),
            bearer_token="test-token",
        )
        assert response.status == 200
        assert tc.testing.decode_json_body(response) == "Hello, World!"

        with pytest.raises(ValueError):
            kernel.resolve_get(tc.uri.service_root(), bearer_token="test-token")
    finally:
        proc.kill()
