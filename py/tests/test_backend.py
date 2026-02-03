import json

import tinychain as tc


def _failing_stub(_request):
    raise AssertionError("python stub should not handle /healthz requests")


def test_backend_healthz_routes_to_rust_handler():
    backend = tc.Backend(_failing_stub, _failing_stub, None)

    # Should return without running the python stubs because the Rust health handler responds
    backend.healthz()


def test_kernel_handle_installs_library_via_rust_handlers():
    initial_schema = json.dumps(
        {"id": "/lib/hello", "version": "0.1.0", "dependencies": []}
    )
    kernel = tc.KernelHandle.with_library_schema(initial_schema)

    get_request = tc.KernelRequest("GET", "/lib", None, None)
    response = kernel.dispatch(get_request)
    assert response.status == 200
    assert tc.testing.decode_json_body(response)["version"] == "0.1.0"

    updated_schema = json.dumps(
        {"id": "/lib/hello", "version": "0.2.0", "dependencies": []}
    )
    put_request = tc.KernelRequest(
        "PUT", "/lib", None, tc.StateHandle(updated_schema)
    )
    put_response = kernel.dispatch(put_request)
    assert put_response.status == 204

    response_after = kernel.dispatch(tc.KernelRequest("GET", "/lib", None, None))
    assert response_after.status == 200
    assert tc.testing.decode_json_body(response_after)["version"] == "0.2.0"
