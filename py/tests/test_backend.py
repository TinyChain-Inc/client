import json

import tinychain as tc

from .support import rjwt_install_token


def _failing_stub(_request):
    raise AssertionError("python stub should not handle /healthz requests")


def test_backend_healthz_routes_to_rust_handler():
    backend = tc.Backend(_failing_stub, _failing_stub, None)

    # Should return without running the python stubs because the Rust health handler responds
    backend.healthz()


def test_kernel_handle_installs_library_via_rust_handlers():
    token = rjwt_install_token(tc.uri("lib", "hello").path)
    initial_schema = json.dumps(
        {"id": f"{tc.uri('lib', 'hello').path}", "version": "0.1.0", "dependencies": []}
    )
    kernel = tc.KernelHandle.with_library_schema_rjwt(
        initial_schema,
        token["host"],
        token["actor_id"],
        token["public_key_b64"],
    )

    get_request = tc.KernelRequest("GET", tc.uri("lib", "hello").path, None, None)
    response = kernel.dispatch(get_request)
    assert response.status == 200
    assert tc.testing.decode_json_body(response)["version"] == "0.1.0"

    updated_schema = json.dumps(
        {"id": f"{tc.uri('lib', 'hello').path}", "version": "0.2.0", "dependencies": []}
    )
    headers = [("authorization", f"Bearer {token['bearer_token']}")]
    put_request = tc.KernelRequest(
        "PUT", tc.uri("lib").path, headers, tc.StateHandle(updated_schema)
    )
    put_response = kernel.dispatch(put_request)
    assert put_response.status == 204

    response_after = kernel.dispatch(
        tc.KernelRequest("GET", tc.uri("lib", "hello").path, None, None)
    )
    assert response_after.status == 200
    assert tc.testing.decode_json_body(response_after)["version"] == "0.2.0"


def test_kernel_handle_rejects_unauthorized_library_install():
    initial_schema = json.dumps(
        {"id": f"{tc.uri('lib', 'hello').path}", "version": "0.1.0", "dependencies": []}
    )
    kernel = tc.KernelHandle.with_library_schema(initial_schema)

    updated_schema = json.dumps(
        {"id": f"{tc.uri('lib', 'hello').path}", "version": "0.2.0", "dependencies": []}
    )
    put_request = tc.KernelRequest(
        "PUT", tc.uri("lib").path, None, tc.StateHandle(updated_schema)
    )
    put_response = kernel.dispatch(put_request)
    assert put_response.status == 401
