import pathlib

import pytest
import tinychain as tc
import tinychain._local as tc_local
import tinychain.testing as tc_testing

from .support import install_token


def _failing_stub(_request):
    raise AssertionError("python stub should not handle /healthz requests")


def test_backend_healthz_routes_to_rust_handler():
    backend = tc_local.backend_handle(_failing_stub, _failing_stub, None)

    # Should return without running the python stubs because the Rust health handler responds
    backend.healthz()


def test_kernel_handle_installs_library_via_rust_handlers(tmp_path: pathlib.Path):
    class Hello(tc.Library):
        publisher = "example-devco"
        resource_name = "hello"
        version = "0.1.0"

    class Updated(tc.Library):
        publisher = "example-devco"
        resource_name = "updated"
        version = "0.2.0"

        @tc.get
        def hello(self):
            return "hello"

    token = install_token(Hello.class_id().path, Updated.class_id().path)
    kernel = tc.kernel.with_library(
        Hello(),
        data_dir=tmp_path,
        token=token,
    )

    get_request = tc_local.kernel_request("GET", Hello.class_id().path, None, None)
    response = kernel.dispatch(get_request)
    assert response.status == 200
    assert tc_testing.decode_json_body(response)["version"] == "0.1.0"

    put_response = tc.install(Updated, kernel=kernel, token=token)
    assert put_response.status == 204

    response_after = kernel.dispatch(tc_local.kernel_request("GET", Updated.class_id().path, None, None))
    assert response_after.status == 200
    assert tc_testing.decode_json_body(response_after)["version"] == "0.2.0"


def test_kernel_handle_rejects_unauthorized_library_install(tmp_path: pathlib.Path):
    class Hello(tc.Library):
        publisher = "example-devco"
        resource_name = "hello"
        version = "0.1.0"

    class Updated(tc.Library):
        publisher = "example-devco"
        resource_name = "updated"
        version = "0.2.0"

        @tc.get
        def hello(self):
            return "hello"

    kernel = tc.kernel.with_library(Hello(), data_dir=tmp_path)
    with pytest.raises(ValueError, match="token"):
        tc.install(Updated, kernel=kernel, token=None)
