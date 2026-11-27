import pathlib

import pytest

import tinychain as tc
import tinychain._local as tc_local
import tinychain.testing as tc_testing
from .support import install_token


def test_backend_healthz_routes_to_rust_handler(tmp_path: pathlib.Path):
    backend = tc_local.backend_handle(
        data_dir=tmp_path / "data",
        workspace=tmp_path / "workspace",
    )
    try:
        # Should return without running the python stubs because the Rust health handler responds
        backend.healthz()
    finally:
        backend.shutdown()


def test_backend_rejects_unknown_application_content_type(tmp_path: pathlib.Path):
    backend = tc_local.backend_handle(
        data_dir=tmp_path / "data",
        workspace=tmp_path / "workspace",
    )
    try:
        request = tc_local.kernel_request(
            "PUT",
            "/lib",
            [("content-type", "text/plain")],
            tc_local.state_handle("{}"),
        )

        with pytest.raises(RuntimeError, match="unsupported application content type") as error:
            backend.dispatch(request)
        assert error.value.code == "bad_request"
    finally:
        backend.shutdown()


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
    try:
        get_request = tc_local.kernel_request("GET", Hello.class_id().path, None, None)
        response = kernel.dispatch(get_request)
        assert response.status == 200
        assert tc_testing.decode_json_body(response) == {}

        put_response = tc.install(Updated, kernel=kernel, token=token)
        assert put_response.status == 204

        response_after = kernel.dispatch(tc_local.kernel_request("GET", Updated.class_id().path, None, None))
        assert response_after.status == 200
        assert "hello" in tc_testing.decode_json_body(response_after)
    finally:
        kernel.shutdown()


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

    kernel = tc.kernel.with_library(
        Hello(), data_dir=tmp_path, token=install_token(Hello.class_id().path)
    )
    try:
        with pytest.raises(ValueError, match="token"):
            tc.install(Updated, kernel=kernel, token=None)
    finally:
        kernel.shutdown()


def test_local_body_limits_and_canonical_definition_errors(tmp_path):
    with tc_local.backend_handle(
        data_dir=tmp_path / "data", workspace=tmp_path / "workspace",
        max_request_bytes_unauth=4,
    ) as kernel:
        for body in ('"1234"', b'"1234"'):
            request = tc_local.kernel_request("GET", "/state", None, tc_local.state_handle(body))
            with pytest.raises(RuntimeError) as error:
                kernel.dispatch(request)
            assert error.value.code == "payload_too_large"

        for root in ("/lib", "/class", "/service"):
            for body in ("{}", '["/lib/example/item/1.0.0",{}]', "{"):
                request = tc_local.kernel_request("PUT", root, None, tc_local.state_handle(body))
                with pytest.raises(RuntimeError) as error:
                    kernel.dispatch(request)
                assert error.value.code == "bad_request"

    kernel.shutdown()
    with pytest.raises(RuntimeError) as error:
        kernel.dispatch(tc_local.kernel_request("GET", "/state", None, None))
    assert error.value.code == "temporarily_unavailable"


def test_local_native_results_and_concurrent_shutdown(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor

    class Point(tc.Class, tc.Map):
        publisher = "example-devco"
        resource_name = "point"
        version = "1.0.0"

    class Echo(tc.Library):
        publisher = "example-devco"
        resource_name = "native_handoff"
        version = "1.0.0"
        classes = (Point,)

        @tc.get
        def echo(self, key: tc.String) -> tc.String:
            return key

        @tc.get
        def hello(self) -> tc.String:
            return "hello"

    token = install_token(Echo.class_id().path, Point.class_id().path)
    with tc.kernel.with_library(Echo(), data_dir=tmp_path, token=token) as kernel:
        response = kernel.dispatch(tc_local.kernel_request(
            "GET", Echo.class_id().path + "/hello", None, None,
        ))
        value = response.body
        clone = value.clone_handle()
        assert value.value() == "hello"
        assert clone.value() == "hello"
        request = tc_local.kernel_request(
            "GET", str(tc.URI(Echo.class_id(), "echo")), None, clone,
        )
        for _ in range(2):
            echoed = kernel.dispatch(request)
            assert echoed.body.value() == "hello"
        assert request.path == str(tc.URI(Echo.class_id(), "echo"))
        assert request.body.value() == "hello"

        expired = kernel.dispatch(tc_local.kernel_request(
            "GET", Echo.class_id().path + "/hello", None, None,
        ))
        # Deliberately exhaust the host's three-second request deadline; this
        # waits for expiry, rather than coordinating concurrent operations.
        time.sleep(3.2)
        with pytest.raises(RuntimeError) as error:
            expired.body.value()
        assert error.value.code == "temporarily_unavailable"
        assert clone.value() == "hello"  # Completed results remain reusable.

        # An unpolled response must release ownership when its last handle drops.
        abandoned = kernel.dispatch(tc_local.kernel_request(
            "GET", Echo.class_id().path + "/hello", None, None,
        ))
        del abandoned
        with ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(lambda _: kernel.shutdown(), range(2)))
    kernel.shutdown()


def test_expired_scalar_and_transferred_results_reject_reuse(tmp_path):
    import time

    class Echo(tc.Library):
        publisher = "example-devco"
        resource_name = "native_expiry"
        version = "1.0.0"

        @tc.get
        def echo(self, key: tc.Number) -> tc.Number:
            return key

    number_path = str(tc.URI(Echo.class_id(), "echo"))
    token = install_token(Echo.class_id().path)
    with tc_local.kernel_handle().local(
        data_dir=str(tmp_path / "data"), workspace=str(tmp_path / "workspace"),
        token=token, request_ttl_secs=1,
    ) as kernel:
        tc.install(Echo, kernel=kernel, token=token)
        # Native scalar handles have the same lifetime contract as collection handles.
        source = kernel.dispatch(tc_local.kernel_request(
            "GET", number_path, None, tc_local.state_handle("7"),
        )).body
        clone = source.clone_handle()
        assert source.value() == 7
        derived = kernel.dispatch(tc_local.kernel_request(
            "GET", number_path, None, clone,
        )).body
        assert derived.value() == 7
        unpolled = kernel.dispatch(tc_local.kernel_request(
            "GET", number_path, None, tc_local.state_handle("8"),
        )).body
        # Deliberately cross fixed TTL plus inbound clock-skew allowance. This is
        # an expiry test, not synchronization between concurrent operations.
        time.sleep(4.2)
        for handle in (source, clone, derived, unpolled):
            with pytest.raises(RuntimeError) as error:
                handle.value()
            assert error.value.code == "conflict"
        with pytest.raises(RuntimeError) as error:
            kernel.dispatch(tc_local.kernel_request(
                "GET", number_path, None, derived,
            ))
        assert error.value.code == "conflict"
        del source, clone, derived, unpolled
