from __future__ import annotations

import pathlib

import tinychain as tc
import tinychain._local as tc_local
import tinychain.testing as tc_testing
from .support import install_token


def test_local_composition_retains_nested_native_handles(tmp_path, monkeypatch):
    class Values(tc.Library):
        publisher = "example-devco"
        resource_name = "native-values"
        version = "0.1.0"

        @tc.get
        def value(self) -> tc.Number:
            return 42

        @tc.post
        def echo(self, value: tc.Number) -> tc.Number:
            return value

    token = install_token(Values.class_id().path)
    with tc.kernel.with_library(Values(), data_dir=tmp_path, token=token) as kernel:
        import json

        def forbidden(*_args, **_kwargs):
            raise AssertionError("unexpected intermediate JSON conversion")

        monkeypatch.setattr(json, "dumps", forbidden)
        monkeypatch.setattr(json, "loads", forbidden)
        path = tc.URI(Values.class_id(), "value").path
        response = kernel.dispatch(tc_local.kernel_request("GET", path, None, None))
        with tc.backend(kernel):
            result = tc.execute(tc.opref.post(tc.URI(Values.class_id(), "echo"), body={"value": response.body}))
            assert result == 42
        del response


def test_install_python_library(tmp_path: pathlib.Path):
    class Example(tc.Library):
        publisher = "example-devco"
        resource_name = "example"
        version = "0.1.0"
        dependencies = ()

        @tc.get
        def hello(self):
            return "hello"

    token = install_token(Example.class_id().path)
    kernel = tc.kernel.with_library(
        Example(),
        data_dir=tmp_path,
        token=token,
    )
    try:
        resp = tc.install(
            Example,
            kernel=kernel,
            data_dir=tmp_path,
            token=token,
        )
        assert resp.status == 204

        class Example(tc.Library):
            publisher = "example-devco"
            resource_name = "example"
            version = "0.1.0"

            @tc.get
            def hello(self) -> tc.String:
                ...

        stub = Example()
        with tc.backend(kernel, mode="deferred"):
            op = stub.hello()
            response = kernel.dispatch(tc_local.kernel_request("GET", op.op.path, None, None))
            assert response.status == 200
            assert tc_testing.decode_json_body(response) == "hello"
    finally:
        kernel.shutdown()


def test_install_python_library_string_concat(tmp_path: pathlib.Path):
    class Greeter(tc.Library):
        publisher = "example-devco"
        resource_name = "greeter"
        version = "0.1.0"

        @tc.get
        def hello(self, name: str) -> tc.String:
            return tc.String("Hello, {{name}}!").render(name=name)

    token = install_token(Greeter.class_id().path)
    kernel = tc.kernel.with_library(
        Greeter(),
        data_dir=tmp_path,
        token=token,
    )
    try:
        resp = tc.install(
            Greeter,
            kernel=kernel,
            data_dir=tmp_path,
            token=token,
        )
        assert resp.status == 204

        with tc.backend(kernel, token=token):
            assert Greeter().hello(name="Ada") == "Hello, Ada!"
    finally:
        kernel.shutdown()
