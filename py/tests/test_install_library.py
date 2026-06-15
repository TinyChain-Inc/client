from __future__ import annotations

import pathlib

import tinychain as tc
import tinychain.testing as tc_testing

from .support import install_token


def test_install_python_library(tmp_path: pathlib.Path):
    class Example(tc.Library):
        publisher = "example-devco"
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
    resp = tc.install(
        Example,
        kernel=kernel,
        data_dir=tmp_path,
        token=token,
    )
    assert resp.status == 204

    class Example(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"

        @tc.get
        def hello(self) -> tc.String:
            ...

    stub = Example()
    with tc.backend(kernel, mode="deferred"):
        op = stub.hello()
        response = kernel.dispatch(tc.KernelRequest("GET", op.op.path, None, None))
        assert response.status == 200
        assert tc_testing.decode_json_body(response) == "hello"


def test_install_python_library_string_concat(tmp_path: pathlib.Path):
    class Greeter(tc.Library):
        publisher = "example-devco"
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
    resp = tc.install(
        Greeter,
        kernel=kernel,
        data_dir=tmp_path,
        token=token,
    )
    assert resp.status == 204

    with tc.backend(kernel, token=token):
        assert Greeter().hello("Ada") == "Hello, Ada!"
