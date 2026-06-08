from __future__ import annotations

import pathlib

import tinychain as tc
import tinychain.testing as tc_testing

from .support import rjwt_install_token

def test_install_python_library(tmp_path: pathlib.Path):
    class Example(tc.Library):
        publisher = "example-devco"
        version = "0.1.0"
        dependencies = ()

        @tc.get
        def hello(self):
            return "hello"

    token = rjwt_install_token(Example.class_id().path)
    kernel = tc.KernelHandle.with_library_schema_rjwt(
        Example.class_schema_json(),
        token["host"],
        token["actor_id"],
        token["public_key_b64"],
        data_dir=str(tmp_path),
    )
    resp = tc.install(
        Example,
        kernel=kernel,
        data_dir=tmp_path,
        bearer_token=token["bearer_token"],
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

    token = rjwt_install_token(Greeter.class_id().path)
    kernel = tc.KernelHandle.with_library_schema_rjwt(
        Greeter.class_schema_json(),
        token["host"],
        token["actor_id"],
        token["public_key_b64"],
        data_dir=str(tmp_path),
    )
    resp = tc.install(
        Greeter,
        kernel=kernel,
        data_dir=tmp_path,
        bearer_token=token["bearer_token"],
    )
    assert resp.status == 204

    with tc.backend(kernel, bearer_token=token["bearer_token"]):
        assert Greeter().hello("Ada") == "Hello, Ada!"
