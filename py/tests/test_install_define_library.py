from __future__ import annotations

import pathlib

import tinychain as tc

from .support import rjwt_install_token

def test_install_python_defined_library(tmp_path: pathlib.Path):
    class Defined(tc.define.Library):
        publisher = "example-devco"
        name = "defined"
        version = "0.1.0"
        dependencies = ()

        @tc.define.get
        def hello(self):
            return "hello"

    defined = Defined()

    token = rjwt_install_token(defined.id().path)
    kernel = tc.KernelHandle.with_library_schema_rjwt(
        defined.schema_json(),
        token["host"],
        token["actor_id"],
        token["public_key_b64"],
        data_dir=str(tmp_path),
    )
    resp = tc.define.install(
        defined,
        kernel=kernel,
        data_dir=tmp_path,
        bearer_token=token["bearer_token"],
    )
    assert resp.status == 204

    class Stub(tc.Library):
        publisher = "example-devco"
        name = "defined"
        version = "0.1.0"

        @tc.define.get
        def hello(self) -> tc.String:
            ...

    stub = Stub()
    with tc.backend(kernel):
        assert tc.execute(stub.hello()) == "hello"
