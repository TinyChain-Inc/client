from __future__ import annotations

import pathlib

import tinychain as tc


def test_install_python_defined_library(tmp_path: pathlib.Path):
    class Defined(tc.define.Library):
        @tc.define.get
        def hello(self):
            return "hello"

    defined = Defined(
        publisher="example-devco",
        name="defined",
        version="0.1.0",
        dependencies=(),
    )

    kernel = tc.KernelHandle.local(data_dir=str(tmp_path))
    resp = tc.define.install(defined, kernel=kernel, data_dir=tmp_path)
    assert resp.status == 204

    class Stub(tc.Library):
        @tc.define.get
        def hello(self) -> tc.String:
            ...

    stub = Stub(publisher="example-devco", name="defined", version="0.1.0")
    with tc.backend(kernel):
        assert tc.execute(stub.hello()) == "hello"
