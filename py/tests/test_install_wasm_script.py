import importlib.util
import json
import pathlib

import pytest

import tinychain as tc

from .support import REPO_ROOT, ensure_wasm_example_built


SCRIPT_PATH = REPO_ROOT / "client" / "py" / "bin" / "install_wasm.py"
SCHEMA_PATH = REPO_ROOT / "tc-server" / "examples" / "library_schema_example.json"


def _load_install_wasm():
    spec = importlib.util.spec_from_file_location("install_wasm", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


INSTALL_WASM = _load_install_wasm()


def _sanitize_id(schema_id: str) -> str:
    return schema_id.lstrip("/").replace("/", "_")


def test_install_wasm_script_registers_routes(tmp_path):
    wasm_path = ensure_wasm_example_built("hello_wasm")
    data_dir = tmp_path / "tc-data"
    data_dir.mkdir()

    response = INSTALL_WASM.install(
        SCHEMA_PATH,
        wasm_path,
        data_dir=data_dir,
    )
    assert response.status == 204

    hydrated_kernel = tc.KernelHandle.local(data_dir=str(data_dir))

    schema_response = hydrated_kernel.dispatch(
        tc.KernelRequest("GET", "/lib", None, None)
    )
    assert schema_response.status == 200
    schema_json = tc.testing.decode_json_body(schema_response)

    hello_path = tc.uri.library(
        publisher="example-devco",
        name="example",
        version="0.1.0",
        path=["hello"],
    ).path
    route_response = hydrated_kernel.dispatch(
        tc.KernelRequest("GET", hello_path, None, tc.StateHandle("world"))
    )
    assert route_response.status == 200
    assert tc.testing.decode_json_body(route_response) == "Hello, world!"

    lib_path = data_dir / "lib" / _sanitize_id(schema_json["id"]) / schema_json["version"]
    assert (lib_path / "schema.json").exists()
    assert (lib_path / "library.wasm").exists()
