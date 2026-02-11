import importlib.util
import json
import pathlib

import os
import tinychain as tc

from .support import REPO_ROOT, ensure_wasm_example_built, rjwt_install_token


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

    token = rjwt_install_token(
        tc.uri("lib", "example-devco", "example", "0.1.0").path
    )
    os.environ["TC_TOKEN_HOST"] = token["host"]
    os.environ["TC_ACTOR_ID"] = token["actor_id"]
    os.environ["TC_PUBLIC_KEY_B64"] = token["public_key_b64"]

    response = INSTALL_WASM.install(
        SCHEMA_PATH,
        wasm_path,
        data_dir=data_dir,
        bearer_token=token["bearer_token"],
    )
    assert response.status == 204

    hydrated_kernel = tc.KernelHandle.local(data_dir=str(data_dir))

    schema_path = tc.uri("lib", "example-devco", "example", "0.1.0").path
    schema_response = hydrated_kernel.dispatch(
        tc.KernelRequest("GET", schema_path, None, None)
    )
    assert schema_response.status == 200
    schema_json = tc.testing.decode_json_body(schema_response)

    hello_path = tc.uri("lib", "example-devco", "example", "0.1.0", "hello").path
    route_response = hydrated_kernel.dispatch(
        tc.KernelRequest("GET", hello_path, None, tc.StateHandle("world"))
    )
    assert route_response.status == 200
    assert tc.testing.decode_json_body(route_response) == "Hello, world!"

    lib_rel = schema_json["id"].lstrip("/").split("/")
    lib_path = data_dir.joinpath(*lib_rel)
    assert (lib_path / "schema.json").exists()
    assert (lib_path / "library.wasm").exists()
