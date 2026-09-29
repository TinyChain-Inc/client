"""One process per restart phase; adapted from v1 Service persistence tests."""

import json
import os
from pathlib import Path
import sys

import pytest
import tinychain as tc
from tinychain import _local
from tinychain.testing import start_rust_example

from .test_service import Example


class TableExample(Example):
    resource_name = "table"
    data = tc.chain.Sync(tc.collection.Table(tc.collection.table.Schema(
        [tc.collection.table.Column("key", tc.Number)],
        [tc.collection.table.Column("value", tc.Number)],
    )))


def run(root, transport, phase, token):
    process = None
    kernel = None
    authority = None
    if transport == "local":
        kernel = _local.kernel_handle().local(
            data_dir=str(root / "data"), workspace=str(root / "workspace"), token=token,
        )
    else:
        os.environ["TC_DATA_DIR"] = str(root / "data")
        os.environ["TC_WORKSPACE"] = str(root / "workspace")
        os.environ["RUST_MIN_STACK"] = "33554432"
        process, address = start_rust_example("http_rpc_native_host", args=[
            "--bind=127.0.0.1:8702", f"--actor-id={token.actor_id}",
            f"--secret-key-b64={token.secret_key_b64}",
        ])
        authority = tc.URI.parse(f"http://{address}")
    try:
        for cls in (Example, TableExample):
            service = cls(authority=authority)
            options = {"kernel": kernel} if kernel else {"remote": str(authority)}
            with tc.backend(kernel=kernel, token=token, headers=[("authorization", f"Bearer {token.bearer_token}")]):
                if phase == "write":
                    tc.install(cls, token=token, **options)
                    assert tc.execute(service.label) == "native"
                    key = [1] if cls is TableExample else None
                    assert service.insert(body=[key, [1]]) is None
                    key = [2] if cls is TableExample else None
                    assert service.append(key=key, value=[2]) is None
                    if cls is TableExample:
                        tc.execute(service.data.insert([3], [3]))
                    else:
                        tc.execute(service.data.insert([3]))
                    assert service.count(key=None) == 3
                    tc.install(cls, token=token, **options)
                    assert service.count(key=None) == 3
                    class Conflict(cls):
                        label = "conflicting definition"

                    with pytest.raises((ValueError, RuntimeError), match="(?i)conflict|different|already"):
                        tc.install(Conflict, token=token, **options)
                    assert service.count(key=None) == 3
                else:
                    assert service.count(key=None) == 3
                    tc.execute(service.data.delete([3]))
                    assert service.count(key=None) == 2
            with tc.backend(kernel=kernel):
                with pytest.raises((ValueError, RuntimeError), match="(?i)unauthorized|forbidden|permission|403|401"):
                    tc.execute(service.data.insert([99], [99]) if cls is TableExample
                               else service.data.insert([99]))
    finally:
        if process is not None:
            process.terminate()
            _, stderr = process.communicate(timeout=15)
            if process.returncode not in (0, -15):
                raise RuntimeError(stderr)


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2], sys.argv[3], tc.auth.SignedBearerToken(**json.load(sys.stdin)))
