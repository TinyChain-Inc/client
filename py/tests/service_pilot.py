"""One process per restart phase; adapted from v1 Service persistence tests."""

import json
import os
import sys
from pathlib import Path

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

    @tc.put
    def insert_then_fail(self, cxt, key, value):
        return tc.after(
            self.data.insert(key, value),
            self.data.insert(["invalid"], [0]),
        )

    @tc.delete
    def delete_then_fail(self, cxt, key):
        return tc.after(
            self.data.delete(key),
            self.data.insert(["invalid"], [0]),
        )

    @tc.post
    def append_then_fail(self, cxt, key, value):
        return tc.after(
            self.data.insert(key, value),
            self.data.insert(["invalid"], [0]),
        )

    @tc.post
    def update_range(self, value):
        return self.data.where(key=slice(1, 3)).update(value=value)

    @tc.delete
    def truncate_range(self, key):
        return self.data.where(key=key).truncate()

    @tc.post
    def update_returned_view(self, value):
        return self.data.where(key=slice(1, 3)).where(key=1).update(value=value)

    @tc.get
    def retained_argument(self, cxt, key):
        return {"table": tc.state.form_of(self.data._get(rtype=tc.collection.Table))}

    @tc.post
    def truncate_argument(self, table: tc.collection.Table):
        return table.truncate()


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

                    if cls is TableExample:
                        with pytest.raises((ValueError, RuntimeError), match="key column 0 expected"):
                            service.insert_then_fail(body=[[4], [4]])
                        assert service.count(key=None) == 3
                        assert tc.execute(service.data.contains([4])) is False

                        with pytest.raises((ValueError, RuntimeError), match="key column 0 expected"):
                            service.delete_then_fail(key=[1])
                        assert service.count(key=None) == 3
                        assert tc.execute(service.data[[1]]) == [1, 1]

                        with pytest.raises((ValueError, RuntimeError), match="key column 0 expected"):
                            service.append_then_fail(key=[4], value=[4])
                        assert service.count(key=None) == 3
                        assert tc.execute(service.data.contains([4])) is False

                        if kernel is not None:
                            response = kernel.dispatch(_local.kernel_request(
                                "POST", cls.class_id().path + "/append",
                                [("authorization", f"Bearer {token.bearer_token}")],
                                _local.state_handle(json.dumps({"key": [4], "value": [4]})),
                            ))
                            # Execution commits before this result is projected;
                            # dropping its request ownership cannot undo the write.
                            assert service.count(key=None) == 4
                            del response
                            assert tc.execute(service.data[[4]]) == [4, 4]
                            tc.execute(service.data.delete([4]))

                        # Both transports must release the failed transaction's
                        # reservations before returning its execution error.
                        tc.execute(service.data.insert([4], [4]))
                        tc.execute(service.data.delete([4]))
                        assert service.count(key=None) == 3

                        assert service.update_range(value=11) is None
                        assert tc.execute(service.data[[1]]) == [1, 11]
                        assert tc.execute(service.data[[2]]) == [2, 11]
                        assert tc.execute(service.data[[3]]) == [3, 3]
                        tc.execute(service.data.insert([5], [5]))
                        assert service.truncate_range(key=5) is None
                        assert tc.execute(service.data.contains([5])) is False
                        with pytest.raises((ValueError, RuntimeError), match="read-only"):
                            service.update_returned_view(value=99)
                        assert tc.execute(service.data[[1]]) == [1, 11]

                        if kernel is not None:
                            headers = [("authorization", f"Bearer {token.bearer_token}")]
                            argument = kernel.dispatch(_local.kernel_request(
                                "GET", cls.class_id().child("retained_argument").path,
                                headers, None,
                            )).body
                            clone = argument.clone_handle()
                            try:
                                # Transfer the native map without projecting its Table.
                                # Cloning or handing off a view cannot restore writes.
                                with pytest.raises(RuntimeError, match="read-only"):
                                    kernel.dispatch(_local.kernel_request(
                                        "POST", cls.class_id().child("truncate_argument").path,
                                        headers, clone,
                                    ))
                                projected = argument.value()
                                rows = projected["table"][str(tc.URI(tc.collection.Table))][1]
                                assert rows == [[1, 11], [2, 11], [3, 3]]
                                assert service.count(key=None) == 3
                            finally:
                                del clone, argument
                else:
                    assert service.count(key=None) == 3
                    if cls is TableExample:
                        assert tc.execute(service.data[[1]]) == [1, 11]
                        assert tc.execute(service.data[[2]]) == [2, 11]
                        assert tc.execute(service.data.contains([5])) is False
                    tc.execute(service.data.delete([3]))
                    assert service.count(key=None) == 2
            with tc.backend(kernel=kernel):
                with pytest.raises((ValueError, RuntimeError), match="(?i)unauthorized|forbidden|permission|403|401"):
                    tc.execute(service.data.insert([99], [99]) if cls is TableExample
                               else service.data.insert([99]))
    finally:
        if kernel is not None:
            kernel.shutdown()
        if process is not None:
            process.terminate()
            _, stderr = process.communicate(timeout=15)
            if process.returncode not in (0, -15):
                raise RuntimeError(stderr)


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2], sys.argv[3], tc.auth.SignedBearerToken(**json.load(sys.stdin)))
