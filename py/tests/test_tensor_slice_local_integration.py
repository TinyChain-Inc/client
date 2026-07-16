from __future__ import annotations

import pathlib

import tinychain as tc
import tinychain.testing as tc_testing

from .support import install_token, require_local_tensor_backend


TIMEOUT_SECONDS = 5


class TensorSlice(tc.Library):
    publisher = "example-devco"
    version = "0.1.0"

    @tc.post
    def slice_tensor(self, x: tc.Tensor) -> tc.Tensor:
        return x.slice([(1, 4)])

    @tc.post
    def slice_sum(self, x: tc.Tensor) -> tc.Number:
        return x.slice([(1, 4)]).sum()


class TensorOps(tc.Library):
    publisher = "example-devco"
    version = "0.1.0"

    @tc.post
    def cast_u64(self, x: tc.Tensor) -> tc.Tensor:
        return x.cast("u64")

    @tc.post
    def sum_axis_keepdims(self, x: tc.Tensor) -> tc.Tensor:
        return x.sum(axes=[1], keepdims=True)

    @tc.post
    def transpose_3d(self, x: tc.Tensor) -> tc.Tensor:
        return x.transpose((2, 0, 1))

    @tc.post
    def add_broadcast(self, x: tc.Tensor, y: tc.Tensor) -> tc.Tensor:
        return x + y


def test_tensor_slice_executes_via_local_python_client(tmp_path: pathlib.Path):
    dense_u64, _ = require_local_tensor_backend()

    library = TensorSlice()
    token = install_token(TensorSlice.class_id().path)
    kernel = tc.kernel.with_library(library, data_dir=tmp_path, token=token)
    install = tc.install(TensorSlice, kernel=kernel, data_dir=tmp_path, token=token)
    assert install.status == 204

    with tc.backend(kernel):
        tensor = tc.Tensor(native=dense_u64([5], [10, 20, 30, 40, 50]))

        sliced = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.slice_tensor(tensor))
        assert isinstance(sliced, tc.Tensor)
        assert sliced.shape == [3]
        assert sliced.values == [20, 30, 40]

        total = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.slice_sum(tensor))
        assert total == 90


def test_tensor_ops_execute_via_local_python_client(tmp_path: pathlib.Path):
    dense_u64, dense_f64 = require_local_tensor_backend()

    library = TensorOps()
    token = install_token(TensorOps.class_id().path)
    kernel = tc.kernel.with_library(library, data_dir=tmp_path, token=token)
    install = tc.install(TensorOps, kernel=kernel, data_dir=tmp_path, token=token)
    assert install.status == 204

    with tc.backend(kernel):
        cast_source = tc.Tensor(native=dense_f64([3], [1.0, 2.0, 3.0]))
        casted = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.cast_u64(cast_source))
        assert isinstance(casted, tc.Tensor)
        assert casted.dtype == "u64"
        assert casted.values == [1, 2, 3]

        reduce_source = tc.Tensor(native=dense_f64([2, 2], [1.0, 2.0, 3.0, 4.0]))
        reduced = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.sum_axis_keepdims(reduce_source))
        assert isinstance(reduced, tc.Tensor)
        assert reduced.shape == [2, 1]
        assert reduced.values == [3.0, 7.0]

        transpose_source = tc.Tensor(native=dense_f64([2, 3, 2], [float(v) for v in range(12)]))
        transposed = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.transpose_3d(transpose_source))
        assert isinstance(transposed, tc.Tensor)
        assert transposed.shape == [2, 2, 3]
        assert transposed.values == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0]

        left = tc.Tensor(native=dense_f64([2, 1], [1.0, 2.0]))
        right = tc.Tensor(native=dense_f64([1, 3], [10.0, 20.0, 30.0]))
        broadcast_sum = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.add_broadcast(left, right))
        assert isinstance(broadcast_sum, tc.Tensor)
        assert broadcast_sum.shape == [2, 3]
        assert broadcast_sum.values == [11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
