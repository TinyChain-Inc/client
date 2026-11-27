from __future__ import annotations

import pathlib

import pytest

import tinychain as tc
import tinychain.testing as tc_testing
from .support import install_token, require_local_tensor_backend


TIMEOUT_SECONDS = 5


class TensorSlice(tc.Library):
    publisher = "example-devco"
    resource_name = "tensor_slice"
    version = "0.1.0"

    @tc.post
    def slice_tensor(self, x: tc.Tensor) -> tc.Tensor:
        return x.slice([(1, 4)])

    @tc.post
    def slice_sum(self, x: tc.Tensor) -> tc.Number:
        return x.slice([(1, 4)]).sum()


class TensorOps(tc.Library):
    publisher = "example-devco"
    resource_name = "tensor_ops"
    version = "0.1.0"

    @tc.post
    def identity(self, x: tc.Tensor) -> tc.Tensor:
        return x

    @tc.post
    def dense_exp(self, x: tc.Tensor) -> tc.Tensor:
        return x.as_dense().exp()

    @tc.post
    def mean_with_implicit_zeros(self, x: tc.Tensor) -> tc.Number:
        return x.mean()

    @tc.post
    def cast_u64(self, x: tc.Tensor) -> tc.Tensor:
        return x.cast(tc.uri(tc.U64))

    @tc.post
    def sum_axis_keepdims(self, x: tc.Tensor) -> tc.Tensor:
        return x.sum(axes=[1], keepdims=True)

    @tc.post
    def transpose_3d(self, x: tc.Tensor) -> tc.Tensor:
        return x.transpose((2, 0, 1))

    @tc.post
    def add_broadcast(self, x: tc.Tensor, y: tc.Tensor) -> tc.Tensor:
        return x + y


    @tc.post
    def chained_geometry(self, x: tc.Tensor) -> tc.Tensor:
        return x.reshape([1, 2, 3]).broadcast([2, 2, 3]).slice([1, (0, 2), (0, 3)]).transpose([1, 0]).slice([(0, 3), 1])

    @tc.post
    def chained_write(self, x: tc.Tensor) -> tc.Tensor:
        return tc.after(x.transpose([1, 0]).slice([(1, 3), 1]).write(9), x)

    @tc.post
    def selected_write(self, x: tc.Tensor) -> tc.Tensor:
        return tc.after(x.slice([1, (1, 3)]).write(9), x)

    @tc.post
    def repeated_selection_write(self, x: tc.Tensor) -> tc.Tensor:
        return tc.after(x.slice([(0, 2), (0, 3)]).slice([1, (1, 3)]).write(9), x)

    @tc.post
    def identity_reshape_write(self, x: tc.Tensor) -> tc.Tensor:
        return tc.after(x.reshape([2, 3]).write(9), x)

    @tc.post
    def transformed_selection_write(self, x: tc.Tensor) -> tc.Tensor:
        return tc.after(x.slice([(0, 2), (0, 3)]).transpose([1, 0]).write(9), x)


def test_tensor_slice_executes_via_local_python_client(tmp_path: pathlib.Path):
    dense_u64, _ = require_local_tensor_backend()

    library = TensorSlice()
    token = install_token(TensorSlice.class_id().path)
    kernel = tc.kernel.with_library(library, data_dir=tmp_path, token=token)
    try:
        install = tc.install(TensorSlice, kernel=kernel, data_dir=tmp_path, token=token)
        assert install.status == 204

        with tc.backend(kernel):
            tensor = dense_u64([5], [10, 20, 30, 40, 50])

            sliced = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.slice_tensor(x=tensor))
            assert isinstance(sliced, tc.Tensor)
            assert sliced.shape == [3]
            assert sliced.values == [20, 30, 40]

            total = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.slice_sum(x=tensor))
            assert total == 90
    finally:
        kernel.shutdown()


def test_tensor_ops_execute_via_local_python_client(tmp_path: pathlib.Path):
    dense_u64, dense_f64 = require_local_tensor_backend()

    library = TensorOps()
    token = install_token(TensorOps.class_id().path)
    kernel = tc.kernel.with_library(library, data_dir=tmp_path, token=token)
    try:
        install = tc.install(TensorOps, kernel=kernel, data_dir=tmp_path, token=token)
        assert install.status == 204

        with tc.backend(kernel):
            cast_source = dense_f64([3], [1.0, 2.0, 3.0])
            casted = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.cast_u64(x=cast_source))
            assert isinstance(casted, tc.Tensor)
            assert casted.dtype == "u64"
            assert casted.values == [1, 2, 3]

            reduce_source = dense_f64([2, 2], [1.0, 2.0, 3.0, 4.0])
            reduced = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.sum_axis_keepdims(x=reduce_source))
            assert isinstance(reduced, tc.Tensor)
            assert reduced.shape == [2, 1]
            assert reduced.values == [3.0, 7.0]

            transpose_source = dense_f64([2, 3, 2], [float(v) for v in range(12)])
            transposed = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.transpose_3d(x=transpose_source))
            assert isinstance(transposed, tc.Tensor)
            assert transposed.shape == [2, 2, 3]
            assert transposed.values == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0]

            u64_transpose_source = dense_u64([2, 3, 2], list(range(12)))
            u64_transposed = tc_testing.run_with_timeout(
                TIMEOUT_SECONDS,
                lambda: library.transpose_3d(x=u64_transpose_source),
            )
            assert isinstance(u64_transposed, tc.Tensor)
            assert u64_transposed.shape == [2, 2, 3]
            assert u64_transposed.values == [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]

            left = dense_f64([2, 1], [1.0, 2.0])
            right = dense_f64([1, 3], [10.0, 20.0, 30.0])
            broadcast_sum = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.add_broadcast(x=left, y=right))
            assert isinstance(broadcast_sum, tc.Tensor)
            assert broadcast_sum.shape == [2, 3]
            assert broadcast_sum.values == [11.0, 21.0, 31.0, 12.0, 22.0, 32.0]

            sparse = tc.Tensor.sparse("f64", [4], [([0], 2.0)])
            mean = tc_testing.run_with_timeout(
                TIMEOUT_SECONDS, lambda: library.mean_with_implicit_zeros(x=sparse)
            )
            assert mean == 0.5
            dense = tc_testing.run_with_timeout(
                TIMEOUT_SECONDS, lambda: library.dense_exp(x=sparse)
            )
            assert dense.layout == "dense"
            assert dense.values == pytest.approx([7.38905609893065, 1.0, 1.0, 1.0])
    finally:
        kernel.shutdown()


@pytest.mark.parametrize("dtype", ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64"])
@pytest.mark.parametrize("sparse", [False, True])
def test_all_real_tensor_layouts_round_trip_through_native_host(tmp_path, dtype, sparse):
    library = TensorOps()
    token = install_token(TensorOps.class_id().path)
    kernel = tc.kernel.with_library(library, data_dir=tmp_path, token=token)
    try:
        with tc.backend(kernel):
            source = tc.Tensor.sparse(dtype, [2, 3], [([1, 2], 7)]) if sparse else tc.Tensor.dense(dtype, [2], [0, 7])
            result = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.identity(x=source))
            assert isinstance(result, tc.Tensor)
            assert result.dtype == dtype
            assert result.layout == source.layout
            assert result.shape == source.shape
            assert result.values == source.values
            if sparse:
                assert result.entries == source.entries
                if dtype == "f64":
                    empty = tc.Tensor.sparse(dtype, [2, 3], [])
                    result = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.identity(x=empty))
                    assert result.shape == empty.shape
                    assert result.entries == []
    finally:
        kernel.shutdown()


def test_chained_geometry_and_write_through_retain_operands_on_native_host(tmp_path):
    library = TensorOps()
    token = install_token(TensorOps.class_id().path)
    kernel = tc.kernel.with_library(library, data_dir=tmp_path, token=token)
    try:
        with tc.backend(kernel):
            source = tc.Tensor.dense("f64", [2, 3], list(range(6)))
            result = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.chained_geometry(x=source))
            assert result.shape == [3]
            assert result.values == [3, 4, 5]
            for operation in (
                library.chained_write,
                library.repeated_selection_write,
                library.identity_reshape_write,
                library.transformed_selection_write,
            ):
                with pytest.raises(RuntimeError, match="read-only"):
                    tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: operation(x=source))
            unchanged = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.identity(x=source))
            assert unchanged.values == [0, 1, 2, 3, 4, 5]
            written = tc_testing.run_with_timeout(TIMEOUT_SECONDS, lambda: library.selected_write(x=source))
            assert written.values == [0, 1, 2, 3, 9, 9]
    finally:
        kernel.shutdown()
