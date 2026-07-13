from __future__ import annotations

import pathlib

import pytest
import tinychain as tc
import tinychain.testing as tc_testing

from tinychain.autodiff import (
    AddOperator,
    ArtifactError,
    AutodiffError,
    DerivativeExecutionDispatcher,
    DerivativeMetadata,
    DerivativeProgram,
    TensorNodeRecord,
    build_derivative_execution_library,
)
from tinychain.library import library_definition
from tests.support import install_token, require_local_tensor_backend, require_tinychain_local


TIMEOUT_SECONDS = 5


def _metadata() -> DerivativeMetadata:
    return DerivativeMetadata(
        source_graph_id="graph",
        transform_version="0.1.0",
        tensor_op_contract_version="0.1.0",
        wrt_signature=("x",),
        seed_contract="seed matches output",
    )


def _program() -> DerivativeProgram:
    return DerivativeProgram(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="gradient",
                operator=AddOperator(),
                op_params={},
                input_value_ids=["seed", "other"],
            )
        ],
        gradients={"x": "gradient"},
        output_gradients=["gradient"],
        metadata=_metadata(),
    )


def test_derivative_execution_library_compiles_as_normal_library() -> None:
    library_cls = build_derivative_execution_library(
        publisher="autodiff-devco",
        class_name="AddDerivativeExecution",
        version="0.1.0",
        program=_program(),
        artifact_class_name="AddDerivativeArtifact",
    )

    definition = library_definition(library_cls)

    assert library_cls.class_id().path == "/lib/autodiff-devco/add_derivative_execution/0.1.0"
    assert definition == {
        "/lib/autodiff-devco/add_derivative_execution/0.1.0": {
            "execute": {
                "/state/scalar/op/post": [
                    ["gradient", {"$seed/add": {"r": {"$other": []}}}],
                    ["result", [{"$gradient": []}]],
                ]
            }
        }
    }


def test_derivative_execution_library_rejects_artifact_identity_collision() -> None:
    with pytest.raises(ArtifactError, match="must not collide"):
        build_derivative_execution_library(
            publisher="autodiff-devco",
            class_name="AddDerivativeArtifact",
            version="0.1.0",
            program=_program(),
            artifact_class_name="AddDerivativeArtifact",
        )


def test_derivative_execution_library_rejects_non_identifier_params() -> None:
    program = DerivativeProgram(
        nodes=[
            TensorNodeRecord(
                node_id="n0",
                output_value_id="gradient",
                operator=AddOperator(),
                op_params={},
                input_value_ids=["seed-value", "other"],
            )
        ],
        gradients={"x": "gradient"},
        output_gradients=["gradient"],
        metadata=_metadata(),
    )

    with pytest.raises(ArtifactError, match="valid Python identifiers"):
        build_derivative_execution_library(
            publisher="autodiff-devco",
            class_name="BadDerivativeExecution",
            version="0.1.0",
            program=program,
        )


def test_real_dispatcher_missing_value_uses_autodiff_error() -> None:
    library_cls = build_derivative_execution_library(
        publisher="autodiff-devco",
        class_name="MissingInputDerivativeExecution",
        version="0.1.0",
        program=_program(),
    )
    dispatcher = DerivativeExecutionDispatcher(
        library_cls=library_cls,
        kernel=object(),
    )

    with pytest.raises(AutodiffError) as error:
        dispatcher.execute(_program(), values={"seed": object()})

    assert error.value.category == "missing_derivative_ir"
    assert "other" in error.value.message


def test_real_dispatcher_executes_installed_route_against_local_backend(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("tinychain_local")
    _, handle = require_tinychain_local(require_library_definition=True)
    _, dense_f64 = require_local_tensor_backend()
    program = _program()
    library_cls = build_derivative_execution_library(
        publisher="autodiff-devco",
        class_name="AddDerivativeRealExecution",
        version="0.1.0",
        program=program,
        artifact_class_name="AddDerivativeArtifact",
    )
    token = install_token(library_cls.class_id().path)
    kernel = handle.local()
    install_response = tc.install(library_cls, kernel=kernel, data_dir=tmp_path, token=token)
    assert install_response.status == 204
    dispatcher = DerivativeExecutionDispatcher(
        library_cls=library_cls,
        kernel=kernel,
    )

    seed = tc.Tensor(native=dense_f64([2], [1.0, 2.0]))
    other = tc.Tensor(native=dense_f64([2], [10.0, 20.0]))

    result = tc_testing.run_with_timeout(
        TIMEOUT_SECONDS,
        lambda: dispatcher.execute(program, values={"seed": seed, "other": other}),
    )

    (gradient,) = result.gradients
    assert isinstance(gradient, tc.Tensor)
    assert gradient.shape == [2]
    assert gradient.values == [11.0, 22.0]
    assert result.metadata == program.metadata
