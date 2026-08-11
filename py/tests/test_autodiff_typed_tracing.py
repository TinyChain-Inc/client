from __future__ import annotations

import gc

import numpy as np
import pytest
import tinychain as tc
from tinychain.autodiff import (
    AddOperator,
    AutodiffError,
    DivOperator,
    ExecutionScheduler,
    MatmulOperator,
    MaxOperator,
    MaxVjpRule,
    MeanOperator,
    MinOperator,
    MinVjpRule,
    MulOperator,
    ProductOperator,
    ProductVjpRule,
    ReshapeOperator,
    SubOperator,
    SumOperator,
    TensorGraphBuilder,
    TransposeOperator,
    captured_operator_types,
    generate,
    get_active_builder,
)
from tinychain.autodiff.finalize import finalize_typed_graph
from tinychain.autodiff.generate import generate as generate_program
from tinychain.autodiff.vjp import default_vjp_registry
from tests.autodiff_execution import NumpyAutodiffDispatcher


def _json(value: object) -> object:
    return tc.state.form_of(value).to_json()


def _symbolic_tensor(name: str) -> tc.Tensor:
    return tc.Tensor(tc.state.TCRef(tc.state.IdRef(name)))


def _assert_category(category: str, operation) -> None:
    with pytest.raises(AutodiffError) as error:
        operation()
    assert error.value.category == category


def _trace_linear_mse(*, dtype: str = "f32"):
    trace = TensorGraphBuilder()
    with tc.state.scoped_context():
        with trace:
            images = trace.input("images", dtype=dtype, shape=(2, 3))
            weights = trace.input("weights", dtype=dtype, shape=(3, 4))
            labels = trace.input("labels", dtype=dtype, shape=(2, 4))
            residual = (images @ weights) - labels
            loss = (residual * residual).mean([0, 1])
    graph = trace.build(outputs=loss)
    return trace, graph, images, weights, labels, residual, loss


def test_typed_tracing_module_boundaries_support_build_and_vjp() -> None:
    assert generate is generate_program

    with TensorGraphBuilder() as trace:
        lhs = trace.input("lhs", dtype="f32", shape=(2, 2))
        rhs = trace.input("rhs", dtype="f32", shape=(2, 2))
        output = lhs * rhs

    graph = trace.build(outputs=output)
    assert finalize_typed_graph(graph) is graph
    assert trace.vjp(output, wrt=[lhs]).gradients


def test_input_creates_named_symbolic_tensor() -> None:
    with TensorGraphBuilder() as trace:
        value = trace.input("features", dtype="f32", shape=(2, "D"))

    assert isinstance(value, tc.Tensor)
    assert _json(value) == {"$features": []}


def test_input_ids_and_metadata_preserve_declaration_order() -> None:
    with TensorGraphBuilder() as trace:
        first = trace.input("first", dtype="f32", shape=(2, 3))
        second = trace.input("second", dtype="f32", shape=(3, 4))
        output = first @ second

    graph = trace.build(outputs=output)
    assert graph.inputs == [
        (trace.value_id(first), {"dtype": "f32", "shape": [2, 3]}),
        (trace.value_id(second), {"dtype": "f32", "shape": [3, 4]}),
    ]


def test_explicit_build_filters_reachable_inputs_in_declaration_order() -> None:
    with TensorGraphBuilder() as trace:
        first = trace.input("first", dtype="f32", shape=(2, 3))
        trace.input("unused", dtype="f32", shape=(2, 3))
        last = trace.input("last", dtype="f32", shape=(2, 3))
        output = last + first

    graph = trace.build(outputs=output)
    assert [value_id for value_id, _ in graph.inputs] == [
        trace.value_id(first),
        trace.value_id(last),
    ]


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"shape": (2, 3)}, TypeError),
        ({"dtype": "f32"}, TypeError),
        ({"dtype": "i32", "shape": (2, 3)}, AutodiffError),
        ({"dtype": "f32", "shape": (-1, 3)}, ValueError),
        ({"dtype": "f32", "shape": (True, 3)}, ValueError),
        ({"dtype": "f32", "shape": ("N-D", 3)}, ValueError),
        ({"dtype": "f32", "shape": "N"}, ValueError),
    ],
)
def test_input_dtype_and_shape_validation(kwargs: dict[str, object], error_type: type[Exception]) -> None:
    with TensorGraphBuilder() as trace:
        with pytest.raises(error_type):
            trace.input("value", **kwargs)


@pytest.mark.parametrize(
    ("name", "error_type"),
    [(None, TypeError), ("", TypeError), ("not-valid", ValueError), ("class", ValueError)],
)
def test_invalid_input_names_fail(name: object, error_type: type[Exception]) -> None:
    with TensorGraphBuilder() as trace:
        with pytest.raises(error_type):
            trace.input(name, dtype="f32", shape=(2, 3))


def test_duplicate_input_name_fails_clearly() -> None:
    with TensorGraphBuilder() as trace:
        trace.input("value", dtype="f32", shape=(2, 3))
        with pytest.raises(ValueError, match="duplicate.*value"):
            trace.input("value", dtype="f32", shape=(2, 3))


def test_typed_sub_records_broadcast_metadata() -> None:
    with TensorGraphBuilder() as trace:
        lhs = trace.input("lhs", dtype="f32", shape=(2, 3))
        rhs = trace.input("rhs", dtype="f32", shape=(1, 3))
        output = lhs - rhs

    node = trace.build(outputs=output).nodes[0]
    assert isinstance(node.operator, SubOperator)
    assert node.output_typespec == {"dtype": "f32", "shape": [2, 3]}


def test_typed_mul_preserves_repeated_input_identity() -> None:
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype="f64", shape=(2, 3))
        output = value * value

    node = trace.build(outputs=output).nodes[0]
    assert isinstance(node.operator, MulOperator)
    assert node.input_value_ids == [trace.value_id(value), trace.value_id(value)]
    assert node.output_typespec == {"dtype": "f64", "shape": [2, 3]}


@pytest.mark.parametrize(
    ("axes", "keepdims", "expected_axes", "expected_shape"),
    [([-1, 0], False, [2, 0], [3]), ([0, 2], True, [0, 2], [1, 3, 1]), ([0, 1, 2], False, [0, 1, 2], [])],
)
def test_typed_mean_normalizes_metadata(axes, keepdims, expected_axes, expected_shape) -> None:
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype="f64", shape=(2, 3, 4))
        output = value.mean(axes, keepdims=keepdims)

    node = trace.build(outputs=output).nodes[0]
    assert isinstance(node.operator, MeanOperator)
    assert node.op_params == {"axes": expected_axes, "keepdims": keepdims}
    assert node.output_typespec == {"dtype": "f64", "shape": expected_shape}


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "output_shape"),
    [((2, 3), (3, 4), [2, 4]), ((5, 2, 3), (1, 3, 4), [5, 2, 4])],
)
def test_typed_matmul_infers_rank2_and_batched_metadata(lhs_shape, rhs_shape, output_shape) -> None:
    with TensorGraphBuilder() as trace:
        lhs = trace.input("lhs", dtype="f32", shape=lhs_shape)
        rhs = trace.input("rhs", dtype="f32", shape=rhs_shape)
        output = lhs @ rhs

    node = trace.build(outputs=output).nodes[0]
    assert isinstance(node.operator, MatmulOperator)
    assert node.output_typespec == {"dtype": "f32", "shape": output_shape}


def test_symbolic_bindings_are_consistent_across_operations() -> None:
    with TensorGraphBuilder() as trace:
        symbolic = trace.input("symbolic", dtype="f32", shape=("N", 3))
        first = trace.input("first", dtype="f32", shape=(2, 3))
        conflicting = trace.input("conflicting", dtype="f32", shape=(4, 3))
        symbolic + first

        with pytest.raises(AutodiffError) as error:
            symbolic * conflicting

    assert error.value.category == "symbolic_shape_mismatch"
    assert len(trace.build().nodes) == 1


def test_matmul_shares_symbol_bindings_between_inner_and_batch_dimensions() -> None:
    with TensorGraphBuilder() as trace:
        lhs = trace.input("lhs", dtype="f32", shape=("N", 2, "N"))
        rhs = trace.input("rhs", dtype="f32", shape=(4, 3, 5))

        with pytest.raises(AutodiffError) as error:
            lhs @ rhs

    assert error.value.category == "symbolic_shape_mismatch"
    assert trace.build().nodes == []


def test_add_and_transpose_capture_with_metadata() -> None:
    with tc.state.scoped_context():
        with TensorGraphBuilder() as trace:
            lhs = trace.input("lhs", dtype="f32", shape=(2, 3))
            rhs = trace.input("rhs", dtype="f32", shape=(1, 3))
            added = lhs + rhs
            output = added.transpose([1, 0])

    nodes = trace.build(outputs=output).nodes
    assert [type(node.operator) for node in nodes] == [AddOperator, TransposeOperator]
    assert [node.output_typespec for node in nodes] == [
        {"dtype": "f32", "shape": [2, 3]},
        {"dtype": "f32", "shape": [3, 2]},
    ]
    assert nodes[1].op_params == {"perm": [1, 0]}


@pytest.mark.parametrize("permutation", ([1.0, 0.0], [True, 0]))
def test_typed_transpose_rejects_non_integer_axes(permutation: list[object]) -> None:
    def perform() -> None:
        with TensorGraphBuilder() as trace:
            value = trace.input("value", dtype="f32", shape=(2, 3))
            value.transpose(permutation)

    _assert_category("invalid_permutation", perform)


def test_typed_transpose_rejects_runtime_permutation_categorically() -> None:
    permutation = tc.state.tuple_of([1, 0])

    def perform() -> None:
        with TensorGraphBuilder() as trace:
            value = trace.input("value", dtype="f32", shape=(2, 3))
            value.transpose(permutation)

    _assert_category("invalid_permutation", perform)


def test_linear_mse_forward_graph_has_complete_ordered_metadata() -> None:
    trace, graph, images, weights, labels, _, loss = _trace_linear_mse()
    assert [type(node.operator) for node in graph.nodes] == [
        MatmulOperator,
        SubOperator,
        MulOperator,
        MeanOperator,
    ]
    assert graph.inputs == [
        (trace.value_id(images), {"dtype": "f32", "shape": [2, 3]}),
        (trace.value_id(weights), {"dtype": "f32", "shape": [3, 4]}),
        (trace.value_id(labels), {"dtype": "f32", "shape": [2, 4]}),
    ]
    assert [node.output_typespec for node in graph.nodes] == [
        {"dtype": "f32", "shape": [2, 4]},
        {"dtype": "f32", "shape": [2, 4]},
        {"dtype": "f32", "shape": [2, 4]},
        {"dtype": "f32", "shape": []},
    ]
    assert graph.outputs == [trace.value_id(loss)]


def test_direct_generate_and_trace_vjp_are_structurally_equivalent_and_ordered() -> None:
    trace, graph, images, weights, _, _, loss = _trace_linear_mse()
    direct = generate(
        graph,
        trace.value_id(loss),
        [trace.value_id(weights), trace.value_id(images)],
        "seed",
        seed_typespec={"dtype": "f32", "shape": []},
    )
    convenient = trace.vjp(loss, wrt=[weights, images])

    assert direct.to_dict() == convenient.to_dict()
    assert list(convenient.gradients) == [trace.value_id(weights), trace.value_id(images)]
    assert convenient.metadata.wrt_signature == (trace.value_id(weights), trace.value_id(images))
    assert convenient.gradients[trace.value_id(weights)]


def test_linear_mse_weight_vjp_owns_transpose_matmul_path() -> None:
    trace, _, images, weights, _, _, loss = _trace_linear_mse()
    program = trace.vjp(loss, wrt=[weights])
    operator_types = [type(node.operator) for node in program.nodes]

    assert TransposeOperator in operator_types
    assert MatmulOperator in operator_types
    weight_gradient_id = program.gradients[trace.value_id(weights)]
    weight_node = next(node for node in program.nodes if node.output_value_id == weight_gradient_id)
    assert isinstance(weight_node.operator, MatmulOperator)
    assert any(
        isinstance(node.operator, TransposeOperator) and node.output_value_id in weight_node.input_value_ids
        for node in program.nodes
    )
    assert trace.value_id(images) in {vid for node in program.nodes for vid in node.input_value_ids}


def test_linear_mse_weight_vjp_matches_numpy_formula() -> None:
    trace, _, images, weights, labels, residual, loss = _trace_linear_mse()
    program = trace.vjp(loss, wrt=[weights])
    image_values = np.arange(6, dtype=np.float32).reshape(2, 3) / 5
    weight_values = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    label_values = np.arange(8, dtype=np.float32).reshape(2, 4) / 9

    result = ExecutionScheduler(NumpyAutodiffDispatcher()).execute(
        program,
        values={
            "seed": np.asarray(1.0, dtype=np.float32),
            trace.value_id(images): image_values,
            trace.value_id(weights): weight_values,
            trace.value_id(labels): label_values,
            trace.value_id(residual): image_values @ weight_values - label_values,
        },
    )
    expected = (2 / (2 * 4)) * image_values.T @ (image_values @ weight_values - label_values)
    np.testing.assert_allclose(result.gradients[0], expected, rtol=1e-5, atol=1e-6)


def test_equivalent_traces_have_same_content_hash() -> None:
    first = _trace_linear_mse()
    second = _trace_linear_mse()
    first_program = first[0].vjp(first[-1], wrt=[first[3]])
    second_program = second[0].vjp(second[-1], wrt=[second[3]])
    assert first_program.metadata.source_graph_id == second_program.metadata.source_graph_id


def test_build_outputs_deduplicates_in_first_occurrence_order() -> None:
    with TensorGraphBuilder() as trace:
        x = trace.input("x", dtype="f32", shape=(2, 2))
        y = trace.input("y", dtype="f32", shape=(2, 2))
        a = x + y
        b = x - y
        c = x * y

    graph = trace.build(outputs=(a, b, a, c, b))
    assert graph.outputs == [trace.value_id(a), trace.value_id(b), trace.value_id(c)]


def test_build_rejects_empty_explicit_outputs_and_infers_the_default_output() -> None:
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype="f32", shape=(2, 2))
        output = value * value

    assert trace.build().outputs == [trace.value_id(output)]
    with pytest.raises(ValueError, match="at least one"):
        trace.build(outputs=[])


@pytest.mark.parametrize("reduce", [False, True])
def test_vjp_seed_metadata_exactly_matches_selected_output(reduce: bool) -> None:
    with tc.state.scoped_context():
        with TensorGraphBuilder() as trace:
            x = trace.input("x", dtype="f64", shape=(2, 3))
            y = trace.input("y", dtype="f64", shape=(2, 3))
            matrix = x * y
            output = matrix.mean([0, 1]) if reduce else matrix

    program = trace.vjp(output, wrt=[x])
    expected = {"dtype": "f64", "shape": [] if reduce else [2, 3]}
    assert program.value_typespecs["seed"] == expected


def test_capture_registry_vjp_registry_parity_with_explicit_classification() -> None:
    captured = captured_operator_types()
    registered = frozenset(default_vjp_registry().supported_types())
    deferred = frozenset({DivOperator, SumOperator, ReshapeOperator})
    intentionally_unsupported = frozenset({MaxOperator, MinOperator, ProductOperator})

    assert captured <= registered
    assert deferred.isdisjoint(captured)
    assert intentionally_unsupported.isdisjoint(captured)
    assert deferred.isdisjoint(intentionally_unsupported)
    assert registered - captured == deferred | intentionally_unsupported
    registry = default_vjp_registry()
    assert isinstance(registry.lookup(MaxOperator()), MaxVjpRule)
    assert isinstance(registry.lookup(MinOperator()), MinVjpRule)
    assert isinstance(registry.lookup(ProductOperator()), ProductVjpRule)
    assert not any(
        isinstance(registry.lookup(operator_type()), (MaxVjpRule, MinVjpRule, ProductVjpRule))
        for operator_type in deferred
    )


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda x, y: x - y, {"$x/sub": {"r": {"$y": []}}}),
        (lambda x, y: x * y, {"$x/mul": {"r": {"$y": []}}}),
        (lambda x, y: x.mean([0, 1]), {"$x/mean": {"axes": [0, 1]}}),
        (lambda x, y: x @ y, {"$x/matmul": {"r": {"$y": []}}}),
        (lambda x, y: x + y, {"$x/add": {"r": {"$y": []}}}),
        (lambda x, y: x.transpose([1, 0]), {"$x/transpose": [[1, 0]]}),
    ],
)
def test_inactive_operations_preserve_symbolic_form(operation, expected) -> None:
    assert get_active_builder() is None
    result = operation(_symbolic_tensor("x"), _symbolic_tensor("y"))
    assert _json(result) == expected
    assert get_active_builder() is None


def test_inactive_mean_return_type_and_transpose_view_metadata() -> None:
    x = _symbolic_tensor("x")
    mean = x.mean([0, 1])
    transposed = x.transpose([1, 0])
    assert isinstance(mean, tc.state.Scalar)
    assert [op.kind for op in transposed.view_spec().ops] == ["transpose"]
    assert transposed.view_spec().ops[0].permutation == (1, 0)


def test_inactive_transpose_allows_runtime_permutation() -> None:
    result = _symbolic_tensor("x").transpose(tc.state.tuple_of([1, 0]))
    assert isinstance(result, tc.Tensor)
    assert get_active_builder() is None


def test_active_builder_is_cleaned_up_after_normal_and_exceptional_exit() -> None:
    assert get_active_builder() is None
    with TensorGraphBuilder() as trace:
        assert get_active_builder() is trace
    assert get_active_builder() is None

    with pytest.raises(RuntimeError, match="boom"):
        with TensorGraphBuilder() as failed:
            assert get_active_builder() is failed
            raise RuntimeError("boom")
    assert get_active_builder() is None


@pytest.mark.parametrize(
    ("category", "lhs_shape", "rhs_shape", "operator"),
    [
        ("broadcast_shape_mismatch", (2, 3), (4, 3), lambda x, y: x + y),
        ("unresolved_symbolic_shape", ("N", 3), ("M", 3), lambda x, y: x - y),
        ("matmul_shape_mismatch", (3,), (3, 4), lambda x, y: x @ y),
        ("matmul_shape_mismatch", (2, 3), (5, 4), lambda x, y: x @ y),
        ("unresolved_symbolic_shape", (2, "K"), ("J", 4), lambda x, y: x @ y),
    ],
)
def test_invalid_elementwise_and_matmul_shapes(category, lhs_shape, rhs_shape, operator) -> None:
    def perform() -> None:
        with TensorGraphBuilder() as trace:
            lhs = trace.input("lhs", dtype="f32", shape=lhs_shape)
            rhs = trace.input("rhs", dtype="f32", shape=rhs_shape)
            operator(lhs, rhs)

    _assert_category(category, perform)


def test_non_floating_input_dtype_has_exact_autodiff_category() -> None:
    with TensorGraphBuilder() as trace:
        with pytest.raises(AutodiffError) as error:
            trace.input("value", dtype="i32", shape=(2, 3))

    assert error.value.category == "dtype_not_differentiable"


def test_mixed_traced_dtypes_fail() -> None:
    def perform() -> None:
        with TensorGraphBuilder() as trace:
            lhs = trace.input("lhs", dtype="f32", shape=(2, 3))
            rhs = trace.input("rhs", dtype="f64", shape=(2, 3))
            lhs * rhs

    _assert_category("dtype_mismatch", perform)


@pytest.mark.parametrize(
    ("axes", "shape", "category"),
    [
        (None, (2, 3), "unsupported_reduction"),
        ([2], (2, 3), "reduction_shape_mismatch"),
        ([0, 0], (2, 3), "reduction_shape_mismatch"),
        (["0"], (2, 3), "reduction_shape_mismatch"),
        ([0], ("N", 3), "unresolved_symbolic_shape"),
    ],
)
def test_invalid_mean_axes(axes, shape, category) -> None:
    def perform() -> None:
        with TensorGraphBuilder() as trace:
            value = trace.input("value", dtype="f32", shape=shape)
            value.mean(axes)

    _assert_category(category, perform)


def test_untraced_output_and_wrt_are_rejected() -> None:
    outsider = _symbolic_tensor("outsider")
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype="f32", shape=(2, 3))
        output = value * value

    with pytest.raises(ValueError, match="outsider"):
        trace.build(outputs=outsider)
    with pytest.raises(ValueError, match="outsider"):
        trace.vjp(output, wrt=[outsider])


def test_untyped_operand_fails_typed_finalization() -> None:
    untyped = _symbolic_tensor("untyped")
    with TensorGraphBuilder() as trace:
        typed = trace.input("typed", dtype="f32", shape=(2, 3))
        output = untyped + typed

    _assert_category("missing_dtype_metadata", lambda: trace.build(outputs=output))


def test_untyped_matmul_defers_metadata_inference_to_typed_finalization() -> None:
    untyped = _symbolic_tensor("untyped")
    with TensorGraphBuilder() as trace:
        typed = trace.input("typed", dtype="f32", shape=(3, 4))
        output = untyped @ typed

    node = trace.build().nodes[0]
    assert node.output_typespec is None
    _assert_category("missing_dtype_metadata", lambda: trace.build(outputs=output))


def test_unsupported_intermediate_fails_typed_finalization() -> None:
    with tc.state.scoped_context():
        with TensorGraphBuilder() as trace:
            lhs = trace.input("lhs", dtype="f32", shape=(2, 3))
            rhs = trace.input("rhs", dtype="f32", shape=(2, 3))
            unsupported = lhs / rhs
            output = unsupported * lhs

    with pytest.raises(AutodiffError) as error:
        trace.build(outputs=output)
    assert error.value.category in {"missing_derivative_behavior", "missing_dtype_metadata"}


def test_input_and_vjp_enforce_trace_lifecycle() -> None:
    trace = TensorGraphBuilder()
    with pytest.raises(RuntimeError, match="active trace"):
        trace.input("value", dtype="f32", shape=(2, 2))

    with trace:
        value = trace.input("value", dtype="f32", shape=(2, 2))
        output = value * value
        with pytest.raises(RuntimeError, match="after the trace context"):
            trace.vjp(output, wrt=[value])


def test_vjp_rejects_exceptionally_exited_trace() -> None:
    trace = TensorGraphBuilder()
    with pytest.raises(RuntimeError, match="boom"):
        with trace:
            value = trace.input("value", dtype="f32", shape=(2, 2))
            output = value * value
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="successfully"):
        trace.vjp(output, wrt=[value])


def test_nested_builder_and_same_builder_reentry_are_rejected() -> None:
    trace = TensorGraphBuilder()
    with trace:
        with pytest.raises(RuntimeError, match="Nested"):
            with TensorGraphBuilder():
                pass
    with pytest.raises(RuntimeError, match="single-trace"):
        with trace:
            pass
    assert get_active_builder() is None


def test_independent_sequential_traces_and_repeated_vjp_do_not_mutate_graph() -> None:
    programs = []
    for name in ("x", "y"):
        with TensorGraphBuilder() as trace:
            value = trace.input(name, dtype="f32", shape=(2, 2))
            output = value * value
        before = trace.build(outputs=output)
        first = trace.vjp(output, wrt=[value])
        second = trace.vjp(output, wrt=[value])
        after = trace.build(outputs=output)
        assert before == after
        assert first.to_dict() == second.to_dict()
        programs.append(first)
    assert programs[0].metadata.source_graph_id == programs[1].metadata.source_graph_id
    assert get_active_builder() is None


def test_gc_of_unassigned_intermediate_cannot_corrupt_dataflow_identity() -> None:
    with tc.state.scoped_context():
        with TensorGraphBuilder() as trace:
            images = trace.input("images", dtype="f32", shape=(2, 3))
            weights = trace.input("weights", dtype="f32", shape=(3, 4))
            labels = trace.input("labels", dtype="f32", shape=(2, 4))
            residual = (images @ weights) - labels
            gc.collect()
            temporaries = [object() for _ in range(10_000)]
            output = residual * residual
            assert temporaries

    graph = trace.build(outputs=output)
    matmul, sub, mul = graph.nodes
    assert sub.input_value_ids[0] == matmul.output_value_id
    assert mul.input_value_ids == [sub.output_value_id, sub.output_value_id]
    assert trace.value_id(residual) == sub.output_value_id
    assert trace.value_id(output) == mul.output_value_id


def test_value_id_on_untraced_object_names_role() -> None:
    trace = TensorGraphBuilder()
    outsider = _symbolic_tensor("outsider")
    with pytest.raises(ValueError, match="outsider"):
        trace.value_id(outsider)


def test_duplicate_wrt_is_rejected_with_named_role() -> None:
    with TensorGraphBuilder() as trace:
        weights = trace.input("weights", dtype="f32", shape=(2, 2))
        output = weights * weights

    with pytest.raises(ValueError, match="duplicate.*weights"):
        trace.vjp(output, wrt=[weights, weights])
