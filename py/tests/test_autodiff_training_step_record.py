"""Unit tests for the lowering, per-parameter update, and compiled-record stage.

This is the first stage of the training-step compiler whose behaviour is
public: `compile_training_step` composes every earlier stage and returns the
three frozen records of the design's §9.2. These tests pin FR-129-001,
FR-129-008, FR-129-010 to FR-129-015, FR-129-021, Inv-2, Inv-3, Inv-6, Inv-8,
Inv-9, Inv-10, Inv-12, §13.2, and §13.4.

Four conventions govern how the file is written.

* **The collaborators are observed at the module seam.** `compile_training_step`
  reaches `lower_graph`, `lower_derivative_program`, and
  `trace_parameter_update` as attributes of `tinychain.autodiff.training_step`,
  so a test can wrap them with a recorder that delegates to the real
  implementation and still observe what was handed across. That is the only way
  to state Inv-9 as an *identity* claim -- `handlers`, `fusion`, and
  `bind_input` reaching every lowering as the same objects, `is` and not `==`
  -- and the only way to prove one update was traced per parameter with that
  parameter's own declared typed spec. Nothing is faked: every recorder
  delegates to the real function it wraps.

* **The registry is opaque, not numerical.** The handlers here return a value
  that is unhashable and non-iterable and carries no dense-array semantics.
  Numerical equivalence over the seam belongs to the shared reference consumer
  and the end-to-end test; what this file proves is the record's *structure*,
  and an opaque target value is also the standing proof of Inv-3 -- the
  framework carries a consumer's value without hashing, comparing, or iterating
  it.

* **Orderings are asserted as sequences and between all four carriers.** Inv-6
  is a four-way agreement, so `parameters`, `provenance.wrt_signature`,
  `provenance.parameter_names`, and `derivative.selected_outputs` are compared
  against each other rather than each against a literal. The two-parameter
  fixture gives its parameters *different shapes* (`w` is 3x4, `b` is 2x4) so a
  swapped wiring is detectable through the gradient typespecs instead of being
  masked by symmetry.

* **Arity is asserted as an invariance, not a pair of separate facts.** Inv-8
  says the record's shape does not vary with the number of parameters, so the
  one- and two-parameter records are compared field-name for field-name rather
  than asserted about individually, and the absence of a scalar convenience
  field is asserted explicitly.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping
from types import MappingProxyType

import pytest
import tinychain as tc
from tinychain.autodiff import graph as graph_module
from tinychain.autodiff import training_step
from tinychain.autodiff.dependencies import DependencyAnalysis, ValueDependency
from tinychain.autodiff.graph import TensorGraph
from tinychain.autodiff.lowering import (
    LoweredProgram,
    OperationHandlerRegistry,
)
from tinychain.autodiff.protocol import AutodiffError
from tinychain.autodiff.reverse import DerivativeProgram
from tinychain.autodiff.training import SGD, Optimizer


# --------------------------------------------------------------------------
# the stage under test, resolved at call time
#
# Resolved through the module rather than imported by name so each case fails
# on its own line with the name it needs, instead of the whole file collapsing
# into one collection-time ImportError.
# --------------------------------------------------------------------------


def _compile(loss: object, **kwargs: object) -> object:
    return training_step.compile_training_step(loss, **kwargs)


# --------------------------------------------------------------------------
# an opaque consumer target value and a registry that emits it
# --------------------------------------------------------------------------


class _OpaqueTargetValue:
    """A consumer target value the framework may only carry (Inv-3).

    Deliberately unhashable and non-iterable: a framework that put one in a
    set, or unpacked one, fails here rather than silently working.
    """

    __hash__ = None

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id

    def __iter__(self):  # pragma: no cover - the point is that it raises
        raise AssertionError("a consumer target value must never be iterated")

    def __repr__(self) -> str:
        return f"<opaque {self.node_id}>"


_OPERATOR_TYPES = (
    graph_module.AddOperator,
    graph_module.SubOperator,
    graph_module.MulOperator,
    graph_module.DivOperator,
    graph_module.SumOperator,
    graph_module.MeanOperator,
    graph_module.MaxOperator,
    graph_module.MinOperator,
    graph_module.ProductOperator,
    graph_module.ReshapeOperator,
    graph_module.BroadcastOperator,
    graph_module.BroadcastReduceOperator,
    graph_module.MatmulOperator,
    graph_module.TransposeOperator,
)


def _stub_registry(recorder: list | None = None) -> OperationHandlerRegistry:
    """A registry resolving every concrete operator to an opaque value.

    Every operator type is registered, so a lowering never fails closed for a
    reason this file is not testing. When *recorder* is supplied each handler
    call appends its node id, in call order.
    """
    registry = OperationHandlerRegistry()
    for operator_type in _OPERATOR_TYPES:

        class _StubHandler:
            def __init__(self, operator_type: type) -> None:
                self.operator_type = operator_type

            def lower(self, context: object) -> object:
                if recorder is not None:
                    recorder.append(context.node_id)
                return _OpaqueTargetValue(context.node_id)

        registry.register(_StubHandler(operator_type))
    return registry


# --------------------------------------------------------------------------
# declarations and losses
# --------------------------------------------------------------------------


SCALAR_SPEC: Mapping[str, object] = {"dtype": "f32", "shape": []}

ONE_PARAMETER_INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": {"dtype": "f32", "shape": (2, 3)},
    "w": {"dtype": "f32", "shape": (3, 4)},
    "y": {"dtype": "f32", "shape": (2, 4)},
}

TWO_PARAMETER_INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": {"dtype": "f32", "shape": (2, 3)},
    "w": {"dtype": "f32", "shape": (3, 4)},
    "b": {"dtype": "f32", "shape": (2, 4)},
    "y": {"dtype": "f32", "shape": (2, 4)},
}

NO_CAPTURE_INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": {"dtype": "f32", "shape": (2, 3)},
    "w": {"dtype": "f32", "shape": (2, 3)},
}


def _residual_loss(*, x: object, w: object, y: object) -> object:
    """The design's §17.3.1 loss: the multiply's VJP captures the residual."""
    residual = x @ w - y
    return (residual * residual).mean([0, 1])


def _two_parameter_loss(*, x: object, w: object, b: object, y: object) -> object:
    """The same residual, read by two parameters with different shapes."""
    residual = x @ w - y
    return ((residual * residual) + (residual * b)).mean([0, 1])


def _no_capture_loss(*, x: object, w: object) -> object:
    """No VJP on this path reads a forward value, so nothing is captured."""
    return (x + w).mean([0, 1])


def _identity_loss(*, w: object) -> object:
    """A parameter that is also the loss output."""
    return w


def _compile_one_parameter(**overrides: object) -> object:
    """Compile the §17.3.1 declaration, one parameter, with stub handlers."""
    kwargs: dict[str, object] = {
        "inputs": ONE_PARAMETER_INPUTS,
        "parameters": ("w",),
        "optimizer": SGD(),
        "optimizer_inputs": {"learning_rate": SCALAR_SPEC},
        "handlers": _stub_registry(),
    }
    loss = overrides.pop("loss", _residual_loss)
    kwargs.update(overrides)
    with tc.state.scoped_context():
        return _compile(loss, **kwargs)


def _compile_two_parameters(**overrides: object) -> object:
    """Compile the two-parameter declaration; `w` is 3x4 and `b` is 2x4."""
    kwargs: dict[str, object] = {
        "inputs": TWO_PARAMETER_INPUTS,
        "parameters": ("w", "b"),
        "optimizer": SGD(),
        "optimizer_inputs": {"learning_rate": SCALAR_SPEC},
        "handlers": _stub_registry(),
    }
    loss = overrides.pop("loss", _two_parameter_loss)
    kwargs.update(overrides)
    with tc.state.scoped_context():
        return _compile(loss, **kwargs)


# --------------------------------------------------------------------------
# seam recorders
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _SeamCall:
    """One recorded call across a module seam, with its result."""

    name: str
    args: tuple
    kwargs: dict
    result: object = None


class _Seams:
    """Every recorded lowering and update-tracing call, in call order."""

    def __init__(self) -> None:
        self.calls: list[_SeamCall] = []

    @property
    def names(self) -> list[str]:
        return [call.name for call in self.calls]

    def of(self, name: str) -> list[_SeamCall]:
        return [call for call in self.calls if call.name == name]

    @property
    def lowerings(self) -> list[_SeamCall]:
        return [
            call
            for call in self.calls
            if call.name in ("lower_graph", "lower_derivative_program")
        ]


_SEAM_NAMES = ("lower_graph", "lower_derivative_program", "trace_parameter_update")


def _record_seams(monkeypatch: pytest.MonkeyPatch) -> _Seams:
    """Wrap each seam with a recorder that delegates to the real function."""
    seams = _Seams()
    for name in _SEAM_NAMES:
        real = getattr(training_step, name)

        def _wrapper(*args, _name: str = name, _real=real, **kwargs):
            call = _SeamCall(name=_name, args=args, kwargs=dict(kwargs))
            seams.calls.append(call)
            call.result = _real(*args, **kwargs)
            return call.result

        monkeypatch.setattr(training_step, name, _wrapper)
    return seams


_STAGE_NAMES = (
    "validate_declaration",
    "trace_loss",
    "differentiate_loss",
    "generate",
    "_check_seed_against_derivative_program",
    "analyze_source_captures",
    "expand_source_artifacts",
    "lower_graph",
    "lower_derivative_program",
    "trace_parameter_update",
    "expand_update_graph",
)


def _instrument_stages(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failing: str | None = None,
    error: BaseException | None = None,
    occurrence: int = 1,
) -> list[str]:
    """Record every stage call in order, optionally failing one of them.

    Each stage is reached by `compile_training_step` as a module attribute, so
    wrapping it here observes -- and can interrupt -- the real sequence without
    replacing any of it.
    """
    calls: list[str] = []
    seen: dict[str, int] = {}
    for name in _STAGE_NAMES:
        real = getattr(training_step, name)

        def _wrapper(*args, _name: str = name, _real=real, **kwargs):
            calls.append(_name)
            seen[_name] = seen.get(_name, 0) + 1
            if _name == failing and seen[_name] == occurrence:
                raise error
            return _real(*args, **kwargs)

        monkeypatch.setattr(training_step, name, _wrapper)
    return calls


# ==========================================================================
# the entry point's signature (AC 1)
# ==========================================================================


def test_compile_training_step_has_the_declared_signature() -> None:
    signature = inspect.signature(training_step.compile_training_step)
    parameters = list(signature.parameters.values())

    assert [parameter.name for parameter in parameters] == [
        "loss",
        "inputs",
        "parameters",
        "optimizer",
        "optimizer_inputs",
        "handlers",
        "fusion",
        "bind_input",
        "seed_label",
        "forward_expansions",
        "derivative_expansions",
        "update_expansions",
    ]

    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters[1:]
    )


def test_compile_training_step_defaults_are_exactly_the_documented_ones() -> None:
    signature = inspect.signature(training_step.compile_training_step)
    defaults = {
        name: parameter.default for name, parameter in signature.parameters.items()
    }
    empty = inspect.Parameter.empty

    assert defaults["loss"] is empty
    assert defaults["inputs"] is empty
    assert defaults["parameters"] is empty
    assert defaults["optimizer"] is empty
    assert defaults["handlers"] is empty
    assert defaults["optimizer_inputs"] is None
    assert defaults["fusion"] is None
    assert defaults["bind_input"] is None
    assert defaults["seed_label"] == "seed"
    assert defaults["forward_expansions"] == ()
    assert defaults["derivative_expansions"] == ()
    assert defaults["update_expansions"] == ()


# ==========================================================================
# the record's own shape (AC 3, Inv-8)
# ==========================================================================


def test_record_types_are_frozen_dataclasses() -> None:
    record = _compile_one_parameter()

    assert dataclasses.is_dataclass(record)
    assert type(record).__dataclass_params__.frozen is True
    assert dataclasses.is_dataclass(record.provenance)
    assert type(record.provenance).__dataclass_params__.frozen is True
    assert dataclasses.is_dataclass(record.parameters[0])
    assert type(record.parameters[0]).__dataclass_params__.frozen is True


def test_record_exposes_exactly_the_declared_fields() -> None:
    record = _compile_one_parameter()

    assert [field.name for field in dataclasses.fields(record)] == [
        "source_forward_graph",
        "source_derivative_program",
        "lowered_forward_graph",
        "lowered_derivative_program",
        "forward",
        "derivative",
        "input_value_ids",
        "loss_value_id",
        "forward_capture_value_ids",
        "seed_value_ids",
        "parameters",
        "provenance",
    ]
    assert [field.name for field in dataclasses.fields(record.parameters[0])] == [
        "name",
        "value_id",
        "gradient_value_id",
        "source_update_graph",
        "lowered_update_graph",
        "update",
        "update_input_value_ids",
        "updated_parameter_value_id",
    ]
    assert [field.name for field in dataclasses.fields(record.provenance)] == [
        "source_graph_id",
        "transform_version",
        "tensor_op_contract_version",
        "wrt_signature",
        "parameter_names",
        "input_names",
        "seed_value_ids",
        "seed_label",
        "optimizer_label",
        "optimizer_input_names",
        "forward_expansions",
        "derivative_expansions",
        "update_expansions",
    ]


def test_record_carries_no_scalar_per_parameter_convenience_field() -> None:
    """§9.4: one parameter is `len(parameters) == 1` and nothing else."""
    record = _compile_one_parameter()
    field_names = {field.name for field in dataclasses.fields(record)}

    assert field_names.isdisjoint(
        {"parameter", "gradient", "update", "updated_parameter", "gradient_value_id"}
    )
    assert callable(record.parameter)


def test_record_shape_does_not_vary_with_parameter_count() -> None:
    """Inv-8: the field structure for one and two parameters is the same."""
    one = _compile_one_parameter()
    two = _compile_two_parameters()

    def structure(value: object) -> list[str]:
        return [field.name for field in dataclasses.fields(value)]

    assert structure(one) == structure(two)
    assert structure(one.provenance) == structure(two.provenance)
    assert structure(one.parameters[0]) == structure(two.parameters[0])
    assert structure(two.parameters[0]) == structure(two.parameters[1])
    assert isinstance(one.parameters, tuple) and isinstance(two.parameters, tuple)
    assert len(one.parameters) == 1
    assert len(two.parameters) == 2


def test_compiled_artifacts_have_the_declared_types() -> None:
    record = _compile_one_parameter()

    assert isinstance(record.source_forward_graph, TensorGraph)
    assert isinstance(record.lowered_forward_graph, TensorGraph)
    assert isinstance(record.source_derivative_program, DerivativeProgram)
    assert isinstance(record.lowered_derivative_program, DerivativeProgram)
    assert isinstance(record.forward, LoweredProgram)
    assert isinstance(record.derivative, LoweredProgram)
    assert isinstance(record.parameters[0].update, LoweredProgram)
    assert isinstance(record.parameters[0].source_update_graph, TensorGraph)
    assert isinstance(record.parameters[0].lowered_update_graph, TensorGraph)


def test_source_and_lowered_artifacts_are_the_same_objects_without_passes() -> None:
    """Inv-10: with no expansion supplied the stage is inert and observably so."""
    record = _compile_two_parameters()

    assert record.lowered_forward_graph is record.source_forward_graph
    assert record.lowered_derivative_program is record.source_derivative_program
    for parameter in record.parameters:
        assert parameter.lowered_update_graph is parameter.source_update_graph


# ==========================================================================
# selections (AC 2, FR-129-006, FR-129-008)
# ==========================================================================


def test_forward_selection_is_the_loss_followed_by_every_capture() -> None:
    record = _compile_one_parameter()

    assert record.forward_capture_value_ids != ()
    assert record.loss_value_id not in record.forward_capture_value_ids
    assert record.forward.selected_outputs == (
        record.loss_value_id,
    ) + record.forward_capture_value_ids


def test_forward_selection_for_a_loss_with_no_captures_is_the_loss_alone() -> None:
    record = _compile_one_parameter(
        loss=_no_capture_loss, inputs=NO_CAPTURE_INPUTS, parameters=("w",)
    )

    assert record.forward_capture_value_ids == ()
    assert record.forward.selected_outputs == (record.loss_value_id,)


def test_derivative_selection_is_the_gradients_in_parameter_order() -> None:
    record = _compile_two_parameters()

    expected = tuple(
        record.source_derivative_program.gradients[parameter.value_id]
        for parameter in record.parameters
    )
    assert record.derivative.selected_outputs == expected
    assert record.derivative.selected_outputs == tuple(
        record.source_derivative_program.output_gradients
    )


def test_each_update_selects_only_its_own_updated_parameter() -> None:
    record = _compile_two_parameters()

    for parameter in record.parameters:
        assert parameter.update.selected_outputs == (
            parameter.updated_parameter_value_id,
        )


# ==========================================================================
# per-parameter fields (AC 4, FR-129-012)
# ==========================================================================


def test_each_parameter_carries_its_forward_value_id() -> None:
    record = _compile_two_parameters()
    declared_input_ids = {value_id for value_id, _ in record.source_forward_graph.inputs}

    for parameter in record.parameters:
        assert parameter.value_id == record.input_value_ids[parameter.name]
        assert parameter.value_id in declared_input_ids


def test_each_parameter_carries_the_source_gradient_for_its_own_value() -> None:
    record = _compile_two_parameters()

    for parameter in record.parameters:
        assert (
            parameter.gradient_value_id
            == record.source_derivative_program.gradients[parameter.value_id]
        )


def test_update_input_value_ids_are_the_traced_records_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seams = _record_seams(monkeypatch)
    record = _compile_two_parameters()

    traced_updates = seams.of("trace_parameter_update")
    assert len(traced_updates) == 2
    for parameter, call in zip(record.parameters, traced_updates):
        assert dict(parameter.update_input_value_ids) == dict(
            call.result.input_value_ids
        )
        assert parameter.source_update_graph is call.result.graph
        assert parameter.updated_parameter_value_id == call.result.updated_parameter_id


def test_update_input_value_ids_name_parameter_gradient_and_optimizer_inputs() -> None:
    record = _compile_two_parameters()

    for parameter in record.parameters:
        assert set(parameter.update_input_value_ids) == {
            "parameter",
            "gradient",
            "learning_rate",
        }


def test_update_input_value_ids_carry_an_optimizer_with_no_optimizer_inputs() -> None:
    class _Halve(Optimizer):
        required_optimizer_inputs: tuple[str, ...] = ()

        def update(self, *, parameter: object, gradient: object) -> object:
            return parameter - gradient

    record = _compile_one_parameter(optimizer=_Halve(), optimizer_inputs=None)

    assert set(record.parameters[0].update_input_value_ids) == {
        "parameter",
        "gradient",
    }
    assert record.provenance.optimizer_input_names == ()


# ==========================================================================
# the four-way ordering agreement (AC 5, Inv-6)
# ==========================================================================


@pytest.mark.parametrize(
    "compile_case, declared",
    [
        (_compile_one_parameter, ("w",)),
        (_compile_two_parameters, ("w", "b")),
    ],
)
def test_all_four_orderings_agree(compile_case, declared) -> None:
    record = compile_case()

    parameter_names = tuple(parameter.name for parameter in record.parameters)
    parameter_value_ids = tuple(parameter.value_id for parameter in record.parameters)
    gradient_value_ids = tuple(
        parameter.gradient_value_id for parameter in record.parameters
    )

    assert parameter_names == declared
    assert record.provenance.parameter_names == declared
    assert record.provenance.wrt_signature == parameter_value_ids
    assert record.provenance.wrt_signature == tuple(
        record.source_derivative_program.metadata.wrt_signature
    )
    assert record.derivative.selected_outputs == gradient_value_ids


def test_parameter_order_follows_the_argument_not_the_declaration_order() -> None:
    """`b` is declared after `w` but requested first; every carrier follows."""
    record = _compile_two_parameters(parameters=("b", "w"))

    assert tuple(parameter.name for parameter in record.parameters) == ("b", "w")
    assert record.provenance.parameter_names == ("b", "w")
    assert record.provenance.wrt_signature == (
        record.input_value_ids["b"],
        record.input_value_ids["w"],
    )
    assert record.derivative.selected_outputs == tuple(
        parameter.gradient_value_id for parameter in record.parameters
    )
    typespecs = record.source_derivative_program.value_typespecs
    assert tuple(typespecs[record.parameters[0].gradient_value_id]["shape"]) == (2, 4)
    assert tuple(typespecs[record.parameters[1].gradient_value_id]["shape"]) == (3, 4)


def test_each_gradient_lands_on_the_parameter_with_that_shape() -> None:
    """A swap between two parameters of different shapes is detectable."""
    record = _compile_two_parameters()
    typespecs = record.source_derivative_program.value_typespecs

    shapes = {
        parameter.name: tuple(typespecs[parameter.gradient_value_id]["shape"])
        for parameter in record.parameters
    }
    assert shapes == {"w": (3, 4), "b": (2, 4)}


# ==========================================================================
# input bindings (AC 6, FR-129-011)
# ==========================================================================


def test_input_value_ids_cover_every_declared_name_exactly_once() -> None:
    record = _compile_two_parameters()

    assert set(record.input_value_ids) == set(TWO_PARAMETER_INPUTS)
    assert len(record.input_value_ids) == len(TWO_PARAMETER_INPUTS)
    assert len(set(record.input_value_ids.values())) == len(TWO_PARAMETER_INPUTS)
    assert set(record.input_value_ids.values()) == {
        value_id for value_id, _ in record.source_forward_graph.inputs
    }


def test_input_value_ids_carry_each_names_own_declared_typespec() -> None:
    record = _compile_two_parameters()
    typespecs = dict(record.source_forward_graph.inputs)

    for name, spec in TWO_PARAMETER_INPUTS.items():
        declared = typespecs[record.input_value_ids[name]]
        assert tuple(declared["shape"]) == tuple(spec["shape"])


# ==========================================================================
# the by-name accessor (AC 7, FR-129-015)
# ==========================================================================


def test_parameter_accessor_returns_the_matching_record() -> None:
    record = _compile_two_parameters()

    assert record.parameter("w") is record.parameters[0]
    assert record.parameter("b") is record.parameters[1]


def test_parameter_accessor_rejects_an_undeclared_name() -> None:
    record = _compile_two_parameters()

    with pytest.raises(AutodiffError) as excinfo:
        record.parameter("absent")

    assert excinfo.value.category == "invalid_training_declaration"
    assert "absent" in excinfo.value.message


def test_parameter_accessor_rejects_an_input_that_is_not_a_parameter() -> None:
    record = _compile_two_parameters()

    with pytest.raises(AutodiffError) as excinfo:
        record.parameter("x")

    assert excinfo.value.category == "invalid_training_declaration"


# ==========================================================================
# Inv-9: the injection points reach every lowering unmodified
# ==========================================================================


def test_every_lowering_receives_the_identical_injection_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _stub_registry()

    class _NeverFuse:
        lookahead = 2

        def fuse(self, context: object) -> None:
            return None

    fusion = _NeverFuse()

    def bind_input(dependency: ValueDependency) -> object:
        return dependency

    seams = _record_seams(monkeypatch)
    _compile_two_parameters(
        handlers=handlers, fusion=fusion, bind_input=bind_input
    )

    lowerings = seams.lowerings
    assert seams.names == [
        "lower_graph",
        "lower_derivative_program",
        "trace_parameter_update",
        "lower_graph",
        "trace_parameter_update",
        "lower_graph",
    ]
    assert len(lowerings) == 4
    for call in lowerings:
        assert call.kwargs["handlers"] is handlers
        assert call.kwargs.get("fusion") is fusion
        assert call.kwargs.get("bind_input") is bind_input


def test_lowerings_receive_no_injection_when_none_was_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seams = _record_seams(monkeypatch)
    _compile_one_parameter()

    for call in seams.lowerings:
        assert call.kwargs.get("fusion") is None
        assert call.kwargs.get("bind_input") is None


def test_derivative_lowering_receives_the_lowered_forward_graph_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seams = _record_seams(monkeypatch)
    record = _compile_two_parameters()

    call = seams.of("lower_derivative_program")[0]
    assert call.kwargs["forward_graph"] is record.lowered_forward_graph
    assert tuple(call.kwargs["seed_value_ids"]) == record.seed_value_ids
    assert tuple(call.kwargs["outputs"]) == record.derivative.selected_outputs


def test_forward_lowering_selects_the_loss_and_captures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seams = _record_seams(monkeypatch)
    record = _compile_one_parameter()

    call = seams.of("lower_graph")[0]
    lowered = call.args[0] if call.args else call.kwargs["graph"]
    assert lowered is record.lowered_forward_graph
    assert tuple(call.kwargs["outputs"]) == record.forward.selected_outputs


def _rebuild_graph_pass(graph: TensorGraph) -> TensorGraph:
    """An expansion returning an equivalent graph as a *different* object.

    An implementation that lowers the pre-expansion artifact instead of the
    expanded one is indistinguishable while every pass is the identity, so the
    tests that follow the expanded artifact through the seam use this pass
    rather than an identity one.
    """
    return TensorGraph(
        nodes=list(graph.nodes), inputs=list(graph.inputs), outputs=list(graph.outputs)
    )


def test_each_update_lowering_selects_only_that_updates_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seams = _record_seams(monkeypatch)
    record = _compile_two_parameters()

    update_calls = seams.of("lower_graph")[1:]
    assert len(update_calls) == 2
    for call, parameter in zip(update_calls, record.parameters):
        assert list(call.kwargs["outputs"]) == [parameter.updated_parameter_value_id]


def test_the_update_lowering_consumes_the_expanded_update_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seams = _record_seams(monkeypatch)
    record = _compile_one_parameter(update_expansions=(_rebuild_graph_pass,))

    parameter = record.parameters[0]
    assert parameter.lowered_update_graph is not parameter.source_update_graph
    call = seams.of("lower_graph")[1]
    lowered = call.args[0] if call.args else call.kwargs["graph"]
    assert lowered is parameter.lowered_update_graph


def test_the_forward_lowering_consumes_the_expanded_forward_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seams = _record_seams(monkeypatch)
    record = _compile_one_parameter(forward_expansions=(_rebuild_graph_pass,))

    assert record.lowered_forward_graph is not record.source_forward_graph
    call = seams.of("lower_graph")[0]
    lowered = call.args[0] if call.args else call.kwargs["graph"]
    assert lowered is record.lowered_forward_graph


def test_a_non_registry_handlers_argument_raises_handler_contract_violation() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _compile_one_parameter(handlers=object())

    assert excinfo.value.category == "handler_contract_violation"


def test_a_malformed_fusion_hook_raises_handler_contract_violation() -> None:
    class _NoLookahead:
        def fuse(self, context: object) -> None:
            return None

    with pytest.raises(AutodiffError) as excinfo:
        _compile_one_parameter(fusion=_NoLookahead())

    assert excinfo.value.category == "handler_contract_violation"


def test_a_declining_fusion_hook_is_offered_operations_from_all_three_lowerings() -> None:
    offered: list[str] = []

    class _Recording:
        lookahead = 2

        def fuse(self, context: object) -> None:
            offered.append(context.candidates[0].node_id)
            return None

    record = _compile_one_parameter(fusion=_Recording())

    forward_ids = {node.node_id for node in record.lowered_forward_graph.nodes}
    derivative_ids = {node.node_id for node in record.lowered_derivative_program.nodes}
    update_ids = {
        node.node_id for node in record.parameters[0].lowered_update_graph.nodes
    }
    offered_ids = set(offered)

    assert offered_ids & forward_ids
    assert offered_ids & derivative_ids
    assert offered_ids & update_ids


def test_bind_input_returning_the_dependency_unchanged_is_carried_through() -> None:
    seen: list[ValueDependency] = []

    def bind_input(dependency: ValueDependency) -> object:
        seen.append(dependency)
        return dependency

    record = _compile_one_parameter(bind_input=bind_input)

    bound = [
        record.forward.values[dependency.value_id]
        for dependency in record.forward.dependencies.required_inputs
    ]
    assert bound
    for dependency, value in zip(
        record.forward.dependencies.required_inputs, bound
    ):
        assert value is dependency
    assert seen


# ==========================================================================
# per-parameter update tracing (AC 8, §8.7)
# ==========================================================================


def test_one_update_is_traced_per_parameter_with_its_own_typed_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = SGD()
    optimizer_inputs = {"learning_rate": SCALAR_SPEC}
    seams = _record_seams(monkeypatch)

    _compile_two_parameters(optimizer=optimizer, optimizer_inputs=optimizer_inputs)

    calls = seams.of("trace_parameter_update")
    assert len(calls) == 2
    for call, name in zip(calls, ("w", "b")):
        update = call.args[0] if call.args else call.kwargs["update"]
        assert update is optimizer
        expected = TWO_PARAMETER_INPUTS[name]
        assert dict(call.kwargs["parameter"]) == dict(expected)
        assert dict(call.kwargs["gradient"]) == dict(expected)
        assert dict(call.kwargs["optimizer_inputs"]) == dict(optimizer_inputs)


def test_the_two_parameters_are_traced_with_different_declared_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An implementation reusing the first parameter's spec fails here."""
    seams = _record_seams(monkeypatch)
    _compile_two_parameters()

    calls = seams.of("trace_parameter_update")
    shapes = [tuple(call.kwargs["parameter"]["shape"]) for call in calls]
    assert shapes == [(3, 4), (2, 4)]


def test_an_exception_from_the_optimizer_update_body_propagates_unchanged() -> None:
    failure = RuntimeError("optimizer body failed")

    class _Exploding(Optimizer):
        required_optimizer_inputs: tuple[str, ...] = ("learning_rate",)

        def update(
            self, *, parameter: object, gradient: object, learning_rate: object
        ) -> object:
            raise failure

    with pytest.raises(RuntimeError) as excinfo:
        _compile_one_parameter(optimizer=_Exploding())

    assert excinfo.value is failure


# ==========================================================================
# provenance (FR-129-014)
# ==========================================================================


def test_provenance_carries_the_derivative_metadata() -> None:
    record = _compile_two_parameters()
    metadata = record.source_derivative_program.metadata

    assert record.provenance.source_graph_id == metadata.source_graph_id
    assert record.provenance.transform_version == metadata.transform_version
    assert (
        record.provenance.tensor_op_contract_version
        == metadata.tensor_op_contract_version
    )


def test_provenance_carries_the_declaration_seed_and_optimizer() -> None:
    optimizer = SGD()
    record = _compile_two_parameters(optimizer=optimizer, seed_label="upstream")

    assert record.provenance.input_names == tuple(TWO_PARAMETER_INPUTS)
    assert record.provenance.seed_value_ids == record.seed_value_ids
    assert len(record.seed_value_ids) == 1
    assert record.provenance.seed_label == "upstream"
    assert record.provenance.optimizer_label == type(optimizer).__name__
    assert record.provenance.optimizer_input_names == ("learning_rate",)


def test_the_callers_seed_label_reaches_the_differentiation_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§8.3: the label travels with the minted seed, not only into provenance."""
    recorded: list[dict] = []
    real = training_step.differentiate_loss

    def _recorder(**kwargs: object) -> object:
        recorded.append(dict(kwargs))
        return real(**kwargs)

    monkeypatch.setattr(training_step, "differentiate_loss", _recorder)
    _compile_one_parameter(seed_label="upstream")

    assert [call["seed_label"] for call in recorded] == ["upstream"]


def test_the_seed_label_is_never_used_as_a_value_id() -> None:
    record = _compile_one_parameter(seed_label="upstream")

    assert "upstream" not in record.seed_value_ids
    assert record.seed_value_ids[0] != "upstream"


def test_provenance_optimizer_input_names_follow_the_optimizers_declaration() -> None:
    """FR-129-014 reads the names off the optimizer, not off the caller's mapping."""

    class _TwoInput(Optimizer):
        required_optimizer_inputs: tuple[str, ...] = ("beta", "alpha")

        def update(
            self, *, parameter: object, gradient: object, beta: object, alpha: object
        ) -> object:
            return parameter - beta * gradient - alpha * gradient

    record = _compile_one_parameter(
        optimizer=_TwoInput(),
        optimizer_inputs={"alpha": SCALAR_SPEC, "beta": SCALAR_SPEC},
    )

    assert record.provenance.optimizer_input_names == ("beta", "alpha")


def test_provenance_records_no_expansion_labels_when_no_pass_was_supplied() -> None:
    record = _compile_two_parameters()

    assert record.provenance.forward_expansions == ()
    assert record.provenance.derivative_expansions == ()
    assert record.provenance.update_expansions == ()


def _identity_forward_pass(graph: TensorGraph) -> TensorGraph:
    """An inert forward pass; its `__qualname__` is the label §9.1 requires."""
    return graph


def _identity_derivative_pass(program: DerivativeProgram) -> DerivativeProgram:
    return program


def _identity_update_pass(graph: TensorGraph) -> TensorGraph:
    return graph


def test_provenance_records_applied_pass_labels_in_application_order() -> None:
    record = _compile_two_parameters(
        forward_expansions=(_identity_forward_pass,),
        derivative_expansions=(_identity_derivative_pass,),
        update_expansions=(_identity_update_pass,),
    )

    assert record.provenance.forward_expansions == ("_identity_forward_pass",)
    assert record.provenance.derivative_expansions == ("_identity_derivative_pass",)
    assert record.provenance.update_expansions == ("_identity_update_pass",)


def test_update_pass_labels_do_not_multiply_with_the_parameter_count() -> None:
    """Inv-8: the record's shape is independent of how many parameters exist."""
    one = _compile_one_parameter(update_expansions=(_identity_update_pass,))
    two = _compile_two_parameters(update_expansions=(_identity_update_pass,))

    assert one.provenance.update_expansions == ("_identity_update_pass",)
    assert two.provenance.update_expansions == one.provenance.update_expansions


# ==========================================================================
# dependency provenance is reachable, and restated nowhere (FR-129-013)
# ==========================================================================


def test_dependency_provenance_is_reachable_through_the_lowered_programs() -> None:
    record = _compile_two_parameters()

    assert isinstance(record.forward.dependencies, DependencyAnalysis)
    assert isinstance(record.derivative.dependencies, DependencyAnalysis)
    for parameter in record.parameters:
        assert isinstance(parameter.update.dependencies, DependencyAnalysis)

    captured = tuple(
        dependency.value_id
        for dependency in record.derivative.dependencies.forward_captures
    )
    assert captured == record.forward_capture_value_ids


def test_the_record_restates_no_per_value_dependency_provenance() -> None:
    record = _compile_two_parameters()

    def restates(value: object) -> bool:
        if isinstance(value, (DependencyAnalysis, ValueDependency)):
            return True
        if isinstance(value, (tuple, list)):
            return any(restates(item) for item in value)
        return False

    for field in dataclasses.fields(record):
        if field.name in ("forward", "derivative"):
            continue
        assert not restates(getattr(record, field.name))
    for parameter in record.parameters:
        for field in dataclasses.fields(parameter):
            if field.name == "update":
                continue
            assert not restates(getattr(parameter, field.name))


# ==========================================================================
# a failure at any stage leaves no partially assembled record (§13.4)
# ==========================================================================


@pytest.mark.parametrize(
    "stage",
    [
        "trace_loss",
        "generate",
        "_check_seed_against_derivative_program",
        "analyze_source_captures",
        "expand_source_artifacts",
        "trace_parameter_update",
        "expand_update_graph",
    ],
)
def test_a_failure_at_any_stage_raises_and_returns_no_record(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    failure = RuntimeError(f"{stage} failed")
    calls = _instrument_stages(monkeypatch, failing=stage, error=failure)

    with pytest.raises(RuntimeError) as excinfo:
        _compile_one_parameter()

    assert excinfo.value is failure
    assert calls[-1] == stage


def test_a_failure_in_the_forward_lowering_stops_the_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("forward lowering failed")
    calls = _instrument_stages(monkeypatch, failing="lower_graph", error=failure)

    with pytest.raises(RuntimeError) as excinfo:
        _compile_two_parameters()

    assert excinfo.value is failure
    assert "lower_derivative_program" not in calls
    assert "trace_parameter_update" not in calls


def test_a_failure_in_the_derivative_lowering_stops_the_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("derivative lowering failed")
    calls = _instrument_stages(
        monkeypatch, failing="lower_derivative_program", error=failure
    )

    with pytest.raises(RuntimeError) as excinfo:
        _compile_two_parameters()

    assert excinfo.value is failure
    assert "trace_parameter_update" not in calls


def test_a_failure_in_the_second_update_lowering_yields_no_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("second update lowering failed")
    calls = _instrument_stages(
        monkeypatch, failing="lower_graph", error=failure, occurrence=3
    )

    with pytest.raises(RuntimeError) as excinfo:
        _compile_two_parameters()

    assert excinfo.value is failure
    assert calls.count("lower_graph") == 3


def test_the_stage_order_is_the_declared_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _instrument_stages(monkeypatch)
    _compile_two_parameters()

    assert calls == [
        "validate_declaration",
        "trace_loss",
        "differentiate_loss",
        "generate",
        "_check_seed_against_derivative_program",
        "analyze_source_captures",
        "expand_source_artifacts",
        "lower_graph",
        "lower_derivative_program",
        "trace_parameter_update",
        "expand_update_graph",
        "lower_graph",
        "trace_parameter_update",
        "expand_update_graph",
        "lower_graph",
    ]


# ==========================================================================
# Inv-12: no module-level mutable state, and two calls are independent
# ==========================================================================


def _module_state_fingerprint() -> tuple[frozenset, tuple]:
    namespace = vars(training_step)
    sizes = tuple(
        sorted(
            (name, len(value))
            for name, value in namespace.items()
            if isinstance(value, (list, dict, set))
        )
    )
    return frozenset(namespace), sizes


def test_the_module_holds_no_module_level_mutable_state() -> None:
    """Inv-12: no cache, no registry, no module-level mutable container."""
    containers = {
        name: type(value).__name__
        for name, value in vars(training_step).items()
        if not name.startswith("__") and isinstance(value, (list, dict, set))
    }

    assert containers == {}


def test_the_module_gains_no_mutable_state_between_calls() -> None:
    _compile_one_parameter()
    before = _module_state_fingerprint()
    _compile_one_parameter()
    after = _module_state_fingerprint()

    assert before == after


def test_two_calls_produce_equal_framework_structure() -> None:
    first = _compile_two_parameters()
    second = _compile_two_parameters()

    assert first.loss_value_id == second.loss_value_id
    assert first.forward_capture_value_ids == second.forward_capture_value_ids
    assert first.seed_value_ids == second.seed_value_ids
    assert dict(first.input_value_ids) == dict(second.input_value_ids)
    assert first.forward.selected_outputs == second.forward.selected_outputs
    assert first.derivative.selected_outputs == second.derivative.selected_outputs
    assert first.provenance == second.provenance
    assert first.source_forward_graph is not second.source_forward_graph


def test_two_calls_share_no_artifact_objects() -> None:
    first = _compile_one_parameter()
    second = _compile_one_parameter()

    assert first.source_derivative_program is not second.source_derivative_program
    assert first.forward is not second.forward
    assert first.parameters[0].update is not second.parameters[0].update


# ==========================================================================
# edge cases
# ==========================================================================


def test_a_parameter_that_is_also_the_loss_output_compiles() -> None:
    record = _compile_one_parameter(
        loss=_identity_loss,
        inputs={"w": {"dtype": "f32", "shape": (2, 3)}},
        parameters=("w",),
    )

    assert record.loss_value_id == record.parameters[0].value_id
    assert record.forward_capture_value_ids == ()
    assert record.forward.selected_outputs == (record.loss_value_id,)
    assert record.derivative.selected_outputs == (
        record.parameters[0].gradient_value_id,
    )


def test_the_record_carries_opaque_consumer_values_untouched() -> None:
    """Inv-3: the framework never hashes, compares, or iterates a target value."""
    record = _compile_one_parameter()

    values = [
        record.forward.values[value_id]
        for value_id in record.forward.selected_outputs
    ]
    assert all(isinstance(value, (_OpaqueTargetValue, ValueDependency)) for value in values)
    assert isinstance(record.forward.values, (Mapping, MappingProxyType))
