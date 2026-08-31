"""Contract, boundary, and compatibility certification for `compile_training_step`.

This module is the §17.6 pass. It certifies a *finished* surface rather than
driving one into existence, so it is deliberately written to be non-vacuous
under mutation rather than merely green: every assertion names the exact
category, the exact identifier, or the exact structural equality it is
protecting, so corrupting the thing it certifies makes a case here fail.

Six groups, in §17.6 order:

1. **The failure matrix** (§17.6.1) -- one case per error category this module
   can reach, asserting the exact category *and* a message naming the
   offending declaration, callable, or value. The malformed-typed-spec and
   optimizer-input cases additionally assert the category their **existing
   owner** raises (§13.3) rather than a new one, and that a counter on the
   loss reads zero -- Inv-5's "fail before the caller's loss body", observed
   rather than assumed.
2. **Collaborator propagation** (§13.2) -- the loss body and the optimizer
   `update` body propagate unchanged and uncategorized; a handler is
   categorized `handler_contract_violation` by the lowering module's existing
   wrapper.
3. **Seed identity** (§17.6.3, Inv-11). Two rules govern every case here and
   both come from §8.3: **the seed's spelling is not the contract**, so no
   assertion in this file names, matches, or pattern-checks a namespace,
   prefix, or literal identifier for the minted seed -- where a test needs a
   graph occupying the minter's natural first candidates it *discovers* them
   by minting against a graph that does not contain them; and **ordering is
   observed, not assumed**, so "before dependency analysis" is proved by a
   seam that raises a `BaseException` sentinel if it is ever entered. A
   mechanical case at the end re-asserts the first rule over this file's own
   source.
4. **Conditional determinism** (Inv-13) -- equal framework structure for equal
   declarations and deterministic collaborators, with a companion case that
   documents the limit by compiling with a handler that returns a fresh object
   every call.
5. **Boundary** (Inv-2, Inv-12, Inv-15) -- mechanical, over the module source,
   in the AST-over-source-tree style `test_autodiff_no_forbidden_dependencies`
   and the symbolic instantiation guards already use.
6. **Compatibility** (BC-2, BC-5) -- the export set, the top-level absence, and
   the four appended categories with every pre-existing member unchanged in
   value and position.

Target values are opaque here on purpose (Inv-3): the handlers below return
plain strings or bare objects rather than arrays, because nothing this file
asserts is numerical and a framework that inspected a target value would have
nothing to inspect. The numerical proof is §17.5's, and lives in
`test_autodiff_training_step_end_to_end`.

No production file is modified by this module.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Optional

import pytest
import tinychain as tc
from tinychain.autodiff import compile_training_step
from tinychain.autodiff import dependencies as _dependencies_module
from tinychain.autodiff import training_step
from tinychain.autodiff.generate import generate as _real_generate
from tinychain.autodiff.graph import MulOperator, TensorGraph, TensorNodeRecord
from tinychain.autodiff.lowering import OperationHandlerRegistry
from tinychain.autodiff.protocol import (
    AUTODIFF_ERROR_CATEGORIES,
    AutodiffError,
    DerivativeMetadata,
)
from tinychain.autodiff.reverse import DerivativeProgram
from tinychain.autodiff.training import SGD, Optimizer
from tinychain.autodiff.training_step import TracedLoss

from tests.autodiff_reference_consumer import training_step_registry


# --------------------------------------------------------------------------
# declarations
#
# The §17.3.1 loss, asymmetrically shaped so a transposed matmul anywhere is a
# shape error rather than a plausible answer.
# --------------------------------------------------------------------------

SCALAR_SPEC: Mapping[str, object] = {"dtype": "f64", "shape": []}

INPUTS: Mapping[str, Mapping[str, object]] = {
    "x": {"dtype": "f64", "shape": (3, 2)},
    "y": {"dtype": "f64", "shape": (3, 4)},
    "w": {"dtype": "f64", "shape": (2, 4)},
}

OPTIMIZER_INPUTS: Mapping[str, Mapping[str, object]] = {
    "learning_rate": SCALAR_SPEC
}


def residual_loss(*, x: object, y: object, w: object) -> object:
    """`mean((x @ w - y) ** 2)` -- the loss every case in this file compiles."""
    residual = x @ w - y
    return (residual * residual).mean([0, 1])


class _CountingLoss:
    """`residual_loss` with an invocation counter, for the Inv-5 cases.

    Accepts `**kwargs` so `inspect.signature(...).bind(...)` succeeds for the
    declared input names: the point of the two cases that use it is that the
    *body* is never reached, which a signature rejection would mask.
    """

    __qualname__ = "_CountingLoss"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **inputs: object) -> object:
        self.calls += 1
        return residual_loss(**inputs)


# --------------------------------------------------------------------------
# collaborators
#
# Two registries over the same nine operator types `training_step_registry`
# resolves. The deterministic one returns a value derived only from the node
# id, so two compiles of equal declarations produce equal target values; the
# fresh one returns a new object every call, which is exactly the collaborator
# Inv-13 makes no claim about.
# --------------------------------------------------------------------------


class _FreshValue:
    """A target value with identity semantics and no `__eq__` of its own."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


class _DeterministicHandler:
    def __init__(self, operator_type: type) -> None:
        self.operator_type = operator_type

    def lower(self, context: object) -> object:
        return f"lowered:{context.node_id}"


class _FreshHandler:
    def __init__(self, operator_type: type) -> None:
        self.operator_type = operator_type

    def lower(self, context: object) -> object:
        return _FreshValue(context.node_id)


class _RaisingHandlerError(Exception):
    """A consumer's own exception class, which the framework cannot enumerate."""


class _RaisingHandler:
    def __init__(self, operator_type: type) -> None:
        self.operator_type = operator_type

    def lower(self, context: object) -> object:
        raise _RaisingHandlerError("this handler cannot lower anything")


def _supported_operator_types() -> tuple[type, ...]:
    """The operator types a full compilation of `residual_loss` reaches.

    Read off the shared reference registry rather than restated, so this file
    cannot drift from the set §17.1's one consumer was measured to need.
    """
    return tuple(training_step_registry().supported_types())


def _registry_of(handler_factory) -> OperationHandlerRegistry:
    registry = OperationHandlerRegistry()
    for operator_type in _supported_operator_types():
        registry.register(handler_factory(operator_type))
    return registry


def deterministic_registry() -> OperationHandlerRegistry:
    return _registry_of(_DeterministicHandler)


def fresh_registry() -> OperationHandlerRegistry:
    return _registry_of(_FreshHandler)


def deterministic_binding(dependency: object) -> object:
    return f"bound:{dependency.value_id}"


def fresh_binding(dependency: object) -> object:
    return _FreshValue(dependency.value_id)


def _compile(**overrides: object) -> object:
    """Compile one training step, with every collaborator deterministic."""
    kwargs: dict[str, object] = {
        "loss": residual_loss,
        "inputs": INPUTS,
        "parameters": ("w",),
        "optimizer": SGD(),
        "optimizer_inputs": OPTIMIZER_INPUTS,
        "handlers": deterministic_registry(),
        "bind_input": deterministic_binding,
    }
    kwargs.update(overrides)
    loss = kwargs.pop("loss")
    with tc.state.scoped_context():
        return compile_training_step(loss, **kwargs)


# ==========================================================================
# §17.6.1 -- the failure matrix
#
# One case per category this module can reach. Every case asserts the exact
# category and that the message names the offending declaration, callable, or
# value -- a category alone would pass for a message that named nothing.
# ==========================================================================


def test_empty_parameters_is_an_invalid_training_declaration() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _compile(parameters=())

    assert excinfo.value.category == "invalid_training_declaration"
    assert "parameters" in excinfo.value.message


def test_a_parameter_absent_from_inputs_names_the_parameter_and_the_inputs() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _compile(parameters=("absent_parameter",))

    assert excinfo.value.category == "invalid_training_declaration"
    message = excinfo.value.message
    assert "absent_parameter" in message
    for declared in INPUTS:
        assert declared in message


def test_a_repeated_parameter_names_the_repeated_parameter() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _compile(parameters=("w", "w"))

    assert excinfo.value.category == "invalid_training_declaration"
    assert "'w'" in excinfo.value.message


def test_an_empty_inputs_mapping_is_an_invalid_training_declaration() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _compile(inputs={}, parameters=("w",))

    assert excinfo.value.category == "invalid_training_declaration"
    assert "inputs" in excinfo.value.message


@pytest.mark.parametrize(
    ("spec", "category"),
    [
        ({"shape": (2, 4)}, "missing_dtype_metadata"),
        ({"dtype": "f64"}, "missing_shape_metadata"),
        ("not-a-mapping", "missing_dtype_metadata"),
    ],
)
def test_a_malformed_typed_spec_raises_its_existing_owners_category_before_the_loss_runs(
    spec: object, category: str
) -> None:
    """§13.3: the typed-input-spec validator owns this, and keeps its category.

    A new category here -- `invalid_training_declaration`, say -- would mean
    this module had re-implemented a check the training module already owns,
    which is exactly what §8.1 forbids. The zero counter is Inv-5: the
    declaration is rejected before the builder is entered, so the caller's
    loss body is never reached.
    """
    loss = _CountingLoss()
    malformed = dict(INPUTS)
    malformed["w"] = spec

    with pytest.raises(AutodiffError) as excinfo:
        _compile(loss=loss, inputs=malformed)

    assert excinfo.value.category == category
    assert "'w'" in excinfo.value.message
    assert loss.calls == 0


def test_optimizer_inputs_disagreeing_raise_the_existing_owners_category_before_the_loss_runs() -> None:
    """§13.3: the training module's declared-inputs check owns this one.

    `SGD` reads `learning_rate`; the declaration names `step_size`. The
    category is the optimizer contract's own `invalid_update_signature`, the
    message names both sides, and the loss body never ran.
    """
    loss = _CountingLoss()

    with pytest.raises(AutodiffError) as excinfo:
        _compile(loss=loss, optimizer_inputs={"step_size": SCALAR_SPEC})

    assert excinfo.value.category == "invalid_update_signature"
    message = excinfo.value.message
    assert "SGD" in message
    assert "learning_rate" in message
    assert "step_size" in message
    assert loss.calls == 0


def test_a_mismatched_loss_signature_names_the_loss_callable() -> None:
    def wrong_arity_loss(*, x: object, w: object) -> object:
        return x * w

    with pytest.raises(AutodiffError) as excinfo:
        _compile(loss=wrong_arity_loss)

    assert excinfo.value.category == "invalid_loss_signature"
    message = excinfo.value.message
    assert "wrong_arity_loss" in message
    assert "y" in message


def test_a_non_callable_loss_is_reported_as_an_invalid_loss_signature() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _compile(loss=object())

    assert excinfo.value.category == "invalid_loss_signature"
    assert "object" in excinfo.value.message


def test_a_loss_returning_a_non_tensor_names_the_type_it_returned() -> None:
    def returns_a_number(*, x: object, y: object, w: object) -> object:
        return 3.5

    with pytest.raises(AutodiffError) as excinfo:
        _compile(loss=returns_a_number)

    assert excinfo.value.category == "invalid_loss_output"
    message = excinfo.value.message
    assert "returns_a_number" in message
    assert "float" in message


class _WrongSignatureOptimizer(Optimizer):
    """Declares `learning_rate` but its `update` does not accept it."""

    required_optimizer_inputs: tuple[str, ...] = ("learning_rate",)

    def update(self, *, parameter: object, gradient: object) -> object:
        return parameter - gradient


def test_an_optimizer_whose_update_signature_does_not_match_names_the_optimizer() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _compile(optimizer=_WrongSignatureOptimizer())

    assert excinfo.value.category == "invalid_update_signature"
    message = excinfo.value.message
    assert "_WrongSignatureOptimizer" in message
    assert "learning_rate" in message


def test_a_plain_callable_optimizer_with_the_wrong_signature_names_the_callable() -> None:
    """The plain-callable branch of §8.1: only the signature binding applies."""

    def plain_update(*, parameter: object, gradient: object) -> object:
        return parameter - gradient

    with pytest.raises(AutodiffError) as excinfo:
        _compile(optimizer=plain_update)

    assert excinfo.value.category == "invalid_update_signature"
    assert "learning_rate" in excinfo.value.message


def test_a_registry_missing_a_handler_names_the_unresolved_operator_type() -> None:
    registry = OperationHandlerRegistry()
    registry.register(_DeterministicHandler(MulOperator))

    with pytest.raises(AutodiffError) as excinfo:
        _compile(handlers=registry)

    assert excinfo.value.category == "unsupported_operator"
    assert "Operator" in excinfo.value.message


def test_a_non_registry_handlers_argument_is_a_handler_contract_violation() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _compile(handlers={"MatmulOperator": _DeterministicHandler(MulOperator)})

    assert excinfo.value.category == "handler_contract_violation"
    assert "OperationHandlerRegistry" in excinfo.value.message


def test_two_handlers_for_one_operator_type_are_rejected_by_the_registry() -> None:
    """The registry owns this, and rejects the second registration at source."""
    registry = OperationHandlerRegistry()
    registry.register(_DeterministicHandler(MulOperator))

    with pytest.raises(AutodiffError) as excinfo:
        registry.register(_FreshHandler(MulOperator))

    assert excinfo.value.category == "handler_contract_violation"
    assert "MulOperator" in excinfo.value.message


class _MalformedFusion:
    """Declares a lookahead that is not a positive integer."""

    lookahead = 0

    def fuse(self, context: object) -> object:
        return None


def test_a_malformed_fusion_hook_names_the_hook_and_its_lookahead() -> None:
    with pytest.raises(AutodiffError) as excinfo:
        _compile(fusion=_MalformedFusion())

    assert excinfo.value.category == "handler_contract_violation"
    message = excinfo.value.message
    assert "_MalformedFusion" in message
    assert "lookahead" in message


def test_parameter_for_an_unknown_name_names_the_declared_parameters() -> None:
    record = _compile()

    with pytest.raises(AutodiffError) as excinfo:
        record.parameter("x")

    assert excinfo.value.category == "invalid_training_declaration"
    message = excinfo.value.message
    assert "'x'" in message
    assert "'w'" in message


def test_parameter_returns_the_declared_parameter_it_names() -> None:
    """The control for the case above: a declared name resolves."""
    record = _compile()

    assert record.parameter("w") is record.parameters[0]
    assert record.parameter("w").name == "w"


# ==========================================================================
# §17.6.2 -- collaborator propagation (§13.2)
# ==========================================================================


class _ApplicationError(Exception):
    """Application code's own exception, which the framework cannot describe."""


def test_a_loss_body_exception_propagates_unchanged_and_uncategorized() -> None:
    marker = _ApplicationError("the application's own failure")

    def raising_loss(*, x: object, y: object, w: object) -> object:
        raise marker

    with pytest.raises(_ApplicationError) as excinfo:
        _compile(loss=raising_loss)

    assert excinfo.value is marker
    assert not isinstance(excinfo.value, AutodiffError)


def test_a_loss_body_system_exit_is_never_wrapped() -> None:
    """Interpreter control flow is never wrapped anywhere (§13.2)."""

    def exiting_loss(*, x: object, y: object, w: object) -> object:
        raise SystemExit(3)

    with pytest.raises(SystemExit) as excinfo:
        _compile(loss=exiting_loss)

    assert excinfo.value.code == 3


class _RaisingOptimizer(Optimizer):
    """A well-formed optimizer whose `update` *body* fails."""

    required_optimizer_inputs: tuple[str, ...] = ("learning_rate",)

    def __init__(self) -> None:
        self.marker = _ApplicationError("the optimizer's own failure")

    def update(
        self, *, parameter: object, gradient: object, learning_rate: object
    ) -> object:
        raise self.marker


def test_an_optimizer_update_body_exception_propagates_unchanged_and_uncategorized() -> None:
    optimizer = _RaisingOptimizer()

    with pytest.raises(_ApplicationError) as excinfo:
        _compile(optimizer=optimizer)

    assert excinfo.value is optimizer.marker
    assert not isinstance(excinfo.value, AutodiffError)


def test_a_handler_exception_is_categorized_handler_contract_violation() -> None:
    """The lowering module's existing wrapper does this, and is not reimplemented."""
    with pytest.raises(AutodiffError) as excinfo:
        _compile(handlers=_registry_of(_RaisingHandler))

    assert excinfo.value.category == "handler_contract_violation"
    message = excinfo.value.message
    assert "_RaisingHandlerError" in message
    assert "this handler cannot lower anything" in message


# ==========================================================================
# §17.6.3 -- seed identity (Inv-11)
#
# Hand-built source graphs, because the property under test is what the minter
# does when specific identifiers are already occupied, and a traced graph's
# identifiers are chosen by the builder rather than by the test.
# ==========================================================================

SEED_SPEC: Mapping[str, object] = {"dtype": "f32", "shape": [2, 3]}
SEED_PARAMETERS = ("w", "b")


def _mul_chain(
    output_value_ids: Sequence[str],
    *,
    input_value_ids: tuple[str, str] = ("pa", "pb"),
) -> TensorGraph:
    first, second = input_value_ids
    nodes: list[TensorNodeRecord] = []
    previous = first
    for index, output_value_id in enumerate(output_value_ids):
        nodes.append(
            TensorNodeRecord(
                node_id=f"m{index}",
                output_value_id=output_value_id,
                operator=MulOperator(),
                op_params={},
                input_value_ids=[previous, second],
                output_typespec=dict(SEED_SPEC),
            )
        )
        previous = output_value_id
    return TensorGraph(
        nodes=nodes,
        inputs=[(first, dict(SEED_SPEC)), (second, dict(SEED_SPEC))],
        outputs=[output_value_ids[-1]],
    )


def _traced_chain(
    output_value_ids: Sequence[str],
    *,
    input_value_ids: tuple[str, str] = ("pa", "pb"),
) -> TracedLoss:
    graph = _mul_chain(output_value_ids, input_value_ids=input_value_ids)
    return TracedLoss(
        graph=graph,
        loss_value_id=output_value_ids[-1],
        input_value_ids={
            SEED_PARAMETERS[0]: input_value_ids[0],
            SEED_PARAMETERS[1]: input_value_ids[1],
        },
    )


def _occupied_value_ids(graph: TensorGraph) -> set[str]:
    """Exactly the set §8.3 requires the candidate search to avoid."""
    occupied = {value_id for value_id, _ in graph.inputs}
    occupied.update(node.output_value_id for node in graph.nodes)
    return occupied


def _mint_seed_against(traced: TracedLoss) -> str:
    return training_step.differentiate_loss(
        traced=traced, parameters=[SEED_PARAMETERS[0]]
    ).seed_value_id


def test_the_minted_seed_walks_past_every_occupied_candidate_including_a_parameters_own_id() -> None:
    """Occupy the minter's own candidates, discovered rather than named.

    Round zero mints against a graph containing none of them; every later
    round rebuilds the graph so every previously observed choice is occupied,
    and the *first* discovered candidate is placed on a **parameter's** own
    value id -- a minter that only scanned node outputs would hand that back.
    """
    discovered: list[str] = []
    for round_index in range(8):
        output_value_ids = list(discovered[1:]) or ["loss_output"]
        parameter_value_id = discovered[0] if discovered else "pa"
        traced = _traced_chain(
            output_value_ids, input_value_ids=(parameter_value_id, "pb")
        )

        seed = _mint_seed_against(traced)

        occupied = _occupied_value_ids(traced.graph)
        assert seed not in occupied, (
            f"round {round_index}: minted seed {seed!r} is already occupied by "
            f"the source forward graph, whose occupied ids are {sorted(occupied)}"
        )
        assert seed != traced.input_value_ids[SEED_PARAMETERS[0]]
        assert seed not in discovered, (
            f"round {round_index}: minted seed {seed!r} repeats a candidate the "
            "minter had already been shown to be occupied"
        )
        discovered.append(seed)

    assert len(set(discovered)) == 8


class _AnalysisEntered(BaseException):
    """Raised by the analysis seam, so entering it can never go unnoticed.

    A `BaseException` rather than an `Exception`, so no broad `except
    Exception` anywhere on the path could swallow it and report a category.
    """


def _forbid_dependency_analysis(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    calls: list[tuple] = []

    def recorder(*args: object, **kwargs: object):
        calls.append((args, kwargs))
        raise _AnalysisEntered("dependency analysis was entered")

    for module in (_dependencies_module, training_step):
        for name in ("analyze_derivative_dependencies", "analyze_graph_dependencies"):
            monkeypatch.setattr(module, name, recorder, raising=False)
    return calls


class _FixedGenerate:
    """Returns a caller-built `DerivativeProgram`, recording the minted seed.

    The collision is therefore constructed out of the identifier the stage
    actually minted -- never one this file guessed.
    """

    def __init__(self, build_program) -> None:
        self._build_program = build_program
        self.seeds: list[object] = []

    def __call__(self, *args: object, **kwargs: object) -> DerivativeProgram:
        bound = inspect.signature(_real_generate).bind(*args, **kwargs)
        bound.apply_defaults()
        seed = bound.arguments["seed"]
        seed_value_id = seed[0] if isinstance(seed, list) else seed
        self.seeds.append(seed_value_id)
        return self._build_program(seed_value_id)


def _fake_program(
    *, nodes: Sequence[TensorNodeRecord], seed_value_id: str
) -> DerivativeProgram:
    metadata = DerivativeMetadata(
        source_graph_id="source",
        transform_version="0.1.0",
        tensor_op_contract_version="0.1.0",
        wrt_signature=("pa",),
        seed_contract="",
    )
    # The seed is always present in `value_typespecs`: reverse traversal
    # records it there for every program it produces, so a check reading that
    # table instead of the produced ids would reject every derivative ever
    # generated.
    value_typespecs = {
        "pa": dict(SEED_SPEC),
        "d0": dict(SEED_SPEC),
        seed_value_id: dict(SEED_SPEC),
    }
    return DerivativeProgram(
        nodes=list(nodes),
        gradients={"pa": "d0"},
        output_gradients=["d0"],
        metadata=metadata,
        value_typespecs=value_typespecs,
    )


@pytest.mark.parametrize("collide_on", ["output_value_id", "node_id"])
def test_a_generated_derivative_colliding_with_the_seed_fails_before_dependency_analysis(
    monkeypatch: pytest.MonkeyPatch, collide_on: str
) -> None:
    """`ambiguous_producer` at the post-generation check, and not one step later.

    The analysis seam raises `_AnalysisEntered`; an implementation that
    analysed the program before checking the seed would surface that sentinel
    instead, so "before dependency analysis" is observed here rather than
    inferred from the fact that something was raised at all.
    """
    analysis_calls = _forbid_dependency_analysis(monkeypatch)

    def build(seed_value_id: str) -> DerivativeProgram:
        return _fake_program(
            nodes=[
                TensorNodeRecord(
                    node_id=seed_value_id if collide_on == "node_id" else "dn0",
                    output_value_id=(
                        seed_value_id if collide_on == "output_value_id" else "d0"
                    ),
                    operator=MulOperator(),
                    op_params={},
                    input_value_ids=["pa", seed_value_id],
                    output_typespec=dict(SEED_SPEC),
                )
            ],
            seed_value_id=seed_value_id,
        )

    fake = _FixedGenerate(build)
    monkeypatch.setattr(training_step, "generate", fake)

    with pytest.raises(AutodiffError) as excinfo:
        training_step.differentiate_loss(
            traced=_traced_chain(["loss_output"]), parameters=[SEED_PARAMETERS[0]]
        )

    assert excinfo.value.category == "ambiguous_producer"
    assert fake.seeds[0] in excinfo.value.message
    assert analysis_calls == []


def _replace_program_nodes(
    program: DerivativeProgram, nodes: Sequence[TensorNodeRecord]
) -> DerivativeProgram:
    return dataclasses.replace(program, nodes=list(nodes))


def test_a_derivative_pass_producing_the_seed_fails_as_an_expansion_contract_violation() -> None:
    """§17.4/Inv-11: the seed stays a required free input across every pass.

    The seed is *discovered* from a clean compile and handed to the pass; the
    pass then mints a node whose output is that same identifier.
    """
    seed_value_id = _compile().seed_value_ids[0]

    def produces_the_seed(program: DerivativeProgram) -> DerivativeProgram:
        template = program.nodes[0]
        extra = dataclasses.replace(
            template, node_id="reused_seed_node", output_value_id=seed_value_id
        )
        return _replace_program_nodes(program, list(program.nodes) + [extra])

    with pytest.raises(AutodiffError) as excinfo:
        _compile(derivative_expansions=(produces_the_seed,))

    assert excinfo.value.category == "expansion_contract_violation"
    message = excinfo.value.message
    assert "produces_the_seed" in message
    assert "position 0" in message
    assert seed_value_id in message


def test_a_derivative_pass_dropping_the_seed_fails_as_an_expansion_contract_violation() -> None:
    """A program that no longer reads the seed is no longer the caller's derivative."""
    seed_value_id = _compile().seed_value_ids[0]

    def drops_the_seed(program: DerivativeProgram) -> DerivativeProgram:
        # Every node that reads the seed is replaced by one carrying a freshly
        # minted node id and reproducing the same output value id, so the
        # artifact stays structurally sound in every respect except the one
        # under test: the seed is no longer a required free input.
        rewritten = [
            dataclasses.replace(
                node,
                node_id=f"exn_seedless{index}",
                input_value_ids=[
                    value_id
                    for value_id in node.input_value_ids
                    if value_id != seed_value_id
                ],
            )
            if seed_value_id in node.input_value_ids
            else node
            for index, node in enumerate(program.nodes)
        ]
        return _replace_program_nodes(program, rewritten)

    # A dropped seed is *also* visible to the final preservation
    # recomputation, whose message names the same pass and the same seed --
    # so the category and the message alone cannot tell the per-pass rule
    # from the recomputation. The next pass never running is what can: §8.6
    # validates each result before the following pass is invoked.
    invoked_after: list[str] = []

    def never_reached(program: DerivativeProgram) -> DerivativeProgram:
        invoked_after.append("invoked")
        return program

    with pytest.raises(AutodiffError) as excinfo:
        _compile(derivative_expansions=(drops_the_seed, never_reached))

    assert excinfo.value.category == "expansion_contract_violation"
    message = excinfo.value.message
    assert "drops_the_seed" in message
    assert seed_value_id in message
    assert invoked_after == [], (
        "the pass after the offending one ran, so the seed rule was not "
        "applied per pass as §8.6 requires"
    )


def _every_value_id(record: object) -> set[str]:
    """Every framework identifier the record carries, across all four artifacts."""
    identifiers: set[str] = set()
    for graph in (record.source_forward_graph, record.lowered_forward_graph):
        identifiers.update(value_id for value_id, _ in graph.inputs)
        identifiers.update(node.output_value_id for node in graph.nodes)
        identifiers.update(node.node_id for node in graph.nodes)
    for program in (
        record.source_derivative_program,
        record.lowered_derivative_program,
    ):
        identifiers.update(node.output_value_id for node in program.nodes)
        identifiers.update(node.node_id for node in program.nodes)
        identifiers.update(program.value_typespecs)
    for compiled in record.parameters:
        for graph in (compiled.source_update_graph, compiled.lowered_update_graph):
            identifiers.update(value_id for value_id, _ in graph.inputs)
            identifiers.update(node.output_value_id for node in graph.nodes)
        identifiers.update(compiled.update.values)
    identifiers.update(record.forward.values)
    identifiers.update(record.derivative.values)
    return identifiers


def test_the_seed_label_appears_only_in_provenance_and_never_as_a_value_id() -> None:
    label = "a_label_no_minter_would_ever_produce"

    record = _compile(seed_label=label)

    assert record.provenance.seed_label == label
    assert label not in _every_value_id(record)
    assert record.seed_value_ids[0] != label
    assert label not in record.seed_value_ids


def test_the_seed_label_does_not_change_a_single_framework_identifier() -> None:
    """Inv-11: the label carries no identity, so it cannot move an id."""
    default = _compile()
    labelled = _compile(seed_label="a_completely_different_label")

    # Provenance is compared separately: `seed_label` is a provenance field,
    # and it is the one thing that is *meant* to differ between the two.
    assert _framework_structure(default, with_provenance=False) == (
        _framework_structure(labelled, with_provenance=False)
    )
    assert default.seed_value_ids == labelled.seed_value_ids
    assert dataclasses.replace(default.provenance, seed_label="") == (
        dataclasses.replace(labelled.provenance, seed_label="")
    )


def test_no_assertion_in_this_file_names_the_minted_seeds_spelling() -> None:
    """§8.3/§17.6.3: the seed's spelling is not the contract, mechanically.

    A file that hard-coded the identifier would pass for an implementation
    whose candidate search never advances, and would break the moment the
    spelling changed -- neither of which is the property under test. The seed
    is obtained from a real compile and this file's own source is searched for
    it, so the guard cannot go stale.
    """
    seed_value_id = _compile().seed_value_ids[0]
    source = pathlib.Path(__file__).read_text(encoding="utf-8")

    assert seed_value_id not in source, (
        f"this file names the minted seed {seed_value_id!r} literally; §8.3 "
        "makes the spelling an implementation detail no test may depend on"
    )


# ==========================================================================
# §17.6.4 -- conditional determinism (Inv-13)
# ==========================================================================


def _dependency_snapshot(analysis: object) -> tuple:
    return (
        tuple(analysis.selected_outputs),
        tuple(
            (
                dependency.value_id,
                dependency.provenance,
                dependency.dtype,
                dependency.shape,
            )
            for dependency in analysis.dependencies
        ),
    )


def _program_snapshot(program: object) -> tuple:
    """A lowered program's *framework* structure, with no target value read.

    `LoweredOperation.value` and `LoweredProgram.values`' values are the
    consumer's own target values (Inv-3), so only their *keys* and the
    framework's own identifiers, ordering, and provenance appear here.
    """
    return (
        tuple(program.selected_outputs),
        tuple(
            (
                operation.output_value_id,
                tuple(operation.source_node_ids),
                tuple(type(operator).__name__ for operator in operation.source_operators),
                operation.is_fused,
            )
            for operation in program.operations
        ),
        tuple(sorted(program.values)),
        _dependency_snapshot(program.dependencies),
    )


def _graph_snapshot(graph: TensorGraph) -> tuple:
    return (
        tuple((value_id, tuple(sorted((k, repr(v)) for k, v in spec.items())))
              for value_id, spec in graph.inputs),
        tuple(
            (
                node.node_id,
                node.output_value_id,
                type(node.operator).__name__,
                tuple(node.input_value_ids),
            )
            for node in graph.nodes
        ),
        tuple(graph.outputs),
    )


def _derivative_snapshot(program: DerivativeProgram) -> tuple:
    return (
        tuple(
            (
                node.node_id,
                node.output_value_id,
                type(node.operator).__name__,
                tuple(node.input_value_ids),
            )
            for node in program.nodes
        ),
        tuple(sorted(program.gradients.items())),
        tuple(program.output_gradients),
        tuple(sorted(program.value_typespecs)),
    )


def _framework_structure(record: object, *, with_provenance: bool = True) -> tuple:
    """Every framework-owned fact of a record: ids, ordering, selections, provenance."""
    return (
        _graph_snapshot(record.source_forward_graph),
        _graph_snapshot(record.lowered_forward_graph),
        _derivative_snapshot(record.source_derivative_program),
        _derivative_snapshot(record.lowered_derivative_program),
        _program_snapshot(record.forward),
        _program_snapshot(record.derivative),
        tuple(sorted(record.input_value_ids.items())),
        record.loss_value_id,
        tuple(record.forward_capture_value_ids),
        tuple(record.seed_value_ids),
        tuple(
            (
                compiled.name,
                compiled.value_id,
                compiled.gradient_value_id,
                compiled.updated_parameter_value_id,
                tuple(sorted(compiled.update_input_value_ids.items())),
                _graph_snapshot(compiled.source_update_graph),
                _graph_snapshot(compiled.lowered_update_graph),
                _program_snapshot(compiled.update),
            )
            for compiled in record.parameters
        ),
        dataclasses.astuple(record.provenance) if with_provenance else None,
    )


def test_two_compiles_with_deterministic_collaborators_are_structurally_equal() -> None:
    first = _compile()
    second = _compile()

    assert first is not second
    assert _framework_structure(first) == _framework_structure(second)
    # The target values are equal too here, but only because *these*
    # collaborators are deterministic -- see the companion case below.
    assert first.forward.output_values == second.forward.output_values


def test_a_fresh_object_handler_still_gives_equal_framework_structure() -> None:
    """The documented limit of Inv-13, made observable.

    These handlers return a brand-new object on every call, so no claim is
    made -- and none is asserted -- about the *target values* inside the two
    records. The framework's own structure is equal all the same, which is
    exactly and only what Inv-13 promises.
    """
    first = _compile(handlers=fresh_registry(), bind_input=fresh_binding)
    second = _compile(handlers=fresh_registry(), bind_input=fresh_binding)

    assert _framework_structure(first) == _framework_structure(second)
    # Deliberately asserted in the negative: the values differ, which is why
    # no equality is claimed for them anywhere above.
    assert first.forward.output_values != second.forward.output_values


def test_declarations_differing_only_in_parameter_ordering_are_not_claimed_equal() -> None:
    """Order is meaning (Inv-6), so a reordering is a different declaration."""
    two_parameter_inputs = dict(INPUTS)
    two_parameter_inputs["b"] = {"dtype": "f64", "shape": (3, 4)}

    def biased_loss(*, x: object, y: object, w: object, b: object) -> object:
        residual = x @ w + b - y
        return (residual * residual).mean([0, 1])

    forward = _compile(
        loss=biased_loss, inputs=two_parameter_inputs, parameters=("w", "b")
    )
    reversed_order = _compile(
        loss=biased_loss, inputs=two_parameter_inputs, parameters=("b", "w")
    )

    assert forward.provenance.parameter_names == ("w", "b")
    assert reversed_order.provenance.parameter_names == ("b", "w")
    assert forward.provenance.wrt_signature != reversed_order.provenance.wrt_signature
    assert forward.derivative.selected_outputs != reversed_order.derivative.selected_outputs


# ==========================================================================
# §17.6.5 -- the boundary (Inv-2, Inv-12, Inv-15)
#
# Mechanical over the module source, in the AST-over-source-tree style
# `test_autodiff_no_forbidden_dependencies` already uses.
# ==========================================================================

_MODULE_PATH = (
    pathlib.Path(training_step.__file__).resolve()
)
_MODULE_TREE = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _absolute_imports(tree: ast.AST) -> list[str]:
    """Every absolute module name the tree imports; relative imports are local."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names.append(node.module)
    return names


def test_the_module_imports_nothing_outside_tinychain_and_the_standard_library() -> None:
    """Inv-15: no backend, no third party, no consumer concept.

    The standard library is not "outside `tinychain`" in the sense Inv-15
    means -- `inspect` and `dataclasses` name no backend -- so the check is
    that every absolute import roots either at `tinychain` or at a stdlib
    module, and nothing else. A third-party or backend import fails here.
    """
    violations = [
        module_name
        for module_name in _absolute_imports(_MODULE_TREE)
        if module_name.split(".")[0] != "tinychain"
        and module_name.split(".")[0] not in sys.stdlib_module_names
    ]

    assert not violations, (
        "the training-step module imports outside tinychain and the standard "
        f"library: {sorted(violations)}"
    )


_FORBIDDEN_DISPATCH_NAMES = frozenset(
    {
        "execute",
        "dispatch",
        "evaluate",
        "resolve",
        "post",
        "put",
        "delete",
        "lower",
        "fuse",
    }
)

_FORBIDDEN_EXECUTION_MODULES = frozenset(
    {"executor", "host", "kernel", "routes", "artifact", "reflection"}
)


def _called_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def test_the_module_dispatches_no_operation_and_evaluates_no_operator() -> None:
    """Inv-2, asserted mechanically rather than by review.

    Three things are checked over the source: it never calls a dispatch or
    evaluation entry point by name (`lower_graph` and
    `lower_derivative_program` are framework *lowering* entry points and are
    not among them), it never constructs a concrete `TensorOperator`, and it
    imports no execution, host, or route module.
    """
    called = _called_names(_MODULE_TREE)
    dispatching = sorted(called & _FORBIDDEN_DISPATCH_NAMES)
    assert not dispatching, (
        f"the training-step module calls dispatch/evaluation entry points {dispatching}"
    )

    constructed_operators = sorted(
        name for name in called if name.endswith("Operator")
    )
    assert not constructed_operators, (
        "the training-step module constructs concrete operators "
        f"{constructed_operators}; it may only carry the ones tracing produced"
    )

    execution_imports = sorted(
        module_name
        for module_name in _absolute_imports(_MODULE_TREE)
        if module_name.split(".")[-1] in _FORBIDDEN_EXECUTION_MODULES
    )
    relative_execution_imports = sorted(
        node.module
        for node in ast.walk(_MODULE_TREE)
        if isinstance(node, ast.ImportFrom)
        and node.level
        and node.module
        and node.module.split(".")[-1] in _FORBIDDEN_EXECUTION_MODULES
    )
    assert not execution_imports and not relative_execution_imports, (
        "the training-step module imports an execution/host/route module: "
        f"{execution_imports + relative_execution_imports}"
    )


def test_the_module_declares_no_mutable_module_level_state() -> None:
    """Inv-12, asserted over the source: no cache, no registry, no counter.

    Every module-level assignment must bind an immutable constant. A dict,
    list, or set literal -- or a call result -- bound at module level is the
    shape a cache or registry takes, so any of them fails here.
    """
    offenders: list[str] = []
    for node in _MODULE_TREE.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        if not isinstance(value, (ast.Constant, ast.Tuple)):
            offenders.append(ast.dump(node)[:120])
        elif isinstance(value, ast.Tuple) and not all(
            isinstance(element, ast.Constant) for element in value.elts
        ):
            offenders.append(ast.dump(node)[:120])

    assert not offenders, (
        f"the training-step module binds mutable module-level state: {offenders}"
    )

    globals_declared = [
        node
        for node in ast.walk(_MODULE_TREE)
        if isinstance(node, (ast.Global, ast.Nonlocal))
    ]
    assert not globals_declared, "the training-step module rebinds module-level state"


def _module_constants() -> dict[str, object]:
    return {
        name: value
        for name, value in vars(training_step).items()
        if not name.startswith("__")
        and isinstance(value, (str, int, float, bool, tuple, frozenset))
    }


def test_two_calls_are_independent_and_leave_no_observable_module_state() -> None:
    """Inv-12: nothing observable persists between two calls.

    Both directions are checked: the records share no mutable object, and the
    module's own attribute set and constants are unchanged by having compiled.
    """
    before_names = set(vars(training_step))
    before_constants = _module_constants()

    first = _compile()
    second = _compile()

    assert first is not second
    assert first.parameters[0] is not second.parameters[0]
    assert first.source_forward_graph is not second.source_forward_graph
    assert first.provenance is not second.provenance
    assert _framework_structure(first) == _framework_structure(second)

    assert set(vars(training_step)) == before_names
    assert _module_constants() == before_constants


def test_the_injection_points_reach_the_lowerings_unwrapped() -> None:
    """Inv-9, the other half of the boundary: no wrapper, no filter, no default.

    A fusion hook that is offered nothing still proves it was *installed*: a
    module that had replaced or defaulted the hook would never call it.
    """
    offered: list[str] = []

    class _RecordingFusion:
        lookahead = 1

        def fuse(self, context: object) -> object:
            offered.append(context.candidates[0].node_id)
            return None

    fusion = _RecordingFusion()
    record = _compile(fusion=fusion)

    assert offered, "the fusion hook was never offered an operation"
    # Every one of the three lowerings must have been offered something: the
    # same object reaches all of them, so a module that threaded it to only
    # some of them -- or replaced it with a default for one -- is caught here
    # rather than by the fact that it was offered anything at all.
    for label, program in (
        ("forward", record.forward),
        ("derivative", record.derivative),
        *((f"update({compiled.name})", compiled.update) for compiled in record.parameters),
    ):
        source_node_ids = {
            node_id
            for operation in program.operations
            for node_id in operation.source_node_ids
        }
        assert source_node_ids <= set(offered), (
            f"the {label} lowering was never offered an operation, so the "
            "fusion hook did not reach it unmodified"
        )
    assert all(not operation.is_fused for operation in record.forward.operations)


# ==========================================================================
# §17.6.6 -- compatibility (BC-2, BC-5)
# ==========================================================================

_TRAINING_STEP_EXPORTS = (
    "CompiledTrainingStep",
    "ParameterCompilation",
    "TrainingStepProvenance",
    "compile_training_step",
)


def test_the_four_public_names_are_exported_from_the_autodiff_package_only() -> None:
    from tinychain import autodiff

    for name in _TRAINING_STEP_EXPORTS:
        assert name in autodiff.__all__
        assert getattr(autodiff, name) is getattr(training_step, name)
        assert not hasattr(tc, name), (
            f"{name!r} leaked onto the top-level tinychain package"
        )

    assert compile_training_step is training_step.compile_training_step


def test_the_stage_functions_stay_module_private() -> None:
    """FR-129-022: `compile_training_step` is the only public composition."""
    from tinychain import autodiff

    for name in (
        "validate_declaration",
        "trace_loss",
        "differentiate_loss",
        "analyze_source_captures",
        "expand_source_artifacts",
        "expand_update_graph",
    ):
        assert hasattr(training_step, name)
        assert name not in autodiff.__all__


_PRE_EXISTING_CATEGORIES = (
    "unsupported_operator",
    "missing_derivative_behavior",
    "missing_derivative_ir",
    "non_differentiable_route",
    "missing_shape_metadata",
    "missing_dtype_metadata",
    "dtype_not_differentiable",
    "shape_mismatch",
    "unresolved_symbolic_shape",
    "symbolic_shape_mismatch",
    "broadcast_shape_mismatch",
    "matmul_shape_mismatch",
    "invalid_permutation",
    "reduction_shape_mismatch",
    "unsupported_reduction",
    "seed_shape_mismatch",
    "malformed_derivative_ir",
    "side_effecting_route_unsupported",
    "autodiff_not_implemented",
    "dtype_mismatch",
    "missing_dependency",
    "ambiguous_producer",
    "invalid_selected_output",
    "handler_contract_violation",
    "invalid_update_signature",
    "invalid_update_output",
)

_APPENDED_CATEGORIES = (
    "invalid_training_declaration",
    "invalid_loss_signature",
    "invalid_loss_output",
    "expansion_contract_violation",
)


def test_error_categories_grew_by_exactly_four_appended_members() -> None:
    """BC-2: every pre-existing member unchanged in value *and* position."""
    assert len(AUTODIFF_ERROR_CATEGORIES) == len(_PRE_EXISTING_CATEGORIES) + len(
        _APPENDED_CATEGORIES
    )
    for position, category in enumerate(_PRE_EXISTING_CATEGORIES):
        assert AUTODIFF_ERROR_CATEGORIES[position] == category, (
            f"pre-existing category {category!r} moved from position {position}"
        )
    assert (
        AUTODIFF_ERROR_CATEGORIES[len(_PRE_EXISTING_CATEGORIES):]
        == _APPENDED_CATEGORIES
    )


def test_every_appended_category_is_accepted_by_the_error_type() -> None:
    for category in _APPENDED_CATEGORIES:
        error = AutodiffError(category, "message")
        assert error.category == category
        assert error.to_dict()["category"] == category
