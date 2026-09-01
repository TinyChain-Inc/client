"""Tests for the caller-chosen reserved identifier prefix of the expansion passes.

Specification §17.7 requires composing both real expansion passes in a single
compile -- `expand_mean_graph` as a `forward_expansions` entry and
`expand_mean_derivative_program` as a `derivative_expansions` entry. Both passes
mint from the same reserved namespace starting at zero, so the lowered forward
graph and the lowered derivative program each contain `exv0`, and the §8.6
preservation recomputation correctly rejects the pair with `ambiguous_producer`.

The fix belongs to the passes: each public entry point accepts an optional,
keyword-only reserved identifier prefix, so a caller composing two passes can
namespace them apart. These tests pin four things:

* the parameter is **keyword-only with a default**, so every pass stays callable
  as a single-positional-argument function and remains usable, unwrapped, as an
  expansion hook -- the property that makes it an expansion pass at all;
* the **default reproduces today's behaviour exactly**, identifier for
  identifier, so no existing caller, artifact, or test observes a change;
* a **custom prefix is honoured and still fail-closed**: minted identifiers are
  spelled under it, and one colliding with an identifier the input artifact
  already uses is rejected rather than silently shadowing it;
* a **prefix that cannot be a safe namespace is rejected** -- empty, not a
  string, or spelled under an identifier namespace another stage owns (the
  tracer's `v`/`n`, the reverse transform's `d`/`dn`).

The last section is the motivation itself: the two passes composed in one
compile with distinct prefixes, where the shared default namespace raises.
No production file other than the expansion module is exercised for its own
sake here; `compile_training_step` appears only as the composition the change
exists to enable.
"""

from __future__ import annotations

import inspect

import pytest
import tinychain as tc
from tinychain.autodiff import (
    AddOperator,
    AutodiffError,
    DerivativeProgram,
    MeanOperator,
    OperationHandlerRegistry,
    TensorGraph,
    TensorGraphBuilder,
    TensorNodeRecord,
    analyze_derivative_dependencies,
    generate,
)
from tinychain.autodiff import training_step
from tinychain.autodiff.training import SGD

from tests import test_autodiff_training_step_end_to_end as e2e
from tests.autodiff_reference_consumer import (
    limited_operation_registry,
    training_step_registry,
)


# --------------------------------------------------------------------------
# lazily resolved surface under test
#
# Resolved inside the test body rather than at import time, so a missing name
# fails a test that has already built its input rather than aborting collection
# of the whole module.
# --------------------------------------------------------------------------


def _passes() -> tuple[object, object, object, object]:
    from tinychain.autodiff import (
        expand_mean_derivative_program,
        expand_mean_graph,
    )
    from tinychain.autodiff.expansion import (
        expand_mean_derivative_program_detailed,
        expand_mean_graph_detailed,
    )

    return (
        expand_mean_graph,
        expand_mean_graph_detailed,
        expand_mean_derivative_program,
        expand_mean_derivative_program_detailed,
    )


def _graph_passes() -> tuple[object, object]:
    forward, forward_detailed, _, _ = _passes()
    return forward, forward_detailed


def _program_passes() -> tuple[object, object]:
    _, _, derivative, derivative_detailed = _passes()
    return derivative, derivative_detailed


def _default_prefixes() -> tuple[str, str]:
    from tinychain.autodiff.expansion import (
        EXPANSION_NODE_ID_PREFIX,
        EXPANSION_VALUE_ID_PREFIX,
    )

    return EXPANSION_NODE_ID_PREFIX, EXPANSION_VALUE_ID_PREFIX


# The parameter name every public pass exposes.
_PARAMETER = "reserved_id_prefix"

# Two prefixes a caller composing the passes would plausibly pick. Neither is
# spelled under a namespace another stage owns.
_FORWARD_PREFIX = "fw"
_DERIVATIVE_PREFIX = "gx"


# --------------------------------------------------------------------------
# artifact fixtures and comparison helpers
# --------------------------------------------------------------------------


def _traced_mean(*, shape=(3, 5), dtype="f64", keepdims=True):
    """Trace `value.mean([0, 1], keepdims=...)` and generate its derivative."""
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype=dtype, shape=shape)
        output = value.mean([0, 1], keepdims=keepdims)
    graph = trace.build(outputs=output)
    program = generate(graph, graph.outputs[0], [graph.inputs[0][0]], "seed")
    return graph, program


def _mean_graph() -> TensorGraph:
    return _traced_mean()[0]


def _mean_program() -> DerivativeProgram:
    return _traced_mean()[1]


def _node_summary(node: TensorNodeRecord) -> tuple:
    """Everything about one node that a rewrite could change."""
    return (
        node.node_id,
        node.output_value_id,
        type(node.operator).__name__,
        node.operator.route_name,
        sorted((str(key), repr(value)) for key, value in node.op_params.items()),
        tuple(node.input_value_ids),
        repr(node.output_typespec),
    )


def _graph_summary(graph: TensorGraph) -> tuple:
    return (
        tuple(_node_summary(node) for node in graph.nodes),
        tuple((value_id, repr(typespec)) for value_id, typespec in graph.inputs),
        tuple(graph.outputs),
    )


def _program_summary(program: DerivativeProgram) -> tuple:
    return (
        tuple(_node_summary(node) for node in program.nodes),
        tuple(sorted(program.gradients.items())),
        tuple(program.output_gradients),
        tuple(
            sorted(
                (value_id, repr(typespec))
                for value_id, typespec in program.value_typespecs.items()
            )
        ),
    )


def _region_summary(regions) -> tuple:
    return tuple(
        (
            region.pass_name,
            tuple(region.source_node_ids),
            tuple(region.emitted_node_ids),
            region.terminal_value_id,
            region.tier,
        )
        for region in regions
    )


def _renamed(text: str, *, custom: str, default: str) -> str:
    """Respell one identifier from the custom namespace back to the default."""
    for suffix in ("n", "v"):
        if text.startswith(f"{custom}{suffix}"):
            return f"{default}{suffix}{text[len(custom) + 1:]}"
    return text


def _renamed_summary(summary, *, custom: str, default: str):
    """Respell every identifier in a summary tree, structure untouched."""
    if isinstance(summary, str):
        return _renamed(summary, custom=custom, default=default)
    if isinstance(summary, tuple):
        return tuple(
            _renamed_summary(item, custom=custom, default=default) for item in summary
        )
    return summary


def _emitted_node_ids(graph_or_program) -> list[str]:
    default_node_prefix, _ = _default_prefixes()
    return [
        node.node_id
        for node in graph_or_program.nodes
        if node.node_id.startswith((default_node_prefix, _FORWARD_PREFIX, _DERIVATIVE_PREFIX))
    ]


# --------------------------------------------------------------------------
# The parameter's shape: keyword-only, defaulted, one positional argument
# --------------------------------------------------------------------------


@pytest.mark.parametrize("index", range(4))
def test_every_pass_exposes_a_keyword_only_defaulted_prefix(index: int) -> None:
    """The prefix is keyword-only with a default, beside one positional artifact."""
    pass_function = _passes()[index]
    signature = inspect.signature(pass_function)
    parameters = list(signature.parameters.values())

    assert len(parameters) == 2, (
        f"{pass_function.__name__} must take exactly the artifact and the prefix, "
        f"got {[parameter.name for parameter in parameters]!r}"
    )
    artifact, prefix = parameters
    assert artifact.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert artifact.default is inspect.Parameter.empty
    assert prefix.name == _PARAMETER
    assert prefix.kind is inspect.Parameter.KEYWORD_ONLY
    assert prefix.default is None


def test_a_pass_stays_callable_as_a_single_argument_expansion() -> None:
    """An unwrapped pass remains a one-positional-argument function."""
    forward, forward_detailed = _graph_passes()
    derivative, derivative_detailed = _program_passes()
    graph, program = _traced_mean()

    for pass_function, artifact in (
        (forward, graph),
        (forward_detailed, graph),
        (derivative, program),
        (derivative_detailed, program),
    ):
        expansion = pass_function  # used unwrapped, exactly as a hook receives it
        assert expansion(artifact) is not None


# --------------------------------------------------------------------------
# The default reproduces today's behaviour exactly
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keepdims", [True, False])
def test_forward_default_matches_an_unprefixed_run(keepdims: bool) -> None:
    """Omitting the prefix and passing `None` produce the identical graph."""
    forward, _ = _graph_passes()
    graph, _ = _traced_mean(keepdims=keepdims)

    baseline = _graph_summary(forward(graph))
    explicit_none = _graph_summary(forward(graph, **{_PARAMETER: None}))

    assert explicit_none == baseline


def test_forward_default_mints_the_documented_namespace() -> None:
    """The default namespace is still `exn…`/`exv…`, indexed from zero."""
    forward, _ = _graph_passes()
    node_prefix, value_prefix = _default_prefixes()
    expanded = forward(_mean_graph())

    minted = [node for node in expanded.nodes if node.node_id.startswith(node_prefix)]
    assert [node.node_id for node in minted] == [
        f"{node_prefix}{index}" for index in range(len(minted))
    ]
    # The region's terminal node carries the replaced mean's own value id, so
    # only the intermediates are minted values.
    minted_values = [
        node.output_value_id
        for node in minted
        if node.output_value_id.startswith(value_prefix)
    ]
    assert minted_values == [
        f"{value_prefix}{index}" for index in range(len(minted_values))
    ]
    assert len(minted_values) == len(minted) - 1


def test_derivative_default_matches_an_unprefixed_run() -> None:
    """Omitting the prefix and passing `None` produce the identical program."""
    derivative, _ = _program_passes()
    program = _mean_program()

    baseline = _program_summary(derivative(program))
    explicit_none = _program_summary(derivative(program, **{_PARAMETER: None}))

    assert explicit_none == baseline


def test_detailed_defaults_match_an_unprefixed_run() -> None:
    """Both detailed forms keep their artifacts and their regions unchanged."""
    _, forward_detailed = _graph_passes()
    _, derivative_detailed = _program_passes()
    graph, program = _traced_mean()

    forward_baseline = forward_detailed(graph)
    forward_explicit = forward_detailed(graph, **{_PARAMETER: None})
    assert _graph_summary(forward_explicit.graph) == _graph_summary(
        forward_baseline.graph
    )
    assert _region_summary(forward_explicit.regions) == _region_summary(
        forward_baseline.regions
    )

    derivative_baseline = derivative_detailed(program)
    derivative_explicit = derivative_detailed(program, **{_PARAMETER: None})
    assert _program_summary(derivative_explicit.program) == _program_summary(
        derivative_baseline.program
    )
    assert _region_summary(derivative_explicit.regions) == _region_summary(
        derivative_baseline.regions
    )


# --------------------------------------------------------------------------
# A custom prefix is honoured, and changes nothing but the spelling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keepdims", [True, False])
def test_forward_custom_prefix_renames_and_nothing_else(keepdims: bool) -> None:
    """A custom prefix respells minted ids and leaves the rewrite identical."""
    forward, _ = _graph_passes()
    node_prefix, value_prefix = _default_prefixes()
    graph, _ = _traced_mean(keepdims=keepdims)

    baseline = forward(graph)
    prefixed = forward(graph, **{_PARAMETER: _FORWARD_PREFIX})

    minted = [
        node
        for node in prefixed.nodes
        if node.node_id.startswith(f"{_FORWARD_PREFIX}n")
    ]
    assert minted, "the pass must mint under the caller's prefix"
    assert [node.node_id for node in minted] == [
        f"{_FORWARD_PREFIX}n{index}" for index in range(len(minted))
    ]
    minted_values = [
        node.output_value_id
        for node in minted
        if node.output_value_id.startswith(f"{_FORWARD_PREFIX}v")
    ]
    assert minted_values == [
        f"{_FORWARD_PREFIX}v{index}" for index in range(len(minted_values))
    ]
    assert not any(
        node.node_id.startswith(node_prefix)
        or node.output_value_id.startswith(value_prefix)
        for node in prefixed.nodes
    ), "no identifier may be left in the default namespace"

    assert _renamed_summary(
        _graph_summary(prefixed), custom=_FORWARD_PREFIX, default="ex"
    ) == _graph_summary(baseline)


def test_derivative_custom_prefix_renames_and_nothing_else() -> None:
    """The gradient-path pass honours a custom prefix the same way."""
    derivative, _ = _program_passes()
    node_prefix, value_prefix = _default_prefixes()
    program = _mean_program()

    baseline = derivative(program)
    prefixed = derivative(program, **{_PARAMETER: _DERIVATIVE_PREFIX})

    minted = [
        node
        for node in prefixed.nodes
        if node.node_id.startswith(f"{_DERIVATIVE_PREFIX}n")
    ]
    assert minted, "the pass must mint under the caller's prefix"
    assert not any(
        node.node_id.startswith(node_prefix)
        or node.output_value_id.startswith(value_prefix)
        for node in prefixed.nodes
    )
    assert _renamed_summary(
        _program_summary(prefixed), custom=_DERIVATIVE_PREFIX, default="ex"
    ) == _program_summary(baseline)


def test_detailed_regions_report_the_custom_identifiers() -> None:
    """Provenance names the identifiers actually emitted, prefix included."""
    _, forward_detailed = _graph_passes()
    result = forward_detailed(_mean_graph(), **{_PARAMETER: _FORWARD_PREFIX})

    assert result.regions
    for region in result.regions:
        assert all(
            node_id.startswith(f"{_FORWARD_PREFIX}n")
            for node_id in region.emitted_node_ids
        )
        assert region.terminal_value_id.startswith(_FORWARD_PREFIX) or (
            region.terminal_value_id in result.graph.outputs
        )


# --------------------------------------------------------------------------
# The collision check still fires under a custom prefix
# --------------------------------------------------------------------------


def _mean_graph_behind_an_add(
    *, node_id: str = "n0", output_value_id: str = "v1"
) -> TensorGraph:
    """A mean whose operand is produced by one add under a chosen identifier.

    The identifier under test sits on the *add*, so it is present in the
    artifact without being carried by the mean -- which is what separates the
    minter's collision check from the mean's own reserved-construct check.
    """
    typespec = {"dtype": "f64", "shape": [3, 5]}
    add = TensorNodeRecord(
        node_id=node_id,
        output_value_id=output_value_id,
        operator=AddOperator(),
        op_params={},
        input_value_ids=["v0", "v0"],
        output_typespec=dict(typespec),
    )
    mean = TensorNodeRecord(
        node_id="n1",
        output_value_id="v2",
        operator=MeanOperator(),
        op_params={"axes": [0, 1], "keepdims": True},
        input_value_ids=[output_value_id],
        output_typespec={"dtype": "f64", "shape": [1, 1]},
    )
    return TensorGraph(
        nodes=[add, mean],
        inputs=[("v0", typespec)],
        outputs=["v2"],
    )


def test_a_minted_node_id_colliding_under_a_custom_prefix_is_rejected() -> None:
    """The fail-closed collision check follows the caller's prefix."""
    forward, _ = _graph_passes()
    graph = _mean_graph_behind_an_add(node_id=f"{_FORWARD_PREFIX}n0")

    with pytest.raises(AutodiffError) as failure:
        forward(graph, **{_PARAMETER: _FORWARD_PREFIX})

    assert failure.value.category == "malformed_derivative_ir"
    assert f"{_FORWARD_PREFIX}n0" in failure.value.message


def test_a_minted_value_id_colliding_under_a_custom_prefix_is_rejected() -> None:
    """A minted value id colliding with an existing one is rejected too."""
    forward, _ = _graph_passes()
    graph = _mean_graph_behind_an_add(output_value_id=f"{_FORWARD_PREFIX}v0")

    with pytest.raises(AutodiffError) as failure:
        forward(graph, **{_PARAMETER: _FORWARD_PREFIX})

    assert failure.value.category == "malformed_derivative_ir"
    assert f"{_FORWARD_PREFIX}v0" in failure.value.message


def test_the_same_graph_expands_cleanly_under_a_non_colliding_prefix() -> None:
    """The control: the collision is the identifier, not the graph."""
    forward, _ = _graph_passes()
    graph = _mean_graph_behind_an_add(node_id=f"{_FORWARD_PREFIX}n0")

    expanded = forward(graph, **{_PARAMETER: _DERIVATIVE_PREFIX})

    assert any(
        node.node_id.startswith(f"{_DERIVATIVE_PREFIX}n") for node in expanded.nodes
    )


def test_a_custom_prefix_does_not_disarm_the_reserved_construct_check() -> None:
    """A mean already spelled in a reserved namespace is still not expandable."""
    forward, _ = _graph_passes()
    node_prefix, _ = _default_prefixes()

    for reserved_mean_id in (f"{node_prefix}0", f"{_FORWARD_PREFIX}n0"):
        graph = TensorGraph(
            nodes=[
                TensorNodeRecord(
                    node_id=reserved_mean_id,
                    output_value_id="v1",
                    operator=MeanOperator(),
                    op_params={"axes": [0, 1], "keepdims": True},
                    input_value_ids=["v0"],
                    output_typespec={"dtype": "f64", "shape": [1, 1]},
                )
            ],
            inputs=[("v0", {"dtype": "f64", "shape": [3, 5]})],
            outputs=["v1"],
        )

        with pytest.raises(AutodiffError) as failure:
            forward(graph, **{_PARAMETER: _FORWARD_PREFIX})

        assert failure.value.category == "unsupported_reduction"
        assert reserved_mean_id in failure.value.message


# --------------------------------------------------------------------------
# Prefix validation
# --------------------------------------------------------------------------

_REJECTED_PREFIXES = [
    "",  # empty: would mint bare `n0`/`v0`, the tracer's own namespace
    " ",  # blank is not a namespace either
    0,  # not a string
    17,
    True,  # a bool reaching a string field is a construction defect
    ("ex",),
    b"ex",
    "v",  # the tracer's value namespace
    "value",
    "n",  # the tracer's node namespace
    "node",
    "d",  # the reverse transform's gradient value namespace
    "dn",  # the reverse transform's gradient node namespace
    "delta",
]


@pytest.mark.parametrize("prefix", _REJECTED_PREFIXES, ids=repr)
@pytest.mark.parametrize("index", range(4))
def test_an_unusable_prefix_is_rejected_by_every_pass(prefix: object, index: int) -> None:
    """Every public entry point refuses a prefix that cannot be a namespace."""
    pass_function = _passes()[index]
    graph, program = _traced_mean()
    artifact = graph if index < 2 else program

    with pytest.raises(AutodiffError) as failure:
        pass_function(artifact, **{_PARAMETER: prefix})

    assert failure.value.category == "malformed_derivative_ir"
    assert repr(prefix) in failure.value.message


def test_the_prefix_is_validated_before_any_rewriting() -> None:
    """A bad prefix is reported even when the artifact has nothing to rewrite."""
    forward, _ = _graph_passes()
    with TensorGraphBuilder() as trace:
        value = trace.input("value", dtype="f64", shape=(3, 5))
        output = value + value
    graph = trace.build(outputs=output)

    with pytest.raises(AutodiffError) as failure:
        forward(graph, **{_PARAMETER: "v"})

    assert failure.value.category == "malformed_derivative_ir"


def test_the_default_prefix_is_itself_accepted_when_passed_explicitly() -> None:
    """`"ex"` is a legal caller prefix and reproduces the default namespace."""
    forward, _ = _graph_passes()
    graph = _mean_graph()

    assert _graph_summary(forward(graph, **{_PARAMETER: "ex"})) == _graph_summary(
        forward(graph)
    )


def test_the_error_category_vocabulary_is_unchanged() -> None:
    """Prefix validation reuses an existing category; none is added."""
    from tinychain.autodiff.protocol import AUTODIFF_ERROR_CATEGORIES

    assert "malformed_derivative_ir" in AUTODIFF_ERROR_CATEGORIES


# --------------------------------------------------------------------------
# The composition this change exists to enable
# --------------------------------------------------------------------------


def _forward_mean_expansion(graph: TensorGraph) -> TensorGraph:
    """The forward pass as a caller would compose it: one positional argument."""
    from tinychain.autodiff import expand_mean_graph

    return expand_mean_graph(graph, **{_PARAMETER: _FORWARD_PREFIX})


def _derivative_mean_expansion(program: DerivativeProgram) -> DerivativeProgram:
    """The gradient-path pass, namespaced apart from the forward one."""
    from tinychain.autodiff import expand_mean_derivative_program

    return expand_mean_derivative_program(program, **{_PARAMETER: _DERIVATIVE_PREFIX})


def test_the_shared_default_namespace_is_what_makes_composition_ambiguous() -> None:
    """The control: both passes at their default prefix collide, as reported."""
    forward, _ = _graph_passes()
    derivative, _ = _program_passes()
    graph, program = _traced_mean()

    with pytest.raises(AutodiffError) as failure:
        analyze_derivative_dependencies(
            derivative(program),
            forward_graph=forward(graph),
            seed_value_ids=["seed"],
            outputs=list(program.output_gradients),
        )

    assert failure.value.category == "ambiguous_producer"


def test_distinct_prefixes_make_the_pair_analyzable() -> None:
    """Namespaced apart, the same two artifacts analyze without ambiguity."""
    graph, program = _traced_mean()

    analysis = analyze_derivative_dependencies(
        _derivative_mean_expansion(program),
        forward_graph=_forward_mean_expansion(graph),
        seed_value_ids=["seed"],
        outputs=list(program.output_gradients),
    )

    assert analysis is not None


def _expansion_capable_registry() -> OperationHandlerRegistry:
    """The training-step registry widened with the handlers expansion needs.

    Composed out of handlers the shared reference consumer already registers --
    none is defined here.
    """
    composed = OperationHandlerRegistry()
    seen: set[type] = set()
    for source in (training_step_registry(), limited_operation_registry()):
        for operator_type in source.supported_types():
            if operator_type in seen:
                continue
            seen.add(operator_type)
            composed.register(source.lookup(operator_type()))
    return composed


def _compile_with(forward_expansions, derivative_expansions):
    with tc.state.scoped_context():
        return training_step.compile_training_step(
            e2e.residual_loss,
            inputs=e2e.ONE_PARAMETER_INPUTS,
            parameters=("w",),
            optimizer=SGD(),
            optimizer_inputs={"learning_rate": e2e.SCALAR_SPEC},
            handlers=_expansion_capable_registry(),
            bind_input=e2e.placeholder_binding,
            forward_expansions=forward_expansions,
            derivative_expansions=derivative_expansions,
        )


def test_both_passes_in_one_compile_share_a_namespace_and_fail() -> None:
    """The reproduction of §17.7: unnamespaced, one compile cannot hold both."""
    from tinychain.autodiff import (
        expand_mean_derivative_program,
        expand_mean_graph,
    )

    with pytest.raises(AutodiffError) as failure:
        _compile_with((expand_mean_graph,), (expand_mean_derivative_program,))

    assert failure.value.category == "ambiguous_producer"


def test_both_passes_in_one_compile_succeed_with_distinct_prefixes() -> None:
    """The requirement of §17.7: both real passes composed in a single compile."""
    record = _compile_with(
        (_forward_mean_expansion,), (_derivative_mean_expansion,)
    )

    assert record.provenance.forward_expansions == ("_forward_mean_expansion",)
    assert record.provenance.derivative_expansions == ("_derivative_mean_expansion",)

    forward_ids = _emitted_node_ids(record.lowered_forward_graph)
    derivative_ids = _emitted_node_ids(record.lowered_derivative_program)
    assert forward_ids and all(
        node_id.startswith(f"{_FORWARD_PREFIX}n") for node_id in forward_ids
    )
    assert derivative_ids and all(
        node_id.startswith(f"{_DERIVATIVE_PREFIX}n") for node_id in derivative_ids
    )
    assert not set(forward_ids) & set(derivative_ids)
