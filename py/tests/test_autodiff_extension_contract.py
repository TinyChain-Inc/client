"""Public export surface tests for the extension contract.

Three modules -- structured dependency analysis, extensible program lowering,
and traced optimizer updates -- are complete and reviewed but only reachable
today by importing their private submodules directly. A downstream consumer
that wants a stable public contract must be able to reach every analysis,
compiler, handler, and update-tracing name from `tinychain.autodiff` itself,
the same way the artifact and route-derivative surfaces already are (see
`tinychain/autodiff/__init__.py`'s `__getattr__` over frozensets).

These tests pin that package-level surface. They do not re-test the
underlying behavior of `dependencies.py`, `lowering.py`, or `training.py`,
which is already covered by their own focused test modules.
"""

from __future__ import annotations

import pytest

import tinychain as tc


_DEPENDENCY_EXPORTS = frozenset(
    {
        "DEPENDENCY_PROVENANCE_DECLARED_INPUT",
        "DEPENDENCY_PROVENANCE_SEED_INPUT",
        "DEPENDENCY_PROVENANCE_FORWARD_CAPTURE",
        "DEPENDENCY_PROVENANCE_LOCAL_VALUE",
        "DEPENDENCY_PROVENANCE_ORDER",
        "DependencyAnalysis",
        "ValueDependency",
        "analyze_derivative_dependencies",
        "analyze_graph_dependencies",
    }
)

_LOWERING_EXPORTS = frozenset(
    {
        "LOWERING_CLAIM_HANDLER",
        "LOWERING_CLAIM_FUSION",
        "FusionContext",
        "FusionHook",
        "FusionResult",
        "LoweredOperation",
        "LoweredProgram",
        "OperationContext",
        "OperationHandler",
        "OperationHandlerRegistry",
        "lower_derivative_program",
        "lower_graph",
    }
)

_TRAINING_EXPORTS = frozenset(
    {
        "TracedUpdate",
        "sgd_update",
        "trace_parameter_update",
    }
)

_EXPANSION_EXPORTS = frozenset(
    {
        "BROADCAST_SCALE_EXPANSION",
        "FillDescriptor",
        "FillOperator",
        "MEAN_EXPANSION_FORWARD",
        "MeanDerivativeExpansionResult",
        "MeanExpansionRegion",
        "MeanGraphExpansionResult",
        "expand_mean_derivative_program",
        "expand_mean_derivative_program_detailed",
        "expand_mean_graph",
        "expand_mean_graph_detailed",
        "fill_descriptor",
    }
)


def test_dependency_analysis_surface_is_exported_from_autodiff_package() -> None:
    import tinychain.autodiff as autodiff
    from tinychain.autodiff import dependencies

    assert _DEPENDENCY_EXPORTS.issubset(set(autodiff.__all__))
    for export_name in _DEPENDENCY_EXPORTS:
        assert hasattr(autodiff, export_name)

    assert autodiff.DependencyAnalysis is dependencies.DependencyAnalysis
    assert autodiff.ValueDependency is dependencies.ValueDependency
    assert autodiff.analyze_graph_dependencies is dependencies.analyze_graph_dependencies
    assert (
        autodiff.analyze_derivative_dependencies
        is dependencies.analyze_derivative_dependencies
    )
    assert autodiff.DEPENDENCY_PROVENANCE_ORDER == dependencies.DEPENDENCY_PROVENANCE_ORDER

    assert not hasattr(tc, "DependencyAnalysis")
    assert not hasattr(tc, "analyze_graph_dependencies")


def test_program_lowering_surface_is_exported_from_autodiff_package() -> None:
    import tinychain.autodiff as autodiff
    from tinychain.autodiff import lowering

    assert _LOWERING_EXPORTS.issubset(set(autodiff.__all__))
    for export_name in _LOWERING_EXPORTS:
        assert hasattr(autodiff, export_name)

    assert autodiff.OperationHandlerRegistry is lowering.OperationHandlerRegistry
    assert autodiff.lower_graph is lowering.lower_graph
    assert autodiff.lower_derivative_program is lowering.lower_derivative_program
    assert autodiff.OperationContext is lowering.OperationContext
    assert autodiff.FusionContext is lowering.FusionContext
    assert autodiff.FusionResult is lowering.FusionResult

    assert not hasattr(tc, "OperationHandlerRegistry")
    assert not hasattr(tc, "lower_graph")


def test_traced_update_surface_is_exported_from_autodiff_package() -> None:
    import tinychain.autodiff as autodiff
    from tinychain.autodiff import training

    assert _TRAINING_EXPORTS.issubset(set(autodiff.__all__))
    for export_name in _TRAINING_EXPORTS:
        assert hasattr(autodiff, export_name)

    assert autodiff.TracedUpdate is training.TracedUpdate
    assert autodiff.sgd_update is training.sgd_update
    assert autodiff.trace_parameter_update is training.trace_parameter_update

    assert not hasattr(tc, "trace_parameter_update")
    assert not hasattr(tc, "sgd_update")


def test_expansion_surface_is_exported_from_autodiff_package() -> None:
    import tinychain.autodiff as autodiff
    from tinychain.autodiff import expansion

    assert _EXPANSION_EXPORTS.issubset(set(autodiff.__all__))
    for export_name in _EXPANSION_EXPORTS:
        assert hasattr(autodiff, export_name)

    assert autodiff.FillOperator is expansion.FillOperator
    assert autodiff.FillDescriptor is expansion.FillDescriptor
    assert autodiff.fill_descriptor is expansion.fill_descriptor
    assert autodiff.expand_mean_graph is expansion.expand_mean_graph
    assert autodiff.expand_mean_graph_detailed is expansion.expand_mean_graph_detailed
    assert (
        autodiff.expand_mean_derivative_program is expansion.expand_mean_derivative_program
    )
    assert (
        autodiff.expand_mean_derivative_program_detailed
        is expansion.expand_mean_derivative_program_detailed
    )
    assert autodiff.MeanExpansionRegion is expansion.MeanExpansionRegion
    assert autodiff.MeanGraphExpansionResult is expansion.MeanGraphExpansionResult
    assert (
        autodiff.MeanDerivativeExpansionResult is expansion.MeanDerivativeExpansionResult
    )
    assert autodiff.MEAN_EXPANSION_FORWARD == expansion.MEAN_EXPANSION_FORWARD
    assert autodiff.BROADCAST_SCALE_EXPANSION == expansion.BROADCAST_SCALE_EXPANSION

    for export_name in _EXPANSION_EXPORTS:
        assert not hasattr(tc, export_name)


def test_unknown_autodiff_attribute_still_raises_attribute_error() -> None:
    """The lazy loader must keep failing closed for names outside every export set."""
    import tinychain.autodiff as autodiff

    with pytest.raises(AttributeError):
        autodiff.definitely_not_a_real_autodiff_export
