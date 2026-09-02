"""Exact stored-value reads declared by GridSearch period kernels."""

from types import MappingProxyType
from typing import cast

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
)
from _lcm.execution.output_layout import VALUE
from _lcm.execution.value_transfer import (
    ValueArtifactKind,
    ValueInputChannel,
)
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.grid_search import (
    _edge_reference_regimes_for_targets,
    _GridSearchArgumentBuilder,
    _GridSearchPeriodKernel,
    _target_value_accesses,
)
from tests.simulation.test_aot_collective_and_gated import _make_consent_model


def _kernel(
    *,
    target_regimes: tuple[str, ...] = (),
    edge_target_regimes: tuple[str, ...] = (),
    same_period_ref_regimes: tuple[str, ...] = (),
    edge_reference_regimes: tuple[str, ...] = (),
) -> _GridSearchPeriodKernel:
    """Build a period-3 adapter whose access declarations need no JAX execution."""
    requirements = CoreExecutionRequirements(
        target_value_accesses=_target_value_accesses(
            regime_name="source",
            period=3,
            target_regimes=target_regimes,
            edge_target_regimes=edge_target_regimes,
            same_period_ref_regimes=same_period_ref_regimes,
            edge_reference_regimes=edge_reference_regimes,
        )
    )
    program = CoreProgram(
        name="main",
        function=lambda: None,
        argument_builder=lambda _context: MappingProxyType({}),
        requirements=requirements,
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason="test_dense_value_accesses",
    )
    return _GridSearchPeriodKernel(_core_programs=MappingProxyType({"main": program}))


def _accesses(kernel: _GridSearchPeriodKernel):
    """Read exact value inputs through the sole native declaration."""
    return kernel.core_programs()["main"].requirements.target_value_accesses


def test_declares_exact_ordinary_next_value_access() -> None:
    """An ordinary continuation reads the target regime's next-period V."""
    (access,) = _accesses(_kernel(target_regimes=("retired",)))

    assert access.target.kind is ValueArtifactKind.REGIME_VALUE
    assert access.target.period == 4
    assert access.target.regime == "retired"
    assert access.target.target_regime is None
    assert access.source.source_period == 3
    assert access.source.source_regime == "source"
    assert access.source.core_key == "main"
    assert access.source.channel is ValueInputChannel.NEXT_REGIME_VALUE
    assert access.source.path == ("retired",)


def test_declares_gated_continuation_in_place_of_raw_target_value() -> None:
    """A gated target names source-owned Wbar at the ordinary next-V leaf."""
    (access,) = _accesses(
        _kernel(
            target_regimes=("couple",),
            edge_target_regimes=("couple",),
        )
    )

    assert access.target.kind is ValueArtifactKind.GATED_CONTINUATION
    assert access.target.period == 4
    assert access.target.regime == "source"
    assert access.target.target_regime == "couple"
    assert access.source.channel is ValueInputChannel.NEXT_REGIME_VALUE
    assert access.source.path == ("couple",)


def test_declares_exact_same_period_reference_value_access() -> None:
    """A value constraint reads its reference regime at the source period."""
    (access,) = _accesses(_kernel(same_period_ref_regimes=("single",)))

    assert access.target.kind is ValueArtifactKind.REGIME_VALUE
    assert access.target.period == 3
    assert access.target.regime == "single"
    assert access.source.source_period == 3
    assert access.source.channel is ValueInputChannel.SAME_PERIOD_VALUE
    assert access.source.path == ("single",)


def test_declares_exact_next_period_edge_reference_value_access() -> None:
    """A gate projection or fallback reads its regime at the landing period."""
    (access,) = _accesses(_kernel(edge_reference_regimes=("outside",)))

    assert access.target.kind is ValueArtifactKind.REGIME_VALUE
    assert access.target.period == 4
    assert access.target.regime == "outside"
    assert access.source.source_period == 3
    assert access.source.channel is ValueInputChannel.EDGE_REFERENCE_VALUE
    assert access.source.path == ("outside",)


def test_final_period_processed_kernel_declares_no_next_value_accesses() -> None:
    """The final source node neither asks reachability for nor declares a successor."""
    model = _make_consent_model(n_subjects=None)
    kernels = model._regimes["single_terminal"].solution.period_kernels
    kernel = kernels[model.ages.n_periods - 1]

    assert isinstance(kernel, _GridSearchPeriodKernel)
    program = kernel.core_programs()["main"]
    builder = cast("_GridSearchArgumentBuilder", program.argument_builder)
    assert builder.edge_reference_regimes == ()
    assert program.requirements.target_value_accesses == ()


def test_edge_references_are_filtered_to_this_periods_reachable_targets() -> None:
    """An edge outside the reachable-target set contributes no references."""
    model = _make_consent_model(n_subjects=None)
    context = object.__new__(SolverBuildContext)
    object.__setattr__(context, "regime_name", "single")
    object.__setattr__(context, "user_regimes", model.user_regimes)

    assert (
        _edge_reference_regimes_for_targets(
            context=context,
            target_regimes=(),
        )
        == ()
    )
    assert _edge_reference_regimes_for_targets(
        context=context,
        target_regimes=("married_terminal",),
    ) == ("single_terminal",)
