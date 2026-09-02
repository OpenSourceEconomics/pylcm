"""Exact stored-value reads declared by GridSearch period kernels."""

from _lcm.execution.value_transfer import (
    ValueArtifactKind,
    ValueInputChannel,
)
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.grid_search import (
    _edge_reference_regimes_for_targets,
    _GridSearchPeriodKernel,
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
    return _GridSearchPeriodKernel(
        core=lambda: None,
        regime_name="source",
        period=3,
        target_regimes=target_regimes,
        edge_target_regimes=edge_target_regimes,
        same_period_ref_regimes=same_period_ref_regimes,
        edge_reference_regimes=edge_reference_regimes,
    )


def test_declares_exact_ordinary_next_value_access() -> None:
    """An ordinary continuation reads the target regime's next-period V."""
    (access,) = _kernel(target_regimes=("retired",)).target_value_accesses(
        core_key="main"
    )

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
    (access,) = _kernel(
        target_regimes=("couple",),
        edge_target_regimes=("couple",),
    ).target_value_accesses(core_key="main")

    assert access.target.kind is ValueArtifactKind.GATED_CONTINUATION
    assert access.target.period == 4
    assert access.target.regime == "source"
    assert access.target.target_regime == "couple"
    assert access.source.channel is ValueInputChannel.NEXT_REGIME_VALUE
    assert access.source.path == ("couple",)


def test_declares_exact_same_period_reference_value_access() -> None:
    """A value constraint reads its reference regime at the source period."""
    (access,) = _kernel(same_period_ref_regimes=("single",)).target_value_accesses(
        core_key="main"
    )

    assert access.target.kind is ValueArtifactKind.REGIME_VALUE
    assert access.target.period == 3
    assert access.target.regime == "single"
    assert access.source.source_period == 3
    assert access.source.channel is ValueInputChannel.SAME_PERIOD_VALUE
    assert access.source.path == ("single",)


def test_declares_exact_next_period_edge_reference_value_access() -> None:
    """A gate projection or fallback reads its regime at the landing period."""
    (access,) = _kernel(edge_reference_regimes=("outside",)).target_value_accesses(
        core_key="main"
    )

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
    assert kernel.target_regimes == ()
    assert kernel.edge_reference_regimes == ()
    assert kernel.target_value_accesses(core_key="main") == ()


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
