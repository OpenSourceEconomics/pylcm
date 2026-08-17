"""Continuation demand and representation travel through one explicit contract."""

from types import SimpleNamespace

import jax.numpy as jnp

import _lcm.engine as engine
import _lcm.solution.contract as contract
from _lcm.continuation import EGMContinuationLayout, EGMContinuationSpec
from _lcm.egm.carry import EGMCarry
from _lcm.reachability import build_phase_reachability
from _lcm.regime_building.processing import _continuation_targets
from lcm.solvers import GridSearch


def _template() -> EGMCarry:
    """Return a finite one-row continuation template."""
    row = jnp.asarray([[0.0, 1.0]])
    return EGMCarry(
        endog_grid=row,
        value=row,
        marginal_utility=jnp.ones_like(row),
        taste_shock_scale=jnp.asarray(0.0),
    )


def test_continuation_payload_alias_has_one_engine_definition():
    """The engine and solver contract import the same continuation alias."""
    assert engine.ContinuationPayload is contract.ContinuationPayload


def test_solution_kernels_bundle_template_with_its_layout():
    """A continuation template cannot be published without its interpretation."""
    layout = EGMContinuationLayout(
        retains_discrete_action_rows=False,
        rows_share_state_grid=True,
        n_stacked_candidates=3,
    )
    spec = EGMContinuationSpec(template=_template(), layout=layout)
    kernels = contract.SolutionKernels(period_kernels={}, continuation_spec=spec)

    assert kernels.continuation_template is spec.template
    assert kernels.continuation_spec.layout is layout


def test_grid_search_declares_the_layout_of_engine_produced_carries():
    """A brute-force child has shared-grid rows and no discrete-action rows."""
    assert GridSearch().egm_continuation_layout == EGMContinuationLayout(
        retains_discrete_action_rows=False,
        rows_share_state_grid=True,
    )


def test_only_reachable_targets_of_continuation_readers_publish_carries():
    """Unreachable and value-only targets do not receive an EGM carry artifact."""
    reachability = build_phase_reachability(
        n_periods=2,
        active_periods_by_regime={
            "reader": (0,),
            "value_only": (0,),
            "needed": (1,),
            "unused": (1,),
        },
        candidate_targets_by_source={
            "reader": ("needed",),
            "value_only": ("unused",),
            "needed": (),
            "unused": (),
        },
        terminal_regimes=("needed", "unused"),
    )
    regimes = {
        "reader": SimpleNamespace(
            solver=SimpleNamespace(requires_continuation=True)
        ),
        "value_only": SimpleNamespace(
            solver=SimpleNamespace(requires_continuation=False)
        ),
        "needed": SimpleNamespace(
            solver=SimpleNamespace(requires_continuation=False)
        ),
        "unused": SimpleNamespace(
            solver=SimpleNamespace(requires_continuation=False)
        ),
    }

    assert _continuation_targets(
        user_regimes=regimes,
        phase_reachability=reachability,
    ) == frozenset({"needed"})
