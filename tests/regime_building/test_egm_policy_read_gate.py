"""Build-time gates for the off-grid simulation policy read.

`SimulationPhase.egm_policy_read` is set only where replaying the solve-phase
policy is valid; every other configuration keeps the grid-argmax simulate path.
"""

import dataclasses

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, NormalIIDProcess, fixed_transition
from lcm.typing import ContinuousState, FloatND, ScalarInt
from lcm_examples.iskhakov_et_al_2017 import WEALTH_GRID, next_wealth_from_savings
from tests.test_models.deterministic import retirement_only
from tests.test_models.deterministic.dcegm_variants import (
    DCEGM_SOLVER,
    dcegm_retirement,
    dead,
)
from tests.test_models.ds2024_housing import build_model


def test_negm_regime_does_not_qualify_for_the_policy_read():
    """A NEGM regime keeps the grid path: its published policy is keeper-only.

    The nested period kernel maximizes the value over keeper and adjuster
    candidates but publishes only the keeper's inner consumption function, so
    replaying it would pair an adjuster-won value with the keeper's policy.
    """
    model = build_model(variant="negm", n_grid=12, n_periods=3)
    assert model._regimes["alive"].simulation.egm_policy_read is None


@pytest.mark.parametrize("backend", ["rfc", "ltm"])
def test_non_crossing_envelope_backends_do_not_qualify(backend: str):
    """RFC/LTM rows keep the grid path: they publish no crossing topology.

    Those backends leave the envelope switch between two retained nodes, so
    linear policy interpolation across the switch would mix two branch
    policies into an action belonging to neither.
    """
    model = _retirement_model_with_backend(backend)
    assert model._regimes["retirement"].simulation.egm_policy_read is None


@pytest.mark.parametrize("backend", ["fues", "mss"])
def test_crossing_inserting_envelope_backends_qualify(backend: str):
    """FUES/MSS rows qualify: duplicated crossing abscissae carry the switch."""
    model = _retirement_model_with_backend(backend)
    assert model._regimes["retirement"].simulation.egm_policy_read is not None


def test_bounded_fues_scan_does_not_qualify_for_the_policy_read():
    """A finite FUES scan window keeps the grid path: its row may miss crossings.

    Only the exhaustive scan (`fues_n_points_to_scan=None`) is proven to find a
    segment's continuation when more than the window's worth of off-segment
    candidates interleave. A bounded window may drop that crossing and retain
    the dominated interlopers, so the row lacks the switch topology a
    branch-faithful policy read interpolates over.
    """
    solver = dataclasses.replace(
        DCEGM_SOLVER, upper_envelope="fues", fues_n_points_to_scan=8
    )
    model = _model_from_alive(
        dcegm_retirement.replace(active=lambda age: age < 50, solver=solver)
    )
    assert model._regimes["retirement"].simulation.egm_policy_read is None


def test_bounded_scan_setting_does_not_disqualify_the_mss_backend():
    """MSS qualifies regardless of `fues_n_points_to_scan`: the knob is FUES-only.

    MSS inserts the exact segment crossing by construction, so a scan-window
    setting left on the solver never touches its published row.
    """
    solver = dataclasses.replace(
        DCEGM_SOLVER, upper_envelope="mss", fues_n_points_to_scan=8
    )
    model = _model_from_alive(
        dcegm_retirement.replace(active=lambda age: age < 50, solver=solver)
    )
    assert model._regimes["retirement"].simulation.egm_policy_read is not None


def test_process_state_regime_does_not_qualify_for_the_policy_read():
    """A regime with a continuous stochastic-process state keeps the grid path.

    A process state is stored as a node-valued row axis, but its simulation
    transition draws a genuinely continuous value that need not land on a
    process node. Selecting one row by nearest node would read the policy of
    the wrong process-state conditional; interpolating across process rows is
    the tracked follow-up.
    """
    model = _model_from_alive(
        dcegm_retirement.replace(
            active=lambda age: age < 50,
            states={
                "wealth": WEALTH_GRID,
                "wage_shock": NormalIIDProcess(
                    n_points=5, gauss_hermite=True, mu=0.0, sigma=0.3
                ),
            },
            functions={
                **dict(dcegm_retirement.functions),
                "resources": _resources_with_shock,
            },
        )
    )
    assert model._regimes["retirement"].simulation.egm_policy_read is None


def test_asset_row_regime_does_not_qualify_for_the_policy_read():
    """An asset-row DCEGM regime keeps the grid path: rows are pointwise optima.

    When a savings-stage function reads the current Euler state, DC-EGM solves
    per exogenous asset node and publishes one optimal point per node rather
    than a crossing-complete resources-space row. Interpolating across nodes
    would mix two endogenous branches wherever the winning branch changes
    between adjacent nodes.
    """
    model = _model_from_alive(
        dcegm_retirement.replace(
            active=lambda age: age < 50,
            transition=_next_regime_reads_wealth,
        )
    )
    assert model._regimes["retirement"].simulation.egm_policy_read is None


def test_passive_state_regime_does_not_qualify_for_the_policy_read():
    """A regime with a passive continuous state keeps the grid path.

    Each published row is already the upper-envelope policy conditional on one
    passive-state node. When the winning endogenous branch differs between two
    passive nodes, linearly blending their policies yields an action from
    neither branch; the passive read returns once conditional-value
    re-decision across the passive axis is available.
    """
    skill_grid = LinSpacedGrid(start=0.5, stop=1.5, n_points=5)
    alive = dcegm_retirement.replace(
        active=lambda age: age < 50,
        states={"wealth": WEALTH_GRID, "skill": skill_grid},
        state_transitions={
            "wealth": next_wealth_from_savings,
            "skill": fixed_transition("skill"),
        },
        functions={**dict(dcegm_retirement.functions), "utility": _skill_alive_utility},
    )
    dead_regime = dead.replace(
        states={"wealth": WEALTH_GRID, "skill": skill_grid},
        functions={"utility": _skill_bequest_utility},
    )
    model = Model(
        regimes={"retirement": alive, "dead": dead_regime},
        ages=AgeGrid(start=40, stop=50, step="10Y"),
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )
    assert model._regimes["retirement"].simulation.egm_policy_read is None


def _retirement_model_with_backend(backend: str) -> Model:
    solver = dataclasses.replace(DCEGM_SOLVER, upper_envelope=backend)
    return _model_from_alive(
        dcegm_retirement.replace(active=lambda age: age < 50, solver=solver)
    )


def _model_from_alive(alive, dead_states=None) -> Model:
    dead_regime = dead if dead_states is None else dead.replace(states=dead_states)
    return Model(
        regimes={"retirement": alive, "dead": dead_regime},
        ages=AgeGrid(start=40, stop=50, step="10Y"),
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )


def _resources_with_shock(
    wealth: ContinuousState, wage_shock: ContinuousState
) -> FloatND:
    return wealth + wage_shock


def _skill_alive_utility(consumption: FloatND, skill: ContinuousState) -> FloatND:
    # `skill` rides along as a passive state; the flow utility keeps it in the
    # DAG (0.0 * skill) without changing the consumption optimum.
    return jnp.log(consumption) + 0.0 * skill


def _skill_bequest_utility(wealth: ContinuousState, skill: ContinuousState) -> FloatND:
    return skill * jnp.log(wealth)


def _next_regime_reads_wealth(
    age: int, wealth: ContinuousState, final_age_alive: float
) -> ScalarInt:
    alive = (age < final_age_alive) & (wealth > -1.0e30)
    return jnp.where(
        alive,
        retirement_only.RetirementOnlyRegimeId.retirement,
        retirement_only.RetirementOnlyRegimeId.dead,
    ).astype("int32")
