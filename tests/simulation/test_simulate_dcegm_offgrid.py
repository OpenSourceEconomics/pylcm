"""Off-grid DC-EGM forward simulation: the continuous action is interpolated.

Standard EGM (Carroll 2006; Iskhakov-Jørgensen-Rust-Schjerning 2017; Druedahl
2021) simulates the continuous choice by interpolating the refined consumption
function at the simulated state, not by an argmax over the action grid. The
solve publishes that function (`EGMSimPolicy`); simulation interpolates it at
each subject's resources.

The spec below is the closed-form two-period retirement model with log utility,
zero interest, and an age-50 bequest `(age/50) log(wealth)`: optimal consumption
is `c* = wealth / (1 + beta)` at every resources level. Seeding subjects at
wealth strictly between consumption-grid nodes, the simulated consumption must
hit `c*` — which a grid-restricted argmax cannot.
"""

import jax.numpy as jnp
import numpy as np

from lcm import AgeGrid, LinSpacedGrid, LogSpacedGrid, Model, Phased, fixed_transition
from lcm.regime import Regime as UserRegime
from lcm.typing import ContinuousState, FloatND
from lcm_examples.iskhakov_et_al_2017 import WEALTH_GRID, next_wealth_from_savings
from tests.test_models.deterministic import base, dcegm_variants, retirement_only
from tests.test_models.deterministic.dcegm_variants import (
    dcegm_retirement,
    get_retirement_only_params,
)

_DISCOUNT_FACTOR = 0.98


def _bequest_utility(wealth: ContinuousState, age: float) -> FloatND:
    return (age / 50.0) * jnp.log(wealth)


def _closed_form_model() -> Model:
    bequest_dead = UserRegime(
        transition=None,
        states={"wealth": LogSpacedGrid(start=0.25, stop=400.0, n_points=400)},
        functions={"utility": _bequest_utility},
    )
    return Model(
        regimes={
            "retirement": dcegm_retirement.replace(active=lambda age: age < 50),
            "dead": bequest_dead,
        },
        ages=AgeGrid(start=40, stop=50, step="10Y"),
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )


def test_dcegm_simulated_consumption_is_off_grid_closed_form():
    """Simulated consumption equals `wealth / (1 + beta)` at off-grid wealth.

    A grid-restricted argmax snaps consumption to the action grid and cannot hit
    the closed form between nodes; off-grid interpolation of the published policy
    must.
    """
    model = _closed_form_model()
    params = get_retirement_only_params(2, discount_factor=_DISCOUNT_FACTOR)

    # Seed subjects at wealth strictly between consumption-grid nodes (the
    # consumption grid shares spacing with the wealth grid in this example).
    wealth_nodes = np.asarray(WEALTH_GRID.to_jax())
    off_grid_wealth = 0.5 * (wealth_nodes[3:-1] + wealth_nodes[4:])
    initial_conditions = {
        "wealth": jnp.asarray(off_grid_wealth),
        "age": jnp.full(off_grid_wealth.shape, 40.0),
        "regime_id": jnp.full(
            off_grid_wealth.shape,
            retirement_only.RetirementOnlyRegimeId.retirement,
        ),
    }

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    df = result.to_dataframe()
    period_0 = df.query("period == 0")
    consumption = period_0["consumption"].to_numpy()
    expected = off_grid_wealth / (1.0 + _DISCOUNT_FACTOR)
    np.testing.assert_allclose(consumption, expected, rtol=2e-2)


def test_discrete_action_regime_keeps_the_grid_consumption_path():
    """With a discrete work choice, simulated consumption stays on the grid.

    The discrete branch is chosen from grid-restricted Q values; a branch
    whose refined conditional optimum lies between action-grid nodes can lose
    that comparison yet win after continuous refinement, so replacing only
    the continuous action could pair the refined policy with the wrong
    branch. Until branch re-decision from published conditional values
    exists, such regimes keep the grid-argmax consumption.
    """
    n_periods = 3
    model = dcegm_variants.get_full_model("dcegm", n_periods)
    params = dcegm_variants.get_full_params(n_periods)

    wealth_nodes = np.asarray(WEALTH_GRID.to_jax())
    off_grid_wealth = 0.5 * (wealth_nodes[5:9] + wealth_nodes[6:10])
    n_subjects = off_grid_wealth.shape[0]
    initial_conditions = {
        "wealth": jnp.asarray(off_grid_wealth),
        "age": jnp.full(n_subjects, 40.0),
        "regime_id": jnp.full(n_subjects, base.RegimeId.working_life),
    }

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    period_0 = (
        result.to_dataframe(use_labels=False)
        .query("period == 0 and regime_name == 'working_life'")
        .sort_values("subject_id")
    )
    consumption = period_0["consumption"].to_numpy()

    consumption_nodes = np.asarray(dcegm_variants.CONSUMPTION_GRID.to_jax())
    node_distance = np.abs(consumption[:, None] - consumption_nodes[None, :]).min(
        axis=1
    )
    np.testing.assert_allclose(node_distance, 0.0, atol=1e-12)


def _skill_bequest_utility(
    wealth: ContinuousState, skill: ContinuousState, age: float
) -> FloatND:
    return skill * (age / 50.0) * jnp.log(wealth)


def _skill_alive_utility(
    consumption: ContinuousState, skill: ContinuousState
) -> FloatND:
    # The zero-weight `skill` term keeps the FOC at `1/c` while satisfying the
    # every-state-is-used model validation; `skill` matters through the
    # bequest weight only.
    return jnp.log(consumption) + 0.0 * skill


def _skill_model() -> Model:
    """Two-period model whose optimal consumption depends on a passive state.

    Alive: `log(c)`; dead bequest: `skill * log(wealth)`. Optimal consumption
    is `c* = wealth / (1 + beta * skill)` — the passive `skill` axis shifts
    the policy row as a function of resources, so an off-grid skill
    exercises the cross-row blend.
    """
    skill_grid = LinSpacedGrid(start=0.5, stop=1.5, n_points=5)
    alive = dcegm_retirement.replace(
        active=lambda age: age < 50,
        states={"wealth": WEALTH_GRID, "skill": skill_grid},
        state_transitions={
            "wealth": next_wealth_from_savings,
            "skill": fixed_transition("skill"),
        },
        functions={
            **dict(dcegm_retirement.functions),
            "utility": _skill_alive_utility,
        },
    )
    bequest_dead = UserRegime(
        transition=None,
        states={
            "wealth": LogSpacedGrid(start=0.25, stop=400.0, n_points=400),
            "skill": skill_grid,
        },
        functions={"utility": _skill_bequest_utility},
    )
    return Model(
        regimes={"retirement": alive, "dead": bequest_dead},
        ages=AgeGrid(start=40, stop=50, step="10Y"),
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )


def test_passive_state_regime_keeps_the_grid_consumption_path():
    """With a passive continuous state, simulated consumption stays on the grid.

    Each published policy row is the upper-envelope consumption conditional on
    one passive-state node. Where the winning endogenous branch differs across
    passive nodes, blending the rows would read an action from neither branch,
    so passive regimes keep the grid-argmax consumption until conditional-value
    re-decision across the passive axis exists.
    """
    model = _skill_model()
    params = get_retirement_only_params(2, discount_factor=_DISCOUNT_FACTOR)

    wealth_nodes = np.asarray(WEALTH_GRID.to_jax())
    off_grid_wealth = 0.5 * (wealth_nodes[5:9] + wealth_nodes[6:10])
    off_grid_skill = np.array([0.625, 0.875, 1.125, 1.375])
    n_subjects = off_grid_wealth.shape[0]
    initial_conditions = {
        "wealth": jnp.asarray(off_grid_wealth),
        "skill": jnp.asarray(off_grid_skill),
        "age": jnp.full(n_subjects, 40.0),
        "regime_id": jnp.full(
            n_subjects, retirement_only.RetirementOnlyRegimeId.retirement
        ),
    }

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    period_0 = result.to_dataframe().query("period == 0")
    consumption = period_0["consumption"].to_numpy()

    consumption_nodes = np.asarray(dcegm_variants.CONSUMPTION_GRID.to_jax())
    node_distance = np.abs(consumption[:, None] - consumption_nodes[None, :]).min(
        axis=1
    )
    np.testing.assert_allclose(node_distance, 0.0, atol=1e-12)


def _phased_alive_utility_solve(consumption: ContinuousState) -> FloatND:
    return jnp.log(consumption)


def _phased_alive_utility_simulate(consumption: ContinuousState) -> FloatND:
    return 2.0 * jnp.log(consumption)


def test_phase_variant_utility_keeps_the_grid_consumption_path():
    """A phase-variant utility disables the solve-policy replay.

    The stored policy solves the solve-phase first-order condition; a
    `Phased` utility (or any phase-variant function) changes the
    simulate-phase optimum even under an unchanged `H`, so the regime keeps
    the grid-argmax consumption.
    """
    alive = dcegm_retirement.replace(
        active=lambda age: age < 50,
        functions={
            **dict(dcegm_retirement.functions),
            "utility": Phased(
                solve=_phased_alive_utility_solve,
                simulate=_phased_alive_utility_simulate,
            ),
        },
    )
    bequest_dead = UserRegime(
        transition=None,
        states={"wealth": LogSpacedGrid(start=0.25, stop=400.0, n_points=400)},
        functions={"utility": _bequest_utility},
    )
    model = Model(
        regimes={"retirement": alive, "dead": bequest_dead},
        ages=AgeGrid(start=40, stop=50, step="10Y"),
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )
    params = get_retirement_only_params(2, discount_factor=_DISCOUNT_FACTOR)

    wealth_nodes = np.asarray(WEALTH_GRID.to_jax())
    off_grid_wealth = 0.5 * (wealth_nodes[5:9] + wealth_nodes[6:10])
    initial_conditions = {
        "wealth": jnp.asarray(off_grid_wealth),
        "age": jnp.full(off_grid_wealth.shape, 40.0),
        "regime_id": jnp.full(
            off_grid_wealth.shape,
            retirement_only.RetirementOnlyRegimeId.retirement,
        ),
    }
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    consumption = result.to_dataframe().query("period == 0")["consumption"].to_numpy()

    consumption_nodes = np.asarray(dcegm_variants.CONSUMPTION_GRID.to_jax())
    node_distance = np.abs(consumption[:, None] - consumption_nodes[None, :]).min(
        axis=1
    )
    np.testing.assert_allclose(node_distance, 0.0, atol=1e-12)
