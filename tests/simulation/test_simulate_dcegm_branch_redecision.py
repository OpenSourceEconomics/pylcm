"""Off-grid branch re-decision from published conditional values (DC-EGM).

For a qualifying regime with discrete actions, the solve publishes per-branch
conditional value and policy rows. Simulation re-decides the discrete branch at
each subject's state by comparing the branches' interpolated conditional values
— each at that branch's own resources — and reads the winning branch's policy
off-grid. Discrete-only user constraints exclude infeasible branches from the
comparison, exactly as they mask the grid argmax.
"""

import jax.numpy as jnp
import numpy as np

from lcm import AgeGrid, DiscreteGrid, LogSpacedGrid, Model, categorical
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from lcm_examples.iskhakov_et_al_2017 import WEALTH_GRID
from tests.test_models.deterministic import retirement_only
from tests.test_models.deterministic.dcegm_variants import (
    dcegm_retirement,
    get_retirement_only_params,
)

_DISCOUNT_FACTOR = 0.98
_BONUS = 10.0
_EFFORT_COST = 0.57
# Taking pays iff `(1 + beta) * log(1 + b / w) > kappa`, so the closed-form
# switch wealth is `w* = b / (exp(kappa / (1 + beta)) - 1)`.
_SWITCH_WEALTH = _BONUS / (np.exp(_EFFORT_COST / (1.0 + _DISCOUNT_FACTOR)) - 1.0)


@categorical(ordered=False)
class BonusChoice:
    skip: ScalarInt
    take: ScalarInt


def _bonus_utility(consumption: ContinuousAction, take_bonus: DiscreteAction) -> FloatND:
    return jnp.log(consumption) - _EFFORT_COST * take_bonus


def _bonus_resources(wealth: ContinuousState, take_bonus: DiscreteAction) -> FloatND:
    return wealth + _BONUS * take_bonus


def _savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _next_wealth(savings: FloatND, interest_rate: float) -> ContinuousState:
    return (1 + interest_rate) * savings


def _inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _bequest_utility(wealth: ContinuousState, age: float) -> FloatND:
    return (age / 50.0) * jnp.log(wealth)


def _bonus_model(constraints: dict | None = None) -> Model:
    alive = dcegm_retirement.replace(
        active=lambda age: age < 50,
        actions={
            "consumption": dcegm_retirement.actions["consumption"],
            "take_bonus": DiscreteGrid(BonusChoice),
        },
        functions={
            "utility": _bonus_utility,
            "resources": _bonus_resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse_marginal_utility,
        },
        state_transitions={"wealth": _next_wealth},
        constraints=constraints or {},
    )
    bequest_dead = UserRegime(
        transition=None,
        states={"wealth": LogSpacedGrid(start=0.25, stop=400.0, n_points=400)},
        functions={"utility": _bequest_utility},
    )
    return Model(
        regimes={"retirement": alive, "dead": bequest_dead},
        ages=AgeGrid(start=40, stop=50, step="10Y"),
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )


def _simulate_period_0(model: Model, off_grid_wealth: np.ndarray):
    params = get_retirement_only_params(2, discount_factor=_DISCOUNT_FACTOR)
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
    return result.to_dataframe(use_labels=False).query("period == 0")


def _off_grid_wealth_straddling_the_switch() -> np.ndarray:
    """Off-node wealth values on both sides of the closed-form switch."""
    wealth_nodes = np.asarray(WEALTH_GRID.to_jax())
    midpoints = 0.5 * (wealth_nodes[:-1] + wealth_nodes[1:])
    below = midpoints[(midpoints > 0.4 * _SWITCH_WEALTH) & (midpoints < 0.9 * _SWITCH_WEALTH)]
    above = midpoints[(midpoints > 1.1 * _SWITCH_WEALTH) & (midpoints < 3.0 * _SWITCH_WEALTH)]
    return np.concatenate([below[:4], above[:4]])


def test_branch_redecision_matches_the_closed_form_switch_and_policy():
    """Simulated branch and consumption hit their closed forms off-grid.

    With `resources = wealth + b * take` and utility `log(c) - kappa * take`,
    the conditional optimum is `c*_d = R_d / (1 + beta)` and taking pays iff
    `(1 + beta) * log(1 + b / wealth) > kappa`. Subjects seeded strictly
    between wealth-grid nodes on both sides of the switch must take exactly
    when the closed form says so, and consume the winning branch's off-grid
    optimum — a grid-restricted pair can do neither.
    """
    off_grid_wealth = _off_grid_wealth_straddling_the_switch()
    period_0 = _simulate_period_0(_bonus_model(), off_grid_wealth).sort_values(
        "subject_id"
    )

    take = period_0["take_bonus"].to_numpy()
    expected_take = (off_grid_wealth < _SWITCH_WEALTH).astype(take.dtype)
    np.testing.assert_array_equal(take, expected_take)

    consumption = period_0["consumption"].to_numpy()
    resources = off_grid_wealth + _BONUS * expected_take
    np.testing.assert_allclose(
        consumption, resources / (1.0 + _DISCOUNT_FACTOR), rtol=2e-2
    )


def _bonus_forbidden(take_bonus: DiscreteAction) -> BoolND:
    return take_bonus == 0


def test_branch_redecision_respects_discrete_action_constraints():
    """An infeasible branch cannot win the off-grid value comparison.

    With the bonus forbidden by a discrete-only constraint, every subject must
    skip — including those for whom taking would win on unconstrained value —
    and consume the skip branch's off-grid optimum `wealth / (1 + beta)`.
    """
    off_grid_wealth = _off_grid_wealth_straddling_the_switch()
    model = _bonus_model(constraints={"bonus_forbidden": _bonus_forbidden})
    period_0 = _simulate_period_0(model, off_grid_wealth).sort_values("subject_id")

    take = period_0["take_bonus"].to_numpy()
    np.testing.assert_array_equal(take, np.zeros_like(take))

    consumption = period_0["consumption"].to_numpy()
    np.testing.assert_allclose(
        consumption, off_grid_wealth / (1.0 + _DISCOUNT_FACTOR), rtol=2e-2
    )
