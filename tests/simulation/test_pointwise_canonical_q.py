"""Pointwise canonical Q evaluation for off-grid action scoring.

Simulation decides discrete branches by maximizing the canonical state-action
value `Q`. On the action grid that maximization is `argmax_and_max_Q_over_a`;
off the grid a candidate continuous action needs the *same* `Q` evaluated at one
action value per subject. `regime.simulation.Q_and_F[period]` is that pointwise
evaluator: it shares the model DAG, transitions, constraints, aggregators,
params, and next-period value arrays with the grid maximization, so a value it
reports is comparable with the grid winner's value.
"""

import dataclasses
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.simulation.simulate as simulate_module
from _lcm.simulation.simulate import _lookup_values_from_indices
from lcm import AgeGrid, DiscreteGrid, LogSpacedGrid, Model, categorical
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.test_models.deterministic import retirement_only
from tests.test_models.deterministic.dcegm_variants import (
    DCEGM_SOLVER,
    dcegm_retirement,
    get_retirement_only_params,
)

_DISCOUNT_FACTOR = 0.98
_BONUS = 10.0
_EFFORT_COST = 0.57


@categorical(ordered=False)
class BonusChoice:
    skip: ScalarInt
    take: ScalarInt


def _bonus_utility(
    consumption: ContinuousAction, take_bonus: DiscreteAction
) -> FloatND:
    return jnp.log(consumption) - _EFFORT_COST * take_bonus


def _bonus_resources(wealth: ContinuousState, take_bonus: DiscreteAction) -> FloatND:
    return wealth + _BONUS * take_bonus


def _savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _bequest_utility(wealth: ContinuousState, age: float) -> FloatND:
    return (age / 50.0) * jnp.log(wealth)


def _bonus_model() -> Model:
    solver = dataclasses.replace(DCEGM_SOLVER, envelope="mss")
    alive = dcegm_retirement.replace(
        active=lambda age: age < 50,
        solver=solver,
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


def test_pointwise_canonical_q_at_the_grid_argmax_action_reproduces_its_value():
    """`Q_and_F` at the grid-argmax action returns that action's grid value.

    The pointwise evaluator and the grid maximization must agree on the value of
    one and the same state-action pair; otherwise an off-grid candidate scored
    pointwise could not be compared with the grid winner.
    """
    model = _bonus_model()
    params = get_retirement_only_params(2, discount_factor=_DISCOUNT_FACTOR)
    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")

    regime = model._regimes["retirement"]
    period = 0
    age = jnp.asarray(model.ages.period_to_age(period))
    wealth = jnp.asarray([12.0, 37.5, 88.25, 210.0])

    flat_params = model._process_params(params)["retirement"]
    # The flat argmax index is unravelled against the canonical action order
    # (discrete actions first), so the grid mapping must follow that order.
    action_names = regime.solution.state_action_space(
        regime_params=flat_params
    ).action_names
    action_grids = MappingProxyType(
        {
            name: jnp.asarray(regime.simulation.grids[name].to_jax())
            for name in action_names
        }
    )
    next_regime_to_V_arr = period_to_regime_to_V_arr[period + 1]

    grid_indices, grid_values = regime.simulation.argmax_and_max_Q_over_a[period](
        wealth=wealth,
        **action_grids,
        next_regime_to_V_arr=next_regime_to_V_arr,
        **flat_params,
        period=jnp.int32(period),
        age=age,
    )
    grid_actions = _lookup_values_from_indices(
        flat_indices=grid_indices, grids=action_grids
    )

    pointwise_values, feasible = regime.simulation.Q_and_F[period](
        wealth=wealth,
        **grid_actions,
        next_regime_to_V_arr=next_regime_to_V_arr,
        **flat_params,
        period=jnp.int32(period),
        age=age,
    )

    assert bool(jnp.all(feasible))
    np.testing.assert_allclose(
        np.asarray(pointwise_values), np.asarray(grid_values), rtol=1e-6
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the off-grid policy read is gated on a crossing-complete env"
        "elope and no shipped backend qualifies; this is the acceptance criteri"
        "on for reopening the gate, so it must start passing again the moment one does"
    ),
)
def test_off_grid_replacement_never_scores_below_the_grid_pair():
    """Every emitted (branch, action) pair is worth at least the grid pair.

    The off-grid read replaces the action-grid argmax only where the replacement
    scores at least as high under the canonical `Q`, so simulating can never
    return a pair the finite-grid decision would have beaten.
    """
    model = _bonus_model()
    params = get_retirement_only_params(2, discount_factor=_DISCOUNT_FACTOR)
    off_grid_wealth = jnp.asarray([12.0, 37.5, 88.25, 210.0])
    n_subjects = off_grid_wealth.shape[0]

    emitted_vs_grid: list[tuple[np.ndarray, np.ndarray]] = []
    original = simulate_module._redecide_branch_and_read_policy

    def record(**kwargs):
        emitted_actions = original(**kwargs)
        emitted_values, _ = kwargs["score_actions"](candidate_actions=emitted_actions)
        emitted_vs_grid.append(
            (np.asarray(emitted_values), np.asarray(kwargs["grid_values"]))
        )
        return emitted_actions

    simulate_module._redecide_branch_and_read_policy = record  # ty: ignore[invalid-assignment]
    try:
        model.simulate(
            params=params,
            initial_conditions={
                "wealth": off_grid_wealth,
                "age": jnp.full(n_subjects, 40.0),
                "regime_id": jnp.full(
                    n_subjects, retirement_only.RetirementOnlyRegimeId.retirement
                ),
            },
            period_to_regime_to_V_arr=None,
            log_level="debug",
        )
    finally:
        simulate_module._redecide_branch_and_read_policy = original

    assert emitted_vs_grid
    for emitted_values, grid_pair_values in emitted_vs_grid:
        assert np.all(emitted_values >= grid_pair_values - 1e-6)
