"""A NEGM durable's declared law of motion governs what the next period carries.

The household chooses a durable stock `s'` this period and enjoys its service
flow, but only `alpha * s'` survives into the next period. That factor lives
where any law of motion lives — in `state_transitions` — so the solver has to
apply it: the chosen stock is what the outer margin searches over, `alpha * s'`
is what the continuation is read at.

The oracle is the grid-search twin, which searches the same durable grid with a
dense consumption grid and evaluates the same declared law by ordinary backward
induction. Substituting the raw search node for the law carries a stock
`1 / alpha` times too large into every continuation, which moves the value far
outside the band the two methods share when they solve the same model.

The law may read the chosen stock, other states, and params. It may not read the
inner consumption-savings margin: NEGM evaluates it to find what the next period
carries, so a law depending on the inner choice would make the outer maximum
range over problems that are no longer independent.
"""

from itertools import pairwise

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    Regime,
    outer_unchanged,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.test_models import negm_serviceflow_toy as toy

_PARAMS = {"discount_factor": 0.95, "alive": {}}

# Share of the chosen durable stock that survives into the next period.
DEPRECIATION = 0.7

# A stock that grows instead — the law is not restricted to `alpha < 1`.
APPRECIATION = 1.2


def _scaled_law(alpha: float):
    """The NEGM regime's durable law `Z' = alpha * s'`."""

    def durable_transition(new_durable: ContinuousState) -> ContinuousState:
        return alpha * new_durable

    return durable_transition


def _scaled_law_brute(alpha: float):
    """The same law in the grid-search twin's action vocabulary."""

    def next_illiquid_brute(new_durable: ContinuousAction) -> ContinuousState:
        return alpha * new_durable

    return next_illiquid_brute


def _build_negm_model(alpha: float, *, durable_law=None) -> Model:
    """The service-flow toy with a scaled durable law, solved by NEGM."""
    alive = NestedConsumptionSavingsRegime(
        active=lambda age, n=toy.FINAL_AGE_ALIVE: age <= n,
        states={"wealth": toy.WEALTH_GRID, "illiquid": toy.ILLIQUID_GRID},
        state_transitions={
            "wealth": toy.next_wealth,
            "illiquid": durable_law if durable_law is not None else _scaled_law(alpha),
        },
        actions={
            "consumption": toy.CONSUMPTION_GRID,
            "illiquid_investment": toy.ILLIQUID_INVESTMENT_GRID,
        },
        transition=toy.next_regime,
        functions={
            "utility": toy.utility,
            "new_durable": toy.new_durable,
            "serviced_durable": toy.serviced_durable,
            "resources_before_outer_cost": toy.resources_before_outer_cost,
            "liquid_savings": toy.liquid_savings,
            "credited": toy.credited,
            "inverse_marginal_utility": toy.inverse_marginal_utility,
        },
        solver=toy.NEGM_SOLVER,
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources=NetOfAdjustmentCost(
                name_in_dag="resources",
                before_cost="resources_before_outer_cost",
                cost="credited",
            ),
            post_decision_state="liquid_savings",
        ),
        outer_continuous=OuterContinuousMargin(
            state="illiquid",
            action="illiquid_investment",
            post_decision_state="new_durable",
            no_adjustment=outer_unchanged,
        ),
    )
    return _model(alive)


def _build_brute_model(alpha: float) -> Model:
    """The same model solved by grid search over the durable and consumption."""
    alive = Regime(
        active=lambda age, n=toy.FINAL_AGE_ALIVE: age <= n,
        states={"wealth": toy.WEALTH_GRID, "illiquid": toy.ILLIQUID_GRID},
        state_transitions={
            "wealth": toy.next_wealth_brute,
            "illiquid": _scaled_law_brute(alpha),
        },
        actions={
            "consumption": toy.CONSUMPTION_GRID_BRUTE,
            "new_durable": toy.OUTER_GRID,
        },
        transition=toy.next_regime,
        functions={
            "utility": toy.utility,
            "serviced_durable": toy.serviced_durable_brute,
        },
        constraints={"feasible": toy.feasible},
    )
    return _model(alive)


def _model(alive: Regime) -> Model:
    return Model(
        regimes={"alive": alive, "dead": toy._build_dead_regime()},
        regime_id_class=toy.RegimeId,
        ages=AgeGrid(start=20, stop=20 + (toy.N_PERIODS - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": toy.FINAL_AGE_ALIVE},
    )


def _solve_period0_alive(model: Model) -> jnp.ndarray:
    return model.solve(params=_PARAMS, log_level="debug")[0]["alive"]


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_negm_applies_the_declared_scaling_of_the_durable_stock():
    """NEGM reproduces the grid-search optimum of the depreciating model.

    Both twins declare `Z' = alpha * s'` and search the same durable grid;
    NEGM's consumption margin is off-grid, so it weakly dominates within the
    band the two share. A solver that carried the raw chosen stock forward
    would answer a different model and leave that band by a wide margin.

    Stated at `alpha < 1`, where the carried stock stays inside the durable
    grid. Above one it does not, and the keeper candidate `s' = Z` reaches
    carried stocks the brute's action grid cannot offer at all, so the two
    choice sets stop coinciding and the gap stops measuring the law. Appreciation
    is covered by the ordering witness below, which needs no shared grid.
    """
    v_negm = np.asarray(_solve_period0_alive(_build_negm_model(DEPRECIATION)))
    v_brute = np.asarray(_solve_period0_alive(_build_brute_model(DEPRECIATION)))
    deviation = np.abs(v_negm - v_brute)
    assert float(deviation.mean()) < 0.06
    assert float(deviation.max()) < 0.15


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_the_value_rises_with_the_share_of_the_stock_that_survives():
    """More of the chosen stock surviving is worth weakly more, everywhere.

    Without this, the agreement test above would still pass if both twins
    ignored `alpha` in the same way — the two would agree on the wrong model.
    The comparison is a strict ordering rather than a tolerance: carrying more
    durable into the next period only ever adds to what the household holds,
    whatever it chose. This is also where `alpha > 1` is covered, since the
    ordering needs no oracle and so no shared choice set.
    """
    values = [
        np.asarray(_solve_period0_alive(_build_negm_model(alpha)))
        for alpha in (DEPRECIATION, 1.0, APPRECIATION)
    ]
    for lower, higher in pairwise(values):
        assert np.all(higher >= lower)
        assert np.any(higher > lower)


def test_an_outer_law_reading_the_inner_savings_margin_is_rejected():
    """A durable law depending on the inner consumption choice is not a NEGM model.

    NEGM evaluates the declared law to find what the next period carries, so a
    law reading the inner post-decision would make the stock carried forward
    depend on the consumption the inner Euler inversion is solving for — the
    outer maximum would no longer range over independent problems.
    """

    def euler_coupled_durable_transition(
        new_durable: ContinuousState, liquid_savings: FloatND
    ) -> ContinuousState:
        return new_durable + 0.1 * liquid_savings

    with pytest.raises(ModelInitializationError, match="inner margin"):
        _build_negm_model(1.0, durable_law=euler_coupled_durable_transition)
