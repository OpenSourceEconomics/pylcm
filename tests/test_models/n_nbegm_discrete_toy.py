"""Two-asset nested toy carrying a hard discrete choice.

The smooth two-asset budget of `n_nbegm_toy` plus a binary insurance decision:
buying costs a premium out of liquid resources and pays a flat utility bonus.
Nothing in the budget is declared piecewise-affine, so the inner NB-EGM
partition is a single interval and the discrete branches sit directly on the
degenerate plain-EGM case.

Conditional on an outer node the inner solve is one-dimensional per discrete
branch, so the nested solve is `max` over outer candidates of the inner
discrete upper envelope — a joint maximum over the `(outer node, branch)`
product, which needs no ordering convention because both aggregations are hard
maxima. The `"brute"` variant maximises over consumption, the durable
investment, and the branch on dense grids, and is the agreement oracle.
"""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    GridSearch,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    OuterContinuousMargin,
    Regime,
    categorical,
)
from lcm.outer_search import FiniteOuterGrid
from lcm.solvers import NBEGM, NNBEGM, TwoMarginSolver
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.test_models import n_nbegm_toy as smooth

# Flat utility gain from holding insurance — makes the branch worth buying.
INSURANCE_UTILITY = 0.15


@categorical(ordered=False)
class BuyPrivate:
    no: ScalarInt
    yes: ScalarInt


def resources(
    wealth: ContinuousState,
    illiquid: ContinuousState,
    new_illiquid: ContinuousState,
    buy_private: DiscreteAction,
    premium: float,
) -> FloatND:
    """Liquid resources at the fixed outer node, net of any premium paid."""
    credited = smooth.credited(illiquid=illiquid, new_illiquid=new_illiquid)
    paid = jnp.where(buy_private == BuyPrivate.yes, premium, 0.0)
    return wealth + smooth.LABOUR_INCOME - credited - paid


def utility(consumption: ContinuousAction, buy_private: DiscreteAction) -> FloatND:
    """CRRA over consumption plus a flat gain in the insured branch."""
    bonus = jnp.where(buy_private == BuyPrivate.yes, INSURANCE_UTILITY, 0.0)
    return (
        consumption ** (1.0 - smooth.RISK_AVERSION) / (1.0 - smooth.RISK_AVERSION)
        + bonus
    )


def build_solver(*, variant: str) -> TwoMarginSolver | GridSearch:
    """Build the requested solver flavour for the alive regime."""
    if variant == "brute":
        return GridSearch()
    if variant == "n_nbegm":
        return NNBEGM(
            inner=NBEGM(savings_grid=smooth.SAVINGS_GRID),
            outer_search=FiniteOuterGrid(grid=smooth.OUTER_GRID),
        )
    msg = f"unknown variant: {variant}"
    raise ValueError(msg)


def build_model(*, variant: str, n_periods: int = 3) -> Model:
    """Build the two-asset discrete-branch toy under the requested flavour."""
    final_age_alive = 20 + (n_periods - 2) * 5
    functions = {
        "utility": utility,
        "new_illiquid": smooth.new_illiquid,
        "resources": resources,
        "liquid_savings": smooth.liquid_savings,
        "keep_illiquid": smooth.keep_illiquid,
        "credited": smooth.credited,
    }
    states = {"wealth": smooth.WEALTH_GRID, "illiquid": smooth.ILLIQUID_GRID}
    state_transitions = {
        "wealth": smooth.next_wealth,
        "illiquid": smooth.durable_transition,
    }
    actions = {
        "consumption": smooth.CONSUMPTION_GRID,
        "illiquid_investment": smooth.ILLIQUID_INVESTMENT_GRID,
        "buy_private": DiscreteGrid(BuyPrivate),
    }
    active = lambda age, n=final_age_alive: age <= n  # noqa: E731
    if variant == "brute":
        # Same oracle correction as the smooth toy: reaching `s'` through an
        # investment action would let the oracle land on only 3 of the 15 outer
        # nodes, and on none at all from 8 of the 10 `illiquid` states, so it
        # would score a candidate set the nested solve never sweeps. Choosing
        # the next stock directly puts both variants on `OUTER_GRID`.
        del functions["new_illiquid"]
        actions = {
            "consumption": smooth.CONSUMPTION_GRID,
            "new_illiquid": smooth.OUTER_GRID,
            "buy_private": DiscreteGrid(BuyPrivate),
        }
        alive = Regime(
            active=active,
            states=states,
            state_transitions=state_transitions,
            actions=actions,
            transition=smooth.next_regime,
            functions=functions,
            constraints={"budget_feasible": smooth.budget_feasible},
            solver=build_solver(variant=variant),
        )
    else:
        alive = NestedConsumptionSavingsRegime(
            active=active,
            states=states,
            state_transitions=state_transitions,
            actions=actions,
            transition=smooth.next_regime,
            functions=functions,
            solver=build_solver(variant=variant),
            liquid=LiquidMargin(
                state="wealth",
                action="consumption",
                resources="resources",
                post_decision_state="liquid_savings",
            ),
            outer_continuous=OuterContinuousMargin(
                state="illiquid",
                action="illiquid_investment",
                post_decision_state="new_illiquid",
                no_adjustment="keep_illiquid",
            ),
        )
    dead = Regime(
        transition=None,
        active=lambda age, n=final_age_alive: age > n,
        states=states,
        functions={"utility": smooth.terminal_utility},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=smooth.RegimeId,
        ages=AgeGrid(start=20, stop=20 + (n_periods - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )
