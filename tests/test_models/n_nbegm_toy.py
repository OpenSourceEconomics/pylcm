"""Smooth two-asset toy for nested outer-search solvers.

The smallest model with a liquid Euler margin `wealth` plus an illiquid durable
margin `illiquid`, and a *numerically smooth* budget: deposits and withdrawals
move one-for-one, the liquid return has a single rate, and utility is pure CRRA
in consumption. Every solver family can represent it, so it isolates the outer
keeper/adjuster wrapper from the inner solver's kink machinery:

- `"brute"` — dense two-action grid search, the finite-grid oracle;
- `"negm"` — `NEGM(inner=DCEGM(...))`, the smooth nested baseline;
- `"n_nbegm"` — `NNBEGM(inner=NBEGM(...))`, the target method.

The nested solvers fix the outer post-decision `next_illiquid` per outer-grid
node; the inner consumption-saving problem is then a 1-D solve on `wealth` with
the credited durable move entering `resources` as a constant. The budget
declares no breakpoints, so the inner NB-EGM partition is a single interval —
the degenerate plain-EGM case.
"""

import jax.numpy as jnp

from lcm import (
    DCEGM,
    NEGM,
    NNBEGM,
    AgeGrid,
    GridSearch,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.outer_search import OuterSearch
from lcm.solvers import NBEGM, Solver
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

N_WEALTH = 12
N_ILLIQUID = 10
N_CONSUMPTION = 30
N_OUTER = 15
N_PERIODS = 3

LIQUID_RATE = 0.05
RISK_AVERSION = 2.0
LABOUR_INCOME = 5.0
TERMINAL_SCALE = 40.0


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def credited(illiquid: ContinuousState, next_illiquid: ContinuousState) -> FloatND:
    """Net liquid cost of moving the durable to `next_illiquid` — one-for-one."""
    return next_illiquid - illiquid


def resources(
    wealth: ContinuousState,
    illiquid: ContinuousState,
    next_illiquid: ContinuousState,
) -> FloatND:
    """Liquid resources consumption is paid out of, given the fixed outer node."""
    return (
        wealth
        + LABOUR_INCOME
        - credited(illiquid=illiquid, next_illiquid=next_illiquid)
    )


def liquid_savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Inner post-decision liquid balance."""
    return resources - consumption


def next_wealth(liquid_savings: FloatND) -> ContinuousState:
    """Inner Euler law with a single liquid rate — smooth everywhere."""
    return (1.0 + LIQUID_RATE) * liquid_savings


def durable_transition(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    """Durable law of motion `s' = Z + Iz` (the `illiquid` state transition)."""
    return illiquid + illiquid_investment


def keep_illiquid(illiquid: ContinuousState) -> FloatND:
    """The no-adjustment candidate `s' = Z`."""
    return illiquid


def utility(consumption: ContinuousAction) -> FloatND:
    """Pure CRRA over consumption."""
    return consumption ** (1.0 - RISK_AVERSION) / (1.0 - RISK_AVERSION)


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    """Inverse of `u'(c) = c^{-gamma}` in the inner consumption slot."""
    return marginal_continuation ** (-1.0 / RISK_AVERSION)


def terminal_utility(wealth: ContinuousState, illiquid: ContinuousState) -> FloatND:
    """Separably curved terminal payoff over both asset stocks.

    Decreasing marginal value in each stock keeps every optimum interior: the
    liquid/illiquid split, the inner savings choice, and consumption all stay
    inside their grids, so no solver family's off-grid extrapolation or cap
    handling enters the comparison.
    """
    return -TERMINAL_SCALE / (wealth + 1.0) - TERMINAL_SCALE / (illiquid + 1.0)


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.alive)


WEALTH_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=N_WEALTH)
ILLIQUID_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=N_ILLIQUID)
CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=20.0, n_points=N_CONSUMPTION)
# Covers `s' = Z + Iz` for every (Z, s') pair with Z in [0, 20] and s' in the
# outer grid, so the brute variant searches the same outer choice set the
# nested solvers sweep (feasibility constraints below trim the excess).
ILLIQUID_INVESTMENT_GRID = LinSpacedGrid(start=-20.0, stop=20.0, n_points=41)
OUTER_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=N_OUTER)
SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=35.0, n_points=60)


def illiquid_feasible(next_illiquid: ContinuousState) -> FloatND:
    """Brute-only constraint pinning `s'` to the N-NB-EGM outer range."""
    return (next_illiquid >= OUTER_GRID.start) & (next_illiquid <= OUTER_GRID.stop)


def budget_feasible(liquid_savings: FloatND) -> FloatND:
    """Brute-only constraint matching the inner solvers' `savings >= 0` grid."""
    return liquid_savings >= 0.0


def build_solver(
    *,
    variant: str,
    outer_batch_size: int = 0,
    outer_search: OuterSearch | None = None,
) -> Solver:
    """Build the requested solver flavour for the alive regime.

    `outer_search` (n_nbegm only) replaces the legacy finite `OUTER_GRID`
    with an explicit strategy — the continuous-outer entry point.
    """
    if variant == "brute":
        return GridSearch()
    if variant == "negm":
        return NEGM(
            inner=DCEGM(
                continuous_state="wealth",
                continuous_action="consumption",
                resources="resources",
                post_decision_function="liquid_savings",
                savings_grid=SAVINGS_GRID,
            ),
            outer_action="illiquid_investment",
            outer_post_decision="next_illiquid",
            outer_grid=OUTER_GRID,
            outer_no_adjustment_candidate="keep_illiquid",
            outer_batch_size=outer_batch_size,
        )
    if variant == "n_nbegm":
        return NNBEGM(
            inner=NBEGM(
                continuous_state="wealth",
                post_decision_function="liquid_savings",
                budget_target="resources",
                savings_grid=SAVINGS_GRID,
            ),
            outer_action="illiquid_investment",
            outer_post_decision="next_illiquid",
            outer_search=outer_search,
            outer_grid=None if outer_search is not None else OUTER_GRID,
            outer_no_adjustment_candidate="keep_illiquid",
            outer_batch_size=0 if outer_search is not None else outer_batch_size,
        )
    msg = f"unknown variant: {variant}"
    raise ValueError(msg)


def build_model(
    *,
    variant: str,
    outer_batch_size: int = 0,
    n_periods: int = N_PERIODS,
    outer_search: OuterSearch | None = None,
) -> Model:
    """Build the smooth two-asset toy under the requested solver flavour.

    With `n_periods=2` the single alive period reads only the terminal carry,
    isolating the outer wrapper from the nested-carry publication; longer
    horizons chain published nested carries between alive periods.
    """
    final_age_alive = 20 + (n_periods - 2) * 5
    functions = {
        "utility": utility,
        "resources": resources,
        "liquid_savings": liquid_savings,
        "keep_illiquid": keep_illiquid,
        "credited": credited,
    }
    if variant == "negm":
        functions["inverse_marginal_utility"] = inverse_marginal_utility
    constraints = (
        {"illiquid_feasible": illiquid_feasible, "budget_feasible": budget_feasible}
        if variant == "brute"
        else {}
    )
    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID},
        state_transitions={"wealth": next_wealth, "illiquid": durable_transition},
        actions={
            "consumption": CONSUMPTION_GRID,
            "illiquid_investment": ILLIQUID_INVESTMENT_GRID,
        },
        transition=next_regime,
        functions=functions,
        constraints=constraints,
        solver=build_solver(
            variant=variant,
            outer_batch_size=outer_batch_size,
            outer_search=outer_search,
        ),
    )
    dead = Regime(
        transition=None,
        active=lambda age, n=final_age_alive: age > n,
        states={"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID},
        functions={"utility": terminal_utility},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=20 + (n_periods - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )
