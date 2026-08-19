"""Smooth two-asset toy for nested outer-search solvers.

The smallest model with a liquid Euler margin `wealth` plus an illiquid durable
margin `illiquid`, and a *numerically smooth* budget: deposits and withdrawals
move one-for-one, the liquid return has a single rate, and utility is pure CRRA
in consumption. Every solver family can represent it, so it isolates the outer
keeper/adjuster wrapper from the inner solver's kink machinery:

- `"brute"` — dense two-action grid search, the finite-grid oracle;
- `"negm"` — `NEGM(inner=DCEGM(...))`, the smooth nested baseline;
- `"n_nbegm"` — `NNBEGM(inner=NBEGM(...))`, the target method.

The nested solvers fix the outer post-decision `new_illiquid` per outer-grid
node; the inner consumption-saving problem is then a 1-D solve on `wealth` with
the credited durable move entering `resources` as a constant. The budget
declares no breakpoints, so the inner NB-EGM partition is a single interval —
the degenerate plain-EGM case.
"""

from collections.abc import Callable

import jax.numpy as jnp

from _lcm.grids.base import Grid
from lcm import (
    DCEGM,
    NEGM,
    NNBEGM,
    AgeGrid,
    GridSearch,
    LinSpacedGrid,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    Regime,
    categorical,
)
from lcm.branch_aggregation import OuterBranchAggregator
from lcm.outer_search import FiniteOuterGrid, OuterSearch
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


def new_illiquid(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    """The durable stock chosen this period, `s' = Z + Iz`.

    The outer post-decision margin: an ordinary function of this period's state
    and action, so the credited cost, the liquid budget, and the `illiquid` law
    of motion all read one value.
    """
    return illiquid + illiquid_investment


def credited(illiquid: ContinuousState, new_illiquid: ContinuousState) -> FloatND:
    """Net liquid cost of moving the durable to `new_illiquid` — one-for-one."""
    return new_illiquid - illiquid


def resources(
    wealth: ContinuousState,
    illiquid: ContinuousState,
    new_illiquid: ContinuousState,
) -> FloatND:
    """Liquid resources consumption is paid out of, given the fixed outer node."""
    return (
        wealth + LABOUR_INCOME - credited(illiquid=illiquid, new_illiquid=new_illiquid)
    )


def resources_before_outer_cost(wealth: ContinuousState) -> FloatND:
    """Cost-free base of the liquid resources, `wealth + y`.

    The NEGM variant declares `outer_cost="credited"`, so pylcm composes its
    resources function as `resources_before_outer_cost - credited` at model
    build — the same quantity `resources` states directly for the other
    variants.
    """
    return wealth + LABOUR_INCOME


def liquid_savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Inner post-decision liquid balance."""
    return resources - consumption


def next_wealth(liquid_savings: FloatND) -> ContinuousState:
    """Inner Euler law with a single liquid rate — smooth everywhere."""
    return (1.0 + LIQUID_RATE) * liquid_savings


def durable_transition(new_illiquid: ContinuousState) -> ContinuousState:
    """Durable law of motion: the stock chosen this period is next period's."""
    return new_illiquid


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


def illiquid_feasible(new_illiquid: ContinuousState) -> FloatND:
    """Brute-only constraint pinning `s'` to the N-NB-EGM outer range."""
    return (new_illiquid >= OUTER_GRID.start) & (new_illiquid <= OUTER_GRID.stop)


def budget_feasible(liquid_savings: FloatND) -> FloatND:
    """Brute-only constraint matching the inner solvers' `savings >= 0` grid."""
    return liquid_savings >= 0.0


def adjustment_scale(period: int) -> FloatND:
    """Per-period scale of the uniform observed fixed adjustment cost."""
    return jnp.asarray(0.3 + 0.05 * period)


def build_solver(
    *,
    variant: str,
    outer_batch_size: int = 0,
    outer_search: OuterSearch | None = None,
    branch_aggregator: OuterBranchAggregator | None = None,
) -> Solver:
    """Build the requested solver flavour for the alive regime.

    `outer_search` (n_nbegm only) replaces the legacy finite `OUTER_GRID`
    with an explicit strategy — the continuous-outer entry point.
    `branch_aggregator` (n_nbegm only) selects the keeper/adjuster fold.
    """
    if variant == "brute":
        return GridSearch()
    if variant == "negm":
        # NEGM takes its DAG names from the regime's two margins, so the solver
        # carries only numerical configuration; see `_negm_margins`.
        return NEGM(
            inner=DCEGM(savings_grid=SAVINGS_GRID),
            outer_grid=OUTER_GRID,
            outer_batch_size=outer_batch_size,
        )
    if variant == "n_nbegm":
        aggregator_kwargs = (
            {}
            if branch_aggregator is None
            else {"branch_aggregator": branch_aggregator}
        )
        return NNBEGM(
            inner=NBEGM(
                continuous_state="wealth",
                post_decision_function="liquid_savings",
                budget_target="resources",
                savings_grid=SAVINGS_GRID,
            ),
            outer_action="illiquid_investment",
            outer_state="illiquid",
            outer_post_decision="new_illiquid",
            outer_search=(
                outer_search
                if outer_search is not None
                else FiniteOuterGrid(grid=OUTER_GRID, batch_size=outer_batch_size)
            ),
            outer_no_adjustment_candidate="keep_illiquid",
            **aggregator_kwargs,
        )
    msg = f"unknown variant: {variant}"
    raise ValueError(msg)


def build_model(
    *,
    variant: str,
    outer_batch_size: int = 0,
    n_periods: int = N_PERIODS,
    illiquid_grid: Grid = ILLIQUID_GRID,
    outer_search: OuterSearch | None = None,
    branch_aggregator: OuterBranchAggregator | None = None,
    durable_law: Callable[..., object] | None = None,
) -> Model:
    """Build the smooth two-asset toy under the requested solver flavour.

    With `n_periods=2` the single alive period reads only the terminal carry,
    isolating the outer wrapper from the nested-carry publication; longer
    horizons chain published nested carries between alive periods.

    `illiquid_grid` overrides the durable state's grid in both regimes.
    `durable_law` overrides the durable's law of motion; every variant reads the
    chosen stock through `new_illiquid`, so one law serves them all and the
    variants keep solving the same model.
    """
    final_age_alive = 20 + (n_periods - 2) * 5
    functions = {
        "utility": utility,
        "new_illiquid": new_illiquid,
        "resources": resources,
        "liquid_savings": liquid_savings,
        "keep_illiquid": keep_illiquid,
        "credited": credited,
    }
    if variant == "negm":
        # With `outer_cost` declared, pylcm composes the resources function
        # as `resources_before_outer_cost - credited` at model build; the
        # direct `resources` definition would double-declare it.
        del functions["resources"]
        functions["resources_before_outer_cost"] = resources_before_outer_cost
        functions["inverse_marginal_utility"] = inverse_marginal_utility
    if branch_aggregator is not None:
        functions["adjustment_scale"] = adjustment_scale
    constraints = (
        {"illiquid_feasible": illiquid_feasible, "budget_feasible": budget_feasible}
        if variant == "brute"
        else {}
    )
    active = lambda age, n=final_age_alive: age <= n  # noqa: E731
    states = {"wealth": WEALTH_GRID, "illiquid": illiquid_grid}
    state_transitions = {
        "wealth": next_wealth,
        "illiquid": durable_law if durable_law is not None else durable_transition,
    }
    actions = {
        "consumption": CONSUMPTION_GRID,
        "illiquid_investment": ILLIQUID_INVESTMENT_GRID,
    }
    if variant == "negm":
        # Built here rather than through `build_solver` so the declared type is
        # the two-margin solver the nested regime accepts, not the base `Solver`.
        negm_solver = NEGM(
            inner=DCEGM(savings_grid=SAVINGS_GRID),
            outer_grid=OUTER_GRID,
            outer_batch_size=outer_batch_size,
        )
        alive = NestedConsumptionSavingsRegime(
            active=active,
            states=states,
            state_transitions=state_transitions,
            actions=actions,
            transition=next_regime,
            functions=functions,
            constraints=constraints,
            solver=negm_solver,
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
                post_decision_state="new_illiquid",
                no_adjustment="keep_illiquid",
            ),
        )
    else:
        solver = build_solver(
            variant=variant,
            outer_batch_size=outer_batch_size,
            outer_search=outer_search,
            branch_aggregator=branch_aggregator,
        )
        alive = Regime(
            active=active,
            states=states,
            state_transitions=state_transitions,
            actions=actions,
            transition=next_regime,
            functions=functions,
            constraints=constraints,
            solver=solver,
        )
    dead = Regime(
        transition=None,
        active=lambda age, n=final_age_alive: age > n,
        states={"wealth": WEALTH_GRID, "illiquid": illiquid_grid},
        functions={"utility": terminal_utility},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=20 + (n_periods - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )
