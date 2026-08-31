"""Smooth two-asset toy for nested outer-search solvers.

The smallest model with a liquid Euler margin `wealth` plus an illiquid durable
margin `illiquid`, and a *numerically smooth* budget: deposits and withdrawals
move one-for-one, the liquid return has a single rate, and utility is pure CRRA
in consumption. Every solver family can represent it, so it isolates the outer
keeper/adjuster wrapper from the inner solver's kink machinery:

- `"brute"` — two-action grid search over `OUTER_GRID` and the consumption
  grid, the finite-grid oracle: it chooses the next durable stock directly, so
  it searches the nested solvers' own outer candidate set;
- `"negm"` — `NEGM(inner=DCEGM(...))`, the smooth nested baseline;
- `"n_nbegm"` — `NNBEGM(inner=NBEGM(...))`, the target method.

The nested solvers fix the outer post-decision `new_illiquid` per outer-grid
node; the inner consumption-saving problem is then a 1-D solve on `wealth` with
the credited durable move entering `resources` as a constant. The budget
declares no breakpoints, so the inner NB-EGM partition is a single interval —
the degenerate plain-EGM case.
"""

from collections.abc import Callable, Mapping

import jax.numpy as jnp

from _lcm.grids.base import Grid
from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    categorical,
)
from lcm.consumption_savings_regime import (
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    outer_unchanged,
)
from lcm.solvers import (
    DCEGM,
    NBEGM,
    NEGM,
    NNBEGM,
    FiniteOuterGrid,
    GridSearch,
    OuterBranchAggregator,
    OuterSearch,
    TwoMarginSolver,
)
from lcm.transition import AgeSpecializedFunction, AgeSpecializedGrid
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


def reserve_unchanged(reserve: ContinuousState) -> ContinuousState:
    """Law of motion of the second passive stock: it is never touched."""
    return reserve


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


def impute_permanent_income(wealth: ContinuousState) -> ContinuousState:
    """Solve-phase imputation for the carried-state capability witness."""
    return 0.1 * wealth


def evolve_permanent_income(
    permanent_income: ContinuousState,
) -> ContinuousState:
    """Simulation law for the carried-state capability witness."""
    return permanent_income


WEALTH_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=N_WEALTH)
ILLIQUID_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=N_ILLIQUID)
CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=20.0, n_points=N_CONSUMPTION)
# The nested solvers' outer action. Spans `s' = Z + Iz` for every (Z, s') pair
# with Z in [0, 20] and s' in [0, 20]; spanning the range is all that is asked
# of it, since the nested solvers reach their candidates through `OUTER_GRID`
# and not through this grid. The brute variant deliberately does *not* use it —
# see `build_model`.
ILLIQUID_INVESTMENT_GRID = LinSpacedGrid(start=-20.0, stop=20.0, n_points=41)
OUTER_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=N_OUTER)
SAVINGS_FLOOR = 0.0  # borrowing limit on the inner post-decision balance
SAVINGS_GRID = LinSpacedGrid(start=SAVINGS_FLOOR, stop=35.0, n_points=60)
# A second passive continuous stock, held fixed and carried only by the alive
# regime. Two points keep the extra state-space axis as small as an axis can be.
RESERVE_GRID = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)


def budget_feasible(liquid_savings: FloatND) -> FloatND:
    """Brute-only constraint matching the inner solvers' `savings >= 0` grid."""
    return liquid_savings >= 0.0


def adjustment_scale(period: int) -> FloatND:
    """Per-period scale of the uniform observed fixed adjustment cost."""
    return jnp.asarray(0.3 + 0.05 * period)


def adjustment_scale_from_param(adjustment_scale_level: float) -> FloatND:
    """Fixed-cost scale read straight from a flat param."""
    return jnp.asarray(adjustment_scale_level)


def build_solver(
    *,
    variant: str,
    outer_batch_size: int = 0,
    outer_search: OuterSearch | None = None,
) -> TwoMarginSolver | GridSearch:
    """Build the requested solver flavour for the alive regime.

    `outer_search` (n_nbegm only) replaces the legacy finite `OUTER_GRID`
    with an explicit strategy — the continuous-outer entry point. The
    keeper/adjuster fold follows the regime's declared `adjustment_cost`, so it
    is not a solver argument.
    """
    if variant == "brute":
        return GridSearch()
    if variant == "negm":
        return NEGM(
            inner=DCEGM(
                savings_grid=SAVINGS_GRID,
            ),
            outer_grid=OUTER_GRID,
            outer_batch_size=outer_batch_size,
        )
    if variant == "n_nbegm":
        return NNBEGM(
            inner=NBEGM(
                savings_grid=SAVINGS_GRID,
            ),
            outer_search=(
                outer_search
                if outer_search is not None
                else FiniteOuterGrid(grid=OUTER_GRID, batch_size=outer_batch_size)
            ),
        )
    msg = f"unknown variant: {variant}"
    raise ValueError(msg)


def build_model(
    *,
    variant: str,
    outer_batch_size: int = 0,
    n_periods: int = N_PERIODS,
    illiquid_grid: Grid | AgeSpecializedGrid = ILLIQUID_GRID,
    outer_search: OuterSearch | None = None,
    adjustment_cost: OuterBranchAggregator | None = None,
    scale_function: Callable[..., object] | None = None,
    illiquid_investment_grid: Grid = ILLIQUID_INVESTMENT_GRID,
    consumption_grid: Grid = CONSUMPTION_GRID,
    durable_law: Callable[..., object] | Phased | None = None,
    constraints: Mapping[str, Callable[..., object]] | None = None,
    utility_function: Callable[..., object] | AgeSpecializedFunction | Phased = utility,
    terminal_utility_function: Callable[..., object] = terminal_utility,
    outer_post_decision_function: Callable[..., object] | None = None,
    regime_transition: Callable[..., object] | Phased = next_regime,
    koopmans_aggregator: Callable[..., object] | Phased | None = None,
    second_passive_state: bool = False,
    carried_state: bool = False,
    terminal_active_from_start: bool = False,
) -> Model:
    """Build the smooth two-asset toy under the requested solver flavour.

    With `n_periods=2` the single alive period reads only the terminal carry,
    isolating the outer wrapper from the nested-carry publication; longer
    horizons chain published nested carries between alive periods.

    `illiquid_grid` overrides the durable state's grid in both regimes, and
    `illiquid_investment_grid` the action that moves it. Together they set
    which `new_illiquid` levels the brute variant can reach, so a caller can
    align the brute candidate set with the nested solvers' outer grid.
    `consumption_grid` refines the inner action, which the grid search ranks
    directly and the endogenous-grid solvers do not, so a refinement sequence
    over it separates the oracle's own discretization error from the solver's.
    `durable_law` overrides the durable's law of motion; every variant reads the
    chosen stock through `new_illiquid`, so one law serves them all and the
    variants keep solving the same model.
    `scale_function` overrides the fixed cost's `adjustment_scale` function,
    so a caller can drive the scale from a flat param.
    `terminal_utility_function` overrides the bequest the terminal regime
    pays. The default is singular one unit below zero durable, so a caller
    giving `illiquid_grid` a domain that reaches into durable debt supplies
    a bequest finite there as well.
    `outer_post_decision_function` overrides `new_illiquid`, the outer
    post-decision margin every variant reads the chosen stock through. It is the
    map N-NB-EGM must invert, so a caller can declare one the solver is required
    to refuse.
    `regime_transition` and `koopmans_aggregator` expose the other public phase
    slots to build-time capability tests without changing the numerical toy.
    `second_passive_state=True` gives the alive regime a second passive
    continuous stock, held fixed and carried by that regime alone, so its carry
    rows span two passive axes instead of one.
    `carried_state=True` adds an otherwise unused solve-imputed/simulate-carried
    state for the NNBEGM replay-capability boundary witness.
    `constraints` overrides the constraint pool, which otherwise carries the
    budget predicate on the grid-search arm and is empty on the endogenous-grid
    arms, whose kernels enforce the budget identity intrinsically.
    `terminal_active_from_start=True` also activates the terminal regime before
    the lifecycle transition. This supports simulations seeded with subjects in
    both regimes at the same age; the default keeps the terminal regime active
    only after the final alive age.
    """
    final_age_alive = 20 + (n_periods - 2) * 5
    functions = {
        "utility": utility_function,
        # Resolved at call time, not captured as a default: a default argument
        # binds at definition, which would make `toy.new_illiquid` unpatchable.
        "new_illiquid": (
            new_illiquid
            if outer_post_decision_function is None
            else outer_post_decision_function
        ),
        "resources": resources,
        "liquid_savings": liquid_savings,
        "credited": credited,
    }
    if variant == "negm":
        # With `outer_cost` declared, pylcm composes the resources function
        # as `resources_before_outer_cost - credited` at model build; the
        # direct `resources` definition would double-declare it.
        del functions["resources"]
        functions["resources_before_outer_cost"] = resources_before_outer_cost
        functions["inverse_marginal_utility"] = inverse_marginal_utility
    if adjustment_cost is not None:
        functions["adjustment_scale"] = (
            adjustment_scale if scale_function is None else scale_function
        )
    if constraints is None:
        constraints = {"budget_feasible": budget_feasible} if variant == "brute" else {}
    active = lambda age, n=final_age_alive: age <= n  # noqa: E731
    states: dict[str, Grid | Phased | AgeSpecializedGrid] = {
        "wealth": WEALTH_GRID,
        "illiquid": illiquid_grid,
    }
    state_transitions = {
        "wealth": next_wealth,
        "illiquid": durable_law if durable_law is not None else durable_transition,
    }
    if second_passive_state:
        states["reserve"] = RESERVE_GRID
        state_transitions["reserve"] = reserve_unchanged
    if carried_state:
        states["permanent_income"] = Phased(
            solve=impute_permanent_income,
            simulate=LinSpacedGrid(start=0.0, stop=3.0, n_points=4),
        )
        state_transitions["permanent_income"] = evolve_permanent_income
    actions = {
        "consumption": consumption_grid,
        "illiquid_investment": illiquid_investment_grid,
    }
    if variant == "brute":
        # The oracle has to search the candidate set the nested solvers sweep,
        # not merely a set that spans its range. Reaching `s'` through an
        # investment action makes the two sets coincide only where `s' = Z + Iz`
        # is an outer node, i.e. where 63 divides 9j - 14k for `Z = 20k/9` and
        # `s' = 20j/14` — 3 of the 15 outer nodes, and those from only 2 of the
        # 10 `illiquid` states; the other 8 states hit none. Choosing the next
        # stock directly puts both solvers on `OUTER_GRID` in the same
        # coordinates, so the comparison is over one candidate set rather than
        # two, and the `illiquid` grid is untouched — the state space is the
        # same before and after, which is what makes the two runs comparable.
        #
        # Dropping `new_illiquid` from the DAG turns its output into an external
        # input the action supplies; every reader — `credited`, `resources`, the
        # durable law — is unchanged.
        del functions["new_illiquid"]
        actions = {"consumption": consumption_grid, "new_illiquid": OUTER_GRID}
    solver = build_solver(
        variant=variant,
        outer_batch_size=outer_batch_size,
        outer_search=outer_search,
    )
    # Built per branch rather than from one shared mapping: the two regime
    # classes narrow `solver` differently, and a `**kwargs` mapping erases the
    # argument types the narrowing is expressed in.
    if variant == "brute":
        alive = Regime(
            active=active,
            states=states,
            state_transitions=state_transitions,
            actions=actions,
            transition=regime_transition,
            functions=functions,
            constraints=constraints,
            solver=solver,
            koopmans_aggregator=koopmans_aggregator,
        )
    else:
        liquid_resources = (
            NetOfAdjustmentCost(
                output="resources",
                before_cost="resources_before_outer_cost",
                cost="credited",
            )
            if variant == "negm"
            else "resources"
        )
        alive = NestedConsumptionSavingsRegime(
            active=active,
            states=states,
            state_transitions=state_transitions,
            actions=actions,
            transition=regime_transition,
            functions=functions,
            constraints=constraints,
            solver=solver,
            koopmans_aggregator=koopmans_aggregator,
            liquid=LiquidMargin(
                state="wealth",
                action="consumption",
                resources=liquid_resources,
                post_decision_state="liquid_savings",
            ),
            outer_continuous=OuterContinuousMargin(
                state="illiquid",
                action="illiquid_investment",
                post_decision_state="new_illiquid",
                no_adjustment=outer_unchanged,
                adjustment_cost=adjustment_cost,
            ),
        )
    dead = Regime(
        transition=None,
        active=lambda age, n=final_age_alive: terminal_active_from_start or age > n,
        states={"wealth": WEALTH_GRID, "illiquid": illiquid_grid},
        functions={"utility": terminal_utility_function},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=20 + (n_periods - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )
