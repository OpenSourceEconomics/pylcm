"""Dobrescu-Shanker (2024) housing model as a pylcm NEGM model.

The DS-2024 multidimensional-method paper (SSRN 4850746) compares RFC against
NEGM on a housing model with liquid assets `a`, a durable housing stock `h`, a
proportional house-trade cost `tau`, depreciation `delta`, a two-state Markov
income `z`, and a discrete adjust/keep choice. Model and calibration are read
from the authors' `InverseDCDP` repo (`housing/housing.py`,
`settings/settings.yml`); see `ds2024-housing-build-plan.md`.

It maps onto pylcm's nested-EGM solver exactly like the DS-2026 App.2 housing
model:

- the **inner** DC-EGM solves liquid consumption-savings, the Euler equation
  inverting on `liquid` assets `a`;
- the **outer** durable margin is the next housing stock `H'` searched over a
  grid, the proportional cost making it non-concave;
- the **keeper** holds the house and the **adjuster** chooses `H'` paying
  `(1 + tau)·H'` while selling the depreciated old house at `(1 + r_H)·h(1-delta)`.

Utility is CRRA non-durable consumption plus a log housing-service flow
`u(c, H') = (c^{1-gamma_C} - 1)/(1 - gamma_C) + alpha·log(H')`, the service
reading the chosen house `H'` (additively separable from consumption — the NEGM
contract).

## Depreciation and the keeper

In the source model the keeper's next house is the depreciated stock
`h(1 - delta)` and only the adjuster liquidates. The NEGM keeper realises this by
injecting the regime's `outer_no_adjustment_candidate` (`keep_housing`, which maps
`h -> h(1 - delta)`) as the keeper's durable transition: the kept stock lands off
the outer housing grid and the inner DC-EGM's passive read blends the continuation
value over the grid's neighbouring nodes, `credited(h, h(1 - delta)) = 0` making
the hold free. The model is therefore faithful at any `delta`.

The brute grid-search twin is a valid oracle only at `delta = 0`: it searches
`next_housing` on the housing grid, so the free-keep level `h(1 - delta)` is on the
grid only when `delta = 0`. At `delta > 0` the brute cannot represent the off-grid
free keep, so the `delta > 0` keeper is validated against a dense host VFI oracle
that includes the free-keep candidate explicitly.
"""

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp

from lcm import (
    DCEGM,
    NEGM,
    AgeGrid,
    DiscreteGrid,
    GridSearch,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)

# Income discretisation (`z_vals`, `Pi` from InverseDCDP housing.py).
INCOME_LOW = 0.1
INCOME_HIGH = 1.0
INCOME_PI = ((0.09, 0.91), (0.06, 0.94))

# Wage-polynomial coefficients (`settings/settings.yml` `lambdas`): a degree-4
# age profile plus a quadratic tenure term. The benchmark's stationary run pins
# the age at the terminal age `T = 60`, so income depends only on the income node.
WAGE_LAMBDAS = (
    4.7651949,
    0.57802016,
    -0.02022858,
    0.00030696,
    -1.71e-6,
    0.0241546,
    -0.00011022,
)
STATIONARY_AGE = 60


def _stationary_log_income_base() -> float:
    """Log labor income at the stationary age, excluding the income node `z`.

    `sum_{i<5} lambda_i * T^i + lambda_5 * T + lambda_6 * T^2` at `T = 60`, so the
    stationary income is `exp(base + z) * 1e-5`.
    """
    age = float(STATIONARY_AGE)
    age_profile = sum(WAGE_LAMBDAS[i] * age**i for i in range(5))
    tenure = WAGE_LAMBDAS[5] * age + WAGE_LAMBDAS[6] * age**2
    return age_profile + tenure


_LOG_INCOME_BASE = _stationary_log_income_base()


@categorical(ordered=False)
class DS2024HousingRegimeId:
    """Lifecycle regimes: an alive housing regime and the terminal bequest."""

    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Income:
    """Two-state Markov income node."""

    low: ScalarInt
    high: ScalarInt


def income_value(income: DiscreteState) -> FloatND:
    """Stationary labor income `y(z) = exp(base + z) * 1e-5` of the income node."""
    z = jnp.where(income == Income.low, INCOME_LOW, INCOME_HIGH)
    return jnp.exp(_LOG_INCOME_BASE + z) * 1e-5


def income_transition(income: DiscreteState) -> FloatND:
    """Markov income law: the row of `Pi` for the current income node."""
    pi = jnp.asarray(INCOME_PI)
    return pi[income]


def _make_keep_housing(delta: float) -> Callable[[ContinuousState], FloatND]:
    """Build the keeper's no-adjustment durable map `H' = h(1 - delta)`.

    The NEGM keeper holds the house at this level for free (the adjustment-cost
    kink). It is the regime's `outer_no_adjustment_candidate`, injected as the
    keeper's durable transition, so it must be **param-free** (a param on the
    durable law binds per target and is read within-period) — `delta` is baked in
    at build time via this closure. At `delta = 0` it returns the stock unchanged
    (`housing * 1.0 == housing`), matching the auto-identity keeper exactly; a
    `delta > 0` keeper depreciates to `h(1 - delta)`, which lands off the housing
    grid and is blended by the inner DC-EGM's passive read.
    """

    def keep_housing(housing: ContinuousState) -> FloatND:
        return housing * (1.0 - delta)

    return keep_housing


def housing_cost(
    housing: ContinuousState,
    next_housing: ContinuousState,
    delta: float,
    return_housing: float,
    tau: float,
) -> FloatND:
    """Net liquid cost of moving the house from `h` to `next_housing` (`H'`).

    - keep (`H' = h(1 - delta)`): cost `0` — the house is retained, no trade;
    - adjust (`H' != h(1 - delta)`): cost `(1 + tau)·H' - (1 + r_H)·h(1 - delta)`
      — sell the depreciated old house at `(1 + r_H)·h(1 - delta)` and buy the new
      house at `(1 + tau)·H'`.

    The proportional cost falls on the whole new stock, so any adjustment pays a
    discrete wedge over keeping (adjusting even to the keep level `h(1 - delta)`
    costs `h(1 - delta)·(tau - r_H)`), opening the (S, s) inaction band. Reads only
    the held housing state and the outer post-decision `next_housing` — a constant
    per outer-grid node, as the NEGM contract requires.
    """
    depreciated = housing * (1.0 - delta)
    round_trip = (1.0 + tau) * next_housing - (1.0 + return_housing) * depreciated
    return jnp.where(next_housing == depreciated, 0.0, round_trip)


def resources(
    liquid: ContinuousState,
    housing_cost: FloatND,
    income_value: FloatND,
    return_liquid: float,
) -> FloatND:
    """Liquid resources consumption is paid from, given the fixed outer node.

    `(1 + r)·a + y - housing_cost`. With the keep cost `0` this is the keeper's
    cash `R·a + y`; with the adjust cost it is `R·a + (1+r_H)·h(1-delta) + y -
    (1+tau)·H'`. The housing cost is bound per outer-grid node, so it enters the
    inner Euler inversion as a constant.
    """
    return (1.0 + return_liquid) * liquid + income_value - housing_cost


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Inner post-decision liquid balance `a' = resources - c`."""
    return resources - consumption


def next_liquid(savings: FloatND) -> ContinuousState:
    """Euler-state law: next liquid assets equal post-decision savings."""
    return savings


def next_housing(
    housing: ContinuousState, housing_investment: ContinuousAction
) -> ContinuousState:
    """Durable law `H' = H + housing_investment`.

    The adjuster's outer search ranges `next_housing` over the outer house-level
    grid (and the brute twin searches it directly via `housing_investment`). The
    keeper's no-adjustment map is the separate `keep_housing` (`H' = h(1 - delta)`),
    injected by the NEGM solver as the keeper's durable transition, so this law is
    the adjuster branch only. The durable transition carries no params (it is read
    within-period by the service flow, so a param would bind per target); the
    depreciation enters through `keep_housing`, the adjust cost, and the bequest.
    """
    return housing + housing_investment


def serviced_housing(next_housing: ContinuousState) -> FloatND:
    """The house serviced this period — the chosen `H'` (keep or adjust)."""
    return next_housing


def utility(
    consumption: ContinuousAction,
    serviced_housing: FloatND,
    gamma_c: float,
    alpha: float,
) -> FloatND:
    """CRRA consumption utility plus a log housing-service flow.

    `u(c, H') = (c^{1 - gamma_C} - 1)/(1 - gamma_C) + alpha·log(H')`. The housing
    term is additively separable, so it drops from the inner consumption Euler
    inversion; utility reads the consumption action and the serviced-housing
    service flow, never the liquid Euler state.
    """
    consumption_utility = (consumption ** (1.0 - gamma_c) - 1.0) / (1.0 - gamma_c)
    return consumption_utility + alpha * jnp.log(serviced_housing)


def inverse_marginal_utility(marginal_continuation: FloatND, gamma_c: float) -> FloatND:
    """Invert the consumption marginal utility `u'(c) = c^{-gamma_C}`.

    `c = mc^{-1/gamma_C}`. The log housing-service term is separable and drops
    from the inner inversion, so unlike the App.2 CES the consumption weight is 1.
    """
    return marginal_continuation ** (-1.0 / gamma_c)


def next_liquid_brute(
    liquid: ContinuousState,
    housing_cost: FloatND,
    income_value: FloatND,
    consumption: ContinuousAction,
    return_liquid: float,
) -> ContinuousState:
    """Brute-force liquid law: resources minus consumption."""
    return (1.0 + return_liquid) * liquid + income_value - housing_cost - consumption


def borrowing_constraint(
    liquid: ContinuousState,
    housing_cost: FloatND,
    income_value: FloatND,
    consumption: ContinuousAction,
    return_liquid: float,
    borrowing_limit: float,
) -> BoolND:
    """Keep post-decision liquid assets at or above the borrowing limit `b`."""
    post = (1.0 + return_liquid) * liquid + income_value - housing_cost - consumption
    return post >= borrowing_limit


def bequest(
    liquid: ContinuousState,
    housing: ContinuousState,
    return_liquid: float,
    theta: float,
    bequest_shift: float,
    gamma_c: float,
) -> FloatND:
    """Terminal bequest `theta·((K + (1+r)·a + h)^{1-gamma_C} - 1)/(1 - gamma_C)`.

    The CRRA bequest (`term_u` in the source, with shift `K`) reads the carried
    liquid and housing states.
    """
    estate = bequest_shift + (1.0 + return_liquid) * liquid + housing
    return theta * (estate ** (1.0 - gamma_c) - 1.0) / (1.0 - gamma_c)


def build_model(
    *,
    variant: Literal["negm", "brute"] = "negm",
    n_grid: int,
    n_periods: int = 4,
    liquid_max: float = 50.0,
    housing_max: float = 50.0,
    housing_min: float = 0.01,
    consumption_max: float = 50.0,
    n_consumption: int = 60,
    n_savings: int = 60,
    delta: float = 0.0,
    n_outer_grid: int | None = None,
) -> Model:
    """Build the DS-2024 housing model.

    Args:
        variant: `"negm"` builds the nested-EGM model (inner liquid DC-EGM, outer
            housing search); `"brute"` builds the grid-search twin solving the same
            economics with no Euler machinery — the accuracy oracle.
        n_grid: Number of points on the liquid, housing, and outer house grids.
        n_periods: Number of model periods (the last is the terminal bequest).
        liquid_max: Upper bound of the liquid grid.
        housing_max: Upper bound of the housing and outer house grids.
        housing_min: Lower bound `b` of the housing grid (and borrowing limit).
        consumption_max: Upper bound of the inner consumption grid.
        n_consumption: Number of inner consumption-grid points.
        n_savings: Number of inner savings-grid points.
        delta: House depreciation rate baked into the keeper's no-adjustment map
            `H' = h(1 - delta)` (param-free, so set at build time). Pass the same
            value to `build_params` (where it enters the adjust cost and bequest
            as a param); `0.0` is the keeper-holds-the-stock case.
        n_outer_grid: Number of points on the NEGM outer house grid; `None`
            uses `n_grid`. Decoupling it from the state grids enables nested
            outer refinements at a fixed state resolution.

    Returns:
        The alive housing regime plus the terminal bequest regime.
    """
    ages = AgeGrid(start=STATIONARY_AGE, stop=STATIONARY_AGE + n_periods - 1, step="Y")
    final_age = int(ages.exact_values[-1])
    keep_housing = _make_keep_housing(delta)

    liquid_grid = LinSpacedGrid(start=housing_min, stop=liquid_max, n_points=n_grid)
    housing_grid = LinSpacedGrid(start=housing_min, stop=housing_max, n_points=n_grid)
    outer_grid = LinSpacedGrid(
        start=housing_min, stop=housing_max, n_points=n_outer_grid or n_grid
    )
    consumption_grid = LinSpacedGrid(
        start=0.05, stop=consumption_max, n_points=n_consumption
    )
    housing_investment_grid = LinSpacedGrid(
        start=-housing_max, stop=housing_max, n_points=n_grid
    )
    savings_grid = LinSpacedGrid(
        start=housing_min, stop=liquid_max + housing_max, n_points=n_savings
    )

    def housing_stays_in_bounds(next_housing: ContinuousState) -> BoolND:
        """The chosen next house must stay within `[housing_min, housing_max]`."""
        return (next_housing >= housing_min) & (next_housing <= housing_max)

    dead = UserRegime(
        transition=None,
        active=lambda age, fa=final_age: age >= fa,
        states={"liquid": liquid_grid, "housing": housing_grid},
        functions={"utility": bequest},
    )

    def next_regime(age: int) -> ScalarInt:
        """Stay alive until the final age, then enter the terminal bequest."""
        return jnp.where(
            age + 1 >= final_age,
            DS2024HousingRegimeId.dead,
            DS2024HousingRegimeId.alive,
        )

    if variant == "brute":
        alive = UserRegime(
            transition=next_regime,
            active=lambda age, fa=final_age: age < fa,
            states={
                "liquid": liquid_grid,
                "housing": housing_grid,
                "income": DiscreteGrid(Income),
            },
            state_transitions={
                "liquid": next_liquid_brute,
                "housing": next_housing,
                "income": MarkovTransition(income_transition),
            },
            actions={
                "consumption": consumption_grid,
                "housing_investment": housing_investment_grid,
            },
            constraints={
                "borrowing_constraint": borrowing_constraint,
                "housing_stays_in_bounds": housing_stays_in_bounds,
            },
            functions={
                "utility": utility,
                "housing_cost": housing_cost,
                "keep_housing": keep_housing,
                "serviced_housing": serviced_housing,
                "income_value": income_value,
            },
            solver=GridSearch(),
        )
        return Model(
            regimes={"alive": alive, "dead": dead},
            ages=ages,
            regime_id_class=DS2024HousingRegimeId,
        )

    negm_solver = NEGM(
        inner=DCEGM(
            continuous_state="liquid",
            continuous_action="consumption",
            resources="resources",
            post_decision_function="savings",
            savings_grid=savings_grid,
        ),
        outer_action="housing_investment",
        outer_post_decision="next_housing",
        outer_grid=outer_grid,
        outer_no_adjustment_candidate="keep_housing",
        outer_cost="housing_cost",
    )

    alive = UserRegime(
        transition=next_regime,
        active=lambda age, fa=final_age: age < fa,
        states={
            "liquid": liquid_grid,
            "housing": housing_grid,
            "income": DiscreteGrid(Income),
        },
        state_transitions={
            "liquid": next_liquid,
            "housing": next_housing,
            "income": MarkovTransition(income_transition),
        },
        actions={
            "consumption": consumption_grid,
            "housing_investment": housing_investment_grid,
        },
        constraints={"housing_stays_in_bounds": housing_stays_in_bounds},
        functions={
            "utility": utility,
            "housing_cost": housing_cost,
            "resources": resources,
            "savings": savings,
            "keep_housing": keep_housing,
            "serviced_housing": serviced_housing,
            "inverse_marginal_utility": inverse_marginal_utility,
            "income_value": income_value,
        },
        solver=negm_solver,
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=ages,
        regime_id_class=DS2024HousingRegimeId,
    )


def build_params(
    *,
    variant: Literal["negm", "brute"] = "negm",
    tau: float = 0.20,
    delta: float = 0.0,
    discount_factor: float = 0.945,
    gamma_c: float = 1.458,
    alpha: float = 0.66,
    return_liquid: float = 0.024,
    return_housing: float = 0.10,
    theta: float = 2.0,
    bequest_shift: float = 200.0,
    housing_min: float = 0.01,
) -> dict:
    """Calibration parameters for the DS-2024 housing model.

    Defaults mirror `InverseDCDP` `housing.py` (`r=0.024`, `r_H=0.10`,
    `beta=0.945`, `alpha=0.66`, `gamma_c=1.458`, `tau=0.20`, `theta=2`, `K=200`),
    except `delta` defaults to `0.0`; pass `delta=0.10` for the paper value (the
    keeper depreciates the held stock to `h(1 - delta)`). Pass the same `delta` to
    `build_model`.

    Args:
        variant: Must match `build_model`.
        tau: Proportional housing-transaction cost.
        delta: Housing depreciation.
        discount_factor: Discount factor `beta`.
        gamma_c: Consumption CRRA `gamma_C`.
        alpha: Housing-service weight.
        return_liquid: Liquid return `r`.
        return_housing: Housing return `r_H`.
        theta: Terminal bequest weight.
        bequest_shift: Terminal bequest shift `K`.
        housing_min: Borrowing limit `b`.

    Returns:
        The nested parameter template keyed by regime then function.
    """
    cost_params = {"delta": delta, "return_housing": return_housing, "tau": tau}
    bequest_params = {
        "return_liquid": return_liquid,
        "theta": theta,
        "bequest_shift": bequest_shift,
        "gamma_c": gamma_c,
    }
    if variant == "brute":
        alive = {
            "utility": {"gamma_c": gamma_c, "alpha": alpha},
            "housing_cost": cost_params,
            "next_liquid": {"return_liquid": return_liquid},
            "borrowing_constraint": {
                "return_liquid": return_liquid,
                "borrowing_limit": housing_min,
            },
        }
    else:
        alive = {
            "utility": {"gamma_c": gamma_c, "alpha": alpha},
            "housing_cost": cost_params,
            "resources": {"return_liquid": return_liquid},
            "next_liquid": {},
            "inverse_marginal_utility": {"gamma_c": gamma_c},
        }
    return {
        "discount_factor": discount_factor,
        "alive": alive,
        "dead": {"utility": bequest_params},
    }
