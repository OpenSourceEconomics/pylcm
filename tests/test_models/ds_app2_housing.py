"""Dobrescu-Shanker (2026) Application 2 housing model as a pylcm NEGM model.

The DS-2026 §2.2 housing model has a liquid financial asset, an illiquid housing
stock with a proportional transaction cost, an AR1 wage, and a discrete
adjust/keep choice. It maps onto pylcm's nested-EGM solver:

- the **inner** DC-EGM solves liquid consumption-savings, with the Euler
  equation inverting on `liquid` assets `a` — the clean inverse-Euler margin;
- the **outer** durable margin is the next housing stock `H'` (`outer_action`
  = `housing_investment`, `outer_post_decision` = `next_housing`), searched over
  a grid rather than inverted — the transaction cost makes the outer value
  non-concave, so a second inverse-Euler would be invalid;
- the **keeper** (`d = 0`) holds the house (`H' = H`, no cost) and the
  **adjuster** (`d = 1`) chooses `H'` paying `(1 + τ)·H'` while selling the old
  house. NEGM builds both cores from one regime — the keeper as a passive
  DC-EGM with `next_housing = housing`, the adjuster as the inner DC-EGM with
  the outer post-decision supplied per grid node — so the adjust/keep choice is
  *not* a user-declared discrete action.

The wage is a Tauchen-discretised AR1 (`rho_w = 0.82`, `sigma_w = 0.11`) carried as a
pylcm process state in working life; it drops at retirement, where income is a
fixed pension. The lifecycle runs working (start age 20) → retired (age 60) →
dead (terminal, T = 70), with a working→retired regime transition.

Utility is separable CES over non-durable consumption and the housing serviced
this period (the new house if adjusting, else the held house). The serviced
house reads the outer post-decision `next_housing`, additively separable from
consumption, so the inner Euler inversion treats the housing term as a constant
— the NEGM contract the housing margin must satisfy.

## Calibration (DS Table 3 note, spec §Calibration)

`β = 0.94`, `gamma_C = 3.5`, `gamma_H = 1.5`, `alpha = 0.70`, `r = 0.04`, `r_H = 0`,
`τ ∈ {0.05, 0.07, 0.12}` (default `0.07`), `rho_w = 0.82`, `sigma_w = 0.11`.

Two values the paper omits are documented defaults pending confirmation
(collected question Q3): the housing-utility scale `κ = 1.0` and the bequest
weight `θ̄ = 1.0`.

The CES exponent follows the standard CRRA/CES reading `(x^{1-gamma} - 1)/(1 - gamma)`,
not the OCR-ambiguous `(x^{gamma-1} - 1)/(gamma - 1)` printed in the paper (collected
question Q4): the standard form is concave and consistent with the listed
`gamma_C = 3.5 > 1`, so it is the sensible reading pending confirmation.
"""

import jax.numpy as jnp

from lcm import (
    DCEGM,
    NEGM,
    AgeGrid,
    LinSpacedGrid,
    Model,
    TauchenAR1Process,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

# Lifecycle anchors (years). Working life starts at 20, retirement at 60, and
# the terminal bequest regime is entered at T = 70.
START_AGE = 20
RETIREMENT_AGE = 60
TERMINAL_AGE = 70

# Number of wage discretisation nodes; kept small so the construction test is
# fast and the eventual solve fits a GPU.
N_WAGE_NODES = 5

# The fixed retirement pension (a placeholder income floor; the paper's
# retirement income process is not part of the App.2 Table 3 sweep).
RETIREMENT_PENSION = 0.3

# The name of the function the transaction cost `τ` parametrises, exposed so the
# construction test can locate `τ` in the params template without hardcoding it.
HOUSING_COST_FUNCTION_NAME = "housing_cost"


@categorical(ordered=False)
class HousingRegimeId:
    """Lifecycle regimes: working, retired, and the terminal bequest regime."""

    working: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def wage_income(wage: ContinuousState) -> FloatND:
    """Map the AR1 wage node to its labor-income level `y = exp(log-wage)`.

    The Tauchen process discretises the log wage; income is its exponential.
    """
    return jnp.exp(wage)


def housing_cost(
    housing: ContinuousState,
    next_housing: ContinuousState,
    return_housing: float,
    tau: float,
) -> FloatND:
    """Net liquid cost of moving the house from `H` to `next_housing` (`H'`).

    Written in net-investment form so it is exactly zero at the no-trade point
    `H' = H` — the requirement the NEGM keeper kernel imposes: the keeper holds
    the stock (`H' = H` is injected) and must face no transaction cost, exactly
    as the kinked-toy `credited` template returns zero at its no-adjustment
    node. Buying additional housing (`H' > H`) costs `(1 + τ)` per unit (face
    value plus the proportional transaction cost); selling (`H' < H`) credits
    `(1 + r_H)` per unit returned:

    - keeper (`H' = H`): cost `0` — keeping is free,
    - adjuster up (`H' > H`): cost `(1 + τ)·(H' - H)`,
    - adjuster down (`H' < H`): cost `(1 + r_H)·(H' - H) < 0` (a credit).

    The DS budget (eq. 12) writes the adjuster as selling the whole old house at
    `(1 + r_H)·H` and rebuying at `(1 + τ)·H'`, which charges a round-trip cost
    even when `H' = H`; the net-investment form here charges the proportional
    cost on the *traded* amount instead, so the no-trade point is free and the
    `max(V_keeper, V_adjuster)` aggregation is continuous through `H' = H`. The
    two coincide for `H' ≠ H` up to the resale/repurchase normalization — a
    collected question (Q6) pending confirmation of which DS intends.

    Reads only the held housing state and the outer post-decision `next_housing`
    — never the inner consumption action or the liquid Euler state — so it is a
    constant per outer-grid node, as the NEGM contract requires.
    """
    investment = next_housing - housing
    return jnp.where(
        investment >= 0.0,
        (1.0 + tau) * investment,
        (1.0 + return_housing) * investment,
    )


def resources(
    liquid: ContinuousState,
    housing_cost: FloatND,
    income: FloatND,
    return_liquid: float,
) -> FloatND:
    """Liquid resources consumption is paid out of, given the fixed outer node.

    `(1 + r)·a + y - housing_cost`. The housing cost is bound to one outer-grid
    node, so it enters the inner Euler inversion as a constant. Strictly
    increasing in the liquid Euler state `liquid`; independent of the inner
    consumption action.
    """
    return (1.0 + return_liquid) * liquid + income - housing_cost


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Inner post-decision liquid balance `a' = resources - c`."""
    return resources - consumption


def next_liquid(savings: FloatND) -> ContinuousState:
    """Euler-state law: next liquid assets equal post-decision savings.

    Reads only the post-decision `savings`, never the outer housing margin, so
    the inner Euler inversion stays independent of the housing choice — the
    NEGM nesting contract.
    """
    return savings


def next_housing(
    housing: ContinuousState, housing_investment: ContinuousAction
) -> ContinuousState:
    """Durable law of motion `H' = H + housing_investment`.

    Used as the `housing` state transition, so pylcm names its output the
    auto-generated `next_housing`; the NEGM solver reads that value as its
    `outer_post_decision`, bound per outer-grid node into the inner resources
    DAG and the serviced-housing service flow.
    """
    return housing + housing_investment


def keep_housing(housing: ContinuousState) -> FloatND:
    """The no-adjustment candidate `H' = H` (the adjustment-cost kink)."""
    return housing


def serviced_housing(next_housing: ContinuousState) -> FloatND:
    """The house lived in this period — the new house `H'`.

    For the adjuster this is the chosen house; for the keeper the injected
    identity `H' = H` makes it the held house. Reads only the outer
    post-decision, so the housing service flow is additively separable from the
    inner consumption action.
    """
    return next_housing


def utility(
    consumption: ContinuousAction,
    serviced_housing: FloatND,
    alpha: float,
    gamma_c: float,
    gamma_h: float,
    kappa: float,
) -> FloatND:
    """Separable CES utility over consumption and serviced housing (eq. 26).

    `u(c, H) = alpha·(c^{1-gamma_C} - 1)/(1 - gamma_C)
             + (1 - alpha)·κ·(H^{1-gamma_H} - 1)/(1 - gamma_H)`.

    Uses the standard CRRA/CES exponent `1 - gamma` (the OCR-ambiguous paper print
    `gamma - 1` is read as the standard concave form pending confirmation — Q4).
    Reads the inner consumption action and the serviced-housing service flow
    additively, never the liquid Euler state — the DC-EGM envelope condition —
    and the two service terms live in distinct functions, so no single function
    couples consumption to the outer margin.
    """
    consumption_utility = (consumption ** (1.0 - gamma_c) - 1.0) / (1.0 - gamma_c)
    housing_utility = (serviced_housing ** (1.0 - gamma_h) - 1.0) / (1.0 - gamma_h)
    return alpha * consumption_utility + (1.0 - alpha) * kappa * housing_utility


def inverse_marginal_utility(
    marginal_continuation: FloatND, gamma_c: float, alpha: float
) -> FloatND:
    """Invert the consumption marginal utility `u'(c) = alpha·c^{-gamma_C}`.

    `c = (mc / alpha)^{-1/gamma_C}`. The housing-service term is additively
    separable from consumption, so it drops out of the inner consumption
    inversion; the consumption weight `alpha` scales the marginal utility, so it
    enters the inverse — the round-trip `(u')^{-1}(u'(c)) = c` the DC-EGM
    validator checks against `jax.grad(utility)` holds only with it.
    """
    return (marginal_continuation / alpha) ** (-1.0 / gamma_c)


def bequest(
    liquid: ContinuousState,
    housing: ContinuousState,
    return_liquid: float,
    theta_bar: float,
    gamma_c: float,
) -> FloatND:
    """Terminal bequest `θ((1 + r)·a + H)`, with `θ(x) = θ̄·u(x, 0)`.

    The CRRA bequest reads only the carried liquid and housing states. The
    bequest weight `θ̄` is a documented default (`1.0`) pending confirmation
    (Q3).
    """
    estate = (1.0 + return_liquid) * liquid + housing
    return theta_bar * (estate ** (1.0 - gamma_c) - 1.0) / (1.0 - gamma_c)


def next_regime(age: int) -> ScalarInt:
    """Working → retired at the retirement age, → dead at the terminal age."""
    return jnp.where(
        age + 1 >= TERMINAL_AGE,
        HousingRegimeId.dead,
        jnp.where(
            age + 1 >= RETIREMENT_AGE,
            HousingRegimeId.retired,
            HousingRegimeId.working,
        ),
    )


def next_regime_from_retired(age: int) -> ScalarInt:
    """Retired → dead at the terminal age, else stay retired."""
    return jnp.where(
        age + 1 >= TERMINAL_AGE,
        HousingRegimeId.dead,
        HousingRegimeId.retired,
    )


def _working_income(wage_income: FloatND) -> FloatND:
    """Working-life income: the AR1 wage level."""
    return wage_income


def _retirement_income() -> FloatND:
    """Retirement income: the fixed pension."""
    return jnp.asarray(RETIREMENT_PENSION)


def _euler_coupled_next_liquid(
    savings: FloatND, next_housing: ContinuousState
) -> ContinuousState:
    """Euler-coupled liquid law for the rejection guardrail test.

    Feeding the outer housing post-decision `next_housing` into the inner
    Euler-state transition couples the inner Euler inversion to the housing
    choice — the DS pension shape NEGM forbids. The validator must reject this
    with the 2-D-EGM pointer.
    """
    return savings + 0.0 * next_housing


def build_model(
    *,
    n_grid: int,
    n_periods: int | None = None,
    liquid_max: float = 50.0,
    housing_max: float = 20.0,
    consumption_max: float = 50.0,
    n_consumption: int = 30,
    n_savings: int = 60,
    _euler_couple_housing: bool = False,
) -> Model:
    """Build the DS App.2 housing NEGM model.

    Args:
        n_grid: Number of points on the liquid, housing, and outer
            housing-investment grids (DS sweeps `NG ∈ {250, 500, 750, 1000}`).
        n_periods: Optional shortened horizon for construction tests; `None`
            uses the paper's working → retired → dead lifecycle from age 20 to
            the terminal age 70.
        liquid_max: Upper bound of the liquid-asset grid.
        housing_max: Upper bound of the housing and outer grids.
        consumption_max: Upper bound of the inner consumption action grid.
        n_consumption: Number of inner consumption-grid points.
        n_savings: Number of inner savings-grid points.
        _euler_couple_housing: Test-only flag that wires the outer housing
            post-decision into the inner Euler-state law, so the NEGM contract
            rejects the model — confirming the accepted model is not accepted by
            accident.

    Returns:
        The two-NEGM-regime (working, retired) plus terminal (dead) housing
        model.
    """
    if n_periods is None:
        ages = AgeGrid(start=START_AGE, stop=TERMINAL_AGE, step="Y")
    else:
        ages = AgeGrid(start=START_AGE, stop=START_AGE + n_periods - 1, step="Y")
    final_age = int(ages.exact_values[-1])
    retirement_age = min(RETIREMENT_AGE, final_age)

    liquid_grid = LinSpacedGrid(start=0.0, stop=liquid_max, n_points=n_grid)
    housing_grid = LinSpacedGrid(start=0.0, stop=housing_max, n_points=n_grid)
    outer_grid = LinSpacedGrid(start=0.0, stop=housing_max, n_points=n_grid)
    consumption_grid = LinSpacedGrid(
        start=0.05, stop=consumption_max, n_points=n_consumption
    )
    housing_investment_grid = LinSpacedGrid(
        start=-housing_max, stop=housing_max, n_points=n_grid
    )
    savings_grid = LinSpacedGrid(
        start=0.0, stop=liquid_max + housing_max, n_points=n_savings
    )

    inner_liquid_law = (
        _euler_coupled_next_liquid if _euler_couple_housing else next_liquid
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
    )

    shared_functions = {
        "utility": utility,
        "housing_cost": housing_cost,
        "resources": resources,
        "savings": savings,
        "keep_housing": keep_housing,
        "serviced_housing": serviced_housing,
        "inverse_marginal_utility": inverse_marginal_utility,
    }

    working = UserRegime(
        transition=next_regime,
        active=lambda age, ra=retirement_age: age < ra,
        states={
            "liquid": liquid_grid,
            "housing": housing_grid,
            "wage": TauchenAR1Process(n_points=N_WAGE_NODES, gauss_hermite=True),
        },
        state_transitions={
            "liquid": inner_liquid_law,
            "housing": next_housing,
        },
        actions={
            "consumption": consumption_grid,
            "housing_investment": housing_investment_grid,
        },
        functions={
            **shared_functions,
            "income": _working_income,
            "wage_income": wage_income,
        },
        solver=negm_solver,
    )

    retired = UserRegime(
        transition=next_regime_from_retired,
        active=lambda age, ra=retirement_age, fa=final_age: ra <= age < fa,
        states={"liquid": liquid_grid, "housing": housing_grid},
        state_transitions={
            "liquid": inner_liquid_law,
            "housing": next_housing,
        },
        actions={
            "consumption": consumption_grid,
            "housing_investment": housing_investment_grid,
        },
        functions={**shared_functions, "income": _retirement_income},
        solver=negm_solver,
    )

    dead = UserRegime(
        transition=None,
        active=lambda age, fa=final_age: age >= fa,
        states={"liquid": liquid_grid, "housing": housing_grid},
        functions={"utility": bequest},
    )

    return Model(
        regimes={"working": working, "retired": retired, "dead": dead},
        ages=ages,
        regime_id_class=HousingRegimeId,
    )


def build_params(
    *,
    tau: float = 0.07,
    discount_factor: float = 0.94,
    gamma_c: float = 3.5,
    gamma_h: float = 1.5,
    alpha: float = 0.70,
    kappa: float = 1.0,
    theta_bar: float = 1.0,
    return_liquid: float = 0.04,
    return_housing: float = 0.0,
    rho_w: float = 0.82,
    sigma_w: float = 0.11,
    mu_w: float = 0.0,
) -> dict:
    """Calibration parameters for the DS App.2 housing model.

    Args:
        tau: Proportional housing-transaction cost (DS sweeps `{0.05, 0.07,
            0.12}`, default `0.07`).
        discount_factor: `β`.
        gamma_c: Consumption CRRA `gamma_C`.
        gamma_h: Housing CES `gamma_H`.
        alpha: Consumption weight `alpha`.
        kappa: Housing-utility scale (documented default `1.0` pending
            confirmation — Q3).
        theta_bar: Bequest weight (documented default `1.0` pending
            confirmation — Q3).
        return_liquid: Liquid return `r`.
        return_housing: Housing return `r_H`.
        rho_w: AR1 wage persistence `rho_w`.
        sigma_w: AR1 wage innovation std `sigma_w`.
        mu_w: AR1 wage drift `μ_w`.

    Returns:
        The nested parameter template keyed by regime then function. The
        transaction cost lives under the housing-cost function, where the outer
        durable margin reads it; the wage-process params nest under the `wage`
        state of the working regime.
    """
    utility_params = {
        "alpha": alpha,
        "gamma_c": gamma_c,
        "gamma_h": gamma_h,
        "kappa": kappa,
    }
    housing_cost_params = {"return_housing": return_housing, "tau": tau}
    wage_params = {"rho": rho_w, "sigma": sigma_w, "mu": mu_w}
    return {
        "discount_factor": discount_factor,
        "working": {
            "utility": utility_params,
            HOUSING_COST_FUNCTION_NAME: housing_cost_params,
            "resources": {"return_liquid": return_liquid},
            "next_liquid": {},
            "inverse_marginal_utility": {"gamma_c": gamma_c, "alpha": alpha},
            "wage": wage_params,
        },
        "retired": {
            "utility": utility_params,
            HOUSING_COST_FUNCTION_NAME: housing_cost_params,
            "resources": {"return_liquid": return_liquid},
            "next_liquid": {},
            "inverse_marginal_utility": {"gamma_c": gamma_c, "alpha": alpha},
        },
        "dead": {
            "utility": {
                "return_liquid": return_liquid,
                "theta_bar": theta_bar,
                "gamma_c": gamma_c,
            },
        },
    }
