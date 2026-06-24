"""Dobrescu-Shanker (2026) Application 2 housing — EGM-FUES discrete-grid variant.

The DS-2026 Section 2.2 housing model is compared in Table 3 by two methods,
**EGM-FUES** and **NEGM**. The NEGM column is pylcm's nested-EGM solver over a
*continuous* housing margin (`ds_app2_housing.py`). This module builds the
**EGM-FUES** column: the paper solves it with one-dimensional FUES nested over a
fixed housing grid (its Box 2) — for each next-housing level `H'` on an exogenous
grid it inverts the liquid-asset Euler equation and pools the candidates, then a
single 1-D FUES upper envelope over the wealth grid selects the optimal policy
across housing choices.

In pylcm that pooled-candidate upper envelope is exactly the discrete-choice
DC-EGM upper envelope (FUES/MSS/LTM): discretise the next-housing choice onto the
housing grid and treat it as a discrete action, and the inner liquid-asset DC-EGM
plus the discrete-choice envelope reproduces the EGM-FUES solve — the same shape
as Application 3's discrete-housing model, with Application 2's separable-CES
utility and proportional transaction cost. As the housing grid refines this
converges to the paper's continuous EGM-FUES.

## The discrete-housing mapping

- liquid financial assets `liquid` (`a >= 0`) are the continuous Euler state the
  Euler equation inverts on, and `consumption` (`c`) is the continuous action;
- the held housing stock `housing` (`H`) is a discrete state carried as a
  value-function grid axis over the housing-level alphabet;
- the next-housing choice is a discrete action `housing_choice` over the same
  alphabet; the discrete-choice upper envelope selects the best `H'` per
  liquid-housing-wage cell (FUES/MSS/LTM), and grid search (VFI/brute) does the
  same by brute force — the Table 3 methods;
- `next_housing = housing_choice`, so the discrete action drives the discrete
  housing transition. Choosing `housing_choice == housing` is the no-trade
  (keeper) option and is always available, so its zero-cost candidate is in the
  envelope by construction — no separate keeper kernel and no below-floor corner.

## Budget and utility (Application 2 calibration)

The housing levels are an `n_housing`-point grid over `[housing_min, housing_max]`
floored at `housing_min = housing_max / (2 * n_housing)`, so the separable CES
housing service flow `H'^{1-gamma_H}` never hits its `H' = 0` singularity. The
proportional transaction cost is charged net-investment form so the no-trade
point is free:

- buying (`H' > H`): cost `(1 + tau) * (H' - H)`,
- selling (`H' < H`): cost `(1 + r_H) * (H' - H)` (a credit),
- keeping (`H' = H`): cost `0`.

Liquid resources are `R = (1 + r) * a + y - housing_cost`, with DC-EGM
consumption recovery `c = R - a'`. Utility is the Application 2 separable CES
`u(c, H') = alpha*(c^{1-gamma_C} - 1)/(1 - gamma_C)
+ (1-alpha)*kappa*(H'^{1-gamma_H} - 1)/(1 - gamma_H)`, so the consumption
marginal utility is `u'(c) = alpha*c^{-gamma_C}` and the housing-service term is
additively separable — the DC-EGM envelope condition on the liquid Euler state
holds, with `c = (alpha / mc)^{1/gamma_C}` the inverse marginal utility.
"""

from typing import Literal

import jax.numpy as jnp

from lcm import (
    DCEGM,
    AgeGrid,
    DiscreteGrid,
    GridSearch,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    TauchenAR1Process,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)

# Lifecycle anchors. The working life starts at 20, retires at 60, and the
# terminal bequest regime is entered at T = 70. The short default horizon keeps
# the construction probe local-safe; the paper uses the full lifecycle.
START_AGE = 20
RETIREMENT_AGE = 60
TERMINAL_AGE = 70

# Number of wage discretisation nodes; small so construction is fast.
N_WAGE_NODES = 5

# Fixed retirement pension (an income floor; the App.2 Table 3 sweep is the
# working-life housing problem).
RETIREMENT_PENSION = 0.3


@categorical(ordered=False)
class HousingFuesRegimeId:
    """Lifecycle regimes: working, retired, and the terminal bequest regime."""

    working: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _make_housing_levels(*, n_housing: int) -> type:
    """Create an ordered categorical with one field per discrete housing level."""
    annotations = {f"h{i}": ScalarInt for i in range(n_housing)}
    cls = type("HousingLevels", (), {"__annotations__": annotations})
    return categorical(ordered=True)(cls)


def wage_income(wage: ContinuousState) -> FloatND:
    """Map the AR1 wage node to its labor income `y = exp(log-wage)`."""
    return jnp.exp(wage)


def _working_income(wage_income: FloatND) -> FloatND:
    """Working-life income is the wage income."""
    return wage_income


def _retirement_income(retirement_pension: float) -> FloatND:
    """Retirement income is the fixed pension."""
    return jnp.asarray(retirement_pension)


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Inner post-decision liquid balance `a' = resources - c`."""
    return resources - consumption


def next_liquid(savings: FloatND) -> ContinuousState:
    """Euler-state law: next liquid assets equal post-decision savings."""
    return savings


def next_housing(housing_choice: DiscreteAction) -> DiscreteState:
    """Discrete housing law: next housing equals the chosen code.

    The discrete action `housing_choice` and the discrete state `housing` share
    the housing-level alphabet, so this deterministic transition maps codes 1:1.
    """
    return housing_choice


def utility(
    consumption: ContinuousAction,
    serviced_housing: FloatND,
    alpha: float,
    kappa: float,
    gamma_c: float,
    gamma_h: float,
) -> FloatND:
    """Application 2 separable CES utility over consumption and serviced housing.

    `u(c, H') = alpha*(c^{1-gamma_C} - 1)/(1 - gamma_C)
    + (1-alpha)*kappa*(H'^{1-gamma_H} - 1)/(1 - gamma_H)`. The housing-service
    term is additively separable from consumption, so it drops from the inner
    consumption Euler inversion. Reads the serviced housing (the chosen stock)
    and the consumption action — never the liquid Euler state.
    """
    consumption_term = (consumption ** (1.0 - gamma_c) - 1.0) / (1.0 - gamma_c)
    housing_term = (serviced_housing ** (1.0 - gamma_h) - 1.0) / (1.0 - gamma_h)
    return alpha * consumption_term + (1.0 - alpha) * kappa * housing_term


def inverse_marginal_utility(
    marginal_continuation: FloatND, alpha: float, gamma_c: float
) -> FloatND:
    """Invert the consumption marginal utility `u'(c) = alpha*c^{-gamma_C}`.

    `c = (alpha / mc)^{1/gamma_C}`. The separable housing term drops from the
    inner inversion; the consumption weight `alpha` and curvature `gamma_C`
    enter the inverse.
    """
    return (alpha / marginal_continuation) ** (1.0 / gamma_c)


def _fail_if_too_few_housing_levels(*, n_housing: int) -> None:
    """Reject a housing alphabet too small to space the stock levels.

    The discrete stock levels are spaced by `(n_housing - 1)`, so at least two
    levels are required.
    """
    if n_housing < 2:
        msg = (
            "n_housing must be at least 2: the discrete housing-level spacing "
            f"divides by (n_housing - 1), got n_housing={n_housing}."
        )
        raise ValueError(msg)


def build_model(  # noqa: C901
    *,
    variant: Literal["dcegm", "brute"] = "dcegm",
    n_grid: int,
    n_housing: int | None = None,
    n_consumption: int = 60,
    n_savings: int | None = None,
    liquid_max: float = 50.0,
    housing_max: float = 20.0,
    n_periods: int | None = None,
    upper_envelope: Literal["fues", "mss", "ltm", "rfc"] = "fues",
) -> Model:
    """Build the DS App.2 housing EGM-FUES discrete-housing model.

    Args:
        variant: `"dcegm"` builds the discrete-choice DC-EGM regime (liquid assets
            the Euler state, consumption inverted, the housing choice a discrete
            action solved by the upper envelope) — the EGM-FUES column; `"brute"`
            builds the grid-search (VFI) twin solving the same economics with no
            Euler machinery.
        n_grid: Number of liquid-asset grid points; also the number of clustered
            exogenous savings nodes the DC-EGM solver scans.
        n_housing: Number of discrete housing levels; defaults to `n_grid` (the
            paper sizes the housing and asset grids together as `NG`).
        n_consumption: Number of consumption-grid points (brute search and the
            DC-EGM simulation draw).
        n_savings: Number of savings-grid nodes; defaults to `n_grid`.
        liquid_max: Upper bound of the liquid-asset grid.
        housing_max: Upper bound of the housing-level grid.
        n_periods: Optional shortened horizon for construction tests; `None` uses
            the full lifecycle to `TERMINAL_AGE`.
        upper_envelope: DC-EGM upper-envelope backend; the Table 3 method is FUES.

    Returns:
        The three-regime (working, retired, dead) discrete-housing model.
    """
    n_housing = n_grid if n_housing is None else n_housing
    n_savings = n_grid if n_savings is None else n_savings
    _fail_if_too_few_housing_levels(n_housing=n_housing)

    if n_periods is None:
        ages = AgeGrid(start=START_AGE, stop=TERMINAL_AGE, step="Y")
        retirement_age = RETIREMENT_AGE
        final_age = TERMINAL_AGE
    else:
        ages = AgeGrid(start=START_AGE, stop=START_AGE + n_periods, step="Y")
        # Split the short horizon: roughly the first half works, then retires,
        # with the terminal bequest regime at the final age.
        retirement_age = START_AGE + max(1, n_periods // 2)
        final_age = START_AGE + n_periods

    def next_regime(age: int) -> ScalarInt:
        """Transition working to retired when next period reaches retirement."""
        return jnp.where(
            age + 1 >= retirement_age,
            HousingFuesRegimeId.retired,
            HousingFuesRegimeId.working,
        )

    def next_regime_from_retired(age: int) -> ScalarInt:
        """Transition retired to dead when next period reaches the final age."""
        return jnp.where(
            age + 1 >= final_age,
            HousingFuesRegimeId.dead,
            HousingFuesRegimeId.retired,
        )

    housing_min = housing_max / (2.0 * n_housing)
    stock_levels = jnp.asarray(
        [
            housing_min + (housing_max - housing_min) * i / (n_housing - 1)
            for i in range(n_housing)
        ]
    )

    housing_class = _make_housing_levels(n_housing=n_housing)
    liquid_grid = LinSpacedGrid(start=0.0, stop=liquid_max, n_points=n_grid)
    consumption_grid = LinSpacedGrid(
        start=0.05, stop=liquid_max, n_points=n_consumption
    )
    savings_grid = IrregSpacedGrid(
        points=tuple(
            (liquid_max + housing_max) * (i / (n_savings - 1)) ** 2
            for i in range(n_savings)
        )
    )

    def housing_stock(housing: DiscreteState) -> FloatND:
        """Held housing stock `H` of the current discrete housing state."""
        return stock_levels[housing]

    def chosen_stock(housing_choice: DiscreteAction) -> FloatND:
        """Next-period housing stock `H'` implied by the discrete choice."""
        return stock_levels[housing_choice]

    def serviced_housing(housing_choice: DiscreteAction) -> FloatND:
        """Serviced housing this period is the chosen next stock `H'`."""
        return stock_levels[housing_choice]

    def housing_cost(
        housing: DiscreteState,
        housing_choice: DiscreteAction,
        tau: float,
        return_housing: float,
    ) -> FloatND:
        """Net liquid cost of moving the house from `H` to `H'`.

        Net-investment form, zero at the no-trade point `H' = H`: buying costs
        `(1 + tau)` per unit, selling credits `(1 + r_H)` per unit. Reads only
        the held housing state and the discrete choice — a constant per
        discrete-choice cell in the inner resources DAG.
        """
        investment = stock_levels[housing_choice] - stock_levels[housing]
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
        """Liquid resources consumption is paid from, given the chosen house.

        `(1 + r) * a + y - housing_cost`. The housing cost is constant per
        discrete-choice cell, so it enters the inner Euler inversion as a
        constant; strictly increasing in the liquid Euler state.
        """
        return (1.0 + return_liquid) * liquid + income - housing_cost

    def next_liquid_brute(
        liquid: ContinuousState,
        housing_cost: FloatND,
        income: FloatND,
        consumption: ContinuousAction,
        return_liquid: float,
    ) -> ContinuousState:
        """Brute-force liquid law: resources minus consumption.

        Reads the consumption action and the shared `housing_cost` directly
        rather than splitting resources and a post-decision `savings`, so the
        grid-searched twin needs no Euler machinery. Equal to
        `resources - consumption` of the DC-EGM regime.
        """
        return (1.0 + return_liquid) * liquid + income - housing_cost - consumption

    def borrowing_constraint(
        liquid: ContinuousState,
        housing_cost: FloatND,
        income: FloatND,
        consumption: ContinuousAction,
        return_liquid: float,
    ) -> BoolND:
        """Keep post-decision liquid assets non-negative (`a' >= 0`)."""
        return (
            (1.0 + return_liquid) * liquid + income - housing_cost - consumption
        ) >= 0.0

    def bequest(
        liquid: ContinuousState,
        housing: DiscreteState,
        return_liquid: float,
        theta_bar: float,
    ) -> FloatND:
        """Terminal bequest `theta_bar * ((1 + r) a + H)`."""
        estate = (1.0 + return_liquid) * liquid + stock_levels[housing]
        return theta_bar * estate

    housing_grid = DiscreteGrid(housing_class)
    dead = UserRegime(
        transition=None,
        active=lambda age, fa=final_age: age >= fa,
        states={"liquid": liquid_grid, "housing": housing_grid},
        functions={"utility": bequest},
    )

    shared_econ = {
        "utility": utility,
        "serviced_housing": serviced_housing,
        "housing_cost": housing_cost,
    }

    if variant == "brute":
        working = UserRegime(
            transition=next_regime,
            active=lambda age, ra=retirement_age: age < ra,
            states={
                "liquid": liquid_grid,
                "housing": housing_grid,
                "wage": TauchenAR1Process(n_points=N_WAGE_NODES, gauss_hermite=True),
            },
            state_transitions={"liquid": next_liquid_brute, "housing": next_housing},
            actions={
                "consumption": consumption_grid,
                "housing_choice": housing_grid,
            },
            constraints={"borrowing_constraint": borrowing_constraint},
            functions={
                **shared_econ,
                "wage_income": wage_income,
                "income": _working_income,
            },
            solver=GridSearch(),
        )
        retired = UserRegime(
            transition=next_regime_from_retired,
            active=lambda age, ra=retirement_age, fa=final_age: ra <= age < fa,
            states={"liquid": liquid_grid, "housing": housing_grid},
            state_transitions={"liquid": next_liquid_brute, "housing": next_housing},
            actions={
                "consumption": consumption_grid,
                "housing_choice": housing_grid,
            },
            constraints={"borrowing_constraint": borrowing_constraint},
            functions={**shared_econ, "income": _retirement_income},
            solver=GridSearch(),
        )
        return Model(
            regimes={"working": working, "retired": retired, "dead": dead},
            ages=ages,
            regime_id_class=HousingFuesRegimeId,
        )

    inner_solver = DCEGM(
        continuous_state="liquid",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
        savings_grid=savings_grid,
        upper_envelope=upper_envelope,
        n_constrained_points=32,
    )
    working = UserRegime(
        transition=next_regime,
        active=lambda age, ra=retirement_age: age < ra,
        states={
            "liquid": liquid_grid,
            "housing": housing_grid,
            "wage": TauchenAR1Process(n_points=N_WAGE_NODES, gauss_hermite=True),
        },
        state_transitions={"liquid": next_liquid, "housing": next_housing},
        actions={
            "consumption": consumption_grid,
            "housing_choice": housing_grid,
        },
        functions={
            **shared_econ,
            "wage_income": wage_income,
            "income": _working_income,
            "resources": resources,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=inner_solver,
    )
    retired = UserRegime(
        transition=next_regime_from_retired,
        active=lambda age, ra=retirement_age, fa=final_age: ra <= age < fa,
        states={"liquid": liquid_grid, "housing": housing_grid},
        state_transitions={"liquid": next_liquid, "housing": next_housing},
        actions={
            "consumption": consumption_grid,
            "housing_choice": housing_grid,
        },
        functions={
            **shared_econ,
            "income": _retirement_income,
            "resources": resources,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=inner_solver,
    )
    return Model(
        regimes={"working": working, "retired": retired, "dead": dead},
        ages=ages,
        regime_id_class=HousingFuesRegimeId,
    )


def build_params(
    *,
    variant: Literal["dcegm", "brute"] = "dcegm",
    tau: float = 0.07,
    discount_factor: float = 0.94,
    gamma_c: float = 3.5,
    gamma_h: float = 1.5,
    alpha: float = 0.70,
    kappa: float = 1.0,
    theta_bar: float = 1.0,
    return_liquid: float = 0.04,
    return_housing: float = 0.0,
    retirement_pension: float = RETIREMENT_PENSION,
    rho_w: float = 0.82,
    sigma_w: float = 0.11,
    mu_w: float = 0.0,
) -> dict:
    """Calibration parameters for the DS App.2 EGM-FUES discrete-housing model.

    Args:
        variant: `"dcegm"` threads the budget params under the resources and
            inverse-marginal-utility functions; `"brute"` threads them under the
            grid-search liquid law and the borrowing constraint. Must match the
            `variant` passed to `build_model`.
        tau: Proportional housing-transaction cost (default `0.07`).
        discount_factor: `beta`.
        gamma_c: Consumption CRRA `gamma_C`.
        gamma_h: Housing CES `gamma_H`.
        alpha: Consumption weight in the separable CES utility.
        kappa: Housing-utility scale.
        theta_bar: Terminal bequest weight.
        return_liquid: Net liquid return `r`.
        return_housing: Net housing return `r_H`.
        retirement_pension: The fixed retirement income.
        rho_w: AR1 wage persistence.
        sigma_w: AR1 wage innovation std.
        mu_w: AR1 wage drift.

    Returns:
        The nested parameter template keyed by regime then function.
    """
    utility_params = {
        "alpha": alpha,
        "kappa": kappa,
        "gamma_c": gamma_c,
        "gamma_h": gamma_h,
    }
    cost_params = {"tau": tau, "return_housing": return_housing}
    wage_params = {"rho": rho_w, "sigma": sigma_w, "mu": mu_w}

    if variant == "brute":
        working = {
            "utility": utility_params,
            "housing_cost": cost_params,
            "next_liquid": {"return_liquid": return_liquid},
            "borrowing_constraint": {"return_liquid": return_liquid},
            "wage": wage_params,
        }
        retired = {
            "utility": utility_params,
            "housing_cost": cost_params,
            "income": {"retirement_pension": retirement_pension},
            "next_liquid": {"return_liquid": return_liquid},
            "borrowing_constraint": {"return_liquid": return_liquid},
        }
    else:
        working = {
            "utility": utility_params,
            "housing_cost": cost_params,
            "resources": {"return_liquid": return_liquid},
            "inverse_marginal_utility": {"alpha": alpha, "gamma_c": gamma_c},
            "wage": wage_params,
        }
        retired = {
            "utility": utility_params,
            "housing_cost": cost_params,
            "income": {"retirement_pension": retirement_pension},
            "resources": {"return_liquid": return_liquid},
            "inverse_marginal_utility": {"alpha": alpha, "gamma_c": gamma_c},
        }
    return {
        "discount_factor": discount_factor,
        "working": working,
        "retired": retired,
        "dead": {
            "utility": {"return_liquid": return_liquid, "theta_bar": theta_bar},
        },
    }
