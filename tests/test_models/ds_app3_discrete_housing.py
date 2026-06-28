"""Dobrescu-Shanker (2026) Application 3 discrete-housing model (no-tax variant).

The DS-2026 Section 2.3 model (an extended Fella 2014) is a finite-horizon
consumption-savings problem with a *discrete* housing stock, an own-vs-rent
choice, a proportional housing-adjustment cost, and a Markov wage. This module
builds the **no-tax** variant (Tables 4 and 7), which is fully specified in the
paper; the piecewise-linear capital-income tax of Table 5 is left as a
clearly-marked hook fixed to zero (its bracket thresholds and rates are not
printed in the paper — see Q7).

## The discrete-housing mapping (Q6)

DS Section 2.3 frames the period as five nested within-period stages. In pylcm
that nest collapses to a single DC-EGM regime with one discrete choice:

- financial assets `assets` (`a >= 0`) are the continuous Euler state the Euler
  equation inverts on, and `consumption` (`c`) is the continuous action;
- the held housing stock `housing` (`H`) is a discrete state carried as a
  value-function grid axis;
- the own-vs-rent-and-level decision is a single discrete action
  `housing_choice` over the same categorical alphabet as `housing` — exactly the
  role Application 1's work/retire `labor_supply` action plays. The
  discrete-choice upper envelope (FUES/MSS/LTM) selects the best choice per
  asset-housing-wage cell, and grid search (VFI) does the same by brute force —
  the four Table 4 methods;
- the next-period housing stock equals the chosen code (`next_housing =
  housing_choice`), so the discrete action drives the discrete-state transition.

This is *not* NEGM: NEGM nests an inner Euler solve inside an outer search over a
*continuous* durable margin, whereas App.3's housing is discrete, so the choice
is a plain discrete action and no outer durable search is needed.

### Own, rent, and housing services

`housing_choice` ranges over `rent` plus the owned stock levels
`own_h1..own_h5`. The serviced housing `h` that enters utility is:

- the chosen stock `H'` when owning (`h = H'`),
- the rental service level `S` when renting (`h = S`).

To keep the problem a single discrete-action DC-EGM regime (one continuous Euler
state, no second continuous margin), the renter's service `S` is tied to the
chosen owned-stock level the agent would otherwise hold — i.e. `rent` provides
the smallest service level on the housing-services grid. Letting the renter pick
`S` on the full owned-stock grid would simply add `rent_s2..rent_s5` codes to the
same discrete action; the single-`rent` form here is the smallest faithful
encoding for the construction probe and is flagged as a judgment call (Q8).

## Budgets (no tax, eq. 27 hook fixed to zero)

The period budget `a' + c = (1 + r) a - T(a) + y + <housing flow>` is written as
a resources function `R = (1 + r) a - T(a) + y + <housing flow>` with the
DC-EGM consumption recovery `c = R - a'`:

- owner: `<flow> = d_adj * H - d_adj * (1 + tau) * H'`, where `d_adj` is `1` when
  the stock changes (`H' != H`) and `0` otherwise — selling the old house and
  buying the new one at the adjustment cost `tau * H'`,
- renter: `<flow> = H - P_r * S` — selling the entire held house and renting
  services `S` at the rental price `P_r`.

The capital-income tax `T(a) = capital_income_tax * a` is the Table 5 hook; the
no-tax variant fixes `capital_income_tax = 0`.

## Calibration (Table 4/7 notes)

`r = 0.06`, `beta = 0.93`, `T = 20`, `alpha = 0.77`, `tau = 0.07`, `kappa =
0.075`, `iota = 0.01`, `A_max = 40`, `H_max = 5`, `P_r = 0.1`, wage AR1 `rho =
0.977`, `sigma_eta = 0.024`, `sigma_eps = 0.063`. Table 7 (Fella replication)
uses the bequest weight `theta = 0.5`.
"""

from typing import Literal

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    RouwenhorstAR1Process,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, GridSearch
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)

# Lifecycle: T = 20 periods. The last period is the terminal bequest regime, so
# there are 19 decision periods. Ages are abstract unit steps from 0.
START_AGE = 0
N_PERIODS = 20

# Housing-stock levels the owner can hold (H_max = 5). `rent` carries stock 0.
OWNED_STOCK_LEVELS = (1.0, 2.0, 3.0, 4.0, 5.0)

# Number of Markov wage discretisation nodes; kept small so the construction
# test is fast and the eventual solve fits a GPU.
N_WAGE_NODES = 5

# The name of the function the adjustment cost `tau` and the capital-income tax
# hook parametrise, exposed so the construction test can locate them in the
# params template without hardcoding the function name.
RESOURCES_FUNCTION_NAME = "resources"

# Piecewise-linear capital-income (asset-return) tax schedule for the with-tax
# variant (Table 5). The paper prints the bracket numbers only in Figure 9; these
# are the authors' replication values, transcribed from the `tax_table.brackets`
# of GitHub `akshayshanker/FUES`, path
# `examples/housing_renting/config_HR/STD_RES_SETTINGS_4_TAXES/master.yml`, pinned
# at commit `00961e0b588fdaa3ea3d740ebc13a8e6d230b26d` (blob
# `41661779df89167c91892bcfe6c7ffe16f259f8d`). It is an Australia-2015-16 schedule
# written directly as a function of assets, `T(a) = B + tau_a * (a - a0)` on each
# bracket `[a0, a1)` (the `B`/`tau_a` of each bracket). Three level discontinuities
# (up +0.10 at a=3.87, down ~0.14 at a=6.97 where the subsidy bracket resets the
# offset to 0.05, up +0.15 at a=15) plus rate kinks make the budget non-monotone —
# why Table 5 compares only FUES vs VFI. The resolved `(LOWER, OFFSET, RATE)` arrays
# below hash to sha256[:16] `4e8d1bf0748f6933`; re-derive from the pinned commit to
# detect upstream drift.
TAX_BRACKET_LOWER = (0.0, 2.20, 2.50, 2.75, 3.87, 6.97, 8.36, 12.0, 15.0, 20.0)
TAX_BRACKET_OFFSET = (
    0.0,
    0.0,
    0.00342,
    0.00852,
    0.11412,
    0.05,
    0.076688,
    0.174968,
    0.378968,
    0.525968,
)
TAX_BRACKET_RATE = (
    0.0,
    0.0114,
    0.0204,
    0.005,
    0.024,
    0.0192,
    0.027,
    0.018,
    0.0294,
    0.0294,
)


def piecewise_capital_income_tax(assets: ContinuousState) -> FloatND:
    """Piecewise-linear capital-income tax `T(a)` on the asset return.

    Selects the bracket `[a0, a1)` containing `assets` (left-closed, so a value on
    a boundary uses the upper bracket) and returns `B + tau_a * (a - a0)`. The
    bracket offsets `B` carry the level discontinuities, so `T` is discontinuous
    at the bracket boundaries — the non-monotone budget the FUES upper envelope is
    built for. There are three: `T` jumps up at a=3.87 (+0.10) and a=15 (+0.15),
    and drops down by about 0.14 at a=6.97, where the subsidy bracket `[6.97,
    8.36)` resets the offset to 0.05 below the preceding bracket's level.
    """
    lower = jnp.asarray(TAX_BRACKET_LOWER)
    offset = jnp.asarray(TAX_BRACKET_OFFSET)
    rate = jnp.asarray(TAX_BRACKET_RATE)
    index = jnp.clip(
        jnp.searchsorted(lower, assets, side="right") - 1, 0, lower.size - 1
    )
    return offset[index] + rate[index] * (assets - lower[index])


@categorical(ordered=False)
class DiscreteHousingRegimeId:
    """Lifecycle regimes: the working life and the terminal bequest regime."""

    working: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Housing:
    """Held / chosen housing: renting (stock 0) plus the owned stock levels."""

    rent: ScalarInt
    own_h1: ScalarInt
    own_h2: ScalarInt
    own_h3: ScalarInt
    own_h4: ScalarInt
    own_h5: ScalarInt


def _owned_stock_array() -> FloatND:
    """Map each `Housing` code to its owned housing stock (`rent` holds 0)."""
    return jnp.asarray((0.0, *OWNED_STOCK_LEVELS))


def housing_stock(housing: DiscreteState) -> FloatND:
    """Held housing stock `H` of the current discrete housing state."""
    return _owned_stock_array()[housing]


def chosen_stock(housing_choice: DiscreteAction) -> FloatND:
    """Next-period owned housing stock `H'` implied by the choice (0 if renting)."""
    return _owned_stock_array()[housing_choice]


def is_renting(housing_choice: DiscreteAction) -> BoolND:
    """Whether the choice is to rent (own nothing next period)."""
    return housing_choice == Housing.rent


def serviced_housing(housing_choice: DiscreteAction, rental_service: float) -> FloatND:
    """Housing services `h` consumed this period.

    Owning serves the chosen stock `H'`; renting serves the rental service level
    `S` (the smallest service on the owned-stock grid — see Q8).
    """
    return jnp.where(
        is_renting(housing_choice), rental_service, chosen_stock(housing_choice)
    )


def utility(
    consumption: ContinuousAction,
    serviced_housing: FloatND,
    alpha: float,
    kappa: float,
    iota: float,
) -> FloatND:
    """Cobb-Douglas log utility over consumption and housing services (eq. 30).

    `u(c, h) = alpha * log(c) + (1 - alpha) * log(kappa * (h + iota))`, with `h`
    the serviced housing (the chosen stock when owning, the rental service when
    renting). Reads only the continuous consumption action and the discrete
    housing choice — never the financial-asset Euler state — so the DC-EGM
    envelope condition on the Euler state holds.
    """
    return alpha * jnp.log(consumption) + (1.0 - alpha) * jnp.log(
        kappa * (serviced_housing + iota)
    )


def wage_income(wage: ContinuousState) -> FloatND:
    """Map the Markov log-wage node to its labor-income level `y = exp(wage)`.

    A discretised AR1 wage is a pylcm stochastic *process* state, annotated as a
    continuous state (its node value is the log wage); income is its exponential.
    """
    return jnp.exp(wage)


def housing_flow(
    housing: DiscreteState,
    housing_choice: DiscreteAction,
    tau: float,
    rental_price: float,
    rental_service: float,
) -> FloatND:
    """Net liquid housing flow in the budget (no tax).

    - owner (`H' > 0`): `d_adj * H - d_adj * (1 + tau) * H'`, where `d_adj` is 1
      when the stock changes and 0 otherwise — selling the held house and buying
      the new one at the adjustment cost `tau * H'`,
    - renter (`H' = 0`): `H - rental_price * rental_service` — selling the entire
      held house and renting services at the rental price.

    Reads only the held housing state and the discrete choice — never the
    consumption action or the financial-asset Euler state — so it enters the
    DC-EGM resources as a constant per discrete-choice cell.
    """
    held = housing_stock(housing)
    chosen = chosen_stock(housing_choice)
    adjusts = housing != housing_choice
    adjustment_indicator = jnp.where(adjusts, 1.0, 0.0)
    owner_flow = (
        adjustment_indicator * held - adjustment_indicator * (1.0 + tau) * chosen
    )
    renter_flow = held - rental_price * rental_service
    return jnp.where(is_renting(housing_choice), renter_flow, owner_flow)


def resources(
    assets: ContinuousState,
    wage_income: FloatND,
    housing_flow: FloatND,
    interest_rate: float,
    capital_income_tax: float,
) -> FloatND:
    """Financial resources consumption is paid out of, given the discrete choice.

    `(1 + r) a - T(a) + y + housing_flow`, with the capital-income tax hook
    `T(a) = capital_income_tax * a` (Q7; the no-tax variant fixes the rate to 0).
    Strictly increasing in the Euler state `assets` for `capital_income_tax <
    1 + r`; independent of the continuous consumption action.
    """
    tax = capital_income_tax * assets
    return (1.0 + interest_rate) * assets - tax + wage_income + housing_flow


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Post-decision financial balance `a' = resources - c`."""
    return resources - consumption


def next_assets(savings: FloatND) -> ContinuousState:
    """Euler-state law: next financial assets equal post-decision savings."""
    return savings


def next_housing(housing_choice: DiscreteAction) -> DiscreteState:
    """Discrete housing law: next housing equals the chosen code.

    The discrete action `housing_choice` and the discrete state `housing` share
    the `Housing` alphabet, so this deterministic transition maps codes 1:1.
    """
    return housing_choice


def inverse_marginal_utility(marginal_continuation: FloatND, alpha: float) -> FloatND:
    """Invert the consumption marginal utility `u'(c) = alpha / c`.

    `c = alpha / mc`. The housing-service term is additively separable from
    consumption, so it drops out of the inner consumption inversion; the
    consumption weight `alpha` scales the marginal utility, so it enters the
    inverse.
    """
    return alpha / marginal_continuation


def next_assets_brute(
    assets: ContinuousState,
    wage_income: FloatND,
    housing_flow: FloatND,
    consumption: ContinuousAction,
    interest_rate: float,
    capital_income_tax: float,
) -> ContinuousState:
    """Brute-force financial-asset law: resources minus consumption.

    Reads the continuous consumption action directly rather than splitting
    resources and a post-decision `savings`, so the grid-searched twin needs no
    Euler machinery. Equal to `resources - consumption` of the DC-EGM regime.
    """
    tax = capital_income_tax * assets
    return (
        (1.0 + interest_rate) * assets - tax + wage_income + housing_flow - consumption
    )


def borrowing_constraint(
    assets: ContinuousState,
    wage_income: FloatND,
    housing_flow: FloatND,
    consumption: ContinuousAction,
    interest_rate: float,
    capital_income_tax: float,
) -> BoolND:
    """Keep post-decision financial assets non-negative (`a' >= 0`).

    Mirrors the DC-EGM savings grid, whose floor is the no-borrowing limit:
    feasible consumption leaves `next_assets_brute >= 0`.
    """
    return (
        next_assets_brute(
            assets=assets,
            wage_income=wage_income,
            housing_flow=housing_flow,
            consumption=consumption,
            interest_rate=interest_rate,
            capital_income_tax=capital_income_tax,
        )
        >= 0.0
    )


def resources_taxed(
    assets: ContinuousState,
    wage_income: FloatND,
    housing_flow: FloatND,
    interest_rate: float,
) -> FloatND:
    """Resources with the piecewise capital-income tax (Table 5 with-tax variant).

    `(1 + r) a - T(a) + y + housing_flow`, with `T(a)` the piecewise-linear
    schedule. The tax's level jumps make resources non-monotone in `assets`, so
    the inner consumption Euler inversion meets the kinked/jumped budget the FUES
    upper envelope resolves.
    """
    return (
        (1.0 + interest_rate) * assets
        - piecewise_capital_income_tax(assets)
        + wage_income
        + housing_flow
    )


def next_assets_brute_taxed(
    assets: ContinuousState,
    wage_income: FloatND,
    housing_flow: FloatND,
    consumption: ContinuousAction,
    interest_rate: float,
) -> ContinuousState:
    """Brute-force financial-asset law with the piecewise capital-income tax."""
    return (
        (1.0 + interest_rate) * assets
        - piecewise_capital_income_tax(assets)
        + wage_income
        + housing_flow
        - consumption
    )


def borrowing_constraint_taxed(
    assets: ContinuousState,
    wage_income: FloatND,
    housing_flow: FloatND,
    consumption: ContinuousAction,
    interest_rate: float,
) -> BoolND:
    """Keep post-decision assets non-negative under the piecewise tax."""
    return (
        next_assets_brute_taxed(
            assets=assets,
            wage_income=wage_income,
            housing_flow=housing_flow,
            consumption=consumption,
            interest_rate=interest_rate,
        )
        >= 0.0
    )


def bequest(
    assets: ContinuousState,
    housing: DiscreteState,
    interest_rate: float,
    theta: float,
) -> FloatND:
    """Terminal bequest `theta * ((1 + r) a + H)` (Table 7 uses `theta = 0.5`).

    Reads only the carried financial-asset and discrete housing states.
    """
    estate = (1.0 + interest_rate) * assets + housing_stock(housing)
    return theta * estate


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    """Transition to `dead` at the final living age, else stay working."""
    return jnp.where(
        age >= final_age_alive,
        DiscreteHousingRegimeId.dead,
        DiscreteHousingRegimeId.working,
    )


def build_model(
    *,
    variant: Literal["dcegm", "brute"] = "dcegm",
    n_assets: int,
    n_wage_nodes: int = N_WAGE_NODES,
    n_periods: int | None = None,
    asset_max: float = 40.0,
    n_consumption: int = 30,
    upper_envelope: Literal["fues", "mss", "ltm", "rfc"] = "fues",
    use_taxes: bool = False,
) -> Model:
    """Build the DS App.3 discrete-housing model.

    Args:
        variant: `"dcegm"` builds the DC-EGM regime (financial assets the Euler
            state, consumption inverted, the housing choice a discrete action);
            `"brute"` builds the grid-search (VFI) twin solving the same
            economics with no Euler machinery — the Table 4 methods.
        use_taxes: When `True`, the budget carries the piecewise-linear
            capital-income tax `T(a)` (the with-tax Table 5 variant, FUES vs VFI
            only); when `False`, the no-tax Table 4/7 budget.
        n_assets: Number of financial-asset grid points; also the number of
            clustered exogenous savings nodes the DC-EGM solver scans.
        n_wage_nodes: Number of Markov wage discretisation nodes (the paper uses
            7; kept small for construction).
        n_periods: Optional shortened horizon for construction tests; `None`
            uses the paper's `T = 20`.
        asset_max: Upper bound of the financial-asset grid `a in [0, asset_max]`.
        n_consumption: Number of consumption-grid points (brute search and the
            DC-EGM simulation draw).
        upper_envelope: DC-EGM upper-envelope backend (`"fues"`, `"mss"`, `"ltm"`,
            `"rfc"`); the Table 4 methods are FUES/MSS/LTM.

    Returns:
        The two-regime (working, dead) discrete-housing model.
    """
    n_periods = N_PERIODS if n_periods is None else n_periods
    ages = AgeGrid(start=START_AGE, stop=START_AGE + n_periods - 1, step="Y")
    final_age_alive = int(ages.exact_values[-1])

    assets_grid = LinSpacedGrid(start=0.0, stop=asset_max, n_points=n_assets)
    consumption_grid = LinSpacedGrid(start=0.05, stop=asset_max, n_points=n_consumption)
    # Cubically clustered savings nodes toward the no-borrowing limit, where the
    # value function curves hardest; the floor (0) encodes `consumption <=
    # resources`.
    savings_grid = IrregSpacedGrid(
        points=tuple(asset_max * (i / (n_assets - 1)) ** 3 for i in range(n_assets))
    )
    wage_process = RouwenhorstAR1Process(n_points=n_wage_nodes)

    resources_func = resources_taxed if use_taxes else resources
    asset_law_brute = next_assets_brute_taxed if use_taxes else next_assets_brute
    borrowing = borrowing_constraint_taxed if use_taxes else borrowing_constraint

    dead = UserRegime(
        transition=None,
        active=lambda age, fa=final_age_alive: age >= fa,
        states={"assets": assets_grid, "housing": DiscreteGrid(Housing)},
        functions={"utility": bequest},
    )

    if variant == "brute":
        working = UserRegime(
            transition=next_regime,
            active=lambda age, fa=final_age_alive: age < fa,
            states={
                "assets": assets_grid,
                "housing": DiscreteGrid(Housing),
                "wage": wage_process,
            },
            state_transitions={
                "assets": asset_law_brute,
                "housing": next_housing,
            },
            actions={
                "consumption": consumption_grid,
                "housing_choice": DiscreteGrid(Housing),
            },
            constraints={"borrowing_constraint": borrowing},
            functions={
                "utility": utility,
                "serviced_housing": serviced_housing,
                "wage_income": wage_income,
                "housing_flow": housing_flow,
            },
            solver=GridSearch(),
        )
        return Model(
            regimes={"working": working, "dead": dead},
            ages=ages,
            regime_id_class=DiscreteHousingRegimeId,
        )

    working = UserRegime(
        transition=next_regime,
        active=lambda age, fa=final_age_alive: age < fa,
        states={
            "assets": assets_grid,
            "housing": DiscreteGrid(Housing),
            "wage": wage_process,
        },
        state_transitions={
            "assets": next_assets,
            "housing": next_housing,
        },
        actions={
            "consumption": consumption_grid,
            "housing_choice": DiscreteGrid(Housing),
        },
        functions={
            "utility": utility,
            "serviced_housing": serviced_housing,
            "wage_income": wage_income,
            "housing_flow": housing_flow,
            "resources": resources_func,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=DCEGM(
            continuous_state="assets",
            continuous_action="consumption",
            resources="resources",
            post_decision_function="savings",
            savings_grid=savings_grid,
            upper_envelope=upper_envelope,
            n_constrained_points=32,
        ),
    )
    return Model(
        regimes={"working": working, "dead": dead},
        ages=ages,
        regime_id_class=DiscreteHousingRegimeId,
    )


def build_params(
    *,
    variant: Literal["dcegm", "brute"] = "dcegm",
    tau: float = 0.07,
    theta: float = 0.5,
    discount_factor: float = 0.93,
    interest_rate: float = 0.06,
    alpha: float = 0.77,
    kappa: float = 0.075,
    iota: float = 0.01,
    rental_price: float = 0.1,
    rental_service: float = 1.0,
    capital_income_tax: float = 0.0,
    use_taxes: bool = False,
    rho_w: float = 0.977,
    sigma_w: float = 0.063,
    mu_w: float = 0.0,
    n_periods: int | None = None,
) -> dict:
    """Calibration parameters for the DS App.3 discrete-housing model (no tax).

    Args:
        variant: `"dcegm"` threads the budget params under the resources and
            inverse-marginal-utility functions; `"brute"` threads them under the
            grid-search financial-asset law and the borrowing constraint, since
            the brute twin reads consumption directly (no resources/savings
            split). Must match the `variant` passed to `build_model`.
        tau: Proportional housing-adjustment cost (Table 4/7 default `0.07`).
        theta: Terminal bequest weight (Table 7 uses `0.5`).
        discount_factor: `beta`.
        interest_rate: Net financial return `r`.
        alpha: Consumption weight in the Cobb-Douglas log utility.
        kappa: Housing-service scale.
        iota: Housing-service shifter (`log(kappa * (h + iota))`).
        rental_price: Rental price `P_r` of housing services.
        rental_service: Rental service level `S` the renter consumes (the
            smallest service on the owned-stock grid — see Q8).
        capital_income_tax: Capital-income tax rate; the Table 5 hook (Q7). The
            no-tax variant fixes it to `0.0`.
        rho_w: Markov wage persistence `rho`.
        sigma_w: Markov wage innovation std (the paper's `sigma_eps = 0.063`
            transitory term; the persistent `sigma_eta = 0.024` enters a fuller
            two-component wage not modelled in this single-process build — Q9).
        mu_w: Markov wage drift.
        n_periods: Optional shortened horizon matching `build_model`; `None` uses
            `T = 20`.

    Returns:
        The nested parameter template keyed by regime then function. The
        adjustment cost lives under the housing-flow function and the
        capital-income tax hook under the budget function; the wage-process
        params nest under the `wage` state of the working regime.
    """
    n_periods = N_PERIODS if n_periods is None else n_periods
    # The final period is the terminal bequest regime, so the last living age is
    # the second-to-last age.
    final_age_alive = n_periods - 2
    # The with-tax budget functions read the piecewise schedule directly, so they
    # take no `capital_income_tax` rate; the no-tax budget keeps the linear hook.
    budget_params = (
        {"interest_rate": interest_rate}
        if use_taxes
        else {"interest_rate": interest_rate, "capital_income_tax": capital_income_tax}
    )
    shared_working = {
        "utility": {"alpha": alpha, "kappa": kappa, "iota": iota},
        "serviced_housing": {"rental_service": rental_service},
        "housing_flow": {
            "tau": tau,
            "rental_price": rental_price,
            "rental_service": rental_service,
        },
        "wage": {"rho": rho_w, "sigma": sigma_w, "mu": mu_w},
    }
    if variant == "brute":
        # The brute law reads consumption directly: the budget params live on
        # the financial-asset law `next_assets` and the borrowing constraint.
        working = {
            **shared_working,
            "next_assets": dict(budget_params),
            "borrowing_constraint": dict(budget_params),
        }
    else:
        working = {
            **shared_working,
            RESOURCES_FUNCTION_NAME: dict(budget_params),
            "inverse_marginal_utility": {"alpha": alpha},
        }
    return {
        "discount_factor": discount_factor,
        "final_age_alive": final_age_alive,
        "working": working,
        "dead": {
            "utility": {"interest_rate": interest_rate, "theta": theta},
        },
    }
