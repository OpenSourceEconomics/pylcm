"""NBEGM ride-along toy whose period utility is CES, not plain CRRA.

The M1 ACA regime's period utility is a CES aggregate of consumption and leisure
with an equivalence scale and a consumption weight, so the marginal utility of
consumption — and therefore the Euler inversion `c = (u')^{-1}(m)` — depends on
the ride-along leisure term, not on consumption alone. A plain-CRRA inversion
`c = (discount_factor * m) ** (-1 / crra)` is the wrong action whenever the
consumption weight is not one or the scale is not one.

This miniature carries the consumption-saving margin on the Euler/liquid state
`liquid` while a continuous co-state `wage` rides along and sets the leisure term
`leisure = leisure_base + leisure_slope * wage`. The period utility is the CES
bundle `scale * (c**w * leisure**(1-w))**(1-crra) / (1-crra)`, whose marginal in
consumption is `scale * w * leisure**((1-w)(1-crra)) * c**(w(1-crra)-1)` — leisure-
and scale-dependent, so the inversion must read them. A subsidy breakpoint on
derived `gross_income = liquid + wage` engages the ride-along schedule path:

- `breakpoint_kind="continuous_kink"` keeps cash-on-hand continuous (a slope kink),
  routing the solve through the continuous multi-interval savings step;
- `breakpoint_kind="jump"` makes cash-on-hand jump at the cliff, routing through the
  unified jump-and-kink savings step.

The `GridSearch` variant productmaps over `(liquid, wage, consumption)` with the
same CES utility and is the dense agreement oracle. The `"nbegm"` variant inverts
the CES marginal utility numerically from the period utility — the NBEGM ride-along
solver carries no `inverse_marginal_utility` function (its `marginal_continuation`
would otherwise become a required parameter), matching the M1 regime.
"""

import jax.numpy as jnp

import lcm
from lcm import LinSpacedGrid, Model
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from tests.test_models.nbegm_common import (
    crra_utility,
    feasible,
    make_alive_dead_model,
    next_liquid,
    next_liquid_from_savings,
    resolve_solver,
    savings,
)


def leisure(
    wage: ContinuousState, leisure_base: float, leisure_slope: float
) -> FloatND:
    """Leisure term riding along with the wage co-state (strictly positive)."""
    return leisure_base + leisure_slope * wage


def utility(
    consumption: ContinuousAction,
    leisure: FloatND,
    consumption_weight: float,
    crra: float,
    util_scale: float,
) -> FloatND:
    """CES utility over a Cobb-Douglas consumption--leisure bundle."""
    bundle = consumption**consumption_weight * leisure ** (1.0 - consumption_weight)
    return util_scale * crra_utility(bundle, crra)


def gross_income(liquid: ContinuousState, wage: ContinuousState) -> FloatND:
    """Pre-tax income: liquid wealth plus the wage co-state (monotone in liquid)."""
    return liquid + wage


@lcm.piecewise_affine(
    "subsidy",
    variable="gross_income",
    breakpoints=(lcm.affine_breakpoint("fpl_cliff", kind="jump"),),
)
def subsidy_jump(
    gross_income: FloatND, subsidy_low: float, subsidy_high: float, fpl_cliff: float
) -> FloatND:
    """Lump-sum subsidy: the higher amount below the income cliff, lower above."""
    return jnp.where(gross_income < fpl_cliff, subsidy_high, subsidy_low)


@lcm.piecewise_affine(
    "subsidy",
    variable="gross_income",
    breakpoints=(lcm.affine_breakpoint("fpl_cliff", kind="continuous_kink"),),
)
def subsidy_kink(
    gross_income: FloatND,
    subsidy_high: float,
    subsidy_slope: float,
    fpl_cliff: float,
) -> FloatND:
    """Subsidy continuous at the cliff: flat below, linearly phased out above."""
    return jnp.where(
        gross_income < fpl_cliff,
        subsidy_high,
        subsidy_high - subsidy_slope * (gross_income - fpl_cliff),
    )


def resources(gross_income: FloatND, subsidy: FloatND) -> FloatND:
    """Cash-on-hand: pre-tax income plus the cliff-contingent subsidy."""
    return gross_income + subsidy


def next_wage(wage: ContinuousState, wage_persistence: float) -> ContinuousState:
    """Wage co-state law: a deterministic decay that never reads the savings slot."""
    return wage_persistence * wage


def build_model(
    *,
    variant: str = "brute",
    breakpoint_kind: str = "jump",
    n_periods: int = 4,
    n_liquid: int = 120,
    n_wage: int = 6,
    n_consumption: int = 160,
    liquid_max: float = 30.0,
    wage_max: float = 6.0,
    n_savings: int = 200,
    savings_max: float = 28.0,
) -> Model:
    """Create the (alive, dead) CES-utility ride-along toy.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch`; `"nbegm"` by the
            `NBEGM` schedule solver, which inverts the CES marginal utility
            numerically from the period utility.
        breakpoint_kind: `"jump"` (cash-on-hand jumps at the cliff) or
            `"continuous_kink"` (cash-on-hand continuous, slope kink).
        n_periods: Number of lifecycle periods (the last is terminal).
        n_liquid: Liquid (Euler) state grid size.
        n_wage: Wage co-state grid size.
        n_consumption: Consumption-action grid size (brute only).
        liquid_max: Upper bound of the liquid grid.
        wage_max: Upper bound of the wage grid.
        n_savings: Post-decision savings grid size (NBEGM only).
        savings_max: Upper bound of the savings grid (NBEGM only).

    Returns:
        The assembled `Model`.

    """
    wage_grid = LinSpacedGrid(start=0.5, stop=wage_max, n_points=n_wage)

    schedule = subsidy_kink if breakpoint_kind == "continuous_kink" else subsidy_jump
    alive_functions = {
        "utility": utility,
        "leisure": leisure,
        "gross_income": gross_income,
        "subsidy": schedule,
        "resources": resources,
    }
    alive_solver = resolve_solver(
        variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
        continuous_state="liquid",
        post_decision_function="savings",
    )
    if variant == "nbegm":
        alive_functions = {**alive_functions, "savings": savings}
        liquid_law = next_liquid_from_savings
        constraints = {}
    else:
        liquid_law = next_liquid
        constraints = {"feasible": feasible}

    return make_alive_dead_model(
        n_periods=n_periods,
        n_liquid=n_liquid,
        liquid_max=liquid_max,
        n_consumption=n_consumption,
        alive_functions=alive_functions,
        liquid_law=liquid_law,
        alive_solver=alive_solver,
        constraints=constraints,
        extra_states={"wage": wage_grid},
        extra_state_transitions={"wage": {"alive": next_wage}},
    )


def build_params(
    *,
    breakpoint_kind: str = "jump",
    discount_factor: float = 0.95,
    crra: float = 2.0,
    consumption_weight: float = 0.55,
    util_scale: float = 1.3,
    leisure_base: float = 1.0,
    leisure_slope: float = 0.4,
    return_liquid: float = 0.03,
    income: float = 1.0,
    wage_persistence: float = 0.9,
    subsidy_low: float = 0.0,
    subsidy_high: float = 3.0,
    subsidy_slope: float = 0.4,
    fpl_cliff: float = 15.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the CES-utility ride-along toy.

    The CES consumption weight and equivalence scale are deliberately away from one,
    so the consumption recovered by the correct CES inversion differs materially from
    the plain-CRRA inversion the solver must not use.
    """
    alive_budget = {"return_liquid": return_liquid, "income": income}
    if breakpoint_kind == "continuous_kink":
        subsidy_params = {
            "subsidy_high": subsidy_high,
            "subsidy_slope": subsidy_slope,
            "fpl_cliff": fpl_cliff,
        }
    else:
        subsidy_params = {
            "subsidy_low": subsidy_low,
            "subsidy_high": subsidy_high,
            "fpl_cliff": fpl_cliff,
        }
    alive_params: dict = {
        "utility": {
            "crra": crra,
            "consumption_weight": consumption_weight,
            "util_scale": util_scale,
        },
        "leisure": {
            "leisure_base": leisure_base,
            "leisure_slope": leisure_slope,
        },
        "H": {"discount_factor": discount_factor},
        "subsidy": subsidy_params,
        "alive": {
            "next_liquid": alive_budget,
            "next_wage": {"wage_persistence": wage_persistence},
            "next_regime": {"final_age_alive": final_age_alive},
        },
        "dead": {
            "next_liquid": alive_budget,
            "next_regime": {"final_age_alive": final_age_alive},
        },
    }
    return {
        "alive": alive_params,
        "dead": {"utility": {"crra": crra}},
    }
