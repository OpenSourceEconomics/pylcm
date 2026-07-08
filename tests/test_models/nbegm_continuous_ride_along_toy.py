"""One-asset NBEGM toy with a subsidy cliff on derived income + a continuous co-state.

Mirrors the M1 ACA structure in miniature: the Euler/liquid state `liquid` carries
the consumption-saving margin, while a *continuous* co-state `wage` rides along. The
subsidy cliff is declared on the derived monotone quantity `gross_income = liquid +
wage`, so the cliff `gross_income == fpl_cliff` maps to the asset preimage `liquid =
fpl_cliff - wage`, a *different* liquid point in every `wage` cell. The NBEGM
ride-along solver names `liquid` as its `continuous_state` (the Euler axis) and treats
`wage` as a ride-along axis the continuation reader integrates, recovering the per-cell
asset preimage of the cliff and solving the jumped budget against the savings-space
continuation. The brute variant (`GridSearch`) productmaps over
`(liquid, wage, consumption)` and is the dense agreement oracle.
"""

import jax.numpy as jnp

import lcm
from lcm import LinSpacedGrid, Model
from lcm.typing import ContinuousState, FloatND
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid,
    next_liquid_from_savings,
    resolve_solver,
    savings,
    utility,
)


def gross_income(liquid: ContinuousState, wage: ContinuousState) -> FloatND:
    """Pre-tax income: liquid wealth plus the wage co-state (monotone in liquid)."""
    return liquid + wage


@lcm.piecewise_affine(
    "subsidy",
    variable="gross_income",
    breakpoints=(lcm.affine_breakpoint("fpl_cliff", kind="jump"),),
)
def subsidy(
    gross_income: FloatND, subsidy_low: float, subsidy_high: float, fpl_cliff: float
) -> FloatND:
    """Lump-sum subsidy: the higher amount below the income cliff, lower above."""
    return jnp.where(gross_income < fpl_cliff, subsidy_high, subsidy_low)


def resources(gross_income: FloatND, subsidy: FloatND) -> FloatND:
    """Cash-on-hand: pre-tax income plus the cliff-contingent subsidy."""
    return gross_income + subsidy


def next_wage(wage: ContinuousState, wage_persistence: float) -> ContinuousState:
    """Wage co-state law: a deterministic decay that never reads the savings slot."""
    return wage_persistence * wage


def build_model(
    *,
    variant: str = "brute",
    n_periods: int = 4,
    n_liquid: int = 120,
    n_wage: int = 6,
    n_consumption: int = 160,
    liquid_max: float = 30.0,
    wage_max: float = 6.0,
    n_savings: int = 200,
    savings_max: float = 28.0,
) -> Model:
    """Create the (alive, dead) toy whose subsidy cliff lives on derived income.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch`; `"nbegm"` by
            the `NBEGM` schedule solver, which names `liquid` as its Euler axis and
            rides along the continuous `wage` co-state.
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

    alive_functions = {
        "utility": utility,
        "gross_income": gross_income,
        "subsidy": subsidy,
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
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    income: float = 1.0,
    wage_persistence: float = 0.9,
    subsidy_low: float = 0.0,
    subsidy_high: float = 3.0,
    fpl_cliff: float = 15.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the derived-income subsidy-cliff toy.

    The cliff at `gross_income == fpl_cliff` maps to the asset preimage `liquid =
    fpl_cliff - wage`, distinct in every `wage` slice and grid-interior for the
    default wage grid. Cash-on-hand drops by `subsidy_high - subsidy_low` as liquid
    crosses each slice's preimage.
    """
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "subsidy": {
                "subsidy_low": subsidy_low,
                "subsidy_high": subsidy_high,
                "fpl_cliff": fpl_cliff,
            },
            "alive": {
                "next_liquid": alive_budget,
                "next_wage": {"wage_persistence": wage_persistence},
                "next_regime": {"final_age_alive": final_age_alive},
            },
            "dead": {
                "next_liquid": alive_budget,
                "next_regime": {"final_age_alive": final_age_alive},
            },
        },
        "dead": {"utility": {"crra": crra}},
    }
