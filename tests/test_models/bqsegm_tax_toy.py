"""One-asset consumption--saving toy with a continuous tax-bracket kink.

A minimal model exercising the continuous-budget BQSEGM path: a single liquid
asset, a single consumption action, and a piecewise-affine tax that bends
cash-on-hand at an exemption threshold. Below the exemption the tax is zero, above
it the marginal tax `tax_rate` applies, so `coh(liquid)` is continuous with a
single downward kink — no jump. The brute variant evaluates the schedule on a
dense grid and is the agreement oracle; the BQSEGM variant reads the declared
schedule and solves each affine segment by EGM.
"""

import jax.numpy as jnp
from dags import rename_arguments

import lcm
from lcm import LinSpacedGrid, Model
from lcm.typing import ContinuousState, FloatND
from tests.test_models.bqsegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid,
    resolve_solver,
    utility,
)


@lcm.piecewise_affine(
    "tax",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint("tax_exemption", kind="continuous_kink"),),
)
def tax(liquid: ContinuousState, tax_rate: float, tax_exemption: float) -> FloatND:
    """Continuous tax: zero below the exemption, `tax_rate` on the excess above."""
    return tax_rate * jnp.maximum(liquid - tax_exemption, 0.0)


def coh(liquid: ContinuousState, tax: FloatND, base_income: float) -> FloatND:
    """Cash-on-hand: liquid wealth plus base income, net of the tax."""
    return liquid + base_income - tax


def build_model(
    *,
    variant: str = "brute",
    n_periods: int = 4,
    n_liquid: int = 120,
    n_consumption: int = 150,
    liquid_max: float = 30.0,
    n_savings: int = 150,
    savings_max: float = 28.0,
    budget_name: str = "coh",
) -> Model:
    """Create the two-regime (alive, dead) tax-bracket one-asset toy.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch` (the dense-grid
            oracle); `"bqsegm"` drives it by the `BQSEGM` schedule solver.
        n_periods: Number of lifecycle periods (the last is terminal).
        n_liquid: Liquid-state grid size.
        n_consumption: Consumption-action grid size (brute only).
        liquid_max: Upper bound of the liquid grid.
        n_savings: Post-decision savings grid size (BQSEGM only).
        savings_max: Upper bound of the savings grid (BQSEGM only).
        budget_name: DAG node name carrying cash-on-hand. The default `"coh"`
            matches the solver convention; any other name exercises the solver's
            `budget_target` selection, with the budget's consumers rewired to read
            it.

    Returns:
        The assembled `Model`.

    """
    next_liquid_func, feasible_func = next_liquid, feasible
    if budget_name != "coh":
        next_liquid_func = rename_arguments(next_liquid, mapper={"coh": budget_name})
        feasible_func = rename_arguments(feasible, mapper={"coh": budget_name})
    return make_alive_dead_model(
        n_periods=n_periods,
        n_liquid=n_liquid,
        liquid_max=liquid_max,
        n_consumption=n_consumption,
        alive_functions={
            "utility": utility,
            "tax": tax,
            budget_name: coh,
        },
        liquid_law=next_liquid_func,
        alive_solver=resolve_solver(
            variant,
            savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
            budget_target=budget_name,
        ),
        constraints={"feasible": feasible_func},
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    income: float = 1.0,
    base_income: float = 2.0,
    tax_rate: float = 0.3,
    tax_exemption: float = 12.0,
    final_age_alive: float = 3.0,
    budget_name: str = "coh",
) -> dict:
    """Get parameters for the tax-bracket one-asset toy."""
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "tax": {"tax_rate": tax_rate, "tax_exemption": tax_exemption},
            budget_name: {"base_income": base_income},
            "alive": {
                "next_liquid": alive_budget,
                "next_regime": {"final_age_alive": final_age_alive},
            },
            "dead": {
                "next_liquid": alive_budget,
                "next_regime": {"final_age_alive": final_age_alive},
            },
        },
        "dead": {"utility": {"crra": crra}},
    }
