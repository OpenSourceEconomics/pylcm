"""One-asset toy with a cash-on-hand floor declared as a hard-constraint schedule.

A means-tested transfer lifts effective wealth to a floor: `coh = max(liquid,
floor_asset) + base_income`. The schedule declares `floor_asset` (the liquid level
below which the transfer binds) as a hard-constraint breakpoint, so cash-on-hand is
flat below it and the value is constant where the floor binds. The BQSEGM schedule
path passes the slope-0 interval to the multi-interval step's floor handling and
must reproduce the dense `GridSearch` value, every age — including the recurring
flat continuation, where the Euler inversion is degenerate.
"""

import jax.numpy as jnp

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
    "coh_floor",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint("floor_asset", kind="hard_constraint"),),
)
def coh_floor(liquid: ContinuousState, floor_asset: float) -> FloatND:
    """Floor top-up: lifts effective wealth to `floor_asset` where liquid is below."""
    return jnp.maximum(liquid, floor_asset) - liquid


def coh(liquid: ContinuousState, coh_floor: FloatND, base_income: float) -> FloatND:
    """Cash-on-hand: `max(liquid, floor_asset) + base_income`."""
    return liquid + coh_floor + base_income


def build_model(
    *,
    variant: str = "brute",
    n_periods: int = 4,
    n_liquid: int = 120,
    n_consumption: int = 150,
    liquid_max: float = 30.0,
    n_savings: int = 150,
    savings_max: float = 28.0,
) -> Model:
    """Create the two-regime (alive, dead) floor one-asset toy."""
    alive_functions = {"utility": utility, "coh_floor": coh_floor, "coh": coh}
    alive_solver = resolve_solver(
        variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
    )

    return make_alive_dead_model(
        n_periods=n_periods,
        n_liquid=n_liquid,
        liquid_max=liquid_max,
        n_consumption=n_consumption,
        alive_functions=alive_functions,
        liquid_law=next_liquid,
        alive_solver=alive_solver,
        constraints={"feasible": feasible},
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    income: float = 1.0,
    base_income: float = 2.0,
    floor_asset: float = 3.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the floor one-asset toy.

    `floor_asset` is the liquid level below which the means-tested transfer binds —
    the hard-constraint breakpoint where cash-on-hand goes flat.
    """
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "coh_floor": {"floor_asset": floor_asset},
            "coh": {"base_income": base_income},
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
