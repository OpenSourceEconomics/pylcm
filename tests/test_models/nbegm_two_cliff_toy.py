"""One-asset toy with two subsidy cliffs declared as a jump `piecewise_affine`.

Cash-on-hand jumps down at each of two asset cliffs (the subsidy steps down
twice), so the budget and the value function jump at both. The NBEGM schedule
path recognises the all-jump schedule and routes it to the recurring N-cliff step,
which must reproduce the dense `GridSearch` value across both cliffs at every age.
"""

import jax.numpy as jnp

import lcm
from lcm import LinSpacedGrid, Model
from lcm.typing import ContinuousState, FloatND
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid,
    resolve_solver,
    utility,
)


@lcm.piecewise_affine(
    "subsidy",
    variable="liquid",
    breakpoints=(
        lcm.affine_breakpoint("cliff_low", kind="jump"),
        lcm.affine_breakpoint("cliff_high", kind="jump"),
    ),
)
def subsidy(
    liquid: ContinuousState,
    subsidy_low: float,
    subsidy_mid: float,
    subsidy_high: float,
    cliff_low: float,
    cliff_high: float,
) -> FloatND:
    """Step subsidy: high below the low cliff, mid between, low above the high cliff."""
    return jnp.where(
        liquid < cliff_low,
        subsidy_high,
        jnp.where(liquid < cliff_high, subsidy_mid, subsidy_low),
    )


def coh(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Cash-on-hand: liquid wealth plus the cliff-contingent subsidy."""
    return liquid + subsidy


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
    """Create the two-regime (alive, dead) two-cliff one-asset toy."""
    alive_functions = {"utility": utility, "subsidy": subsidy, "coh": coh}
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
    subsidy_high: float = 3.0,
    subsidy_mid: float = 1.5,
    subsidy_low: float = 0.5,
    cliff_low: float = 6.0,
    cliff_high: float = 14.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the two-cliff one-asset toy."""
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "subsidy": {
                "subsidy_low": subsidy_low,
                "subsidy_mid": subsidy_mid,
                "subsidy_high": subsidy_high,
                "cliff_low": cliff_low,
                "cliff_high": cliff_high,
            },
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
