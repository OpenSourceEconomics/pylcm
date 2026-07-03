"""One-asset toy with a subsidy cliff declared as a jump `piecewise_affine`.

The same Medicaid-style asset cliff as `bqsegm_medicaid_toy`, but expressed through
a jump-kind `lcm.piecewise_affine` schedule rather than `case_boundary`/`piece`.
Cash-on-hand jumps down as liquid crosses the cliff. The BQSEGM schedule path
recognises the single jump and routes it to the binary case solver, so it must
reproduce the dense `GridSearch` value across the cliff at every age.
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
    "subsidy",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint("cliff", kind="jump"),),
)
def subsidy(
    liquid: ContinuousState, subsidy_low: float, subsidy_high: float, cliff: float
) -> FloatND:
    """Lump-sum subsidy: the higher amount below the cliff, the lower above it."""
    return jnp.where(liquid < cliff, subsidy_high, subsidy_low)


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
    """Create the two-regime (alive, dead) jump-schedule one-asset toy."""
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
    subsidy_low: float = 0.5,
    cliff: float = 8.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the jump-schedule one-asset toy."""
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "subsidy": {
                "subsidy_low": subsidy_low,
                "subsidy_high": subsidy_high,
                "cliff": cliff,
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
