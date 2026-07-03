"""One-asset toy mixing a subsidy cliff (jump) and a tax bracket (kink).

A single `piecewise_affine` net-transfer schedule declares two thresholds of
different kinds: a jump where the subsidy steps down at an asset cliff, and a
continuous kink where a tax phases in above an exemption. Cash-on-hand therefore
jumps at the cliff and bends at the exemption. The BQSEGM schedule path routes this
mixed schedule to the unified step, which must reproduce the dense `GridSearch`
value through both, every age.
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
    "net_transfer",
    variable="liquid",
    breakpoints=(
        lcm.affine_breakpoint("cliff", kind="jump"),
        lcm.affine_breakpoint("exemption", kind="continuous_kink"),
    ),
)
def net_transfer(
    liquid: ContinuousState,
    subsidy_low: float,
    subsidy_high: float,
    cliff: float,
    tax_rate: float,
    exemption: float,
) -> FloatND:
    """Subsidy that steps down at the cliff, net of a tax above the exemption."""
    subsidy = jnp.where(liquid < cliff, subsidy_high, subsidy_low)
    tax = tax_rate * jnp.maximum(liquid - exemption, 0.0)
    return subsidy - tax


def coh(liquid: ContinuousState, net_transfer: FloatND) -> FloatND:
    """Cash-on-hand: liquid wealth plus the net transfer."""
    return liquid + net_transfer


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
    """Create the two-regime (alive, dead) mixed jump-and-kink one-asset toy."""
    alive_functions = {"utility": utility, "net_transfer": net_transfer, "coh": coh}
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
    cliff: float = 6.0,
    tax_rate: float = 0.3,
    exemption: float = 16.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the mixed jump-and-kink one-asset toy.

    The cliff (jump) precedes the exemption (kink), so the schedule's declared
    breakpoint order is ascending in value.
    """
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "net_transfer": {
                "subsidy_low": subsidy_low,
                "subsidy_high": subsidy_high,
                "cliff": cliff,
                "tax_rate": tax_rate,
                "exemption": exemption,
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
