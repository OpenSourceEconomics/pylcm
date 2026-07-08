"""One-asset consumption--saving toy with a Medicaid case boundary on assets.

A minimal model exercising NBEGM: a single liquid asset, a single consumption
action, and a binary Medicaid eligibility test on current assets. Eligibility
(`liquid < medicaid_asset_limit`) grants a larger lump-sum subsidy into
cash-on-hand than the private side, so the budget — and hence the value function —
jumps down as assets cross the limit upward. Within each case the budget is
smooth, so NBEGM solves each case by EGM and stitches them at the boundary; the
brute variant evaluates the `jnp.where` combination on a dense grid and is the
agreement oracle.

`build_model(variant="brute")` uses `GridSearch`; `build_model(variant="nbegm")`
uses the `NBEGM` solver over the decorated case pieces.
"""

import functools

import jax.numpy as jnp

import lcm
from lcm import LinSpacedGrid, Model
from lcm.typing import BoolND, ContinuousState, FloatND

# RegimeId, bequest, and the survival probabilities are re-exported: the
# medicaid agreement test assembles its own validation models from these
# toy internals.
from tests.test_models.nbegm_common import (
    RegimeId,  # noqa: F401
    bequest,  # noqa: F401
    feasible,
    make_alive_dead_model,
    next_liquid,
    prob_die,  # noqa: F401
    prob_stay_alive,  # noqa: F401
    resolve_solver,
    utility,
)


@lcm.case_boundary(
    lcm.boundary("liquid", "medicaid_asset_limit", equality="otherwise", kind="jump")
)
def medicaid_eligible(liquid: ContinuousState, medicaid_asset_limit: float) -> BoolND:
    """Medicaid asset test: eligible while liquid wealth is below the limit."""
    return liquid < medicaid_asset_limit


@lcm.piece("subsidy", when=medicaid_eligible)
def subsidy_medicaid(subsidy_high: float) -> FloatND:
    """Subsidy into cash-on-hand for the Medicaid-eligible (low-asset) case."""
    return jnp.asarray(subsidy_high)


@lcm.piece("subsidy", otherwise=medicaid_eligible)
def subsidy_private(subsidy_low: float) -> FloatND:
    """Subsidy into cash-on-hand for the private (high-asset) case."""
    return jnp.asarray(subsidy_low)


def subsidy(
    subsidy_medicaid: FloatND, subsidy_private: FloatND, medicaid_eligible: BoolND
) -> FloatND:
    """Brute-force combination of the two subsidy pieces (NBEGM ignores this)."""
    return jnp.where(medicaid_eligible, subsidy_medicaid, subsidy_private)


def resources(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Cash-on-hand: liquid wealth plus the Medicaid-contingent subsidy."""
    return liquid + subsidy


def build_model(
    *,
    variant: str = "brute",
    n_periods: int = 4,
    n_liquid: int = 80,
    n_consumption: int = 120,
    liquid_max: float = 20.0,
    n_savings: int = 100,
    savings_max: float = 20.0,
) -> Model:
    """Create the two-regime (alive, dead) Medicaid one-asset toy.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch` (the dense-grid
            oracle); `"nbegm"` drives it by the `NBEGM` case-piece solver.
        n_periods: Number of lifecycle periods (the last is terminal).
        n_liquid: Liquid-state grid size.
        n_consumption: Consumption-action grid size (brute only).
        liquid_max: Upper bound of the liquid grid.
        n_savings: Post-decision savings grid size (NBEGM only).
        savings_max: Upper bound of the savings grid (NBEGM only).

    Returns:
        The assembled `Model`.

    """
    return make_alive_dead_model(
        n_periods=n_periods,
        n_liquid=n_liquid,
        liquid_max=liquid_max,
        n_consumption=n_consumption,
        alive_functions={
            "utility": utility,
            "medicaid_eligible": medicaid_eligible,
            "subsidy_medicaid": subsidy_medicaid,
            "subsidy_private": subsidy_private,
            "subsidy": subsidy,
            "resources": resources,
        },
        liquid_law=next_liquid,
        alive_solver=resolve_solver(
            variant,
            savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
        ),
        constraints={"feasible": feasible},
    )


@functools.cache
def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    income: float = 1.0,
    subsidy_high: float = 3.0,
    subsidy_low: float = 0.5,
    medicaid_asset_limit: float = 8.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the Medicaid one-asset toy.

    The Medicaid-eligible subsidy (`subsidy_high`) exceeds the private subsidy
    (`subsidy_low`), so cash-on-hand — and the value function — jumps down as
    liquid wealth crosses `medicaid_asset_limit` upward.
    """
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "medicaid_eligible": {"medicaid_asset_limit": medicaid_asset_limit},
            "subsidy_medicaid": {"subsidy_high": subsidy_high},
            "subsidy_private": {"subsidy_low": subsidy_low},
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
