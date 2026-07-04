"""One-asset toy composing a binary discrete choice with a cliff schedule.

Each period the agent chooses whether to buy private insurance (a premium that
lowers cash-on-hand) against a budget that also carries a declared jump: a lump
tax kicks in above an exemption on the liquid state. BQSEGM must solve the
continuous consumption/savings subproblem inside each insurance branch *and*
respect the cliff within each branch, then take the discrete choice by the
upper envelope over the branch values. The brute variant maximises over the
discrete choice and consumption on a dense grid and is the agreement oracle.

This is the F-E minimal composition: the discrete upper envelope over a
non-smooth (cliffed) budget, on a single liquid axis with no ride-along.
"""

import jax.numpy as jnp

import lcm
from lcm import DiscreteGrid, LinSpacedGrid, Model, categorical
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.test_models.bqsegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid,
    resolve_solver,
    utility,
)


@categorical(ordered=False)
class BuyPrivate:
    no: ScalarInt
    yes: ScalarInt


@lcm.piecewise_affine(
    "tax",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint("tax_exemption", kind="jump"),),
)
def tax_cliff(
    liquid: ContinuousState, tax_exemption: float, tax_lump: float
) -> FloatND:
    """Cliff tax: zero below the exemption, a flat lump above (a pure jump).

    A level drop in cash-on-hand at the exemption with slope 1 on both sides —
    the shape of an income-tested subsidy cliff (a lump loss of eligibility).
    """
    return jnp.where(liquid >= tax_exemption, tax_lump, 0.0)


@lcm.piecewise_affine(
    "tax",
    variable="liquid",
    breakpoints=(
        lcm.affine_breakpoint("tax_exemption", kind="jump"),
        lcm.affine_breakpoint("tax_bracket", kind="continuous_kink"),
    ),
)
def tax_mixed(
    liquid: ContinuousState,
    tax_exemption: float,
    tax_lump: float,
    tax_bracket: float,
    tax_rate: float,
) -> FloatND:
    """Cliff lump at the exemption plus a rising bracket above a higher threshold.

    Cash-on-hand drops by `tax_lump` at the exemption (slope unchanged) and then
    slopes down by `tax_rate` above `tax_bracket`. The bracket case has non-unit
    liquid slope, so the save-to-cliff candidate landing at the exemption must
    scale its published marginal by that slope.
    """
    lump = jnp.where(liquid >= tax_exemption, tax_lump, 0.0)
    bracket = tax_rate * jnp.maximum(liquid - tax_bracket, 0.0)
    return lump + bracket


def coh(
    liquid: ContinuousState,
    tax: FloatND,
    buy_private: DiscreteAction,
    premium: float,
) -> FloatND:
    """Cash-on-hand: liquid plus base income, net of tax and any premium."""
    return liquid + 6.0 - tax - premium * buy_private


def build_model(
    *,
    variant: str = "brute",
    n_periods: int = 4,
    n_liquid: int = 120,
    n_consumption: int = 150,
    liquid_max: float = 30.0,
    n_savings: int = 150,
    savings_max: float = 28.0,
    mixed_schedule: bool = False,
) -> Model:
    """Create the two-regime (alive, dead) discrete-choice-plus-cliff toy.

    With `mixed_schedule`, the budget carries a jump *and* a kink (a rising
    bracket) so each discrete branch routes to the mixed-kind unified step, whose
    save-to-cliff candidate must scale its marginal by the bracket case's slope.
    """
    tax_func = tax_mixed if mixed_schedule else tax_cliff
    alive_functions = {"utility": utility, "tax": tax_func, "coh": coh}
    alive_solver = resolve_solver(
        variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
        continuous_state="liquid",
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
        extra_actions={"buy_private": DiscreteGrid(BuyPrivate)},
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    income: float = 1.0,
    premium: float = 1.5,
    tax_exemption: float = 12.0,
    tax_lump: float = 1.0,
    final_age_alive: float = 3.0,
    mixed_schedule: bool = False,
    tax_bracket: float = 18.0,
    tax_rate: float = 0.2,
) -> dict:
    """Get parameters for the discrete-choice-plus-cliff one-asset toy."""
    alive_budget = {"return_liquid": return_liquid, "income": income}
    tax_params = {"tax_exemption": tax_exemption, "tax_lump": tax_lump}
    if mixed_schedule:
        tax_params = {**tax_params, "tax_bracket": tax_bracket, "tax_rate": tax_rate}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "coh": {"premium": premium},
            "tax": tax_params,
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
