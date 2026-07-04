"""Ride-along toy with a binary discrete choice over a kinked budget.

A stochastic income node rides along a kink tax schedule (the continuation
integrates over the child income nodes), and each period the agent also chooses
whether to buy private insurance — a premium that lowers cash-on-hand. BQSEGM
must solve the continuous consumption/savings subproblem per ride cell *and*
per discrete branch, then take the discrete choice by the upper envelope over
the branch values. The brute variant maximises over the discrete choice and
consumption on a dense grid and is the agreement oracle.

The schedule here is a pure kink (no jump), so the continuation carries no
in-period value jump: this isolates the discrete-envelope-over-ride-cells
composition from the published-jump topology.
"""

import jax.numpy as jnp

import lcm
from _lcm.grids.base import Grid
from lcm import DiscreteGrid, LinSpacedGrid, Model, NormalIIDProcess, categorical
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.test_models.bqsegm_common import (
    feasible,
    make_alive_dead_model,
    resolve_solver,
    savings,
    utility,
)

N_INCOME_NODES = 5
INCOME_SCALE = 0.5


@categorical(ordered=False)
class BuyPrivate:
    no: ScalarInt
    yes: ScalarInt


@lcm.piecewise_affine(
    "tax",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint("tax_exemption", kind="continuous_kink"),),
)
def tax(liquid: ContinuousState, tax_rate: float, tax_exemption: float) -> FloatND:
    """Continuous tax: zero below the exemption, `tax_rate` on the excess above."""
    return tax_rate * jnp.maximum(liquid - tax_exemption, 0.0)


def coh(
    liquid: ContinuousState,
    tax: FloatND,
    buy_private: DiscreteAction,
    base_income: float,
    premium: float,
) -> FloatND:
    """Cash-on-hand: liquid plus base income, net of tax and any premium."""
    return liquid + base_income - tax - premium * buy_private


def next_liquid(
    coh: FloatND,
    consumption: ContinuousAction,
    income: ContinuousState,
    return_liquid: float,
) -> ContinuousState:
    """Liquid law: saved cash earns the return, plus the realized income draw."""
    return (1.0 + return_liquid) * (coh - consumption) + INCOME_SCALE * jnp.exp(income)


def next_liquid_from_savings(
    savings: FloatND,
    income: ContinuousState,
    return_liquid: float,
) -> ContinuousState:
    """Liquid law in post-decision form: saved cash earns the return, plus income."""
    return (1.0 + return_liquid) * savings + INCOME_SCALE * jnp.exp(income)


def next_streak(
    streak: ContinuousState, buy_private: DiscreteAction
) -> ContinuousState:
    """Coverage streak: grows while insured, resets otherwise (reads the action)."""
    return jnp.where(buy_private == BuyPrivate.yes, streak + 1.0, 0.0)


def build_model(
    *,
    variant: str = "brute",
    n_periods: int = 4,
    n_liquid: int = 120,
    n_consumption: int = 150,
    liquid_max: float = 30.0,
    n_savings: int = 150,
    savings_max: float = 28.0,
    action_in_costate: bool = False,
) -> Model:
    """Create the (alive, dead) ride-along toy with a discrete insurance choice.

    With `action_in_costate`, a `streak` co-state carries a law of motion that
    reads `buy_private` — so the discrete action shifts the continuation, not just
    the current budget, which the shared-continuation envelope cannot represent.
    """
    income_grid = NormalIIDProcess(n_points=N_INCOME_NODES, gauss_hermite=True)
    alive_functions = {"utility": utility, "tax": tax, "coh": coh}
    extra_states: dict[str, Grid] = {"income": income_grid}
    extra_state_transitions: dict[str, object] = {}
    if action_in_costate:
        extra_states["streak"] = LinSpacedGrid(start=0.0, stop=4.0, n_points=5)
        extra_state_transitions["streak"] = {
            "alive": next_streak,
            "dead": next_streak,
        }
    alive_solver = resolve_solver(
        variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
        continuous_state="liquid",
        post_decision_function="savings",
    )
    if variant == "bqsegm":
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
        extra_states=extra_states,
        extra_state_transitions=extra_state_transitions,
        extra_actions={"buy_private": DiscreteGrid(BuyPrivate)},
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    base_income: float = 3.0,
    premium: float = 1.5,
    tax_rate: float = 0.2,
    tax_exemption: float = 12.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the ride-along discrete-choice toy."""
    alive_budget = {"return_liquid": return_liquid}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "coh": {"base_income": base_income, "premium": premium},
            "income": {"mu": 0.0, "sigma": 1.0},
            "tax": {"tax_rate": tax_rate, "tax_exemption": tax_exemption},
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
