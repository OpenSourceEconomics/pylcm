"""Ride-along toy whose budget is shifted by several discrete choices at once.

A stochastic income node rides along a kink tax schedule, and each period the
agent picks a combination of discrete choices: whether to buy private insurance,
whether to claim a benefit, and (in the three-action variant) how much to work.
Every choice shifts cash-on-hand and period utility, so NBEGM must solve the
continuous consumption/savings subproblem per ride cell and per element of the
*product* of the discrete grids, then take the joint choice by the upper
envelope over those branches. The brute variant maximises over the same product
and consumption on a dense grid and is the agreement oracle.

The two-action variant has 2x2 = 4 branches; the three-action variant has
2x2x5 = 20, the shape a lifecycle model with an insurance, a claiming, and a
labor-supply margin presents.
"""

import jax.numpy as jnp

import lcm
from lcm import DiscreteGrid, Model, NormalIIDProcess, categorical
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.test_models.nbegm_common import (
    crra_utility,
    feasible,
    make_alive_dead_model,
    resolve_solver,
    savings,
)

N_INCOME_NODES = 3
INCOME_SCALE = 0.5
COVERAGE_UTILITY = 0.30
CLAIM_UTILITY = 0.15
LEISURE_UTILITY = 0.20
HOURS_PER_LEVEL = 0.5


@categorical(ordered=False)
class BuyPrivate:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=False)
class ClaimBenefit:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=True)
class LaborSupply:
    none: ScalarInt
    quarter: ScalarInt
    half: ScalarInt
    three_quarters: ScalarInt
    full: ScalarInt


@lcm.piecewise_affine(
    output="tax",
    variable="liquid",
    breakpoints=(
        lcm.affine_breakpoint(threshold="tax_exemption", kind="continuous_kink"),
    ),
)
def tax(*, liquid: ContinuousState, tax_rate: float, tax_exemption: float) -> FloatND:
    """Continuous tax: zero below the exemption, `tax_rate` on the excess above."""
    return tax_rate * jnp.maximum(liquid - tax_exemption, 0.0)


def resources_two_actions(
    *,
    liquid: ContinuousState,
    tax: FloatND,
    buy_private: DiscreteAction,
    claim_benefit: DiscreteAction,
    base_income: float,
    premium: float,
    benefit: float,
) -> FloatND:
    """Cash-on-hand net of tax and the premium, plus the claimed benefit."""
    return liquid + base_income - tax - premium * buy_private + benefit * claim_benefit


def resources_three_actions(
    *,
    liquid: ContinuousState,
    tax: FloatND,
    buy_private: DiscreteAction,
    claim_benefit: DiscreteAction,
    labor_supply: DiscreteAction,
    base_income: float,
    premium: float,
    benefit: float,
    wage: float,
) -> FloatND:
    """Cash-on-hand net of tax and premium, plus the benefit and labor earnings."""
    return (
        liquid
        + base_income
        - tax
        - premium * buy_private
        + benefit * claim_benefit
        + wage * HOURS_PER_LEVEL * labor_supply
    )


def utility_two_actions(
    *,
    consumption: ContinuousAction,
    crra: float,
    buy_private: DiscreteAction,
    claim_benefit: DiscreteAction,
) -> FloatND:
    """CRRA consumption utility plus the value of coverage and of claiming."""
    return (
        crra_utility(consumption=consumption, crra=crra)
        + COVERAGE_UTILITY * buy_private
        + CLAIM_UTILITY * claim_benefit
    )


def utility_three_actions(
    *,
    consumption: ContinuousAction,
    crra: float,
    buy_private: DiscreteAction,
    claim_benefit: DiscreteAction,
    labor_supply: DiscreteAction,
) -> FloatND:
    """CRRA consumption, plus coverage and claiming value, less leisure lost."""
    return (
        crra_utility(consumption=consumption, crra=crra)
        + COVERAGE_UTILITY * buy_private
        + CLAIM_UTILITY * claim_benefit
        - LEISURE_UTILITY * labor_supply
    )


def next_liquid_from_savings(
    *,
    savings: FloatND,
    income: ContinuousState,
    return_liquid: float,
) -> ContinuousState:
    """Liquid law in post-decision form: saved cash earns the return, plus income."""
    return (1.0 + return_liquid) * savings + INCOME_SCALE * jnp.exp(income)


def build_model(
    *,
    variant: str = "brute",
    n_actions: int = 2,
    n_periods: int = 3,
    n_liquid: int = 40,
    n_consumption: int = 120,
    liquid_max: float = 30.0,
    n_savings: int = 60,
    savings_max: float = 28.0,
    jump_read: str = "bridged",
    envelope_arithmetic: str = "certified",
    branch_batch_size: int = 0,
) -> Model:
    """Create the (alive, dead) ride-along toy with several discrete choices.

    With `n_actions=2` the agent chooses insurance and claiming (4 branches);
    with `n_actions=3` a five-level labor supply joins them (20 branches). Every
    choice enters the current budget and period utility only, so all branches
    share one next-period continuation and the discrete choice is taken by the
    upper envelope over the product of the grids.
    """
    if n_actions not in (2, 3):
        msg = f"n_actions must be 2 or 3, got {n_actions}."
        raise ValueError(msg)
    extra_actions = {
        "buy_private": DiscreteGrid(category_class=BuyPrivate),
        "claim_benefit": DiscreteGrid(category_class=ClaimBenefit),
    }
    if n_actions == 3:
        extra_actions["labor_supply"] = DiscreteGrid(category_class=LaborSupply)
    alive_functions = {
        "utility": utility_two_actions if n_actions == 2 else utility_three_actions,
        "tax": tax,
        "resources": (
            resources_two_actions if n_actions == 2 else resources_three_actions
        ),
        "savings": savings,
    }
    alive_solver = resolve_solver(
        variant=variant,
        savings_grid=lcm.LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
        jump_read=jump_read,
        envelope_arithmetic=envelope_arithmetic,
        branch_batch_size=branch_batch_size,
    )
    return make_alive_dead_model(
        n_periods=n_periods,
        n_liquid=n_liquid,
        liquid_max=liquid_max,
        n_consumption=n_consumption,
        alive_functions=alive_functions,
        liquid_law=next_liquid_from_savings,
        alive_solver=alive_solver,
        constraints={} if variant == "nbegm" else {"feasible": feasible},
        extra_states={
            "income": NormalIIDProcess(n_points=N_INCOME_NODES, gauss_hermite=True)
        },
        extra_actions=extra_actions,
    )


def build_params(
    *,
    n_actions: int = 2,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    base_income: float = 3.0,
    premium: float = 1.5,
    benefit: float = 2.0,
    wage: float = 1.2,
    tax_rate: float = 0.2,
    tax_exemption: float = 12.0,
    final_age_alive: float = 2.0,
) -> dict:
    """Get parameters for the multi-discrete ride-along toy."""
    budget_params = {
        "base_income": base_income,
        "premium": premium,
        "benefit": benefit,
    }
    if n_actions == 3:
        budget_params["wage"] = wage
    alive_budget = {"return_liquid": return_liquid}
    return {
        "alive": {
            "utility": {"crra": crra},
            "koopmans_aggregator": {"discount_factor": discount_factor},
            "resources": budget_params,
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
