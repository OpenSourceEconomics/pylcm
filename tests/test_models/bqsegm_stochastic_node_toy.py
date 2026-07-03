"""One-asset BQSEGM toy with a stochastic income node riding along a tax schedule.

Extends the tax-bracket schedule toy with a genuine *stochastic* ride-along: an IID
income process `income` whose node enters next period's liquid wealth additively. The
income node is not on the consumption--saving Euler axis (it rides along), but unlike a
deterministic co-state its child node is *distributed*: the continuation must weight the
per-node child reads by the process's intrinsic transition probabilities. That weighted
node expectation is the quantity `BQSEGM.stochastic_node_batch_size` splays into blocks,
so this toy is the minimal lock that the splay path integrates the same value either
way.
The brute variant (`GridSearch`) productmaps over `(liquid, income, consumption)` and
averages the action-aggregated next-period V over the income nodes — the dense oracle.
"""

import jax.numpy as jnp

import lcm
from lcm import LinSpacedGrid, Model, NormalIIDProcess
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from tests.test_models.bqsegm_common import (
    feasible,
    make_alive_dead_model,
    resolve_solver,
    savings,
    utility,
)

# Discretization nodes of the IID income process. Odd, as Gauss-Hermite requires.
N_INCOME_NODES = 5

# Scale on `exp(income)` entering next liquid. Kept small so that, even at the top of
# the savings grid and the highest income node, continuation wealth stays inside the
# liquid grid: an out-of-grid continuation edge-clamps, and BQSEGM's savings-space
# inversion would then extrapolate past its last endogenous point, desyncing it from the
# brute oracle at the top liquid nodes.
INCOME_SCALE = 0.5


@lcm.piecewise_affine(
    "tax",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint("tax_exemption", kind="continuous_kink"),),
)
def tax(liquid: ContinuousState, tax_rate: float, tax_exemption: float) -> FloatND:
    """Continuous tax: zero below the exemption, `tax_rate` on the excess above."""
    return tax_rate * jnp.maximum(liquid - tax_exemption, 0.0)


def coh(liquid: ContinuousState, tax: FloatND, base_income: float) -> FloatND:
    """Cash-on-hand: liquid plus a flat base income, net of the tax."""
    return liquid + base_income - tax


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


def build_model(
    *,
    variant: str = "brute",
    stochastic_node_batch_size: int = 0,
    n_periods: int = 4,
    n_liquid: int = 120,
    n_consumption: int = 150,
    liquid_max: float = 30.0,
    n_savings: int = 150,
    savings_max: float = 28.0,
) -> Model:
    """Create the (alive, dead) tax toy with a stochastic ride-along income node.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch`; `"bqsegm"` by the
            `BQSEGM` schedule solver, which batches the 1-D liquid step over the income
            node and takes the continuation expectation over the child nodes.
        stochastic_node_batch_size: Block size for splaying the continuation's
            income-node expectation (BQSEGM only); `0` reads the whole mesh in one pass.
        n_periods: Number of lifecycle periods (the last is terminal).
        n_liquid: Liquid-state grid size.
        n_consumption: Consumption-action grid size (brute only).
        liquid_max: Upper bound of the liquid grid.
        n_savings: Post-decision savings grid size (BQSEGM only).
        savings_max: Upper bound of the savings grid (BQSEGM only).

    Returns:
        The assembled `Model`.

    """
    income_grid = NormalIIDProcess(n_points=N_INCOME_NODES, gauss_hermite=True)

    alive_functions = {"utility": utility, "tax": tax, "coh": coh}
    alive_solver = resolve_solver(
        variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
        continuous_state="liquid",
        post_decision_function="savings",
        stochastic_node_batch_size=stochastic_node_batch_size,
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
        extra_states={"income": income_grid},
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    base_income: float = 2.0,
    income_mu: float = 0.0,
    income_sigma: float = 0.2,
    tax_rate: float = 0.3,
    tax_exemption: float = 12.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the stochastic-node tax toy.

    `income` carries the IID process distribution params (`mu`, `sigma`); the intrinsic
    Gauss-Hermite weights flow through the same params channel for both solvers.
    """
    alive_budget = {"return_liquid": return_liquid}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "tax": {"tax_rate": tax_rate, "tax_exemption": tax_exemption},
            "coh": {"base_income": base_income},
            "income": {"mu": income_mu, "sigma": income_sigma},
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
