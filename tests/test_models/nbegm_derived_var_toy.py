"""One-asset NBEGM toy whose budget kink lives on a derived monotone income var.

Mirrors the ride-along tax toy, but the tax bracket kinks on `gross_income`, a
derived quantity that is monotone in the liquid state and offset by the
ride-along `kind`. The kink's location in liquid space therefore *moves* per
ride-along cell: at `gross_income = liquid + base_income[kind]`, the threshold
`gross_income == tax_kink` maps to the asset preimage `liquid = tax_kink -
base_income[kind]`, a different liquid point in each `kind` slice. The NBEGM
schedule solver must recover that per-cell asset preimage before partitioning
the liquid axis. The brute variant (`GridSearch`) productmaps over
`(liquid, kind, consumption)` and is the dense agreement oracle.
"""

import jax.numpy as jnp

import lcm
from lcm import (
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
)
from lcm.typing import (
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid,
    next_liquid_from_savings,
    resolve_solver,
    savings,
    utility,
)


@categorical(ordered=False)
class ConsumerKind:
    lo: ScalarInt
    hi: ScalarInt


def gross_income(
    liquid: ContinuousState,
    kind: DiscreteState,
    base_income: FloatND,
) -> FloatND:
    """Pre-tax income: liquid wealth plus the kind's base income (monotone)."""
    return liquid + base_income[kind]


@lcm.piecewise_affine(
    "tax",
    variable="gross_income",
    breakpoints=(lcm.affine_breakpoint("tax_kink", kind="continuous_kink"),),
)
def tax(gross_income: FloatND, tax_rate: float, tax_kink: float) -> FloatND:
    """Continuous tax: zero below the kink, `tax_rate` on income above it."""
    return tax_rate * jnp.maximum(gross_income - tax_kink, 0.0)


def coh(gross_income: FloatND, tax: FloatND) -> FloatND:
    """Cash-on-hand: pre-tax income net of the tax."""
    return gross_income - tax


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
    """Create the (alive, dead) toy whose tax kink lives on derived `gross_income`.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch`; `"nbegm"` by
            the `NBEGM` schedule solver, which must map the derived-income kink
            to its per-`kind` asset preimage.
        n_periods: Number of lifecycle periods (the last is terminal).
        n_liquid: Liquid-state grid size.
        n_consumption: Consumption-action grid size (brute only).
        liquid_max: Upper bound of the liquid grid.
        n_savings: Post-decision savings grid size (NBEGM only).
        savings_max: Upper bound of the savings grid (NBEGM only).

    Returns:
        The assembled `Model`.

    """
    alive_functions = {
        "utility": utility,
        "gross_income": gross_income,
        "tax": tax,
        "coh": coh,
    }
    alive_solver = resolve_solver(
        variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
        post_decision_function="savings",
    )
    if variant == "nbegm":
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
        extra_states={"kind": DiscreteGrid(ConsumerKind)},
        extra_state_transitions={"kind": {"alive": lcm.fixed_transition("kind")}},
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    income: float = 1.0,
    base_income_lo: float = 1.0,
    base_income_hi: float = 4.0,
    tax_rate: float = 0.3,
    tax_kink: float = 15.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the derived-income tax toy.

    `base_income` is a length-2 array indexed by the `kind` code (`lo`, `hi`). The
    tax kink at `gross_income == tax_kink` therefore maps to the asset preimage
    `liquid = tax_kink - base_income[kind]`, distinct across the two slices
    (`14.0` for `lo`, `11.0` for `hi` at the defaults), both grid-interior.
    """
    base_income = jnp.array([base_income_lo, base_income_hi])
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "gross_income": {"base_income": base_income},
            "tax": {"tax_rate": tax_rate, "tax_kink": tax_kink},
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
