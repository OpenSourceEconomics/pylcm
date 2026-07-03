"""One-asset BQSEGM toy mixing a jump and a kink on two distinct income vars.

The budget combines a continuous tax bracketing on `income_a` (a `continuous_kink`)
with a lump-sum subsidy that drops at a cliff on `income_b` (a `jump`). The two
income concepts are offset differently by the ride-along `kind`, so the kink and
the jump map to per-`kind` asset preimages that *reorder* between slices: the jump
sits below the kink in one slice and above it in the other. BQSEGM must merge the
mixed-kind breakpoints across the two variables into one per-cell sorted partition
whose jump position is recovered per cell. The brute variant (`GridSearch`)
productmaps over `(liquid, kind, consumption)` and is the dense agreement oracle.
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
from tests.test_models.bqsegm_common import (
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


def income_a(liquid: ContinuousState, kind: DiscreteState, base_a: FloatND) -> FloatND:
    """Income concept the tax bracket reads: liquid plus the kind's `a` offset."""
    return liquid + base_a[kind]


def income_b(liquid: ContinuousState, kind: DiscreteState, base_b: FloatND) -> FloatND:
    """Income concept the subsidy cliff reads: liquid plus the kind's `b` offset."""
    return liquid + base_b[kind]


@lcm.piecewise_affine(
    "tax_a",
    variable="income_a",
    breakpoints=(lcm.affine_breakpoint("kink_a", kind="continuous_kink"),),
)
def tax_a(income_a: FloatND, rate_a: float, kink_a: float) -> FloatND:
    """Continuous tax on `income_a`: zero below the kink, `rate_a` on the excess."""
    return rate_a * jnp.maximum(income_a - kink_a, 0.0)


@lcm.piecewise_affine(
    "subsidy_b",
    variable="income_b",
    breakpoints=(lcm.affine_breakpoint("cliff_b", kind="jump"),),
)
def subsidy_b(
    income_b: FloatND, subsidy_low: float, subsidy_high: float, cliff_b: float
) -> FloatND:
    """Lump-sum subsidy on `income_b`: the higher amount below the cliff."""
    return jnp.where(income_b < cliff_b, subsidy_high, subsidy_low)


def coh(income_a: FloatND, tax_a: FloatND, subsidy_b: FloatND) -> FloatND:
    """Cash-on-hand: income net of the tax plus the cliff-contingent subsidy."""
    return income_a - tax_a + subsidy_b


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
    """Create the (alive, dead) toy mixing a jump and a kink on two income vars.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch`; `"bqsegm"` by
            the `BQSEGM` schedule solver, which must merge the kink and the jump —
            declared on two variables — into one per-`kind` asset partition whose
            jump position is recovered per cell.
        n_periods: Number of lifecycle periods (the last is terminal).
        n_liquid: Liquid-state grid size.
        n_consumption: Consumption-action grid size (brute only).
        liquid_max: Upper bound of the liquid grid.
        n_savings: Post-decision savings grid size (BQSEGM only).
        savings_max: Upper bound of the savings grid (BQSEGM only).

    Returns:
        The assembled `Model`.

    """
    alive_functions = {
        "utility": utility,
        "income_a": income_a,
        "income_b": income_b,
        "tax_a": tax_a,
        "subsidy_b": subsidy_b,
        "coh": coh,
    }
    alive_solver = resolve_solver(
        variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
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
        extra_states={"kind": DiscreteGrid(ConsumerKind)},
        extra_state_transitions={"kind": {"alive": lcm.fixed_transition("kind")}},
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    income: float = 1.0,
    base_a_lo: float = 1.0,
    base_a_hi: float = 4.0,
    base_b_lo: float = 2.0,
    base_b_hi: float = 2.0,
    rate_a: float = 0.2,
    kink_a: float = 15.0,
    subsidy_low: float = 0.0,
    subsidy_high: float = 2.0,
    cliff_b: float = 14.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the mixed jump-and-kink two-variable budget toy.

    The kink maps to the asset preimage `kink_a - base_a[kind]` and the jump to
    `cliff_b - base_b[kind]`. At the defaults the kink sits at `14.0` (`lo`) /
    `11.0` (`hi`) and the jump at `12.0` (both slices), so the jump is *below* the
    kink for `lo` (`12 < 14`) but *above* it for `hi` (`11 < 12`): the per-cell
    sorted order swaps which position is the jump.
    """
    base_a = jnp.array([base_a_lo, base_a_hi])
    base_b = jnp.array([base_b_lo, base_b_hi])
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "income_a": {"base_a": base_a},
            "income_b": {"base_b": base_b},
            "tax_a": {"rate_a": rate_a, "kink_a": kink_a},
            "subsidy_b": {
                "subsidy_low": subsidy_low,
                "subsidy_high": subsidy_high,
                "cliff_b": cliff_b,
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
