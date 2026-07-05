"""One-asset NBEGM toy whose cliff threshold is a per-ride-along-cell table.

Isolates state-indexed *thresholds* from state-indexed *offsets*: the derived
income variable `gross_income = liquid + base_income` carries the SAME offset in
both `kind` slices, but the subsidy cliff sits at `gross_income == fpl_cliff[kind]`
— a threshold read from a length-2 table indexed by the ride-along `kind` state.
The cliff therefore maps to the asset preimage `liquid = fpl_cliff[kind] -
base_income`, a different liquid point per slice driven entirely by the threshold.
The NBEGM ride-along solver must resolve each cell's threshold from the table
before mapping it to the per-cell preimage. The brute variant (`GridSearch`)
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


def gross_income(liquid: ContinuousState, base_income: float) -> FloatND:
    """Pre-tax income: liquid wealth plus a kind-invariant base income (monotone)."""
    return liquid + base_income


@lcm.piecewise_affine(
    "subsidy",
    variable="gross_income",
    breakpoints=(lcm.affine_breakpoint("fpl_cliff", kind="jump", indexed_by="kind"),),
)
def subsidy(
    gross_income: FloatND,
    kind: DiscreteState,
    subsidy_low: float,
    subsidy_high: float,
    fpl_cliff: FloatND,
) -> FloatND:
    """Lump-sum subsidy: higher below the kind's income cliff, lower above."""
    return jnp.where(gross_income < fpl_cliff[kind], subsidy_high, subsidy_low)


def coh(gross_income: FloatND, subsidy: FloatND) -> FloatND:
    """Cash-on-hand: pre-tax income plus the cliff-contingent subsidy."""
    return gross_income + subsidy


def build_model(
    *,
    variant: str = "brute",
    n_periods: int = 4,
    n_liquid: int = 160,
    n_consumption: int = 200,
    liquid_max: float = 30.0,
    n_savings: int = 200,
    savings_max: float = 28.0,
) -> Model:
    """Create the (alive, dead) toy whose subsidy cliff threshold is kind-indexed.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch`; `"nbegm"` by
            the `NBEGM` schedule solver, which must read each cell's cliff
            threshold from the `fpl_cliff` table and map it to its asset preimage.
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
        "subsidy": subsidy,
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
    base_income: float = 5.0,
    subsidy_low: float = 0.0,
    subsidy_high: float = 3.0,
    fpl_cliff_lo: float = 14.0,
    fpl_cliff_hi: float = 11.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the kind-indexed-threshold subsidy-cliff toy.

    `fpl_cliff` is a length-2 array indexed by the `kind` code (`lo`, `hi`). With
    the kind-invariant `base_income`, the cliff `gross_income == fpl_cliff[kind]`
    maps to the asset preimage `liquid = fpl_cliff[kind] - base_income` — `9.0` for
    `lo`, `6.0` for `hi` at the defaults, both grid-interior — so the cash-on-hand
    discontinuity sits at different assets across slices, driven only by the
    per-cell threshold.
    """
    fpl_cliff = jnp.array([fpl_cliff_lo, fpl_cliff_hi])
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "gross_income": {"base_income": base_income},
            "subsidy": {
                "subsidy_low": subsidy_low,
                "subsidy_high": subsidy_high,
                "fpl_cliff": fpl_cliff,
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
