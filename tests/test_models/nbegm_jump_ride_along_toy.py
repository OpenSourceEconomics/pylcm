"""One-asset NBEGM toy with a subsidy cliff on a derived income var + ride-along.

Combines the two M1 ingredients in miniature: a budget *jump* (a subsidy that
drops to zero once income crosses a cliff) declared on the derived monotone
quantity `gross_income`, together with a ride-along co-state `kind` that offsets
income. The cliff `gross_income == fpl_cliff` therefore maps to the asset
preimage `liquid = fpl_cliff - base_income[kind]`, a *different* liquid point in
each `kind` slice — so the cash-on-hand discontinuity sits at different assets
across cells. The NBEGM ride-along solver must recover the per-cell asset
preimage of the cliff and solve the jumped budget against the savings-space
continuation. The brute variant (`GridSearch`) productmaps over
`(liquid, kind, consumption)` and is the dense agreement oracle.
"""

import jax.numpy as jnp
from dags import with_signature

import lcm
from lcm import DiscreteGrid, LinSpacedGrid, Model, categorical
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
    "subsidy",
    variable="gross_income",
    breakpoints=(lcm.affine_breakpoint("fpl_cliff", kind="jump"),),
)
def subsidy(
    gross_income: FloatND, subsidy_low: float, subsidy_high: float, fpl_cliff: float
) -> FloatND:
    """Lump-sum subsidy: the higher amount below the income cliff, lower above."""
    return jnp.where(gross_income < fpl_cliff, subsidy_high, subsidy_low)


def resources(gross_income: FloatND, subsidy: FloatND) -> FloatND:
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
    extra_actions: dict | None = None,
    jump_read: str = "one_sided",
) -> Model:
    """Create the (alive, dead) toy whose subsidy cliff lives on derived income.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch`; `"nbegm"` by
            the `NBEGM` schedule solver, which must map the derived-income cliff
            to its per-`kind` asset preimage and solve the jumped budget.
        n_periods: Number of lifecycle periods (the last is terminal).
        n_liquid: Liquid-state grid size.
        n_consumption: Consumption-action grid size (brute only).
        liquid_max: Upper bound of the liquid grid.
        n_savings: Post-decision savings grid size (NBEGM only).
        savings_max: Upper bound of the savings grid (NBEGM only).
        jump_read: `NBEGM.jump_read` — how the parent's continuation read
            treats the child value's cliffs (NBEGM only).

    Returns:
        The assembled `Model`.

    """
    alive_functions = {
        "utility": utility,
        "gross_income": gross_income,
        "subsidy": subsidy,
        "resources": resources,
    }
    alive_solver = resolve_solver(
        variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
        post_decision_function="savings",
        jump_read=jump_read,
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
        constraints={
            **constraints,
            # Each extra action gets a discrete-only consumer so the model's
            # unused-variable gate passes and the action reaches the solver.
            **{
                f"{name}_nonnegative": with_signature(args=[name])(
                    lambda *values: values[0] >= 0
                )
                for name in (extra_actions or {})
            },
        },
        extra_actions=extra_actions,
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
    subsidy_low: float = 0.0,
    subsidy_high: float = 3.0,
    fpl_cliff: float = 15.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the derived-income subsidy-cliff toy.

    `base_income` is a length-2 array indexed by the `kind` code (`lo`, `hi`). The
    cliff at `gross_income == fpl_cliff` maps to the asset preimage `liquid =
    fpl_cliff - base_income[kind]`, distinct across the two slices (`14.0` for
    `lo`, `11.0` for `hi` at the defaults), both grid-interior. Cash-on-hand drops
    by `subsidy_high - subsidy_low` as liquid crosses each slice's preimage.
    """
    base_income = jnp.array([base_income_lo, base_income_hi])
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
