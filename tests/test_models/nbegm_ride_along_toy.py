"""One-asset NBEGM toy with a deterministic ride-along co-state.

Extends the continuous tax-bracket toy with a second, discrete state `kind` that
shifts base income but does not itself enter the consumption--saving Euler axis:
it rides along. `kind` evolves deterministically (it stays), so the continuation
value `V'[liquid, kind]` is read within the same `kind` slice — no expectation
over a stochastic transition is involved. The budget therefore varies per
ride-along cell (`base_income` depends on `kind`), and the NBEGM step must solve
the 1-D liquid problem once per `kind` slice (batched), each against that slice's
budget and continuation. The brute variant (`GridSearch`) productmaps over
`(liquid, kind, consumption)` and is the dense agreement oracle.
"""

from collections.abc import Mapping

import jax.numpy as jnp

import lcm
from lcm import DiscreteGrid, LinSpacedGrid, Model, categorical
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.test_models.nbegm_common import (
    crra_utility,
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


def crra_of_kind(kind: DiscreteState, crra_by_kind: FloatND) -> FloatND:
    """Per-kind CRRA coefficient read off the ride-along `kind` code."""
    return crra_by_kind[kind]


def utility_per_kind(consumption: ContinuousAction, crra_of_kind: FloatND) -> FloatND:
    """CRRA consumption utility whose curvature differs across `kind` slices.

    Stands in for a model whose utility parameters are indexed by a ride-along
    preference type through an intermediate DAG node, so the Euler inversion
    must use each cell's own curvature.
    """
    return crra_utility(consumption, crra_of_kind)


def discount_factor(kind: DiscreteState, discount_factor_by_kind: FloatND) -> FloatND:
    """Per-kind subjective discount factor read off the ride-along `kind` code.

    Stands in for a model whose discount factor is a DAG function of a ride-along
    state (e.g. a preference type) rather than the default flat `H__discount_factor`
    parameter, so the budget's Euler weight differs across ride-along slices.
    """
    return discount_factor_by_kind[kind]


@lcm.piecewise_affine(
    "tax",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint("tax_exemption", kind="continuous_kink"),),
)
def tax(liquid: ContinuousState, tax_rate: float, tax_exemption: float) -> FloatND:
    """Continuous tax: zero below the exemption, `tax_rate` on the excess above."""
    return tax_rate * jnp.maximum(liquid - tax_exemption, 0.0)


def resources(
    liquid: ContinuousState,
    kind: DiscreteState,
    tax: FloatND,
    base_income: FloatND,
) -> FloatND:
    """Cash-on-hand: liquid plus the kind's base income, net of the tax."""
    return liquid + base_income[kind] - tax


def build_model(
    *,
    variant: str = "brute",
    per_kind_discount: bool = False,
    per_kind_crra: bool = False,
    n_periods: int = 4,
    n_liquid: int = 120,
    n_consumption: int = 150,
    liquid_max: float = 30.0,
    n_savings: int = 150,
    savings_max: float = 28.0,
    nbegm_overrides: Mapping[str, object] | None = None,
    distributed_kind: bool = False,
) -> Model:
    """Create the (alive, dead) tax toy with a deterministic ride-along `kind`.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch`; `"nbegm"` by
            the `NBEGM` schedule solver, which must batch the 1-D liquid step over
            the `kind` co-state.
        n_periods: Number of lifecycle periods (the last is terminal).
        n_liquid: Liquid-state grid size.
        n_consumption: Consumption-action grid size (brute only).
        liquid_max: Upper bound of the liquid grid.
        n_savings: Post-decision savings grid size (NBEGM only).
        savings_max: Upper bound of the savings grid (NBEGM only).

    Returns:
        The assembled `Model`.

    """
    alive_functions = {"utility": utility, "tax": tax, "resources": resources}
    if per_kind_crra:
        # Route the utility curvature through a DAG node indexed by the
        # ride-along `kind`, exercising per-cell utility parameters.
        alive_functions = {
            **alive_functions,
            "utility": utility_per_kind,
            "crra_of_kind": crra_of_kind,
        }
    if per_kind_discount:
        # Drive the discount factor off the ride-along `kind` code so the Euler
        # weight differs across slices, exercising DAG-resolved discounting.
        alive_functions = {**alive_functions, "discount_factor": discount_factor}
    alive_solver = resolve_solver(
        variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
        post_decision_function="savings",
        **(dict(nbegm_overrides) if nbegm_overrides else {}),
    )
    if variant == "nbegm":
        # The case-piece path inverts the Euler equation against the
        # post-decision savings node, so the liquid law is in savings form and
        # the budget exposes the `savings` slot the continuation reader consumes.
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
        extra_states=({} if distributed_kind else {"kind": DiscreteGrid(ConsumerKind)}),
        extra_state_transitions={"kind": {"alive": lcm.fixed_transition("kind")}},
        model_states=(
            {"kind": DiscreteGrid(ConsumerKind, distributed=True)}
            if distributed_kind
            else None
        ),
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    per_kind_discount: bool = False,
    per_kind_crra: bool = False,
    crra_lo: float = 3.0,
    crra_hi: float = 1.5,
    discount_factor_lo: float = 0.90,
    discount_factor_hi: float = 0.97,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    income: float = 1.0,
    base_income_lo: float = 1.0,
    base_income_hi: float = 4.0,
    tax_rate: float = 0.3,
    tax_exemption: float = 12.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the ride-along tax toy.

    `base_income` is a length-2 array indexed by the `kind` code (`lo`, `hi`), so
    the budget differs across the ride-along slices. With `per_kind_discount`, the
    discount factor is supplied as a length-2 `discount_factor_by_kind` array under
    the `discount_factor` DAG function instead of the flat `H__discount_factor`.
    """
    base_income = jnp.array([base_income_lo, base_income_hi])
    alive_budget = {"return_liquid": return_liquid, "income": income}
    discount_slot = (
        {
            "discount_factor": {
                "discount_factor_by_kind": jnp.array(
                    [discount_factor_lo, discount_factor_hi]
                )
            }
        }
        if per_kind_discount
        else {"H": {"discount_factor": discount_factor}}
    )
    utility_slot = (
        {"crra_of_kind": {"crra_by_kind": jnp.array([crra_lo, crra_hi])}}
        if per_kind_crra
        else {"utility": {"crra": crra}}
    )
    return {
        "alive": {
            **utility_slot,
            **discount_slot,
            "tax": {"tax_rate": tax_rate, "tax_exemption": tax_exemption},
            "resources": {"base_income": base_income},
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
