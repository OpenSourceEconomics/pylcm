"""One-asset BQSEGM toy with a deterministic ride-along co-state.

Extends the continuous tax-bracket toy with a second, discrete state `kind` that
shifts base income but does not itself enter the consumption--saving Euler axis:
it rides along. `kind` evolves deterministically (it stays), so the continuation
value `V'[liquid, kind]` is read within the same `kind` slice — no expectation
over a stochastic transition is involved. The budget therefore varies per
ride-along cell (`base_income` depends on `kind`), and the BQSEGM step must solve
the 1-D liquid problem once per `kind` slice (batched), each against that slice's
budget and continuation. The brute variant (`GridSearch`) productmaps over
`(liquid, kind, consumption)` and is the dense agreement oracle.
"""

import dataclasses
from collections.abc import Mapping

import jax.numpy as jnp

import lcm
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.regime import Regime
from lcm.solvers import GridSearch
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class ConsumerKind:
    lo: ScalarInt
    hi: ScalarInt


def _crra(consumption: FloatND, crra: float) -> FloatND:
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption ** (1.0 - crra) / (1.0 - crra),
    )


def utility(consumption: ContinuousAction, crra: float) -> FloatND:
    """CRRA consumption utility."""
    return _crra(consumption, crra)


def bequest(liquid: ContinuousState, crra: float) -> FloatND:
    """Terminal value: consume remaining liquid wealth."""
    return _crra(liquid, crra)


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


def coh(
    liquid: ContinuousState,
    kind: DiscreteState,
    tax: FloatND,
    base_income: FloatND,
) -> FloatND:
    """Cash-on-hand: liquid plus the kind's base income, net of the tax."""
    return liquid + base_income[kind] - tax


def next_liquid(
    coh: FloatND,
    consumption: ContinuousAction,
    return_liquid: float,
    income: float,
) -> ContinuousState:
    """Liquid law of motion: saved cash earns the liquid return, plus income."""
    return (1.0 + return_liquid) * (coh - consumption) + income


def savings(coh: FloatND, consumption: ContinuousAction) -> FloatND:
    """Post-decision savings: the cash-on-hand not consumed."""
    return coh - consumption


def next_liquid_from_savings(
    savings: FloatND,
    return_liquid: float,
    income: float,
) -> ContinuousState:
    """Liquid law in post-decision form: saved cash earns the return, plus income."""
    return (1.0 + return_liquid) * savings + income


def feasible(coh: FloatND, consumption: ContinuousAction) -> BoolND:
    """Borrowing constraint: consumption cannot exceed cash-on-hand."""
    return consumption <= coh


def prob_stay_alive(age: int, final_age_alive: float) -> FloatND:
    """Deterministic (0/1) probability of staying alive next period."""
    return jnp.where(age + 1 < final_age_alive, 1.0, 0.0)


def prob_die(age: int, final_age_alive: float) -> FloatND:
    """Deterministic (0/1) probability of dying next period."""
    return jnp.where(age + 1 >= final_age_alive, 1.0, 0.0)


def build_model(
    *,
    variant: str = "brute",
    per_kind_discount: bool = False,
    n_periods: int = 4,
    n_liquid: int = 120,
    n_consumption: int = 150,
    liquid_max: float = 30.0,
    n_savings: int = 150,
    savings_max: float = 28.0,
    bqsegm_overrides: Mapping[str, object] | None = None,
) -> Model:
    """Create the (alive, dead) tax toy with a deterministic ride-along `kind`.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch`; `"bqsegm"` by
            the `BQSEGM` schedule solver, which must batch the 1-D liquid step over
            the `kind` co-state.
        n_periods: Number of lifecycle periods (the last is terminal).
        n_liquid: Liquid-state grid size.
        n_consumption: Consumption-action grid size (brute only).
        liquid_max: Upper bound of the liquid grid.
        n_savings: Post-decision savings grid size (BQSEGM only).
        savings_max: Upper bound of the savings grid (BQSEGM only).

    Returns:
        The assembled `Model`.

    """
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    final_age = ages.exact_values[-1]
    liquid_grid = LinSpacedGrid(start=0.1, stop=liquid_max, n_points=n_liquid)

    alive_functions = {"utility": utility, "tax": tax, "coh": coh}
    if per_kind_discount:
        # Drive the discount factor off the ride-along `kind` code so the Euler
        # weight differs across slices, exercising DAG-resolved discounting.
        alive_functions = {**alive_functions, "discount_factor": discount_factor}
    if variant == "brute":
        alive_solver = GridSearch()
        liquid_law = next_liquid
        constraints = {"feasible": feasible}
    elif variant == "bqsegm":
        from lcm.solvers import BQSEGM  # noqa: PLC0415

        # The case-piece path inverts the Euler equation against the
        # post-decision savings node, so the liquid law is in savings form and
        # the budget exposes the `savings` slot the continuation reader consumes.
        alive_functions = {**alive_functions, "savings": savings}
        liquid_law = next_liquid_from_savings
        constraints = {}
        alive_solver = BQSEGM(
            savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
            post_decision_function="savings",
        )
        if bqsegm_overrides:
            alive_solver = dataclasses.replace(alive_solver, **bqsegm_overrides)
    else:
        msg = f"unknown variant {variant!r}; use 'brute' or 'bqsegm'."
        raise ValueError(msg)

    alive = Regime(
        actions={
            "consumption": LinSpacedGrid(
                start=0.1, stop=liquid_max, n_points=n_consumption
            )
        },
        states={"liquid": liquid_grid, "kind": DiscreteGrid(ConsumerKind)},
        state_transitions={
            "liquid": {"alive": liquid_law, "dead": liquid_law},
            "kind": {"alive": lcm.fixed_transition("kind")},
        },
        constraints=constraints,
        transition={
            "alive": MarkovTransition(prob_stay_alive),
            "dead": MarkovTransition(prob_die),
        },
        functions=alive_functions,
        active=lambda age, fa=final_age: age < fa,
        solver=alive_solver,
    )
    dead = Regime(
        transition=None,
        states={"liquid": liquid_grid},
        functions={"utility": bequest},
        active=lambda age, fa=final_age: age >= fa,
        solver=GridSearch(),
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    per_kind_discount: bool = False,
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
    return {
        "alive": {
            "utility": {"crra": crra},
            **discount_slot,
            "tax": {"tax_rate": tax_rate, "tax_exemption": tax_exemption},
            "coh": {"base_income": base_income},
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
