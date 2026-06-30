"""One-asset BQSEGM toy whose budget kink lives on a derived monotone income var.

Mirrors the ride-along tax toy, but the tax bracket kinks on `gross_income`, a
derived quantity that is monotone in the liquid state and offset by the
ride-along `kind`. The kink's location in liquid space therefore *moves* per
ride-along cell: at `gross_income = liquid + base_income[kind]`, the threshold
`gross_income == tax_kink` maps to the asset preimage `liquid = tax_kink -
base_income[kind]`, a different liquid point in each `kind` slice. The BQSEGM
schedule solver must recover that per-cell asset preimage before partitioning
the liquid axis. The brute variant (`GridSearch`) productmaps over
`(liquid, kind, consumption)` and is the dense agreement oracle.
"""

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
    n_periods: int = 4,
    n_liquid: int = 120,
    n_consumption: int = 150,
    liquid_max: float = 30.0,
    n_savings: int = 150,
    savings_max: float = 28.0,
) -> Model:
    """Create the (alive, dead) toy whose tax kink lives on derived `gross_income`.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch`; `"bqsegm"` by
            the `BQSEGM` schedule solver, which must map the derived-income kink
            to its per-`kind` asset preimage.
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

    alive_functions = {
        "utility": utility,
        "gross_income": gross_income,
        "tax": tax,
        "coh": coh,
    }
    if variant == "brute":
        alive_solver = GridSearch()
        liquid_law = next_liquid
        constraints = {"feasible": feasible}
    elif variant == "bqsegm":
        from lcm.solvers import BQSEGM  # noqa: PLC0415

        alive_functions = {**alive_functions, "savings": savings}
        liquid_law = next_liquid_from_savings
        constraints = {}
        alive_solver = BQSEGM(
            savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
            post_decision_function="savings",
        )
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
