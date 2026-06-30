"""BQSEGM ride-along toy whose next-asset law jumps at a declared liquid cliff.

The aca M1 budget's next-period assets carry current-asset boundaries:
`next_assets = savings + corrections(assets)`, where the corrections (a Medicaid
asset-test transfer, a pension imputation adjustment) are piecewise-constant in
the current liquid state through declared cliffs. The continuation therefore
cannot read `next_assets` as a function of post-decision savings alone — within
each declared interval the correction is a constant, so the continuation must be
evaluated per interval with that constant bound.

This toy isolates that structure: a single declared liquid cliff
(`medicaid_limit`) adds a constant transfer to next-period liquid below it and
nothing above it, so `next_liquid` jumps at the cliff. The BQSEGM solve must
reproduce the dense `GridSearch` value across the asset interior in both `kind`
slices — the value oracle for the per-interval continuation.
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


def coh(liquid: ContinuousState, kind: DiscreteState, base_income: FloatND) -> FloatND:
    """Cash-on-hand: liquid plus the kind's base income (no in-period cliff)."""
    return liquid + base_income[kind]


@lcm.piecewise_affine(
    "medicaid_transfer",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint("medicaid_limit", kind="jump"),),
)
def medicaid_transfer(
    liquid: ContinuousState, medicaid_limit: float, transfer_amount: float
) -> FloatND:
    """A constant transfer to next-period liquid while assets are below the limit.

    Reads the *current* liquid state, so it is a current-asset boundary: the
    next-asset law inherits its jump at `medicaid_limit`.
    """
    return jnp.where(liquid < medicaid_limit, transfer_amount, 0.0)


def next_liquid(
    coh: FloatND,
    consumption: ContinuousAction,
    medicaid_transfer: FloatND,
    return_liquid: float,
    income: float,
) -> ContinuousState:
    """Liquid law of motion with the current-asset Medicaid transfer."""
    return (1.0 + return_liquid) * (coh - consumption) + income + medicaid_transfer


def savings(coh: FloatND, consumption: ContinuousAction) -> FloatND:
    """Post-decision savings: the cash-on-hand not consumed."""
    return coh - consumption


def next_liquid_from_savings(
    savings: FloatND,
    medicaid_transfer: FloatND,
    return_liquid: float,
    income: float,
) -> ContinuousState:
    """Liquid law in post-decision form, carrying the current-asset transfer."""
    return (1.0 + return_liquid) * savings + income + medicaid_transfer


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
    """Create the (alive, dead) toy whose next-asset law jumps at a liquid cliff."""
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    final_age = ages.exact_values[-1]
    liquid_grid = LinSpacedGrid(start=0.1, stop=liquid_max, n_points=n_liquid)

    alive_functions = {
        "utility": utility,
        "coh": coh,
        "medicaid_transfer": medicaid_transfer,
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
            continuous_state="liquid",
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
    medicaid_limit: float = 12.0,
    transfer_amount: float = 2.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the next-asset-cliff toy."""
    base_income = jnp.array([base_income_lo, base_income_hi])
    alive_budget = {"return_liquid": return_liquid, "income": income}
    transfer = {"medicaid_limit": medicaid_limit, "transfer_amount": transfer_amount}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "coh": {"base_income": base_income},
            "medicaid_transfer": transfer,
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
