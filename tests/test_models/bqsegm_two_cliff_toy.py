"""One-asset toy with two subsidy cliffs declared as a jump `piecewise_affine`.

Cash-on-hand jumps down at each of two asset cliffs (the subsidy steps down
twice), so the budget and the value function jump at both. The BQSEGM schedule
path recognises the all-jump schedule and routes it to the recurring N-cliff step,
which must reproduce the dense `GridSearch` value across both cliffs at every age.
"""

import jax.numpy as jnp

import lcm
from lcm import AgeGrid, LinSpacedGrid, MarkovTransition, Model, categorical
from lcm.regime import Regime
from lcm.solvers import GridSearch
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


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


@lcm.piecewise_affine(
    "subsidy",
    variable="liquid",
    breakpoints=(
        lcm.affine_breakpoint("cliff_low", kind="jump"),
        lcm.affine_breakpoint("cliff_high", kind="jump"),
    ),
)
def subsidy(
    liquid: ContinuousState,
    subsidy_low: float,
    subsidy_mid: float,
    subsidy_high: float,
    cliff_low: float,
    cliff_high: float,
) -> FloatND:
    """Step subsidy: high below the low cliff, mid between, low above the high cliff."""
    return jnp.where(
        liquid < cliff_low,
        subsidy_high,
        jnp.where(liquid < cliff_high, subsidy_mid, subsidy_low),
    )


def coh(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Cash-on-hand: liquid wealth plus the cliff-contingent subsidy."""
    return liquid + subsidy


def next_liquid(
    coh: FloatND,
    consumption: ContinuousAction,
    return_liquid: float,
    income: float,
) -> ContinuousState:
    """Liquid law of motion: saved cash earns the liquid return, plus income."""
    return (1.0 + return_liquid) * (coh - consumption) + income


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
    """Create the two-regime (alive, dead) two-cliff one-asset toy."""
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    final_age = ages.exact_values[-1]
    liquid_grid = LinSpacedGrid(start=0.1, stop=liquid_max, n_points=n_liquid)

    alive_functions = {"utility": utility, "subsidy": subsidy, "coh": coh}
    if variant == "brute":
        alive_solver = GridSearch()
    elif variant == "bqsegm":
        from lcm.solvers import BQSEGM  # noqa: PLC0415

        alive_solver = BQSEGM(
            savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings)
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
        states={"liquid": liquid_grid},
        state_transitions={"liquid": {"alive": next_liquid, "dead": next_liquid}},
        constraints={"feasible": feasible},
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
    subsidy_high: float = 3.0,
    subsidy_mid: float = 1.5,
    subsidy_low: float = 0.5,
    cliff_low: float = 6.0,
    cliff_high: float = 14.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the two-cliff one-asset toy."""
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "subsidy": {
                "subsidy_low": subsidy_low,
                "subsidy_mid": subsidy_mid,
                "subsidy_high": subsidy_high,
                "cliff_low": cliff_low,
                "cliff_high": cliff_high,
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
