"""One-asset consumption--saving toy with a Medicaid case boundary on assets.

A minimal model exercising BQSEGM: a single liquid asset, a single consumption
action, and a binary Medicaid eligibility test on current assets. Eligibility
(`liquid < medicaid_asset_limit`) grants a larger lump-sum subsidy into
cash-on-hand than the private side, so the budget — and hence the value function —
jumps down as assets cross the limit upward. Within each case the budget is
smooth, so BQSEGM solves each case by EGM and stitches them at the boundary; the
brute variant evaluates the `jnp.where` combination on a dense grid and is the
agreement oracle.

`build_model(variant="brute")` uses `GridSearch`; `build_model(variant="bqsegm")`
uses the `BQSEGM` solver over the decorated case pieces.
"""

import functools

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


@lcm.case_boundary(
    lcm.boundary("liquid", "medicaid_asset_limit", equality="otherwise", kind="jump")
)
def medicaid_eligible(liquid: ContinuousState, medicaid_asset_limit: float) -> BoolND:
    """Medicaid asset test: eligible while liquid wealth is below the limit."""
    return liquid < medicaid_asset_limit


@lcm.piece("subsidy", when=medicaid_eligible)
def subsidy_medicaid(subsidy_high: float) -> FloatND:
    """Subsidy into cash-on-hand for the Medicaid-eligible (low-asset) case."""
    return jnp.asarray(subsidy_high)


@lcm.piece("subsidy", otherwise=medicaid_eligible)
def subsidy_private(subsidy_low: float) -> FloatND:
    """Subsidy into cash-on-hand for the private (high-asset) case."""
    return jnp.asarray(subsidy_low)


def subsidy(
    subsidy_medicaid: FloatND, subsidy_private: FloatND, medicaid_eligible: BoolND
) -> FloatND:
    """Brute-force combination of the two subsidy pieces (BQSEGM ignores this)."""
    return jnp.where(medicaid_eligible, subsidy_medicaid, subsidy_private)


def coh(liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Cash-on-hand: liquid wealth plus the Medicaid-contingent subsidy."""
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
    n_liquid: int = 80,
    n_consumption: int = 120,
    liquid_max: float = 20.0,
    n_savings: int = 100,
    savings_max: float = 20.0,
) -> Model:
    """Create the two-regime (alive, dead) Medicaid one-asset toy.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch` (the dense-grid
            oracle); `"bqsegm"` drives it by the `BQSEGM` case-piece solver.
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
        "medicaid_eligible": medicaid_eligible,
        "subsidy_medicaid": subsidy_medicaid,
        "subsidy_private": subsidy_private,
        "subsidy": subsidy,
        "coh": coh,
    }
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


@functools.cache
def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    income: float = 1.0,
    subsidy_high: float = 3.0,
    subsidy_low: float = 0.5,
    medicaid_asset_limit: float = 8.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the Medicaid one-asset toy.

    The Medicaid-eligible subsidy (`subsidy_high`) exceeds the private subsidy
    (`subsidy_low`), so cash-on-hand — and the value function — jumps down as
    liquid wealth crosses `medicaid_asset_limit` upward.
    """
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "medicaid_eligible": {"medicaid_asset_limit": medicaid_asset_limit},
            "subsidy_medicaid": {"subsidy_high": subsidy_high},
            "subsidy_private": {"subsidy_low": subsidy_low},
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
