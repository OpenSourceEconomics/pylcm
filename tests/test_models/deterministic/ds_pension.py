"""Dobrescu--Shanker / Druedahl--Jorgensen two-asset pension benchmark (brute form).

The 2-D pension model the RFC-vs-G2EGM comparison solves: a finite-horizon
consumption--saving problem with a **liquid** account and an illiquid **pension**
account. While working, the agent chooses consumption `c` and a one-directional
pension `deposit` (`d >= 0`); the liquid post-decision balance `liquid - c - d` is
kept non-negative by a borrowing constraint, and the pension post-decision balance
`pension + d + chi*log(1 + d)` carries a concave employer match. Liquid earns gross
return `1 + return_liquid`, pension the higher `1 + return_pension`.

Retirement is a **deterministic lifecycle transition** at a fixed age (not an
endogenous per-period choice -- the Druedahl--Jorgensen `solve.m` solves the retired
sub-problem as a separate 1-D continuation, with no per-period work/retire max). At
the working->retired transition the pension is paid out as a lump sum into liquid and
the problem collapses to a 1-D liquid-only consumption problem; the retired agent
receives a flat retirement income and earns the liquid return. Working utility carries
an additive disutility of work `work_disutility` (the retired agent pays none).

This is the dense-grid brute-force **oracle** the 2-D EGM kernel (G2EGM / multidim
RFC) is validated against. It is written in brute-solvable form -- two continuous
states (`liquid`, `pension`), two continuous actions (`consumption`, `deposit`), both
coupled through the budget -- with the faithful calibration read from the G2EGM
`SetupPar.m` (`get_params`). Default grids are small for fast local oracle solves;
pass larger sizes (and the full calibration) for the reference comparison.

Two micro-conventions are parameterized pending confirmation against `fun.m`:
`pension_payout_return` (the return factor applied to the paid-out pension balance at
retirement; `1 + return_pension` by default) and `retirement_income_in_first_period`
(whether `retirement_income` is received in the first retired period).
"""

import functools

import jax.numpy as jnp

from lcm import AgeGrid, LinSpacedGrid, MarkovTransition, Model, categorical
from lcm.regime import Regime
from lcm.solvers import GridSearch, Solver
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _crra(consumption: FloatND, crra: float) -> FloatND:
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption ** (1.0 - crra) / (1.0 - crra),
    )


def utility_working(
    consumption: ContinuousAction, crra: float, work_disutility: float
) -> FloatND:
    """CRRA consumption utility net of the additive disutility of work."""
    return _crra(consumption, crra) - work_disutility


def utility_retired(consumption: ContinuousAction, crra: float) -> FloatND:
    """CRRA consumption utility; the retired agent pays no work disutility."""
    return _crra(consumption, crra)


def bequest(liquid: ContinuousState, crra: float) -> FloatND:
    """Terminal value: consume remaining liquid wealth (the pension is paid out)."""
    return _crra(liquid, crra)


def _pension_post_decision(
    pension: ContinuousState,
    deposit: ContinuousAction,
    match_rate: float,
) -> FloatND:
    """Pension post-decision balance `pension + deposit + chi*log(1 + deposit)`."""
    return pension + deposit + match_rate * jnp.log(1.0 + deposit)


def next_liquid_working(
    liquid: ContinuousState,
    consumption: ContinuousAction,
    deposit: ContinuousAction,
    return_liquid: float,
    wage: float,
) -> ContinuousState:
    """Liquid law of motion while staying in the working regime."""
    return (1.0 + return_liquid) * (liquid - consumption - deposit) + wage


def next_liquid_retiring(
    liquid: ContinuousState,
    pension: ContinuousState,
    consumption: ContinuousAction,
    deposit: ContinuousAction,
    return_liquid: float,
    pension_payout_return: float,
    match_rate: float,
    retirement_income: float,
) -> ContinuousState:
    """Liquid on the working->retired transition: the pension is paid out as a lump sum.

    The liquid post-decision balance earns the liquid return, the pension
    post-decision balance is paid out scaled by `pension_payout_return`, and the
    first retirement income is added.
    """
    liquid_post = liquid - consumption - deposit
    pension_post = _pension_post_decision(pension, deposit, match_rate)
    return (
        (1.0 + return_liquid) * liquid_post
        + pension_payout_return * pension_post
        + retirement_income
    )


def next_pension_working(
    pension: ContinuousState,
    deposit: ContinuousAction,
    return_pension: float,
    match_rate: float,
) -> ContinuousState:
    """Pension law of motion while staying in the working regime."""
    return (1.0 + return_pension) * _pension_post_decision(pension, deposit, match_rate)


def next_liquid_retired(
    liquid: ContinuousState,
    consumption: ContinuousAction,
    return_liquid: float,
    retirement_income: float,
) -> ContinuousState:
    """Liquid law of motion within retirement (1-D consumption--saving)."""
    return (1.0 + return_liquid) * (liquid - consumption) + retirement_income


def feasible_working(
    liquid: ContinuousState,
    consumption: ContinuousAction,
    deposit: ContinuousAction,
) -> BoolND:
    """Liquid borrowing constraint: the liquid post-decision balance stays >= 0."""
    return consumption + deposit <= liquid


def feasible_retired(
    liquid: ContinuousState,
    consumption: ContinuousAction,
) -> BoolND:
    """Liquid borrowing constraint in retirement (no deposit margin)."""
    return consumption <= liquid


def prob_stay_working(age: int, retirement_age: float) -> FloatND:
    """Deterministic (0/1) probability of staying in the working regime next period."""
    return jnp.where(age + 1 < retirement_age, 1.0, 0.0)


def prob_retire(age: int, retirement_age: float) -> FloatND:
    """Deterministic (0/1) probability of transitioning working->retired next period."""
    return jnp.where(age + 1 >= retirement_age, 1.0, 0.0)


def prob_stay_retired(age: int, final_age_alive: float) -> FloatND:
    """Deterministic (0/1) probability of remaining retired next period."""
    return jnp.where(age + 1 < final_age_alive, 1.0, 0.0)


def prob_die(age: int, final_age_alive: float) -> FloatND:
    """Deterministic (0/1) probability of transitioning retired->dead next period."""
    return jnp.where(age + 1 >= final_age_alive, 1.0, 0.0)


def get_model(
    *,
    n_periods: int = 5,
    retirement_period: int = 3,
    n_liquid: int = 12,
    n_pension: int = 10,
    n_consumption: int = 14,
    n_deposit: int = 8,
    liquid_max: float = 20.0,
    pension_max: float = 15.0,
    solvers: dict[str, Solver] | None = None,
) -> Model:
    """Create the three-regime (working, retired, dead) DS pension model.

    The agent works for `retirement_period` periods, then is retired for the rest of
    the `n_periods`-period horizon, with a terminal (dead) period. Grid sizes default
    to a small oracle scale; pass larger values for a finer reference solve.

    Args:
        solvers: Optional mapping of regime name to its `Solver`. A name absent from
            the mapping (or `solvers=None`) keeps the default `GridSearch` — so the
            default model is the dense-grid brute oracle. Pass
            `{"working": TwoDimEGM(...)}` to drive the working regime by the two-asset
            G2EGM method, and `{"retired": OneAssetEGM(...)}` for the 1-D retired EGM.
    """
    solvers = solvers or {}
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    retirement_age = ages.exact_values[retirement_period]
    final_age = ages.exact_values[-1]
    liquid_grid = LinSpacedGrid(start=0.1, stop=liquid_max, n_points=n_liquid)
    pension_grid = LinSpacedGrid(start=0.0, stop=pension_max, n_points=n_pension)
    consumption_grid = LinSpacedGrid(start=0.1, stop=liquid_max, n_points=n_consumption)

    working = Regime(
        actions={
            "consumption": consumption_grid,
            "deposit": LinSpacedGrid(start=0.0, stop=pension_max, n_points=n_deposit),
        },
        states={"liquid": liquid_grid, "pension": pension_grid},
        state_transitions={
            "liquid": {
                "working": next_liquid_working,
                "retired": next_liquid_retiring,
            },
            "pension": {"working": next_pension_working},
        },
        constraints={"feasible": feasible_working},
        transition={
            "working": MarkovTransition(prob_stay_working),
            "retired": MarkovTransition(prob_retire),
        },
        functions={"utility": utility_working},
        active=lambda age, ra=retirement_age: age < ra,
        solver=solvers.get("working", GridSearch()),
    )
    retired = Regime(
        actions={"consumption": consumption_grid},
        states={"liquid": liquid_grid},
        state_transitions={
            "liquid": {
                "retired": next_liquid_retired,
                "dead": next_liquid_retired,
            }
        },
        constraints={"feasible": feasible_retired},
        transition={
            "retired": MarkovTransition(prob_stay_retired),
            "dead": MarkovTransition(prob_die),
        },
        functions={"utility": utility_retired},
        active=lambda age, ra=retirement_age, fa=final_age: ra <= age < fa,
        solver=solvers.get("retired", GridSearch()),
    )
    dead = Regime(
        transition=None,
        states={"liquid": liquid_grid},
        functions={"utility": bequest},
        solver=solvers.get("dead", GridSearch()),
    )
    return Model(
        regimes={"working": working, "retired": retired, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )


@functools.cache
def get_params(
    *,
    discount_factor: float = 0.98,
    crra: float = 2.0,
    work_disutility: float = 0.25,
    return_liquid: float = 0.02,
    return_pension: float = 0.04,
    match_rate: float = 0.10,
    wage: float = 1.0,
    retirement_income: float = 0.50,
    retirement_age: float = 3.0,
    final_age_alive: float = 4.0,
    pension_payout_return: float | None = None,
) -> dict:
    """Get parameters for the DS pension model (faithful calibration from `SetupPar.m`).

    `pension_payout_return` defaults to `1 + return_pension` (the pension balance is
    paid out earning its own return at retirement); pass a value to override the
    convention pending confirmation against `fun.m`.
    """
    if pension_payout_return is None:
        pension_payout_return = 1.0 + return_pension
    return {
        "working": {
            "utility": {"crra": crra, "work_disutility": work_disutility},
            "H": {"discount_factor": discount_factor},
            "working": {
                "next_liquid": {"return_liquid": return_liquid, "wage": wage},
                "next_pension": {
                    "match_rate": match_rate,
                    "return_pension": return_pension,
                },
                "next_regime": {"retirement_age": retirement_age},
            },
            "retired": {
                "next_liquid": {
                    "match_rate": match_rate,
                    "pension_payout_return": pension_payout_return,
                    "retirement_income": retirement_income,
                    "return_liquid": return_liquid,
                },
                "next_regime": {"retirement_age": retirement_age},
            },
        },
        "retired": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "retired": {
                "next_liquid": {
                    "retirement_income": retirement_income,
                    "return_liquid": return_liquid,
                },
                "next_regime": {"final_age_alive": final_age_alive},
            },
            "dead": {
                "next_liquid": {
                    "retirement_income": retirement_income,
                    "return_liquid": return_liquid,
                },
                "next_regime": {"final_age_alive": final_age_alive},
            },
        },
        "dead": {"utility": {"crra": crra}},
    }
