"""Kinked two-asset toy as an `NEGM` model (the G1 parity target).

The smallest model carrying the Laibson frictions a NEGM solve must reproduce:
a liquid margin `wealth` (X) the Euler equation inverts on, plus an
illiquid/durable margin `illiquid` (Z) with a withdrawal penalty (a kink at
`illiquid_investment = 0`) and a `Z >= 0` floor. The brute oracle for the
equivalent spec is committed in `negm_phase0/kinked_toy_oracle.py` (§2 of
`negm_phase0/negm-phase0-findings.md`).

The NEGM reparametrisation fixes the outer post-decision `next_illiquid`
(`s' = Z + Iz`) per outer-grid node; the inner consumption-savings problem is
then a standard 1-D DC-EGM solve on `wealth`:

- inner Euler state `wealth`, inner action `consumption`, inner post-decision
  `liquid_savings` (`a^X`), inner resources `resources`,
- the outer margin enters the inner `resources` as a constant (the credited
  withdrawal/deposit) and indexes the child durable state `illiquid`,
- the credit-card rate kink lives in the inner Euler law `next_wealth`.

The outer search runs over `next_illiquid` with the no-adjustment candidate
`s' = illiquid` (`Iz = 0`, the withdrawal-penalty kink).
"""

from dataclasses import replace

import jax.numpy as jnp

from lcm import (
    DCEGM,
    NEGM,
    AgeGrid,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

N_X = 12
N_Z = 12
N_C = 25
N_AZ = 25
N_PERIODS = 4

ILLIQUID_FLOW = 0.05  # iota
WITHDRAWAL_PENALTY = 0.10  # kappa on a withdrawal (next_illiquid < illiquid)
BORROW_RATE = 0.12  # credit-card rate on liquid_savings < 0
SAVE_RATE = 0.03  # rate on liquid_savings >= 0
RISK_AVERSION = 2.0
LABOUR_INCOME = 5.0


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def credited(illiquid: ContinuousState, next_illiquid: ContinuousState) -> FloatND:
    """Net liquid cost of moving the durable to `next_illiquid` (`s'`).

    A deposit (`s' > Z`) costs its face value; a withdrawal (`s' < Z`) returns
    only `(1 - kappa)` of the amount pulled out — the penalty kink at `s' = Z`.
    """
    investment = next_illiquid - illiquid
    return jnp.where(
        investment < 0.0,
        (1.0 - WITHDRAWAL_PENALTY) * investment,
        investment,
    )


def resources(
    wealth: ContinuousState, illiquid: ContinuousState, next_illiquid: ContinuousState
) -> FloatND:
    """Liquid resources consumption is paid out of, given the fixed outer node.

    `next_illiquid` (`s'`) is bound to one outer-grid node, so the credited
    durable move is a constant in the inner Euler inversion.
    """
    return (
        wealth
        + LABOUR_INCOME
        - credited(illiquid=illiquid, next_illiquid=next_illiquid)
    )


def liquid_savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Inner post-decision liquid balance `a^X = R_inner - c`."""
    return resources - consumption


def next_wealth(liquid_savings: FloatND) -> ContinuousState:
    """Inner Euler law with the credit-card rate kink at `a^X = 0`."""
    rate = jnp.where(liquid_savings < 0.0, BORROW_RATE, SAVE_RATE)
    return (1.0 + rate) * liquid_savings


def durable_transition(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    """Durable law of motion `s' = Z + Iz`, the `illiquid` state transition.

    Used as the `illiquid` state transition, so pylcm refers to its output as
    the auto-generated `next_illiquid`; the NEGM solver names that value as its
    `outer_post_decision`, which the inner `resources` reads as a kernel-bound
    constant per outer-grid node.
    """
    return illiquid + illiquid_investment


def keep_illiquid(illiquid: ContinuousState) -> FloatND:
    """The no-adjustment candidate `s' = Z` (the withdrawal-penalty kink)."""
    return illiquid


def utility(consumption: ContinuousAction, illiquid: ContinuousState) -> FloatND:
    """CRRA over `consumption + iota * illiquid` — additively separable in `s'`."""
    flow = consumption + ILLIQUID_FLOW * illiquid
    return flow ** (1.0 - RISK_AVERSION) / (1.0 - RISK_AVERSION)


def inverse_marginal_utility(
    marginal_continuation: FloatND, illiquid: ContinuousState
) -> FloatND:
    """Inverse of `u'(c) = (c + iota*Z)^{-gamma}` in the inner consumption slot.

    Inverting $u'(c) = (c + \\iota Z)^{-\\gamma} = m$ for the consumption action
    gives $c = m^{-1/\\gamma} - \\iota Z$: the `iota * Z` shift the durable state
    contributes to the utility flow is a constant offset that must be subtracted
    when recovering `c` from the marginal continuation. The kernel binds the
    durable state `illiquid` (`Z`) from the combo pool, so the offset is exact at
    every durable node rather than a quantity the kernel re-adds downstream.

    The unconstrained inverse can fall below zero where the marginal continuation
    is large (low savings): that is the borrowing-constrained region the kernel's
    constrained-candidate segment and upper envelope represent, so the raw
    inverse is returned without a positivity clamp.
    """
    return marginal_continuation ** (-1.0 / RISK_AVERSION) - ILLIQUID_FLOW * illiquid


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.alive)


WEALTH_GRID = LinSpacedGrid(start=-6.0, stop=30.0, n_points=N_X)
ILLIQUID_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=N_Z)
CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=20.0, n_points=N_C)
ILLIQUID_INVESTMENT_GRID = LinSpacedGrid(start=-8.0, stop=8.0, n_points=N_AZ)
OUTER_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=N_AZ)
SAVINGS_GRID = LinSpacedGrid(start=-5.0, stop=35.0, n_points=80)


NEGM_SOLVER = NEGM(
    inner=DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="liquid_savings",
        savings_grid=SAVINGS_GRID,
    ),
    outer_action="illiquid_investment",
    outer_post_decision="next_illiquid",
    outer_grid=OUTER_GRID,
    outer_no_adjustment_candidate="keep_illiquid",
)


def build_alive_regime(*, outer_batch_size: int = 0) -> Regime:
    """The non-terminal NEGM regime (two assets, two continuous actions).

    `outer_batch_size` chunks the NEGM outer durable search (`0` = all at once).
    """
    final_age_alive = 20 + (N_PERIODS - 2) * 5
    solver = replace(NEGM_SOLVER, outer_batch_size=outer_batch_size)
    return Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID},
        state_transitions={"wealth": next_wealth, "illiquid": durable_transition},
        actions={
            "consumption": CONSUMPTION_GRID,
            "illiquid_investment": ILLIQUID_INVESTMENT_GRID,
        },
        transition=next_regime,
        functions={
            "utility": utility,
            "resources": resources,
            "liquid_savings": liquid_savings,
            "keep_illiquid": keep_illiquid,
            "credited": credited,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=solver,
    )


def build_dead_regime() -> Regime:
    """The terminal regime."""
    final_age_alive = 20 + (N_PERIODS - 2) * 5
    return Regime(
        transition=None,
        active=lambda age, n=final_age_alive: age > n,
        functions={"utility": lambda: 0.0},
    )


def build_model(*, outer_batch_size: int = 0) -> Model:
    """Build the kinked-toy NEGM model (the G1 parity target).

    `outer_batch_size` chunks the NEGM outer durable search (`0` = all at once).
    """
    final_age_alive = 20 + (N_PERIODS - 2) * 5
    return Model(
        regimes={
            "alive": build_alive_regime(outer_batch_size=outer_batch_size),
            "dead": build_dead_regime(),
        },
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=20 + (N_PERIODS - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )
