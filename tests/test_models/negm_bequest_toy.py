"""Two-asset NEGM toy whose terminal bequest reads *both* continuous states.

A kinked two-asset toy (liquid `wealth`, durable `illiquid`) solved by the
nested EGM, whose terminal `dead` regime values a bequest over **both** the
liquid and the durable stock — `bequest(wealth, illiquid)`. The durable is the
NEGM outer (passive) margin, so the terminal carry the EGM parent reads has the
Euler state `wealth` plus the passive continuous state `illiquid` as a leading
axis — the Dobrescu-Shanker housing pattern (the bequest reads the resale value
of the house alongside liquid wealth).

`utility` reads only the *held* durable, so the inner consumption margin is
isolated from the outer one — the single new feature exercised is the
two-continuous-state terminal carry. `build_negm_model` solves it by the nested
EGM; `build_brute_model` is the economically identical grid-search twin used as
the parity oracle.
"""

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
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

N_X = 8
N_Z = 8
N_C = 15
N_AZ = 12
N_PERIODS = 3

ILLIQUID_FLOW = 0.05  # iota: service flow from the held durable
WITHDRAWAL_PENALTY = 0.10  # kappa on a withdrawal (next_illiquid < illiquid)
SAVE_RATE = 0.03  # single interest rate on liquid savings
RISK_AVERSION = 2.0
LABOUR_INCOME = 5.0
BEQUEST_WEIGHT = 0.8  # weight on the terminal bequest
SAVINGS_FLOOR = -5.0  # borrowing limit on the inner post-decision a^X


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def credited(illiquid: ContinuousState, next_illiquid: ContinuousState) -> FloatND:
    """Net liquid cost of moving the durable to `next_illiquid` (`s'`)."""
    investment = next_illiquid - illiquid
    return jnp.where(
        investment < 0.0,
        (1.0 - WITHDRAWAL_PENALTY) * investment,
        investment,
    )


def resources_before_outer_cost(wealth: ContinuousState) -> FloatND:
    """Cost-free base of the liquid resources consumption is paid out of.

    With `NEGM.outer_cost` declared, pylcm composes the resources function as
    `resources_before_outer_cost - credited` at model build, so the credited
    durable move enters resources additively by construction.
    """
    return wealth + LABOUR_INCOME


def liquid_savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Inner post-decision liquid balance `a^X = R_inner - c`."""
    return resources - consumption


def next_wealth(liquid_savings: FloatND) -> ContinuousState:
    """Inner Euler law (single interest rate)."""
    return (1.0 + SAVE_RATE) * liquid_savings


def durable_transition(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    """Durable law of motion `s' = Z + Iz`, the `illiquid` state transition."""
    return illiquid + illiquid_investment


def keep_illiquid(illiquid: ContinuousState) -> FloatND:
    """The no-adjustment candidate `s' = Z` (the withdrawal-penalty kink)."""
    return illiquid


def utility(consumption: ContinuousAction, illiquid: ContinuousState) -> FloatND:
    """CRRA over `consumption + iota * illiquid` — reads only the held durable."""
    flow = consumption + ILLIQUID_FLOW * illiquid
    return flow ** (1.0 - RISK_AVERSION) / (1.0 - RISK_AVERSION)


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    """Inverse of `u'(c) = (c + iota*Z)^{-gamma}` in the inner consumption slot."""
    return marginal_continuation ** (-1.0 / RISK_AVERSION)


def bequest(wealth: ContinuousState, illiquid: ContinuousState) -> FloatND:
    """Terminal value over both liquid wealth and the durable resale value."""
    total = wealth + illiquid + 1.0
    return BEQUEST_WEIGHT * total ** (1.0 - RISK_AVERSION) / (1.0 - RISK_AVERSION)


def feasible(liquid_savings: FloatND) -> BoolND:
    """The inner post-decision balance must stay at or above the borrowing floor."""
    return liquid_savings >= SAVINGS_FLOOR


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.alive)


WEALTH_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=N_X)
ILLIQUID_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=N_Z)
CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=20.0, n_points=N_C)
CONSUMPTION_GRID_BRUTE = LinSpacedGrid(start=0.1, stop=20.0, n_points=300)
ILLIQUID_INVESTMENT_GRID = LinSpacedGrid(start=-8.0, stop=8.0, n_points=N_AZ)
OUTER_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=N_AZ)
SAVINGS_GRID = LinSpacedGrid(start=SAVINGS_FLOOR, stop=35.0, n_points=80)

FINAL_AGE_ALIVE = 20 + (N_PERIODS - 2) * 5

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
    outer_cost="credited",
)


def _build_dead_regime() -> Regime:
    """The terminal regime: a bequest over both continuous states."""
    return Regime(
        transition=None,
        active=lambda age, n=FINAL_AGE_ALIVE: age > n,
        states={"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID},
        functions={"utility": bequest},
    )


def build_negm_model() -> Model:
    """Build the bequest toy solved by the nested EGM."""
    alive = Regime(
        active=lambda age, n=FINAL_AGE_ALIVE: age <= n,
        states={"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID},
        state_transitions={"wealth": next_wealth, "illiquid": durable_transition},
        actions={
            "consumption": CONSUMPTION_GRID,
            "illiquid_investment": ILLIQUID_INVESTMENT_GRID,
        },
        transition=next_regime,
        functions={
            "utility": utility,
            "resources_before_outer_cost": resources_before_outer_cost,
            "liquid_savings": liquid_savings,
            "keep_illiquid": keep_illiquid,
            "credited": credited,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=NEGM_SOLVER,
    )
    return Model(
        regimes={"alive": alive, "dead": _build_dead_regime()},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=20 + (N_PERIODS - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": FINAL_AGE_ALIVE},
    )


def next_wealth_brute(
    wealth: ContinuousState,
    illiquid: ContinuousState,
    new_durable: ContinuousAction,
    consumption: ContinuousAction,
) -> ContinuousState:
    """Brute-twin liquid law of motion, in state + action terms."""
    investment = new_durable - illiquid
    credited_move = jnp.where(
        investment < 0.0, (1.0 - WITHDRAWAL_PENALTY) * investment, investment
    )
    savings = wealth + LABOUR_INCOME - credited_move - consumption
    return (1.0 + SAVE_RATE) * savings


def next_illiquid_brute(new_durable: ContinuousAction) -> ContinuousState:
    """Brute-twin durable law of motion: the chosen new stock is next period's."""
    return new_durable


def utility_brute(consumption: ContinuousAction, illiquid: ContinuousState) -> FloatND:
    """Brute-twin flow utility — reads the held durable, as the NEGM twin does."""
    flow = consumption + ILLIQUID_FLOW * illiquid
    return flow ** (1.0 - RISK_AVERSION) / (1.0 - RISK_AVERSION)


def feasible_brute(
    wealth: ContinuousState,
    illiquid: ContinuousState,
    new_durable: ContinuousAction,
    consumption: ContinuousAction,
) -> BoolND:
    """The implied post-decision balance must stay at or above the borrowing floor."""
    investment = new_durable - illiquid
    credited_move = jnp.where(
        investment < 0.0, (1.0 - WITHDRAWAL_PENALTY) * investment, investment
    )
    savings = wealth + LABOUR_INCOME - credited_move - consumption
    return savings >= SAVINGS_FLOOR


def build_brute_model() -> Model:
    """Build the economically identical grid-search twin (the parity oracle)."""
    alive = Regime(
        active=lambda age, n=FINAL_AGE_ALIVE: age <= n,
        states={"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID},
        state_transitions={
            "wealth": next_wealth_brute,
            "illiquid": next_illiquid_brute,
        },
        actions={
            "consumption": CONSUMPTION_GRID_BRUTE,
            "new_durable": OUTER_GRID,
        },
        transition=next_regime,
        functions={"utility": utility_brute},
        constraints={"feasible": feasible_brute},
    )
    return Model(
        regimes={"alive": alive, "dead": _build_dead_regime()},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=20 + (N_PERIODS - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": FINAL_AGE_ALIVE},
    )
