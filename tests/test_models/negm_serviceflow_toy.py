"""Service-flow kinked two-asset toy: `utility` reads the *new* durable stock.

A variant of the kinked two-asset toy (`negm_kinked_toy.py`) in which the
durable yields its service flow from the **newly chosen** stock `s'`
(`next_illiquid`) rather than the held stock `Z` (`illiquid`):

```{math}
u(c, s') = \\frac{c^{1-\\gamma}}{1-\\gamma} + \\iota\\, s'
```

The service term is additively separable from consumption, so the inner
consumption Euler equation is the plain CRRA inversion (no offset). The novelty
is that `utility` reads `next_illiquid`, the auto-named output of the `illiquid`
state transition and the NEGM `outer_post_decision`. The NEGM solver binds that
value per outer-grid node (the adjuster) or holds it at the durable stock (the
keeper), so the inner kernel evaluates the service flow from the chosen house —
the Dobrescu-Shanker housing pattern (`utility` reads `next_housing`).

`build_negm_model` solves it by the nested EGM; `build_brute_model` is the
economically identical grid-search twin used as the parity oracle. Brute
restricts the continuous policy to the action grid, so its value is a lower
bound: NEGM weakly dominates and approaches it as the brute grids refine.
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

ILLIQUID_FLOW = 0.05  # iota: service flow per unit of the new durable stock s'
WITHDRAWAL_PENALTY = 0.10  # kappa on a withdrawal (next_illiquid < illiquid)
BORROW_RATE = 0.12  # credit-card rate on liquid_savings < 0
SAVE_RATE = 0.03  # rate on liquid_savings >= 0
RISK_AVERSION = 2.0
LABOUR_INCOME = 5.0
SAVINGS_FLOOR = -5.0  # borrowing limit on the inner post-decision a^X


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


def resources(wealth: ContinuousState, credited: FloatND) -> FloatND:
    """Liquid resources consumption is paid out of, given the fixed outer node.

    Reads the outer margin exclusively through the `credited` cost function —
    the NEGM outer-cost contract.
    """
    return wealth + LABOUR_INCOME - credited


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

    pylcm names its output `next_illiquid`; the NEGM solver names that value as
    its `outer_post_decision`, read by the inner `resources` and `utility` as a
    kernel-bound constant per outer-grid node.
    """
    return illiquid + illiquid_investment


def keep_illiquid(illiquid: ContinuousState) -> FloatND:
    """The no-adjustment candidate `s' = Z` (the withdrawal-penalty kink)."""
    return illiquid


def serviced_durable(next_illiquid: ContinuousState) -> FloatND:
    """Service flow from the newly chosen durable stock `s'`.

    Routing the outer post-decision through its own function (rather than into
    `utility`'s signature directly) keeps `utility` additively separable in the
    inner and outer margins — the Dobrescu-Shanker housing structure, where
    `utility` reads a `serviced_housing(next_housing)` flow.
    """
    return ILLIQUID_FLOW * next_illiquid


def utility(consumption: ContinuousAction, serviced_durable: FloatND) -> FloatND:
    """CRRA over consumption plus an additive service flow from the new durable.

    The service term is constant given the bound outer node, so it leaves the
    inner consumption Euler inversion a plain CRRA inverse.
    """
    consumption_utility = consumption ** (1.0 - RISK_AVERSION) / (1.0 - RISK_AVERSION)
    return consumption_utility + serviced_durable


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    """Inverse of `u'(c) = c^{-gamma}` — the additive service term drops out."""
    return marginal_continuation ** (-1.0 / RISK_AVERSION)


def credited_brute(illiquid: ContinuousState, new_durable: ContinuousAction) -> FloatND:
    """Brute-twin credited durable move to the chosen new stock `s'`."""
    investment = new_durable - illiquid
    return jnp.where(
        investment < 0.0,
        (1.0 - WITHDRAWAL_PENALTY) * investment,
        investment,
    )


def resources_brute(
    wealth: ContinuousState, illiquid: ContinuousState, new_durable: ContinuousAction
) -> FloatND:
    """Brute-twin liquid resources, in state + action terms (no next-state read)."""
    return (
        wealth
        + LABOUR_INCOME
        - credited_brute(illiquid=illiquid, new_durable=new_durable)
    )


def next_wealth_brute(
    wealth: ContinuousState,
    illiquid: ContinuousState,
    new_durable: ContinuousAction,
    consumption: ContinuousAction,
) -> ContinuousState:
    """Brute-twin liquid law of motion with the credit-card rate kink."""
    savings = (
        resources_brute(wealth=wealth, illiquid=illiquid, new_durable=new_durable)
        - consumption
    )
    rate = jnp.where(savings < 0.0, BORROW_RATE, SAVE_RATE)
    return (1.0 + rate) * savings


def next_illiquid_brute(new_durable: ContinuousAction) -> ContinuousState:
    """Brute-twin durable law of motion: the chosen new stock is next period's."""
    return new_durable


def serviced_durable_brute(new_durable: ContinuousAction) -> FloatND:
    """Brute-twin service flow from the chosen durable stock `s'`."""
    return ILLIQUID_FLOW * new_durable


def feasible(
    wealth: ContinuousState,
    illiquid: ContinuousState,
    new_durable: ContinuousAction,
    consumption: ContinuousAction,
) -> BoolND:
    """The implied post-decision balance must stay at or above the borrowing floor."""
    savings = (
        resources_brute(wealth=wealth, illiquid=illiquid, new_durable=new_durable)
        - consumption
    )
    return savings >= SAVINGS_FLOOR


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.alive)


WEALTH_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=N_X)
ILLIQUID_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=N_Z)
CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=20.0, n_points=N_C)
CONSUMPTION_GRID_BRUTE = LinSpacedGrid(start=0.1, stop=20.0, n_points=300)
ILLIQUID_INVESTMENT_GRID = LinSpacedGrid(start=-8.0, stop=8.0, n_points=N_AZ)
OUTER_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=N_AZ)
SAVINGS_GRID = LinSpacedGrid(start=SAVINGS_FLOOR, stop=35.0, n_points=80)


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


FINAL_AGE_ALIVE = 20 + (N_PERIODS - 2) * 5


def _build_dead_regime() -> Regime:
    """The terminal regime (shared by both twins)."""
    return Regime(
        transition=None,
        active=lambda age, n=FINAL_AGE_ALIVE: age > n,
        functions={"utility": lambda: 0.0},
    )


def build_negm_model() -> Model:
    """Build the service-flow toy solved by the nested EGM."""
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
            "serviced_durable": serviced_durable,
            "resources": resources,
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
        functions={
            "utility": utility,
            "serviced_durable": serviced_durable_brute,
        },
        constraints={"feasible": feasible},
    )
    return Model(
        regimes={"alive": alive, "dead": _build_dead_regime()},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=20 + (N_PERIODS - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": FINAL_AGE_ALIVE},
    )
