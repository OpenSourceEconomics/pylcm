"""Epstein-Zin lifecycle savings model with health-dependent mortality.

An example consumer block in the spirit of Atal, Fang, Karlsson &
Ziebarth (2025, JPE 133(6), doi:10.1086/734781) - savings, a two-state
health Markov chain, and health-dependent survival into a terminal `dead`
regime - with the recursion swapped to Epstein-Zin:

- `V = ((1 - b) * c^r + b * CE^r)^(1/r)` via a user-supplied `H`,
- `CE = (E[V'^(1-g)])^(1/(1-g))` via `PowerMean`.

Utility is consumption itself, so values stay in (positive) consumption
units and the power transform is well-defined. The `dead` bequest value
`sqrt(wealth)` is strictly positive at every reachable wealth. At default
grid sizes next wealth `w - c + income` stays inside the wealth grids
(`income` equals the consumption-grid lower bound), so linear interpolation
never extrapolates and an in-test numpy backward induction on the same
grids reproduces the solve exactly.

See docs/examples/epstein_zin.ipynb for a full exposition of the model,
its recursion, and usage examples.
"""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    CertaintyEquivalent,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    Float1D,
    FloatND,
    Period,
    ScalarInt,
)

WEALTH_GRID = LinSpacedGrid(start=0.5, stop=12.0, n_points=6)
DEAD_WEALTH_GRID = LinSpacedGrid(start=0.0, stop=12.0, n_points=25)
CONSUMPTION_GRID = LinSpacedGrid(start=0.5, stop=5.0, n_points=7)

INCOME = 0.5
SURVIVAL_PROBS = (0.95, 0.85, 0.0)
BAD_HEALTH_SURVIVAL_FACTOR = 0.9
HEALTH_TRANSITION = ((0.8, 0.2), (0.1, 0.9))  # rows: bad, good


@categorical(ordered=False)
class EZRegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class HealthStatus:
    bad: ScalarInt
    good: ScalarInt


def utility_alive(consumption: ContinuousAction) -> FloatND:
    return consumption


def utility_dead(wealth: ContinuousState) -> FloatND:
    return jnp.sqrt(wealth)


def H_epstein_zin(
    utility: FloatND, E_next_V: FloatND, discount_factor: FloatND, rho: FloatND
) -> FloatND:
    return (
        (1.0 - discount_factor) * utility**rho + discount_factor * E_next_V**rho
    ) ** (1.0 / rho)


def next_wealth(
    wealth: ContinuousState, consumption: ContinuousAction, income: FloatND
) -> ContinuousState:
    return wealth - consumption + income


def health_probs(health: DiscreteState) -> FloatND:
    return jnp.where(
        health == HealthStatus.good,
        jnp.array(HEALTH_TRANSITION[1]),
        jnp.array(HEALTH_TRANSITION[0]),
    )


def next_regime(
    health: DiscreteState, period: Period, survival_probs: Float1D
) -> FloatND:
    sp = survival_probs[period] * jnp.where(
        health == HealthStatus.good, 1.0, BAD_HEALTH_SURVIVAL_FACTOR
    )
    return jnp.array([sp, 1.0 - sp])


def budget_constraint(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def get_model(
    *,
    certainty_equivalent: CertaintyEquivalent | None,
    n_periods: int = 4,
    n_wealth_points: int = 6,
    n_consumption_points: int = 7,
    n_dead_wealth_points: int = 25,
) -> Model:
    """Create the Epstein-Zin lifecycle model with health-dependent mortality.

    Args:
        certainty_equivalent: Certainty equivalent specification for the expectation.
            Pass `None` to use the linear-expectation (expected-utility) recursion.
        n_periods: Number of lifecycle periods (annual, starting at age 60).
        n_wealth_points: Number of points in the alive wealth grid.
        n_consumption_points: Number of points in the consumption grid.
        n_dead_wealth_points: Number of points in the dead (bequest) wealth grid.

    Returns:
        A configured Model instance.

    """
    last_age = 60 + n_periods - 1
    wealth_grid = LinSpacedGrid(
        start=WEALTH_GRID.start, stop=WEALTH_GRID.stop, n_points=n_wealth_points
    )
    dead_wealth_grid = LinSpacedGrid(
        start=DEAD_WEALTH_GRID.start,
        stop=DEAD_WEALTH_GRID.stop,
        n_points=n_dead_wealth_points,
    )
    consumption_grid = LinSpacedGrid(
        start=CONSUMPTION_GRID.start,
        stop=CONSUMPTION_GRID.stop,
        n_points=n_consumption_points,
    )
    alive = Regime(
        transition=MarkovTransition(next_regime),
        states={
            "wealth": wealth_grid,
            "health": DiscreteGrid(HealthStatus),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": {"alive": MarkovTransition(health_probs)},
        },
        actions={"consumption": consumption_grid},
        constraints={"budget_constraint": budget_constraint},
        functions={"utility": utility_alive, "H": H_epstein_zin},
        certainty_equivalent=certainty_equivalent,
        active=lambda age, la=last_age: age < la,
    )
    dead = Regime(
        transition=None,
        states={"wealth": dead_wealth_grid},
        functions={"utility": utility_dead},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=60, stop=last_age, step="Y"),
        regime_id_class=EZRegimeId,
    )


def get_params(
    *,
    risk_aversion: float | None,
    discount_factor: float = 0.9,
    rho: float = 0.5,
    survival_probs: tuple[float, ...] = SURVIVAL_PROBS,
) -> dict:
    """Get parameters for the Epstein-Zin lifecycle model.

    Args:
        risk_aversion: Coefficient of relative risk aversion (gamma) for the certainty
            equivalent. Pass `None` when solving without a certainty equivalent.
        discount_factor: Time discount factor (beta).
        rho: Intertemporal elasticity exponent (rho = 1 - 1/psi).
        survival_probs: Per-period survival probabilities (last entry must be 0.0);
            its length is `n_periods - 1`.

    Returns:
        Parameter dict ready for `model.solve()`.

    """
    params: dict = {
        "alive": {
            "H": {"discount_factor": discount_factor, "rho": rho},
            "next_wealth": {"income": INCOME},
            "next_regime": {"survival_probs": jnp.array(survival_probs)},
        },
    }
    if risk_aversion is not None:
        params["alive"]["certainty_equivalent"] = {"risk_aversion": risk_aversion}
    return params
