"""Consumption-savings model with state-conditioned income risk (stochastic volatility).

Income is a named current Normal shock observed before consumption. Its scale reads
`lagged_uncertainty`, copied from `uncertainty` by the preceding transition, so income
next period retains the source period's volatility regime even when uncertainty
switches. The persistent uncertainty state follows its original Markov law.

Initial conditions must declare both uncertainty states. Supply known entry income
with `PhysicalValues`; omitting that observation requests a current draw. Positive
standard-normal Gauss-Hermite quadrature prices current policy values instead of the
former fixed physical income grid. The economic timing is retained, while that old
finite interpolation operator and seeded numerical records are not reference answers
for this declaration.
"""

import functools
import math

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DeterministicTransition,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.exceptions import GridInitializationError
from lcm.processes import StateConditioned
from lcm.shocks import GaussHermite, Normal, Product, ShockGroup
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserParamsNode,
)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Uncertainty:
    low: ScalarInt
    high: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def next_wealth(
    *,
    wealth: ContinuousState,
    consumption: ContinuousAction,
    income: ContinuousState,
    interest_rate: float,
) -> FloatND:
    return (1 + interest_rate) * (wealth - consumption) + jnp.exp(income)


def next_uncertainty(*, uncertainty: DiscreteState, persistence: float) -> FloatND:
    """Symmetric two-state regime: stay with probability `persistence`."""
    stay = persistence
    return jnp.where(
        uncertainty == Uncertainty.low,
        jnp.array([stay, 1.0 - stay]),
        jnp.array([1.0 - stay, stay]),
    )


def next_lagged_uncertainty(uncertainty: DiscreteState) -> DiscreteState:
    """Keep the source volatility regime for next period's income law."""
    return uncertainty


def _validate_income_sigmas(*, sigma_low: float, sigma_high: float) -> None:
    values = {"low": sigma_low, "high": sigma_high}
    unusable = {
        category: value
        for category, value in values.items()
        if not isinstance(value, float | int)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0.0
    }
    if unusable:
        raise GridInitializationError(
            "a conditioned `sigma` needs finite positive values for every "
            f"category, but got {unusable}."
        )


def next_regime(*, age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.alive)


def wealth_constraint(
    *, consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


@functools.cache
def get_model(
    *,
    n_periods: int = 6,
    sigma_low: float = 0.05,
    sigma_high: float = 0.30,
    wealth_n_points: int = 20,
    consumption_n_points: int = 30,
    income_n_points: int = 7,
) -> Model:
    """Create the stochastic-volatility consumption-savings model."""
    _validate_income_sigmas(sigma_low=sigma_low, sigma_high=sigma_high)
    final_age_alive = 20 + (n_periods - 2) * 10
    alive_law = DeterministicTransition(func=next_regime)
    alive = Regime(
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=40.0, n_points=wealth_n_points),
            "uncertainty": DiscreteGrid(category_class=Uncertainty),
            "lagged_uncertainty": DiscreteGrid(category_class=Uncertainty),
        },
        state_transitions={
            "wealth": next_wealth,
            "uncertainty": StochasticTransition(func=next_uncertainty),
            "lagged_uncertainty": next_lagged_uncertainty,
        },
        actions={
            "consumption": LinSpacedGrid(
                start=0.1, stop=10.0, n_points=consumption_n_points
            )
        },
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
        shocks=("income_risk",),
    )
    dead = Regime(
        functions={"utility": lambda: 0.0},
    )
    alive_targets = {
        "dead": tuple(range(20, final_age_alive + 1, 10)),
        **(
            {"alive": tuple(range(20, final_age_alive, 10))}
            if tuple(range(20, final_age_alive, 10))
            else {}
        ),
    }
    return Model(
        edges={"alive": Transition(targets=alive_targets, law=alive_law)},
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        initial_nodes={20: "alive"},
        ages=AgeGrid(start=20, inclusive_stop=20 + (n_periods - 1) * 10, step="10Y"),
        fixed_params={"final_age_alive": final_age_alive},
        shocks={
            "income_risk": ShockGroup(
                shocks={
                    "income": Normal(
                        mu=0.0,
                        sigma=StateConditioned(
                            on="lagged_uncertainty",
                            by={"low": sigma_low, "high": sigma_high},
                        ),
                    )
                },
                nodes=Product(income=GaussHermite(n=income_n_points)),
            )
        },
    )


def get_params(
    *,
    interest_rate: float = 0.03,
    discount_factor: float = 0.95,
    persistence: float = 0.9,
) -> dict[str, UserParamsNode]:
    """Parameters for the stochastic-volatility model.

    The income sigmas are fixed in the law at construction; only the
    regime-persistence, interest rate, and discount factor are runtime params here.
    """
    return {
        "discount_factor": discount_factor,
        "alive": {
            "next_wealth": {"interest_rate": interest_rate},
            "next_uncertainty": {"persistence": persistence},
        },
    }
