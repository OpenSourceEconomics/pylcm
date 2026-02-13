from types import MappingProxyType
from typing import Literal

from jax import numpy as jnp

import lcm
from lcm.ages import AgeGrid
from lcm.grids import DiscreteGrid, LinSpacedGrid, ShockGrid, categorical
from lcm.model import Model
from lcm.regime import Regime
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)


def get_model(
    n_periods: int,
    distribution_type: Literal["uniform", "normal", "tauchen", "rouwenhorst"],
):
    @lcm.mark.stochastic
    def next_health(health: DiscreteState, health_transition: FloatND) -> FloatND:
        return health_transition[health]

    @lcm.mark.stochastic
    def next_income() -> None:
        pass

    def next_wealth(consumption: ContinuousAction, wealth: ContinuousState) -> FloatND:
        return wealth - consumption

    def next_regime(period: int) -> ScalarInt:
        terminal = period >= n_periods - 1  # is test_term in last period
        return jnp.where(terminal, RegimeId.test_regime_term, RegimeId.test_regime)

    def wealth_constraint(
        wealth: ContinuousState, income: ContinuousState, consumption: ContinuousAction
    ):
        return wealth - consumption + jnp.exp(income) >= 0

    def utility(
        wealth: ContinuousState,  # noqa: ARG001
        income: ContinuousState,  # noqa: ARG001
        health: DiscreteState,
        consumption: ContinuousAction,
    ) -> FloatND:
        return jnp.log(consumption) * (1.0 - (1.0 - health) * 0.3)

    def test_active(age):
        return age < n_periods

    def test_term_active(age):
        return age == n_periods

    @categorical
    class Health:
        bad: int = 0
        good: int = 1

    @categorical
    class RegimeId:
        test_regime: int
        test_regime_term: int

    test_regime = Regime(
        active=test_active,
        states={
            "wealth": LinSpacedGrid(start=1, stop=5, n_points=5),
            "income": ShockGrid(
                n_points=5,
                distribution_type=distribution_type,
                shock_params=MappingProxyType({"rho": 0.975})
                if distribution_type in ["rouwenhorst", "tauchen"]
                else MappingProxyType({}),
            ),
            "health": DiscreteGrid(Health),
        },
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=2, n_points=4),
        },
        transitions={
            "next_wealth": next_wealth,
            "next_income": next_income,
            "next_health": next_health,
            "next_regime": next_regime,
        },
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )
    test_regime_term = Regime(
        terminal=True,
        active=test_term_active,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"test_regime": test_regime, "test_regime_term": test_regime_term},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, stop=n_periods, step="Y"),
        fixed_params={"income": {"rho": 0.975}}
        if distribution_type in ["rouwenhorst", "tauchen"]
        else {},
    )


def get_params():
    return {
        "test_regime": {
            "discount_factor": 1,
            "next_health": {"health_transition": jnp.full((2, 2), fill_value=0.5)},
        },
        "test_regime_term": {},
    }
