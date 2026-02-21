from typing import Literal

from jax import numpy as jnp

import lcm
from lcm.ages import AgeGrid
from lcm.grids import DiscreteGrid, LinSpacedGrid, categorical
from lcm.model import Model
from lcm.regime import Regime
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)

_SHOCK_GRID_CLASSES = {
    "uniform": lcm.shocks.iid.Uniform,
    "normal": lcm.shocks.iid.Normal,
    "tauchen": lcm.shocks.ar1.Tauchen,
    "rouwenhorst": lcm.shocks.ar1.Rouwenhorst,
}


def get_model(
    n_periods: int,
    distribution_type: Literal["uniform", "normal", "tauchen", "rouwenhorst"],
):
    @lcm.mark.stochastic
    def next_health(health: DiscreteState, health_transition: FloatND) -> FloatND:
        return health_transition[health]

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
            "wealth": LinSpacedGrid(
                start=1, stop=5, n_points=5, transition=next_wealth
            ),
            "income": _SHOCK_GRID_CLASSES[distribution_type](n_points=5),
            "health": DiscreteGrid(Health, transition=next_health),
        },
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=2, n_points=4),
        },
        transition=next_regime,
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )
    test_regime_term = Regime(
        transition=None,
        active=test_term_active,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"test_regime": test_regime, "test_regime_term": test_regime_term},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, stop=n_periods, step="Y"),
    )


_SHOCK_PARAMS: dict[str, dict[str, float]] = {
    "uniform": {"start": 0.0, "stop": 1.0},
    "normal": {"mu": 0.0, "sigma": 1.0, "n_std": 3.0},
    "tauchen": {"rho": 0.975, "sigma": 1.0, "mu": 0.0, "n_std": 2},
    "rouwenhorst": {"rho": 0.975, "sigma": 1.0, "mu": 0.0},
}


def get_params(
    distribution_type: Literal[
        "uniform", "normal", "tauchen", "rouwenhorst"
    ] = "tauchen",
):
    return {
        "test_regime": {
            "discount_factor": 1,
            "next_health": {"health_transition": jnp.full((2, 2), fill_value=0.5)},
            "income": _SHOCK_PARAMS[distribution_type],
        },
        "test_regime_term": {},
    }
