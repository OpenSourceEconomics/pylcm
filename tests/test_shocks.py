from __future__ import annotations

import pytest
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


@pytest.mark.parametrize(
    "distribution_type", ["uniform", "normal", "tauchen", "rouwenhorst"]
)
def test_model_with_shock(distribution_type):
    @lcm.mark.stochastic
    def next_state2(state2: DiscreteState, state2_transition: FloatND) -> FloatND:
        return state2_transition[state2]

    @lcm.mark.stochastic
    def next_state() -> None:
        pass

    def next_regime(period: int) -> ScalarInt:
        terminal = period >= 4 - 1  # is test_term in last period
        return jnp.where(terminal, RegimeId.test_regime_term, RegimeId.test_regime)

    def utility(
        state: ContinuousState,  # noqa: ARG001
        state2: DiscreteState,  # noqa: ARG001
        action: ContinuousAction,  # noqa: ARG001
    ) -> float:
        return 0.0

    def test_active(age):
        return age < 4

    def test_term_active(age):
        return age == 4

    @categorical
    class Discrete:
        a: int = 0
        b: int = 1

    @categorical
    class RegimeId:
        test_regime: int
        test_regime_term: int

    test_regime = Regime(
        active=test_active,
        states={
            "state": ShockGrid(
                n_points=5,
                distribution_type=distribution_type,
                shock_params={"rho": 0.975}
                if distribution_type in ["rouwenhorst", "tauchen"]
                else {},
            ),
            "state2": DiscreteGrid(Discrete),
        },
        actions={
            "action": LinSpacedGrid(start=1, stop=5, n_points=2),
        },
        utility=utility,
        transitions={
            "next_state": next_state,
            "next_state2": next_state2,
            "next_regime": next_regime,
        },
    )
    test_regime_term = Regime(
        terminal=True,
        active=test_term_active,
        utility=lambda: 0.0,
    )
    model = Model(
        regimes={"test_regime": test_regime, "test_regime_term": test_regime_term},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, stop=4, step="Y"),
        fixed_params={"state": {"rho": 0.975}}
        if distribution_type in ["rouwenhorst", "tauchen"]
        else {},
    )
    params = {
        "test_regime": {
            "discount_factor": 1.0,
            "next_state2": {"state2_transition": jnp.full((2, 2), fill_value=0.5)},
        },
        "test_regime_term": {
            "discount_factor": 1.0,
        },
    }

    model.solve_and_simulate(
        params=params,
        initial_regimes=["test_regime"],
        initial_states={"state": jnp.asarray([0]), "state2": jnp.asarray([0])},
    )
