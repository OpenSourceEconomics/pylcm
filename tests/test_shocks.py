from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from jax import numpy as jnp

import lcm
from lcm.ages import AgeGrid
from lcm.grids import DiscreteGrid, LinspaceGrid, ShockGrid
from lcm.model import Model
from lcm.regime import Regime

if TYPE_CHECKING:
    from lcm.typing import (
        ContinuousAction,
        ContinuousState,
        DiscreteState,
        FloatND,
        ScalarInt,
    )


@pytest.fixture
def params_for_shocks():
    return {
        "uniform": {
            "test": {
                "discount_factor": 1.0,
                "next_state": {"start": 0, "stop": 1},
                "next_state2": {"state2_transition": jnp.full((2, 2), fill_value=0.5)},
            },
            "test_term": {
                "discount_factor": 1.0,
            },
        },
        "normal": {
            "test": {
                "discount_factor": 1.0,
                "next_state": {"mu_eps": 0, "sigma_eps": 1, "n_std": 2},
            },
            "test_term": {
                "discount_factor": 1.0,
            },
        },
        "tauchen": {
            "test": {
                "discount_factor": 1.0,
                "next_state": {"rho": 0.8, "mu_eps": 0, "sigma_eps": 1, "n_std": 2},
            },
            "test_term": {
                "discount_factor": 1.0,
            },
        },
        "rouwenhorst": {
            "test": {
                "discount_factor": 1.0,
                "next_state": {"rho": 0.8, "mu_eps": 0, "sigma_eps": 1},
            },
            "test_term": {
                "discount_factor": 1.0,
            },
        },
    }


@pytest.mark.parametrize("distribution_type", ["uniform"])
def test_model_with_shock(distribution_type, params_for_shocks):
    @lcm.mark.stochastic(type=distribution_type)
    def next_state(state: ContinuousState) -> ContinuousState:
        pass

    @lcm.mark.stochastic
    def next_state2(state2: DiscreteState, state2_transition: FloatND) -> FloatND:
        return state2_transition[state2]

    def next_regime(period: int) -> ScalarInt:
        terminal = period >= 5 - 2  # is test_term in last period
        return jnp.where(terminal, RegimeId.test_term, RegimeId.test)

    def utility(
        state: ContinuousState,  # noqa: ARG001
        state2: DiscreteState,  # noqa: ARG001
        action: ContinuousAction,  # noqa: ARG001
    ) -> FloatND:
        return 0

    def test_active(age):
        return age < 4

    def test_term_active(age):
        return age == 4

    @dataclass
    class Discrete:
        test: int = 0
        test_term: int = 1

    @dataclass
    class RegimeId:
        test: int = 0
        test_term: int = 1

    test_regime = Regime(
        name="test",
        active=test_active,
        states={
            "state": ShockGrid(n_points=5, type=distribution_type),
            "state2": DiscreteGrid(Discrete),
        },
        actions={
            "action": LinspaceGrid(start=1, stop=5, n_points=2),
        },
        utility=utility,
        constraints={},
        transitions={
            "next_state": next_state,
            "next_state2": next_state2,
            "next_regime": next_regime,
        },
    )
    test_regime_term = Regime(
        name="test_term",
        terminal=True,
        active=test_term_active,
        utility=lambda: 0.0,
    )
    model = Model(
        regimes=[test_regime, test_regime_term],
        regime_id_cls=RegimeId,
        ages=AgeGrid(start=0, stop=4, step="Y"),
    )
    params = params_for_shocks[distribution_type]

    model.solve_and_simulate(
        params=params,
        initial_regimes=["test"],
        initial_states={"state": jnp.asarray([0]), "state2": jnp.asarray([0])},
    )
