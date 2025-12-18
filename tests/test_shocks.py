from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from jax import numpy as jnp

import lcm
from lcm.grids import LinspaceGrid, ShockGrid
from lcm.model import Model
from lcm.regime import Regime

if TYPE_CHECKING:
    from lcm.typing import ContinuousAction, ContinuousState, FloatND


@pytest.fixture
def params_for_shocks():
    return {
        "uniform": {"test": {"beta": 1.0, "next_state": {"start": 0, "stop": 1}}},
        "normal": {
            "test": {
                "beta": 1.0,
                "next_state": {"mu_eps": 0, "sigma_eps": 1, "n_std": 2},
            }
        },
        "tauchen": {
            "test": {
                "beta": 1.0,
                "next_state": {"rho": 0.8, "mu_eps": 0, "sigma_eps": 1, "n_std": 2},
            }
        },
        "rouwenhorst": {
            "test": {
                "beta": 1.0,
                "next_state": {"rho": 0.8, "mu_eps": 0, "sigma_eps": 1},
            }
        },
    }


@pytest.mark.parametrize(
    "distribution_type", ["uniform", "normal", "tauchen", "rouwenhorst"]
)
def test_model_with_shock(distribution_type, params_for_shocks):
    @lcm.mark.stochastic(type=distribution_type)
    def next_state(state: ContinuousState) -> ContinuousState:
        pass

    def utility(state: ContinuousState, action: ContinuousAction) -> FloatND:  # noqa: ARG001
        return 0

    test_regime = Regime(
        name="test",
        states={
            "state": ShockGrid(n_points=5, type=distribution_type),
        },
        actions={
            "action": LinspaceGrid(start=1, stop=5, n_points=2),
        },
        utility=utility,
        constraints={},
        transitions={
            "next_state": next_state,
        },
    )
    model = Model(
        regimes=[test_regime],
        n_periods=5,
    )
    params = params_for_shocks[distribution_type]

    model.solve_and_simulate(
        params=params,
        initial_regimes=["test"],
        initial_states={"state": jnp.asarray([0])},
    )
