from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

import lcm
from lcm import Model
from tests.test_models.stochastic import get_model, get_params

if TYPE_CHECKING:
    from lcm.typing import DiscreteState, FloatND

# ======================================================================================
# Simulate
# ======================================================================================


def test_model_solve_and_simulate_with_stochastic_model():
    model = get_model(n_periods=4)
    params = get_params(n_periods=4)

    res: pd.DataFrame = model.solve_and_simulate(
        params=params,
        initial_states={
            "health": jnp.array([1, 1, 0, 0]),
            "partner": jnp.array([0, 0, 1, 0]),
            "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
        },
        initial_regimes=["working"] * 4,
    )["working"]

    # This is derived from the partner transition in get_params.

    expected_next_partner = (
        (res.working.astype(bool) | ~res.partner.astype(bool)).astype(int).loc[:7]
    )

    pd.testing.assert_series_equal(
        res["partner"].loc[4:],
        expected_next_partner,
        check_index=False,
        check_names=False,
    )


# ======================================================================================
# Solve
# ======================================================================================


def test_model_solve_with_stochastic_model():
    model = get_model(n_periods=4)
    model.solve(params=get_params(n_periods=4))


# ======================================================================================
# Comparison with deterministic results
# ======================================================================================


@pytest.fixture
def models_and_params() -> tuple[Model, Model, dict]:
    """Return a deterministic and stochastic model with parameters.

    TODO(@timmens): Add this to tests/test_models/stochastic.py.

    """
    from tests.test_models.stochastic import RegimeID, dead, retired, working

    # Define functions first
    @lcm.mark.stochastic
    def next_health_stochastic(health: DiscreteState) -> FloatND:
        return jnp.identity(2)[health]

    def next_health_deterministic(health: DiscreteState) -> DiscreteState:
        return health

    # Create deterministic model with modified function
    working_deterministic = working.replace(
        transitions={
            **working.transitions,
            "next_health": next_health_deterministic,
        }
    )
    retired_deterministic = retired.replace(
        transitions={
            **retired.transitions,
            "next_health": next_health_deterministic,
        }
    )

    # Create stochastic model with identity transition function
    working_stochastic = working.replace(
        transitions={
            **working.transitions,
            "next_health": next_health_stochastic,
        }
    )
    retired_stochastic = retired.replace(
        transitions={
            **retired.transitions,
            "next_health": next_health_stochastic,
        }
    )

    model_deterministic = Model(
        [working_deterministic, retired_deterministic, dead],
        n_periods=4,
        regime_id_cls=RegimeID,
    )

    model_stochastic = Model(
        [working_stochastic, retired_stochastic, dead],
        n_periods=4,
        regime_id_cls=RegimeID,
    )

    return model_deterministic, model_stochastic, get_params(n_periods=4)


def test_compare_deterministic_and_stochastic_results_value_function(
    models_and_params: tuple[Model, Model, dict],
):
    """Test that the deterministic and stochastic models produce the same results."""
    model_deterministic, model_stochastic, params = models_and_params

    # ==================================================================================
    # Compare value function arrays
    # ==================================================================================
    solution_deterministic: dict[int, dict[str, FloatND]] = model_deterministic.solve(
        params
    )
    solution_stochastic: dict[int, dict[str, FloatND]] = model_stochastic.solve(params)

    for period in range(model_deterministic.n_periods - 1):
        assert_array_almost_equal(
            solution_deterministic[period]["working"],
            solution_stochastic[period]["working"],
            decimal=14,
        )

    # ==================================================================================
    # Compare simulation results
    # ==================================================================================
    initial_states = {
        "health": jnp.array([1, 1, 0, 0]),
        "partner": jnp.array([0, 0, 0, 0]),
        "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
    }
    initial_regimes = ["working"] * 4

    simulation_deterministic = model_deterministic.simulate(
        params,
        V_arr_dict=solution_deterministic,
        initial_states=initial_states,
        initial_regimes=initial_regimes,
    )
    simulation_stochastic = model_stochastic.simulate(
        params,
        V_arr_dict=solution_stochastic,
        initial_states=initial_states,
        initial_regimes=initial_regimes,
    )
    pd.testing.assert_frame_equal(
        simulation_deterministic["working"],
        simulation_stochastic["working"],
    )
