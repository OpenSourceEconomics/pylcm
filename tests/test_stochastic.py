from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

import lcm
from lcm.model import Model
from tests.test_models.utils import get_model, get_params, get_regime

if TYPE_CHECKING:
    from lcm.typing import FloatND

# ======================================================================================
# Simulate
# ======================================================================================


def test_model_solve_and_simulate_with_stochastic_model():
    model = get_model("iskhakov_et_al_2017_stochastic", n_periods=3)

    res: pd.DataFrame = model.solve_and_simulate(
        params=get_params(),
        initial_states={
            "health": jnp.array([1, 1, 0, 0]),
            "partner": jnp.array([0, 0, 1, 0]),
            "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
        },
    )

    # This is derived from the partner transition in get_params.
    expected_next_partner = (
        (res.working.astype(bool) | ~res.partner.astype(bool)).astype(int).loc[:1]
    )

    pd.testing.assert_series_equal(
        res["partner"].loc[1:],
        expected_next_partner,
        check_index=False,
        check_names=False,
    )


# ======================================================================================
# Solve
# ======================================================================================


def test_model_solve_with_stochastic_model():
    model = get_model("iskhakov_et_al_2017_stochastic", n_periods=3)
    model.solve(params=get_params())


# ======================================================================================
# Comparison with deterministic results
# ======================================================================================


@pytest.fixture
def model_and_params():
    """Return a simple deterministic and stochastic model with parameters.

    TODO(@timmens): Add this to tests/test_models/stochastic.py.

    """

    # Define functions first
    @lcm.mark.stochastic
    def next_health_stochastic(health):
        pass

    def next_health_deterministic(health):
        return health

    # Get the base models and create modified versions
    base_regime = get_regime("iskhakov_et_al_2017_stochastic", n_periods=3)

    # Create deterministic model with modified function
    regime_deterministic = base_regime.replace(
        transitions={
            **base_regime.transitions,
            "next_health": next_health_deterministic,
        }
    )

    # Create stochastic model with modified function
    regime_stochastic = base_regime.replace(
        transitions={**base_regime.transitions, "next_health": next_health_stochastic}
    )

    params = get_params(
        beta=0.95,
        disutility_of_work=1.0,
        interest_rate=0.05,
        wage=10.0,
        health_transition=jnp.identity(2),
    )

    model_deterministic = Model(regime_deterministic, n_periods=base_regime.n_periods)
    model_stochastic = Model(regime_stochastic, n_periods=base_regime.n_periods)
    return model_deterministic, model_stochastic, params


def test_compare_deterministic_and_stochastic_results_value_function(model_and_params):
    """Test that the deterministic and stochastic models produce the same results."""
    model_deterministic, model_stochastic, params = model_and_params

    # ==================================================================================
    # Compare value function arrays
    # ==================================================================================
    solution_deterministic: dict[int, FloatND] = model_deterministic.solve(params)
    solution_stochastic: dict[int, FloatND] = model_stochastic.solve(params)

    for period in range(model_deterministic.n_periods):
        assert_array_almost_equal(
            solution_deterministic[period],
            solution_stochastic[period],
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

    simulation_deterministic = model_deterministic.simulate(
        params,
        V_arr_dict=solution_deterministic,
        initial_states=initial_states,
    )
    simulation_stochastic = model_stochastic.simulate(
        params,
        V_arr_dict=solution_stochastic,
        initial_states=initial_states,
    )
    pd.testing.assert_frame_equal(simulation_deterministic, simulation_stochastic)
