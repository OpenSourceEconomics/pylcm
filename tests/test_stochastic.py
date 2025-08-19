from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pandas as pd
import pytest

import lcm
from lcm.entry_point import (
    get_lcm_function,
)
from tests.test_models import get_model_config, get_params

if TYPE_CHECKING:
    from lcm.typing import FloatND

# ======================================================================================
# Simulate
# ======================================================================================


def test_get_lcm_function_with_simulate_target():
    simulate_model, _ = get_lcm_function(
        model=get_model_config("iskhakov_et_al_2017_stochastic", n_periods=3),
        targets="solve_and_simulate",
    )

    res: pd.DataFrame = simulate_model(  # type: ignore[assignment]
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


def test_get_lcm_function_with_solve_target():
    solve_model, _ = get_lcm_function(
        model=get_model_config("iskhakov_et_al_2017_stochastic", n_periods=3),
        targets="solve",
    )
    solve_model(params=get_params())


# ======================================================================================
# Comparison with deterministic results
# ======================================================================================


@pytest.fixture
def model_and_params():
    """Return a simple deterministic and stochastic model with parameters.

    TODO(@timmens): Add this to tests/test_models/stochastic.py.

    """
    model_deterministic = get_model_config(
        "iskhakov_et_al_2017_stochastic", n_periods=3
    )
    model_stochastic = get_model_config("iskhakov_et_al_2017_stochastic", n_periods=3)

    # Overwrite health transition with simple stochastic version and deterministic one
    # ==================================================================================
    @lcm.mark.stochastic
    def next_health_stochastic(health):
        pass

    def next_health_deterministic(health):
        return health

    model_deterministic.functions["next_health"] = next_health_deterministic
    model_stochastic.functions["next_health"] = next_health_stochastic

    params = get_params(
        beta=0.95,
        disutility_of_work=1.0,
        interest_rate=0.05,
        wage=10.0,
        health_transition=jnp.identity(2),
    )

    return model_deterministic, model_stochastic, params


def test_compare_deterministic_and_stochastic_results_value_function(model_and_params):
    """Test that the deterministic and stochastic models produce the same results."""
    model_deterministic, model_stochastic, params = model_and_params

    # ==================================================================================
    # Compare value function arrays
    # ==================================================================================
    solve_model_deterministic, _ = get_lcm_function(
        model=model_deterministic,
        targets="solve",
    )
    solve_model_stochastic, _ = get_lcm_function(
        model=model_stochastic,
        targets="solve",
    )

    solution_deterministic: dict[int, FloatND] = solve_model_deterministic(params)  # type: ignore[assignment]
    solution_stochastic: dict[int, FloatND] = solve_model_stochastic(params)  # type: ignore[assignment]

    for period in range(model_deterministic.n_periods):
        assert jnp.array_equal(
            solution_deterministic[period],
            solution_stochastic[period],
            equal_nan=True,
        )

    # ==================================================================================
    # Compare simulation results
    # ==================================================================================
    simulate_model_deterministic, _ = get_lcm_function(
        model=model_deterministic,
        targets="simulate",
    )
    simulate_model_stochastic, _ = get_lcm_function(
        model=model_stochastic,
        targets="simulate",
    )

    initial_states = {
        "health": jnp.array([1, 1, 0, 0]),
        "partner": jnp.array([0, 0, 0, 0]),
        "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
    }

    simulation_deterministic = simulate_model_deterministic(
        params,
        V_arr_dict=solution_deterministic,
        initial_states=initial_states,
    )
    simulation_stochastic = simulate_model_stochastic(
        params,
        V_arr_dict=solution_stochastic,
        initial_states=initial_states,
    )
    pd.testing.assert_frame_equal(simulation_deterministic, simulation_stochastic)  # type: ignore[arg-type]
