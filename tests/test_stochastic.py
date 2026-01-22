from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

import lcm
from lcm import AgeGrid, Model
from lcm.typing import DiscreteState, FloatND
from tests.test_models.stochastic import (
    RegimeId,
    dead,
    get_model,
    get_params,
    retired,
    working,
)

# ======================================================================================
# Simulate
# ======================================================================================


def test_model_solve_and_simulate_with_stochastic_model():
    model = get_model(n_periods=4)
    params = get_params(n_periods=4)

    result = model.solve_and_simulate(
        params=params,
        initial_states={
            "health": jnp.array([1, 1, 0, 0]),
            "partner": jnp.array([0, 0, 1, 0]),
            "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
        },
        initial_regimes=["working"] * 4,
    )
    df = result.to_dataframe().query('regime == "working"')

    # Verify expected columns
    required_cols = {"period", "subject_id", "partner", "labor_supply"}
    assert required_cols <= set(df.columns)
    assert len(df) > 0

    # Check partner transition follows expected pattern:
    # Partner becomes single if working and partnered, otherwise stays partnered
    period_0 = df.query("period == 0").set_index("subject_id")
    period_1 = df.query("period == 1").set_index("subject_id")
    common = period_0.index.intersection(period_1.index)

    if len(common) > 0:
        p0, p1 = period_0.loc[common], period_1.loc[common]
        should_be_single = (p0["labor_supply"] == "work") & (
            p0["partner"] == "partnered"
        )
        expected = should_be_single.map({True: "single", False: "partnered"})

        pd.testing.assert_series_equal(
            p1["partner"],
            expected,
            check_names=False,
            check_dtype=False,
            check_categorical=False,
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
def models_and_params() -> tuple[Model, Model, dict[str, Any]]:
    """Return a deterministic and stochastic model with parameters.

    TODO(@timmens): Add this to tests/test_models/stochastic.py.

    """

    @lcm.mark.stochastic
    def next_health_stochastic(health: DiscreteState) -> FloatND:
        return jnp.identity(2)[health]

    def next_health_deterministic(health: DiscreteState) -> DiscreteState:
        return health

    n_periods = 4
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")

    # Create deterministic model with modified function
    working_deterministic = working.replace(
        transitions={
            **working.transitions,
            "next_health": next_health_deterministic,
        },
        active=lambda age: age < n_periods - 1,
    )
    retired_deterministic = retired.replace(
        transitions={
            **retired.transitions,
            "next_health": next_health_deterministic,
        },
        active=lambda age: age < n_periods - 1,
    )

    # Create stochastic model with identity transition function
    working_stochastic = working.replace(
        transitions={
            **working.transitions,
            "next_health": next_health_stochastic,
        },
        active=lambda age: age < n_periods - 1,
    )
    retired_stochastic = retired.replace(
        transitions={
            **retired.transitions,
            "next_health": next_health_stochastic,
        },
        active=lambda age: age < n_periods - 1,
    )

    dead_updated = dead.replace(active=lambda age: age >= n_periods - 1)

    model_deterministic = Model(
        regimes={
            "working": working_deterministic,
            "retired": retired_deterministic,
            "dead": dead_updated,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    model_stochastic = Model(
        regimes={
            "working": working_stochastic,
            "retired": retired_stochastic,
            "dead": dead_updated,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    return model_deterministic, model_stochastic, get_params(n_periods=n_periods)


def test_compare_deterministic_and_stochastic_results_value_function(
    models_and_params: tuple[Model, Model, dict[str, Any]],
) -> None:
    """Test that the deterministic and stochastic models produce the same results."""
    model_deterministic, model_stochastic, params = models_and_params

    # ==================================================================================
    # Compare value function arrays
    # ==================================================================================
    solution_deterministic: Mapping[int, Mapping[str, FloatND]] = (
        model_deterministic.solve(params)
    )
    solution_stochastic: Mapping[int, Mapping[str, FloatND]] = model_stochastic.solve(
        params
    )

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
    df_deterministic = simulation_deterministic.to_dataframe().query(
        'regime == "working"'
    )
    df_stochastic = simulation_stochastic.to_dataframe().query('regime == "working"')
    pd.testing.assert_frame_equal(
        df_deterministic.reset_index(drop=True),
        df_stochastic.reset_index(drop=True),
    )
