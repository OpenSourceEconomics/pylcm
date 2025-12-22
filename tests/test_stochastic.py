from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

import lcm
from lcm import Model
from tests.test_models.stochastic import (
    RegimeId,
    dead,
    get_model,
    get_params,
    retired,
    working,
)

if TYPE_CHECKING:
    from lcm.typing import DiscreteState, FloatND

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
    res: pd.DataFrame = result.to_dataframe().query('regime == "working"')

    # Verify simulation produced expected columns and some rows
    assert "period" in res.columns
    assert "subject_id" in res.columns
    assert "partner" in res.columns
    assert "labor_supply" in res.columns
    assert len(res) > 0

    # Check that partner transition follows the transition matrix from get_params
    period_0 = res[res.period == 0].set_index("subject_id")
    period_1 = res[res.period == 1].set_index("subject_id")

    # Only test subjects present in both periods
    common_subjects = period_0.index.intersection(period_1.index)

    if len(common_subjects) > 0:
        # Create expected partner values based on period 0 state (using category labels)
        expected_partner = period_0.loc[common_subjects].apply(
            lambda row: "single"
            if (row["labor_supply"] == "work" and row["partner"] == "partnered")
            else "partnered",
            axis=1,
        )

        actual_partner = period_1.loc[common_subjects, "partner"]

        pd.testing.assert_series_equal(
            actual_partner,
            expected_partner,
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

    # Create deterministic model with modified function
    working_deterministic = working.replace(
        transitions={
            **working.transitions,
            "next_health": next_health_deterministic,
        },
        active=range(n_periods - 1),
    )
    retired_deterministic = retired.replace(
        transitions={
            **retired.transitions,
            "next_health": next_health_deterministic,
        },
        active=range(n_periods - 1),
    )

    # Create stochastic model with identity transition function
    working_stochastic = working.replace(
        transitions={
            **working.transitions,
            "next_health": next_health_stochastic,
        },
        active=range(n_periods - 1),
    )
    retired_stochastic = retired.replace(
        transitions={
            **retired.transitions,
            "next_health": next_health_stochastic,
        },
        active=range(n_periods - 1),
    )

    dead_updated = dead.replace(active=[n_periods - 1])

    model_deterministic = Model(
        [working_deterministic, retired_deterministic, dead_updated],
        n_periods=n_periods,
        regime_id_cls=RegimeId,
    )

    model_stochastic = Model(
        [working_stochastic, retired_stochastic, dead_updated],
        n_periods=n_periods,
        regime_id_cls=RegimeId,
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
    df_deterministic = simulation_deterministic.to_dataframe().query(
        'regime == "working"'
    )
    df_stochastic = simulation_stochastic.to_dataframe().query('regime == "working"')
    pd.testing.assert_frame_equal(
        df_deterministic.reset_index(drop=True),
        df_stochastic.reset_index(drop=True),
    )
