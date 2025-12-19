from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

import lcm
from lcm import Model
from tests.test_models.stochastic import (
    RegimeID,
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

    res: pd.DataFrame = model.solve_and_simulate(
        params=params,
        initial_states={
            "health": jnp.array([1, 1, 0, 0]),
            "partner": jnp.array([0, 0, 1, 0]),
            "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
        },
        initial_regimes=["working"] * 4,
    )["working"]

    # Verify simulation produced expected columns and some rows
    assert "period" in res.columns
    assert "subject_id" in res.columns
    assert "partner" in res.columns
    assert "labor_supply" in res.columns
    assert len(res) > 0

    # Check that partner transition follows the transition matrix from get_params:
    # Working (labor=0) + single (partner=0) -> partnered (1)
    # Working (labor=0) + partnered (partner=1) -> single (0)
    # Not working (labor=1) + single (partner=0) -> partnered (1)
    # Not working (labor=1) + partnered (partner=1) -> partnered (1)
    period_0 = res[res.period == 0].set_index("subject_id")
    period_1 = res[res.period == 1].set_index("subject_id")

    # Only test subjects present in both periods
    common_subjects = period_0.index.intersection(period_1.index)
    if len(common_subjects) > 0:
        for subj in common_subjects:
            is_working_p0 = period_0.loc[subj, "labor_supply"] == 0
            is_partnered_p0 = period_0.loc[subj, "partner"] == 1

            if is_working_p0 and is_partnered_p0:
                expected_partner_p1 = 0  # Working + partnered -> single
            elif is_working_p0 and not is_partnered_p0:
                expected_partner_p1 = 1  # Working + single -> partnered
            elif not is_working_p0 and is_partnered_p0:
                expected_partner_p1 = 1  # Not working + partnered -> partnered
            else:  # not working and single
                expected_partner_p1 = 1  # Not working + single -> partnered

            # Partner status at period 1 should match expected
            assert period_1.loc[subj, "partner"] == expected_partner_p1, (
                f"Subject {subj}: expected partner={expected_partner_p1} at period 1, "
                f"got {period_1.loc[subj, 'partner']} "
                f"(was working={is_working_p0}, partnered={is_partnered_p0})"
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
    pd.testing.assert_frame_equal(
        simulation_deterministic["working"],
        simulation_stochastic["working"],
    )
