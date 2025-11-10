from __future__ import annotations

import jax.numpy as jnp
import pandas as pd
import pytest
from pybaum import tree_map

from tests.test_models.utils import get_model


def test_internal_regime_has_required_attributes():
    """Test that Model has all required attributes after initialization."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=3)

    # Check all required attributes exist
    assert hasattr(model, "internal_regimes")
    assert hasattr(
        model.internal_regimes["iskhakov_et_al_2017_stripped_down"], "params_template"
    )
    assert hasattr(
        model.internal_regimes["iskhakov_et_al_2017_stripped_down"],
        "state_action_spaces",
    )
    assert hasattr(
        model.internal_regimes["iskhakov_et_al_2017_stripped_down"], "state_space_infos"
    )
    assert hasattr(
        model.internal_regimes["iskhakov_et_al_2017_stripped_down"],
        "max_Q_over_a_functions",
    )
    assert hasattr(
        model.internal_regimes["iskhakov_et_al_2017_stripped_down"],
        "argmax_and_max_Q_over_a_functions",
    )
    assert hasattr(
        model.internal_regimes["iskhakov_et_al_2017_stripped_down"],
        "next_state_simulation_function",
    )


def test_model_solve_method():
    """Test Model.solve() method works correctly."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=3)
    params = tree_map(lambda _: 0.2, model.params_template)

    # Test solve method
    solution = model.solve(params)

    assert isinstance(solution, dict)
    assert len(solution) == 3
    assert all(period in solution for period in range(3))

    # Check solution has correct structure
    for period in range(3):
        assert isinstance(
            solution[period]["iskhakov_et_al_2017_stripped_down"], jnp.ndarray
        )


def test_model_simulate_method():
    """Test Model.simulate() method works correctly."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=3)
    params = tree_map(lambda _: 0.2, model.params_template)

    # Solve first
    solution = model.solve(params)

    # Create initial states
    initial_states = {
        "iskhakov_et_al_2017_stripped_down": {
            "wealth": jnp.array([10.0, 20.0]),
            "lagged_retirement": jnp.array([0, 0]),
        }
    }

    # Test simulate method
    results = model.simulate(
        params=params,
        initial_states=initial_states,
        V_arr_dict=solution,
        initial_regimes=["iskhakov_et_al_2017_stripped_down"] * 2,
    )["iskhakov_et_al_2017_stripped_down"]

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0


def test_model_solve_and_simulate_method():
    """Test Model.solve_and_simulate() method works correctly."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=3)
    params = tree_map(lambda _: 0.2, model.params_template)

    # Create initial states
    initial_states = {
        "iskhakov_et_al_2017_stripped_down": {
            "wealth": jnp.array([10.0, 20.0]),
            "lagged_retirement": jnp.array([0, 0]),
        }
    }

    # Test combined method
    results = model.solve_and_simulate(
        params=params,
        initial_states=initial_states,
        initial_regimes=["iskhakov_et_al_2017_stripped_down"] * 2,
    )["iskhakov_et_al_2017_stripped_down"]

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0


@pytest.mark.skip
def test_model_params_template_matches_internal():
    """Test that params_template matches internal_regime.params."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=3)

    assert hasattr(model.internal_regimes, "params_template")


@pytest.mark.parametrize(
    "model_name",
    [
        "iskhakov_et_al_2017_stripped_down",
        "iskhakov_et_al_2017_discrete",
    ],
)
def test_model_initialization_all_configs(model_name):
    """Test Model initialization works for test configurations."""
    model = get_model(model_name, n_periods=2)

    # Should complete without error
    assert model.internal_regimes is not None
