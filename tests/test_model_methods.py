"""Test new Model solve/simulate methods.

This module tests the new Model class methods (solve, simulate, solve_and_simulate)
that were added to replace the get_lcm_function approach. It includes:

1. Tests for the new Model methods themselves
2. Backward compatibility tests to ensure get_lcm_function still works
3. Equivalence tests to ensure both approaches give identical results

The backward compatibility tests
(test_model_solve_method_equivalent_to_get_lcm_function) explicitly test that the
deprecated get_lcm_function gives identical results to the new Model methods.
"""

from __future__ import annotations

import re

import jax.numpy as jnp
import pandas as pd
import pytest
from pybaum import tree_equal, tree_map

from lcm.entry_point import get_lcm_function
from tests.test_models import get_model


def test_model_has_required_attributes():
    """Test that Model has all required attributes after initialization."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=3)

    # Check all required attributes exist
    assert hasattr(model, "internal_model")
    assert hasattr(model, "params_template")
    assert hasattr(model, "state_action_spaces")
    assert hasattr(model, "state_space_infos")
    assert hasattr(model, "max_Q_over_a_functions")
    assert hasattr(model, "argmax_and_max_Q_over_a_functions")

    # Check they have correct types and lengths
    assert len(model.state_action_spaces) == 3
    assert len(model.max_Q_over_a_functions) == 3
    assert len(model.argmax_and_max_Q_over_a_functions) == 3


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
        assert isinstance(solution[period], jnp.ndarray)


def test_model_simulate_method():
    """Test Model.simulate() method works correctly."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=3)
    params = tree_map(lambda _: 0.2, model.params_template)

    # Solve first
    solution = model.solve(params)

    # Create initial states
    initial_states = {
        "wealth": jnp.array([10.0, 20.0]),
        "lagged_retirement": jnp.array([0, 0]),
    }

    # Test simulate method
    results = model.simulate(
        params=params,
        initial_states=initial_states,
        V_arr_dict=solution,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0


def test_model_solve_and_simulate_method():
    """Test Model.solve_and_simulate() method works correctly."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=3)
    params = tree_map(lambda _: 0.2, model.params_template)

    initial_states = {
        "wealth": jnp.array([10.0, 20.0]),
        "lagged_retirement": jnp.array([0, 0]),
    }

    # Test combined method
    results = model.solve_and_simulate(
        params=params,
        initial_states=initial_states,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0


def test_model_params_template_matches_internal():
    """Test that params_template matches internal_model.params."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=3)

    assert model.params_template == model.internal_model.params


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
    assert model.internal_model is not None
    assert len(model.state_action_spaces) == 2
    assert len(model.max_Q_over_a_functions) == 2


def test_model_solve_method_equivalent_to_get_lcm_function():
    """Test new Model.solve() gives same results as get_lcm_function."""
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=3)
    params = tree_map(lambda _: 0.2, model.params_template)

    # Old approach
    warn_msg = re.escape("get_lcm_function() is deprecated.")
    with pytest.warns(DeprecationWarning, match=warn_msg):
        solve_old, _ = get_lcm_function(model=model, targets="solve")
    solution_old = solve_old(params)

    # New approach
    solution_new = model.solve(params)

    # Should give identical results
    assert tree_equal(solution_old, solution_new)
