"""Tests for _compute_starting_periods."""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid
from lcm.simulation.simulate import _compute_starting_periods


def test_all_subjects_start_at_first_age():
    ages = AgeGrid(start=25, stop=75, step="Y")
    initial_ages = jnp.array([25.0, 25.0, 25.0])
    result = _compute_starting_periods(initial_ages=initial_ages, ages=ages)
    expected = jnp.array([0, 0, 0])
    assert jnp.array_equal(result, expected)


def test_all_subjects_start_at_last_age():
    ages = AgeGrid(start=25, stop=75, step="Y")
    initial_ages = jnp.array([75.0, 75.0])
    result = _compute_starting_periods(initial_ages=initial_ages, ages=ages)
    n_periods = len(ages.values) - 1
    expected = jnp.full(2, n_periods)
    assert jnp.array_equal(result, expected)


def test_heterogeneous_ages():
    ages = AgeGrid(start=25, stop=30, step="Y")
    initial_ages = jnp.array([25.0, 27.0, 30.0])
    result = _compute_starting_periods(initial_ages=initial_ages, ages=ages)
    expected = jnp.array([0, 2, 5])
    assert jnp.array_equal(result, expected)


def test_single_subject():
    ages = AgeGrid(start=25, stop=75, step="Y")
    initial_ages = jnp.array([40.0])
    result = _compute_starting_periods(initial_ages=initial_ages, ages=ages)
    expected = jnp.array([15])
    assert jnp.array_equal(result, expected)


def test_sub_annual_grid():
    ages = AgeGrid(start=25, stop=26, step="Q")
    # Grid values: 25.0, 25.25, 25.5, 25.75, 26.0
    initial_ages = jnp.array([25.0, 25.25, 25.5, 25.75, 26.0])
    result = _compute_starting_periods(initial_ages=initial_ages, ages=ages)
    expected = jnp.array([0, 1, 2, 3, 4])
    assert jnp.array_equal(result, expected)


def test_irregular_grid():
    ages = AgeGrid(exact_values=[0, 1, 5, 10, 20])
    initial_ages = jnp.array([0.0, 5.0, 20.0])
    result = _compute_starting_periods(initial_ages=initial_ages, ages=ages)
    expected = jnp.array([0, 2, 4])
    assert jnp.array_equal(result, expected)


def test_multi_year_steps():
    ages = AgeGrid(start=40, stop=60, step="10Y")
    # Grid values: 40, 50, 60
    initial_ages = jnp.array([40.0, 50.0, 60.0])
    result = _compute_starting_periods(initial_ages=initial_ages, ages=ages)
    expected = jnp.array([0, 1, 2])
    assert jnp.array_equal(result, expected)


def test_age_below_grid_minimum():
    ages = AgeGrid(start=25, stop=75, step="Y")
    initial_ages = jnp.array([20.0])
    with pytest.raises(ValueError, match="not valid age grid points"):
        _compute_starting_periods(initial_ages=initial_ages, ages=ages)


def test_age_above_grid_maximum():
    ages = AgeGrid(start=25, stop=75, step="Y")
    initial_ages = jnp.array([80.0])
    with pytest.raises(ValueError, match="not valid age grid points"):
        _compute_starting_periods(initial_ages=initial_ages, ages=ages)


def test_age_between_grid_points():
    ages = AgeGrid(start=25, stop=75, step="Y")
    initial_ages = jnp.array([26.5])
    with pytest.raises(ValueError, match="not valid age grid points"):
        _compute_starting_periods(initial_ages=initial_ages, ages=ages)


def test_mix_of_valid_and_invalid_ages():
    ages = AgeGrid(start=25, stop=30, step="Y")
    initial_ages = jnp.array([25.0, 26.5, 30.0, 80.0])
    with pytest.raises(ValueError, match="not valid age grid points") as exc_info:
        _compute_starting_periods(initial_ages=initial_ages, ages=ages)
    msg = str(exc_info.value)
    assert "26.5" in msg
    assert "80.0" in msg
    # Valid ages should not appear in the error message
    assert "25.0" not in msg


def test_empty_array():
    ages = AgeGrid(start=25, stop=75, step="Y")
    initial_ages = jnp.array([], dtype=jnp.float32)
    result = _compute_starting_periods(initial_ages=initial_ages, ages=ages)
    assert result.shape == (0,)


def test_sub_annual_monthly_grid():
    ages = AgeGrid(start=25, stop=26, step="M")
    # Grid has 13 monthly points: 25.0, 25.0833..., ..., 26.0
    initial_ages = jnp.array([25.0, 26.0])
    result = _compute_starting_periods(initial_ages=initial_ages, ages=ages)
    expected = jnp.array([0, 12])
    assert jnp.array_equal(result, expected)
