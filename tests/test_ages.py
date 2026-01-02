"""Tests for the ages module (AgeGrid and step parsing)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import Model
from lcm.ages import AgeGrid, parse_step
from lcm.exceptions import GridInitializationError
from tests.test_models.deterministic.base import (
    RegimeId,
    dead,
    retired,
    working,
)

# ======================================================================================
# parse_step tests
# ======================================================================================


def test_parse_step_valid_formats():
    assert parse_step("Y") == 1.0
    assert parse_step("2Y") == 2.0
    assert parse_step("M") == pytest.approx(1 / 12)
    assert parse_step("3M") == pytest.approx(0.25)
    assert parse_step("Q") == 0.25


def test_parse_step_invalid():
    with pytest.raises(ValueError, match="Invalid step format"):
        parse_step("X")


# ======================================================================================
# AgeGrid creation tests
# ======================================================================================


def test_age_grid_from_range():
    ages = AgeGrid(start=18, stop=22, step="Y")
    assert ages.n_periods == 4
    np.testing.assert_array_equal(ages.ages, [18, 19, 20, 21])
    assert ages.step_size == 1.0


def test_age_grid_from_values():
    ages = AgeGrid(values=(18, 25, 35, 65))
    assert ages.n_periods == 4
    np.testing.assert_array_equal(ages.ages, [18, 25, 35, 65])
    assert ages.step_size is None


def test_age_grid_period_to_age():
    ages = AgeGrid(start=18, stop=22, step="Y")
    assert ages.period_to_age(0) == 18.0
    assert ages.period_to_age(3) == 21.0


def test_age_grid_get_periods_where():
    ages = AgeGrid(start=18, stop=23, step="Y")  # [18, 19, 20, 21, 22]
    periods = ages.get_periods_where(lambda age: age >= 21)
    assert periods == [3, 4]


# ======================================================================================
# AgeGrid validation tests
# ======================================================================================


def test_age_grid_no_params_raises():
    with pytest.raises(GridInitializationError):
        AgeGrid()


def test_age_grid_values_and_range_raises():
    with pytest.raises(GridInitializationError, match="Cannot specify both"):
        AgeGrid(start=18, stop=22, step="Y", values=(18, 19))


def test_age_grid_start_greater_than_stop_raises():
    with pytest.raises(GridInitializationError, match="must be less than"):
        AgeGrid(start=30, stop=20, step="Y")


def test_age_grid_non_increasing_values_raises():
    with pytest.raises(GridInitializationError, match="strictly increasing"):
        AgeGrid(values=(18, 20, 19))


# ======================================================================================
# Integration test with non-yearly steps
# ======================================================================================


def test_model_with_quarterly_steps():
    """Test that solve/simulate works with quarterly (Q) step size."""
    # Quarterly steps: 18.0, 18.25, 18.5, 18.75 (4 periods)
    ages = AgeGrid(start=18, stop=19, step="Q")
    assert ages.n_periods == 4
    assert ages.step_size == 0.25

    model = Model(
        [
            working.replace(active=lambda age: age < 18.75),
            retired.replace(active=lambda age: age < 18.75),
            dead.replace(active=lambda age: age >= 18.75),
        ],
        ages=ages,
        regime_id_cls=RegimeId,
    )

    params = {
        "working": {
            "discount_factor": 0.99,
            "utility": {"disutility_of_work": 0.5},
            "next_wealth": {"interest_rate": 0.01},
            "next_regime": {"final_age": 18.5},  # Transition to dead at age 18.5
            "borrowing_constraint": {},
            "labor_income": {"wage": 5.0},
        },
        "retired": {
            "discount_factor": 0.99,
            "utility": {},
            "next_wealth": {"interest_rate": 0.01, "labor_income": 0.0},
            "next_regime": {"final_age": 18.5},
            "borrowing_constraint": {},
        },
        "dead": {},
    }

    # Solve and simulate
    result = model.solve_and_simulate(
        params=params,
        initial_states={"wealth": jnp.array([50.0, 100.0, 150.0])},
        initial_regimes=["working"] * 3,
    )

    df = result.to_dataframe()

    # Check that age column has quarterly values
    assert set(df["age"].unique()) == {18.0, 18.25, 18.5, 18.75}

    # Check working/retired regimes only have ages < 18.75
    non_dead_df = df.query('regime != "dead"')
    assert all(non_dead_df["age"] < 18.75)
