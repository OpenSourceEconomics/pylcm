"""Tests for the ages module (AgeGrid and step parsing)."""

from fractions import Fraction

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


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        ("Y", 1),
        ("2Y", 2),
        ("M", Fraction(1, 12)),
        ("3M", Fraction(1, 4)),
        ("Q", Fraction(1, 4)),
    ],
)
def test_parse_step_valid_formats(step, expected):
    assert parse_step(step) == expected


def test_parse_step_invalid():
    with pytest.raises(GridInitializationError, match="Invalid step format"):
        parse_step("X")


# ======================================================================================
# AgeGrid creation tests
# ======================================================================================


def test_age_grid_from_range():
    ages = AgeGrid(start=18, stop=21, step="Y")
    assert ages.n_periods == 4
    np.testing.assert_array_equal(ages.values, [18, 19, 20, 21])
    assert ages.step_size == 1.0


def test_age_grid_with_int_and_fraction_annual():
    """Test AgeGrid with int start and Fraction stop."""
    ages = AgeGrid(start=18, stop=Fraction(21, 1), step="Y")
    assert ages.n_periods == 4
    np.testing.assert_array_equal(ages.values, [18, 19, 20, 21])
    assert ages.step_size == 1.0
    assert type(ages.precise_step_size) is int
    assert isinstance(ages.precise_values, tuple)
    assert all(isinstance(age, int) for age in ages.precise_values)


def test_age_grid_with_int_and_fraction_quarterly():
    """Test AgeGrid with int start and Fraction stop."""
    ages = AgeGrid(start=20, stop=21 + Fraction(1, 4), step="Q")
    assert ages.n_periods == 6
    np.testing.assert_array_equal(ages.values, [20.0, 20.25, 20.5, 20.75, 21.0, 21.25])
    assert ages.step_size == 0.25
    assert type(ages.precise_step_size) is Fraction
    assert isinstance(ages.precise_values, tuple)
    assert all(isinstance(age, Fraction) for age in ages.precise_values)


def test_age_grid_from_values():
    ages = AgeGrid(precise_values=(18, 25, 35, 65))
    assert ages.n_periods == 4
    np.testing.assert_array_equal(ages.values, [18, 25, 35, 65])
    assert ages.step_size is None


def test_age_grid_period_to_age():
    ages = AgeGrid(start=18, stop=22, step="Y")
    assert ages.period_to_age(0) == 18.0
    assert ages.period_to_age(3) == 21.0


def test_age_grid_get_periods_where():
    ages = AgeGrid(start=18, stop=22, step="Y")
    periods = ages.get_periods_where(lambda age: age >= 21)
    assert periods == (3, 4)


# ======================================================================================
# AgeGrid validation tests
# ======================================================================================


def test_age_grid_no_params_raises():
    with pytest.raises(GridInitializationError):
        AgeGrid()


def test_age_grid_values_and_range_raises():
    with pytest.raises(GridInitializationError, match="Cannot specify both"):
        AgeGrid(start=18, stop=22, step="Y", precise_values=(18, 19))  # ty: ignore[no-matching-overload]


def test_age_grid_start_greater_than_stop_raises():
    with pytest.raises(GridInitializationError, match="must be less than"):
        AgeGrid(start=30, stop=20, step="Y")


def test_age_grid_non_increasing_values_raises():
    with pytest.raises(GridInitializationError, match="strictly increasing"):
        AgeGrid(precise_values=(18, 20, 19))


def test_age_grid_step_size_not_divisible_raises():
    """Test that step size must divide evenly into the range."""
    with pytest.raises(GridInitializationError, match="does not divide evenly"):
        AgeGrid(start=18, stop=21, step="2Y")


# ======================================================================================
# Integration test with non-yearly steps
# ======================================================================================


def test_model_with_quarterly_steps():
    """Test that solve/simulate works with quarterly (Q) step size."""
    # Quarterly steps: 18.0, 18.25, 18.5, 18.75, 19.0 (5 periods)
    ages = AgeGrid(start=18, stop=19, step="Q")
    final_age_alive = 18.75
    assert ages.n_periods == 5
    assert ages.step_size == 0.25

    model = Model(
        regimes={
            "working": working.replace(active=lambda age: age <= final_age_alive),
            "retired": retired.replace(active=lambda age: age <= final_age_alive),
            "dead": dead.replace(active=lambda age: age > final_age_alive),
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    params = {
        "discount_factor": 0.99,
        "working": {
            "utility": {"disutility_of_work": 0.5},
            "next_wealth": {"interest_rate": 0.01},
            "next_regime": {"final_age_alive": final_age_alive},
            "labor_income": {"wage": 5.0},
        },
        "retired": {
            "next_wealth": {"interest_rate": 0.01, "labor_income": 0.0},
            "next_regime": {"final_age_alive": final_age_alive},
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
    assert set(df["age"].unique()) == {18.0, 18.25, 18.5, 18.75, 19.0}

    # Check working/retired regimes only have ages < 19
    non_dead_df = df.query('regime != "dead"')
    assert all(non_dead_df["age"] < 19)
