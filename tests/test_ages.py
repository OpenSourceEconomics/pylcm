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
    retirement,
    working_life,
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
    assert type(ages.exact_step_size) is int
    assert isinstance(ages.exact_values, tuple)
    assert all(isinstance(age, int) for age in ages.exact_values)


def test_age_grid_with_int_and_fraction_quarterly():
    """Test AgeGrid with int start and Fraction stop."""
    ages = AgeGrid(start=20, stop=21 + Fraction(1, 4), step="Q")
    assert ages.n_periods == 6
    np.testing.assert_array_equal(ages.values, [20.0, 20.25, 20.5, 20.75, 21.0, 21.25])
    assert ages.step_size == 0.25
    assert type(ages.exact_step_size) is Fraction
    assert isinstance(ages.exact_values, tuple)
    assert all(isinstance(age, Fraction) for age in ages.exact_values)


def test_age_grid_from_values():
    ages = AgeGrid(exact_values=(18, 25, 35, 65))
    assert ages.n_periods == 4
    np.testing.assert_array_equal(ages.values, [18, 25, 35, 65])
    assert ages.step_size is None


def test_age_grid_period_to_age():
    ages = AgeGrid(start=18, stop=22, step="Y")
    assert ages.period_to_age(0) == 18
    assert type(ages.period_to_age(0)) is int
    assert ages.period_to_age(3) == 21


def test_age_grid_get_periods_where():
    ages = AgeGrid(start=18, stop=22, step="Y")
    periods = ages.get_periods_where(lambda age: age >= 21)
    assert periods == (3, 4)


# ======================================================================================
# AgeGrid validation tests
# ======================================================================================


def test_age_grid_no_params_raises():
    with pytest.raises(GridInitializationError):
        AgeGrid()  # ty: ignore[no-matching-overload]


def test_age_grid_values_and_range_raises():
    with pytest.raises(GridInitializationError, match="Cannot specify both"):
        AgeGrid(start=18, stop=22, step="Y", exact_values=(18, 19))  # ty: ignore[no-matching-overload]


def test_age_grid_start_greater_than_stop_raises():
    with pytest.raises(GridInitializationError, match="must be less than"):
        AgeGrid(start=30, stop=20, step="Y")


def test_age_grid_non_increasing_values_raises():
    with pytest.raises(GridInitializationError, match="strictly increasing"):
        AgeGrid(exact_values=(18, 20, 19))


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
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive
            ),
            "retirement": retirement.replace(active=lambda age: age <= final_age_alive),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    params = {
        "discount_factor": 0.99,
        "final_age_alive": final_age_alive,
        "working_life": {
            "utility": {"disutility_of_work": 0.5},
            "next_wealth": {"interest_rate": 0.01},
            "labor_income": {"wage": 5.0},
        },
        "retirement": {
            "next_wealth": {"interest_rate": 0.01, "labor_income": 0.0},
        },
    }

    # Solve and simulate
    result = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.array([50.0, 100.0, 150.0]),
            "age": jnp.array([18.0, 18.0, 18.0]),
            "regime": jnp.array([RegimeId.working_life] * 3),
        },
        period_to_regime_to_V_arr=None,
    )

    df = result.to_dataframe()

    # Check that age column has quarterly values
    assert set(df["age"].unique()) == {18.0, 18.25, 18.5, 18.75, 19.0}

    # Check working/retired regimes only have ages < 19
    non_dead_df = df.query('regime != "dead"')
    assert all(non_dead_df["age"] < 19)


# ======================================================================================
# AgeGrid.age_to_period tests
# ======================================================================================


def test_age_grid_age_to_period():
    ages = AgeGrid(start=18, stop=22, step="Y")
    assert ages.age_to_period(18) == 0
    assert ages.age_to_period(20) == 2
    assert ages.age_to_period(22) == 4


def test_age_grid_age_to_period_invalid():
    ages = AgeGrid(start=18, stop=22, step="Y")
    with pytest.raises(ValueError, match="not a valid grid point"):
        ages.age_to_period(17)


# ======================================================================================
# Integer auto-detection tests
# ======================================================================================


def test_annual_step_produces_int():
    ages = AgeGrid(start=18, stop=21, step="Y")
    assert ages.is_integer
    assert ages.values.dtype == jnp.int32
    np.testing.assert_array_equal(ages.values, [18, 19, 20, 21])
    assert all(isinstance(v, int) for v in ages.exact_values)


def test_multiannual_step_produces_int():
    ages = AgeGrid(start=40, stop=70, step="10Y")
    assert ages.is_integer
    assert ages.values.dtype == jnp.int32
    np.testing.assert_array_equal(ages.values, [40, 50, 60, 70])


def test_integer_exact_values_produce_int():
    ages = AgeGrid(exact_values=(18, 25, 35, 65))
    assert ages.is_integer
    assert ages.values.dtype == jnp.int32
    np.testing.assert_array_equal(ages.values, [18, 25, 35, 65])
    assert ages.step_size is None


def test_integer_fraction_exact_values_produce_int():
    """Fraction(18, 1) is integer-valued and should produce int."""
    ages = AgeGrid(start=Fraction(18, 1), stop=Fraction(21, 1), step="Y")
    assert ages.is_integer
    assert ages.values.dtype == jnp.int32
    assert all(isinstance(v, int) for v in ages.exact_values)


def test_quarterly_step_produces_float():
    ages = AgeGrid(start=20, stop=21, step="Q")
    assert not ages.is_integer
    assert jnp.issubdtype(ages.values.dtype, jnp.floating)


def test_monthly_step_produces_float():
    ages = AgeGrid(start=20, stop=Fraction(20 * 12 + 1, 12), step="M")
    assert not ages.is_integer
    assert jnp.issubdtype(ages.values.dtype, jnp.floating)


def test_fractional_exact_values_produce_float():
    ages = AgeGrid(exact_values=(18, Fraction(51, 2)))
    assert not ages.is_integer
    assert jnp.issubdtype(ages.values.dtype, jnp.floating)


def test_integer_period_to_age_returns_int():
    ages = AgeGrid(start=18, stop=21, step="Y")
    result = ages.period_to_age(0)
    assert result == 18
    assert type(result) is int


def test_float_period_to_age_returns_float():
    ages = AgeGrid(start=20, stop=21, step="Q")
    result = ages.period_to_age(0)
    assert result == 20.0
    assert type(result) is float


def test_integer_age_to_period():
    ages = AgeGrid(start=40, stop=70, step="10Y")
    assert ages.age_to_period(40) == 0
    assert ages.age_to_period(60) == 2
    assert ages.age_to_period(70) == 3


def test_integer_get_periods_where_passes_int():
    ages = AgeGrid(start=18, stop=22, step="Y")
    received_types = []
    periods = ages.get_periods_where(
        lambda age: (received_types.append(type(age)), age >= 21)[1]
    )
    assert periods == (3, 4)
    assert all(t is int for t in received_types)


# ======================================================================================
# Integer age integration test
# ======================================================================================


def test_model_with_integer_ages():
    """Test that solve/simulate works with integer ages."""
    ages = AgeGrid(start=40, stop=70, step="10Y")
    last_age = ages.exact_values[-1]

    model = Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age, la=last_age: age < la
            ),
            "retirement": retirement.replace(active=lambda age, la=last_age: age < la),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )

    params = {
        "discount_factor": 0.95,
        "final_age_alive": 60,
        "working_life": {
            "utility": {"disutility_of_work": 0.5},
            "next_wealth": {"interest_rate": 0.05},
            "labor_income": {"wage": 10.0},
        },
        "retirement": {
            "next_wealth": {"interest_rate": 0.05, "labor_income": 0.0},
        },
    }

    result = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.array([50.0, 100.0, 150.0]),
            "age": jnp.array([40, 40, 40]),
            "regime": jnp.array([RegimeId.working_life] * 3),
        },
        period_to_regime_to_V_arr=None,
    )

    df = result.to_dataframe()

    # Age column should have integer values
    assert set(df["age"].unique()) == {40, 50, 60, 70}
