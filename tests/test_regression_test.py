from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from jax import numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from lcm import (
    LinspaceGrid,
    LogspaceGrid,
)
from lcm._config import TEST_DATA
from tests.test_models.deterministic.regression import get_model, get_params

if TYPE_CHECKING:
    from lcm.grids import ContinuousGrid
    from lcm.typing import FloatND


def test_regression_test():
    """Test that the output of lcm does not change."""
    # Load expected output
    expected_simulate = pd.read_pickle(
        TEST_DATA / "regression_tests" / "simulation.pkl"
    )
    expected_solve = pd.read_pickle(TEST_DATA / "regression_tests" / "solution.pkl")

    # Generate current lcm output
    n_periods = 4
    model = get_model(n_periods=n_periods)
    params = get_params(
        n_periods=n_periods,
        discount_factor=0.95,
        disutility_of_work=1.0,
        interest_rate=0.05,
    )

    got_solve: dict[int, dict[str, FloatND]] = model.solve(params)
    got_simulate = model.solve_and_simulate(
        params=params,
        initial_states={"wealth": jnp.array([5.0, 20, 40, 70])},
        initial_regimes=["working"] * 4,
    ).to_dataframe()

    # Compare solution
    for period in range(n_periods - 1):
        for regime in got_solve[period]:
            aaae(expected_solve[period][regime], got_solve[period][regime], decimal=5)

    # Compare simulation (use tolerance to match solution comparison precision)
    assert_frame_equal(got_simulate, expected_simulate, check_dtype=False, atol=1e-5)


# ======================================================================================
# Test that all grid types work in model solution and simulation
# ======================================================================================


def _create_grid(
    grid_type: str, start: float, stop: float, n_points: int
) -> ContinuousGrid:
    """Create a grid of the specified type."""
    if grid_type == "LinspaceGrid":
        return LinspaceGrid(start=start, stop=stop, n_points=n_points)
    if grid_type == "LogspaceGrid":
        return LogspaceGrid(start=start, stop=stop, n_points=n_points)
    msg = f"Unknown grid type: {grid_type}"
    raise ValueError(msg)


@pytest.mark.parametrize(
    "grid_type",
    [
        "LinspaceGrid",
        "LogspaceGrid",
    ],
)
def test_model_with_different_grid_types(grid_type: str):
    """Test that model solution and simulation work with all grid types."""
    n_periods = 4
    # As the borrowing constraint uses weak inequality, we cannot use log-spaced grids
    # for wealth. Consuming everything this period is allowed, but cannot be
    # represented.
    wealth_grid = _create_grid(
        grid_type=grid_type.replace("Log", "Lin"), n_points=100, start=1, stop=400
    )
    consumption_grid = _create_grid(
        grid_type=grid_type, start=1, stop=400, n_points=500
    )

    model = get_model(
        n_periods=n_periods,
        wealth_grid=wealth_grid,
        consumption_grid=consumption_grid,
    )
    params = get_params(
        n_periods=n_periods,
        discount_factor=0.95,
        disutility_of_work=1.0,
        interest_rate=0.05,
    )

    # This should complete without error
    result = model.solve_and_simulate(
        params=params,
        initial_states={"wealth": jnp.array([5.0, 20, 40, 70])},
        initial_regimes=["working"] * 4,
    )
    df = result.to_dataframe()

    # Basic sanity checks
    assert len(df) == n_periods * 4  # 4 periods * 4 subjects
    assert "wealth" in df.columns
    assert "consumption" in df.columns
