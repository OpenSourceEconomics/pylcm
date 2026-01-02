from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from jax import numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from lcm._config import TEST_DATA
from tests.test_models.deterministic.regression import get_model, get_params

if TYPE_CHECKING:
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
