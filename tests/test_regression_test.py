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
    # Load generated output
    # ==================================================================================
    expected_simulate = pd.read_pickle(
        TEST_DATA.joinpath("regression_tests", "simulation.pkl"),
    )

    expected_solve = pd.read_pickle(
        TEST_DATA.joinpath("regression_tests", "solution.pkl"),
    )

    # Generate current lcm ouput
    # ==================================================================================
    n_periods = 4

    model = get_model(n_periods=n_periods)

    params = get_params(
        n_periods=n_periods,
        beta=0.95,
        disutility_of_work=1.0,
        interest_rate=0.05,
    )
    got_solve: dict[int, dict[str, FloatND]] = model.solve(params)

    result = model.solve_and_simulate(
        params=params,
        initial_states={"wealth": jnp.array([5.0, 20, 40, 70])},
        initial_regimes=["working"] * 4,
    )
    got_simulate_df = result.to_dataframe()

    # Compare solution
    # ==================================================================================
    for period in range(n_periods - 1):
        for regime in got_solve[period]:
            aaae(expected_solve[period][regime], got_solve[period][regime], decimal=5)

    # Compare simulation (convert flat DataFrame to dict by regime for comparison)
    # ==================================================================================
    for regime in expected_simulate:
        expected_cols = expected_simulate[regime].columns.tolist()
        got_regime_df = (
            got_simulate_df.query(f'regime == "{regime}"')
            .drop(columns="regime")[expected_cols]  # Only select expected columns
            .sort_values(["period", "subject_id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(
            expected_simulate[regime], got_regime_df, check_like=True, check_dtype=False
        )
