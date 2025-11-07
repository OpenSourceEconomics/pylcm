from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from jax import numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from lcm._config import TEST_DATA
from tests.test_models.utils import get_model, get_params

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
    model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=3)

    params = get_params(
        regime_name="iskhakov_et_al_2017_stripped_down",
        beta=0.95,
        disutility_of_work=1.0,
        interest_rate=0.05,
    )
    got_solve: dict[int, FloatND] = model.solve(params)

    got_simulate = model.solve_and_simulate(
        params=params,
        initial_states={
            "iskhakov_et_al_2017_stripped_down__wealth": jnp.array([5.0, 20, 40, 70]),
        },
        initial_regimes=["iskhakov_et_al_2017_stripped_down"] * 4,
    )
    # Compare
    # ==================================================================================
    for period in expected_solve:
        for regime in expected_solve[period]:
            aaae(expected_solve[period][regime], got_solve[period][regime], decimal=5)

    for regime in expected_simulate:
        assert_frame_equal(expected_simulate[regime], got_simulate[regime])
