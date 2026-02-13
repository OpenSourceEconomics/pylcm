import pandas as pd
import pytest
from jax import numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from lcm._config import TEST_DATA
from tests.conftest import X64_ENABLED
from tests.test_models.shocks import get_model, get_params


@pytest.mark.skipif(not X64_ENABLED, reason="Not working with 32-Bit because of RNG")
@pytest.mark.parametrize(
    "distribution_type", ["uniform", "normal", "tauchen", "rouwenhorst"]
)
def test_model_with_shock(distribution_type):
    n_periods = 3

    model = get_model(n_periods, distribution_type)
    params = get_params()

    got_solve = model.solve(
        params=params,
    )

    got_simulate = model.simulate(
        params=params,
        initial_regimes=["test_regime"] * 2,
        initial_states={
            "health": jnp.asarray([0, 0]),
            "income": jnp.asarray([0, 0]),
            "wealth": jnp.asarray([1, 1]),
        },
        V_arr_dict=got_solve,
        seed=42,
    ).to_dataframe()

    expected_simulate = pd.read_pickle(
        TEST_DATA / "shocks" / f"simulation_{distribution_type}.pkl"
    )
    expected_solve = pd.read_pickle(
        TEST_DATA / "shocks" / f"solution_{distribution_type}.pkl"
    )
    # Compare solution
    for period in range(n_periods - 1):
        for regime in got_solve[period]:
            aaae(expected_solve[period][regime], got_solve[period][regime], decimal=5)

    # Compare simulation (use tolerance to match solution comparison precision)
    assert_frame_equal(
        got_simulate,
        expected_simulate,
        check_dtype=False,
        atol=1e-5,
        check_column_type=False,
        check_categorical=False,
    )
