"""The retirement-only model reproduces the full-model retired analytical solution.

The `retirement` regime is absorbing, so a two-regime model (retirement + dead)
must yield the same retired value functions as the full Iskhakov et al. (2017)
model. This anchors the oracle that the DC-EGM concave tests compare against:
if this test holds, `iskhakov_2017_*__values_retired.csv` is a valid target for
any solver of the two-regime model.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.config import TEST_DATA
from _lcm.typing import PeriodToRegimeToVArr
from tests.test_models.deterministic.retirement_only import get_model, get_params

ANALYTICAL_CASES = {
    "iskhakov_2017_five_periods": 6,
    "iskhakov_2017_low_delta": 4,
}


def load_analytical_values_retired(case: str) -> np.ndarray:
    return np.genfromtxt(
        TEST_DATA.joinpath("analytical_solution", f"{case}__values_retired.csv"),
        delimiter=",",
    )


def stack_retirement_V(period_to_regime_to_V_arr: PeriodToRegimeToVArr) -> np.ndarray:
    periods = sorted(period_to_regime_to_V_arr)[:-1]
    return np.stack(
        [np.asarray(period_to_regime_to_V_arr[p]["retirement"]) for p in periods]
    )


@pytest.mark.parametrize(("case", "n_periods"), ANALYTICAL_CASES.items())
def test_retirement_only_brute_force_matches_analytical(case, n_periods):
    """Brute-force V of the two-regime model equals the analytical retired values."""
    model = get_model(n_periods)
    params = get_params(n_periods, discount_factor=0.98, interest_rate=0.0)

    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")

    numerical = stack_retirement_V(period_to_regime_to_V_arr)
    analytical = load_analytical_values_retired(case)
    mse = np.mean((analytical - numerical) ** 2, axis=0)
    # Same tolerance as the full-model analytical test: the brute-force solution is
    # unstable at the two lowest wealth levels, so they are excluded.
    aaae(mse[2:], 0, decimal=1)
