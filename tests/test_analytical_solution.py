"""Testing against the analytical solution of Iskhakov et al. (2017).

The benchmark is taken from the paper "The endogenous grid method for
discrete-continuous dynamic action models with (or without) taste shocks" by Fedor
Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning (2017,
https://doi.org/10.3982/QE643).

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm._config import TEST_DATA
from lcm.entry_point import get_lcm_function
from tests.test_models import get_model_config, get_params

if TYPE_CHECKING:
    from jax import Array

# ======================================================================================
# Model specifications
# ======================================================================================


TEST_CASES = {
    "iskhakov_2017_five_periods": {
        "model": get_model_config("iskhakov_et_al_2017", n_periods=5),
        "params": get_params(
            beta=0.98,
            disutility_of_work=1.0,
            interest_rate=0.0,
            wage=20.0,
        ),
    },
    "iskhakov_2017_low_delta": {
        "model": get_model_config("iskhakov_et_al_2017", n_periods=3),
        "params": get_params(
            beta=0.98,
            disutility_of_work=0.1,
            interest_rate=0.0,
            wage=20.0,
        ),
    },
}


def mean_square_error(x, y, axis=None):
    return np.mean((x - y) ** 2, axis=axis)


# ======================================================================================
# Test
# ======================================================================================


@pytest.mark.parametrize(("model_name", "model_and_params"), TEST_CASES.items())
def test_analytical_solution(model_name, model_and_params):
    """Test that the numerical solution matches the analytical solution.

    The analytical solution is from Iskhakov et al (2017) and is generated
    in the development repository: github.com/opensourceeconomics/pylcm-dev.

    """
    # Compute LCM solution
    # ==================================================================================
    solve_model, _ = get_lcm_function(model=model_and_params["model"], targets="solve")

    V_arr_dict: dict[int, Array] = solve_model(params=model_and_params["params"])  # type: ignore[assignment]
    V_arr_list = list(dict(sorted(V_arr_dict.items(), key=lambda x: x[0])).values())

    _numerical = np.stack(V_arr_list)
    numerical = {
        "worker": _numerical[:, 0, :],
        "retired": _numerical[:, 1, :],
    }

    # Load analytical solution
    # ==================================================================================
    analytical = {
        _type: np.genfromtxt(
            TEST_DATA.joinpath(
                "analytical_solution",
                f"{model_name}__values_{_type}.csv",
            ),
            delimiter=",",
        )
        for _type in ["worker", "retired"]
    }

    # Compare
    # ==================================================================================
    for _type in ["worker", "retired"]:
        _analytical = np.array(analytical[_type])
        _numerical = numerical[_type]

        # Compare the whole trajectory over time
        mse = mean_square_error(_analytical, _numerical, axis=0)
        # Exclude the first two initial wealth levels from the comparison, because the
        # numerical solution is unstable for very low wealth levels.
        aaae(mse[2:], 0, decimal=1)
