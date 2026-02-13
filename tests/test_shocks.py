from types import MappingProxyType

import pandas as pd
import pytest
from jax import numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from lcm import ShockGrid
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
    params = get_params(distribution_type)

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


@pytest.mark.parametrize(
    "distribution_type", ["uniform", "normal", "tauchen", "rouwenhorst"]
)
def test_shock_grid_correct_shape_without_params(distribution_type):
    """ShockGrid without shock_params returns correct-shape arrays."""
    grid = ShockGrid(distribution_type=distribution_type, n_points=3)
    assert not grid.is_fully_specified
    assert len(grid.params_to_pass_at_runtime) > 0
    assert grid.shock.get_gridpoints().shape == (3,)
    assert grid.shock.get_transition_probs().shape == (3, 3)


@pytest.mark.parametrize(
    ("distribution_type", "shock_params"),
    [
        ("uniform", {"start": 0.0, "stop": 1.0}),
        ("normal", {"mu_eps": 0.0, "sigma_eps": 1.0, "n_std": 3.0}),
        ("tauchen", {"rho": 0.9, "sigma_eps": 1.0, "mu_eps": 0.0, "n_std": 2}),
        ("rouwenhorst", {"rho": 0.9, "sigma_eps": 1.0, "mu_eps": 0.0}),
    ],
)
def test_shock_grid_fully_specified_with_all_params(distribution_type, shock_params):
    """ShockGrid with all params provided is fully specified."""
    grid = ShockGrid(
        distribution_type=distribution_type,
        n_points=3,
        shock_params=MappingProxyType(shock_params),
    )
    assert grid.is_fully_specified
    result = grid.shock.get_gridpoints()
    assert result.shape == (3,)
