from types import MappingProxyType

import jax
import pandas as pd
import pytest
from jax import numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from lcm import (
    ShockGridAR1Rouwenhorst,
    ShockGridAR1Tauchen,
    ShockGridIIDNormal,
    ShockGridIIDUniform,
)
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


_GRID_CLASSES = [
    ShockGridIIDUniform,
    ShockGridIIDNormal,
    ShockGridAR1Tauchen,
    ShockGridAR1Rouwenhorst,
]


@pytest.mark.parametrize("grid_cls", _GRID_CLASSES)
def test_shock_grid_correct_shape_without_params(grid_cls):
    """ShockGrid without params returns correct-shape arrays."""
    grid = grid_cls(n_points=3)
    assert not grid.is_fully_specified
    assert grid.params_to_pass_at_runtime
    assert grid.get_gridpoints().shape == (3,)
    assert grid.get_transition_probs().shape == (3, 3)


@pytest.mark.parametrize(
    ("grid_cls", "kwargs"),
    [
        (ShockGridIIDUniform, {"start": 0.0, "stop": 1.0}),
        (ShockGridIIDNormal, {"mean": 0.0, "std": 1.0, "n_std": 3.0}),
        (ShockGridAR1Tauchen, {"ar1_coeff": 0.9, "std": 1.0, "mean": 0.0, "n_std": 2}),
        (ShockGridAR1Rouwenhorst, {"ar1_coeff": 0.9, "std": 1.0, "mean": 0.0}),
    ],
)
def test_shock_grid_fully_specified_with_all_params(grid_cls, kwargs):
    """ShockGrid with all params provided is fully specified."""
    grid = grid_cls(n_points=3, **kwargs)
    assert grid.is_fully_specified
    result = grid.get_gridpoints()
    assert result.shape == (3,)


# ======================================================================================
# draw_shock tests
# ======================================================================================

_N_DRAWS = 10_000


def _draw_many(grid, params, key_seed=0, current_value=None):
    """Helper: draw _N_DRAWS samples from grid.draw_shock."""
    keys = jax.random.split(jax.random.key(key_seed), _N_DRAWS)
    if current_value is not None:
        prev = jnp.array(current_value)
        return jnp.array([grid.draw_shock(params, k, prev) for k in keys])
    return jnp.array([grid.draw_shock(params, k) for k in keys])


@pytest.mark.parametrize("params_at_init", [True, False])
def test_draw_shock_uniform(params_at_init):
    """ShockGridIIDUniform.draw_shock uses start/stop params."""
    kwargs = {"start": 2.0, "stop": 4.0}
    if params_at_init:
        grid = ShockGridIIDUniform(n_points=5, **kwargs)
        params = grid.params
    else:
        grid = ShockGridIIDUniform(n_points=5)
        params = MappingProxyType(kwargs)
    draws = _draw_many(grid, params)
    assert jnp.all(draws >= 2.0)
    assert jnp.all(draws <= 4.0)
    aaae(draws.mean(), 3.0, decimal=1)


@pytest.mark.parametrize("params_at_init", [True, False])
def test_draw_shock_normal(params_at_init):
    """ShockGridIIDNormal.draw_shock uses mean/std params."""
    kwargs = {"mean": 5.0, "std": 0.1, "n_std": 3.0}
    if params_at_init:
        grid = ShockGridIIDNormal(n_points=5, **kwargs)
        params = grid.params
    else:
        grid = ShockGridIIDNormal(n_points=5)
        params = MappingProxyType(kwargs)
    draws = _draw_many(grid, params)
    aaae(draws.mean(), 5.0, decimal=1)
    aaae(draws.std(), 0.1, decimal=1)


@pytest.mark.parametrize("params_at_init", [True, False])
def test_draw_shock_tauchen(params_at_init):
    """ShockGridAR1Tauchen.draw_shock uses mean/std/ar1_coeff params."""
    kwargs = {"ar1_coeff": 0.5, "std": 0.1, "mean": 2.0, "n_std": 3.0}
    if params_at_init:
        grid = ShockGridAR1Tauchen(n_points=5, **kwargs)
        params = grid.params
    else:
        grid = ShockGridAR1Tauchen(n_points=5)
        params = MappingProxyType(kwargs)
    draws = _draw_many(grid, params, current_value=3.0)
    aaae(draws.mean(), 2.5, decimal=1)
    aaae(draws.std(), 0.1, decimal=1)


@pytest.mark.parametrize("params_at_init", [True, False])
def test_draw_shock_rouwenhorst(params_at_init):
    """ShockGridAR1Rouwenhorst.draw_shock uses mean/std/ar1_coeff params."""
    kwargs = {"ar1_coeff": 0.5, "std": 0.1, "mean": 2.0}
    if params_at_init:
        grid = ShockGridAR1Rouwenhorst(n_points=5, **kwargs)
        params = grid.params
    else:
        grid = ShockGridAR1Rouwenhorst(n_points=5)
        params = MappingProxyType(kwargs)
    draws = _draw_many(grid, params, current_value=3.0)
    aaae(draws.mean(), 2.5, decimal=1)
    aaae(draws.std(), 0.1, decimal=1)
