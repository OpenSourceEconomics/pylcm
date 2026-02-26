from types import MappingProxyType

import jax
import pandas as pd
import pytest
from jax import numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

import lcm
from lcm._config import TEST_DATA
from lcm.exceptions import GridInitializationError
from tests.conftest import DECIMAL_PRECISION, X64_ENABLED
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
            "age": jnp.asarray([0.0, 0.0]),
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


# ======================================================================================
# Shape tests
# ======================================================================================

_GRID_CLASSES_WITH_GH_KWARG = [
    (lcm.shocks.iid.Uniform, {}),
    (lcm.shocks.iid.Normal, {"gauss_hermite": True}),
    (lcm.shocks.iid.LogNormal, {"gauss_hermite": True}),
    (lcm.shocks.ar1.Tauchen, {"gauss_hermite": True}),
    (lcm.shocks.ar1.Rouwenhorst, {}),
]


@pytest.mark.parametrize(("grid_cls", "extra_kw"), _GRID_CLASSES_WITH_GH_KWARG)
def test_shock_grid_correct_shape_without_params(grid_cls, extra_kw):
    """ShockGrid without params returns correct-shape arrays."""
    grid = grid_cls(n_points=3, **extra_kw)
    assert not grid.is_fully_specified
    assert grid.params_to_pass_at_runtime
    assert grid.get_gridpoints().shape == (3,)
    assert grid.get_transition_probs().shape == (3, 3)
    assert jnp.all(jnp.isnan(grid.get_gridpoints()))
    assert jnp.all(jnp.isnan(grid.get_transition_probs()))


@pytest.mark.parametrize(
    ("grid_cls", "kwargs"),
    [
        (lcm.shocks.iid.Uniform, {"start": 0.0, "stop": 1.0}),
        (
            lcm.shocks.iid.Normal,
            {"gauss_hermite": True, "mu": 0.0, "sigma": 1.0},
        ),
        (
            lcm.shocks.iid.Normal,
            {"gauss_hermite": False, "mu": 0.0, "sigma": 1.0, "n_std": 3.0},
        ),
        (
            lcm.shocks.iid.LogNormal,
            {"gauss_hermite": True, "mu": 0.0, "sigma": 1.0},
        ),
        (
            lcm.shocks.iid.LogNormal,
            {"gauss_hermite": False, "mu": 0.0, "sigma": 1.0, "n_std": 3.0},
        ),
        (
            lcm.shocks.ar1.Tauchen,
            {"gauss_hermite": True, "rho": 0.9, "sigma": 1.0, "mu": 0.0},
        ),
        (
            lcm.shocks.ar1.Tauchen,
            {"gauss_hermite": False, "rho": 0.9, "sigma": 1.0, "mu": 0.0, "n_std": 2},
        ),
        (lcm.shocks.ar1.Rouwenhorst, {"rho": 0.9, "sigma": 1.0, "mu": 0.0}),
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
    """Uniform.draw_shock uses start/stop params."""
    kwargs = {"start": 2.0, "stop": 4.0}
    if params_at_init:
        grid = lcm.shocks.iid.Uniform(n_points=5, **kwargs)
        params = grid.params
    else:
        grid = lcm.shocks.iid.Uniform(n_points=5)
        params = MappingProxyType(kwargs)
    draws = _draw_many(grid, params)
    assert jnp.all(draws >= 2.0)
    assert jnp.all(draws <= 4.0)
    aaae(draws.mean(), 3.0, decimal=1)


@pytest.mark.parametrize("params_at_init", [True, False])
def test_draw_shock_normal(params_at_init):
    """Normal.draw_shock uses mu/sigma params."""
    kwargs = {"mu": 5.0, "sigma": 0.1}
    if params_at_init:
        grid = lcm.shocks.iid.Normal(n_points=5, gauss_hermite=True, **kwargs)
        params = grid.params
    else:
        grid = lcm.shocks.iid.Normal(n_points=5, gauss_hermite=True)
        params = MappingProxyType(kwargs)
    draws = _draw_many(grid, params)
    aaae(draws.mean(), 5.0, decimal=1)
    aaae(draws.std(), 0.1, decimal=1)


@pytest.mark.parametrize("params_at_init", [True, False])
def test_draw_shock_lognormal(params_at_init):
    """LogNormal.draw_shock produces positive samples with correct log-moments."""
    kwargs = {"mu": 1.0, "sigma": 0.1}
    if params_at_init:
        grid = lcm.shocks.iid.LogNormal(n_points=5, gauss_hermite=True, **kwargs)
        params = grid.params
    else:
        grid = lcm.shocks.iid.LogNormal(n_points=5, gauss_hermite=True)
        params = MappingProxyType(kwargs)
    draws = _draw_many(grid, params)
    assert jnp.all(draws > 0)
    aaae(jnp.log(draws).mean(), 1.0, decimal=1)
    aaae(jnp.log(draws).std(), 0.1, decimal=1)


@pytest.mark.parametrize("params_at_init", [True, False])
def test_draw_shock_tauchen(params_at_init):
    """Tauchen.draw_shock uses mu/sigma/rho params."""
    kwargs = {"rho": 0.5, "sigma": 0.1, "mu": 2.0}
    if params_at_init:
        grid = lcm.shocks.ar1.Tauchen(n_points=5, gauss_hermite=True, **kwargs)
        params = grid.params
    else:
        grid = lcm.shocks.ar1.Tauchen(n_points=5, gauss_hermite=True)
        params = MappingProxyType(kwargs)
    draws = _draw_many(grid, params, current_value=3.0)
    aaae(draws.mean(), 3.5, decimal=1)
    aaae(draws.std(), 0.1, decimal=1)


@pytest.mark.parametrize("params_at_init", [True, False])
def test_draw_shock_rouwenhorst(params_at_init):
    """Rouwenhorst.draw_shock uses mu/sigma/rho params."""
    kwargs = {"rho": 0.5, "sigma": 0.1, "mu": 2.0}
    if params_at_init:
        grid = lcm.shocks.ar1.Rouwenhorst(n_points=5, **kwargs)
        params = grid.params
    else:
        grid = lcm.shocks.ar1.Rouwenhorst(n_points=5)
        params = MappingProxyType(kwargs)
    draws = _draw_many(grid, params, current_value=3.0)
    aaae(draws.mean(), 3.5, decimal=1)
    aaae(draws.std(), 0.1, decimal=1)


# ======================================================================================
# AR(1) grid property tests
# ======================================================================================

_AR1_GRID_CLASSES = [lcm.shocks.ar1.Tauchen, lcm.shocks.ar1.Rouwenhorst]


@pytest.mark.parametrize("grid_cls", _AR1_GRID_CLASSES)
def test_ar1_grid_centers_on_unconditional_mean(grid_cls):
    """Midpoint of AR(1) gridpoints is approximately mu / (1 - rho)."""
    mu, rho = 2.0, 0.8
    kwargs = {"rho": rho, "sigma": 0.5, "mu": mu}
    if grid_cls is lcm.shocks.ar1.Tauchen:
        kwargs["gauss_hermite"] = True
    grid = grid_cls(n_points=11, **kwargs)
    points = grid.get_gridpoints()
    midpoint = (points[0] + points[-1]) / 2
    expected = mu / (1 - rho)
    aaae(midpoint, expected, decimal=10)


@pytest.mark.parametrize("grid_cls", _AR1_GRID_CLASSES)
def test_ar1_transition_probs_rows_sum_to_one(grid_cls):
    """Each row of the transition matrix sums to 1."""
    kwargs = {"rho": 0.9, "sigma": 0.5, "mu": 1.0}
    if grid_cls is lcm.shocks.ar1.Tauchen:
        kwargs["gauss_hermite"] = True
    grid = grid_cls(n_points=7, **kwargs)
    P = grid.get_transition_probs()
    row_sums = P.sum(axis=1)
    aaae(row_sums, jnp.ones(7), decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("grid_cls", _AR1_GRID_CLASSES)
def test_ar1_draw_shock_unconditional_moments(grid_cls):
    """Long-run simulated moments match AR(1) unconditional moments."""
    mu, rho, sigma = 0.5, 0.7, 0.3
    kwargs = {"rho": rho, "sigma": sigma, "mu": mu}
    if grid_cls is lcm.shocks.ar1.Tauchen:
        kwargs["gauss_hermite"] = True
    grid = grid_cls(n_points=11, **kwargs)
    params = grid.params

    n_steps = 20_000
    burn_in = 2_000
    key = jax.random.key(42)
    y = jnp.array(mu / (1 - rho))
    trajectory = []
    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        y = grid.draw_shock(params, subkey, y)
        trajectory.append(y)
    samples = jnp.array(trajectory[burn_in:])

    expected_mean = mu / (1 - rho)
    expected_std = sigma / jnp.sqrt(1 - rho**2)
    aaae(samples.mean(), expected_mean, decimal=1)
    aaae(samples.std(), expected_std, decimal=1)


# ======================================================================================
# Validation tests
# ======================================================================================


@pytest.mark.parametrize(
    "grid_cls_and_kwargs",
    [
        (lcm.shocks.iid.Normal, {"gauss_hermite": True}),
        (lcm.shocks.iid.LogNormal, {"gauss_hermite": True}),
        (lcm.shocks.ar1.Tauchen, {"gauss_hermite": True}),
        (lcm.shocks.ar1.Rouwenhorst, {}),
    ],
)
def test_even_n_points_rejected(grid_cls_and_kwargs):
    """Normal, LogNormal, Tauchen, and Rouwenhorst reject even n_points."""
    grid_cls, extra_kw = grid_cls_and_kwargs
    with pytest.raises(GridInitializationError, match="n_points must be odd"):
        grid_cls(n_points=4, **extra_kw)


@pytest.mark.parametrize(
    "grid_cls",
    [lcm.shocks.iid.Normal, lcm.shocks.iid.LogNormal, lcm.shocks.ar1.Tauchen],
)
def test_gauss_hermite_and_n_std_mutual_exclusion(grid_cls):
    """gauss_hermite=True and n_std are mutually exclusive."""
    with pytest.raises(GridInitializationError, match="mutually exclusive"):
        grid_cls(n_points=3, gauss_hermite=True, n_std=2.0)


def test_gauss_hermite_required():
    """Normal(n_points=5) without gauss_hermite raises TypeError."""
    with pytest.raises(TypeError):
        lcm.shocks.iid.Normal(n_points=5)  # ty: ignore[missing-argument]


# ======================================================================================
# Gauss-Hermite specific tests
# ======================================================================================


def test_normal_gauss_hermite_weights_sum_to_one():
    """GH weights for iid Normal sum to 1."""
    grid = lcm.shocks.iid.Normal(n_points=7, gauss_hermite=True, mu=0.0, sigma=1.0)
    P = grid.get_transition_probs()
    aaae(P[0].sum(), 1.0, decimal=DECIMAL_PRECISION)


def test_normal_gauss_hermite_n_std_not_in_params():
    """n_std is excluded from params_to_pass_at_runtime when gauss_hermite=True."""
    grid = lcm.shocks.iid.Normal(n_points=5, gauss_hermite=True)
    assert "n_std" not in grid.params_to_pass_at_runtime


def test_tauchen_gauss_hermite_transition_probs_rows_sum_to_one():
    """Each row of the GH Tauchen transition matrix sums to 1."""
    grid = lcm.shocks.ar1.Tauchen(
        n_points=7, gauss_hermite=True, rho=0.9, sigma=0.5, mu=1.0
    )
    P = grid.get_transition_probs()
    row_sums = P.sum(axis=1)
    aaae(row_sums, jnp.ones(7), decimal=DECIMAL_PRECISION)


def test_tauchen_gauss_hermite_centers_on_unconditional_mean():
    """GH Tauchen gridpoints center on mu / (1 - rho)."""
    mu, rho = 2.0, 0.8
    grid = lcm.shocks.ar1.Tauchen(
        n_points=11, gauss_hermite=True, rho=rho, sigma=0.5, mu=mu
    )
    points = grid.get_gridpoints()
    midpoint = (points[0] + points[-1]) / 2
    expected = mu / (1 - rho)
    aaae(midpoint, expected, decimal=10)


# ======================================================================================
# LogNormal specific tests
# ======================================================================================


def test_lognormal_correct_shape_without_params():
    """LogNormal without params returns correct-shape NaN arrays."""
    grid = lcm.shocks.iid.LogNormal(n_points=3, gauss_hermite=True)
    assert not grid.is_fully_specified
    assert grid.get_gridpoints().shape == (3,)
    assert grid.get_transition_probs().shape == (3, 3)


def test_lognormal_fully_specified_gauss_hermite():
    """LogNormal with gauss_hermite=True is fully specified with mu/sigma."""
    grid = lcm.shocks.iid.LogNormal(n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0)
    assert grid.is_fully_specified
    assert grid.get_gridpoints().shape == (5,)


def test_lognormal_fully_specified_n_std():
    """LogNormal with gauss_hermite=False is fully specified with mu/sigma/n_std."""
    grid = lcm.shocks.iid.LogNormal(
        n_points=5, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=3.0
    )
    assert grid.is_fully_specified
    assert grid.get_gridpoints().shape == (5,)


def test_lognormal_gridpoints_are_positive():
    """All LogNormal gridpoints are strictly positive."""
    grid = lcm.shocks.iid.LogNormal(n_points=7, gauss_hermite=True, mu=0.0, sigma=1.0)
    assert jnp.all(grid.get_gridpoints() > 0)


def test_lognormal_gauss_hermite_weights_sum_to_one():
    """GH weights for iid LogNormal sum to 1."""
    grid = lcm.shocks.iid.LogNormal(n_points=7, gauss_hermite=True, mu=0.0, sigma=1.0)
    P = grid.get_transition_probs()
    aaae(P[0].sum(), 1.0, decimal=DECIMAL_PRECISION)
