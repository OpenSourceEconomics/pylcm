"""Tests for draw_shock sampling correctness across all shock grid types."""

from types import MappingProxyType

import jax
import pytest
from jax import numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae

import lcm

_N_DRAWS = 10_000

_NORMAL_MIXTURE_KWARGS = {
    "n_std": 3.0,
    "p1": 0.9,
    "mu1": 0.0,
    "sigma1": 0.1,
    "mu2": -0.5,
    "sigma2": 0.3,
}

_TAUCHEN_NORMAL_MIXTURE_KWARGS = {
    "rho": 0.8,
    "mu": 1.0,
    "n_std": 3.0,
    "p1": 0.9,
    "mu1": 0.0,
    "sigma1": 0.1,
    "mu2": -0.5,
    "sigma2": 0.3,
}

_AR1_GRID_CLASSES = [lcm.shocks.ar1.Tauchen, lcm.shocks.ar1.Rouwenhorst]


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


@pytest.mark.parametrize("params_at_init", [True, False])
def test_draw_shock_normal_mixture(params_at_init):
    """NormalMixture.draw_shock produces draws with correct mixture moments."""
    kwargs = _NORMAL_MIXTURE_KWARGS
    if params_at_init:
        grid = lcm.shocks.iid.NormalMixture(n_points=5, **kwargs)
        params = grid.params
    else:
        grid = lcm.shocks.iid.NormalMixture(n_points=5)
        params = MappingProxyType(kwargs)
    draws = _draw_many(grid, params)
    p1 = kwargs["p1"]
    mu1, mu2 = kwargs["mu1"], kwargs["mu2"]
    sigma1, sigma2 = kwargs["sigma1"], kwargs["sigma2"]
    expected_mean = p1 * mu1 + (1 - p1) * mu2
    expected_var = (
        p1 * (sigma1**2 + mu1**2) + (1 - p1) * (sigma2**2 + mu2**2) - expected_mean**2
    )
    aaae(draws.mean(), expected_mean, decimal=1)
    aaae(draws.var(), expected_var, decimal=1)


@pytest.mark.parametrize("params_at_init", [True, False])
def test_draw_shock_tauchen_normal_mixture(params_at_init):
    """TauchenNormalMixture.draw_shock produces yields correct conditional moments."""
    kwargs = _TAUCHEN_NORMAL_MIXTURE_KWARGS
    if params_at_init:
        grid = lcm.shocks.ar1.TauchenNormalMixture(n_points=5, **kwargs)
        params = grid.params
    else:
        grid = lcm.shocks.ar1.TauchenNormalMixture(n_points=5)
        params = MappingProxyType(kwargs)
    current_value = 3.0
    draws = _draw_many(grid, params, current_value=current_value)
    p1, mu1, mu2 = kwargs["p1"], kwargs["mu1"], kwargs["mu2"]
    mean_eps = p1 * mu1 + (1 - p1) * mu2
    expected_mean = kwargs["mu"] + kwargs["rho"] * current_value + mean_eps
    aaae(draws.mean(), expected_mean, decimal=1)


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
