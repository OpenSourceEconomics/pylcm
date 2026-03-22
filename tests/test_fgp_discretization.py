"""Tests verifying discretization accuracy against Fella, Gallipoli & Pan (2019).

FGP (*Review of Economic Dynamics*, 2019) compare Tauchen, Adda-Cooper, and
Rouwenhorst for discretizing non-stationary AR(1) income processes in lifecycle
models. Rouwenhorst dominates across all configurations.

Reference parameters (FGP Section 4, p. 191):
    sigma_eps^2 = 0.0161, mu = 0, rho in {0.95, 0.98}, N in {5, 25}.

"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats.norm import cdf as jax_cdf
from numpy.testing import assert_array_almost_equal as aaae

import lcm
from lcm.shocks._base import _mixture_cdf
from tests.conftest import DECIMAL_PRECISION

SIGMA_EPS_SQ = 0.0161
SIGMA_EPS = np.sqrt(SIGMA_EPS_SQ)

# Mixture parameters (FGP Section 4.3, p. 195)
FGP_P1 = 0.9
FGP_MU1 = 0.0089
FGP_MU2 = -FGP_MU1 * FGP_P1 / (1 - FGP_P1)  # -0.0800 (approx)
FGP_SIGMA1 = 0.0635
FGP_SIGMA2 = 0.3430


@pytest.mark.parametrize(
    ("rho", "n_points"),
    [(0.95, 5), (0.95, 25), (0.98, 5), (0.98, 25)],
    ids=str,
)
def test_tauchen_grid_span_matches_fgp(rho, n_points):
    """Tauchen grid span matches FGP Eq. 5: +/- n_std * sigma_eps / sqrt(1 - rho^2)."""
    n_std = 3.0
    grid = lcm.shocks.ar1.Tauchen(
        n_points=n_points,
        gauss_hermite=False,
        rho=rho,
        sigma=SIGMA_EPS,
        mu=0.0,
        n_std=n_std,
    )
    points = grid.get_gridpoints()

    expected_std_y = SIGMA_EPS / np.sqrt(1 - rho**2)
    expected_span = 2 * n_std * expected_std_y
    aaae(float(points[-1] - points[0]), expected_span, decimal=DECIMAL_PRECISION)
    aaae(float((points[0] + points[-1]) / 2), 0.0, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(
    ("rho", "n_points"),
    [(0.95, 5), (0.95, 25), (0.98, 5), (0.98, 25)],
    ids=str,
)
def test_rouwenhorst_grid_span_matches_fgp(rho, n_points):
    """Rouwenhorst grid span matches FGP Eq. 13."""
    grid = lcm.shocks.ar1.Rouwenhorst(
        n_points=n_points, rho=rho, sigma=SIGMA_EPS, mu=0.0
    )
    points = grid.get_gridpoints()

    expected_std_y = SIGMA_EPS / np.sqrt(1 - rho**2)
    expected_span = 2 * np.sqrt(n_points - 1) * expected_std_y
    aaae(float(points[-1] - points[0]), expected_span, decimal=DECIMAL_PRECISION)
    aaae(float((points[0] + points[-1]) / 2), 0.0, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(
    ("method", "rho", "n_points"),
    [
        ("tauchen", 0.95, 5),
        ("tauchen", 0.95, 25),
        ("tauchen", 0.98, 5),
        ("tauchen", 0.98, 25),
        ("rouwenhorst", 0.95, 5),
        ("rouwenhorst", 0.95, 25),
        ("rouwenhorst", 0.98, 5),
        ("rouwenhorst", 0.98, 25),
    ],
    ids=str,
)
def test_transition_rows_sum_to_one(method, rho, n_points):
    """Each row of the transition matrix sums to 1."""
    grid = _make_grid(method, rho, n_points)
    P = grid.get_transition_probs()
    aaae(P.sum(axis=1), jnp.ones(n_points), decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(
    ("method", "rho", "n_points"),
    [
        ("tauchen", 0.95, 5),
        ("tauchen", 0.95, 25),
        ("tauchen", 0.98, 5),
        ("tauchen", 0.98, 25),
        ("rouwenhorst", 0.95, 5),
        ("rouwenhorst", 0.95, 25),
        ("rouwenhorst", 0.98, 5),
        ("rouwenhorst", 0.98, 25),
    ],
    ids=str,
)
def test_transition_probs_nonnegative(method, rho, n_points):
    """All entries of the transition matrix are non-negative."""
    grid = _make_grid(method, rho, n_points)
    P = grid.get_transition_probs()
    assert jnp.all(P >= 0)


@pytest.mark.parametrize(
    ("rho", "n_points"),
    [(0.95, 5), (0.95, 25), (0.98, 5), (0.98, 25)],
    ids=str,
)
def test_rouwenhorst_conditional_mean(rho, n_points):
    """Rouwenhorst conditional mean from state i equals rho * y_i (FGP Eq. 9)."""
    grid = lcm.shocks.ar1.Rouwenhorst(
        n_points=n_points, rho=rho, sigma=SIGMA_EPS, mu=0.0
    )
    points = grid.get_gridpoints()
    P = grid.get_transition_probs()

    # E[y' | y_i] = sum_j P[i,j] * y_j should equal rho * y_i
    conditional_means = P @ points
    expected = rho * points
    aaae(conditional_means, expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(
    ("rho", "n_points"),
    [(0.95, 5), (0.95, 25), (0.98, 5), (0.98, 25)],
    ids=str,
)
def test_rouwenhorst_conditional_variance(rho, n_points):
    """Rouwenhorst conditional variance from state i equals sigma_eps^2 (FGP Eq. 11)."""
    grid = lcm.shocks.ar1.Rouwenhorst(
        n_points=n_points, rho=rho, sigma=SIGMA_EPS, mu=0.0
    )
    points = grid.get_gridpoints()
    P = grid.get_transition_probs()

    conditional_means = P @ points
    # Var[y' | y_i] = sum_j P[i,j] * (y_j - E[y'|y_i])^2
    deviations = points[None, :] - conditional_means[:, None]
    conditional_vars = (P * deviations**2).sum(axis=1)
    aaae(conditional_vars, jnp.full(n_points, SIGMA_EPS_SQ), decimal=DECIMAL_PRECISION)


def _stationary_distribution(P):
    """Compute stationary distribution via eigendecomposition."""
    eigvals, eigvecs = jnp.linalg.eig(P.T)
    idx = jnp.argmin(jnp.abs(eigvals - 1.0))
    pi = jnp.real(eigvecs[:, idx])
    return pi / pi.sum()


def _markov_chain_moments(points, P):
    """Compute mean, variance, and lag-1 autocorrelation."""
    pi = _stationary_distribution(P)
    mean = jnp.dot(pi, points)
    var = jnp.dot(pi, (points - mean) ** 2)
    cross = jnp.sum(pi[:, None] * P * points[:, None] * points[None, :])
    autocorr = (cross - mean**2) / var
    return float(mean), float(var), float(autocorr)


@pytest.mark.parametrize(
    ("method", "rho", "n_points"),
    [
        ("tauchen", 0.95, 5),
        ("tauchen", 0.95, 25),
        ("tauchen", 0.98, 5),
        ("tauchen", 0.98, 25),
        ("rouwenhorst", 0.95, 5),
        ("rouwenhorst", 0.95, 25),
        ("rouwenhorst", 0.98, 5),
        ("rouwenhorst", 0.98, 25),
    ],
    ids=str,
)
def test_stationary_moments(method, rho, n_points):
    """Markov chain stationary mean, variance, and autocorrelation match theory."""
    grid = _make_grid(method, rho, n_points)
    points = grid.get_gridpoints()
    P = grid.get_transition_probs()

    expected_mean = 0.0
    expected_var = SIGMA_EPS_SQ / (1 - rho**2)

    got_mean, got_var, got_autocorr = _markov_chain_moments(points, P)

    # Tauchen with high persistence + few points is known to be inaccurate —
    # this is precisely the FGP finding. Use wider tolerance for that case.
    if method == "tauchen" and rho >= 0.98 and n_points <= 5:
        assert abs(got_mean - expected_mean) < 0.5 * np.sqrt(expected_var)
        assert got_var > 0  # at least positive
        assert 0 < got_autocorr < 1  # at least in the right range
    else:
        aaae(got_mean, expected_mean, decimal=1)
        aaae(got_var, expected_var, decimal=1)
        aaae(got_autocorr, rho, decimal=1)


@pytest.mark.parametrize(
    ("rho", "n_points"),
    [(0.95, 5), (0.95, 25), (0.98, 5), (0.98, 25)],
    ids=str,
)
def test_rouwenhorst_more_accurate_than_tauchen(rho, n_points):
    """Rouwenhorst variance and autocorrelation errors are <= Tauchen's."""
    expected_var = SIGMA_EPS_SQ / (1 - rho**2)

    tauchen_grid = _make_grid("tauchen", rho, n_points)
    t_points = tauchen_grid.get_gridpoints()
    t_P = tauchen_grid.get_transition_probs()
    _, t_var, t_autocorr = _markov_chain_moments(t_points, t_P)

    rouw_grid = _make_grid("rouwenhorst", rho, n_points)
    r_points = rouw_grid.get_gridpoints()
    r_P = rouw_grid.get_transition_probs()
    _, r_var, r_autocorr = _markov_chain_moments(r_points, r_P)

    # Rouwenhorst should have smaller (or equal) errors
    assert abs(r_var - expected_var) <= abs(t_var - expected_var) + 1e-12
    assert abs(r_autocorr - rho) <= abs(t_autocorr - rho) + 1e-12


def _make_fgp_mixture_grid(n_points=5, rho=0.95, n_std=3.0):
    """Create a TauchenNormalMixture grid with FGP parameters."""
    return lcm.shocks.ar1.TauchenNormalMixture(
        n_points=n_points,
        rho=rho,
        mu=0.0,
        n_std=n_std,
        p1=FGP_P1,
        mu1=FGP_MU1,
        sigma1=FGP_SIGMA1,
        mu2=FGP_MU2,
        sigma2=FGP_SIGMA2,
    )


def test_mixture_grid_span_matches_tauchen_equivalent():
    """Mixture grid span equals standard Tauchen span for equivalent overall sigma."""
    rho = 0.95
    n_std = 3.0
    n_points = 5

    # Overall innovation std for the mixture
    mean_eps = FGP_P1 * FGP_MU1 + (1 - FGP_P1) * FGP_MU2
    sigma_eps_sq_mix = (
        FGP_P1 * (FGP_SIGMA1**2 + FGP_MU1**2)
        + (1 - FGP_P1) * (FGP_SIGMA2**2 + FGP_MU2**2)
        - mean_eps**2
    )
    std_y_mix = np.sqrt(sigma_eps_sq_mix / (1 - rho**2))
    expected_span = 2 * n_std * std_y_mix

    grid = _make_fgp_mixture_grid(n_points=n_points, rho=rho, n_std=n_std)
    points = grid.get_gridpoints()
    aaae(float(points[-1] - points[0]), expected_span, decimal=DECIMAL_PRECISION)


def test_mixture_transition_rows_sum_to_one():
    """Each row of the TauchenNormalMixture transition matrix sums to 1."""
    grid = _make_fgp_mixture_grid(n_points=5)
    P = grid.get_transition_probs()
    aaae(P.sum(axis=1), jnp.ones(5), decimal=DECIMAL_PRECISION)


def test_mixture_transition_probs_nonnegative():
    """All entries of the mixture transition matrix are non-negative."""
    grid = _make_fgp_mixture_grid(n_points=5)
    P = grid.get_transition_probs()
    assert jnp.all(P >= 0)


def test_mixture_cdf_spot_check():
    """Spot-check mixture CDF against manual scipy-style computation."""
    x = jnp.array([0.0, 0.1, -0.1])
    got = _mixture_cdf(
        x=x, p1=FGP_P1, mu1=FGP_MU1, sigma1=FGP_SIGMA1, mu2=FGP_MU2, sigma2=FGP_SIGMA2
    )

    expected = FGP_P1 * jax_cdf((x - FGP_MU1) / FGP_SIGMA1) + (1 - FGP_P1) * jax_cdf(
        (x - FGP_MU2) / FGP_SIGMA2
    )
    aaae(got, expected, decimal=DECIMAL_PRECISION)


def test_mixture_draw_shock_moments():
    """Draw shock from mixture produces draws with correct unconditional moments."""

    rho = 0.95
    grid = _make_fgp_mixture_grid(n_points=5, rho=rho)
    params = grid.params

    mean_eps = FGP_P1 * FGP_MU1 + (1 - FGP_P1) * FGP_MU2
    sigma_eps_sq_mix = (
        FGP_P1 * (FGP_SIGMA1**2 + FGP_MU1**2)
        + (1 - FGP_P1) * (FGP_SIGMA2**2 + FGP_MU2**2)
        - mean_eps**2
    )

    expected_mean = mean_eps / (1 - rho)
    expected_var = sigma_eps_sq_mix / (1 - rho**2)

    n_steps = 30_000
    burn_in = 5_000
    key = jax.random.key(42)
    y = jnp.array(expected_mean)
    trajectory = []
    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        y = grid.draw_shock(params, subkey, y)
        trajectory.append(y)
    samples = jnp.array(trajectory[burn_in:])

    aaae(float(samples.mean()), expected_mean, decimal=0)
    aaae(float(samples.var()), expected_var, decimal=0)


def test_mixture_stationary_moments():
    """Stationary moments of the mixture Markov chain approximate theory."""
    rho = 0.95
    grid = _make_fgp_mixture_grid(n_points=25, rho=rho, n_std=3.0)
    points = grid.get_gridpoints()
    P = grid.get_transition_probs()

    mean_eps = FGP_P1 * FGP_MU1 + (1 - FGP_P1) * FGP_MU2
    sigma_eps_sq_mix = (
        FGP_P1 * (FGP_SIGMA1**2 + FGP_MU1**2)
        + (1 - FGP_P1) * (FGP_SIGMA2**2 + FGP_MU2**2)
        - mean_eps**2
    )

    expected_mean = mean_eps / (1 - rho)
    expected_var = sigma_eps_sq_mix / (1 - rho**2)

    got_mean, got_var, got_autocorr = _markov_chain_moments(points, P)

    aaae(got_mean, expected_mean, decimal=1)
    aaae(got_var, expected_var, decimal=1)
    aaae(got_autocorr, rho, decimal=1)


def test_mixture_without_params_returns_nan():
    """TauchenNormalMixture without params returns NaN arrays of correct shape."""
    grid = lcm.shocks.ar1.TauchenNormalMixture(n_points=5)
    assert not grid.is_fully_specified
    assert grid.get_gridpoints().shape == (5,)
    assert grid.get_transition_probs().shape == (5, 5)
    assert jnp.all(jnp.isnan(grid.get_gridpoints()))
    assert jnp.all(jnp.isnan(grid.get_transition_probs()))


def test_mixture_fully_specified_with_all_params():
    """TauchenNormalMixture with all params is fully specified."""
    grid = _make_fgp_mixture_grid(n_points=5)
    assert grid.is_fully_specified
    assert grid.get_gridpoints().shape == (5,)


def _make_grid(method, rho, n_points):
    """Create a Tauchen or Rouwenhorst grid with FGP parameters."""
    if method == "tauchen":
        return lcm.shocks.ar1.Tauchen(
            n_points=n_points,
            gauss_hermite=False,
            rho=rho,
            sigma=SIGMA_EPS,
            mu=0.0,
            n_std=3.0,
        )
    return lcm.shocks.ar1.Rouwenhorst(
        n_points=n_points, rho=rho, sigma=SIGMA_EPS, mu=0.0
    )
