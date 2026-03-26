import pandas as pd
import pytest
from jax import numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal
from quantecon.markov.approximation import rouwenhorst as qe_rouwenhorst
from quantecon.markov.approximation import tauchen as qe_tauchen

import lcm
from lcm._config import TEST_DATA
from lcm.exceptions import GridInitializationError
from tests.conftest import DECIMAL_PRECISION, X64_ENABLED
from tests.test_models.shock_grids import (
    MultiRegimeId,
    RegimeId,
    get_model,
    get_multi_regime_model,
    get_multi_regime_params,
    get_params,
)


@pytest.mark.skipif(not X64_ENABLED, reason="Not working with 32-Bit because of RNG")
@pytest.mark.parametrize(
    "distribution_type", ["uniform", "normal", "lognormal", "tauchen", "rouwenhorst"]
)
def test_model_with_shock(distribution_type):
    model = get_model(n_periods=4, distribution_type=distribution_type)
    params = get_params(distribution_type)

    got_solve = model.solve(
        params=params,
    )

    got_simulate = model.simulate(
        params=params,
        initial_conditions={
            "health": jnp.asarray([0, 0]),
            "income": jnp.asarray([0, 0]),
            "wealth": jnp.asarray([1, 1]),
            "age": jnp.asarray([0.0, 0.0]),
            "regime": jnp.array([RegimeId.alive] * 2),
        },
        period_to_regime_to_V_arr=got_solve,
        seed=42,
    ).to_dataframe()

    expected_simulate = pd.read_pickle(
        TEST_DATA / "shocks" / f"simulation_{distribution_type}.pkl"
    )
    expected_solve = pd.read_pickle(
        TEST_DATA / "shocks" / f"solution_{distribution_type}.pkl"
    )
    # Compare solution (iterate over expected regimes — got may have additional ones)
    for period in expected_solve:
        for regime in expected_solve[period]:
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


@pytest.mark.skipif(not X64_ENABLED, reason="Not working with 32-Bit because of RNG")
@pytest.mark.parametrize(
    "distribution_type", ["uniform", "normal", "lognormal", "tauchen", "rouwenhorst"]
)
def test_model_with_cross_regime_shocks(distribution_type: str) -> None:
    """Verify cross-regime transitions work when both regimes have shock grids.

    Reproducer for a crash where shock-state `lambda: None` stubs leak from the
    source regime into cross-regime transition dicts, causing a `vmap` error in
    `joint_weights_from_marginals` (receives `None` instead of arrays).
    """
    model = get_multi_regime_model(n_periods=6, distribution_type=distribution_type)
    params = get_multi_regime_params(distribution_type)  # ty: ignore[invalid-argument-type]

    result = model.simulate(
        params=params,
        initial_conditions={
            "health": jnp.zeros(2, dtype=int),
            "income": jnp.zeros(2),
            "wealth": jnp.ones(2),
            "age": jnp.zeros(2),
            "regime": jnp.full(2, MultiRegimeId.work, dtype=int),
        },
        period_to_regime_to_V_arr=None,
        seed=42,
    ).to_dataframe()
    assert set(result["regime"]) >= {"work", "retire"}


_GRID_CLASSES_WITH_GH_KWARG = [
    (lcm.shocks.iid.Uniform, {}),
    (lcm.shocks.iid.Normal, {"gauss_hermite": True}),
    (lcm.shocks.iid.LogNormal, {"gauss_hermite": True}),
    (lcm.shocks.iid.NormalMixture, {}),
    (lcm.shocks.ar1.Tauchen, {"gauss_hermite": True}),
    (lcm.shocks.ar1.Rouwenhorst, {}),
    (lcm.shocks.ar1.TauchenNormalMixture, {}),
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
        (
            lcm.shocks.iid.NormalMixture,
            {
                "n_std": 3.0,
                "p1": 0.9,
                "mu1": 0.0,
                "sigma1": 0.1,
                "mu2": -0.5,
                "sigma2": 0.3,
            },
        ),
        (lcm.shocks.ar1.Rouwenhorst, {"rho": 0.9, "sigma": 1.0, "mu": 0.0}),
        (
            lcm.shocks.ar1.TauchenNormalMixture,
            {
                "rho": 0.9,
                "mu": 1.0,
                "n_std": 3.0,
                "p1": 0.9,
                "mu1": 0.0,
                "sigma1": 0.1,
                "mu2": -0.5,
                "sigma2": 0.3,
            },
        ),
    ],
)
def test_shock_grid_fully_specified_with_all_params(grid_cls, kwargs):
    """ShockGrid with all params provided is fully specified."""
    grid = grid_cls(n_points=3, **kwargs)
    assert grid.is_fully_specified
    result = grid.get_gridpoints()
    assert result.shape == (3,)


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


@pytest.mark.parametrize(
    "grid_cls",
    [lcm.shocks.iid.Normal, lcm.shocks.iid.LogNormal, lcm.shocks.ar1.Tauchen],
    ids=lambda c: c.__name__,
)
def test_even_n_points_rejected_for_gauss_hermite(grid_cls):
    """Gauss-Hermite grids reject even n_points (no node at the mean)."""
    with pytest.raises(GridInitializationError, match="n_points must be odd"):
        grid_cls(n_points=4, gauss_hermite=True)


@pytest.mark.parametrize(
    "grid_cls_and_kwargs",
    [
        (lcm.shocks.iid.Normal, {"gauss_hermite": False, "n_std": 3.0}),
        (lcm.shocks.iid.LogNormal, {"gauss_hermite": False, "n_std": 3.0}),
        (lcm.shocks.ar1.Tauchen, {"gauss_hermite": False, "n_std": 3.0}),
        (lcm.shocks.iid.Uniform, {}),
        (lcm.shocks.iid.NormalMixture, {}),
        (lcm.shocks.ar1.Rouwenhorst, {}),
        (lcm.shocks.ar1.TauchenNormalMixture, {}),
    ],
)
def test_even_n_points_accepted_for_non_gauss_hermite(grid_cls_and_kwargs):
    """Non-GH grids accept even n_points (linspace grids work fine without a node at
    the mean)."""
    grid_cls, extra_kw = grid_cls_and_kwargs
    grid_cls(n_points=4, **extra_kw)  # should not raise


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


def test_normal_gauss_hermite_weights_sum_to_one():
    """GH weights for iid Normal sum to 1."""
    grid = lcm.shocks.iid.Normal(n_points=7, gauss_hermite=True, mu=0.0, sigma=1.0)
    P = grid.get_transition_probs()
    aaae(P[0].sum(), 1.0, decimal=DECIMAL_PRECISION)


def test_normal_linspace_transition_probs_rows_sum_to_one():
    """Non-GH Normal iid transition probability rows sum to 1."""
    grid = lcm.shocks.iid.Normal(
        n_points=7, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=3.0
    )
    P = grid.get_transition_probs()
    row_sums = P.sum(axis=1)
    aaae(row_sums, jnp.ones(7), decimal=DECIMAL_PRECISION)


def test_lognormal_linspace_transition_probs_rows_sum_to_one():
    """Non-GH LogNormal iid transition probability rows sum to 1."""
    grid = lcm.shocks.iid.LogNormal(
        n_points=7, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=3.0
    )
    P = grid.get_transition_probs()
    row_sums = P.sum(axis=1)
    aaae(row_sums, jnp.ones(7), decimal=DECIMAL_PRECISION)


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


def test_tauchen_linspace_transition_probs_rows_sum_to_one():
    """Each row of the non-GH Tauchen transition matrix sums to 1."""
    grid = lcm.shocks.ar1.Tauchen(
        n_points=7, gauss_hermite=False, rho=0.9, sigma=0.5, mu=1.0, n_std=3.0
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


_NORMAL_MIXTURE_KWARGS = {
    "n_std": 3.0,
    "p1": 0.9,
    "mu1": 0.0,
    "sigma1": 0.1,
    "mu2": -0.5,
    "sigma2": 0.3,
}


def test_normal_mixture_transition_probs_rows_sum_to_one():
    """NormalMixture transition probability rows sum to 1."""
    grid = lcm.shocks.iid.NormalMixture(n_points=7, **_NORMAL_MIXTURE_KWARGS)
    P = grid.get_transition_probs()
    row_sums = P.sum(axis=1)
    aaae(row_sums, jnp.ones(7), decimal=DECIMAL_PRECISION)


def test_iid_normal_mixture_stationary_moments():
    """IID NormalMixture stationary mean and std match mixture moments."""
    kwargs = _NORMAL_MIXTURE_KWARGS
    grid = lcm.shocks.iid.NormalMixture(n_points=21, **kwargs)
    got_mean, got_std = _stationary_moments(
        grid.get_gridpoints(), grid.get_transition_probs()
    )
    p1 = kwargs["p1"]
    mu1, mu2 = kwargs["mu1"], kwargs["mu2"]
    sigma1, sigma2 = kwargs["sigma1"], kwargs["sigma2"]
    expected_mean = p1 * mu1 + (1 - p1) * mu2
    expected_var = (
        p1 * (sigma1**2 + mu1**2) + (1 - p1) * (sigma2**2 + mu2**2) - expected_mean**2
    )
    aaae(got_mean, expected_mean, decimal=1)
    aaae(got_std, float(jnp.sqrt(expected_var)), decimal=1)


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


def test_tauchen_normal_mixture_transition_probs_rows_sum_to_one():
    """TauchenNormalMixture transition probability rows sum to 1."""
    grid = lcm.shocks.ar1.TauchenNormalMixture(
        n_points=7, **_TAUCHEN_NORMAL_MIXTURE_KWARGS
    )
    P = grid.get_transition_probs()
    row_sums = P.sum(axis=1)
    aaae(row_sums, jnp.ones(7), decimal=DECIMAL_PRECISION)


def test_tauchen_normal_mixture_centers_on_unconditional_mean():
    """TauchenNormalMixture gridpoints center on (mu + mean_eps) / (1 - rho)."""
    kwargs = _TAUCHEN_NORMAL_MIXTURE_KWARGS
    grid = lcm.shocks.ar1.TauchenNormalMixture(n_points=11, **kwargs)
    points = grid.get_gridpoints()
    midpoint = (points[0] + points[-1]) / 2
    mean_eps = kwargs["p1"] * kwargs["mu1"] + (1 - kwargs["p1"]) * kwargs["mu2"]
    expected = (kwargs["mu"] + mean_eps) / (1 - kwargs["rho"])
    aaae(midpoint, expected, decimal=10)


def test_tauchen_normal_mixture_stationary_moments_and_autocorrelation():
    """TauchenNormalMixture stationary mean, std, and autocorrelation match theory."""
    kwargs = _TAUCHEN_NORMAL_MIXTURE_KWARGS
    grid = lcm.shocks.ar1.TauchenNormalMixture(n_points=21, **kwargs)
    points = grid.get_gridpoints()
    P = grid.get_transition_probs()

    p1, mu1, sigma1 = kwargs["p1"], kwargs["mu1"], kwargs["sigma1"]
    mu2, sigma2 = kwargs["mu2"], kwargs["sigma2"]
    rho = kwargs["rho"]

    mean_eps = p1 * mu1 + (1 - p1) * mu2
    expected_mean = (kwargs["mu"] + mean_eps) / (1 - rho)
    sigma_eps_sq = (
        p1 * (sigma1**2 + mu1**2) + (1 - p1) * (sigma2**2 + mu2**2) - mean_eps**2
    )
    expected_std = float(jnp.sqrt(sigma_eps_sq / (1 - rho**2)))

    got_mean, got_std = _stationary_moments(points, P)
    aaae(got_mean, expected_mean, decimal=1)
    aaae(got_std, expected_std, decimal=1)

    got_rho = _lag1_autocorrelation(points, P)
    aaae(got_rho, rho, decimal=1)


TAUCHEN_CASES = [
    {"rho": 0.9, "sigma": 0.5, "mu": 0.0, "n_std": 3, "n": 7},
    {"rho": 0.7, "sigma": 1.0, "mu": 2.0, "n_std": 2, "n": 5},
    {"rho": 0.99, "sigma": 0.1, "mu": 0.0, "n_std": 3, "n": 11},
]

ROUWENHORST_CASES = [
    {"rho": 0.9, "sigma": 0.5, "mu": 0.0, "n": 7},
    {"rho": 0.7, "sigma": 1.0, "mu": 2.0, "n": 5},
    {"rho": 0.99, "sigma": 0.1, "mu": 0.0, "n": 11},
]


@pytest.mark.parametrize(
    "case", TAUCHEN_CASES, ids=lambda c: f"rho={c['rho']}_n={c['n']}"
)
def test_tauchen_matches_quantecon(case):
    """Tauchen (non-GH) gridpoints and transition probs match QuantEcon."""
    qe = qe_tauchen(**case)
    grid = lcm.shocks.ar1.Tauchen(
        n_points=case["n"],
        gauss_hermite=False,
        rho=case["rho"],
        sigma=case["sigma"],
        mu=case["mu"],
        n_std=case["n_std"],
    )
    aaae(grid.get_gridpoints(), qe.state_values, decimal=DECIMAL_PRECISION)
    aaae(grid.get_transition_probs(), qe.P, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(
    "case", ROUWENHORST_CASES, ids=lambda c: f"rho={c['rho']}_n={c['n']}"
)
@pytest.mark.filterwarnings("ignore:The API of rouwenhorst has changed:UserWarning")
def test_rouwenhorst_matches_quantecon(case):
    """Rouwenhorst gridpoints and transition probs match QuantEcon."""
    qe = qe_rouwenhorst(**case)
    grid = lcm.shocks.ar1.Rouwenhorst(
        n_points=case["n"],
        rho=case["rho"],
        sigma=case["sigma"],
        mu=case["mu"],
    )
    aaae(grid.get_gridpoints(), qe.state_values, decimal=DECIMAL_PRECISION)
    aaae(grid.get_transition_probs(), qe.P, decimal=DECIMAL_PRECISION)


def _stationary_moments(gridpoints, P):
    """Compute mean and variance from the stationary distribution of a Markov chain."""
    # Stationary distribution: solve pi @ P = pi via eigendecomposition
    eigvals, eigvecs = jnp.linalg.eig(P.T)
    idx = jnp.argmin(jnp.abs(eigvals - 1.0))
    pi = jnp.real(eigvecs[:, idx])
    pi = pi / pi.sum()
    mean = jnp.dot(pi, gridpoints)
    var = jnp.dot(pi, (gridpoints - mean) ** 2)
    return float(mean), float(jnp.sqrt(var))


def _lag1_autocorrelation(gridpoints, P):
    """Compute lag-1 autocorrelation from gridpoints and transition matrix."""
    eigvals, eigvecs = jnp.linalg.eig(P.T)
    idx = jnp.argmin(jnp.abs(eigvals - 1.0))
    pi = jnp.real(eigvecs[:, idx])
    pi = pi / pi.sum()
    mean = jnp.dot(pi, gridpoints)
    var = jnp.dot(pi, (gridpoints - mean) ** 2)
    # E[X_t * X_{t+1}] = sum_i sum_j pi_i * P_ij * x_i * x_j
    cross = float(jnp.sum(pi[:, None] * P * gridpoints[:, None] * gridpoints[None, :]))
    return (cross - float(mean) ** 2) / float(var)


@pytest.mark.parametrize("gauss_hermite", [True, False])
def test_iid_normal_stationary_moments(gauss_hermite):
    """IID Normal stationary mean and std match mu and sigma."""
    mu, sigma = 1.5, 0.8
    extra = {"gauss_hermite": gauss_hermite}
    if not gauss_hermite:
        extra["n_std"] = 4.0
    grid = lcm.shocks.iid.Normal(n_points=21, mu=mu, sigma=sigma, **extra)
    got_mean, got_std = _stationary_moments(
        grid.get_gridpoints(), grid.get_transition_probs()
    )
    aaae(got_mean, mu, decimal=1)
    aaae(got_std, sigma, decimal=1)


@pytest.mark.parametrize("gauss_hermite", [True, False])
def test_iid_lognormal_stationary_moments(gauss_hermite):
    """IID LogNormal stationary log-mean and log-std match mu and sigma."""
    mu, sigma = 0.5, 0.3
    extra = {"gauss_hermite": gauss_hermite}
    if not gauss_hermite:
        extra["n_std"] = 4.0
    grid = lcm.shocks.iid.LogNormal(n_points=21, mu=mu, sigma=sigma, **extra)
    points = grid.get_gridpoints()
    P = grid.get_transition_probs()
    got_mean, got_std = _stationary_moments(jnp.log(points), P)
    aaae(got_mean, mu, decimal=1)
    aaae(got_std, sigma, decimal=1)


@pytest.mark.parametrize(
    ("grid_cls", "extra_kw"),
    [
        (lcm.shocks.ar1.Tauchen, {"gauss_hermite": True}),
        (lcm.shocks.ar1.Tauchen, {"gauss_hermite": False, "n_std": 3.0}),
        (lcm.shocks.ar1.Rouwenhorst, {}),
    ],
    ids=["tauchen-gh", "tauchen-linspace", "rouwenhorst"],
)
def test_ar1_stationary_moments_and_autocorrelation(grid_cls, extra_kw):
    """AR(1) stationary mean, std, and lag-1 autocorrelation match theory."""
    rho, sigma, mu = 0.8, 0.4, 1.0
    grid = grid_cls(n_points=21, rho=rho, sigma=sigma, mu=mu, **extra_kw)
    points = grid.get_gridpoints()
    P = grid.get_transition_probs()

    expected_mean = mu / (1 - rho)
    expected_std = sigma / jnp.sqrt(1 - rho**2)

    got_mean, got_std = _stationary_moments(points, P)
    aaae(got_mean, expected_mean, decimal=1)
    aaae(got_std, expected_std, decimal=1)

    got_rho = _lag1_autocorrelation(points, P)
    aaae(got_rho, rho, decimal=1)
