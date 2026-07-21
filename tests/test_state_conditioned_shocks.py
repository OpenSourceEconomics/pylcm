"""Tests for the state-conditioned-shock direct-CDF row builders (v1 core).

Pins the load-bearing correctness the pro-comp-method audit (2026-07-14) demanded:
reduction to pylcm's own transition math at the nodes, F1-safety (direct-CDF, not row
interpolation), F2-safety (sigma genuinely moves the kernel on fixed nodes), and
analytic conditional moments. The DAG plumbing is a later increment; this is the kernel.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.processes.state_conditioned import (
    StateConditioned,
    conditioned_row,
    gather_sigma,
    iid_normal_row,
    sigma_array_by_code,
    tauchen_row,
)
from lcm import DiscreteGrid, NormalIIDProcess, TauchenAR1Process, categorical
from lcm.typing import ScalarInt


@categorical(ordered=True)
class Uncertainty:
    low: ScalarInt
    high: ScalarInt


def test_tauchen_row_reduces_to_pylcm_at_nodes():
    """At a node with sigma == grid_sigma, tauchen_row == the pylcm Tauchen row."""
    p = TauchenAR1Process(
        n_points=7, gauss_hermite=False, rho=0.9, sigma=0.2, mu=0.0, n_std=3.0
    )
    nodes = p.get_gridpoints()
    P = p.get_transition_probs()
    for i in range(7):
        row = tauchen_row(nodes, rho=0.9, sigma=0.2, from_value=float(nodes[i]))
        np.testing.assert_allclose(np.asarray(row), np.asarray(P[i]), atol=1e-10)


def test_iid_row_reduces_to_pylcm():
    """iid_normal_row == the pylcm CDF-binned NormalIID row (rows identical)."""
    p = NormalIIDProcess(n_points=7, gauss_hermite=False, mu=0.0, sigma=0.2, n_std=3.0)
    nodes = p.get_gridpoints()
    P = p.get_transition_probs()
    row = iid_normal_row(nodes, mu=0.0, sigma=0.2)
    np.testing.assert_allclose(np.asarray(row), np.asarray(P[0]), atol=1e-10)


@pytest.mark.parametrize("mu", [0.2, -0.5])
def test_tauchen_row_reduces_to_pylcm_at_nodes_with_nonzero_mu(mu):
    """F2 regression: the AR(1) intercept must survive into the row.

    Stock pylcm builds its Tauchen rows in demeaned coordinates, where the intercept
    drops out; here the nodes and the from-value are PHYSICAL (centred on mu/(1-rho)),
    so the conditional mean is `mu + rho*y` and dropping `mu` misplaces every row.
    """
    p = TauchenAR1Process(
        n_points=7, gauss_hermite=False, rho=0.9, sigma=0.2, mu=mu, n_std=3.0
    )
    nodes = p.get_gridpoints()
    P = p.get_transition_probs()
    for i in range(7):
        row = tauchen_row(nodes, rho=0.9, sigma=0.2, from_value=float(nodes[i]), mu=mu)
        np.testing.assert_allclose(np.asarray(row), np.asarray(P[i]), atol=1e-10)


@pytest.mark.parametrize("mu", [1.0, -0.3])
def test_iid_row_reduces_to_pylcm_with_nonzero_mu(mu):
    """F2 regression: a non-zero IID mean must survive into the row."""
    p = NormalIIDProcess(n_points=7, gauss_hermite=False, mu=mu, sigma=0.2, n_std=3.0)
    nodes = p.get_gridpoints()
    P = p.get_transition_probs()
    row = iid_normal_row(nodes, mu=mu, sigma=0.2)
    np.testing.assert_allclose(np.asarray(row), np.asarray(P[0]), atol=1e-10)


def test_dropping_mu_would_misplace_the_row():
    """Pins WHY F2 mattered: at mu=1, sigma=0.2 the mu=0 row is a different law that
    reverses a continuation-vs-sure choice paying 1 on the middle node against 0.5."""
    nodes = jnp.array([0.0, 1.0, 2.0])
    correct = np.asarray(iid_normal_row(nodes, mu=1.0, sigma=0.2))
    dropped = np.asarray(iid_normal_row(nodes, mu=0.0, sigma=0.2))
    tv = 0.5 * np.abs(correct - dropped).sum()
    assert tv > 0.9
    assert dropped[1] < 0.5 < correct[1]


@pytest.mark.parametrize("sigma", [0.05, 0.2, 1.0])
def test_rows_are_distributions(sigma):
    nodes = jnp.linspace(-3.0, 3.0, 9)
    r = np.asarray(tauchen_row(nodes, rho=0.8, sigma=sigma, from_value=0.3))
    assert r.sum() == pytest.approx(1.0, abs=1e-12)
    assert (r >= -1e-12).all()


def test_state_conditioning_moves_sigma_on_fixed_nodes():
    """F2-safety: on the SAME fixed nodes, two sigmas give genuinely different variances
    (the failure Rouwenhorst has, which is why it is excluded)."""
    nodes = jnp.linspace(-3.0, 3.0, 41)  # fixed common grid (grid_sigma wide)
    r_lo = np.asarray(iid_normal_row(nodes, mu=0.0, sigma=0.2))
    r_hi = np.asarray(iid_normal_row(nodes, mu=0.0, sigma=1.0))
    g = np.asarray(nodes)
    v_lo = (g**2 * r_lo).sum()
    v_hi = (g**2 * r_hi).sum()
    assert v_hi > 4 * v_lo


def test_f1_direct_cdf_is_not_row_interpolation_at_low_sigma():
    """F1: the direct row differs O(1) from interpolating node rows at low sigma — this
    builder avoids the defect *by construction* (it evaluates the CDF at from_value)."""
    nodes = jnp.array([-1.0, 0.0, 1.0])
    y, rho, sigma = 0.52, 0.9, 0.01
    direct = np.asarray(tauchen_row(nodes, rho, sigma, from_value=y))
    K = np.vstack(
        [np.asarray(tauchen_row(nodes, rho, sigma, from_value=float(x))) for x in nodes]
    )
    interp = 0.48 * K[1] + 0.52 * K[2]
    tv = 0.5 * np.abs(direct - interp).sum()
    assert tv > 0.5
    assert direct[-1] < 0.25 < interp[-1]  # interpolation would reverse a decision


def test_conditional_mean_matches_rho_y_on_fine_grid():
    """Analytic anchor: on a fine grid the Tauchen conditional mean -> rho * y."""
    nodes = jnp.linspace(-6.0, 6.0, 401)
    y, rho, sigma = 0.5, 0.8, 0.3
    r = np.asarray(tauchen_row(nodes, rho, sigma, from_value=y))
    mean = (np.asarray(nodes) * r).sum()
    assert abs(mean - rho * y) < 1e-3


def test_iid_conditional_variance_matches_sigma_on_fine_grid():
    """Analytic anchor: on a fine grid the IID binned variance -> sigma**2."""
    nodes = jnp.linspace(-8.0, 8.0, 801)
    sigma = 0.7
    r = np.asarray(iid_normal_row(nodes, mu=0.0, sigma=sigma))
    g = np.asarray(nodes)
    var = (g**2 * r).sum() - (g * r).sum() ** 2
    assert abs(var - sigma**2) < 1e-3


# --- StateConditioned value object + code-ordered sigma resolver (F5) ------------- #


def test_sigma_array_ordered_by_code_not_insertion():
    """F5/RT5: sigma is ordered by the categorical's integer code, so indexing by the
    conditioning state's code is correct even when `by` is given in another order."""
    grid = DiscreteGrid(Uncertainty)  # codes: low=0, high=1
    by = {"high": 1.0, "low": 0.2}  # deliberately reverse insertion order
    arr = np.asarray(sigma_array_by_code(grid, by))
    np.testing.assert_allclose(arr, [0.2, 1.0])  # [code0=low, code1=high]
    assert float(gather_sigma(jnp.asarray(arr), 0)) == 0.2
    assert float(gather_sigma(jnp.asarray(arr), 1)) == 1.0


def test_sigma_array_missing_category_raises():
    grid = DiscreteGrid(Uncertainty)
    with pytest.raises(ValueError, match="missing categories"):
        sigma_array_by_code(grid, {"low": 0.2})


def test_state_conditioned_dataclass_is_frozen():
    sc = StateConditioned(on="uncertainty", by={"low": 0.2, "high": 1.0})
    assert sc.on == "uncertainty"
    with pytest.raises((AttributeError, TypeError)):
        sc.on = "other"  # ty: ignore[invalid-assignment]


def test_conditioned_row_dispatch_sums_to_one():
    nodes = jnp.linspace(-3.0, 3.0, 7)
    r_iid = conditioned_row(
        family="iid_normal", nodes=nodes, sigma=0.5, from_value=0.0, mu=0.0
    )
    r_tau = conditioned_row(
        family="tauchen", nodes=nodes, sigma=0.5, from_value=0.3, mu=0.0, rho=0.8
    )
    assert np.asarray(r_iid).sum() == pytest.approx(1.0)
    assert np.asarray(r_tau).sum() == pytest.approx(1.0)


def test_conditioned_row_tauchen_requires_rho():
    nodes = jnp.linspace(-3.0, 3.0, 7)
    with pytest.raises(ValueError, match="requires rho"):
        conditioned_row(
            family="tauchen", nodes=nodes, sigma=0.5, from_value=0.0, mu=0.0
        )
