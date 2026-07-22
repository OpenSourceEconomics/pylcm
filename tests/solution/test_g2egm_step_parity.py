"""The four-segment G2EGM step reproduces the brute solve, corners included.

The single-segment (`ucon`-only) step matches the dense grid-search solve only where
the borrowing and deposit constraints are slack; at the constrained corners (low
liquid, low pension) its unconstrained cloud does not cover the target and its
extrapolation is far from the brute value. The four-segment upper envelope adds the
`dcon`, `acon`, and `con` clouds, so those corners are covered by their own segments.
On the region the post-decision grid can reach, the envelope matches the brute solve
and is dramatically better than `ucon`-only at the corners.

The objective masks any candidate whose reconstructed post-decision balance leaves the
post-decision grid (the continuation reader clamps to the boundary, so an out-of-grid
candidate would otherwise receive a fabricated value), and the direct-Bellman hole-fill
covers common-grid targets no segment reaches.

Residual: the top pension edge (`n` at the grid maximum) stays an uncovered hole. There
the optimal pension post-decision balance exceeds the grid maximum, so neither a segment
candidate nor the hole-fill has an in-domain continuation — a grid-coverage limit, not
an algorithm gap. The test asserts on the covered region and pins the edge as known.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.two_asset_g2egm_step import g2egm_step
from _lcm.egm.two_asset_step import egm_step
from tests.test_models.deterministic.two_asset import get_model, get_params

_P = {
    "discount_factor": 0.95,
    "crra": 2.0,
    "match_rate": 1.0,
    "return_liquid": 0.02,
    "return_pension": 0.06,
    "wage": 10.0,
}
_M_GRID = jnp.linspace(1.0, 100.0, 12)
_N_GRID = jnp.linspace(0.0, 50.0, 10)
_B_GRID = jnp.linspace(0.0, 46.0, 16)


def _solve():
    model = get_model(n_periods=2)
    params = get_params(n_periods=2, pension_bequest_weight=0.5)
    brute = model.solve(params=params, log_level="off")
    next_value = jnp.asarray(brute[1]["dead"])
    g2egm = np.asarray(
        g2egm_step(
            next_value=next_value,
            m_grid=_M_GRID,
            n_grid=_N_GRID,
            a_grid=jnp.linspace(0.0, 85.0, 18),
            b_grid=_B_GRID,
            consumption_grid=jnp.linspace(0.5, 90.0, 18),
            **_P,
        ).value
    )
    ucon_only = np.asarray(
        egm_step(
            next_value=next_value,
            m_grid=_M_GRID,
            n_grid=_N_GRID,
            a_grid=jnp.linspace(1.0, 85.0, 18),
            b_grid=_B_GRID,
            **_P,
        )
    )
    return g2egm, np.asarray(brute[0]["working"]), ucon_only


# The top pension column (n at the grid maximum) is the uncovered edge hole.
_COVERED = np.s_[:, :9]
_CORNER = np.s_[:4, :5]


def test_g2egm_matches_brute_on_the_covered_region():
    """Where the post-decision grid reaches, the envelope matches the brute solve."""
    g2egm, brute, _ucon = _solve()
    assert np.isfinite(g2egm[_COVERED]).all()
    rel = np.abs(g2egm[_COVERED] - brute[_COVERED]) / np.abs(brute[_COVERED])
    assert np.median(rel) < 0.01
    assert np.percentile(rel, 95) < 0.10


def test_g2egm_covers_constrained_corners_far_better_than_ucon_only():
    """The envelope covers the low-liquid/low-pension corner; ucon-only does not.

    At the constrained corner the multi-segment envelope is within ~9% of brute, while
    the unconstrained-only step is off by tens of percent — the `dcon`/`acon`/`con`
    segments supply the candidates the unconstrained cloud cannot reach.
    """
    g2egm, brute, ucon_only = _solve()
    g2egm_rel = np.abs(g2egm[_CORNER] - brute[_CORNER]) / np.abs(brute[_CORNER])
    ucon_rel = np.abs(ucon_only[_CORNER] - brute[_CORNER]) / np.abs(brute[_CORNER])
    assert g2egm_rel.max() < 0.10
    # The unconstrained-only step is an order of magnitude worse at the corner.
    assert ucon_rel.max() > 0.5
    assert g2egm_rel.max() < 0.25 * ucon_rel.max()


def test_g2egm_value_is_monotone_in_both_assets_on_the_covered_region():
    """More liquid and more pension are weakly valuable on the covered region."""
    g2egm, _brute, _ucon = _solve()
    covered = g2egm[_COVERED]
    assert np.all(np.diff(covered, axis=0) >= -1e-6)
    assert np.all(np.diff(covered, axis=1) >= -1e-6)


def test_g2egm_top_pension_edge_is_a_known_uncovered_hole():
    """The top pension column stays uncovered: its optimal post-state leaves the grid.

    Neither a segment candidate nor the direct-Bellman hole-fill has an in-domain
    continuation there, so the value is `-inf` — a grid-coverage limit, not a gap the
    hole-fill can close.
    """
    g2egm, _brute, _ucon = _solve()
    assert np.all(np.isneginf(g2egm[:, 9]))


def test_g2egm_publishes_a_feasible_policy_on_the_covered_region():
    """The published consumption/deposit policy is feasible where the grid covers.

    On the covered region the optimal consumption is positive, the deposit is
    non-negative, and the two respect the liquid budget `c + d <= m`.
    """
    model = get_model(n_periods=2)
    params = get_params(n_periods=2, pension_bequest_weight=0.5)
    next_value = jnp.asarray(model.solve(params=params, log_level="off")[1]["dead"])
    result = g2egm_step(
        next_value=next_value,
        m_grid=_M_GRID,
        n_grid=_N_GRID,
        a_grid=jnp.linspace(0.0, 85.0, 18),
        b_grid=_B_GRID,
        consumption_grid=jnp.linspace(0.5, 90.0, 18),
        **_P,
    )
    consumption = np.asarray(result.consumption)[_COVERED]
    deposit = np.asarray(result.deposit)[_COVERED]
    liquid = np.asarray(_M_GRID)[:, None] * np.ones_like(consumption)
    assert np.all(consumption > 0.0)
    assert np.all(deposit >= -1e-9)
    assert np.all(consumption + deposit <= liquid + 1e-6)
