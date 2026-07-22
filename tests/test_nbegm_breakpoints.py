"""Breakpoint geometry helpers map thresholds to a liquid interval partition.

Preimages are clamped to the grid span, intervals are lower-closed/upper-open
(a value exactly on a breakpoint belongs to the interval above), and each
interval's representative point recovers the active affine budget segment.
"""

import jax.numpy as jnp
import numpy as np

import lcm
from _lcm.egm.nbegm_breakpoints import (
    affine_coefficients,
    clamp_breakpoints_to_grid,
    interval_midpoints,
    interval_segment_coefficients,
    linear_asset_preimage,
)


def _medicaid_pool():
    """A one-boundary Medicaid asset-test pool with both subsidy pieces."""

    @lcm.case_boundary(
        lcm.boundary(
            "assets", "medicaid_asset_limit", equality="otherwise", kind="jump"
        )
    )
    def medicaid_eligible(assets, medicaid_asset_limit):
        return assets < medicaid_asset_limit

    @lcm.piece("subsidy", when=medicaid_eligible)
    def subsidy_medicaid(subsidy_amount):
        return subsidy_amount

    @lcm.piece("subsidy", otherwise=medicaid_eligible)
    def subsidy_none(subsidy_amount):
        return 0.0 * subsidy_amount

    return {
        "medicaid_eligible": medicaid_eligible,
        "subsidy_medicaid": subsidy_medicaid,
        "subsidy_none": subsidy_none,
    }


def test_clamp_sends_a_degenerate_preimage_outside_the_grid_and_keeps_it_finite():
    """A non-finite breakpoint is clamped to a finite point just outside the grid.

    A boundary whose derived variable has (near-)zero slope in the liquid state has no
    finite asset preimage — `linear_asset_preimage` sends it to ±inf (never crossed) or
    NaN (exactly on the boundary). Clamping keeps every breakpoint finite and just
    outside `[grid_min, grid_max]`, so it collapses to an empty edge interval whose
    midpoint stays finite, while a genuine in-grid breakpoint is left untouched.
    """
    liquid_grid = jnp.asarray([-159132.0, 170433.0, 500000.0])
    breakpoints = jnp.asarray([-jnp.inf, 250000.0, jnp.inf, jnp.nan])

    clamped = clamp_breakpoints_to_grid(
        breakpoints=breakpoints, liquid_grid=liquid_grid
    )

    assert bool(jnp.all(jnp.isfinite(clamped)))
    np.testing.assert_allclose(float(clamped[1]), 250000.0)
    assert float(clamped[0]) < -159132.0
    assert float(clamped[2]) > 500000.0
    assert float(clamped[3]) > 500000.0


def test_clamped_degenerate_breakpoint_yields_finite_interval_midpoints():
    """A degenerate boundary does not make the live interval's midpoint non-finite.

    The interval that holds the most-negative asset point must have a finite midpoint
    even when another boundary is degenerate, so the affine segment recovered there is
    finite rather than poisoned by a `±inf` interval edge.
    """
    liquid_grid = jnp.asarray([-159132.0, 170433.0, 500000.0])
    degenerate = clamp_breakpoints_to_grid(
        breakpoints=jnp.asarray([-jnp.inf, 300000.0]), liquid_grid=liquid_grid
    )

    midpoints = interval_midpoints(liquid_grid=liquid_grid, breakpoints=degenerate)

    assert bool(jnp.all(jnp.isfinite(midpoints)))


def test_affine_coefficients_recover_slope_and_intercept():
    """An affine boundary variable's slope and intercept are recovered exactly."""

    def gross_income(assets):
        return 0.04 * assets + 12_000.0

    slope, intercept = affine_coefficients(gross_income)
    np.testing.assert_allclose(slope, 0.04, atol=1e-9)
    np.testing.assert_allclose(intercept, 12_000.0, atol=1e-6)


def test_linear_asset_preimage_inverts_an_increasing_boundary_variable():
    """A threshold in an increasing affine variable maps to its asset value."""

    def gross_income(assets):
        return 0.04 * assets + 12_000.0

    asset_value = linear_asset_preimage(gross_income, threshold=jnp.asarray(20_000.0))
    np.testing.assert_allclose(asset_value, 200_000.0, atol=1e-3)


def test_linear_asset_preimage_inverts_a_decreasing_boundary_variable():
    """A threshold in a decreasing affine variable maps to its asset value."""

    def countable(assets):
        return -0.5 * assets + 3_000.0

    asset_value = linear_asset_preimage(countable, threshold=jnp.asarray(1_000.0))
    np.testing.assert_allclose(asset_value, 4_000.0, atol=1e-6)


def test_linear_asset_preimage_traces_a_runtime_threshold_and_slope():
    """The preimage is a traced quantity in the runtime threshold and slope."""

    def make_gross_income(rate_of_return):
        def gross_income(assets):
            return rate_of_return * assets + 12_000.0

        return gross_income

    def preimage(rate_of_return, threshold):
        return linear_asset_preimage(
            make_gross_income(rate_of_return), threshold=threshold
        )

    jitted = jnp.vectorize(preimage)
    # z = 0.05 a + 12000, threshold 17000 -> a = 5000 / 0.05 = 100000.
    np.testing.assert_allclose(
        jitted(jnp.asarray(0.05), jnp.asarray(17_000.0)), 100_000.0, atol=1e-3
    )


def test_interval_midpoints_pick_a_representative_point_per_interval():
    """Each interval gets an interior point between its bounding breakpoints."""
    grid = jnp.asarray([0.0, 30.0])
    breakpoints = jnp.asarray([8.0, 20.0])
    np.testing.assert_allclose(
        np.asarray(interval_midpoints(liquid_grid=grid, breakpoints=breakpoints)),
        np.asarray([4.0, 14.0, 25.0]),
        atol=1e-6,
    )


def test_interval_segment_coefficients_recover_the_active_affine_segment():
    """Per interval, a piecewise-affine schedule's active slope and intercept hold."""

    def tax(liquid):
        return (
            0.1 * jnp.minimum(liquid, 8.0)
            + 0.2 * jnp.clip(liquid - 8.0, 0.0, 12.0)
            + 0.3 * jnp.maximum(liquid - 20.0, 0.0)
        )

    midpoints = jnp.asarray([4.0, 14.0, 25.0])
    slopes, intercepts = interval_segment_coefficients(
        schedule=tax, interval_midpoints=midpoints
    )
    np.testing.assert_allclose(np.asarray(slopes), [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(np.asarray(intercepts), [0.0, -0.8, -2.8], atol=1e-5)
