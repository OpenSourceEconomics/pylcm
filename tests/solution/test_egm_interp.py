"""Spec for interpolation on NaN-padded, weakly-ascending carry rows.

Contract under test — `_lcm.egm.interp.interp_on_padded_grid`:

    interp_on_padded_grid(
        *, x_query: FloatND, xp: Float1D, fp: Float1D,
        fp_slopes: Float1D | None = None,
    ) -> FloatND

`xp` is a NaN-padded, weakly ascending grid row (NaNs only in the tail).
Behavior:

- linear interpolation between neighboring non-NaN nodes,
- edge clamp outside the non-NaN range,
- tie-safe at duplicated abscissae (zero-width brackets from envelope kinks):
  queries strictly below the duplicate interpolate toward the left value, queries
  at or above it use the right value — never a division by the zero bracket,
- `-inf` endpoints (infeasible values) force their bracket interior to `-inf`
  instead of NaN; a query exactly on a finite neighbor returns that value,
- with `fp_slopes` (the exact node derivatives), brackets whose endpoint
  values and slopes are all finite get a monotone cubic Hermite correction;
  all other brackets — and the contracts above — behave exactly as in the
  linear case.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm import interp


def test_matches_numpy_interp_on_clean_grid():
    """On an unpadded strictly ascending grid, results equal `np.interp`."""
    xp = jnp.array([1.0, 2.0, 4.0, 8.0])
    fp = jnp.array([0.0, 3.0, 5.0, 6.0])
    x_query = jnp.array([1.0, 1.5, 3.0, 7.9, 8.0])

    got = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)

    expected = np.interp(np.asarray(x_query), np.asarray(xp), np.asarray(fp))
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_nan_tail_is_ignored():
    """NaN padding in the tail does not affect interpolation on the prefix."""
    xp = jnp.array([1.0, 2.0, 4.0, jnp.nan, jnp.nan])
    fp = jnp.array([0.0, 3.0, 5.0, jnp.nan, jnp.nan])
    x_query = jnp.array([1.5, 3.0, 4.0])

    got = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)

    expected = np.interp(np.asarray(x_query), [1.0, 2.0, 4.0], [0.0, 3.0, 5.0])
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_edge_clamp_below_and_above():
    """Queries outside the non-NaN range return the boundary values."""
    xp = jnp.array([1.0, 2.0, 4.0, jnp.nan])
    fp = jnp.array([0.0, 3.0, 5.0, jnp.nan])
    x_query = jnp.array([0.0, 100.0])

    got = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)

    np.testing.assert_allclose(got, jnp.array([0.0, 5.0]), atol=1e-12)


def test_duplicated_abscissa_is_tie_safe():
    """A zero-width bracket (envelope kink) yields finite one-sided values."""
    xp = jnp.array([0.0, 1.0, 1.0, 2.0])
    fp = jnp.array([0.0, 10.0, 20.0, 30.0])
    x_query = jnp.array([0.5, 1.0, 1.5])

    got = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)

    assert not bool(jnp.isnan(got).any())
    np.testing.assert_allclose(got[0], 5.0, atol=1e-12)
    np.testing.assert_allclose(got[2], 25.0, atol=1e-12)
    np.testing.assert_allclose(got[1], 20.0, atol=1e-12)


def test_neg_inf_node_yields_neg_inf_inside_brackets():
    """A `-inf` node forces its bracket interiors to `-inf`, never NaN.

    Value rows may carry `-inf` at a node (e.g. a terminal bequest at zero
    wealth); a query exactly on the finite neighbor returns that neighbor's
    value.
    """
    xp = jnp.array([0.0, 1.0, 2.0])
    fp = jnp.array([-jnp.inf, 3.0, 5.0])
    x_query = jnp.array([0.0, 0.5, 1.0, 1.5])

    got = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)

    np.testing.assert_array_equal(np.asarray(got), [-np.inf, -np.inf, 3.0, 4.0])


def test_all_neg_inf_row_yields_neg_inf_everywhere():
    """An all-`-inf` row (infeasible combo) interpolates to `-inf`, never NaN."""
    xp = jnp.array([0.0, 1.0, 2.0, jnp.nan])
    fp = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, jnp.nan])
    x_query = jnp.array([-1.0, 0.5, 1.0, 3.0])

    got = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)

    assert bool(jnp.isneginf(got).all())


def test_exact_slopes_reproduce_a_monotone_cubic():
    """With exact node slopes, a monotone cubic is interpolated exactly.

    Per bracket the interpolant matches endpoint values and derivatives — four
    degrees of freedom — so a cubic whose slopes pass the monotonicity limiter
    untouched is reproduced to round-off, where linear interpolation is biased.
    """
    xp = jnp.array([1.0, 1.5, 2.0])
    fp = xp**3
    fp_slopes = 3.0 * xp**2
    x_query = jnp.array([1.1, 1.25, 1.7, 1.95])

    got = interp.interp_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    np.testing.assert_allclose(got, np.asarray(x_query) ** 3, atol=1e-12)


def test_slopes_keep_nan_tail_edge_clamp_and_node_values():
    """The Hermite path preserves NaN-tail handling, edge clamps, and nodes."""
    xp = jnp.array([1.0, 2.0, 4.0, jnp.nan])
    fp = jnp.array([0.0, 3.0, 5.0, jnp.nan])
    fp_slopes = jnp.array([3.0, 2.0, 0.5, jnp.nan])
    x_query = jnp.array([0.0, 1.0, 2.0, 4.0, 100.0])

    got = interp.interp_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    np.testing.assert_allclose(got, jnp.array([0.0, 0.0, 3.0, 5.0, 5.0]), atol=1e-12)


def test_slopes_keep_duplicated_abscissa_tie_safe():
    """Zero-width kink brackets yield the same one-sided values as linear."""
    xp = jnp.array([0.0, 1.0, 1.0, 2.0])
    fp = jnp.array([0.0, 10.0, 20.0, 30.0])
    fp_slopes = jnp.array([10.0, 10.0, 10.0, 10.0])
    x_query = jnp.array([1.0])

    got = interp.interp_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    np.testing.assert_allclose(got, jnp.array([20.0]), atol=1e-12)


def test_slopes_on_neg_inf_bracket_fall_back_to_linear_rule():
    """A `-inf` endpoint forces its bracket interior to `-inf` with slopes given.

    Carry rows store slope `0.0` at `-inf` value nodes; the bracket must
    behave exactly as in the linear case — `-inf` inside, the finite
    neighbor's value on it.
    """
    xp = jnp.array([0.0, 1.0, 2.0])
    fp = jnp.array([-jnp.inf, 3.0, 5.0])
    fp_slopes = jnp.array([0.0, 2.0, 2.0])
    x_query = jnp.array([0.0, 0.5, 1.0, 1.5])

    got = interp.interp_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    np.testing.assert_array_equal(np.asarray(got), [-np.inf, -np.inf, 3.0, 4.0])


def test_steep_slopes_are_limited_to_keep_monotonicity():
    """Node slopes far above the secant cannot produce non-monotone output.

    An unlimited cubic Hermite with endpoint slopes ten times the secant
    overshoots inside the bracket; the limiter caps slopes at three times the
    secant, so the interpolant stays within the bracket's value range and
    weakly increasing on increasing data.
    """
    xp = jnp.array([0.0, 1.0])
    fp = jnp.array([0.0, 1.0])
    fp_slopes = jnp.array([10.0, 10.0])
    x_query = jnp.linspace(0.0, 1.0, 101)

    got = interp.interp_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    assert bool(jnp.all(jnp.diff(got) >= 0.0))
    assert bool(jnp.all((got >= 0.0) & (got <= 1.0)))
