"""Spec for linear interpolation on NaN-padded, weakly-ascending carry rows.

Contract under test — `_lcm.egm.interp.interp_on_padded_grid`:

    interp_on_padded_grid(*, x_query: FloatND, xp: Float1D, fp: Float1D) -> FloatND

`xp` is a NaN-padded, weakly ascending grid row (NaNs only in the tail).
Behavior:

- linear interpolation between neighboring non-NaN nodes,
- edge clamp outside the non-NaN range,
- tie-safe at duplicated abscissae (zero-width brackets from envelope kinks):
  queries strictly below the duplicate interpolate toward the left value, queries
  at or above it use the right value — never a division by the zero bracket,
- `-inf` endpoints (infeasible values) force their bracket interior to `-inf`
  instead of NaN; a query exactly on a finite neighbor returns that value.
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
