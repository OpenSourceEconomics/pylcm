"""Breakpoint-aware interpolation never averages across a value jump.

A child value function with a declared jump (e.g. a benefit cliff) is smooth
on each side of the breakpoint but discontinuous across it. Plain linear
interpolation between the two grid points straddling the breakpoint blends the
two sides, handing queries near the cliff a value that belongs to neither.
The breakpoint-aware read restricts each query's stencil to its own side:
queries between the breakpoint and the first grid point beyond it extrapolate
one-sidedly from their own side's boundary segment.
"""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.interp import (
    interp_across_breakpoints,
    interp_across_breakpoints_on_prepared_grid,
    interp_on_prepared_grid,
    prepare_padded_grid,
)


def test_queries_near_a_jump_read_their_own_side():
    """Queries on each side of the breakpoint get that side's values exactly.

    The value is `x` below the breakpoint at 5.0 and `x - 100` above it, on a
    grid whose points straddle the breakpoint (4.0 and 6.0). A query at 4.9
    must read ~4.9 (own-side extrapolation), not the linear blend ~-40; a
    query at 5.1 must read ~-94.9.
    """
    grid = jnp.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    values = jnp.where(grid < 5.0, grid, grid - 100.0)
    breakpoints = jnp.array([5.0])

    queries = jnp.array([4.9, 5.1, 3.0, 7.0])
    read = interp_across_breakpoints(
        queries=queries, grid=grid, values=values, breakpoints=breakpoints
    )

    np.testing.assert_allclose(np.asarray(read), [4.9, -94.9, 3.0, -93.0], atol=1e-6)


def test_sparse_interval_never_reads_across_a_second_breakpoint():
    """A query in an interval with one grid node reads that node, constant.

    With breakpoints at 5.0 and 7.0 and a grid of [0, 4, 6, 10], the interval
    (5, 7) contains only the node 6.0. A query at 6.5 must read 6.0's value
    (own-side constant extrapolation), never a stencil like [4.0, 6.0] that
    crosses the 5.0 breakpoint into the wrong side.
    """
    grid = jnp.array([0.0, 4.0, 6.0, 10.0])
    values = jnp.array([0.0, 4.0, 100.0, 0.0])
    breakpoints = jnp.array([5.0, 7.0])

    read = interp_across_breakpoints(
        queries=jnp.array([6.5]), grid=grid, values=values, breakpoints=breakpoints
    )

    np.testing.assert_allclose(np.asarray(read), [100.0], atol=1e-6)


def test_padded_row_read_is_side_faithful():
    """The NaN-padded-row variant reads each query from its own side.

    Same jumped values as the clean-grid case, on a row with a NaN tail: the
    query below the breakpoint must not blend the above-side node into its
    read, and the padding must stay inert.
    """
    xp = jnp.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, jnp.nan, jnp.nan])
    fp = jnp.where(xp < 5.0, xp, xp - 100.0)
    search_grid, valid_length = prepare_padded_grid(xp)

    read = jax.vmap(
        lambda x: interp_across_breakpoints_on_prepared_grid(
            x_query=x,
            search_grid=search_grid,
            valid_length=valid_length,
            xp=xp,
            fp=fp,
            breakpoints=jnp.array([5.0]),
            breakpoint_side_values=jnp.array([[5.0, -95.0]]),
        )
    )(jnp.array([4.9, 5.1, 3.0, 7.0]))

    np.testing.assert_allclose(np.asarray(read), [4.9, -94.9, 3.0, -93.0], atol=1e-6)


def test_padded_row_read_keeps_hermite_accuracy_away_from_jumps():
    """Away from every breakpoint, the jumped-row read equals the Hermite read.

    A curved row read linearly is biased between nodes; the side-faithful
    read must not pay that bias where its bracket never touches a jump, so
    given the row's slopes it reproduces the Hermite interpolant there.
    """
    xp = jnp.array([1.0, 2.0, 3.0, 4.0, 9.0, 10.0, jnp.nan, jnp.nan])
    fp = jnp.where(xp < 5.0, -(xp**-1.0), 50.0 - (xp**-1.0))
    slopes = xp**-2.0
    search_grid, valid_length = prepare_padded_grid(xp)

    smooth_query = jnp.asarray(2.5)
    hermite = interp_on_prepared_grid(
        x_query=smooth_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=slopes,
    )
    side_faithful = interp_across_breakpoints_on_prepared_grid(
        x_query=smooth_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=slopes,
        breakpoints=jnp.array([5.0]),
        breakpoint_side_values=jnp.array([[-0.2, 49.8]]),
    )
    linear = jnp.interp(smooth_query, xp[:6], fp[:6])

    np.testing.assert_allclose(float(side_faithful), float(hermite), atol=1e-9)
    assert abs(float(hermite) - float(linear)) > 1e-3


def test_straddling_bracket_interpolates_to_the_side_limit():
    """A breakpoint-straddling bracket anchors on the exact boundary limit.

    On a coarse row whose bracket [4, 8] straddles a jump at 5.0, a query at
    4.5 interpolates between the node (4, 16) and the left limit (5, 25) —
    it must neither extrapolate from the below-side boundary segment nor
    blend the above-side node into the read. A query at 6.0 interpolates
    between the right limit (5, 1025) and the node (8, 1064).
    """
    xp = jnp.array([0.0, 2.0, 4.0, 8.0, 10.0, jnp.nan])
    fp = jnp.where(xp < 5.0, xp**2, xp**2 + 1000.0)
    search_grid, valid_length = prepare_padded_grid(xp)

    read = jax.vmap(
        lambda x: interp_across_breakpoints_on_prepared_grid(
            x_query=x,
            search_grid=search_grid,
            valid_length=valid_length,
            xp=xp,
            fp=fp,
            breakpoints=jnp.array([5.0]),
            breakpoint_side_values=jnp.array([[25.0, 1025.0]]),
        )
    )(jnp.array([4.5, 6.0]))

    expected_below = 16.0 + (25.0 - 16.0) / (5.0 - 4.0) * (4.5 - 4.0)
    expected_above = 1025.0 + (1064.0 - 1025.0) / (8.0 - 5.0) * (6.0 - 5.0)
    np.testing.assert_allclose(
        np.asarray(read), [expected_below, expected_above], atol=1e-6
    )


def test_sparse_interval_interpolates_between_facing_limits():
    """An interval holding no grid node reads between its two boundary limits.

    With breakpoints at 5.0 and 7.0 and nodes at [0, 4, 8, 10], the interval
    (5, 7) contains no node: a query at 6.0 interpolates between 5.0's right
    limit and 7.0's left limit, touching neither side's nodes.
    """
    xp = jnp.array([0.0, 4.0, 8.0, 10.0, jnp.nan, jnp.nan])
    fp = jnp.array([0.0, 4.0, -100.0, -98.0, jnp.nan, jnp.nan])
    search_grid, valid_length = prepare_padded_grid(xp)

    read = interp_across_breakpoints_on_prepared_grid(
        x_query=jnp.asarray(6.0),
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        breakpoints=jnp.array([5.0, 7.0]),
        breakpoint_side_values=jnp.array([[5.0, 40.0], [42.0, -101.0]]),
    )

    np.testing.assert_allclose(float(read), 41.0, atol=1e-6)


def test_anchored_bracket_uses_the_node_slope_when_given():
    """An anchored read with slopes recovers a quadratic row exactly.

    For `f(x) = x²` with a jump declared at 5.0 on a bracket [4, 8], the
    query at 4.5 anchors on the node (4, 16) with exact slope 8 and the left
    limit (5, 25): the quadratic interpolant through them reproduces
    `4.5² = 20.25`, where the linear secant would give 20.5.
    """
    xp = jnp.array([0.0, 2.0, 4.0, 8.0, 10.0, jnp.nan])
    fp = jnp.where(xp < 5.0, xp**2, xp**2 + 1000.0)
    slopes = 2.0 * xp
    search_grid, valid_length = prepare_padded_grid(xp)

    read = interp_across_breakpoints_on_prepared_grid(
        x_query=jnp.asarray(4.5),
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=slopes,
        breakpoints=jnp.array([5.0]),
        breakpoint_side_values=jnp.array([[25.0, 1025.0]]),
    )

    np.testing.assert_allclose(float(read), 20.25, atol=1e-9)
