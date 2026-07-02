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
        )
    )(jnp.array([4.9, 5.1, 3.0, 7.0]))

    np.testing.assert_allclose(np.asarray(read), [4.9, -94.9, 3.0, -93.0], atol=1e-6)
