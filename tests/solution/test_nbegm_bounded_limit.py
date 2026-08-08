"""One-sided continuation limits never read across their own cliff.

`_bounded_limit_below` estimates the continuation just below a cliff and
`_bounded_limit_above` just above it, each restricted to grid nodes between that
cliff and its neighbour so the stencil never spans a discontinuity. When two
cliffs fall inside one grid cell the enclosed interval holds no node at all;
the helpers then read the nearest node on their own side of the cliff rather
than the first node past it, which belongs to the branch they exist to exclude.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.nbegm_step import _bounded_limit_above, _bounded_limit_below

GRID = jnp.arange(6.0)
VALUES = GRID**2


def test_limit_from_below_stays_below_the_cliff_when_no_node_lies_between():
    """With `(0.3, 0.7)` empty, the limit below `0.7` reads the node at `0.0`.

    The first node above `0.3` is `1.0`, on the far side of the cliff at `0.7`,
    so reading it would pair a save-to-`0.7` policy with the continuation of
    the branch above `0.7`.
    """
    below = float(
        _bounded_limit_below(GRID, VALUES, limit=0.7, prev_limit=0.3, n=GRID.shape[0])
    )
    np.testing.assert_allclose(below, 0.0, atol=1e-12)


def test_limit_from_above_stays_above_the_cliff_when_no_node_lies_between():
    """With `(0.3, 0.7)` empty, the limit above `0.3` reads the node at `1.0`."""
    above = float(
        _bounded_limit_above(GRID, VALUES, limit=0.3, next_limit=0.7, n=GRID.shape[0])
    )
    np.testing.assert_allclose(above, 1.0, atol=1e-12)


def test_limit_from_below_extrapolates_from_two_in_interval_nodes():
    """With `(0.5, 3.5)` holding three nodes, the limit is the linear extension.

    The two nodes nearest the cliff are `2.0` and `3.0` with values `4.0` and
    `9.0`, so the limit at `3.5` extends that secant to `11.5`.
    """
    below = float(
        _bounded_limit_below(GRID, VALUES, limit=3.5, prev_limit=0.5, n=GRID.shape[0])
    )
    np.testing.assert_allclose(below, 11.5, atol=1e-12)


def test_limit_from_above_extrapolates_from_two_in_interval_nodes():
    """With `(0.5, 3.5)` holding three nodes, the limit above `0.5` is `0.0`.

    The two nodes nearest the cliff are `1.0` and `2.0` with values `1.0` and
    `4.0`; extending that secant back to `0.5` gives `-0.5`.
    """
    above = float(
        _bounded_limit_above(GRID, VALUES, limit=0.5, next_limit=3.5, n=GRID.shape[0])
    )
    np.testing.assert_allclose(above, -0.5, atol=1e-12)
