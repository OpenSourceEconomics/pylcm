"""The jump-aware continuation read must not bridge a neighboring jump.

`_jump_aware_interp` reconstructs a one-sided limit at each cliff by extrapolating
from the nearest same-side grid nodes. When two cliffs are close relative to the
grid spacing, the naive nearest-two-nodes stencil reaches across the neighboring
cliff and bridges its discontinuity, so the recovered one-sided value is wrong.
Each side's stencil must stay within the interval bounded by the adjacent cliffs.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.bqsegm_step import (
    _bounded_limit_above,
    _bounded_limit_below,
    _jump_aware_interp,
)


def test_jump_aware_interp_does_not_bridge_a_neighboring_jump():
    """A query just above the first of two close cliffs takes its limit from the
    middle segment, not from nodes across the second cliff."""
    grid = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # Two close cliffs at 2.5 and 3.5; node index 3 is the lone middle-segment node.
    # Below first cliff -> 10, middle -> 50, above second cliff -> 100.
    values = jnp.array([10.0, 10.0, 10.0, 50.0, 100.0, 100.0, 100.0])
    breakpoints = jnp.array([2.5, 3.5])

    # 2.6 sits in the middle segment (level 50). Its above-first-cliff limit must be
    # the middle segment's value, not a bridge toward the above-second-cliff 100.
    query = jnp.array([2.6])
    out = np.asarray(_jump_aware_interp(query, grid, values, breakpoints, "otherwise"))
    np.testing.assert_allclose(out, [50.0], atol=1e-6)


def test_jump_aware_interp_below_side_does_not_bridge_previous_jump():
    """The below-side limit of the second cliff stays within the middle segment."""
    grid = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    values = jnp.array([10.0, 10.0, 10.0, 50.0, 100.0, 100.0, 100.0])
    breakpoints = jnp.array([2.5, 3.5])

    # 3.4 sits in the middle segment (level 50). Its below-second-cliff limit must be
    # the middle segment's value, not a bridge back toward the below-first-cliff 10.
    query = jnp.array([3.4])
    out = np.asarray(_jump_aware_interp(query, grid, values, breakpoints, "otherwise"))
    np.testing.assert_allclose(out, [50.0], atol=1e-6)


def test_bounded_limit_below_uses_only_nodes_above_the_previous_cliff():
    """The below-side limit reads the middle segment, not nodes across the prior
    cliff — the shared helper used by both the jump-aware read and the
    boundary-targeting branch."""
    grid = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    values = jnp.array([10.0, 10.0, 10.0, 50.0, 100.0, 100.0, 100.0])
    # Approaching the second cliff (3.5) from below, bounded above the first (2.5):
    # only node 3 (value 50) qualifies, so the limit is the middle level 50.
    out = float(_bounded_limit_below(grid, values, limit=3.5, prev_limit=2.5, n=7))
    np.testing.assert_allclose(out, 50.0, atol=1e-6)


def test_bounded_limit_above_uses_only_nodes_below_the_next_cliff():
    """The above-side limit reads the middle segment, not nodes across the next
    cliff."""
    grid = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    values = jnp.array([10.0, 10.0, 10.0, 50.0, 100.0, 100.0, 100.0])
    # Approaching the first cliff (2.5) from above, bounded below the second (3.5):
    # only node 3 (value 50) qualifies, so the limit is the middle level 50.
    out = float(_bounded_limit_above(grid, values, limit=2.5, next_limit=3.5, n=7))
    np.testing.assert_allclose(out, 50.0, atol=1e-6)
