"""Value jumps ride on padded rows as duplicated abscissae.

A row holding a jump carries its location twice — once with the left limit,
once with the right — so the ordinary padded-row read brackets against the
exact one-sided values and can never blend the two sides of the jump.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.interp import interp_on_padded_grid


def test_duplicated_abscissa_jump_reads_each_side_exactly():
    """Queries around a duplicated-abscissa jump read their own side.

    The row is `x` below the jump at 5.0 and `x - 100` at and above it; the
    node 5.0 appears twice with the two limits. Queries below interpolate on
    the left branch, queries at or above the jump on the right branch.
    """
    xp = jnp.array([0.0, 2.0, 4.0, 5.0, 5.0, 6.0, 8.0, jnp.nan])
    fp = jnp.array([0.0, 2.0, 4.0, 5.0, -95.0, -94.0, -92.0, jnp.nan])

    read = interp_on_padded_grid(x_query=jnp.array([4.5, 5.0, 5.5, 3.0]), xp=xp, fp=fp)

    np.testing.assert_allclose(np.asarray(read), [4.5, -95.0, -94.5, 3.0], atol=1e-9)
