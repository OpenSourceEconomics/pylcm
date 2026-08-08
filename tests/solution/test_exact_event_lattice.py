"""An envelope switch hands over on the first state the incoming branch owns.

The refined row carries a switch as a duplicated abscissa holding both branches'
records, so a right-continuous read at that abscissa returns the incoming branch.
That makes the abscissa structural rather than cosmetic: placing it on a state the
*outgoing* branch still owns publishes the incoming policy one representable state
too early, and the exact crossing almost never lands on a float, so the placement
has to be certified rather than rounded to.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.certified_sign import certified_margin_sign
from _lcm.egm.upper_envelope.segment_envelope import refine_envelope_exact
from tests.conftest import X64_ENABLED


def _row_read(grid: np.ndarray, payload: np.ndarray, query: float) -> float:
    """Right-continuous read of a refined row, as the simulation reader does."""
    upper = int(np.clip(np.searchsorted(grid, query, side="right"), 1, len(grid) - 1))
    lower = upper - 1
    width = grid[upper] - grid[lower]
    if width == 0:
        return float(payload[upper])
    weight = (query - grid[lower]) / width
    return float((1 - weight) * payload[lower] + weight * payload[upper])


def _bracketing_floats(dtype, b0, b1):
    """The two adjacent floats straddling the exact crossing of the two links."""
    exact = Fraction(float(b0)) / (
        Fraction(float(dtype(0.125))) - Fraction(float(b1)) + Fraction(float(b0))
    )
    near = dtype(float(exact))
    if Fraction(float(near)) < exact:
        return near, np.nextafter(near, dtype(np.inf))
    if Fraction(float(near)) > exact:
        return np.nextafter(near, dtype(-np.inf)), near
    return near, near


def _links(dtype, jax_dtype, target):
    """Two links crossing at `target`, with policies 8 (outgoing) and 2 (incoming)."""
    b0 = dtype((0.125 - 0.5) * target)
    b1 = dtype(b0 + dtype(0.5))
    return (
        jnp.asarray([0.0, 1.0, 0.0, 1.0], dtype=jax_dtype),
        jnp.asarray([8.0, 8.0, 2.0, 2.0], dtype=jax_dtype),
        jnp.asarray([0.0, 0.125, b0, b1], dtype=jax_dtype),
        b0,
        b1,
    )


@pytest.mark.parametrize(
    ("dtype", "jax_dtype", "target"),
    [
        (np.float32, jnp.float32, 0.012873900293255133),
        (np.float64, jnp.float64, 0.010957966764418377),
    ],
)
def test_switch_lands_on_the_first_state_the_incoming_branch_owns(
    dtype, jax_dtype, target
) -> None:
    """The state below the crossing keeps the outgoing policy, the one above takes over.

    The two links are certified to swap ownership between one pair of adjacent
    floats. Read at the lower one the row must still publish the outgoing policy
    `8`; read at its successor it must publish the incoming `2`.
    """
    if (jax_dtype is jnp.float64) != X64_ENABLED:
        pytest.skip("dtype is not the configured working precision")
    grid, policy, value, b0, b1 = _links(dtype, jax_dtype, target)

    refined_grid, refined_policy, _value, n_kept = jax.jit(
        lambda g, p, v: refine_envelope_exact(
            endog_grid=g, policy=p, value=v, n_refined=8, max_runs=4
        )
    )(grid, policy, value)
    n = int(n_kept)
    assert n <= 8

    lower, upper = _bracketing_floats(dtype, b0, b1)
    signs = [
        int(
            certified_margin_sign(
                a_x0=grid[0],
                a_x1=grid[1],
                a_v0=value[0],
                a_v1=value[1],
                b_x0=grid[2],
                b_x1=grid[3],
                b_v0=value[2],
                b_v1=value[3],
                x_query=jnp.asarray(q, dtype=jax_dtype),
            )
        )
        for q in (lower, upper)
    ]
    assert signs == [1, -1], "the geometry must certifiably swap across this pair"

    live_grid = np.asarray(refined_grid[:n])
    live_policy = np.asarray(refined_policy[:n])
    assert _row_read(live_grid, live_policy, lower) == 8.0
    assert _row_read(live_grid, live_policy, upper) == 2.0
