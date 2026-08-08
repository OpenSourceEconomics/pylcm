"""Which branch owns a query follows the values, and not how segments are blocked.

`segment_block_size` partitions the envelope's segment axis. Value parity under
that partition is a numerical statement and tolerates rounding; *ownership* is
not — the published policy and marginal come from one branch, and a partition
that changed which one would change the decision, not its precision.

Two branches separated by a single ULP is the sharp case, and the envelope
decides it: each candidate's read carries a measured error bound, so where the
bounds do not overlap the higher branch owns however small the gap. A branch is
never handed the query for want of resolution its own arithmetic already has.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import envelope_at_query

_LOSER_POLICY = 10.0
_WINNER_POLICY = 20.0

_BASE_VALUE = -1.0


def _ulp_gap(gap_in_ulp: int) -> float:
    """The distance spanned by `gap_in_ulp` steps above the base value."""
    stepped = np.asarray(jnp.asarray(_BASE_VALUE))
    for _ in range(gap_in_ulp):
        stepped = np.nextafter(stepped, np.asarray(stepped.dtype.type(np.inf)))
    return float(stepped - _BASE_VALUE)


def _owner_policy(*, block_size: int, gap: float) -> float:
    """The policy published at the shared query, at a given segment block size."""
    grid = jnp.asarray([0.0, 1.0, 0.0, 1.0])
    winner = _BASE_VALUE + gap
    value = jnp.asarray([_BASE_VALUE, _BASE_VALUE, winner, winner])
    policy = jnp.asarray([_LOSER_POLICY, _LOSER_POLICY, _WINNER_POLICY, _WINNER_POLICY])
    segment_id = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    _value, owner_policy, _marginal = envelope_at_query(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal=jnp.ones_like(grid),
        segment_id=segment_id,
        x_query=jnp.asarray([0.5]),
        segment_block_size=block_size,
    )
    return float(np.asarray(owner_policy)[0])


@pytest.mark.parametrize("block_size", [0, 1])
@pytest.mark.parametrize("gap_in_ulp", [1, 2, 4])
def test_the_higher_branch_owns_a_gap_of_a_few_ulp(gap_in_ulp, block_size):
    """A branch higher by one ULP owns the query, under either evaluation path."""
    owner = _owner_policy(block_size=block_size, gap=_ulp_gap(gap_in_ulp))
    assert owner == _WINNER_POLICY


@pytest.mark.parametrize("block_size", [0, 1])
def test_the_higher_branch_owns_a_gap_far_above_the_resolution(block_size):
    """A branch higher by much more than a rounding owns the query."""
    assert _owner_policy(block_size=block_size, gap=1e-5) == _WINNER_POLICY
