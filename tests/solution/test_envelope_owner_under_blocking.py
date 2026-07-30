"""Which branch owns a query does not depend on how the segments are blocked.

`segment_block_size` partitions the envelope's segment axis. Value parity under
that partition is a numerical statement and tolerates rounding; *ownership* is
not — the published policy and marginal come from one branch, and a partition
that changed which one would change the decision, not its precision. Two
branches separated by a single ULP is the case where a partition-dependent
reduction order could flip the winner, so that is where it is pinned.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import envelope_at_query

_LOSER_POLICY = 10.0
_WINNER_POLICY = 20.0


_BASE_VALUE = -1.0


def _gap_above_the_tie_band() -> float:
    """A value gap the envelope resolves by magnitude, at either precision.

    It has to clear both the envelope's relative tie band and the working
    format's own resolution, so the winner is the higher branch by construction.
    """
    return 1e-5


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


@pytest.mark.parametrize("gap_in_ulp", [1, 2, 4])
def test_blocked_and_dense_agree_on_the_owner_inside_the_tie_band(gap_in_ulp):
    """Blocking does not move ownership between two branches separated by rounding.

    A gap of a few ULP is inside the envelope's tie band, so the owner is settled
    by the documented right-continuous tie-break rather than by the values. That
    rule has to reach the same branch under either evaluation.
    """
    gap = _ulp_gap(gap_in_ulp)
    assert _owner_policy(block_size=1, gap=gap) == _owner_policy(block_size=0, gap=gap)


def test_blocked_and_dense_agree_on_the_owner_of_a_decided_gap():
    """Where the values decide the owner, blocking reaches the same branch."""
    gap = _gap_above_the_tie_band()
    assert _owner_policy(block_size=1, gap=gap) == _owner_policy(block_size=0, gap=gap)


def test_the_higher_branch_owns_a_gap_the_values_decide():
    """Above the tie band the higher branch owns, so a flip would be detectable."""
    assert _owner_policy(block_size=0, gap=_gap_above_the_tie_band()) == _WINNER_POLICY
