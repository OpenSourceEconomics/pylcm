"""A branch dominating by many orders of magnitude is the one published.

Dekker's split reaches its halves by scaling its operand, and that scaling leaves
the format's range well below the point where the product it serves would. A
comparison whose operands sit in that band is still an ordinary comparison — the
margin is enormous and its sign is never in doubt — so the envelope owes it an
owner rather than an abstention.

Getting this wrong has a direction that matters. Publishing the *ordinary* branch
would be a fail-open: the case where a large true margin is read as no margin is
exactly the case where a branch dominated by many orders of magnitude can be
handed the query. Publishing NaN would be the opposite failure, an abstention on
a decision that was in fact established. This file rules out both, and does so
identically however the segment axis is blocked — a query whose answer depended
on the partition would be a defect on top of whichever it hid.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import envelope_at_query
from tests.conftest import EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(
    reason=EXACT_KERNEL_SKIP_REASON
)

_X0 = 100.0
_X1 = 1100.0
_QUERY = 434.0

_DOMINANT_POLICY = 2.0
_ORDINARY_POLICY = 0.5


def _beyond_the_split_scaling() -> float:
    """A value whose split intermediate overflows while its own product does not.

    The split scales by roughly the square root of the format's precision, so an
    operand this size sends that intermediate out of range even though the
    determinant it serves stays an ordinary finite number.

    The result is returned in the working dtype, so that the level the setup
    feeds in and the level the assertions expect back are the same number.
    """
    dtype = jnp.zeros(1).dtype
    largest = float(jnp.finfo(dtype).max)
    return float(jnp.asarray(largest / 1_000.0, dtype))


def _published(*, block_size: int) -> tuple[float, float, float]:
    """Value, policy, and marginal at a query one huge branch dominates."""
    dominant = _beyond_the_split_scaling()
    grid = jnp.asarray([_X0, _X1, _X0, _X1])
    value = jnp.asarray([dominant, dominant, 1.0, 1.0])
    policy = jnp.asarray(
        [
            _DOMINANT_POLICY,
            _DOMINANT_POLICY,
            _ORDINARY_POLICY,
            _ORDINARY_POLICY,
        ]
    )
    marginal = jnp.zeros(4)
    segment_id = jnp.asarray([0.0, 0.0, 1.0, 1.0])

    got_value, got_policy, got_marginal = envelope_at_query(
        endog_grid=grid,
        policy=policy,
        value=value,
        marginal=marginal,
        segment_id=segment_id,
        x_query=jnp.asarray([_QUERY]),
        segment_block_size=block_size,
    )
    return float(got_value[0]), float(got_policy[0]), float(got_marginal[0])


@pytest.mark.parametrize("block_size", [0, 1, 2, 3])
def test_the_dominant_branch_supplies_the_policy(block_size: int):
    """The winner's policy is published, not the branch it dominates."""
    _value, policy, _marginal = _published(block_size=block_size)

    assert policy == _DOMINANT_POLICY


@pytest.mark.parametrize("block_size", [0, 1, 2, 3])
def test_the_dominant_branch_supplies_the_value(block_size: int):
    """The value channel reports the winner's own level, finite and unpoisoned."""
    value, _policy, _marginal = _published(block_size=block_size)

    assert value == _beyond_the_split_scaling()


@pytest.mark.parametrize("block_size", [0, 1, 2, 3])
def test_the_marginal_channel_survives_the_split_scaling(block_size: int):
    """A NaN tail from the split would surface here first."""
    _value, _policy, marginal = _published(block_size=block_size)

    assert not np.isnan(marginal)
    assert marginal == 0.0
