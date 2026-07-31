"""A margin that cannot be certified is published as NaN, in either backend.

Two things can stop the envelope naming an owner, and they are not the same. Two
branches within a rounding of each other are *known* to be that close, and either
may be taken: no state between them is demonstrably better, and the tie-break
picks one deterministically. A comparison that left the range where the error-free
transforms are exact is the opposite situation — nothing at all was established,
and the branches may be arbitrarily far apart.

Treating the second as though it were the first is a fail-open: it is precisely
the case where a large true margin can be reported as no margin, so the branch
that is dominated by many orders of magnitude can be handed the query. The
envelope publishes NaN there instead, and does so identically however the segment
axis is blocked — a query whose answer depends on the partition would be a second
defect on top of the first.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import envelope_at_query

_X0 = 100.0
_X1 = 1100.0
_QUERY = 434.0

_DOMINANT_POLICY = 2.0
_ORDINARY_POLICY = 0.5


def _beyond_the_transform_domain() -> float:
    """A finite value whose affine numerator, times a width, overflows the format."""
    largest = float(jnp.finfo(jnp.zeros(1).dtype).max)
    # The determinant is a numerator (value times width) times a width, so a value
    # this size is finite while the product it enters is not.
    return largest / 1_000.0


def _published(*, block_size: int) -> tuple[float, float, float]:
    """Value, policy, and marginal at a query one huge branch dominates."""
    dominant = _beyond_the_transform_domain()
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
def test_an_uncertifiable_margin_publishes_no_policy(block_size: int):
    """No branch supplies a policy where the comparison established nothing."""
    _value, policy, _marginal = _published(block_size=block_size)

    assert np.isnan(policy)


@pytest.mark.parametrize("block_size", [0, 1, 2, 3])
def test_an_uncertifiable_margin_publishes_no_value(block_size: int):
    """The value channel fails with the policy rather than reporting a rival's."""
    value, _policy, _marginal = _published(block_size=block_size)

    assert np.isnan(value)


@pytest.mark.parametrize("block_size", [0, 1, 2, 3])
def test_an_uncertifiable_margin_publishes_no_marginal(block_size: int):
    """The marginal channel fails with the other two."""
    _value, _policy, marginal = _published(block_size=block_size)

    assert np.isnan(marginal)
