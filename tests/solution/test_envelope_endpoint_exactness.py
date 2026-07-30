"""The envelope reproduces a candidate exactly at that candidate's own abscissa.

A query landing on a candidate is not an interpolation problem — the answer is
that candidate's own value, policy, and marginal. The link it is read from may
join it to a neighbour whose value is orders of magnitude larger, which happens
whenever a grid carries a node with a near-zero budget: consuming it yields a
CRRA utility that dwarfs every other value on the grid. Reading the link must
still return the queried candidate bit for bit, or the neighbour's magnitude
sets the resolution of an unrelated node's published value.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import envelope_at_query


def _read_at(*, values: list[float], grid: list[float], query: float):
    """Envelope value and policy at one query, beside the working-precision inputs.

    The candidate arrays are returned too: an exactness assertion has to compare
    against the value as the working precision actually holds it, not against the
    decimal literal it was written as.
    """
    value_arr = jnp.asarray(values)
    grid_arr = jnp.asarray(grid)
    value, policy, _marginal = envelope_at_query(
        endog_grid=grid_arr,
        policy=grid_arr,
        value=value_arr,
        marginal=jnp.ones_like(grid_arr),
        segment_id=jnp.zeros_like(grid_arr),
        x_query=jnp.asarray([query]),
    )
    return np.asarray(value)[0], np.asarray(policy)[0], np.asarray(value_arr)


@pytest.mark.parametrize("neighbour", [-1.6777216e7, -1e12, -1e30])
def test_query_on_a_node_returns_that_node_beside_a_huge_neighbour(neighbour):
    """A candidate's value survives a link to a vastly larger-magnitude neighbour."""
    value, _policy, candidates = _read_at(
        values=[neighbour, -2.475, -1.475],
        grid=[1e-7, 0.5, 1.0],
        query=0.5,
    )
    np.testing.assert_array_equal(value, candidates[1])


def test_query_on_the_left_endpoint_returns_that_endpoint():
    """Reading at the left end of a link returns the left candidate exactly."""
    value, _policy, candidates = _read_at(
        values=[-1.6777216e7, -2.475, -1.475],
        grid=[1e-7, 0.5, 1.0],
        query=1e-7,
    )
    np.testing.assert_array_equal(value, candidates[0])


def test_interpolated_policy_is_exact_at_a_node():
    """The published policy at a node is that node's own policy, not a blend."""
    _value, policy, _candidates = _read_at(
        values=[-1.6777216e7, -2.475, -1.475],
        grid=[1e-7, 0.5, 1.0],
        query=0.5,
    )
    np.testing.assert_array_equal(policy, np.asarray(jnp.asarray(0.5)))
