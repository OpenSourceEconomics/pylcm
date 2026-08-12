"""A lone point owns the queries it brackets, wherever it sits on the axis.

A candidate that is the only node of its branch has no consecutive neighbour, so
its only link is a zero-width self-bracket at its own abscissa. It still carries a
stored value there, and the upper envelope has to see it: where a rival branch
passes over that abscissa without a node of its own, whichever of the two stores
the larger value owns the query.

`"ordinary"` arithmetic is the reference throughout. It has no certificate to
abstain on, so it publishes what the geometry says, and the certified route owes
the same answer wherever the geometry is separated by far more than the format's
resolution.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import envelope_at_query
from tests.conftest import DECIMAL_PRECISION

_ROUTES = (
    pytest.param(0, id="dense"),
    pytest.param(1, id="blocked"),
)


def _envelope(*, endog, value, segment, query, arithmetic, block_size=0):
    """Evaluate the envelope, taking policy as half the value and a unit marginal."""
    grid = jnp.asarray(endog)
    return envelope_at_query(
        endog_grid=grid,
        policy=jnp.asarray(value) * 0.5,
        value=jnp.asarray(value),
        marginal=jnp.ones_like(grid),
        segment_id=jnp.asarray(segment),
        x_query=jnp.asarray(query),
        segment_block_size=block_size,
        arithmetic=arithmetic,
    )


class _Straddle(NamedTuple):
    """A lone point together with an ordinary link that passes strictly over it."""

    endog: tuple[float, float, float]
    """Abscissae: the lone point, then the rival link's lower and upper nodes."""
    query: float
    """The lone point's own abscissa, midway along the rival link."""


# Branch 1 is the single node; branch 0 is the rival, rising and carrying no node
# at the queried abscissa. The rival reads 4.5 there, so a lone point storing 5.0
# owns the query and one storing 1.0 does not.
#
# The rival's nodes straddle the point by a fraction of the abscissa rather than
# by a fixed step, so the arrangement is the same geometry in both precisions. A
# fixed step of 1.0 around 1e8 is below float32's spacing there, which would
# collapse all three abscissae onto one point and test the format instead.
_STRADDLED = {
    "at_zero": _Straddle(endog=(0.0, -1.0, 1.0), query=0.0),
    "off_zero": _Straddle(endog=(1.0, 0.0, 2.0), query=1.0),
    "large_abscissa": _Straddle(endog=(1e8, 0.99e8, 1.01e8), query=1e8),
}


@pytest.mark.parametrize("block_size", _ROUTES)
@pytest.mark.parametrize("position", sorted(_STRADDLED))
def test_a_straddled_lone_point_owns_the_query_it_brackets(position, block_size):
    """A lone point storing the larger value wins against a rival passing over it."""
    case = _STRADDLED[position]
    published, _, _ = _envelope(
        endog=case.endog,
        value=[5.0, 0.0, 9.0],
        segment=[1.0, 0.0, 0.0],
        query=[case.query],
        arithmetic="certified",
        block_size=block_size,
    )
    np.testing.assert_array_almost_equal(
        np.asarray(published), [5.0], decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("block_size", _ROUTES)
@pytest.mark.parametrize("position", sorted(_STRADDLED))
def test_a_straddled_lone_point_loses_to_a_rival_reading_higher(position, block_size):
    """A lone point storing the smaller value does not take the query from the rival."""
    case = _STRADDLED[position]
    published, _, _ = _envelope(
        endog=case.endog,
        value=[1.0, 0.0, 9.0],
        segment=[1.0, 0.0, 0.0],
        query=[case.query],
        arithmetic="certified",
        block_size=block_size,
    )
    np.testing.assert_array_almost_equal(
        np.asarray(published), [4.5], decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("position", sorted(_STRADDLED))
def test_the_certified_route_matches_the_ordinary_read_at_a_lone_point(position):
    """Where the geometry is well separated, the certificate costs no answers."""
    case = _STRADDLED[position]
    shared = {
        "endog": case.endog,
        "value": [5.0, 0.0, 9.0],
        "segment": [1.0, 0.0, 0.0],
        "query": [case.query],
    }
    certified, _, _ = _envelope(**shared, arithmetic="certified")
    ordinary, _, _ = _envelope(**shared, arithmetic="ordinary")
    np.testing.assert_array_almost_equal(
        np.asarray(certified), np.asarray(ordinary), decimal=DECIMAL_PRECISION
    )


def test_a_lone_point_is_read_at_its_own_abscissa_and_nowhere_else():
    """A single point contributes its value at its own abscissa only.

    Away from it the lone point brackets nothing, so the rival branch owns the
    query and the published value is the rival's interpolation.
    """
    published, _, _ = _envelope(
        endog=[0.0, -1.0, 1.0],
        value=[5.0, 0.0, 9.0],
        segment=[1.0, 0.0, 0.0],
        query=[-1.0, -0.5, 0.0, 0.5, 1.0],
        arithmetic="certified",
    )
    np.testing.assert_array_almost_equal(
        np.asarray(published), [0.0, 2.25, 5.0, 6.75, 9.0], decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("position", sorted(_STRADDLED))
def test_coincident_nodes_storing_different_values_publish_the_larger(position):
    """Where a branch takes two values at one abscissa, the envelope takes the larger.

    The link between two coincident nodes carries no affine line and settles
    nothing, but each node is also a point of the correspondence in its own
    right, so the upper envelope reads the higher of the two.
    """
    case = _STRADDLED[position]
    point = case.query
    published, _, _ = _envelope(
        endog=[point, point, case.endog[2]],
        value=[5.0, 8.0, 9.0],
        segment=[0.0, 0.0, 0.0],
        query=[point],
        arithmetic="certified",
    )
    np.testing.assert_array_almost_equal(
        np.asarray(published), [8.0], decimal=DECIMAL_PRECISION
    )
