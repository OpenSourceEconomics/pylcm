"""The envelope's arithmetic is selectable, and the choice is a compile-time one.

`envelope_at_query` decides which link owns each query. Deciding it in
double-double arithmetic keeps the ordering right where two candidates' endpoint
values nearly cancel, and costs roughly an order of magnitude more work per read.
Not every solve needs that: a well-scaled problem is decided identically by the
ordinary affine read.

`arithmetic` selects between them. It is a Python-level choice, not a traced one,
so the ordinary mode never emits the error-free transforms at all — a mask over
both would pay for the arithmetic it was meant to avoid.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import envelope_at_query

# A crossing of two branches, well scaled: the endpoint values are the same order
# of magnitude as the differences that decide ownership, so no cancellation
# separates the two arithmetics.
_ENDOG_GRID = jnp.asarray([10.0, 11.0, 12.0, 10.0, 11.0, 12.0])
_POLICY = jnp.asarray([3.0, 3.0, 3.0, 0.5, 0.5, 0.5])
_VALUE = jnp.asarray([5 / 3, 2.0, 7 / 3, 0.0, 2.0, 4.0])
_SEGMENT_ID = jnp.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
_MARGINAL = jnp.asarray([1 / 3, 1 / 3, 1 / 3, 2.0, 2.0, 2.0])
_X_QUERY = jnp.asarray([10.0, 10.5, 11.0, 11.1, 11.5, 12.0])


def _envelope(arithmetic):
    return envelope_at_query(
        endog_grid=_ENDOG_GRID,
        policy=_POLICY,
        value=_VALUE,
        marginal=_MARGINAL,
        segment_id=_SEGMENT_ID,
        x_query=_X_QUERY,
        arithmetic=arithmetic,
    )


@pytest.mark.parametrize("channel", [0, 1, 2])
def test_the_ordinary_read_decides_a_well_scaled_envelope_as_the_certified_one_does(
    channel,
):
    """On a well-scaled crossing both arithmetics publish the same envelope."""
    certified = _envelope("certified")[channel]
    ordinary = _envelope("ordinary")[channel]
    np.testing.assert_allclose(ordinary, certified, rtol=1e-12, atol=1e-12)


def test_the_ordinary_read_emits_none_of_the_error_free_transforms():
    """The ordinary mode compiles to strictly fewer operations."""
    counts = {
        mode: len(jax.make_jaxpr(partial(_envelope, mode))().jaxpr.eqns)
        for mode in ("certified", "ordinary")
    }
    assert counts["ordinary"] < counts["certified"]
