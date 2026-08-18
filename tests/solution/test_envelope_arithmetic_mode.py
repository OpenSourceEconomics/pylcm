"""The envelope's arithmetic is selectable, and the choice is a compile-time one.

`envelope_at_query` decides which link owns each query. The certified mode
decides it in exact arithmetic over the stored operands, which keeps the ordering
right where two candidates' endpoint values nearly cancel. Not every solve needs
that: a well-scaled problem is decided identically by the ordinary affine read.

`arithmetic` selects between them. It is a Python-level choice, not a traced one,
so the ordinary mode never emits the certified machinery at all — a mask over
both would pay for the arithmetic it was meant to avoid.

The two are not ordered by program size. The certified mode delegates its whole
reduction to a few opaque calls, so it compiles to fewer equations than the
ordinary dense read while doing the more careful arithmetic inside them.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import envelope_at_query
from tests.conftest import EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

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


def test_the_ordinary_read_emits_none_of_the_certified_machinery():
    """Selecting the ordinary mode leaves the exact kernel out of the program.

    The choice is made in Python rather than traced, so the certified path is
    absent from the ordinary program rather than masked within it — a mask over
    both would pay for the arithmetic it was meant to avoid.

    An operation count cannot express this. The certified mode delegates the
    whole reduction to a handful of opaque calls and so compiles to *fewer*
    equations than the ordinary dense read, which says nothing about whether
    either one dragged in the other's machinery.
    """
    certified = str(jax.make_jaxpr(partial(_envelope, "certified"))().jaxpr)
    ordinary = str(jax.make_jaxpr(partial(_envelope, "ordinary"))().jaxpr)

    assert "ExactQueryWinner" in certified, (
        "the certified mode must reach the exact kernel, or this test cannot "
        "detect the ordinary mode reaching it either"
    )
    assert "ExactQueryWinner" not in ordinary
    assert "ExactAffineRead" not in ordinary
