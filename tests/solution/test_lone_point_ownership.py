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
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

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
def test_a_straddled_lone_point_owns_the_query_it_brackets(*, position, block_size):
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
def test_a_straddled_lone_point_loses_to_a_rival_reading_higher(
    *, position, block_size
):
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


def _straddled_case(*, rng, dtype):
    """Draw one lone point at zero with a rival strictly straddling it.

    The rival's endpoints are placed so its reading at zero is a definite
    distance from the lone point's stored value, in units of the working
    format's spacing at the larger of the two. That separation is what makes an
    abstention a defect here rather than an honest near-tie: the two candidates
    are further apart than the arithmetic's own resolution by construction, so a
    certificate that cannot order them has lost something it was given.
    """
    scale = float(10.0 ** rng.integers(-4, 5))
    left_x = -scale * float(rng.uniform(0.25, 4.0))
    right_x = scale * float(rng.uniform(0.25, 4.0))
    rival_at_zero = float(rng.uniform(-50.0, 50.0))
    slope = float(rng.uniform(-20.0, 20.0)) / scale
    left_v = rival_at_zero + slope * left_x
    right_v = rival_at_zero + slope * right_x

    gap = float(np.finfo(dtype).eps) * max(abs(rival_at_zero), 1.0) * 1e4
    point_v = rival_at_zero + float(rng.choice([-1.0, 1.0])) * gap * rng.uniform(
        1.0, 1e3
    )
    return {
        "endog": [0.0, left_x, right_x],
        "value": [point_v, left_v, right_v],
        "segment": [1.0, 0.0, 0.0],
        "query": [0.0],
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_certified_owner_at_zero_agrees_with_the_ordinary_read(seed):
    """Over the straddled-lone-point class, the certificate costs no answers.

    The class is a lone point at abscissa zero — the arrangement whose flat line
    the arithmetic cannot read without a width — against rivals swept over
    magnitude, slope, sign and which of the two stores the larger value. Wherever
    the two candidates are separated by far more than the format's resolution,
    the certified route owes the same winner as the plain read, in all three
    published channels, and owes an answer at all.
    """
    dtype = jnp.zeros(()).dtype
    rng = np.random.default_rng(seed=seed)
    for _ in range(24):
        case = _straddled_case(rng=rng, dtype=dtype)
        certified = _envelope(**case, arithmetic="certified")
        ordinary = _envelope(**case, arithmetic="ordinary")
        for channel, got, expected in zip(
            ("value", "policy", "marginal"), certified, ordinary, strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(got),
                np.asarray(expected),
                rtol=10.0**-DECIMAL_PRECISION,
                atol=10.0**-DECIMAL_PRECISION,
                err_msg=f"{channel} disagrees for {case}",
            )


def _step_ulps(*, value, steps, dtype):
    """Return `value` moved `steps` representable steps, signed toward ±inf."""
    out = dtype(value)
    toward = dtype(np.inf if steps > 0 else -np.inf)
    for _ in range(abs(steps)):
        out = np.nextafter(out, toward, dtype=dtype)
    return out


@pytest.mark.parametrize("block_size", _ROUTES)
@pytest.mark.parametrize("width_exponent", [-8, -60, -120, -400, -1000])
@pytest.mark.parametrize("lone_point_is_higher", [True, False])
def test_the_higher_line_owns_the_abscissa_at_any_width_ratio(
    *, lone_point_is_higher, width_exponent, block_size
):
    """Whichever line reads higher owns the abscissa however narrow the rival is.

    The rival's width and the gap being decided are independent quantities: a
    branch may pass over the point across an interval many orders of magnitude
    narrower than the values it carries, and the ordering of the two candidates
    at the shared abscissa does not depend on how wide either of them is. A
    comparison that loses the ordering as the rival narrows is reporting the
    scale of the operands rather than the geometry.

    Both orderings are asserted. A comparison that has stopped deciding and is
    merely defaulting can still publish the right winner for one of them, so a
    test that only ever expects the lone point to win cannot tell a decision
    from a lucky default.

    The gap here is 64 representable steps of the rival's own value, so it is a
    normal-input difference at every width, not a near-tie.
    """
    dtype = np.dtype(jnp.zeros(()).dtype).type
    if np.ldexp(1.0, width_exponent) < float(np.finfo(dtype).tiny):
        pytest.skip("rival width is subnormal in this format")
    half_width = dtype(np.ldexp(1.0, width_exponent))
    rival_value = dtype(0.75)
    point_value = _step_ulps(
        value=rival_value, steps=64 if lone_point_is_higher else -64, dtype=dtype
    )
    expected = (
        (point_value, dtype(1.0), dtype(11.0))
        if lone_point_is_higher
        else (rival_value, dtype(0.0), dtype(22.0))
    )

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.asarray([dtype(0.0), -half_width, half_width]),
        policy=jnp.asarray([dtype(1.0), dtype(0.0), dtype(0.0)]),
        value=jnp.asarray([point_value, rival_value, rival_value]),
        marginal=jnp.asarray([dtype(11.0), dtype(22.0), dtype(22.0)]),
        segment_id=jnp.asarray([dtype(1.0), dtype(0.0), dtype(0.0)]),
        x_query=jnp.asarray([dtype(0.0)]),
        segment_block_size=block_size,
        arithmetic="certified",
    )

    # Ownership is a discrete decision, so it is asserted exactly: the published
    # triple either comes from the winning candidate or it does not.
    for channel, got, want in zip(
        ("value", "policy", "marginal"),
        (value, policy, marginal),
        expected,
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(got), [want], err_msg=f"{channel} came from the losing line"
        )
