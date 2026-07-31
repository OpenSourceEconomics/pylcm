"""Reading a link reproduces the affine correspondence its endpoints define.

Two properties, and the second is the one a normalized coordinate cannot deliver:

- A query that *is* an endpoint returns that endpoint. The link's other end may be
  orders of magnitude away — a node whose budget is near zero carries a CRRA
  utility that dwarfs the grid — and the answer must not inherit its resolution.
- A query strictly inside returns the affine value there. Endpoint identity has to
  come from the query itself: the normalized coordinate `(q - left) / (right -
  left)` is a rounded quotient, and for `q = nextafter(right, -inf)` it evaluates
  to exactly `1.0` at both precisions. Trusting it publishes the wrong endpoint's
  value and policy for a point that is not an endpoint at all.

The oracle is exact rational arithmetic over the stored floats, so it shares no
rounding behaviour with the implementation.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.query import envelope_at_query


def _working_dtype() -> np.dtype:
    return np.dtype(jnp.result_type(1.0))


def _spacing_at(magnitude: float) -> float:
    """The working format's spacing at `magnitude`, so a bound reads in its ULPs."""
    return float(np.spacing(_working_dtype().type(abs(magnitude))))


def _exact_affine(*, left_grid, right_grid, left_value, right_value, query) -> float:
    """The affine value at `query`, computed exactly over the stored floats."""
    relative = (Fraction(float(query)) - Fraction(float(left_grid))) / (
        Fraction(float(right_grid)) - Fraction(float(left_grid))
    )
    exact = Fraction(float(left_value)) + relative * (
        Fraction(float(right_value)) - Fraction(float(left_value))
    )
    return float(exact)


def _read(*, grid, values, policies, query, block_size=0):
    """Envelope value and policy at one query over the given candidate cloud."""
    value, policy, _marginal = envelope_at_query(
        endog_grid=jnp.asarray(grid),
        policy=jnp.asarray(policies),
        value=jnp.asarray(values),
        marginal=jnp.ones(len(grid)),
        segment_id=jnp.zeros(len(grid)),
        x_query=jnp.asarray([query]),
        segment_block_size=block_size,
    )
    return float(np.asarray(value)[0]), float(np.asarray(policy)[0])


@pytest.mark.parametrize("neighbour", [-1.6777216e7, -1e12, -1e30])
def test_query_on_a_node_returns_that_node_beside_a_huge_neighbour(neighbour):
    """A candidate's value survives a link to a vastly larger-magnitude neighbour."""
    values = [neighbour, -2.475, -1.475]
    expected = float(jnp.asarray(values)[1])
    value, _policy = _read(
        grid=[1e-7, 0.5, 1.0], values=values, policies=[1e-7, 0.5, 1.0], query=0.5
    )
    np.testing.assert_array_equal(value, expected)


def test_query_on_the_left_endpoint_returns_that_endpoint():
    """Reading at the left end of a link returns the left candidate exactly."""
    values = [-1.6777216e7, -2.475, -1.475]
    expected = float(jnp.asarray(values)[0])
    value, _policy = _read(
        grid=[1e-7, 0.5, 1.0], values=values, policies=[1e-7, 0.5, 1.0], query=1e-7
    )
    np.testing.assert_array_equal(value, expected)


def _steep_link():
    """A two-point link with far-apart endpoint channels, and an interior query.

    The query is the float just below the right node, so the normalized coordinate
    rounds to exactly `1.0` while the query is strictly inside the link. Value and
    policy share the same steep spread, so each channel's affine reading at the
    query is far from the endpoint it nearly touches.
    """
    dtype = _working_dtype()
    left_grid = dtype.type(-1.0)
    right_grid = dtype.type(1.0)
    span = 2.0**26 if dtype == np.float32 else 2.0**55
    channel = [dtype.type(-span), dtype.type(1.0)]
    return (
        [left_grid, right_grid],
        channel,
        channel,
        np.nextafter(right_grid, dtype.type(-np.inf)),
    )


@pytest.mark.parametrize("block_size", [0, 1])
def test_an_interior_query_reads_the_affine_value_not_an_endpoint(block_size):
    """A query one ULP inside the right node is interpolated, not snapped to it."""
    grid, values, policies, query = _steep_link()
    value, _policy = _read(
        grid=grid, values=values, policies=policies, query=query, block_size=block_size
    )
    expected = _exact_affine(
        left_grid=grid[0],
        right_grid=grid[1],
        left_value=values[0],
        right_value=values[1],
        query=query,
    )
    assert abs(value - expected) <= 4.0 * _spacing_at(expected)


def test_an_interior_query_reads_the_affine_policy_not_an_endpoint():
    """The policy at an interior query is the interpolated one, not the endpoint's."""
    grid, values, policies, query = _steep_link()
    _value, policy = _read(grid=grid, values=values, policies=policies, query=query)
    expected = _exact_affine(
        left_grid=grid[0],
        right_grid=grid[1],
        left_value=policies[0],
        right_value=policies[1],
        query=query,
    )
    assert abs(policy - expected) <= 4.0 * _spacing_at(expected)


def _generated_links():
    """Two-node links spanning the magnitude regimes a link read has to survive.

    Each case pairs a grid interval with endpoint channel values, drawn so that
    the spread between the two ends ranges from none at all to many orders of
    magnitude, in both directions and both signs.
    """
    dtype = _working_dtype()
    rng = np.random.default_rng(20260728)
    exponent = 26 if dtype == np.float32 else 55
    spreads = [0.0, 1.0, 2.0**8, 2.0**exponent]
    cases = []
    for spread in spreads:
        for sign in (-1.0, 1.0):
            left_grid = dtype.type(rng.uniform(-3.0, 3.0))
            right_grid = dtype.type(left_grid + rng.uniform(0.1, 4.0))
            ordinary = dtype.type(rng.uniform(-3.0, 3.0))
            extreme = dtype.type(sign * spread + float(ordinary))
            cases.append((left_grid, right_grid, extreme, ordinary))
            cases.append((left_grid, right_grid, ordinary, extreme))
    return cases


def _queries_across(left_grid, right_grid):
    """Endpoints, the floats just inside them, and interior points between."""
    dtype = _working_dtype()
    inward_left = np.nextafter(left_grid, dtype.type(np.inf))
    inward_right = np.nextafter(right_grid, dtype.type(-np.inf))
    interior = [
        dtype.type(left_grid + share * (float(right_grid) - float(left_grid)))
        for share in (0.25, 0.5, 0.75)
    ]
    return [left_grid, inward_left, *interior, inward_right, right_grid]


@pytest.mark.parametrize("block_size", [0, 1])
@pytest.mark.parametrize("link", _generated_links())
def test_every_read_along_a_link_agrees_with_exact_rational_arithmetic(
    link, block_size
):
    """Both evaluation paths read a link as the affine map its endpoints define.

    Endpoint reads are exact; interior reads land within a few ULP of the value
    exact rational arithmetic assigns them, whatever the spread between the two
    endpoints.
    """
    left_grid, right_grid, left_value, right_value = link
    for query in _queries_across(left_grid, right_grid):
        value, _policy = _read(
            grid=[left_grid, right_grid],
            values=[left_value, right_value],
            policies=[left_value, right_value],
            query=query,
            block_size=block_size,
        )
        if query == left_grid:
            assert value == float(left_value)
        elif query == right_grid:
            assert value == float(right_value)
        else:
            expected = _exact_affine(
                left_grid=left_grid,
                right_grid=right_grid,
                left_value=left_value,
                right_value=right_value,
                query=query,
            )
            assert abs(value - expected) <= 4.0 * _spacing_at(expected)


def test_a_rival_candidate_still_owns_a_query_the_link_falls_below():
    """A point candidate above the link's affine value at the query wins the envelope.

    The link's right endpoint sits far above the rival, so snapping to it would hand
    the query — and the published policy — to the wrong branch.
    """
    grid, values, policies, query = _steep_link()
    rival_value, rival_policy = 0.0, 0.5
    value, policy = _read(
        grid=[*grid, query],
        values=[*values, rival_value],
        policies=[*policies, rival_policy],
        query=query,
    )
    assert (value, policy) == (rival_value, rival_policy)
