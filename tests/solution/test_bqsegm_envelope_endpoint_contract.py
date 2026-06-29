"""Endpoint-ownership contract of the query-side upper envelope.

At a case boundary two one-sided segments meet at the same abscissa. The side
that does not own the equality point must not win there even if its value is
higher: its endpoint at the boundary is open. The envelope honors per-candidate
open/closed endpoint flags; with no flags every endpoint is closed, so the
inclusive-bracket behavior the DC-EGM path relies on is unchanged.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.upper_envelope.query import envelope_at_query

# Two monotone segments meeting at the boundary abscissa x = 1.0:
# - segment 0 owns the boundary (value 1.0 there),
# - segment 1 is the excluded side (value 5.0 there, but not eligible at 1.0).
_ENDOG_GRID = jnp.array([0.0, 1.0, 1.0, 2.0])
_VALUE = jnp.array([0.0, 1.0, 5.0, 6.0])
_POLICY = jnp.array([0.0, 1.0, 5.0, 6.0])
_MARGINAL = jnp.array([1.0, 1.0, 1.0, 1.0])
_SEGMENT_ID = jnp.array([0.0, 0.0, 1.0, 1.0])


def test_default_closed_endpoints_keep_the_inclusive_bracket():
    """Without endpoint flags, the higher excluded segment wins at the boundary."""
    env_value, _, _ = envelope_at_query(
        endog_grid=_ENDOG_GRID,
        policy=_POLICY,
        value=_VALUE,
        marginal=_MARGINAL,
        segment_id=_SEGMENT_ID,
        x_query=jnp.array([1.0]),
    )
    np.testing.assert_allclose(np.asarray(env_value), [5.0])


def test_open_endpoint_excludes_its_segment_at_the_boundary():
    """An open endpoint at the boundary makes the owning side win there."""
    # Candidate 2 is segment 1's left endpoint at x = 1.0; mark it open.
    left_endpoint_closed = jnp.array([True, True, False, True])
    right_endpoint_closed = jnp.array([True, True, True, True])
    env_value, env_policy, _ = envelope_at_query(
        endog_grid=_ENDOG_GRID,
        policy=_POLICY,
        value=_VALUE,
        marginal=_MARGINAL,
        segment_id=_SEGMENT_ID,
        x_query=jnp.array([1.0]),
        left_endpoint_closed=left_endpoint_closed,
        right_endpoint_closed=right_endpoint_closed,
    )
    np.testing.assert_allclose(np.asarray(env_value), [1.0])
    np.testing.assert_allclose(np.asarray(env_policy), [1.0])


def test_strict_interior_query_ignores_endpoint_flags():
    """Endpoint flags only gate the exact boundary; interior queries are unaffected."""
    left_endpoint_closed = jnp.array([True, True, False, True])
    right_endpoint_closed = jnp.array([True, True, True, True])
    env_value, _, _ = envelope_at_query(
        endog_grid=_ENDOG_GRID,
        policy=_POLICY,
        value=_VALUE,
        marginal=_MARGINAL,
        segment_id=_SEGMENT_ID,
        x_query=jnp.array([1.5]),
        left_endpoint_closed=left_endpoint_closed,
        right_endpoint_closed=right_endpoint_closed,
    )
    np.testing.assert_allclose(np.asarray(env_value), [5.5])
