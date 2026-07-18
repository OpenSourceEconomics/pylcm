"""The validated local cubic Hermite outer interpolant.

The PR-4 interpolation battery: exact node reproduction, quadratic-exact
recovery between nodes (value and derivative), derivative-vs-central-FD
agreement on a smooth surface, refusal to bridge nonfinite gaps or declared
invalid intervals, no poisoning of valid intervals by a nonfinite neighbor,
out-of-domain refusal, and vectorized heterogeneous state cells.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.outer_interpolation import (
    LocalCubicOuterInterpolant,
    OuterInterpolant,
)

_INTERP = LocalCubicOuterInterpolant()


def test_satisfies_the_protocol() -> None:
    assert isinstance(_INTERP, OuterInterpolant)


def test_reproduces_node_values_exactly() -> None:
    nodes = jnp.array([0.0, 0.4, 1.0, 2.5])
    values = jnp.sin(nodes)
    read = _INTERP.evaluate(nodes=nodes, values=values, query=nodes)
    np.testing.assert_array_equal(np.asarray(read), np.asarray(values))


def test_quadratic_is_recovered_exactly_with_derivative() -> None:
    """Parabolic slope estimates make the Hermite read quadratic-exact."""
    nodes = jnp.array([0.0, 0.3, 0.7, 1.1, 2.0])

    def f(x: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * x**2 - 3.0 * x + 0.5

    query = jnp.array([0.11, 0.5, 0.99, 1.7])
    value, derivative = _INTERP.evaluate_with_derivative(
        nodes=nodes, values=f(nodes), query=query
    )
    np.testing.assert_allclose(np.asarray(value), np.asarray(f(query)), atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(derivative), np.asarray(4.0 * query - 3.0), atol=1e-12
    )


def test_derivative_matches_central_finite_difference() -> None:
    nodes = jnp.linspace(0.0, 1.0, 33)
    values = jnp.sin(3.0 * nodes)
    query = jnp.array([0.21, 0.52, 0.83])
    eps = 1e-6
    _, derivative = _INTERP.evaluate_with_derivative(
        nodes=nodes, values=values, query=query
    )
    fd = (
        _INTERP.evaluate(nodes=nodes, values=values, query=query + eps)
        - _INTERP.evaluate(nodes=nodes, values=values, query=query - eps)
    ) / (2.0 * eps)
    np.testing.assert_allclose(np.asarray(derivative), np.asarray(fd), atol=1e-6)


def test_nonfinite_gap_is_not_bridged() -> None:
    """Intervals touching a nonfinite candidate read `-inf`, not a bridge."""
    nodes = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([0.0, -jnp.inf, 2.0, 3.0])
    inside_gap = _INTERP.evaluate(
        nodes=nodes, values=values, query=jnp.array([0.5, 1.5])
    )
    assert float(inside_gap[0]) == -jnp.inf
    assert float(inside_gap[1]) == -jnp.inf


def test_nonfinite_neighbor_does_not_poison_valid_interval() -> None:
    """The [2, 3] interval stays finite and accurate next to the NaN node."""
    nodes = jnp.array([0.0, 1.0, 2.0, 3.0])
    values = jnp.array([0.0, jnp.nan, 2.0, 3.0])
    read = _INTERP.evaluate(nodes=nodes, values=values, query=jnp.array([2.5]))
    assert np.isfinite(float(read[0]))
    np.testing.assert_allclose(float(read[0]), 2.5, atol=0.1)


def test_declared_invalid_interval_is_refused() -> None:
    nodes = jnp.array([0.0, 1.0, 2.0])
    values = jnp.array([0.0, 1.0, 2.0])
    interval_valid = jnp.array([True, False])
    ok = _INTERP.evaluate(
        nodes=nodes,
        values=values,
        query=jnp.array([0.5]),
        interval_valid=interval_valid,
    )
    refused = _INTERP.evaluate(
        nodes=nodes,
        values=values,
        query=jnp.array([1.5]),
        interval_valid=interval_valid,
    )
    np.testing.assert_allclose(float(ok[0]), 0.5, atol=1e-12)
    assert float(refused[0]) == -jnp.inf


def test_out_of_domain_query_is_invalid_not_extrapolated() -> None:
    nodes = jnp.array([0.0, 1.0])
    values = jnp.array([0.0, 1.0])
    read = _INTERP.evaluate(nodes=nodes, values=values, query=jnp.array([-0.1, 1.1]))
    assert float(read[0]) == -jnp.inf
    assert float(read[1]) == -jnp.inf
    validity = _INTERP.evaluate_validity(
        nodes=nodes, values=values, query=jnp.array([-0.1, 0.5, 1.1])
    )
    assert not bool(validity[0])
    assert bool(validity[1])
    assert not bool(validity[2])


def test_vectorized_state_cells_interpolate_independently() -> None:
    """Two state cells with different surfaces read their own values."""
    nodes = jnp.array([0.0, 0.5, 1.0, 1.5])
    surfaces = jnp.stack(
        [nodes**2, 1.0 - nodes], axis=1
    )  # shape (C, 2): cell 0 quadratic, cell 1 affine
    query = jnp.array([0.75, 0.75])
    read = _INTERP.evaluate(nodes=nodes, values=surfaces, query=query)
    np.testing.assert_allclose(np.asarray(read), [0.75**2, 0.25], atol=1e-12)


def test_bad_nodes_raise() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _INTERP.evaluate(
            nodes=jnp.array([0.0, 0.0, 1.0]),
            values=jnp.zeros(3),
            query=jnp.array([0.5]),
        )
    with pytest.raises(ValueError, match="two nodes"):
        _INTERP.evaluate(
            nodes=jnp.array([0.0]), values=jnp.zeros(1), query=jnp.array([0.0])
        )
    with pytest.raises(ValueError, match="leading axis"):
        _INTERP.evaluate(
            nodes=jnp.array([0.0, 1.0]), values=jnp.zeros(3), query=jnp.array([0.5])
        )
