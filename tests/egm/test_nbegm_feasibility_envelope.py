"""Feasibility partitions are enforced by the NBEGM upper envelope."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.axis_boundaries import (
    AxisBoundary,
    boundary_owner_for_feasible_region,
    resolve_axis_partition,
)
from _lcm.egm.nbegm_step import _interp_continuation_value
from _lcm.egm.upper_envelope.query import envelope_at_query


@pytest.mark.parametrize(
    ("feasible_side", "includes_boundary", "expected_value"),
    [
        ("below", True, 4.0),
        ("below", False, -jnp.inf),
        ("above", True, 4.0),
        ("above", False, -jnp.inf),
    ],
)
def test_feasibility_partition_owns_equality_from_the_comparison(
    *, feasible_side, includes_boundary, expected_value
) -> None:
    """All four inequality forms give the threshold to exactly one side."""
    boundary = jnp.float32(4.0)
    partition = resolve_axis_partition(
        start=jnp.float32(0.0),
        stop=jnp.float32(8.0),
        boundaries=(
            AxisBoundary(
                value=boundary,
                owner=boundary_owner_for_feasible_region(
                    feasible_side=feasible_side,
                    includes_boundary=includes_boundary,
                ),
                effect="feasibility",
            ),
        ),
    )
    feasible_intervals = (
        jnp.array([True, False])
        if feasible_side == "below"
        else jnp.array([False, True])
    )
    endog = jnp.array(
        [
            0.0,
            jnp.nextafter(boundary, -jnp.inf),
            boundary,
            jnp.nextafter(boundary, jnp.inf),
            8.0,
        ]
    )

    value, policy, marginal = envelope_at_query(
        endog_grid=endog,
        value=endog,
        policy=2.0 * endog,
        marginal=3.0 * endog,
        segment_id=jnp.zeros_like(endog),
        x_query=boundary,
        feasibility_partition=partition,
        feasible_interval_mask=feasible_intervals,
    )

    assert value == expected_value
    if jnp.isneginf(expected_value):
        assert jnp.isnan(policy)
        assert marginal == 0.0
    else:
        assert policy == 8.0
        assert marginal == 12.0


def test_feasibility_mask_prevents_a_candidate_link_from_crossing_the_boundary() -> (
    None
):
    """Two live endpoints on opposite sides never interpolate through the boundary."""
    boundary = jnp.float32(4.0)
    partition = resolve_axis_partition(
        start=jnp.float32(0.0),
        stop=jnp.float32(8.0),
        boundaries=(AxisBoundary(value=boundary, owner="right", effect="feasibility"),),
    )

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.array([3.0, 5.0]),
        value=jnp.array([30.0, 50.0]),
        policy=jnp.array([300.0, 500.0]),
        marginal=jnp.array([3_000.0, 5_000.0]),
        segment_id=jnp.array([0.0, 0.0]),
        x_query=jnp.array([4.0, 5.0]),
        feasibility_partition=partition,
        feasible_interval_mask=jnp.array([False, True]),
    )

    assert jnp.isnan(value[0])
    np.testing.assert_array_equal(
        (value[1], policy[1], marginal[1]),
        np.array([50.0, 500.0, 5_000.0]),
    )


def test_infeasible_queries_mask_all_published_channels_together() -> None:
    """An infeasible grid row publishes -inf value, NaN policy, and zero marginal."""
    partition = resolve_axis_partition(
        start=jnp.float32(0.0),
        stop=jnp.float32(8.0),
        boundaries=(
            AxisBoundary(value=jnp.float32(4.0), owner="right", effect="feasibility"),
        ),
    )

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.array([0.0, 2.0, 4.0, 8.0]),
        value=jnp.array([1.0, 2.0, 4.0, 8.0]),
        policy=jnp.array([10.0, 20.0, 40.0, 80.0]),
        marginal=jnp.array([100.0, 200.0, 400.0, 800.0]),
        segment_id=jnp.zeros((4,)),
        x_query=jnp.array([1.0, 6.0]),
        feasibility_partition=partition,
        feasible_interval_mask=jnp.array([False, True]),
    )

    assert jnp.isneginf(value[0])
    assert jnp.isnan(policy[0])
    assert marginal[0] == 0.0
    np.testing.assert_array_equal(
        (value[1], policy[1], marginal[1]),
        np.array([6.0, 60.0, 600.0]),
    )


@pytest.mark.parametrize("fixture_name", ["x64_enabled", "x64_disabled"])
def test_feasibility_envelope_supports_both_precisions_jit_and_vmap(
    *,
    request: pytest.FixtureRequest,
    fixture_name: str,
) -> None:
    """Ownership and channel masking survive both JAX transforms and precisions."""
    request.getfixturevalue(fixture_name)
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    boundary = jnp.asarray(4.0, dtype=dtype)
    partition = resolve_axis_partition(
        start=jnp.asarray(0.0, dtype=dtype),
        stop=jnp.asarray(8.0, dtype=dtype),
        boundaries=(AxisBoundary(value=boundary, owner="right", effect="feasibility"),),
    )
    endog = jnp.asarray([0.0, 2.0, 4.0, 6.0, 8.0], dtype=dtype)

    def solve(query):
        return envelope_at_query(
            endog_grid=endog,
            value=endog,
            policy=2.0 * endog,
            marginal=3.0 * endog,
            segment_id=jnp.zeros_like(endog),
            x_query=query,
            feasibility_partition=partition,
            feasible_interval_mask=jnp.array([False, True]),
        )

    queries = jnp.asarray([2.0, 4.0, 6.0], dtype=dtype)
    eager = solve(queries)
    jitted = jax.jit(solve)(queries)
    vmapped = jax.vmap(solve)(queries)

    expected_value = np.array([-np.inf, 4.0, 6.0])
    expected_policy = np.array([np.nan, 8.0, 12.0])
    expected_marginal = np.array([0.0, 12.0, 18.0])
    for actual in (eager, jitted, vmapped):
        np.testing.assert_allclose(actual[0], expected_value)
        np.testing.assert_allclose(actual[1], expected_policy, equal_nan=True)
        np.testing.assert_allclose(actual[2], expected_marginal)


def test_continuation_value_read_preserves_an_isolated_feasible_point() -> None:
    """An exact feasible child value survives adjacent infeasible carry rows."""
    boundary = jnp.float32(4.0)
    grid = jnp.array(
        [
            3.0,
            jnp.nextafter(boundary, -jnp.inf),
            boundary,
            jnp.nextafter(boundary, jnp.inf),
            5.0,
        ]
    )
    value = jnp.array([-jnp.inf, -jnp.inf, 7.0, -jnp.inf, -jnp.inf])
    query = jnp.array(
        [
            jnp.float32(3.5),
            boundary,
            jnp.float32(4.5),
        ]
    )

    actual = _interp_continuation_value(query=query, grid=grid, value=value)

    np.testing.assert_array_equal(actual, np.array([-np.inf, 7.0, -np.inf]))


def test_negative_infinity_candidate_cannot_poison_a_finite_exact_point() -> None:
    """A dead continuation candidate loses to a finite point at the same node."""
    boundary = jnp.float32(4.0)

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.array([boundary, boundary]),
        value=jnp.array([-jnp.inf, 7.0], dtype=boundary.dtype),
        policy=jnp.array([jnp.nan, 2.0], dtype=boundary.dtype),
        marginal=jnp.array([0.0, 3.0], dtype=boundary.dtype),
        segment_id=jnp.array([0.0, 1.0], dtype=boundary.dtype),
        x_query=boundary,
    )

    np.testing.assert_array_equal(
        (value, policy, marginal),
        np.array([7.0, 2.0, 3.0]),
    )
