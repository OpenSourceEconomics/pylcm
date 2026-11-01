"""NEGM current publication uses its actual domain and certified floor source."""

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike
from typing_extensions import TypedDict

from _lcm.egm import step_core
from _lcm.egm.upper_envelope.segment_envelope import _NegmBindingEnvelope
from lcm.typing import Float1D, FloatND, ScalarFloat, ScalarInt
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution.test_negm_native_binding import _bits_equal, _refine
from tests.solution.test_segment_envelope import F1_GRID, F1_POLICY, F1_VALUE

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

type _NumericalInput = ArrayLike | Sequence[ArrayLike]


class _PublicationInputs(TypedDict, closed=True):
    refined_grid: Float1D
    refined_policy: Float1D
    refined_value: Float1D
    n_kept: ScalarInt
    n_pad: int
    publish_resources: FloatND
    lowest_euler_point: ScalarFloat
    borrowing_limit: ScalarFloat
    utility_of_action: Callable[[ScalarFloat], ScalarFloat]
    discounted_expected_value_at_limit: ScalarFloat


def _publish(
    *,
    row: _NegmBindingEnvelope,
    resources: _NumericalInput,
    bound: ArrayLike,
    discounted_corner: ArrayLike,
    active: bool,
) -> tuple[FloatND, Float1D, Float1D]:
    kwargs: _PublicationInputs = {
        "refined_grid": row.grid,
        "refined_policy": row.policy,
        "refined_value": row.value,
        "n_kept": row.n_kept,
        "n_pad": row.grid.size,
        "publish_resources": jnp.asarray(resources),
        "lowest_euler_point": row.grid[0],
        "borrowing_limit": jnp.asarray(bound),
        "utility_of_action": jnp.log,
        "discounted_expected_value_at_limit": jnp.asarray(discounted_corner),
    }
    if active:
        return step_core._publish_V_and_carry_rows(
            **kwargs, _negm_current_publication=True, _binding_query=row.binding_query
        )
    return step_core._publish_V_and_carry_rows(**kwargs)


def test_binding_publication_uses_nonzero_floor_and_discounted_continuation() -> None:
    """A represented binding query has log(R-b)+betaW(b), in resource units."""
    bound, beta, w_at_bound = 0.25, 0.5, 3.0
    # Existing native concave fixture coordinates/capacity. Explicit source group
    # owns A=b; no resource-minus-policy inference supplies that ownership.
    grid = jnp.asarray([1.0, 2.0, 3.0])
    policy = grid - bound
    discounted_corner = beta * w_at_bound
    row = _refine(
        grid=grid,
        policy=policy,
        value=jnp.log(policy) + discounted_corner,
        floor_sources=3,
        queries=[1.25, 1.75, 2.5],
    )
    active = _publish(
        row=row,
        resources=[1.25, 1.75, 2.5],
        bound=bound,
        discounted_corner=discounted_corner,
        active=True,
    )
    expected = jnp.log(jnp.asarray([1.25, 1.75, 2.5]) - bound) + discounted_corner
    np.testing.assert_array_almost_equal(active[0], expected, decimal=DECIMAL_PRECISION)
    legacy = _publish(
        row=row,
        resources=[1.25, 1.75, 2.5],
        bound=bound,
        discounted_corner=discounted_corner,
        active=False,
    )
    # Current V correction must not rewrite the original carried value/M rows.
    for actual, expected in zip(active[1:], legacy[1:], strict=True):
        _bits_equal(actual=actual, expected=expected)
    np.testing.assert_array_almost_equal(
        active[2][:3], 1.0 / policy, decimal=DECIMAL_PRECISION
    )


def test_domain_uses_own_cost_once_before_comparison_with_the_floor() -> None:
    """Own R=2*x+3-2 is compared with b=.25 without rescaling the floor."""
    bound, discounted_corner = 0.25, 1.5
    state = jnp.asarray([-0.5, -0.375, 0.125, 1.0])
    own_cost = 2.0
    resources = 2.0 * state + 3.0 - own_cost
    np.testing.assert_array_equal(resources, [0.0, 0.25, 1.25, 3.0])
    grid = jnp.asarray([1.0, 2.0, 3.0])
    row = _refine(
        grid=grid,
        policy=grid - bound,
        value=jnp.log(grid - bound) + discounted_corner,
        floor_sources=3,
        queries=resources,
    )
    result = _publish(
        row=row,
        resources=resources,
        bound=bound,
        discounted_corner=discounted_corner,
        active=True,
    )
    np.testing.assert_array_equal(result[0][:2], [-np.inf, -np.inf])
    np.testing.assert_array_almost_equal(
        result[0][2:],
        jnp.log(resources[2:] - bound) + discounted_corner,
        decimal=DECIMAL_PRECISION,
    )
    # This fixed-cell unit consumes actual post-cost R. Genuine source plumbing
    # is separately checked by the public installed-producer test and original6.


def test_a_winning_nonconcave_interior_value_is_never_capped_at_the_corner() -> None:
    """An interior source retains its original interpolation when it wins."""
    queries = jnp.asarray([0.8, 1.1])
    row = _refine(
        grid=F1_GRID, policy=F1_POLICY, value=F1_VALUE, floor_sources=2, queries=queries
    )
    np.testing.assert_array_equal(row.binding_query, [False, False])
    active = _publish(
        row=row, resources=queries, bound=0.0, discounted_corner=-10.0, active=True
    )
    legacy = _publish(
        row=row, resources=queries, bound=0.0, discounted_corner=-10.0, active=False
    )
    for actual, expected in zip(active, legacy, strict=True):
        _bits_equal(actual=actual, expected=expected)
    assert np.all(np.asarray(active[0]) > np.asarray(jnp.log(queries) - 10.0))


def test_empty_domain_does_not_hide_native_overflow_poison() -> None:
    """A known empty R<=b cell still surfaces the existing overflow failure."""
    # Four branches cover (13, 14), one more than the capacity holds.
    grid = [10.0, 14.0, 11.0, 17.0, 12.0, 20.0, 13.0, 23.0, 15.0, 26.0]
    row = _refine(
        grid=grid,
        policy=[1.0] * len(grid),
        value=[float(i) for i in range(len(grid))],
        floor_sources=2,
        queries=[0.0, 0.25],
        max_runs=3,
    )
    result = _publish(
        row=row, resources=[0.0, 0.25], bound=0.25, discounted_corner=1.5, active=True
    )
    assert all(np.isnan(np.asarray(channel)).all() for channel in result)


def test_invalid_resource_arithmetic_is_not_laundered_into_no_action() -> None:
    """NaN and positive infinity retain nonfinite arithmetic classification."""
    grid = jnp.asarray([1.0, 2.0, 3.0])
    queries = jnp.asarray([jnp.nan, jnp.inf, -jnp.inf])
    row = _refine(
        grid=grid,
        policy=grid - 0.25,
        value=jnp.log(grid - 0.25) + 1.5,
        floor_sources=3,
        queries=queries,
    )
    result = _publish(
        row=row, resources=queries, bound=0.25, discounted_corner=1.5, active=True
    )
    np.testing.assert_array_equal(np.isnan(np.asarray(result[0])), [True, True, True])


def test_emitted_capacity_failure_precedes_finite_empty_publication() -> None:
    """A truncated native row poisons all channels even where finite R<=b."""
    row = _refine(
        grid=[1.0, 2.0, 3.0],
        policy=[0.5, 1.0, 1.5],
        value=[0.0, 1.0, 1.5],
        floor_sources=3,
        queries=[0.0, 0.25],
        capacity=2,
    )
    assert int(row.n_kept) > 2
    result = _publish(
        row=row, resources=[0.0, 0.25], bound=0.25, discounted_corner=1.5, active=True
    )
    assert all(np.isnan(np.asarray(channel)).all() for channel in result)


def test_refused_gap_emission_cannot_be_replaced_by_a_finite_corner() -> None:
    """A selected floor source cannot clear the native emission's numeric failure."""
    maximum = jnp.finfo(jnp.asarray(0.0).dtype).max
    row = _refine(
        grid=[0.0, 1.0, np.nan, 3.0, 4.0],
        policy=[maximum / 2.0, maximum, np.nan, 13.0, 14.0],
        value=[0.0, 1.0, np.nan, 3.0, 4.0],
        floor_sources=2,
        queries=[0.5, 0.25],
        capacity=16,
    )
    assert 0 < int(row.n_kept) <= 16
    assert np.isnan(np.asarray(row.policy)[: int(row.n_kept)]).any()
    assert not np.asarray(row.binding_query).any()
    result = _publish(
        row=row, resources=[0.5, 0.25], bound=0.25, discounted_corner=1.5, active=True
    )
    assert np.isnan(np.asarray(result[0])).all()
    assert np.isnan(np.asarray(result[1])).any()
    assert np.isnan(np.asarray(result[2])).any()
