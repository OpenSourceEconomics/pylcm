"""Finite owned components preserve their published extreme-scale arithmetic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import _read_owned_row_with_support
from _lcm.egm.finite_negm_query import (
    _read_finite_native_row,
    _validate_finite_outer_rows,
)
from _lcm.egm.interp import _hermite_query_derivative
from _lcm.egm.upper_envelope.segment_envelope import DEFAULT_MAX_RUNS
from lcm.typing import ScalarFloat
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm._finite_selection_component import component_view
from tests.egm.test_finite_negm_owned_germs import _emit

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def test_owned_overflow_component_preserves_ordinary_and_paired_channels() -> None:
    """Finite midpoint records retain ordinary M1 and the published paired M0."""
    dtype = jnp.asarray(0.0).dtype
    largest = jnp.asarray(jnp.finfo(dtype).max, dtype=dtype)
    extreme = largest * jnp.asarray(0.75, dtype=dtype)
    coordinate = extreme / 2.0
    assert bool(jnp.isfinite(extreme) & jnp.isfinite(coordinate))
    assert bool(extreme > largest / 2.0)
    assert bool(jnp.isposinf(extreme - (-extreme)))
    row, marginal, data = _emit(
        coordinates=(1.0, float(extreme)),
        values=(-float(extreme), float(extreme)),
        marginals=(1.0, 1.0),
    )
    local_grid = row.grid[:2]
    local_value = row.value[:2]
    local_marginal = marginal[:2]
    np.testing.assert_array_equal(
        (coordinate - local_grid[0]) / (local_grid[1] - local_grid[0]), 0.5
    )
    assert bool(jnp.all(jnp.isfinite(local_grid)))
    assert bool(jnp.all(jnp.isfinite(local_value)))
    assert bool(jnp.all(jnp.isfinite(local_marginal)))
    assert bool(row.link_valid[0])
    np.testing.assert_array_equal(row.owner_id[:2], [0, 0])

    def owned_value(point: ScalarFloat) -> ScalarFloat:
        return _read_owned_row_with_support(
            query=point,
            grid=row.grid,
            value=row.value,
            marginal=marginal,
            data=data,
            owner_capacity=DEFAULT_MAX_RUNS,
        )[0].value

    ordinary, support = _read_owned_row_with_support(
        query=coordinate,
        grid=row.grid,
        value=row.value,
        marginal=marginal,
        data=data,
        owner_capacity=DEFAULT_MAX_RUNS,
    )
    assert bool(ordinary.supported & ~ordinary.invalid_numeric)
    np.testing.assert_array_equal(support[:2], [True, True])
    np.testing.assert_array_equal(ordinary.value, 0.0)
    np.testing.assert_array_equal(ordinary.marginal, 1.0)
    np.testing.assert_array_equal(ordinary.savings, 0.1875)
    published_derivative = _hermite_query_derivative(
        x_query=coordinate,
        search_grid=local_grid,
        valid_length=jnp.int32(2),
        xp=local_grid,
        fp=local_value,
        fp_slopes=local_marginal,
    )
    np.testing.assert_array_equal(published_derivative, 0.0)
    np.testing.assert_array_equal(jax.grad(owned_value)(coordinate), 0.0)

    raw = _read_finite_native_row(
        coordinate=coordinate,
        grid=row.grid,
        value=row.value,
        marginal=marginal,
        data=data,
        owner_capacity=DEFAULT_MAX_RUNS,
        side_is_row_side=True,
    )
    for paired, expected_marginal in ((False, 1.0), (True, 0.0)):
        candidate = component_view(
            profile=_validate_finite_outer_rows(
                raw=raw,
                queries=coordinate,
                gradients=jnp.asarray(1.0),
                floor=jnp.asarray(0.0),
                upper_saving=jnp.asarray(2.0),
                resources=coordinate,
                eligible=jnp.asarray(a=True),
                corner_value=jnp.asarray(-10.0),
                corner_marginal=jnp.asarray(0.0),
                supported_corner=jnp.asarray(a=False),
                finite_corner=jnp.asarray(a=False),
                invalid_corner=jnp.asarray(a=False),
                paired_marginal_read=paired,
            )
        )
        assert not bool(candidate.coverage_missing)
        np.testing.assert_array_equal(candidate.value, 0.0)
        np.testing.assert_array_equal(candidate.right_marginal, expected_marginal)
        np.testing.assert_array_equal(candidate.left_marginal, expected_marginal)
        assert bool(candidate.right_germ[0] & candidate.left_germ[0])
        np.testing.assert_array_equal(jnp.stack(candidate.right_germ[1:]), [0.0] * 3)
        np.testing.assert_array_equal(jnp.stack(candidate.left_germ[1:]), [0.0] * 3)
