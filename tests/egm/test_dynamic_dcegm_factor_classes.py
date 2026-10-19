"""Preserve signed term scales and derivatives in canonical passive reads."""

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.dcegm_query import DCEGMQuery, DCEGMQueryResult
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from lcm.typing import FloatND, ScalarFloat
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm import test_dynamic_dcegm_passive_products as cases

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _query() -> DCEGMQuery:
    model, _ = cases._model_and_params(epsilon=0.25, scale=1.0)
    return cases._installed_query(model=model)


def _read(
    *,
    query: DCEGMQuery,
    z: ScalarFloat,
    w: ScalarFloat,
    bank: FloatND,
    values: FloatND | None = None,
    marginals: FloatND | None = None,
    gradients: FloatND | None = None,
) -> DCEGMQueryResult:
    capacity = query.model.authority.row_capacity
    shape = (2, 3, capacity)
    data = EGMQueryData(
        savings=jnp.zeros(shape),
        node_kind=jnp.full(shape, QueryNodeKind.PAD, dtype=jnp.int32),
        link_valid=jnp.zeros((2, 3, capacity - 1), dtype=jnp.bool_),
        owner_id=jnp.zeros(shape, dtype=jnp.int32),
        side_id=jnp.zeros(shape, dtype=jnp.int32),
        floor_used=jnp.zeros((2, 3)),
        corner_continuation=bank[..., None],
        corner_kind=jnp.full((2, 3, 1), QueryNodeKind.FINITE, dtype=jnp.int32),
        corner_interval_id=jnp.zeros((2, 3, 1), dtype=jnp.int32),
        corner_side_id=jnp.zeros((2, 3, 1), dtype=jnp.int32),
    )
    if values is not None:
        data = replace(
            data,
            node_kind=jnp.full(shape, QueryNodeKind.FINITE, dtype=jnp.int32),
            link_valid=jnp.ones((2, 3, capacity - 1), dtype=jnp.bool_),
        )
    return query(
        grid=jnp.broadcast_to(jnp.arange(1.0, capacity + 1.0), shape),
        value=jnp.zeros(shape) if values is None else values,
        marginal=jnp.zeros(shape) if marginals is None else marginals,
        data=data,
        row_queries=jnp.ones((2, 3)),
        row_gradients=jnp.ones((2, 3)) if gradients is None else gradients,
        passive_values=(z, w),
        passive_grids=(jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 1.0, 2.0])),
        pool={
            "liquid": jnp.asarray(0.0),
            "z": z,
            "w": w,
            "period": jnp.asarray(1),
            "age": jnp.asarray(41.0),
            "borrowing_lower__floor": jnp.asarray(0.0),
            "koopmans_aggregator__discount_factor": jnp.asarray(0.5),
        },
        resource_gradient=jnp.asarray(1.0),
    )


@pytest.mark.parametrize("reverse_axes", [False, True])
def test_signed_rescaling_retains_one_eighth_in_either_axis_order(
    *, reverse_axes: bool
) -> None:
    """Keep a normal term when a large signed coefficient rescales a small one."""
    query = _query()
    binary32 = jnp.asarray(1.0).dtype == jnp.float32
    small_exponent, large_exponent, value_exponent = (
        (80, 126, 48) if binary32 else (600, 1022, 424)
    )
    small = jnp.asarray(math.ldexp(1.0, -small_exponent))
    large = jnp.asarray(math.ldexp(1.0, large_exponent))
    amplitude = jnp.asarray(math.ldexp(1.0, -value_exponent))
    for number in (small, large, amplitude):
        assert np.isfinite(number)
        assert number >= np.finfo(number.dtype).tiny
    assert -small_exponent + large_exponent - value_exponent == -2
    z, w = (large, small) if reverse_axes else (small, large)
    bank = jnp.zeros((2, 3)).at[1, 1 if reverse_axes else 2].set(amplitude)
    result = _read(query=query, z=z, w=w, bank=bank)
    np.testing.assert_allclose(result.value, 0.125, rtol=10**-DECIMAL_PRECISION, atol=0)
    np.testing.assert_array_equal([result.marginal, result.savings], [1.0, 0.0])
    np.testing.assert_array_equal(
        [*result.bound_status, result.coverage_missing], [False, False, False]
    )


@pytest.mark.parametrize("material", [False, True])
@pytest.mark.parametrize("zero", [False, True])
def test_passive_corner_retains_forward_reverse_and_mixed_derivatives(
    *, material: bool, zero: bool
) -> None:
    """Differentiate the retained finite cross term in either passive coordinate."""
    query = _query()
    binary32 = jnp.asarray(1.0).dtype == jnp.float32
    exponent, scale_exponent = (64, 126) if binary32 else (512, 1022)
    epsilon = math.ldexp(1.0, -exponent) if material else 0.25
    scale = math.ldexp(1.0, scale_exponent) if material else 8.0

    # keyword-only-exempt: library-callback=jax.jvp
    def value(z: ScalarFloat, w: ScalarFloat) -> ScalarFloat:
        return _read(
            query=query,
            z=z,
            w=w,
            bank=cases._corner_bank(jnp.asarray(scale))[..., 0],
        ).value

    point = (jnp.asarray(0.0 if zero else epsilon), jnp.asarray(epsilon))
    expected_value = 0.0 if zero else (0.125 if material else 0.25)
    expected_slope = math.ldexp(1.0, scale_exponent - exponent - 1) if material else 1.0
    expected_mixed = 0.5 * scale
    for number in (expected_slope, expected_mixed):
        assert np.isfinite(jnp.asarray(number))
        assert number >= np.finfo(jnp.asarray(number).dtype).tiny
    primal, tangent = jax.jvp(value, point, (jnp.asarray(2.0), jnp.asarray(3.0)))
    reverse = jax.grad(value, argnums=(0, 1))(*point)
    mixed = jax.grad(jax.grad(value, argnums=0), argnums=1)(*point)
    np.testing.assert_allclose(
        [primal, tangent, *reverse, mixed],
        [
            expected_value,
            (2 if zero else 5) * expected_slope,
            expected_slope,
            0.0 if zero else expected_slope,
            expected_mixed,
        ],
        rtol=10**-DECIMAL_PRECISION,
        atol=0,
    )
    np.testing.assert_allclose(
        jax.jit(value)(*point), expected_value, rtol=10**-DECIMAL_PRECISION, atol=0
    )
    _, pullback = jax.vjp(value, *point)
    np.testing.assert_allclose(
        pullback(jnp.asarray(2.0)),
        [2 * expected_slope, 0.0 if zero else 2 * expected_slope],
        rtol=10**-DECIMAL_PRECISION,
        atol=0,
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(value, argnums=0))(*point),
        expected_slope,
        rtol=10**-DECIMAL_PRECISION,
        atol=0,
    )


@pytest.mark.parametrize("active_poison", [False, True])
def test_exact_zero_corner_poison_keeps_stencil_authority(
    *, active_poison: bool
) -> None:
    """Reject active poison and annihilate an exactly inactive nonfinite corner."""
    query = _query()
    bank = jnp.zeros((2, 3)).at[1, 1].set(jnp.nan)
    result = _read(
        query=query,
        z=jnp.asarray(0.25 if active_poison else 0.0),
        w=jnp.asarray(0.25),
        bank=bank,
    )
    assert bool(result.coverage_missing) is active_poison
    if not active_poison:
        np.testing.assert_array_equal(
            [result.value, result.marginal, result.savings], [0.0, 1.0, 0.0]
        )


def test_associated_row_marginal_preserves_its_gradient_factor() -> None:
    """Keep native marginal and transition factors associated through weighting."""
    query = _query()
    binary32 = jnp.asarray(1.0).dtype == jnp.float32
    small_exponent, large_exponent = (80, 126) if binary32 else (600, 1022)
    small = jnp.asarray(math.ldexp(1.0, -small_exponent))
    large = jnp.asarray(math.ldexp(1.0, large_exponent))
    shape = (2, 3, query.model.authority.row_capacity)
    values = jnp.zeros(shape).at[1, 0, :].set(1.0)
    marginals = jnp.zeros(shape).at[1, 0, :].set(small)
    expected = math.ldexp(1.0, large_exponent - 2 * small_exponent)
    assert expected >= np.finfo(small.dtype).tiny
    result = _read(
        query=query,
        z=large,
        w=jnp.asarray(0.0),
        bank=jnp.zeros((2, 3)),
        values=values,
        marginals=marginals,
        gradients=jnp.full((2, 3), small),
    )
    np.testing.assert_allclose(result.value, large, rtol=10**-DECIMAL_PRECISION, atol=0)
    np.testing.assert_allclose(
        result.marginal, expected, rtol=10**-DECIMAL_PRECISION, atol=0
    )
    np.testing.assert_array_equal(
        [result.savings, *result.bound_status, result.coverage_missing], 0
    )


def test_finite_zero_corner_survives_large_signed_axis_products() -> None:
    """Keep a zero finite bank at zero despite overflowing intermediate weights."""
    query = _query()
    exponent = 126 if jnp.asarray(1.0).dtype == jnp.float32 else 1022
    coordinate = jnp.asarray(math.ldexp(1.0, exponent))
    assert np.isfinite(coordinate)
    result = _read(query=query, z=coordinate, w=coordinate, bank=jnp.zeros((2, 3)))
    np.testing.assert_array_equal(
        [result.value, result.marginal, result.savings], [0.0, 1.0, 0.0]
    )
    np.testing.assert_array_equal(
        [*result.bound_status, result.coverage_missing], [False, False, False]
    )
