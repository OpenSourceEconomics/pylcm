"""Derivatives through the native certified owner/read boundary."""

from __future__ import annotations

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope._exact_affine import (
    UNRESOLVED_STATUS,
    exact_affine_read,
    exact_query_winner,
    exact_query_winner_batched,
)
from _lcm.egm.upper_envelope.query import envelope_at_query

_DTYPES = (jnp.float32, jnp.float64) if jax.config.x64_enabled else (jnp.float32,)
_REQUIRES_X64 = pytest.mark.skipif(
    not jax.config.x64_enabled,
    reason="requires the suite's float64 configuration",
)


# keyword-only-exempt: library-callback=jax.jvp
def _read_pair(x0, x1, v0, v1, q):
    return exact_affine_read(x0=x0, x1=x1, v0=v0, v1=v1, x_query=q)


# keyword-only-exempt: library-callback=jax.vmap
def _read_value(x0, x1, v0, v1, q):
    return _read_pair(x0, x1, v0, v1, q)[0]


def _directional_fraction(*, primals, tangents) -> float:
    x0, x1, v0, v1, q = (Fraction(float(x)) for x in primals)
    dx0, dx1, dv0, dv1, dq = (Fraction(float(x)) for x in tangents)
    width = x1 - x0
    alpha = (q - x0) / width
    slope = (v1 - v0) / width
    return float(
        (1 - alpha) * dv0 + alpha * dv1 + slope * (dq - (1 - alpha) * dx0 - alpha * dx1)
    )


def _public_value(*, dtype, q):
    value, _, _ = envelope_at_query(
        endog_grid=jnp.asarray([0.0, 1.0], dtype=dtype),
        policy=jnp.asarray([1.0, 3.0], dtype=dtype),
        value=jnp.asarray([0.0, 2.0], dtype=dtype),
        marginal=jnp.asarray([4.0, 6.0], dtype=dtype),
        segment_id=jnp.asarray([0.0, 0.0], dtype=dtype),
        x_query=jnp.reshape(q, (1,)),
        arithmetic="certified",
    )
    return value[0]


@pytest.mark.parametrize("dtype", _DTYPES)
def test_public_certified_envelope_has_fixed_owner_query_derivative(dtype) -> None:
    q = jnp.asarray(0.25, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    primal, tangent = jax.jvp(lambda z: _public_value(dtype=dtype, q=z), (q,), (one,))
    np.testing.assert_allclose(np.asarray(primal), np.asarray(0.5, dtype=dtype))
    np.testing.assert_allclose(np.asarray(tangent), np.asarray(2.0, dtype=dtype))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_public_certified_envelope_supports_forward_over_forward(dtype) -> None:
    q = jnp.asarray(0.25, dtype=dtype)
    second = jax.jacfwd(jax.jacfwd(lambda z: _public_value(dtype=dtype, q=z)))(q)
    tolerance = 1e-5 if dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(np.asarray(second), 0.0, atol=tolerance, rtol=0)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_direct_exact_read_all_five_operands_match_fraction_oracle(dtype) -> None:
    primals = tuple(jnp.asarray(x, dtype=dtype) for x in (0.2, 1.3, -2.0, 4.0, 0.7))
    tangents = tuple(jnp.asarray(x, dtype=dtype) for x in (0.1, -0.2, 0.3, -0.4, 0.5))
    (value, status), (value_dot, status_dot) = jax.jvp(_read_pair, primals, tangents)
    expected = _directional_fraction(primals=primals, tangents=tangents)
    tolerance = 2e-5 if dtype == jnp.float32 else 2e-13
    assert int(status) == 0
    assert np.isfinite(float(value))
    np.testing.assert_allclose(
        float(value_dot), expected, rtol=tolerance, atol=tolerance
    )
    assert status_dot.dtype == jax.dtypes.float0


@pytest.mark.parametrize("dtype", _DTYPES)
def test_generated_smooth_family_matches_fraction_oracle(dtype) -> None:
    rng = np.random.default_rng(40701 + np.dtype(dtype).itemsize)
    raw_x0 = rng.uniform(-2.0, 2.0, 64)
    raw_width = np.exp2(rng.integers(-4, 3, 64))
    raw_v0 = rng.normal(size=64)
    raw_v1 = rng.normal(size=64)
    raw_q = raw_x0 + rng.uniform(0.05, 0.95, 64) * raw_width
    raw_directions = [rng.normal(size=64) for _ in range(5)]
    primals = tuple(
        jnp.asarray(x, dtype=dtype)
        for x in (raw_x0, raw_x0 + raw_width, raw_v0, raw_v1, raw_q)
    )
    tangents = tuple(jnp.asarray(x, dtype=dtype) for x in raw_directions)
    (values, status), (observed, status_dot) = jax.jvp(_read_pair, primals, tangents)
    expected = np.asarray(
        [
            _directional_fraction(
                primals=tuple(np.asarray(x)[i] for x in primals),
                tangents=tuple(np.asarray(x)[i] for x in tangents),
            )
            for i in range(64)
        ]
    )
    tolerance = 8e-5 if dtype == jnp.float32 else 2e-12
    np.testing.assert_array_equal(np.asarray(status), 0)
    assert np.isfinite(np.asarray(values)).all()
    np.testing.assert_allclose(
        np.asarray(observed), expected, rtol=tolerance, atol=tolerance
    )
    assert status_dot.dtype == jax.dtypes.float0


@pytest.mark.parametrize("batched", [False, True])
@_REQUIRES_X64
def test_owner_and_status_have_float0_tangents(*, batched: bool) -> None:
    dtype = jnp.float64
    if batched:
        left = jnp.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=dtype)
        right = jnp.asarray([[1.0, 2.0], [1.0, 2.0]], dtype=dtype)
        values = jnp.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=dtype)
        query = jnp.asarray([[0.5], [0.5]], dtype=dtype)
        live = jnp.ones_like(left, dtype=bool)

        def owner(*, a, b, c, d, q):
            return exact_query_winner_batched(
                left_grid=a,
                right_grid=b,
                left_value=c,
                right_value=d,
                live=live,
                x_query=q,
            )

    else:
        left = jnp.asarray([0.0, 1.0], dtype=dtype)
        right = jnp.asarray([1.0, 2.0], dtype=dtype)
        values = jnp.asarray([0.0, 1.0], dtype=dtype)
        query = jnp.asarray([0.5], dtype=dtype)
        live = jnp.ones_like(left, dtype=bool)

        # keyword-only-exempt: library-callback=jax.jvp
        def owner(a, b, c, d, q):
            return exact_query_winner(
                left_grid=a,
                right_grid=b,
                left_value=c,
                right_value=d,
                live=live,
                x_query=q,
            )

    primals = (left, right, values, values + 1.0, query)
    tangents = tuple(jnp.ones_like(x) for x in primals)
    (winner, status), (winner_dot, status_dot) = jax.jvp(owner, primals, tangents)
    np.testing.assert_array_equal(np.asarray(status), 0)
    assert np.isfinite(np.asarray(winner)).all()
    assert winner_dot.dtype == jax.dtypes.float0
    assert status_dot.dtype == jax.dtypes.float0


@_REQUIRES_X64
def test_jit_preserves_exact_primal_bits_and_tangent() -> None:
    dtype = jnp.float64
    primals = tuple(jnp.asarray(x, dtype=dtype) for x in (0.2, 1.3, -2.0, 4.0, 0.7))
    tangents = tuple(jnp.asarray(x, dtype=dtype) for x in (0.1, -0.2, 0.3, -0.4, 0.5))
    eager = jax.jvp(_read_pair, primals, tangents)
    compiled = jax.jit(lambda *z: jax.jvp(_read_pair, z, tangents))(*primals)
    np.testing.assert_array_equal(np.asarray(eager[0][0]), np.asarray(compiled[0][0]))
    np.testing.assert_array_equal(np.asarray(eager[0][1]), np.asarray(compiled[0][1]))
    np.testing.assert_allclose(np.asarray(eager[1][0]), np.asarray(compiled[1][0]))
    assert compiled[1][1].dtype == jax.dtypes.float0


def test_jvp_of_vmap_matches_vmap_of_scalar_jvp() -> None:
    primals = (
        jnp.asarray([0.0, 0.2, -1.0]),
        jnp.asarray([1.0, 1.4, 2.0]),
        jnp.asarray([-1.0, 2.0, 0.5]),
        jnp.asarray([3.0, -2.0, 4.0]),
        jnp.asarray([0.25, 0.8, 1.0]),
    )
    tangents = tuple(jnp.linspace(0.1, 0.3, 3) for _ in primals)

    def batched_value(*z):
        return jax.vmap(_read_value)(*z)

    value_a, tangent_a = jax.jvp(batched_value, primals, tangents)
    value_b, tangent_b = jax.vmap(
        lambda a, b, c, d, e, da, db, dc, dd, de: jax.jvp(
            _read_value, (a, b, c, d, e), (da, db, dc, dd, de)
        )
    )(*primals, *tangents)
    np.testing.assert_array_equal(np.asarray(value_a), np.asarray(value_b))
    np.testing.assert_allclose(np.asarray(tangent_a), np.asarray(tangent_b))


def test_shared_owner_custom_vmap_composes_with_jvp_and_jit() -> None:
    left = jnp.asarray([0.0, 1.0])
    right = jnp.asarray([1.0, 2.0])
    v0 = jnp.asarray([0.0, 1.0])
    v1 = jnp.asarray([1.0, 2.0])
    live = jnp.asarray([True, True])
    queries = jnp.asarray([0.25, 0.5, 0.75])

    def one(q):
        return exact_query_winner(
            left_grid=left,
            right_grid=right,
            left_value=v0,
            right_value=v1,
            live=live,
            x_query=q,
        )

    bank = jax.vmap(one)
    (winner, status), (winner_dot, status_dot) = jax.jit(
        lambda q, dq: jax.jvp(bank, (q,), (dq,))
    )(queries, jnp.ones_like(queries))
    np.testing.assert_array_equal(np.asarray(winner), 0)
    np.testing.assert_array_equal(np.asarray(status), 0)
    assert winner_dot.dtype == jax.dtypes.float0
    assert status_dot.dtype == jax.dtypes.float0


def test_varying_segment_vmap_composes_with_jvp() -> None:
    left = jnp.asarray([[0.0, 1.0], [0.0, 1.0]])
    right = jnp.asarray([[1.0, 2.0], [1.0, 2.0]])
    v0 = jnp.asarray([[0.0, 1.0], [1.0, 0.0]])
    v1 = jnp.asarray([[1.0, 2.0], [2.0, 1.0]])
    live = jnp.ones_like(left, dtype=bool)
    query = jnp.asarray([0.5, 0.5])

    # keyword-only-exempt: library-callback=jax.vmap
    def one(a, b, c, d, mask, q):
        return exact_query_winner(
            left_grid=a,
            right_grid=b,
            left_value=c,
            right_value=d,
            live=mask,
            x_query=q,
        )

    bank = jax.vmap(one)
    primals = (left, right, v0, v1, live, query)
    tangents = (
        jnp.ones_like(left),
        jnp.ones_like(right),
        jnp.ones_like(v0),
        jnp.ones_like(v1),
        jnp.zeros_like(live, dtype=jax.dtypes.float0),
        jnp.ones_like(query),
    )
    (winner, status), (winner_dot, status_dot) = jax.jvp(bank, primals, tangents)
    np.testing.assert_array_equal(np.asarray(status), 0)
    assert np.isfinite(np.asarray(winner)).all()
    assert winner_dot.dtype == jax.dtypes.float0
    assert status_dot.dtype == jax.dtypes.float0


def test_direct_forward_over_forward_read_is_finite() -> None:
    x0, x1, v0, v1 = map(jnp.asarray, (0.2, 1.3, -2.0, 4.0))
    q = jnp.asarray(0.7)
    second = jax.jacfwd(jax.jacfwd(lambda z: _read_value(x0, x1, v0, v1, z)))(q)
    np.testing.assert_allclose(np.asarray(second), 0.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
def test_alternate_shapes_preserve_tangent_shape(*, dtype, shape) -> None:
    base = np.ones(shape or (), dtype=np.dtype(dtype))
    primals = tuple(
        jnp.asarray(x, dtype=dtype)
        for x in (0.0 * base, 2.0 * base, -1.0 * base, 3.0 * base, 0.5 * base)
    )
    tangents = tuple(jnp.ones_like(x) for x in primals)
    (_, status), (value_dot, status_dot) = jax.jvp(_read_pair, primals, tangents)
    assert value_dot.shape == shape
    np.testing.assert_array_equal(np.asarray(status), 0)
    assert np.isfinite(np.asarray(value_dot)).all()
    assert status_dot.dtype == jax.dtypes.float0


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("scale", [2.0**-20, 1.0, 2.0**20])
@pytest.mark.parametrize("translation", [-1e6, 0.0, 1e6])
def test_scale_translation_mutations_follow_working_geometry(
    *, dtype, scale, translation
) -> None:
    primals = tuple(
        jnp.asarray(x, dtype=dtype)
        for x in (
            translation + scale * 0.2,
            translation + scale * 1.2,
            -3.0,
            4.0,
            translation + scale * 0.7,
        )
    )
    tangents = tuple(
        jnp.asarray(x, dtype=dtype)
        for x in (scale * 0.1, scale * -0.2, 0.4, -0.5, scale * 0.3)
    )
    (_, status), (observed, _) = jax.jvp(_read_pair, primals, tangents)
    if bool(np.asarray(primals[1] > primals[0])):
        expected = _directional_fraction(primals=primals, tangents=tangents)
        tolerance = 1e-3 if dtype == jnp.float32 else 2e-9
        assert int(status) == 0
        np.testing.assert_allclose(
            float(observed), expected, rtol=tolerance, atol=tolerance
        )
    else:
        assert int(status) == UNRESOLVED_STATUS
        assert np.isnan(float(observed))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_zero_width_fails_closed(dtype) -> None:
    primals = tuple(jnp.asarray(x, dtype=dtype) for x in (1.0, 1.0, 2.0, 2.0, 1.0))
    tangents = tuple(jnp.ones_like(x) for x in primals)
    (value, status), (value_dot, status_dot) = jax.jvp(_read_pair, primals, tangents)
    assert int(status) == UNRESOLVED_STATUS
    assert np.isnan(float(value))
    assert np.isnan(float(value_dot))
    assert status_dot.dtype == jax.dtypes.float0


@pytest.mark.parametrize("dtype", _DTYPES)
def test_reversed_adjacent_width_fails_closed(dtype) -> None:
    high = np.asarray(1.0, dtype=np.dtype(dtype))
    low = np.nextafter(high, np.asarray(0.0, dtype=np.dtype(dtype)))
    primals = tuple(jnp.asarray(x, dtype=dtype) for x in (high, low, 2.0, 3.0, high))
    tangents = tuple(jnp.ones_like(x) for x in primals)
    (_, status), (value_dot, _) = jax.jvp(_read_pair, primals, tangents)
    assert int(status) == UNRESOLVED_STATUS
    assert np.isnan(float(value_dot))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_positive_adjacent_width_is_differentiable(dtype) -> None:
    x0 = np.asarray(1.0, dtype=np.dtype(dtype))
    x1 = np.nextafter(x0, np.asarray(np.inf, dtype=np.dtype(dtype)))
    v0 = np.asarray(2.0, dtype=np.dtype(dtype))
    v1 = np.nextafter(v0, np.asarray(np.inf, dtype=np.dtype(dtype)))
    primals = tuple(jnp.asarray(x, dtype=dtype) for x in (x0, x1, v0, v1, x0))
    tangents = tuple(jnp.asarray(x, dtype=dtype) for x in (0.0, 0.0, 0.0, 0.0, 1.0))
    (_, status), (value_dot, _) = jax.jvp(_read_pair, primals, tangents)
    assert int(status) == 0
    assert np.isfinite(float(value_dot))
    assert float(value_dot) > 0.0


@pytest.mark.parametrize("bad_index", range(5))
def test_nonfinite_primal_fails_closed(bad_index: int) -> None:
    values = [0.2, 1.3, -2.0, 4.0, 0.7]
    values[bad_index] = np.nan
    primals = tuple(jnp.asarray(x) for x in values)
    tangents = tuple(jnp.ones_like(x) for x in primals)
    (value, status), (value_dot, _) = jax.jvp(_read_pair, primals, tangents)
    assert int(status) == UNRESOLVED_STATUS
    assert np.isnan(float(value))
    assert np.isnan(float(value_dot))


def test_nonfinite_direction_fails_closed_without_changing_primal() -> None:
    primals = tuple(map(jnp.asarray, (0.2, 1.3, -2.0, 4.0, 0.7)))
    tangents = [jnp.ones_like(x) for x in primals]
    tangents[2] = jnp.asarray(np.inf)
    (value, status), (value_dot, _) = jax.jvp(_read_pair, primals, tuple(tangents))
    assert int(status) == 0
    assert np.isfinite(float(value))
    assert np.isnan(float(value_dot))


def test_permuting_unique_segments_preserves_physical_owner_and_float0() -> None:
    left = np.asarray([0.0, 0.0])
    right = np.asarray([1.0, 1.0])
    v0 = np.asarray([0.0, 2.0])
    v1 = np.asarray([1.0, 3.0])
    query = jnp.asarray([0.5])
    live = jnp.asarray([True, True])
    for permutation in (np.asarray([0, 1]), np.asarray([1, 0])):
        primals = (
            *(jnp.asarray(x[permutation]) for x in (left, right, v0, v1)),
            query,
        )

        # keyword-only-exempt: library-callback=jax.jvp
        def resolve(a, b, c, d, q):
            return exact_query_winner(
                left_grid=a,
                right_grid=b,
                left_value=c,
                right_value=d,
                live=live,
                x_query=q,
            )

        tangents = tuple(jnp.ones_like(x) for x in primals)
        (winner, status), (winner_dot, status_dot) = jax.jvp(resolve, primals, tangents)
        physical_owner = int(permutation[int(np.asarray(winner)[0])])
        assert physical_owner == 1
        assert int(np.asarray(status)[0]) == 0
        assert winner_dot.dtype == jax.dtypes.float0
        assert status_dot.dtype == jax.dtypes.float0


@pytest.mark.parametrize("dtype", _DTYPES)
def test_certified_read_carries_the_slope_in_forward_mode(dtype) -> None:
    """`jax.jacfwd` reports the exact affine slope of the certified read."""
    q = jnp.asarray(0.25, dtype=dtype)
    observed = jax.jacfwd(lambda z: _public_value(dtype=dtype, q=z))(q)
    np.testing.assert_allclose(np.asarray(observed), np.asarray(2.0, dtype=dtype))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_certified_read_refuses_reverse_mode(dtype) -> None:
    """Reverse mode is unavailable because no reverse rule is registered.

    On finite directions the affine differential is linear and has a
    mathematical transpose. The custom JVP also inspects tangent finiteness, so
    JAX cannot automatically transpose the registered rule. Callers needing
    `jax.grad` use `arithmetic="ordinary"`.
    """
    q = jnp.asarray(0.25, dtype=dtype)
    with pytest.raises((AssertionError, TypeError, ValueError)):
        jax.grad(lambda z: _public_value(dtype=dtype, q=z))(q)
