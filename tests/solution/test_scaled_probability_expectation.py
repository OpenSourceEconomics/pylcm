"""Regression tests for the production scaled-probability reduction."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.probability import scaled_down_by_power_of_two
from _lcm.regime_building.zero_safe import zero_safe_average
from lcm.typing import FloatND, IntND


def _maybe_jit(
    *, func: Callable[..., FloatND], compiled: bool
) -> Callable[..., FloatND]:
    return jax.jit(func) if compiled else func


def _mean(*, values: FloatND, weights: FloatND, shifts: IntND) -> FloatND:
    """The scaled weighted mean, as a positional-argument function to transform."""
    return zero_safe_average(a=values, weights=weights, shifts=shifts)


@pytest.mark.parametrize("primitive", ["ldexp", "frexp"])
def test_the_reduction_scales_by_bit_arithmetic_not_by_a_general_primitive(
    primitive: str,
) -> None:
    """Keep the general exponent primitives off the stochastic-node hot path.

    Every scale this reduction applies is a non-positive power of two, which
    `scaled_down_by_power_of_two` performs on the bits. Reaching for `ldexp` or
    `frexp` instead would run a general exponent decomposition over the whole
    value surface — twice, since the mass and the numerator each need the scale.
    """
    dtype = jnp.zeros(()).dtype
    traced = jax.make_jaxpr(_mean)(
        jnp.ones(5, dtype=dtype),
        jnp.full(5, 0.2, dtype=dtype),
        jnp.zeros(5, dtype=jnp.int32),
    )

    assert f"name={primitive}" not in str(traced)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_downward_scaling_matches_ieee_ldexp_at_boundaries(*, compiled: bool) -> None:
    """The fast scaler is exact across normal, subnormal and zero results."""
    dtype = jnp.zeros(()).dtype
    finfo = np.finfo(np.dtype(dtype))
    values = np.asarray(
        [
            1.0,
            -1.0,
            finfo.max,
            finfo.tiny,
            np.nextafter(finfo.tiny, 0, dtype=np.dtype(dtype)),
            finfo.smallest_subnormal,
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=np.dtype(dtype),
    )
    min_normal_exponent = int(finfo.minexp)
    min_subnormal_exponent = min_normal_exponent - int(finfo.nmant)
    shifts = np.asarray(
        [
            0,
            -1,
            -2,
            min_normal_exponent,
            min_normal_exponent - 1,
            min_subnormal_exponent,
            min_subnormal_exponent - 1,
            -1,
            -7,
            -7,
            -7,
        ],
        dtype=np.int32,
    )
    scale = _maybe_jit(func=scaled_down_by_power_of_two, compiled=compiled)

    got = np.asarray(scale(jnp.asarray(values), jnp.asarray(shifts)))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        expected = np.ldexp(values, shifts).astype(values.dtype)

    integer_dtype = np.uint32 if values.dtype == np.float32 else np.uint64
    got_bits = got.view(integer_dtype)
    expected_bits = expected.view(integer_dtype)
    both_nan = np.isnan(got) & np.isnan(expected)
    np.testing.assert_array_equal(got_bits[~both_nan], expected_bits[~both_nan])


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_downward_scaling_rounds_subnormal_ties_to_even(*, compiled: bool) -> None:
    """Halfway subnormal results round according to the retained low bit."""
    dtype = jnp.zeros(()).dtype
    np_dtype = np.dtype(dtype)
    integer_dtype = np.uint32 if np_dtype == np.dtype(np.float32) else np.uint64
    smallest_normal_bits = np.asarray(np.finfo(np_dtype).tiny).view(integer_dtype)
    bits = np.asarray(
        [smallest_normal_bits + 1, smallest_normal_bits + 3],
        dtype=integer_dtype,
    )
    values = bits.view(np_dtype)
    shifts = np.asarray([-1, -1], dtype=np.int32)
    scale = _maybe_jit(func=scaled_down_by_power_of_two, compiled=compiled)

    got = np.asarray(scale(jnp.asarray(values), jnp.asarray(shifts)))
    expected = np.ldexp(values, shifts).astype(np_dtype)

    np.testing.assert_array_equal(got.view(integer_dtype), expected.view(integer_dtype))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_downward_scaling_matches_ieee_ldexp_on_random_finite_values(
    *, compiled: bool
) -> None:
    """The bit implementation matches IEEE rounding over the finite bit space."""
    dtype = jnp.zeros(()).dtype
    np_dtype = np.dtype(dtype)
    integer_dtype = np.uint32 if np_dtype == np.dtype(np.float32) else np.uint64
    rng = np.random.default_rng(20260809 + np_dtype.itemsize)
    raw = rng.integers(0, np.iinfo(integer_dtype).max, size=4096, dtype=integer_dtype)
    values = raw.view(np_dtype)
    values = values[np.isfinite(values)]
    minimum_shift = -350 if np_dtype == np.dtype(np.float32) else -3000
    shifts = rng.integers(minimum_shift, 1, size=values.size, dtype=np.int32)
    scale = _maybe_jit(func=scaled_down_by_power_of_two, compiled=compiled)

    got = np.asarray(scale(jnp.asarray(values), jnp.asarray(shifts)))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        expected = np.ldexp(values, shifts).astype(np_dtype)

    np.testing.assert_array_equal(got.view(integer_dtype), expected.view(integer_dtype))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_downward_scaling_has_the_expected_value_gradient(*, compiled: bool) -> None:
    """The bit-level primal retains the ordinary power-of-two derivative."""
    dtype = jnp.zeros(()).dtype
    shifts = jnp.asarray([0, -1, -7], dtype=jnp.int32)

    def total(values: FloatND) -> FloatND:
        return jnp.sum(scaled_down_by_power_of_two(values=values, shift=shifts))

    gradient = jax.grad(_maybe_jit(func=total, compiled=compiled))(
        jnp.asarray([1.0, 1.0, 1.0], dtype=dtype)
    )
    expected = jnp.asarray([1.0, 0.5, 2.0**-7], dtype=dtype)

    np.testing.assert_array_equal(np.asarray(gradient), np.asarray(expected))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_a_large_rare_contribution_is_formed_before_its_scale_is_applied(
    *, compiled: bool
) -> None:
    """A probability below the common scale can still contribute one quarter.

    The scaled coefficient is the smallest normal number and therefore safe to
    multiply. Applying its residual scale before it meets the value would make
    the probability subnormal and let the backend flush a contribution of 1/4.
    """
    dtype = jnp.zeros(()).dtype
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    large = jnp.asarray(1.0, dtype=dtype) / tiny
    reduce = _maybe_jit(func=_mean, compiled=compiled)

    got = reduce(
        jnp.asarray([0.0, large], dtype=dtype),
        jnp.asarray([1.0, tiny], dtype=dtype),
        jnp.asarray([0, 2], dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(0.25, dtype=dtype))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("kind", ["nan", "positive_infinity", "negative_infinity"])
def test_a_live_nonfinite_node_is_not_reclassified_as_null_by_its_scale(
    *, compiled: bool, kind: str
) -> None:
    """A live node stays non-finite even when its plain weight underflows."""
    reduce = _maybe_jit(func=_mean, compiled=compiled)
    spread = 300 if jnp.zeros(()).dtype == jnp.float32 else 2800
    nonfinite = {
        "nan": jnp.nan,
        "positive_infinity": jnp.inf,
        "negative_infinity": -jnp.inf,
    }[kind]

    got = reduce(
        jnp.asarray([1.0, nonfinite]),
        jnp.ones(2),
        jnp.asarray([0, spread], dtype=jnp.int32),
    )

    if kind == "nan":
        assert bool(jnp.isnan(got))
    elif kind == "positive_infinity":
        assert bool(jnp.isposinf(got))
    else:
        assert bool(jnp.isneginf(got))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_a_represented_zero_remains_the_null_event_at_any_scale(
    *, compiled: bool
) -> None:
    """A genuine zero coefficient annihilates a non-finite value."""
    reduce = _maybe_jit(func=_mean, compiled=compiled)

    got = reduce(
        jnp.asarray([1.0, jnp.nan]),
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([0, 300], dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(1.0))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_the_mass_is_read_on_the_same_common_scale(*, compiled: bool) -> None:
    """Weights `(1, 1/4)` price the second node at one fifth."""
    reduce = _maybe_jit(func=_mean, compiled=compiled)

    got = reduce(
        jnp.asarray([0.0, 1.0]),
        jnp.ones(2),
        jnp.asarray([0, 2], dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(0.2))
