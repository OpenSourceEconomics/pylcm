"""The power-mean kernels preserve constant payoffs at representation boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.power_mean import weighted_power_mean, weighted_power_mean_of_pair

_EXPONENTS = (-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0)


def _bits(value):
    arr = np.asarray(value)
    uint = np.uint32 if arr.dtype.itemsize == 4 else np.uint64
    return arr.view(uint)


def _boundary_payoffs():
    dtype = jnp.zeros(()).dtype
    info = jnp.finfo(dtype)
    return (
        dtype.type(0.0),
        dtype.type(info.smallest_subnormal),
        dtype.type(info.smallest_subnormal * 8),
        dtype.type(info.tiny),
        dtype.type(info.max),
    )


@pytest.mark.parametrize("exponent", _EXPONENTS)
def test_constant_lottery_keeps_boundary_payoff_bits(exponent: float) -> None:
    """Every constant lottery returns its payoff's exact representation."""
    dtype = jnp.zeros(()).dtype
    wide = 1100 if dtype.itemsize == 8 else 200

    def aggregate(*, values, shifts):
        return weighted_power_mean(
            values=values,
            weights=jnp.asarray([2.0, 1.0, 0.5], dtype=dtype),
            exponent=jnp.asarray(exponent, dtype=dtype),
            shifts=shifts,
        )

    compiled = jax.jit(aggregate)
    for payoff in _boundary_payoffs():
        values = jnp.full((3,), payoff, dtype=dtype)
        for shifts in (
            jnp.zeros((3,), dtype=jnp.int32),
            jnp.asarray([0, wide - 20, wide], dtype=jnp.int32),
        ):
            expected_bits = _bits(payoff)
            assert _bits(aggregate(values=values, shifts=shifts)) == expected_bits
            assert _bits(compiled(values, shifts)) == expected_bits


@pytest.mark.parametrize("exponent", _EXPONENTS)
def test_constant_pair_keeps_boundary_payoff_bits(exponent: float) -> None:
    """The two-node Koopmans specialization obeys the same invariant."""
    dtype = jnp.zeros(()).dtype

    def aggregate(payoff):
        return weighted_power_mean_of_pair(
            first=jnp.asarray(payoff, dtype=dtype),
            second=jnp.asarray(payoff, dtype=dtype),
            first_weight=jnp.asarray(2.0, dtype=dtype),
            second_weight=jnp.asarray(0.5, dtype=dtype),
            exponent=jnp.asarray(exponent, dtype=dtype),
        )

    compiled = jax.jit(aggregate)
    for payoff in _boundary_payoffs():
        expected_bits = _bits(payoff)
        assert _bits(aggregate(payoff)) == expected_bits
        assert _bits(compiled(payoff)) == expected_bits


def test_wide_scaled_all_zero_geometric_lottery_is_zero() -> None:
    """A live all-zero row is zero even when a normalized weight vanishes."""
    dtype = jnp.zeros(()).dtype
    wide = 1100 if dtype.itemsize == 8 else 200
    got = jax.jit(weighted_power_mean)(
        values=jnp.zeros((2,), dtype=dtype),
        weights=jnp.ones((2,), dtype=dtype),
        exponent=jnp.asarray(0.0, dtype=dtype),
        shifts=jnp.asarray([0, wide], dtype=jnp.int32),
    )
    assert _bits(got) == _bits(dtype.type(0.0))
