"""A scaled lottery reduces to its exact mean across the whole admissible class.

A node of a continuation lottery carries probability `coefficient * 2**-shift`,
and the reduction has to survive every combination the pair can legitimately
take: a significand anywhere in its normalized range, a shift anywhere the
exponent field reaches, and a value anywhere from zero to the largest the format
holds. Two orderings each fail at one end — forming the weight first loses a rare
node whose product is ordinary, forming `coefficient * value` first overflows
when a significand above one meets a value near the top of the range — so the
class is only covered by generating both ends together rather than by pinning
either witness.

The reference is exact rational arithmetic on the same inputs, which shares no
code, no floating-point rounding, and no ordering decision with the
implementation under test.
"""

import itertools
from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcm import LinearExpectation, QuasiArithmeticMean

# Normalized significands, including the above-one range that can overflow.
_SIGNIFICANDS = (1.0, 1.5, 1.9)

# Value as a fraction of the format's maximum; `1.0` means an ordinary `1.0`.
_VALUE_FRACTIONS = (0.0, 1.0, 0.75)


def _dtype_case() -> tuple[Any, tuple[int, ...]]:
    """The working dtype and a spread of shifts its exponent field reaches."""
    dtype = jnp.zeros(()).dtype
    if dtype.itemsize == 4:
        return dtype, (0, 16, 40, 64, 100, 126)
    return dtype, (0, 64, 200, 512, 800, 1020)


def _value(fraction: float, dtype: Any) -> float:
    """Turn a value fraction into a concrete value of the working dtype."""
    if fraction == 1.0:
        return 1.0
    return fraction * float(jnp.finfo(dtype).max)


def _exact_mean(
    *,
    significands: tuple[float, ...],
    shifts: tuple[int, ...],
    values: tuple[float, ...],
) -> Fraction:
    """The lottery's mean in exact rational arithmetic."""
    weights = [
        Fraction(*float(significand).as_integer_ratio()) / Fraction(1 << shift)
        for significand, shift in zip(significands, shifts, strict=True)
    ]
    numerator = sum(
        (
            weight * Fraction(*float(value).as_integer_ratio())
            for weight, value in zip(weights, values, strict=True)
        ),
        start=Fraction(0),
    )
    return numerator / sum(weights, start=Fraction(0))


def _rows() -> tuple[
    list[list[float]], list[list[int]], list[list[float]], list[float]
]:
    """Every two-node lottery in the class, with its exact mean."""
    dtype, shift_grid = _dtype_case()
    significands: list[list[float]] = []
    shifts: list[list[int]] = []
    values: list[list[float]] = []
    exact: list[float] = []

    for (
        rare_significand,
        rare_shift,
        rare_fraction,
        likely_fraction,
    ) in itertools.product(
        _SIGNIFICANDS, shift_grid, _VALUE_FRACTIONS, _VALUE_FRACTIONS
    ):
        row_significands = (1.0, rare_significand)
        row_shifts = (0, rare_shift)
        row_values = (_value(likely_fraction, dtype), _value(rare_fraction, dtype))
        mean = _exact_mean(
            significands=row_significands, shifts=row_shifts, values=row_values
        )
        # A row whose exact mean the format cannot hold is outside the contract:
        # the reduction is asked to represent the answer, not to invent range.
        if abs(float(mean)) > float(jnp.finfo(dtype).max):
            continue
        significands.append(list(row_significands))
        shifts.append(list(row_shifts))
        values.append(list(row_values))
        exact.append(float(mean))

    return significands, shifts, values, exact


@pytest.mark.parametrize(
    "mean",
    [
        LinearExpectation(),
        QuasiArithmeticMean(transform=lambda value: value, inverse=lambda value: value),
    ],
    ids=["linear", "identity_quasi"],
)
def test_every_admissible_scaled_lottery_reduces_to_its_exact_mean(mean) -> None:
    """The reduction agrees with exact rational arithmetic over the whole class.

    Every row is reduced in one batched call, so the comparison also pins that a
    row's scale stays its own: no row may be moved by the spread of the rows
    beside it.
    """
    dtype, _ = _dtype_case()
    significands, shifts, values, exact = _rows()

    got = jax.jit(mean.aggregate_scaled)(
        values=jnp.asarray(values, dtype=dtype),
        coefficients=jnp.asarray(significands, dtype=dtype),
        shifts=jnp.asarray(shifts, dtype=jnp.int32),
        params={},
    )

    reference = np.asarray(exact)
    np.testing.assert_allclose(
        np.asarray(got),
        reference,
        rtol=32.0 * float(jnp.finfo(dtype).eps),
        atol=float(jnp.finfo(dtype).tiny),
    )


def test_a_malformed_coefficient_poisons_its_own_row_only() -> None:
    """A negative or NaN weight is not a dead node and does not pass silently.

    It fails loudly in the row that carries it, and a well-formed row sharing
    the batch keeps its own answer — the poison travels with the lottery, not
    with the reduction.
    """
    dtype, _ = _dtype_case()
    coefficients = jnp.asarray([[1.0, 1.0], [1.0, -1.0], [1.0, jnp.nan]], dtype=dtype)
    shifts = jnp.zeros((3, 2), dtype=jnp.int32)
    values = jnp.asarray([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]], dtype=dtype)

    got = np.asarray(
        jax.jit(LinearExpectation().aggregate_scaled)(
            values=values, coefficients=coefficients, shifts=shifts, params={}
        )
    )

    np.testing.assert_allclose(got[0], 2.0, rtol=8.0 * float(jnp.finfo(dtype).eps))
    assert np.isnan(got[1])
    assert np.isnan(got[2])
