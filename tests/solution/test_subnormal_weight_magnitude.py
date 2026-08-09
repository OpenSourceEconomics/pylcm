"""A weight too small to use is never enlarged into one that carries real mass.

Subnormal probabilities span a wide range: the smallest is `2**23` times below the
largest in single precision and `2**52` times below it in double. Treating them
alike — by promoting every one of them to the smallest normal magnitude — hands
the tiniest probabilities the mass of the largest, and against a big continuation
that invents a contribution the model never specified.

A weight that arrives representable is never enlarged: it may contribute *less*
than its true share, down to nothing at all, and the omission is at most
`tiny * |V|` — below every declared tolerance for any value function a model can
also add utility to.

A joint product that underflows is the one case that can err upward, because the
format has nothing smaller than its smallest magnitude to stand in with. That
overstatement is bounded by `smallest_subnormal * |V|`, tighter than the omission
bound by `2**23` in single precision and `2**52` in double.

The one place magnitude stops mattering is an infinite continuation, where any
strictly positive weight yields that infinity. Only there is such a weight raised.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.power_mean import weighted_power_mean
from _lcm.zero_safe import joint_weight, zero_safe_weighted_term
from lcm.typing import FloatND, ScalarFloat


def _active_dtype() -> np.dtype:
    """The precision the suite is running at."""
    return np.dtype(jnp.zeros(()).dtype)


def _tolerance() -> float:
    """The declared profile tolerance at the active precision."""
    return 1e-12 if _active_dtype() == np.float64 else 1e-5


def _smallest_subnormal() -> ScalarFloat:
    """The smallest positive value the dtype can represent at all."""
    dtype = _active_dtype()
    return jnp.asarray(np.nextafter(dtype.type(0.0), dtype.type(1.0)), dtype=dtype)


def _largest_subnormal() -> ScalarFloat:
    """The largest probability the dtype cannot hold as a normal number."""
    dtype = _active_dtype()
    tiny = np.asarray(np.finfo(dtype).tiny, dtype=dtype)
    return jnp.asarray(np.nextafter(tiny, dtype.type(0.0)), dtype=dtype)


def _smallest_normal() -> ScalarFloat:
    """The smallest probability the dtype holds as a normal number."""
    return jnp.asarray(np.finfo(_active_dtype()).tiny, dtype=_active_dtype())


def _subnormal_at(fraction_of_the_range: float) -> ScalarFloat:
    """A subnormal a given fraction of the way up the subnormal range."""
    dtype = _active_dtype()
    smallest = float(_smallest_subnormal())
    return jnp.asarray(
        smallest * 2 ** (fraction_of_the_range * _mantissa_bits()), dtype=dtype
    )


def _mantissa_bits() -> int:
    """Width of the subnormal range in powers of two."""
    return 52 if _active_dtype() == np.float64 else 23


def _largest_finite() -> ScalarFloat:
    """The largest finite value of the dtype — where an inflated weight shows most."""
    return jnp.asarray(np.finfo(_active_dtype()).max, dtype=_active_dtype())


def _lottery_value(weight: ScalarFloat, rare_value: ScalarFloat) -> float:
    """A two-target continuation: unit mass on one, `weight` on `rare_value`."""
    common = zero_safe_weighted_term(
        weight=jnp.asarray(1.0, dtype=_active_dtype()),
        value=jnp.asarray(1.0, dtype=_active_dtype()),
        subnormal_is_accounted_for=False,
    )
    rare = zero_safe_weighted_term(
        weight=weight, value=rare_value, subnormal_is_accounted_for=False
    )
    return float((common + rare) / (1.0 + float(weight)))


def _exact_lottery_value(weight: ScalarFloat, rare_value: ScalarFloat) -> float:
    """The same continuation in exact arithmetic."""
    w = float(weight)
    return (1.0 + w * float(rare_value)) / (1.0 + w)


@pytest.mark.parametrize(
    "weight_at",
    [0.0, 0.25, 0.5, 0.75],
    ids=["smallest", "quarter", "half", "three-quarters"],
)
def test_a_subnormal_weight_never_contributes_more_than_its_true_share(
    weight_at: float,
) -> None:
    """Across the subnormal range, a supplied weight never gains mass.

    A weight that arrives representable is passed through untouched, so the
    answer is either exact or short by the dropped node — never above it.
    """
    weight = _subnormal_at(weight_at)
    value = _largest_finite()

    got = _lottery_value(weight, value)
    exact = _exact_lottery_value(weight, value)

    assert got <= exact * (1.0 + _tolerance())


def test_the_smallest_subnormal_weight_does_not_invent_a_contribution() -> None:
    """A probability at the bottom of the range prices its target at ~nothing."""
    weight = _smallest_subnormal()
    value = _largest_finite()

    got = _lottery_value(weight, value)
    exact = _exact_lottery_value(weight, value)

    np.testing.assert_allclose(got, exact, rtol=_tolerance())


def test_a_negative_smallest_subnormal_does_not_subtract_real_mass() -> None:
    """An invalid negative weight stays negligible rather than becoming ~-4."""
    weight = -_smallest_subnormal()
    value = _largest_finite()

    assert abs(_lottery_value(weight, value) - 1.0) <= _tolerance()


def test_a_subnormal_weight_does_not_distort_a_nonlinear_aggregation() -> None:
    """A harmonic mean is not moved by a node whose probability underflowed.

    A harmonic mean divides by each value, so an enlarged weight on a small
    value moves the answer far more than its probability warrants. Both values
    here are ordinary, so the node is negligible on any backend.
    """
    dtype = _active_dtype()
    root = np.sqrt(float(_smallest_normal())) / 2.0
    underflowed = joint_weight(jnp.asarray([root, root], dtype=dtype))

    got = float(
        weighted_power_mean(
            values=jnp.asarray([1.0, 0.5], dtype=dtype),
            weights=jnp.stack([jnp.asarray(1.0, dtype=dtype), underflowed]),
            exponent=jnp.asarray(-1.0, dtype=dtype),
            shifts=jnp.zeros((), jnp.int32),
        )
    )

    np.testing.assert_allclose(got, 1.0, rtol=_tolerance())


def test_an_underflowed_joint_product_is_never_inflated() -> None:
    """A product that underflows stays below the normal range and below its truth.

    How far below is a backend fact — a flushing backend receives the smallest
    representable magnitude, one that represents subnormals keeps the product's
    own value — so what is asserted is the contract both satisfy.
    """
    dtype = _active_dtype()
    root = np.sqrt(float(_smallest_normal())) / 2.0
    factors = jnp.asarray([root, root], dtype=dtype)

    weight = float(joint_weight(factors))

    assert weight != 0.0
    assert weight <= root * root
    assert weight < float(_smallest_normal())


def test_an_underflowed_joint_product_contributes_nothing_to_a_finite_value() -> None:
    """The node exists, and prices an ordinary finite continuation at ~nothing.

    The continuation is ordinary on purpose. Against a value near the top of the
    dtype's range the answer is backend-visible — a flushing backend drops the
    node while one that represents subnormals prices it — and the claim here is
    the one that holds everywhere.
    """
    dtype = _active_dtype()
    root = np.sqrt(float(_smallest_normal())) / 2.0
    weight = joint_weight(jnp.asarray([root, root], dtype=dtype))

    term = float(
        zero_safe_weighted_term(
            weight=weight,
            value=jnp.asarray(1.0, dtype=dtype),
            subnormal_is_accounted_for=False,
        )
    )

    assert abs(term) <= _tolerance()


def test_an_underflowed_joint_product_still_keeps_an_infinity() -> None:
    """A node that can occur makes an infeasible continuation infinite."""
    dtype = _active_dtype()
    root = np.sqrt(float(_smallest_normal())) / 2.0
    weight = joint_weight(jnp.asarray([root, root], dtype=dtype))

    term = zero_safe_weighted_term(
        weight=weight,
        value=jnp.asarray(-jnp.inf, dtype=dtype),
        subnormal_is_accounted_for=False,
    )

    assert bool(jnp.isneginf(term))


@pytest.mark.parametrize("compile_it", [False, True], ids=["eager", "jit"])
def test_the_largest_subnormal_weight_still_keeps_an_infinity(
    *, compile_it: bool
) -> None:
    """The infinite case is unchanged: magnitude cannot matter against `-inf`."""
    dtype = _active_dtype()
    func: Callable[..., FloatND] = (
        jax.jit(zero_safe_weighted_term, static_argnames="subnormal_is_accounted_for")
        if compile_it
        else zero_safe_weighted_term
    )

    term = func(
        weight=_largest_subnormal(),
        value=jnp.asarray(-jnp.inf, dtype=dtype),
        subnormal_is_accounted_for=False,
    )

    assert bool(jnp.isneginf(term))
