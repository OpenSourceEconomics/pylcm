"""Every consumer reads a probability the same way, from its bits.

A probability below the dtype's normal range is a strictly positive number
that arithmetic reports as zero: `w == 0` is true of it, `w > 0` is false, and
a negative one compares as `-0`. Answering "is this the null event", "is this
a valid probability" and "how large is it" with comparisons therefore gives
one answer in the module that reads bits and another everywhere else, and the
difference is not a rounding difference — it decides whether an infinity
survives, whether a malformed transition is refused, and whether a rare node
contributes at all.

The rescaling is the other half of the contract: a weighted mean depends on
its weights only through their ratios, so lifting a whole lottery by one power
of two changes no answer and leaves every downstream multiplication with
operands the dtype can use.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.probability import (
    is_below_smallest_normal,
    is_live,
    is_negative,
    is_represented_zero,
    rescaled_lottery_weights,
    rescaled_weight_group,
    rescaled_weight_pair,
)


def _dtype() -> np.dtype:
    """The precision the suite is running at."""
    return np.dtype(jnp.zeros(()).dtype)


def _smallest_subnormal() -> Any:
    dtype = _dtype()
    return np.nextafter(dtype.type(0.0), dtype.type(1.0), dtype=dtype)


def _largest_subnormal() -> Any:
    dtype = _dtype()
    return np.nextafter(dtype.type(np.finfo(dtype).tiny), dtype.type(0.0), dtype=dtype)


def _interior_subnormal() -> Any:
    """A subnormal from the middle of the range, not either end of it."""
    dtype = _dtype()
    mantissa_bits = 23 if dtype.itemsize == 4 else 52
    int_dtype = np.int32 if dtype.itemsize == 4 else np.int64
    return np.asarray(1 << (mantissa_bits // 2), dtype=int_dtype).view(dtype)[()]


_SUBNORMALS = [
    pytest.param(_smallest_subnormal, id="smallest"),
    pytest.param(_interior_subnormal, id="interior"),
    pytest.param(_largest_subnormal, id="largest"),
]


@pytest.mark.parametrize("magnitude", _SUBNORMALS)
def test_a_positive_subnormal_is_live(magnitude) -> None:
    """A probability the dtype cannot multiply is still an event that occurs."""
    weight = jnp.asarray(magnitude(), dtype=_dtype())

    assert bool(is_live(weight))
    assert not bool(is_represented_zero(weight))
    assert bool(is_below_smallest_normal(weight))


@pytest.mark.parametrize("magnitude", _SUBNORMALS)
def test_a_negative_subnormal_is_negative_and_not_live(magnitude) -> None:
    """Negative mass is a misspecification, at any size the format can hold."""
    weight = jnp.asarray(np.negative(magnitude()), dtype=_dtype())

    assert bool(is_negative(weight))
    assert not bool(is_live(weight))
    assert not bool(is_represented_zero(weight))


@pytest.mark.parametrize("signed_zero", [0.0, -0.0], ids=["positive", "negative"])
def test_a_signed_zero_is_the_null_event(signed_zero: float) -> None:
    """Both zeros are the event that cannot occur, and neither is negative."""
    weight = jnp.asarray(signed_zero, dtype=_dtype())

    assert bool(is_represented_zero(weight))
    assert not bool(is_live(weight))
    assert not bool(is_negative(weight))


@pytest.mark.parametrize(
    ("value", "live"),
    [(1.0, True), (0.5, True), (np.inf, True), (np.nan, False), (-1.0, False)],
    ids=["one", "half", "infinite", "nan", "negative"],
)
def test_liveness_of_an_ordinary_weight(value: float, *, live: bool) -> None:
    """The bit-level test agrees with the arithmetic one wherever both are valid."""
    assert bool(is_live(jnp.asarray(value, dtype=_dtype()))) is live


@pytest.mark.parametrize("magnitude", _SUBNORMALS)
@pytest.mark.parametrize("compile_it", [False, True], ids=["eager", "jit"])
def test_rescaling_preserves_every_ratio_exactly(
    magnitude, *, compile_it: bool
) -> None:
    """The rescaled lottery has the ratios it was given, bit for bit."""
    dtype = _dtype()
    rare = magnitude()
    weights = jnp.asarray([1.0, rare], dtype=dtype)
    rescale = (
        jax.jit(rescaled_lottery_weights) if compile_it else (rescaled_lottery_weights)
    )

    rescaled = np.asarray(rescale(weights))

    exact_ratio = np.longdouble(rare) / np.longdouble(1.0)
    assert np.longdouble(rescaled[1]) / np.longdouble(rescaled[0]) == exact_ratio


@pytest.mark.parametrize("magnitude", _SUBNORMALS)
def test_rescaling_lifts_every_live_weight_into_the_normal_range(magnitude) -> None:
    """No live weight is left where the arithmetic would flush it."""
    weights = jnp.asarray([1.0, magnitude()], dtype=_dtype())

    rescaled = rescaled_lottery_weights(weights)

    assert not bool(jnp.any(is_below_smallest_normal(rescaled) & is_live(rescaled)))


def test_rescaling_lifts_a_lottery_that_is_subnormal_throughout() -> None:
    """The factor is the largest each entry needs, so the widest spread survives."""
    weights = jnp.asarray([_smallest_subnormal(), _largest_subnormal()], dtype=_dtype())

    rescaled = rescaled_lottery_weights(weights)

    assert bool(jnp.all(is_live(rescaled)))
    assert not bool(jnp.any(is_below_smallest_normal(rescaled)))


def test_rescaling_leaves_the_null_event_null() -> None:
    """A zero weight is not lifted into something that can occur."""
    weights = jnp.asarray([1.0, 0.0, _smallest_subnormal()], dtype=_dtype())

    rescaled = rescaled_lottery_weights(weights)

    assert bool(is_represented_zero(rescaled[1]))


@pytest.mark.parametrize(
    "unusable", [np.nan, np.inf, -np.inf], ids=["nan", "positive", "negative"]
)
def test_rescaling_leaves_a_non_finite_weight_alone(unusable: float) -> None:
    """Neither a NaN nor an infinity has a scale to change."""
    weights = jnp.asarray([1.0, unusable], dtype=_dtype())

    rescaled = np.asarray(rescaled_lottery_weights(weights))

    assert np.isnan(rescaled[1]) if np.isnan(unusable) else rescaled[1] == unusable


def test_rescaling_keeps_a_negative_weight_negative() -> None:
    """The sign survives the rescaling, so a malformed weight stays visible."""
    weights = jnp.asarray([1.0, np.negative(_smallest_subnormal())], dtype=_dtype())

    rescaled = rescaled_lottery_weights(weights)

    assert bool(is_negative(rescaled[1]))


def test_rescaling_a_lottery_of_normal_weights_changes_nothing() -> None:
    """With nothing to lift, the weights come back as they went in."""
    weights = jnp.asarray([0.25, 0.5, 0.25], dtype=_dtype())

    np.testing.assert_array_equal(
        np.asarray(rescaled_lottery_weights(weights)), np.asarray(weights)
    )


def test_a_pair_is_rescaled_by_one_common_factor() -> None:
    """The two-node form gives the ratio the general form gives."""
    dtype = _dtype()
    first, second = rescaled_weight_pair(
        jnp.asarray(1.0, dtype=dtype), jnp.asarray(_smallest_subnormal(), dtype=dtype)
    )

    assert np.longdouble(np.asarray(second)) / np.longdouble(
        np.asarray(first)
    ) == np.longdouble(_smallest_subnormal())


def test_a_group_of_branches_is_rescaled_by_one_common_factor() -> None:
    """Branches held one array per target share the factor their lottery needs."""
    dtype = _dtype()
    scaled = rescaled_weight_group(
        [
            jnp.asarray(1.0, dtype=dtype),
            jnp.asarray(_smallest_subnormal(), dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
        ]
    )

    assert np.longdouble(np.asarray(scaled[1])) / np.longdouble(
        np.asarray(scaled[0])
    ) == np.longdouble(_smallest_subnormal())
    assert bool(is_represented_zero(scaled[2]))
