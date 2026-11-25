"""Maxima paired with the identity of the element attaining them.

`max_and_smallest_id` and `max_lexicographic_and_smallest_id` reduce values
over their trailing axis and return the smallest identity attaining the maximum.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Int

from _lcm.probability import without_derivative
from lcm.typing import BoolND, FloatND, IntND

# Encodings follow the float width, independently of canonical int32 action IDs.
type _FloatBitsND = Int[jax.Array, "..."]

# Identity of no element; it loses every tie against a real identity.
NO_ID = jnp.iinfo(jnp.int32).max


def max_and_smallest_id(
    *, values: FloatND, ids: IntND, initial: float
) -> tuple[FloatND, IntND]:
    """Return the max over the last axis and the smallest `int32` id attaining it.

    Both come out of one reduction over `(value, id)` pairs, so the id always
    names an element whose value is the returned max. Matching the values
    against a separately reduced max would be unsafe: the compiler may evaluate
    the values once per reduction, and evaluations that round differently leave
    no value equal to the max. When `initial` exceeds every value the id is
    `NO_ID`; mask an element out by giving it the lowest value and `NO_ID`.

    Differentiating through the reduction raises
    `UnsupportedDifferentiationError`.
    """
    initial_arr = jnp.asarray(initial, dtype=values.dtype)
    return _paired_max_no_derivative(values, ids, initial_arr)


# keyword-only-exempt: library-callback=jax.custom_jvp
def _paired_max(values: FloatND, ids: IntND, initial: FloatND) -> tuple[FloatND, IntND]:
    """Reduce `(value, id)` pairs over the last axis, without a derivative."""
    best_key, best_id = _reduce_float_keys(
        keys=_float_order_key(values),
        ids=ids,
        initial_key=_float_order_key(initial),
        infinity_key=_float_infinity_key(values.dtype),
    )
    return jax.lax.bitcast_convert_type(
        _reverse_negative_bits(best_key), values.dtype
    ), best_id


_paired_max_no_derivative = without_derivative(_paired_max)


def _float_order_key(values: FloatND) -> _FloatBitsND:
    """Encode floats as reversible integer keys for the paired reduction.

    Keys order represented IEEE values, with `-0` at -1 and `+0` at 0.
    Decoding the winning key preserves subnormal bits without a float accumulator.

    The paired reduction over the last axis is the keys' only reader. Over more
    than one element the compiler computes the keys inside that reduction, so one
    evaluation of the values both picks the winner and becomes the published
    maximum. Over a single element the compiler replaces the reduction by separate
    elementwise programs for the maximum and for the identity, each of which would
    evaluate the values' producer on its own; an optimization barrier then holds
    the keys in one array that both programs read.
    """
    if values.dtype.itemsize not in (2, 4, 8):
        msg = f"Use 16-, 32- and 64-bit IEEE floats; got {values.dtype}."
        raise ValueError(msg)
    integer = jnp.dtype(f"int{values.dtype.itemsize * 8}")
    bits = jax.lax.bitcast_convert_type(values, integer)
    if values.shape[-1:] == (1,):
        return jax.lax.optimization_barrier(_reverse_negative_bits(bits))
    return _reverse_negative_bits(bits)


def _reverse_negative_bits(bits: _FloatBitsND) -> _FloatBitsND:
    """Reverse negative IEEE encodings; applying this twice restores the input."""
    magnitude_mask = jnp.iinfo(bits.dtype).max
    return bits ^ ((bits >> (bits.dtype.itemsize * 8 - 1)) & magnitude_mask)


def _float_infinity_key(dtype: np.dtype[np.generic]) -> int:
    """Return the positive IEEE infinity encoding without allocating float data."""
    info = jnp.finfo(dtype)
    return (2 * info.maxexp - 1) << info.nmant


def _float_keys_equal(*, left: _FloatBitsND, right: _FloatBitsND) -> BoolND:
    """Equate represented values, including the two signed-zero keys."""
    both_zero = (left >= -1) & (left <= 0) & (right >= -1) & (right <= 0)
    return (left == right) | both_zero


def _reduce_float_keys(
    *, keys: _FloatBitsND, ids: IntND, initial_key: _FloatBitsND, infinity_key: int
) -> tuple[_FloatBitsND, IntND]:
    """Reduce the keys once, then let a finite initial value compete."""
    no_id = jnp.asarray(NO_ID, dtype=jnp.int32)
    choose = partial(_larger_float_key_then_smaller_id, infinity_key=infinity_key)
    best = jax.lax.reduce(
        (keys, ids),
        (jnp.asarray(~infinity_key, dtype=keys.dtype), no_id),
        choose,
        (keys.ndim - 1,),
    )
    return choose(best, (initial_key, no_id))


# keyword-only-exempt: library-callback=jax.lax.reduce
def _larger_float_key_then_smaller_id(
    left: tuple[_FloatBitsND, IntND],
    right: tuple[_FloatBitsND, IntND],
    *,
    infinity_key: int,
) -> tuple[_FloatBitsND, IntND]:
    """Compare precomputed keys, retaining the smallest id even at a zero tie."""
    left_key, left_id = left
    right_key, right_id = right
    valid = (
        (left_key >= ~infinity_key)
        & (left_key <= infinity_key)
        & (right_key >= ~infinity_key)
        & (right_key <= infinity_key)
    )
    tie = _float_keys_equal(left=left_key, right=right_key)
    key = jnp.where(valid, jnp.maximum(left_key, right_key), left_key)
    identity = jnp.where(
        tie,
        jnp.minimum(left_id, right_id),
        jnp.where(right_key > left_key, right_id, left_id),
    )
    return key, jnp.where(valid, identity, left_id)


def max_lexicographic_and_smallest_id(
    *, values: tuple[FloatND, ...], ids: IntND, where: BoolND
) -> tuple[tuple[FloatND, ...], IntND]:
    """Return the lexicographic maximum over the last axis and its smallest id.

    Elements are ranked on `values[0]`; ties go to the larger `values[1]`, then
    `values[2]`, and so on; full ties go to the smaller `int32` id. The order is
    total, so the winner does not depend on how the axis is split or merged.
    Within each position a NaN ranks below every number, and `-0` and `+0` are
    equal. An element outside `where` ranks below every element inside it.

    One reduction carries the order keys of every position beside the id, so the
    winner's values are decoded from their keys rather than gathered. They are
    the winner's own bits, except that a NaN comes back as some NaN. Where no
    element is inside `where`, every value is NaN and the id is `NO_ID`.

    Differentiating through the reduction raises
    `UnsupportedDifferentiationError`.
    """
    return _lexicographic_max_no_derivative(values, ids, where)


# keyword-only-exempt: library-callback=jax.custom_jvp
def _lexicographic_max(
    values: tuple[FloatND, ...], ids: IntND, where: BoolND
) -> tuple[tuple[FloatND, ...], IntND]:
    """Reduce `(key, ..., key, id)` tuples over the last axis, without a derivative."""
    no_id = jnp.asarray(NO_ID, dtype=jnp.int32)
    keys = tuple(_ranked_key(values=value, where=where) for value in values)
    *best_keys, best_id = jax.lax.reduce(
        (*keys, jnp.where(where, ids, no_id)),
        (*(_lowest_key(key.dtype) for key in keys), no_id),
        _lexicographically_larger,
        (ids.ndim - 1,),
    )
    return tuple(
        jax.lax.bitcast_convert_type(_reverse_negative_bits(key), value.dtype)
        for key, value in zip(best_keys, values, strict=True)
    ), best_id


_lexicographic_max_no_derivative = without_derivative(_lexicographic_max)


def _ranked_key(*, values: FloatND, where: BoolND) -> _FloatBitsND:
    """Order key of each value; NaN and elements outside `where` take the lowest."""
    keys = _float_order_key(values)
    infinity_key = _float_infinity_key(values.dtype)
    is_number = (keys >= ~infinity_key) & (keys <= infinity_key)
    return jnp.where(where & is_number, keys, _lowest_key(keys.dtype))


def _lowest_key(dtype: np.dtype[np.generic]) -> _FloatBitsND:
    """Return the key just below `-inf`'s; it decodes to a NaN."""
    infinity_key = _float_infinity_key(jnp.dtype(f"float{dtype.itemsize * 8}"))
    return jnp.asarray(~infinity_key - 1, dtype=dtype)


# keyword-only-exempt: library-callback=jax.lax.reduce
def _lexicographically_larger(
    left: tuple[_FloatBitsND | IntND, ...], right: tuple[_FloatBitsND | IntND, ...]
) -> tuple[_FloatBitsND | IntND, ...]:
    """Keep the tuple whose first unequal key is larger, else the smaller id."""
    right_wins = right[-1] < left[-1]
    for left_key, right_key in zip(left[-2::-1], right[-2::-1], strict=True):
        right_wins = jnp.where(
            _float_keys_equal(left=left_key, right=right_key),
            right_wins,
            right_key > left_key,
        )
    return tuple(
        jnp.where(right_wins, right_item, left_item)
        for left_item, right_item in zip(left, right, strict=True)
    )


def _move_axes_to_back(
    *, a: FloatND | IntND | BoolND, axes: tuple[int, ...]
) -> FloatND | IntND | BoolND:
    """Move specified axes to the back of the array.

    Args:
        a: Multidimensional jax array.
        axes: Axes to move to the back.

    Returns:
        Array a with shifted axes.

    """
    front_axes = sorted(set(range(a.ndim)) - set(axes))
    return a.transpose((*front_axes, *axes))


def _flatten_last_n_axes(
    *, a: FloatND | IntND | BoolND, n: int
) -> FloatND | IntND | BoolND:
    """Flatten the last n axes of a to 1 dimension.

    Args:
        a: Multidimensional jax array.
        n: Number of axes to flatten.

    Returns:
        Array a with flattened last n axes.

    """
    return a.reshape(*a.shape[:-n], -1)
