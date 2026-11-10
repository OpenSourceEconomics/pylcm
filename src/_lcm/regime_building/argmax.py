"""Argmax and max over the trailing action axes of a Q array.

`argmax_and_max` flattens the requested number of trailing axes, reduces over
the flat axis, and returns the winning flat index alongside its value.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.dtypes import float0
from jaxtyping import Int

from lcm.typing import BoolND, FloatND, IntND

# Encodings follow the float width, independently of canonical int32 action IDs.
type _FloatBitsND = Int[jax.Array, "..."]

# Identity of no element; it loses every tie against a real identity.
NO_ID = jnp.iinfo(jnp.int32).max


def argmax_and_max(
    *,
    a: FloatND,
    axis: int | tuple[int, ...] | None = None,
    initial: float | None = None,
    where: BoolND | None = None,
) -> tuple[IntND, FloatND]:
    """Compute the argmax of an n-dim array along axis.

    If multiple maxima exist, the first index will be selected.

    Args:
        a: Multidimensional array.
        axis: Axis along which to compute the argmax. If None, the argmax is computed
            over all axes.
        initial: The minimum value of an output element. Must be present to
            allow computation on empty slice. See ~numpy.ufunc.reduce for details.
        where: Elements to compare for the maximum. See ~numpy.ufunc.reduce
            for details.

    Returns:
        - The argmax indices. Array with the same shape as a, except for the dimensions
          specified in axis, which are dropped. The value corresponds to an index that
          can be translated into a tuple of indices using jnp.unravel_index.
        - The corresponding maximum values.

    """
    if axis is None:
        axis = tuple(range(a.ndim))
    elif isinstance(axis, int):
        axis = (axis,)

    if a.ndim == 0 or len(axis) == 0:
        return _argmax_and_max_of_one_candidate(a=a, where=where, initial=initial)

    a = _move_axes_to_back(a=a, axes=axis)
    a = _flatten_last_n_axes(a=a, n=len(axis))

    if where is not None and where.ndim != 0:
        where = _move_axes_to_back(a=where, axes=axis)
        where = _flatten_last_n_axes(a=where, n=len(axis))

    where = (
        jnp.ones(a.shape, dtype=bool)
        if where is None
        else jnp.broadcast_to(where, a.shape)
    )
    is_nan = jnp.isnan(a)
    comparable = where & ~is_nan
    lowest = -jnp.inf
    positions = jnp.broadcast_to(jnp.arange(a.shape[-1], dtype=jnp.int32), a.shape)
    _max, _argmax = max_and_smallest_id(
        values=jnp.where(comparable, a, lowest),
        ids=jnp.where(comparable, positions, NO_ID),
        initial=lowest if initial is None else initial,
    )
    # A NaN among the compared elements makes the max NaN and the index zero; an
    # `initial` above every element is the max and also reports index zero.
    any_nan = jnp.any(where & is_nan, axis=-1)
    _max = jnp.where(any_nan, jnp.full_like(_max, jnp.nan), _max)
    _argmax = jnp.where(any_nan | (_argmax == NO_ID), 0, _argmax)

    return _argmax, _max


def _argmax_and_max_of_one_candidate(
    *, a: FloatND, where: BoolND | None, initial: float | None
) -> tuple[IntND, FloatND]:
    """Return the index and value of each cell of `a` when nothing is reduced.

    Every cell has one candidate, its own value, at index zero. The value is the
    larger of the candidate and `initial` where `where` admits the candidate, and
    `initial` elsewhere; without `initial`, `-inf` stands in. A
    feasible NaN stays NaN, and the derivative splits a tie as `jnp.max` does. No
    reduction is involved, so a candidate above `initial` keeps its exact bits.
    """
    floor = jnp.asarray(-jnp.inf if initial is None else initial, dtype=a.dtype)
    admitted = (
        jnp.ones(a.shape, dtype=bool)
        if where is None
        else jnp.broadcast_to(where, a.shape)
    )
    return (
        jnp.zeros(a.shape, dtype=jnp.int32),
        jnp.where(admitted, jnp.maximum(a, floor), floor),
    )


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

    The max differentiates like `jnp.max` with the same `initial`:
    - no tie with `initial`: the average of the tangents of the elements equal
      to the max;
    - `initial` equal to the max: half that average plus half the tangent of
      `initial`;
    - `initial` above every element: the tangent of `initial`.

    The id carries no derivative.
    """
    initial_arr = jnp.asarray(initial, dtype=values.dtype)
    return _paired_max_with_tangent(values, ids, initial_arr)


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


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _paired_max_jvp(
    primals: tuple[FloatND, IntND, FloatND],
    tangents: tuple[FloatND, jax.Array | np.ndarray, FloatND],
) -> tuple[tuple[FloatND, IntND], tuple[FloatND, np.ndarray]]:
    """Average the tangents of the elements equal to the max, as `jnp.max` does.

    One protected key array owns both the primal reduction and its tie mask.
    Signed-zero keys compare equal for identity and tangent ownership.
    """
    values, ids, initial = primals
    values_dot, _, initial_dot = tangents
    value_keys = _float_order_key(values)
    initial_key = _float_order_key(initial)
    infinity_key = _float_infinity_key(values.dtype)
    best_key, best_id = _reduce_float_keys(
        keys=value_keys,
        ids=ids,
        initial_key=initial_key,
        infinity_key=infinity_key,
    )
    best = jax.lax.bitcast_convert_type(_reverse_negative_bits(best_key), values.dtype)
    attains = (
        _float_keys_equal(left=value_keys, right=best_key[..., jnp.newaxis])
        & (value_keys >= ~infinity_key)
        & (value_keys <= infinity_key)
    ).astype(values.dtype)
    count = jnp.sum(attains, axis=-1)
    elements_dot = jnp.sum(values_dot * attains, axis=-1) / jnp.maximum(count, 1)
    initial_dot = jnp.broadcast_to(initial_dot, best.shape)
    best_dot = jnp.where(
        count > 0,
        jnp.where(
            _float_keys_equal(left=initial_key, right=best_key),
            (elements_dot + initial_dot) / 2,
            elements_dot,
        ),
        initial_dot,
    )
    return (best, best_id), (best_dot, np.zeros(best_id.shape, dtype=float0))


# Built by call rather than by decorator: `@jax.custom_jvp` produces a callable
# instance, which the package claw rebinds to a bound method of its `__call__`,
# losing `defjvp` along with everything else the object knows. The generic
# differentiation of a two-operand `lax.reduce` cannot carry the integer id's
# tangent, so the pair needs its own rule.
_paired_max_with_tangent = jax.custom_jvp(_paired_max)
# JAX types a JVP rule's tangents as its primals; an integer's `float0` tangent
# is no `IntND`.
_paired_max_with_tangent.defjvp(_paired_max_jvp)  # ty: ignore[invalid-argument-type]


def _float_order_key(values: FloatND) -> _FloatBitsND:
    """Protect reversible integer keys once, before any reduction or tie comparison.

    Keys order represented IEEE values, with `-0` at -1 and `+0` at 0.
    Decoding the winning key preserves subnormal bits without a float accumulator.
    """
    if values.dtype.itemsize not in (2, 4, 8):
        msg = f"Use 16-, 32- and 64-bit IEEE floats; got {values.dtype}."
        raise ValueError(msg)
    integer = jnp.dtype(f"int{values.dtype.itemsize * 8}")
    bits = jax.lax.bitcast_convert_type(values, integer)
    return jax.lax.optimization_barrier(_reverse_negative_bits(bits))


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
    """Reduce protected keys once, then let a finite initial value compete."""
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
