"""Argmax and max over the trailing action axes of a Q array.

`argmax_and_max` flattens the requested number of trailing axes, reduces over
the flat axis, and returns the winning flat index alongside its value.
"""

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
    a: FloatND | IntND,
    axis: int | tuple[int, ...] | None = None,
    initial: float | None = None,
    where: BoolND | None = None,
) -> tuple[IntND, FloatND | IntND]:
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
    lowest = (
        -jnp.inf if jnp.issubdtype(a.dtype, jnp.floating) else jnp.iinfo(a.dtype).min
    )
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
    *, a: FloatND | IntND, where: BoolND | None, initial: float | None
) -> tuple[IntND, FloatND | IntND]:
    """Return the index and value of each cell of `a` when nothing is reduced.

    Every cell has one candidate, its own value, at index zero. The value is the
    larger of the candidate and `initial` where `where` admits the candidate, and
    `initial` elsewhere; without `initial` the dtype's lowest value stands in. A
    feasible NaN stays NaN, and the derivative splits a tie as `jnp.max` does. No
    reduction is involved, so a candidate above `initial` keeps its exact bits.
    """
    lowest = (
        -jnp.inf if jnp.issubdtype(a.dtype, jnp.floating) else jnp.iinfo(a.dtype).min
    )
    floor = jnp.asarray(lowest if initial is None else initial, dtype=a.dtype)
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
    *, values: FloatND | IntND, ids: IntND, initial: float
) -> tuple[FloatND | IntND, IntND]:
    """Return the max over the last axis and the smallest `int32` id attaining it.

    Both come out of one reduction over `(value, id)` pairs, so the id always
    names an element whose value is the returned max. Matching the values
    against a separately reduced max would be unsafe: the compiler may evaluate
    the values once per reduction, and evaluations that round differently leave
    no value equal to the max. When `initial` exceeds every value the id is
    `NO_ID`; mask an element out by giving it the lowest value and `NO_ID`.

    A floating max differentiates like `jnp.max` with the same `initial`:
    - no tie with `initial`: the average of the tangents of the elements equal
      to the max;
    - `initial` equal to the max: half that average plus half the tangent of
      `initial`;
    - `initial` above every element: the tangent of `initial`.

    The id carries no derivative.
    """
    initial_arr = jnp.asarray(initial, dtype=values.dtype)
    if jnp.issubdtype(values.dtype, jnp.floating):
        return _paired_max_with_tangent(values, ids, initial_arr)
    return _paired_max(values, ids, initial_arr)


# keyword-only-exempt: library-callback=jax.custom_jvp
def _paired_max(
    values: FloatND | IntND, ids: IntND, initial: FloatND | IntND
) -> tuple[FloatND | IntND, IntND]:
    """Reduce `(value, id)` pairs over the last axis, without a derivative."""
    lowest = (
        -jnp.inf
        if jnp.issubdtype(values.dtype, jnp.floating)
        else jnp.iinfo(values.dtype).min
    )
    no_id = jnp.asarray(NO_ID, dtype=jnp.int32)
    # XLA requires an identity initializer; a finite floor must compete afterward.
    best = jax.lax.reduce(
        (values, ids),
        (jnp.asarray(lowest, dtype=values.dtype), no_id),
        _larger_value_then_smaller_id,
        (values.ndim - 1,),
    )
    return _larger_value_then_smaller_id(best, (initial, no_id))


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _paired_max_jvp(
    primals: tuple[FloatND, IntND, FloatND],
    tangents: tuple[FloatND, jax.Array | np.ndarray, FloatND],
) -> tuple[tuple[FloatND, IntND], tuple[FloatND, np.ndarray]]:
    """Average the tangents of the elements equal to the max, as `jnp.max` does.

    The id's tangent, in and out, is JAX's `float0` zero tangent of an integer.

    The tangent is linear in the value and `initial` tangents with weights
    fixed by the primals, so reverse mode transposes it. `{-0, +0}` compare
    equal, so a signed-zero tie averages both elements whatever sign the max
    carries. The max and the equality test read one materialized copy of the
    values: a producer the compiler evaluated once per use could round
    differently and leave no element equal to the max.
    """
    values, ids, initial = primals
    values = jax.lax.optimization_barrier(values)
    values_dot, _, initial_dot = tangents
    best, best_id = _paired_max(values, ids, initial)
    _, value_keys = _float_order_parts(values)
    _, best_keys = _float_order_parts(best)
    _, initial_key = _float_order_parts(initial)
    attains = ((value_keys == best_keys[..., jnp.newaxis]) & ~jnp.isnan(values)).astype(
        values.dtype
    )
    count = jnp.sum(attains, axis=-1)
    elements_dot = jnp.sum(values_dot * attains, axis=-1) / jnp.maximum(count, 1)
    initial_dot = jnp.broadcast_to(initial_dot, best.shape)
    best_dot = jnp.where(
        count > 0,
        jnp.where(
            initial_key == best_keys, (elements_dot + initial_dot) / 2, elements_dot
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


# keyword-only-exempt: library-callback=jax.lax.reduce
def _larger_value_then_smaller_id(
    left: tuple[FloatND | IntND, IntND], right: tuple[FloatND | IntND, IntND]
) -> tuple[FloatND | IntND, IntND]:
    """Keep the pair with the larger value, and the smaller id on a tie.

    A `{-0, +0}` tie keeps `+0` unless both are `-0`.
    """
    left_value, left_id = left
    right_value, right_id = right
    if jnp.issubdtype(left_value.dtype, jnp.floating):
        left_bits, left_key = _float_order_parts(left_value)
        right_bits, right_key = _float_order_parts(right_value)
        valid = ~jnp.isnan(left_value) & ~jnp.isnan(right_value)
        tie = (right_key == left_key) & valid
        take_right = ((right_key > left_key) & valid) | (tie & (right_id < left_id))
        bits = jnp.where(take_right, right_bits, left_bits)
        both_zero = (left_key == 0) & (right_key == 0)
        bits = jnp.where(both_zero, left_bits & right_bits, bits)
        value = jax.lax.bitcast_convert_type(bits, left_value.dtype)
    else:
        tie = right_value == left_value
        take_right = (right_value > left_value) | (tie & (right_id < left_id))
        value = jnp.where(take_right, right_value, left_value)
    return value, jnp.where(take_right, right_id, left_id)


def _float_order_parts(values: FloatND) -> tuple[_FloatBitsND, _FloatBitsND]:
    """Return stored bits and an integer order key, equating signed zeros.

    Negative encodings reverse magnitude order. Barriers prevent the compiler
    from replacing this stored-value order with flushing float comparisons.
    """
    integer = jnp.int32 if values.dtype == jnp.float32 else jnp.int64
    bits = jax.lax.optimization_barrier(jax.lax.bitcast_convert_type(values, integer))
    magnitude_mask = jnp.iinfo(integer).max
    key = bits ^ ((bits >> (values.dtype.itemsize * 8 - 1)) & magnitude_mask)
    key = jnp.where((bits & magnitude_mask) == 0, jnp.zeros_like(bits), key)
    return bits, jax.lax.optimization_barrier(key)


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
