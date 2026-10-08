"""Argmax and max over the trailing action axes of a Q array.

`argmax_and_max` flattens the requested number of trailing axes, reduces over
the flat axis, and returns the winning flat index alongside its value.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.dtypes import float0

from lcm.typing import BoolND, FloatND, IntND

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

    # Handle scalar or empty axis case (no actions to maximize over)
    if a.ndim == 0 or len(axis) == 0:
        # When there are no dimensions to reduce over, return:
        # - index 0 (trivial argmax since there's only one element)
        # - the array itself (already the maximum)
        return jnp.array(0, dtype=jnp.int32), a

    if a.ndim != 0:
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

    A floating max differentiates like `jnp.max`: its tangent is the average of
    the tangents of the elements equal to it, and zero when `initial` exceeds
    every element. The id carries no derivative.
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
    return jax.lax.reduce(
        (values, ids),
        (initial, jnp.asarray(NO_ID, dtype=jnp.int32)),
        _larger_value_then_smaller_id,
        (values.ndim - 1,),
    )


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _paired_max_jvp(
    primals: tuple[FloatND, IntND, FloatND],
    tangents: tuple[FloatND, jax.Array | np.ndarray, FloatND],
) -> tuple[tuple[FloatND, IntND], tuple[FloatND, np.ndarray]]:
    """Average the tangents of the elements equal to the max.

    The id's tangent, in and out, is JAX's `float0` zero tangent of an integer.

    The tangent is linear in the value tangent with weights fixed by the
    primals, so reverse mode transposes it. `{-0, +0}` compare equal, so a
    signed-zero tie averages both elements whatever sign the max carries. The
    max and the equality test read one materialized copy of the values: a
    producer the compiler evaluated once per use could round differently and
    leave no element equal to the max.
    """
    values, ids, initial = primals
    values = jax.lax.optimization_barrier(values)
    values_dot = tangents[0]
    best, best_id = _paired_max(values, ids, initial)
    attains = (values == best[..., jnp.newaxis]).astype(values.dtype)
    count = jnp.sum(attains, axis=-1)
    best_dot = jnp.where(
        count > 0,
        jnp.sum(values_dot * attains, axis=-1) / jnp.maximum(count, 1),
        jnp.zeros_like(best),
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
    tie = right_value == left_value
    take_right = (right_value > left_value) | (tie & (right_id < left_id))
    value = jnp.where(take_right, right_value, left_value)
    value = jnp.where(tie & (left_value == 0), left_value + right_value, value)
    return value, jnp.where(take_right, right_id, left_id)


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
