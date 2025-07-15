from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from lcm.typing import IntND

# ======================================================================================
# argmax
# ======================================================================================


def argmax_and_max(
    a: Array,
    axis: int | tuple[int, ...] | None = None,
    initial: float | None = None,
    where: Array | None = None,
) -> tuple[IntND, Array]:
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
    # Preparation
    # ==================================================================================
    if axis is None:
        axis = tuple(range(a.ndim))
    elif isinstance(axis, int):
        axis = (axis,)

    # Move axis over which to compute the argmax to the back and flatten last dims
    # ==================================================================================
    a = _move_axes_to_back(a, axes=axis)
    a = _flatten_last_n_axes(a, n=len(axis))

    # Do same transformation for where
    # ==================================================================================
    if where is not None:
        where = _move_axes_to_back(where, axes=axis)
        where = _flatten_last_n_axes(where, n=len(axis))

    # Compute argmax over last dimension
    # ----------------------------------------------------------------------------------
    # Note: If multiple maxima exist, this approach will select the first index.
    # ==================================================================================
    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)
    max_value_mask = a == _max
    if where is not None:
        max_value_mask = jnp.logical_and(max_value_mask, where)
    _argmax = jnp.argmax(max_value_mask, axis=-1)

    return _argmax, _max.reshape(_argmax.shape)


def _move_axes_to_back(a: Array, axes: tuple[int, ...]) -> Array:
    """Move specified axes to the back of the array.

    Args:
        a: Multidimensional jax array.
        axes: Axes to move to the back.

    Returns:
        Array a with shifted axes.

    """
    front_axes = sorted(set(range(a.ndim)) - set(axes))
    return a.transpose((*front_axes, *axes))


def _flatten_last_n_axes(a: Array, n: int) -> Array:
    """Flatten the last n axes of a to 1 dimension.

    Args:
        a: Multidimensional jax array.
        n: Number of axes to flatten.

    Returns:
        Array a with flattened last n axes.

    """
    return a.reshape(*a.shape[:-n], -1)
