"""Saturating int32 counts: a positive population never reduces to zero.

Landing diagnostics count cells of a state-action product that can exceed the
int32 range, and int64 is unavailable without x64. Every count reduction
therefore saturates at `COUNT_CAPACITY`:

- a total below the capacity is exact;
- a total at the capacity means "at least that many";
- a positive population never wraps to zero or to a negative number, so a
  mandatory refusal that tests a total for being nonzero cannot be lost.

Reductions use only `jnp.sum` and elementwise operations, so they partition
over sharded axes like any other sum.
"""

import math
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from lcm.exceptions import ExecutionPlanningError
from lcm.typing import BoolND, IntND

# The largest exact count; a saturated total reads as at least this many.
COUNT_CAPACITY = int(np.iinfo(np.int32).max)

# The value bits of a non-negative int32.
_INT32_BITS = 31


def add_counts(*, left: IntND, right: IntND) -> IntND:
    """Add non-negative int32 counts elementwise, saturating at the capacity."""
    left = jnp.asarray(left, dtype=jnp.int32)
    right = jnp.asarray(right, dtype=jnp.int32)
    return jnp.where(left > COUNT_CAPACITY - right, COUNT_CAPACITY, left + right)


def sum_counts(
    *,
    counts: IntND | BoolND,
    axis: int | Sequence[int] | None = None,
    largest: int | None = None,
) -> IntND:
    """Sum non-negative counts (or booleans) over `axis`, saturating at the capacity.

    A reduction over `n` elements whose largest possible total fits int32 is one
    exact `jnp.sum`. Otherwise each element is split into base-$2^k$ digits with
    $n (2^k - 1)$ within the capacity, so every digit sum is exact, and the digit
    sums are recombined with saturating additions.

    Args:
        counts: Non-negative integer counts, or booleans counted as zero or one.
        axis: The axes to reduce; `None` reduces all of them.
        largest: A static bound on every element that the caller guarantees;
            `None` means one for booleans and `COUNT_CAPACITY` otherwise.

    Returns:
        The int32 totals, exact below `COUNT_CAPACITY` and equal to it otherwise.

    Raises:
        ExecutionPlanningError: The reduction spans more elements than an int32
            count can address.

    """
    counts = jnp.asarray(counts)
    if largest is not None:
        largest_element = largest
    elif counts.dtype == jnp.bool_:
        largest_element = 1
    else:
        largest_element = COUNT_CAPACITY
    counts = counts.astype(jnp.int32)
    if axis is None:
        axes = tuple(range(counts.ndim))
    elif isinstance(axis, int):
        axes = (axis % counts.ndim,)
    else:
        axes = tuple(sorted(a % counts.ndim for a in axis))
    n_elements = math.prod(counts.shape[a] for a in axes)
    if n_elements > COUNT_CAPACITY:
        msg = (
            f"An off-grid landing count reduces {n_elements} elements, more than "
            "an int32 count can address."
        )
        raise ExecutionPlanningError(msg)
    if n_elements * largest_element <= COUNT_CAPACITY:
        return jnp.sum(counts, axis=axes, dtype=jnp.int32)
    digit_bits = (COUNT_CAPACITY // n_elements + 1).bit_length() - 1
    mask = (1 << digit_bits) - 1
    total = jnp.zeros(
        tuple(size for a, size in enumerate(counts.shape) if a not in axes),
        dtype=jnp.int32,
    )
    for shift in range(0, _INT32_BITS, digit_bits):
        digits = jnp.sum((counts >> shift) & mask, axis=axes, dtype=jnp.int32)
        scaled = jnp.where(
            digits > COUNT_CAPACITY >> shift, COUNT_CAPACITY, digits << shift
        )
        total = add_counts(left=total, right=scaled)
    return total
