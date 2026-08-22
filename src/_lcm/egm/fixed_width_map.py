"""Map a per-row body over a leading axis with the block size as a runtime operand.

`jax.lax.map(func, xs, batch_size=b)` reshapes `xs` by a *static* `b`, so the
block size joins the compilation key and every value of it compiles a
differently vectorized body. Two block sizes then publish neighbouring floats
for the same real number, because XLA is free to reassociate each body
differently.

Here the compilation key is `(max_block_size, microtile_width)` alone. The
block size arrives as a scalar array and changes only the loop's stride and the
lanes each iteration commits, so every admitted block size runs the same
executable and publishes the same bits.

Two properties do the work:

- A global row `index` always occupies lane `index % microtile_width`, because
  the stride is a multiple of the microtile width. Lane assignment is therefore
  independent of the block size.
- Each iteration evaluates a full `max_block_size` window whatever the block
  size is, so no shape anywhere depends on the runtime value.

The cost is that work and workspace follow `max_block_size` rather than the
smaller runtime block: a smaller block changes how many iterations run, not how
much each one computes. `max_block_size` is therefore kept small — it bounds
both the per-iteration work and the padding the row buffers carry.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

type PyTree = Any


@dataclass(frozen=True, slots=True)
class FixedWidthMapGeometry:
    """Static vector width and window selected for one partitioned axis.

    Geometry is explicit at every call site.  It is numerical profile
    configuration, not a user partition knob: changing either field changes
    the executable, while ``requested_block_size`` remains a runtime operand of
    that executable.
    """

    microtile_width: int
    """Rows of one economic callback evaluated together."""

    profile_window: int
    """Maximum rows evaluated by one fixed-shape loop iteration."""

    def __post_init__(self) -> None:
        _fail_if_static_shape_invalid(
            max_block_size=self.profile_window,
            microtile_width=self.microtile_width,
        )


def map_fixed_width(
    *,
    func: Callable[[PyTree], PyTree],
    xs: PyTree,
    max_block_size: int,
    microtile_width: int,
    block_size: Array,
) -> PyTree:
    """Apply `func` to every row of `xs`, committing `block_size` rows per step.

    Args:
        func: Body applied to one row of `xs`. Evaluated `microtile_width` rows
            at a time, so it must be row-independent.
        xs: Pytree whose leaves share a leading row axis.
        max_block_size: Static window evaluated per iteration. Must be a
            positive multiple of `microtile_width`.
        microtile_width: Static vector width `func` is evaluated at.
        block_size: Scalar integer array of rows committed per iteration.
            Validate it with `validate_block_size` before tracing.

    Returns:
        Pytree of results with the same leading row count as `xs`.

    """
    _fail_if_static_shape_invalid(
        max_block_size=max_block_size, microtile_width=microtile_width
    )
    n_rows = _leading_size(xs)
    # One trailing window so the last iteration's full window stays in bounds:
    # a start below `n_rows` must admit `start + max_block_size` lanes.
    n_padded = math.ceil(n_rows / max_block_size) * max_block_size + max_block_size
    padded = jax.tree.map(lambda leaf: _pad_rows(leaf, n_padded=n_padded), xs)

    stride = jnp.asarray(block_size, dtype=jnp.int32)
    lane_offsets = jnp.arange(max_block_size, dtype=jnp.int32)

    def window(start: Array) -> PyTree:
        """Evaluate one fixed-shape window as microtiles of the declared width."""
        block = jax.tree.map(
            lambda leaf: jax.lax.dynamic_slice_in_dim(
                leaf, start, max_block_size, axis=0
            ),
            padded,
        )
        tiles = jax.tree.map(
            lambda leaf: leaf.reshape(
                (max_block_size // microtile_width, microtile_width, *leaf.shape[1:])
            ),
            block,
        )
        tiled = jax.lax.map(jax.vmap(func), tiles)
        return jax.tree.map(
            lambda leaf: leaf.reshape((max_block_size, *leaf.shape[2:])), tiled
        )

    template = jax.eval_shape(window, jnp.int32(0))
    out = jax.tree.map(
        lambda leaf: jnp.zeros((n_padded, *leaf.shape[1:]), dtype=leaf.dtype), template
    )

    def body(carry: tuple[Array, PyTree]) -> tuple[Array, PyTree]:
        start, published = carry
        fresh = window(start)
        committed = lane_offsets < stride

        def merge(new_leaf: Array, published_leaf: Array) -> Array:
            existing = jax.lax.dynamic_slice_in_dim(
                published_leaf, start, max_block_size, axis=0
            )
            mask = committed.reshape((max_block_size, *(1,) * (new_leaf.ndim - 1)))
            merged = jnp.where(mask, new_leaf, existing)
            return jax.lax.dynamic_update_slice_in_dim(
                published_leaf, merged, start, axis=0
            )

        return start + stride, jax.tree.map(merge, fresh, published)

    def keep_going(carry: tuple[Array, PyTree]) -> Array:
        return carry[0] < jnp.int32(n_rows)

    _, published = jax.lax.while_loop(keep_going, body, (jnp.int32(0), out))
    return jax.tree.map(lambda leaf: leaf[:n_rows], published)


def validate_block_size(
    *, block_size: int, max_block_size: int, microtile_width: int
) -> None:
    """Raise unless `block_size` is an admitted runtime partition."""
    _fail_if_static_shape_invalid(
        max_block_size=max_block_size, microtile_width=microtile_width
    )
    if block_size <= 0 or block_size > max_block_size:
        msg = (
            f"block_size must be in (0, {max_block_size}], got {block_size}. "
            "A larger partition needs a larger max_block_size, which is a new "
            "compilation."
        )
        raise ValueError(msg)
    if block_size % microtile_width:
        msg = (
            f"block_size must be a multiple of microtile_width "
            f"{microtile_width}, got {block_size}. A partial microtile would "
            "move rows to different lanes for different partitions."
        )
        raise ValueError(msg)


def _fail_if_static_shape_invalid(*, max_block_size: int, microtile_width: int) -> None:
    if microtile_width <= 0 or max_block_size <= 0:
        msg = (
            f"max_block_size and microtile_width must be positive, got "
            f"{max_block_size} and {microtile_width}."
        )
        raise ValueError(msg)
    if max_block_size % microtile_width:
        msg = (
            f"max_block_size {max_block_size} must be a multiple of "
            f"microtile_width {microtile_width}."
        )
        raise ValueError(msg)


def _leading_size(xs: PyTree) -> int:
    leaves = jax.tree.leaves(xs)
    if not leaves:
        msg = "xs must contain at least one array."
        raise ValueError(msg)
    sizes = {int(leaf.shape[0]) for leaf in leaves}
    if len(sizes) != 1:
        msg = f"every leaf of xs must share a leading row count, got {sorted(sizes)}."
        raise ValueError(msg)
    return sizes.pop()


def _pad_rows(leaf: Array, *, n_padded: int) -> Array:
    """Extend the row axis by repeating the first row.

    `func` is row-independent, so the padded lanes only ever produce values that
    are masked out or sliced away; repeating a real row keeps them in the same
    numeric range as the genuine ones.
    """
    pad = n_padded - leaf.shape[0]
    if pad <= 0:
        return leaf
    return jnp.concatenate([leaf, jnp.repeat(leaf[:1], pad, axis=0)], axis=0)


def enclosing_max_block_size(*, n_rows: int, microtile_width: int) -> int:
    """Return the smallest admitted static window covering `n_rows`."""
    if n_rows <= 0:
        msg = f"n_rows must be positive, got {n_rows}."
        raise ValueError(msg)
    return math.ceil(n_rows / microtile_width) * microtile_width


def max_block_size_for_axis(*, n_rows: int, geometry: FixedWidthMapGeometry) -> int:
    """Return this call site's static window for an axis with ``n_rows`` rows."""
    return min(
        geometry.profile_window,
        enclosing_max_block_size(
            n_rows=n_rows,
            microtile_width=geometry.microtile_width,
        ),
    )


def admitted_block_size(
    *, requested: int, max_block_size: int, microtile_width: int
) -> int:
    """Round a requested partition up to the nearest admitted one.

    A partition that is not a microtile multiple would move rows to different
    lanes for different requests, so requests are coarsened rather than refused.
    The caller's value is a requested commit stride; the nearest admitted stride
    at or above it keeps every request on one executable. A non-positive request
    selects `max_block_size`, the largest stride admitted by this static profile.
    It covers the whole axis in one iteration only when the axis fits that window.
    """
    if requested <= 0:
        return max_block_size
    rounded = math.ceil(requested / microtile_width) * microtile_width
    return min(rounded, max_block_size)


def map_partitioned(
    *,
    func: Callable[[PyTree], PyTree],
    xs: PyTree,
    requested_block_size: int,
    geometry: FixedWidthMapGeometry,
) -> PyTree:
    """Map ``func`` over ``xs`` at one call site's declared geometry.

    The static window is derived from the axis length and the explicit
    call-site geometry.  The request is coarsened to an admitted partition but
    remains a runtime loop stride, so all requests admitted by one geometry use
    one executable.
    """
    n_rows = _leading_size(xs)
    max_block_size = max_block_size_for_axis(n_rows=n_rows, geometry=geometry)
    return map_fixed_width(
        func=func,
        xs=xs,
        max_block_size=max_block_size,
        microtile_width=geometry.microtile_width,
        block_size=jnp.int32(
            admitted_block_size(
                requested=requested_block_size,
                max_block_size=max_block_size,
                microtile_width=geometry.microtile_width,
            )
        ),
    )
