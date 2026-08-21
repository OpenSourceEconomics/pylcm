"""Shared ownership rules for one-dimensional interior boundaries."""

from typing import Literal

import jax.numpy as jnp

from lcm.typing import Float1D, ScalarFloat

type BoundaryOwner = Literal["left", "right"]


def effective_segment_bounds(
    *,
    start: ScalarFloat,
    stop: ScalarFloat,
    breakpoints: Float1D,
    owners: tuple[BoundaryOwner, ...],
) -> tuple[Float1D, Float1D]:
    """Return representable segment bounds with each breakpoint owned once.

    Args:
        start: Closed lower endpoint of the complete axis.
        stop: Closed upper endpoint of the complete axis.
        breakpoints: Strictly increasing interior boundary values.
        owners: Equality owner for each interior boundary.

    Returns:
        Effective starts and stops for every segment. An endpoint on the open
        side of a breakpoint is shifted by one representable floating-point
        value, while the owning side retains the exact breakpoint.

    """
    nominal_starts = jnp.concatenate((start[None], breakpoints))
    nominal_stops = jnp.concatenate((breakpoints, stop[None]))
    left_is_closed = jnp.asarray(
        [True, *(owner == "right" for owner in owners)], dtype=jnp.bool_
    )
    right_is_closed = jnp.asarray(
        [*(owner == "left" for owner in owners), True], dtype=jnp.bool_
    )
    effective_starts = jnp.where(
        left_is_closed,
        nominal_starts,
        jnp.nextafter(nominal_starts, jnp.inf),
    )
    effective_stops = jnp.where(
        right_is_closed,
        nominal_stops,
        jnp.nextafter(nominal_stops, -jnp.inf),
    )
    return effective_starts, effective_stops
