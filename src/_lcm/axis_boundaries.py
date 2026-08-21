"""Shared ownership rules for one-dimensional interior boundaries."""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, FloatND, Int1D, IntND, ScalarFloat

type BoundaryOwner = Literal["left", "right"]
type PartitionEffect = Literal[
    "continuous_kink",
    "jump",
    "flat_budget",
    "feasibility",
]

_EFFECT_CODES: dict[PartitionEffect, int] = {
    "continuous_kink": 0,
    "jump": 1,
    "flat_budget": 2,
    "feasibility": 3,
}


@dataclass(frozen=True, kw_only=True)
class AxisBoundary:
    """One liquid-axis boundary before runtime values are sorted."""

    value: FloatND | float | int | bool
    """Boundary location in the liquid coordinate."""

    owner: BoundaryOwner
    """Adjacent interval containing the exact boundary value."""

    effect: PartitionEffect
    """Economic effect whose implementation consumes the boundary."""


@dataclass(frozen=True, kw_only=True)
class ResolvedAxisPartition:
    """One sorted, ownership-aware partition of the liquid axis."""

    values: Float1D
    """Sorted boundary locations, retaining coincident declarations."""

    owner_is_right: BoolND
    """Whether the right interval owns equality at each sorted boundary."""

    effect_codes: Int1D
    """Integer code of each sorted boundary's consuming economic effect."""

    effective_starts: Float1D
    """Representable closed start of every interval."""

    effective_stops: Float1D
    """Representable closed stop of every interval."""

    selection_thresholds: Float1D
    """Right-interval starts used to classify arbitrary coordinates."""


def effect_code(effect: PartitionEffect) -> int:
    """Return the stable internal integer code for one partition effect."""
    return _EFFECT_CODES[effect]


def partition_effect_for_schedule_kind(
    kind: Literal["continuous_kink", "jump", "hard_constraint"],
) -> PartitionEffect:
    """Translate the public schedule vocabulary into economic partition effects."""
    return "flat_budget" if kind == "hard_constraint" else kind


def boundary_owner_for_feasible_region(
    *,
    feasible_side: Literal["below", "above"],
    includes_boundary: bool,
) -> BoundaryOwner:
    """Return the adjacent interval owning equality for one feasible half-space."""
    feasible_owns = includes_boundary
    if feasible_side == "below":
        return "left" if feasible_owns else "right"
    return "right" if feasible_owns else "left"


def resolve_axis_partition(
    *,
    start: ScalarFloat,
    stop: ScalarFloat,
    boundaries: tuple[AxisBoundary, ...],
) -> ResolvedAxisPartition:
    """Sort boundary values and their ownership/effect metadata as one partition."""
    dtype = jnp.result_type(start, stop)
    if boundaries:
        unsorted_values = jnp.stack(
            [jnp.asarray(boundary.value, dtype=dtype) for boundary in boundaries]
        )
        order = jnp.argsort(unsorted_values, stable=True)
        values = unsorted_values[order]
        owner_is_right = jnp.asarray(
            [boundary.owner == "right" for boundary in boundaries],
            dtype=jnp.bool_,
        )[order]
        effect_codes = jnp.asarray(
            [effect_code(boundary.effect) for boundary in boundaries],
            dtype=jnp.int32,
        )[order]
    else:
        values = jnp.zeros((0,), dtype=dtype)
        owner_is_right = jnp.zeros((0,), dtype=jnp.bool_)
        effect_codes = jnp.zeros((0,), dtype=jnp.int32)

    effective_starts, effective_stops = _effective_segment_bounds_from_owner_codes(
        start=start,
        stop=stop,
        breakpoints=values,
        owner_is_right=owner_is_right,
    )
    return ResolvedAxisPartition(
        values=values,
        owner_is_right=owner_is_right,
        effect_codes=effect_codes,
        effective_starts=effective_starts,
        effective_stops=effective_stops,
        selection_thresholds=effective_starts[1:],
    )


def axis_interval_indices(
    *, partition: ResolvedAxisPartition, values: FloatND
) -> IntND:
    """Classify coordinates into intervals using boundary ownership."""
    return jnp.searchsorted(
        partition.selection_thresholds,
        values,
        side="right",
    ).astype(jnp.int32)


def feasibility_region_indices(
    *, partition: ResolvedAxisPartition, values: FloatND
) -> IntND:
    """Classify coordinates by the feasibility boundaries they have crossed.

    Non-feasibility boundaries refine the shared axis partition but do not split
    an envelope branch. This region label therefore advances only at boundaries
    owned by a feasibility constraint.
    """
    interval_indices = axis_interval_indices(partition=partition, values=values)
    is_feasibility = partition.effect_codes == effect_code("feasibility")
    region_by_interval = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(is_feasibility, dtype=jnp.int32),
        )
    )
    return region_by_interval[interval_indices]


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
    return _effective_segment_bounds_from_owner_codes(
        start=start,
        stop=stop,
        breakpoints=breakpoints,
        owner_is_right=jnp.asarray(
            [owner == "right" for owner in owners],
            dtype=jnp.bool_,
        ),
    )


def _effective_segment_bounds_from_owner_codes(
    *,
    start: ScalarFloat,
    stop: ScalarFloat,
    breakpoints: Float1D,
    owner_is_right: BoolND,
) -> tuple[Float1D, Float1D]:
    """Apply one-ULP open sides from an array aligned with sorted boundaries."""
    nominal_starts = jnp.concatenate((start[None], breakpoints))
    nominal_stops = jnp.concatenate((breakpoints, stop[None]))
    left_is_closed = jnp.concatenate((jnp.ones((1,), dtype=jnp.bool_), owner_is_right))
    right_is_closed = jnp.concatenate(
        (~owner_is_right, jnp.ones((1,), dtype=jnp.bool_))
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
