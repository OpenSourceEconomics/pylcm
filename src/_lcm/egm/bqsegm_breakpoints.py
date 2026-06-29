"""Mixed-kind breakpoint IR: the solver's uniform view of a case boundary.

BQSEGM merges every institutional boundary on the liquid axis into one sorted
interval partition. That requires a representation independent of which user form
declared the boundary: a Medicaid asset-test jump, a tax-bracket continuous kink,
and a consumption-floor hard constraint all become a `BreakpointSource` carrying
the monotone boundary variable, its threshold, the discontinuity kind, and the
side that owns the exact boundary point.

This module only lifts the declared metadata (`lcm.case_boundary` /
`lcm.boundary`) into that IR. Mapping each source to its asset preimage, merging
the breakpoints into intervals, specializing per-interval formulas, and emitting
per-kind candidates are downstream solver steps.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from _lcm.egm.bqsegm import BQSEGMRegistry
from _lcm.typing import FunctionName
from lcm.case_piece import EqualityOwner
from lcm.typing import Float1D, Int1D, ScalarFloat

type BreakpointKind = Literal[
    "jump", "continuous_kink", "hard_constraint", "open_boundary"
]


@dataclass(frozen=True)
class BreakpointSource:
    """One breakpoint on the liquid axis, in mixed-kind solver IR form.

    A user-declared boundary surface (jump, continuous kink, hard constraint) and
    a solver-synthesized open one-sided limit (`open_boundary`) share this record,
    so the downstream interval partition treats them uniformly.
    """

    variable: FunctionName
    """Name of the monotone boundary variable compared against the threshold."""
    threshold: str
    """Name of the DAG variable or parameter holding the threshold value."""
    kind: BreakpointKind
    """Discontinuity kind: jump, continuous kink, hard constraint, or open limit."""
    equality_owner: EqualityOwner
    """Side that owns the exact boundary point (`when` or `otherwise`)."""


def breakpoint_sources_from_registry(
    registry: BQSEGMRegistry,
) -> tuple[BreakpointSource, ...]:
    """Lift every declared boundary surface to a breakpoint source.

    Args:
        registry: Collected case-piece metadata of one regime's function pool.

    Returns:
        Tuple of breakpoint sources, one per declared equality surface, in
        predicate-then-surface declaration order.

    """
    return tuple(
        BreakpointSource(
            variable=surface.variable,
            threshold=surface.threshold,
            kind=surface.kind,
            equality_owner=surface.equality_owner,
        )
        for meta in registry.boundaries.values()
        for surface in meta.boundaries
    )


def affine_coefficients(
    z_of_liquid: Callable[[ScalarFloat], ScalarFloat],
) -> tuple[ScalarFloat, ScalarFloat]:
    """Recover the slope and intercept of an affine boundary variable.

    For a boundary variable `z(assets) = slope * assets + intercept` the slope is
    its derivative (constant for an affine map) and the intercept is its value at
    zero assets. Both are read at zero, so the result is a traced quantity in any
    runtime parameter the function closes over.

    Args:
        z_of_liquid: The boundary variable as a function of the scalar liquid
            state, with every other argument bound.

    Returns:
        Tuple `(slope, intercept)` of the affine map in the liquid state.

    """
    zero = jnp.zeros(())
    slope = jax.grad(z_of_liquid)(zero)
    intercept = z_of_liquid(zero)
    return slope, intercept


def linear_asset_preimage(
    z_of_liquid: Callable[[ScalarFloat], ScalarFloat],
    *,
    threshold: ScalarFloat,
) -> ScalarFloat:
    """Map a threshold in a monotone-affine boundary variable to its asset value.

    A boundary `z(assets) == threshold` on a derived quantity becomes a breakpoint
    on the liquid axis at `assets = (threshold - intercept) / slope`, the exact
    inverse for an affine `z`. The monotonicity gate guarantees a nonzero,
    single-signed slope, so the preimage is unique. The piecewise-affine inversion
    needed once `z` bends (post-claiming SS-taxation tiers) is a later step.

    Args:
        z_of_liquid: The boundary variable as a function of the scalar liquid
            state, with every other argument bound.
        threshold: Threshold value in the boundary variable's units.

    Returns:
        The asset breakpoint — the liquid-state value where `z` equals the
        threshold.

    """
    slope, intercept = affine_coefficients(z_of_liquid)
    return (threshold - intercept) / slope


def n_intervals(*, n_breakpoints: int) -> int:
    """Return the interval count for a partition of N breakpoints on one axis.

    N breakpoints merged on the liquid axis split it into N+1 ordered intervals.
    This is a build-time static count — the number of per-cell EGM solves the
    partition fans out into.

    Args:
        n_breakpoints: Number of asset breakpoints on the liquid axis.

    Returns:
        The number of intervals, `n_breakpoints + 1`.

    """
    return n_breakpoints + 1


def interval_index(*, liquid_grid: Float1D, breakpoints: Float1D) -> Int1D:
    """Assign each liquid grid point the index of the interval it falls in.

    The breakpoints are sorted ascending and merged into one partition; interval
    `i` spans `[b_{i-1}, b_i)` with the open ends `[-inf, b_0)` and
    `[b_{N-1}, +inf)`. A liquid point exactly on a breakpoint joins the interval
    above it (lower-closed, upper-open cells); which case's candidate ultimately
    owns that exact point is resolved later by the branch-aware envelope using the
    per-breakpoint equality owner. Coincident breakpoints leave an empty interval,
    which is harmless.

    Args:
        liquid_grid: Liquid-state grid points to classify.
        breakpoints: Asset breakpoints on the liquid axis, in any order.

    Returns:
        Per-grid-point interval index in `0 .. len(breakpoints)`.

    """
    sorted_breakpoints = jnp.sort(breakpoints)
    return jnp.searchsorted(sorted_breakpoints, liquid_grid, side="right").astype(
        jnp.int32
    )
