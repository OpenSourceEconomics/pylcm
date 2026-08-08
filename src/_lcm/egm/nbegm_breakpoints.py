"""Breakpoint geometry on the liquid axis.

NBEGM merges every institutional boundary on the liquid axis into one sorted
interval partition: each interval is lower-closed/upper-open, so a liquid value
exactly on a breakpoint belongs to the interval above it. The helpers here map
declared thresholds to their liquid-space preimages, clamp them to the grid
span, and derive per-interval representative points and affine segment
coefficients — consumed by the NBEGM solver's core builders.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from lcm.typing import Float1D, ScalarFloat


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


def clamp_breakpoints_to_grid(*, breakpoints: Float1D, liquid_grid: Float1D) -> Float1D:
    """Clamp asset breakpoints to a margin just outside the liquid grid.

    A boundary whose derived variable has (near-)zero slope in the liquid state has no
    finite asset preimage: `linear_asset_preimage` sends it to `±inf` (the threshold is
    never crossed within reach) or `NaN` (the whole cell sits on the boundary). A
    non-finite breakpoint would make the adjacent interval's midpoint non-finite and
    poison that interval's recovered affine segment — including the live interval that
    holds the query grid. Clamping every breakpoint into a margin one grid-width outside
    `[grid_min, grid_max]` collapses such a boundary to an empty edge interval whose
    midpoint stays finite, while leaving every genuine in-grid breakpoint untouched. A
    `NaN` breakpoint is sent to the upper margin.

    Args:
        breakpoints: Asset breakpoints on the liquid axis, in any order.
        liquid_grid: Ascending liquid-state grid; its endpoints bound the grid.

    Returns:
        The breakpoints clamped into `[grid_min - margin, grid_max + margin]`, all
        finite.

    """
    grid_min = liquid_grid[0]
    grid_max = liquid_grid[-1]
    margin = jnp.maximum(grid_max - grid_min, 1.0)
    lower = grid_min - margin
    upper = grid_max + margin
    finite = jnp.where(jnp.isnan(breakpoints), upper, breakpoints)
    return jnp.clip(finite, lower, upper)


def interval_midpoints(*, liquid_grid: Float1D, breakpoints: Float1D) -> Float1D:
    """Return one interior representative liquid point per interval.

    The outer intervals are unbounded, so the liquid grid's endpoints close them:
    interval 0 spans `[grid_min, b_0)`, interval N spans `[b_{N-1}, grid_max]`, and
    each interior interval the gap between consecutive sorted breakpoints. The
    representative is the midpoint of each closed interval — a point strictly
    inside it (away from the kinks) where the active affine segment is read.

    Args:
        liquid_grid: Liquid-state grid; its endpoints bound the outer intervals.
        breakpoints: Asset breakpoints on the liquid axis, in any order.

    Returns:
        One representative liquid point per interval, length
        `len(breakpoints) + 1`.

    """
    sorted_breakpoints = jnp.sort(breakpoints)
    grid_min = liquid_grid[0]
    grid_max = liquid_grid[-1]
    lower_edges = jnp.concatenate([grid_min[None], sorted_breakpoints])
    upper_edges = jnp.concatenate([sorted_breakpoints, grid_max[None]])
    return 0.5 * (lower_edges + upper_edges)


def interval_segment_coefficients(
    *,
    schedule: Callable[[ScalarFloat], ScalarFloat],
    interval_midpoints: Float1D,
) -> tuple[Float1D, Float1D]:
    """Recover a schedule's active affine segment per interval.

    A piecewise-affine schedule of the liquid state is affine inside each interval.
    Reading its slope (by differentiation) and value at the interval's interior
    representative recovers the active segment `value = slope * liquid + intercept`
    that holds throughout that interval. Sampling away from the kinks gives the
    correct one-sided segment at each interval.

    Args:
        schedule: The schedule as a function of the scalar liquid state, with every
            other argument bound.
        interval_midpoints: One interior representative liquid point per interval.

    Returns:
        Tuple `(slopes, intercepts)` of the active affine segment per interval.

    """
    slopes = jax.vmap(jax.grad(schedule))(interval_midpoints)
    values = jax.vmap(schedule)(interval_midpoints)
    intercepts = values - slopes * interval_midpoints
    return slopes, intercepts
