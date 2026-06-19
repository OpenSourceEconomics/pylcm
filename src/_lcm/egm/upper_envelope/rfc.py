"""Rooftop-Cut (RFC) upper-envelope refinement of EGM candidates.

Implements the roof-top cut of Dobrescu, L., & Shanker, A. (2024). Using
Inverse Euler Equations to Solve Multidimensional Discrete-Continuous Dynamic
Models: A General Method. SSRN 4850746 (their Box 1).

Inverting the Euler equation in models with discrete choices yields a value
*correspondence*: in non-concave regions, several candidate points share a
neighborhood of the endogenous grid, each on a different choice-specific value
segment. RFC selects the upper envelope by a dense dominance test: at each
candidate $j$ it builds the tangent (value) line from the supgradient
$\\mu_j = \\partial v / \\partial R$ (the exact value-row slope by the envelope
theorem) and deletes every neighbor $l$ within a search radius that

- lies *below* $j$'s tangent — $v_l - v_j < \\mu_j\\,(R_l - R_j)$ — and
- sits across a policy jump — $|\\Delta c / \\Delta R| > $ `jump_thresh`.

The jump gate is what spares a strictly concave segment: every point on a
concave curve lies below its neighbors' tangents, but only a genuine
segment-switch carries the policy discontinuity, so only switches are cut.

Unlike FUES, RFC *only deletes* points — it never inserts the exact
segment-crossing abscissa. The refined output is therefore a NaN-padded,
weakly-ascending *subset* of the input candidates: the dominated points are
gone and the envelope points remain, with a kink landing between two retained
points (recovered by the downstream Hermite carry read to within local grid
spacing). The per-pair test has no sequential carry, so the kernel is a
`vmap`-friendly dense computation that generalizes to multidimensional grids.

All shapes are static, so the kernel can be `jax.jit`-compiled and `jax.vmap`-
batched over a leading dimension of the candidate arrays.
"""

import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, FloatND, ScalarInt


def refine_envelope(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal_utility: Float1D,
    n_refined: int,
    search_radius: int = 10,
    jump_thresh: float = 2.0,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
    """Refine a candidate value correspondence to its upper envelope.

    The candidates may arrive in any order; they are sorted (NaN-stable)
    ascending in `endog_grid` first. The refined arrays have static length
    `n_refined`, hold the surviving envelope points in weakly ascending grid
    order, and are NaN-padded in the tail. No crossing point is inserted — the
    output is a subset of the (sorted) input candidates.

    Args:
        endog_grid: Candidate endogenous grid points (resources), any order.
        policy: Candidate policy values at `endog_grid`.
        value: Candidate value-correspondence points at `endog_grid`.
        marginal_utility: Candidate supgradient $\\mu = \\partial v / \\partial
            R$, the exact value-row slope by the envelope theorem.
        n_refined: Static length of the refined output arrays.
        search_radius: Number of neighbors on each side (in sorted grid order)
            inspected by the dominance test.
        jump_thresh: Threshold on $|\\Delta c / \\Delta R|$ above which two
            candidates are treated as lying on different value-function
            segments.

    Returns:
        Tuple of refined endogenous grid, refined policy, refined value (each
        of length `n_refined`, NaN-padded), and the number of envelope points
        `n_kept`. `n_kept > n_refined` signals overflow; the arrays then hold a
        valid truncated prefix of the envelope. Callers must check the counter
        rather than publish the truncated arrays silently — the EGM step
        NaN-poisons its published rows on overflow so the solve loop's NaN
        diagnostics name the offending (regime, period).

    """
    order = jnp.argsort(endog_grid)
    grid = endog_grid[order]
    policy_sorted = policy[order]
    value_sorted = value[order]
    mu = marginal_utility[order]

    # A dead candidate arrives NaN-filled (the EGM step poisons `-inf`-valued
    # corners to NaN before refinement). Treat its value as `-inf` in the
    # dominance test: it is then dominated by every finite point and can never
    # dominate one, so it never survives and never deletes a live candidate.
    dead = jnp.isnan(grid) | jnp.isnan(value_sorted)
    value_for_test = jnp.where(dead, -jnp.inf, value_sorted)
    grid_for_test = jnp.where(dead, jnp.inf, grid)
    mu_for_test = jnp.where(dead, 0.0, mu)

    deleted_by_neighbor = _dominated_within_radius(
        grid=grid_for_test,
        policy=policy_sorted,
        value=value_for_test,
        marginal_utility=mu_for_test,
        search_radius=search_radius,
        jump_thresh=jump_thresh,
    )
    survives = ~dead & ~deleted_by_neighbor

    # Compact survivors into the NaN-padded prefix, preserving sorted order.
    position = jnp.cumsum(survives.astype(jnp.int32)) - 1
    slot = jnp.where(survives, position, n_refined)
    refined_grid = jnp.full(n_refined, jnp.nan, dtype=grid.dtype)
    refined_policy = jnp.full(n_refined, jnp.nan, dtype=policy_sorted.dtype)
    refined_value = jnp.full(n_refined, jnp.nan, dtype=value_sorted.dtype)
    refined_grid = refined_grid.at[slot].set(grid, mode="drop")
    refined_policy = refined_policy.at[slot].set(policy_sorted, mode="drop")
    refined_value = refined_value.at[slot].set(value_sorted, mode="drop")

    n_kept = jnp.sum(survives, dtype=jnp.int32)
    return refined_grid, refined_policy, refined_value, n_kept


def _dominated_within_radius(
    *,
    grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal_utility: Float1D,
    search_radius: int,
    jump_thresh: float,
) -> BoolND:
    """Indicate which candidates a neighbor's tangent dominates across a jump.

    Builds the dense (point, offset) test: each candidate `l` is compared
    against the `search_radius` candidates on each side. `l` is deleted iff
    some in-radius neighbor `j` has `l` below `j`'s tangent *and* a policy jump
    between them.

    Args:
        grid: Sorted candidate endogenous grid points.
        policy: Candidate policy values at `grid`.
        value: Candidate value-correspondence points at `grid`.
        marginal_utility: Candidate supgradient at `grid`.
        search_radius: Number of neighbors on each side to inspect.
        jump_thresh: Threshold on $|\\Delta c / \\Delta R|$.

    Returns:
        Boolean indicator per candidate; `True` marks a dominated candidate.

    """
    n_input = grid.shape[0]
    point = jnp.arange(n_input, dtype=jnp.int32)
    # Signed offsets covering `search_radius` neighbors on each side.
    offsets = jnp.concatenate(
        [
            -jnp.arange(1, search_radius + 1, dtype=jnp.int32)[::-1],
            jnp.arange(1, search_radius + 1, dtype=jnp.int32),
        ]
    )
    # neighbor[l, o] = the anchor j that tests candidate l at offset o.
    neighbor = point[:, None] + offsets[None, :]
    in_bounds = (neighbor >= 0) & (neighbor < n_input)
    clipped = jnp.clip(neighbor, 0, n_input - 1)

    grid_j = grid[clipped]
    policy_j = policy[clipped]
    value_j = value[clipped]
    mu_j = marginal_utility[clipped]

    grid_l = grid[:, None]
    delta_grid = grid_l - grid_j
    # `l` below `j`'s tangent line: v_l - v_j < mu_j * (R_l - R_j).
    below_tangent = (value[:, None] - value_j) < mu_j * delta_grid - 1e-9
    policy_jump = _has_policy_jump(
        grid_a=grid_j,
        policy_a=policy_j,
        grid_b=grid_l,
        policy_b=policy[:, None],
        jump_thresh=jump_thresh,
    )
    deletes = in_bounds & below_tangent & policy_jump
    return jnp.any(deletes, axis=1)


def _has_policy_jump(
    *,
    grid_a: FloatND,
    policy_a: FloatND,
    grid_b: FloatND,
    policy_b: FloatND,
    jump_thresh: float,
) -> BoolND:
    """Indicate whether two points lie on different value-function segments.

    Points lie on different segments iff the policy secant $|\\Delta c /
    \\Delta R|$ between them exceeds `jump_thresh`. Coincident abscissae carry
    no jump.

    Args:
        grid_a: Endogenous grid point(s) of the first point.
        policy_a: Policy value(s) of the first point.
        grid_b: Endogenous grid point(s) of the second point.
        policy_b: Policy value(s) of the second point.
        jump_thresh: Threshold on $|\\Delta c / \\Delta R|$.

    Returns:
        Boolean indicator(s), broadcast over the inputs.

    """
    delta_grid = grid_b - grid_a
    safe_delta = jnp.where(delta_grid == 0.0, 1.0, delta_grid)
    policy_slope = jnp.where(delta_grid == 0.0, 0.0, (policy_b - policy_a) / safe_delta)
    return jnp.abs(policy_slope) > jump_thresh
