"""Local-upper-bound (LTM) upper-envelope refinement of EGM candidates.

Implements the brute upper-envelope method of Druedahl's `consav` package (the
`upperenvelope` routine), referenced in Dobrescu & Shanker (2026) as the
quadratic baseline (`LTM`). Inverting the Euler equation in models with discrete
choices yields a value *correspondence*: the candidates form a chain of linear
segments between consecutive nodes that, in non-concave regions, overlap in the
endogenous grid. LTM selects the upper envelope by a dense scan: at every output
abscissa it inspects every segment, keeps those that bracket the abscissa,
linearly interpolates the value (and policy) along each, and reports the
highest-value interpolant.

The cost is `O(N_query x N_segments) = O(K^2)` by construction — the
`(N_query, N_segments)` bracket-and-interpolate matrix is materialized in full.
This is deliberate: the method is the paper's quadratic reference, so the kernel
is not folded into a sequential scan. All shapes are static, so the kernel can
be `jax.jit`-compiled and `jax.vmap`-batched over a leading dimension of the
candidate arrays.

Unlike FUES, LTM inserts no exact segment-crossing abscissa: it evaluates the
envelope at the candidate abscissae, so a kink lands between two output nodes
and the downstream interpolation read recovers it to within the local grid
spacing. Unlike RFC, it consumes no supgradient — the segments carry their own
slopes.
"""

import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, ScalarInt


def refine_envelope(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    n_refined: int,
    segment_id: Float1D | None = None,
) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
    """Refine a candidate value correspondence to its upper envelope.

    The candidates arrive as a chain of consecutive linear segments (one segment
    per consecutive input pair), as the Euler inversion produces them: the
    constrained run followed by the interior run, each ascending along its own
    margin but jointly non-monotone in the endogenous grid. The envelope is
    evaluated at the candidate abscissae, sorted ascending; the refined arrays
    have static length `n_refined`, hold the envelope points in weakly ascending
    grid order, and are NaN-padded in the tail. No crossing point is inserted —
    a kink lands between two output nodes, recovered by the downstream read.

    Args:
        endog_grid: Candidate endogenous grid points (resources). Consecutive
            entries form the linear segments scanned for the envelope.
        policy: Candidate policy values at `endog_grid`.
        value: Candidate value-correspondence points at `endog_grid`.
        n_refined: Static length of the refined output arrays.
        segment_id: Optional per-candidate branch label, aligned with
            `endog_grid`. When supplied, a consecutive-pair segment is real iff
            both endpoints carry the same label, so unrelated branches are never
            bridged. `None` (the default) links every consecutive pair.

    Returns:
        Tuple of refined endogenous grid, refined policy, refined value (each
        of length `n_refined`, NaN-padded), and the number of envelope points
        `n_kept`. `n_kept > n_refined` signals overflow; the arrays then hold a
        valid truncated prefix of the envelope. Callers must check the counter
        rather than publish the truncated arrays silently — the EGM step
        NaN-poisons its published rows on overflow so the solve loop's NaN
        diagnostics name the offending (regime, period).

    """
    # A dead candidate arrives NaN-filled (the EGM step poisons `-inf`-valued
    # corners to NaN before refinement). A segment touching a dead endpoint is
    # excluded from the scan, and a dead abscissa sorts to the NaN tail, so it
    # is neither queried nor evaluated.
    dead = jnp.isnan(endog_grid) | jnp.isnan(value)

    # Sort the candidate abscissae ascending so the output row is weakly
    # ascending and the NaN tail is contiguous; dead nodes sort last.
    grid_key = jnp.where(dead, jnp.inf, endog_grid)
    order = jnp.argsort(grid_key)
    query_grid = jnp.where(dead, jnp.nan, endog_grid)[order]
    query_dead = dead[order]

    envelope_value, envelope_policy = _evaluate_envelope(
        query_grid=query_grid,
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        dead=dead,
        segment_id=segment_id,
    )

    # A query point no segment brackets (e.g. the lone dead-padded tail) yields
    # no envelope value: poison the whole triple to NaN so it joins the tail.
    no_segment = jnp.isneginf(envelope_value)
    drop = query_dead | no_segment
    refined_grid = jnp.where(drop, jnp.nan, query_grid)
    refined_policy = jnp.where(drop, jnp.nan, envelope_policy)
    refined_value = jnp.where(drop, jnp.nan, envelope_value)

    # Compact the live nodes into the NaN-padded prefix, preserving sorted
    # order; the live query points are already a contiguous ascending prefix, so
    # the scatter only trims the dropped tail.
    survives = ~drop
    position = jnp.cumsum(survives.astype(jnp.int32)) - 1
    slot = jnp.where(survives, position, n_refined)
    out_grid = jnp.full(n_refined, jnp.nan, dtype=endog_grid.dtype)
    out_policy = jnp.full(n_refined, jnp.nan, dtype=policy.dtype)
    out_value = jnp.full(n_refined, jnp.nan, dtype=value.dtype)
    out_grid = out_grid.at[slot].set(refined_grid, mode="drop")
    out_policy = out_policy.at[slot].set(refined_policy, mode="drop")
    out_value = out_value.at[slot].set(refined_value, mode="drop")

    n_kept = jnp.sum(survives, dtype=jnp.int32)
    return out_grid, out_policy, out_value, n_kept


def _evaluate_envelope(
    *,
    query_grid: Float1D,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    dead: BoolND,
    segment_id: Float1D | None = None,
) -> tuple[Float1D, Float1D]:
    """Evaluate the upper envelope at every query abscissa over all segments.

    Builds the dense `(N_query, N_segments)` bracket-and-interpolate matrix: each
    query `m_j` is tested against every segment `(k, k+1)` of consecutive input
    candidates. A segment brackets the query iff `m_j` lies in its abscissa
    range; the value and policy are then linearly interpolated along the
    segment. The envelope value is the maximum over bracketing segments, and the
    policy is the one carried by the winning segment. A query no segment
    brackets reports `-inf` value (the absent-envelope sentinel).

    Args:
        query_grid: Abscissae at which to evaluate the envelope; NaN tail.
        endog_grid: Candidate endogenous grid points (segment endpoints).
        policy: Candidate policy values at `endog_grid`.
        value: Candidate value-correspondence points at `endog_grid`.
        dead: Per-candidate dead indicator; a segment with a dead endpoint is
            excluded from the scan.

    Returns:
        Tuple of the envelope value and the envelope policy at each query
        abscissa.

    """
    # Segment endpoints: candidate `k` to candidate `k+1`, consecutive in input
    # order (the EGM cloud's natural segment chain).
    left_grid = endog_grid[:-1]
    right_grid = endog_grid[1:]
    left_policy = policy[:-1]
    right_policy = policy[1:]
    left_value = value[:-1]
    right_value = value[1:]
    segment_live = ~dead[:-1] & ~dead[1:]
    # With explicit topology, a consecutive-pair link is a real segment only
    # within one branch: drop links whose endpoints carry different labels so
    # unrelated branches are never bridged.
    if segment_id is not None:
        segment_live = segment_live & (segment_id[:-1] == segment_id[1:])

    query = query_grid[:, None]
    lower = jnp.minimum(left_grid, right_grid)[None, :]
    upper = jnp.maximum(left_grid, right_grid)[None, :]
    brackets = segment_live[None, :] & (query >= lower) & (query <= upper)

    # Linear position of the query along each segment; a zero-width segment
    # (coincident abscissae) takes weight 0, so its left endpoint applies.
    width = (right_grid - left_grid)[None, :]
    safe_width = jnp.where(width == 0.0, 1.0, width)
    relative = jnp.where(width == 0.0, 0.0, (query - left_grid[None, :]) / safe_width)

    value_interp = left_value[None, :] + relative * (right_value - left_value)[None, :]
    policy_interp = (
        left_policy[None, :] + relative * (right_policy - left_policy)[None, :]
    )

    # Keep only bracketing segments; non-bracketing ones get `-inf` value so the
    # row maximum ignores them and reports `-inf` when no segment brackets.
    masked_value = jnp.where(brackets, value_interp, -jnp.inf)
    best_segment = jnp.argmax(masked_value, axis=1)
    envelope_value = jnp.max(masked_value, axis=1)
    envelope_policy = jnp.take_along_axis(policy_interp, best_segment[:, None], axis=1)[
        :, 0
    ]
    return envelope_value, envelope_policy
