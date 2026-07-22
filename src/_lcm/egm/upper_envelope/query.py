"""Exact query-side upper envelope of an EGM candidate correspondence.

The query-side counterpart of the full-row refiners (`fues`, `rfc`, `ltm`,
`mss`). Those materialise the whole refined envelope row and the caller then
reads it at a query; this evaluates the envelope *directly* at a set of query
abscissae without ever building the row.

For one query the value is the maximum, over every live branch segment that
brackets it, of the segment's linear value; the policy and marginal are the
winning segment's. A folded branch contributes several bracketing segments, so
the maximum is exact for the piecewise-linear correspondence. Topology is
explicit: a segment is the link between two consecutive candidates carrying the
same `segment_id`, so unrelated branches are never bridged — the contract the
host oracle enforces.

By default the evaluation is a fixed-shape `(n_query, n_segment)`
bracket-and-reduce: no sequential scan, no NaN-padded refined row,
branch-parallel and reduction-heavy, which is the shape an accelerator runs
fastest. This is the backend asset-row mode wants — one query per Euler node, no
full envelope to refine. For a large `(n_query, n_segment)` that dense matrix is
itself the memory wall; `segment_block_size` swaps it for a two-pass blocked
scan over segment blocks (running max, then max-slope-among-near-max against the
fixed envelope value), which peaks at `(n_query, block)` instead of
`(n_query, n_segment)` and returns the identical result.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, FloatND

# Right-continuous tie band: among bracketing segments whose interpolated value
# is within this of the envelope maximum, the larger value-slope wins (it is
# higher just to the right). Both the dense and blocked paths use it, so they
# select the same policy/marginal at a tie.
#
# The band MUST be a dtype- and scale-aware ULP floor, not a fixed absolute
# constant (round-3 audit F6, defect class DC-1). Two segments that tie exactly
# in real arithmetic can differ by one rounding ULP after interpolation; in
# float32 a single ULP at unit value scale (~1e-7) dwarfs a fixed 1e-12 band, so
# the mathematically tied segment with the larger right-hand slope is excluded
# before the right-continuous rule runs and the wrong branch wins. Scale the band
# to the values being compared: `_TIE_BAND_ULPS * eps(dtype) * max(|a|, |b|)`.
# This supersedes an interim `1e-12 * max(1, |ref|)` relative band: that scaled
# with magnitude but kept a `1e-12` coefficient with no `eps`, so it stayed below
# one float32 ULP at large value scale — still precision-blind, which is exactly
# what F6 flags.
_TIE_BAND_ULPS = 64.0


def _value_tie_band(a: FloatND, b: FloatND) -> FloatND:
    """Dtype- and magnitude-scaled value-tie half-width (DC-1 floor)."""
    eps = jnp.finfo(jnp.result_type(a, b)).eps
    return _TIE_BAND_ULPS * eps * jnp.maximum(jnp.abs(a), jnp.abs(b))


class _SegmentLinks(NamedTuple):
    """Per-link endpoints of the candidate correspondence (length `n - 1`).

    A link is the consecutive-candidate pair `(i, i+1)`; it is a real envelope
    segment only where `live` (both endpoints finite and sharing a branch label).
    """

    left_grid: Float1D
    right_grid: Float1D
    left_value: Float1D
    right_value: Float1D
    left_policy: Float1D
    right_policy: Float1D
    left_marginal: Float1D
    right_marginal: Float1D
    live: BoolND


def envelope_at_query(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    x_query: FloatND,
    segment_block_size: int = 0,
) -> tuple[FloatND, FloatND, FloatND]:
    """Evaluate the branch-aware upper envelope at each query abscissa.

    Args:
        endog_grid: Candidate endogenous grid points (resources), any order
            within a branch; a NaN entry is a dead/padding candidate.
        policy: Candidate policy values at `endog_grid`.
        value: Candidate value-correspondence points at `endog_grid`.
        marginal: Candidate marginal values (the supgradient) at `endog_grid`.
        segment_id: Per-candidate branch label. A segment is a consecutive-pair
            link whose endpoints share a label, so unrelated branches never join.
        x_query: Abscissae at which to evaluate the envelope.
        segment_block_size: When `0` (or at least the number of segments), the
            dense `(n_query, n_segment)` reduction. A positive value below the
            segment count instead runs the two-pass blocked scan, peaking at
            `(n_query, segment_block_size)`; the result is identical.

    Returns:
        Tuple of the envelope value, the winning segment's policy, and the
        winning segment's marginal at each query, each shaped like `x_query`. A
        query no live segment brackets yields NaN in all three.
    """
    dead = jnp.isnan(endog_grid) | jnp.isnan(value)
    # A link is a real segment only within one branch: both endpoints live and
    # carrying the same label.
    consecutive = _SegmentLinks(
        left_grid=endog_grid[:-1],
        right_grid=endog_grid[1:],
        left_value=value[:-1],
        right_value=value[1:],
        left_policy=policy[:-1],
        right_policy=policy[1:],
        left_marginal=marginal[:-1],
        right_marginal=marginal[1:],
        live=~dead[:-1] & ~dead[1:] & (segment_id[:-1] == segment_id[1:]),
    )
    # Every live candidate is also a zero-width self-bracket at its own abscissa,
    # so a lone point — a folded-out or boundary-collapsed candidate with no
    # consecutive same-segment neighbour — stays visible where a query lands on
    # it, instead of collapsing to a lower multi-point branch. A right-extending
    # consecutive link outranks a zero-width self-bracket in the right-continuous
    # tie-break, so multi-point chains and their interpolation are unchanged; a
    # self-bracket wins only where nothing brackets the query from the right.
    self_bracket = _SegmentLinks(
        left_grid=endog_grid,
        right_grid=endog_grid,
        left_value=value,
        right_value=value,
        left_policy=policy,
        right_policy=policy,
        left_marginal=marginal,
        right_marginal=marginal,
        live=~dead,
    )
    links = _SegmentLinks(
        *(
            jnp.concatenate([pair, point])
            for pair, point in zip(consecutive, self_bracket, strict=True)
        )
    )

    query = jnp.asarray(x_query)
    n_segment = links.left_grid.shape[0]
    if 0 < segment_block_size < n_segment:
        return _envelope_at_query_blocked(
            links=links, query=query, block_size=segment_block_size
        )

    flat = query.reshape(-1)[:, None]
    left_grid, right_grid = links.left_grid, links.right_grid
    left_value, right_value = links.left_value, links.right_value
    left_policy, right_policy = links.left_policy, links.right_policy
    left_marginal, right_marginal = links.left_marginal, links.right_marginal
    segment_live = links.live
    lower = jnp.minimum(left_grid, right_grid)[None, :]
    upper = jnp.maximum(left_grid, right_grid)[None, :]
    brackets = segment_live[None, :] & (flat >= lower) & (flat <= upper)

    width = (right_grid - left_grid)[None, :]
    safe_width = jnp.where(width == 0.0, 1.0, width)
    relative = jnp.where(width == 0.0, 0.0, (flat - left_grid[None, :]) / safe_width)
    value_interp = left_value[None, :] + relative * (right_value - left_value)[None, :]
    policy_interp = (
        left_policy[None, :] + relative * (right_policy - left_policy)[None, :]
    )
    marginal_interp = (
        left_marginal[None, :] + relative * (right_marginal - left_marginal)[None, :]
    )

    masked_value = jnp.where(brackets, value_interp, -jnp.inf)
    any_bracket = jnp.any(brackets, axis=1)
    max_value = jnp.max(masked_value, axis=1, keepdims=True)
    # Break a value tie right-continuously, matching the kernel's `side="right"`
    # read: among the bracketing segments attaining the maximum, prefer one that
    # extends strictly to the right of the query (so "larger value-slope is higher
    # just to the right" is meaningful), and among those the larger slope. Only at the
    # global upper endpoint, where nothing continues right, fall back to the largest
    # near-max slope. `_right_continuous_rank` folds both keys into one comparable
    # scalar so this dense reduction and the blocked scan select the same winner.
    slope = (right_value - left_value)[None, :] / safe_width
    near_max = brackets & (
        masked_value >= max_value - _value_tie_band(masked_value, max_value)
    )
    right_available = flat < upper
    best = jnp.argmax(
        _right_continuous_rank(
            near_max=near_max, right_available=right_available, slope=slope
        ),
        axis=1,
    )
    env_value = jnp.where(any_bracket, max_value[:, 0], jnp.nan)
    env_policy = jnp.where(
        any_bracket,
        jnp.take_along_axis(policy_interp, best[:, None], axis=1)[:, 0],
        jnp.nan,
    )
    env_marginal = jnp.where(
        any_bracket,
        jnp.take_along_axis(marginal_interp, best[:, None], axis=1)[:, 0],
        jnp.nan,
    )
    return (
        env_value.reshape(query.shape),
        env_policy.reshape(query.shape),
        env_marginal.reshape(query.shape),
    )


def _right_continuous_rank(
    *, near_max: BoolND, right_available: BoolND, slope: FloatND
) -> FloatND:
    """One comparable scalar per segment for the right-continuous tie-break.

    Ranks a right-extending near-max segment above one that ends at the query, and
    among equally-eligible segments the larger value-slope. `arctan` bounds the slope
    into `(-pi/2, pi/2)`, so the integer right-extends bit dominates it; non-near-max
    segments get `-inf`. `argmax` over this key reproduces "prefer a right-extending
    near-max segment, else the largest near-max slope" with no global reduction, so
    the dense path and the blocked scan (which compares this scalar across blocks)
    select the same winner.
    """
    bounded_slope = jnp.arctan(slope) / jnp.pi + 0.5
    rank = right_available.astype(bounded_slope.dtype) + bounded_slope
    return jnp.where(near_max, rank, -jnp.inf)


def _block_query_terms(
    *, block: FloatND, live: BoolND, flat: Float1D
) -> tuple[BoolND, FloatND, FloatND, FloatND, FloatND, FloatND]:
    """Bracket-and-interpolate one segment block against every query.

    `block` is one `(block_size, 8)` slice of the stacked link endpoint columns
    and `live` its `(block_size,)` live-flag slice. Returns the
    `(n_query, block_size)` bracket mask; the value, policy, marginal, and
    value-slope interpolated at each query for each link in the block; and the
    link's upper endpoint (for the right-continuous tie-break) — the same
    quantities the dense path forms over all segments at once, but only for this
    block, so the peak working set is `(n_query, block_size)`.
    """
    left_grid, right_grid = block[:, 0], block[:, 1]
    left_value, right_value = block[:, 2], block[:, 3]
    left_policy, right_policy = block[:, 4], block[:, 5]
    left_marginal, right_marginal = block[:, 6], block[:, 7]

    q = flat[:, None]
    lower = jnp.minimum(left_grid, right_grid)[None, :]
    upper = jnp.maximum(left_grid, right_grid)[None, :]
    brackets = live[None, :] & (q >= lower) & (q <= upper)

    width = (right_grid - left_grid)[None, :]
    safe_width = jnp.where(width == 0.0, 1.0, width)
    relative = jnp.where(width == 0.0, 0.0, (q - left_grid[None, :]) / safe_width)
    value_interp = left_value[None, :] + relative * (right_value - left_value)[None, :]
    policy_interp = (
        left_policy[None, :] + relative * (right_policy - left_policy)[None, :]
    )
    marginal_interp = (
        left_marginal[None, :] + relative * (right_marginal - left_marginal)[None, :]
    )
    slope = (right_value - left_value)[None, :] / safe_width
    return brackets, value_interp, policy_interp, marginal_interp, slope, upper


def _envelope_at_query_blocked(
    *, links: _SegmentLinks, query: FloatND, block_size: int
) -> tuple[FloatND, FloatND, FloatND]:
    """Two-pass blocked equivalent of the dense `(n_query, n_segment)` reduction.

    Both passes are exact associative folds against a fixed target, so the result
    matches the dense path (up to floating-point reassociation between the two
    XLA lowerings):

    - Pass 1 accumulates the running per-query max over segment blocks — the
      envelope value, with a running `any_bracket` flag.
    - Pass 2 re-scans the blocks and, among segments whose value is within
      the dtype-scaled `_value_tie_band` of that (now fixed) envelope value,
      keeps the winner of the
      right-continuous rank (`_right_continuous_rank`: a right-extending near-max
      segment over one ending at the query, then larger value-slope) — the dense
      path's tie-break. The strict cross-block `>` keeps the earliest such winner,
      matching the dense `argmax`.

    The links are padded to a multiple of `block_size` with dead segments (which
    never bracket) and reshaped to `(n_block, block_size)`; the scan peaks at
    `(n_query, block_size)` working memory.
    """
    flat = query.reshape(-1)
    n_query = flat.shape[0]
    n_segment = links.live.shape[0]
    pad = (-n_segment) % block_size

    def _padded(column: FloatND, fill: float) -> FloatND:
        if pad == 0:
            return column
        return jnp.concatenate([column, jnp.full((pad,), fill, dtype=column.dtype)])

    columns = jnp.stack(
        [
            _padded(links.left_grid, 0.0),
            _padded(links.right_grid, 0.0),
            _padded(links.left_value, 0.0),
            _padded(links.right_value, 0.0),
            _padded(links.left_policy, 0.0),
            _padded(links.right_policy, 0.0),
            _padded(links.left_marginal, 0.0),
            _padded(links.right_marginal, 0.0),
        ],
        axis=1,
    )

    live = (
        links.live
        if pad == 0
        else jnp.concatenate([links.live, jnp.zeros((pad,), dtype=bool)])
    )
    blocks = columns.reshape(-1, block_size, columns.shape[1])
    live_blocks = live.reshape(-1, block_size)
    dtype = links.left_grid.dtype

    def max_step(
        carry: tuple[FloatND, BoolND],
        block_and_live: tuple[FloatND, BoolND],
    ) -> tuple[tuple[FloatND, BoolND], None]:
        running_max, any_bracket = carry
        block, block_live = block_and_live
        brackets, value_interp, *_ = _block_query_terms(
            block=block, live=block_live, flat=flat
        )
        block_max = jnp.max(jnp.where(brackets, value_interp, -jnp.inf), axis=1)
        return (
            jnp.maximum(running_max, block_max),
            any_bracket | jnp.any(brackets, axis=1),
        ), None

    (running_max, any_bracket), _ = jax.lax.scan(
        max_step,
        (
            jnp.full((n_query,), -jnp.inf, dtype=dtype),
            jnp.zeros((n_query,), dtype=bool),
        ),
        (blocks, live_blocks),
    )
    env_value = jnp.where(any_bracket, running_max, jnp.nan)

    def policy_step(
        carry: tuple[FloatND, FloatND, FloatND],
        block_and_live: tuple[FloatND, BoolND],
    ) -> tuple[tuple[FloatND, FloatND, FloatND], None]:
        best_rank, best_policy, best_marginal = carry
        block, block_live = block_and_live
        brackets, value_interp, policy_interp, marginal_interp, slope, upper = (
            _block_query_terms(block=block, live=block_live, flat=flat)
        )
        near_max = brackets & (
            value_interp
            >= env_value[:, None] - _value_tie_band(value_interp, env_value[:, None])
        )
        rank = _right_continuous_rank(
            near_max=near_max, right_available=flat[:, None] < upper, slope=slope
        )
        winner = jnp.argmax(rank, axis=1)[:, None]
        block_rank = jnp.take_along_axis(rank, winner, axis=1)[:, 0]
        block_policy = jnp.take_along_axis(policy_interp, winner, axis=1)[:, 0]
        block_marginal = jnp.take_along_axis(marginal_interp, winner, axis=1)[:, 0]
        take = block_rank > best_rank
        return (
            jnp.where(take, block_rank, best_rank),
            jnp.where(take, block_policy, best_policy),
            jnp.where(take, block_marginal, best_marginal),
        ), None

    (_, env_policy_flat, env_marginal_flat), _ = jax.lax.scan(
        policy_step,
        (
            jnp.full((n_query,), -jnp.inf, dtype=dtype),
            jnp.full((n_query,), jnp.nan, dtype=dtype),
            jnp.full((n_query,), jnp.nan, dtype=dtype),
        ),
        (blocks, live_blocks),
    )
    env_policy = jnp.where(any_bracket, env_policy_flat, jnp.nan)
    env_marginal = jnp.where(any_bracket, env_marginal_flat, jnp.nan)
    return (
        env_value.reshape(query.shape),
        env_policy.reshape(query.shape),
        env_marginal.reshape(query.shape),
    )
