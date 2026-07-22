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
# The band MUST bound the interpolation ROUNDING ERROR, which is set by the
# endpoint OPERANDS, not by the interpolated output magnitude (round-3 audit F6
# opened this as DC-1; round-4 audit F2 reopened it). `value = left + r*(right -
# left)` rounds with error ~ eps*(|left| + |r|*(|right| + |left|)); near a
# crossing the output cancels to ~0, so scaling the band by `max(|a|, |b|)` of
# the OUTPUTS collapses it to ~0 and excludes the mathematically tied segment
# with the larger right-hand slope — the wrong branch then wins. Scale the band
# to the operands instead (`_interp_error_scale`), and compare against the max
# segment's own operand error too, since `max_value - value` carries rounding on
# both sides. This supersedes the earlier output-magnitude band
# (`64*eps*max(|a|,|b|)`) and, before it, an interim `1e-12*max(1,|ref|)` band —
# both precision-/cancellation-blind, exactly what F6/F2 flag.
_TIE_BAND_ULPS = 64.0


def _interp_error_scale(
    left_value: FloatND, right_value: FloatND, relative: FloatND
) -> FloatND:
    """Operand-scaled bound on the linear-interpolation rounding error.

    For `value = left + relative*(right - left)`, the rounding error tracks the
    endpoint magnitudes (and the `right - left` cancellation), NOT the possibly-
    cancelled result. Multiply by `_TIE_BAND_ULPS * eps(dtype)` at the call site
    to get the tie half-width (DC-1 floor).
    """
    return jnp.abs(left_value) + jnp.abs(relative) * (
        jnp.abs(right_value) + jnp.abs(left_value)
    )


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
    # just to the right" is meaningful), and among those the larger slope. Only at
    # the global upper endpoint, where nothing continues right, fall back to the
    # largest near-max slope. `_tie_break_slope_key` compares the slope at native
    # precision so this dense reduction and the blocked scan select the same winner.
    slope = (right_value - left_value)[None, :] / safe_width
    # Operand-scaled tie band (round-4 audit F2, DC-1): scale to the endpoint
    # operands, not the interpolated output (which cancels near a crossing), and
    # add the max segment's own error since both sides of `max_value - value`
    # round.
    eps = jnp.finfo(value_interp.dtype).eps
    err = (
        _TIE_BAND_ULPS
        * eps
        * _interp_error_scale(left_value[None, :], right_value[None, :], relative)
    )
    err_at_max = jnp.take_along_axis(
        err, jnp.argmax(masked_value, axis=1)[:, None], axis=1
    )
    near_max = brackets & (masked_value >= max_value - (err + err_at_max))
    _, tie_key = _tie_break_slope_key(
        near_max=near_max, right_available=flat < upper, slope=slope
    )
    best = jnp.argmax(tie_key, axis=1)
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


def _tie_break_slope_key(
    *, near_max: BoolND, right_available: BoolND, slope: FloatND
) -> tuple[BoolND, FloatND]:
    """Per-query eligibility flag + per-segment slope key for the tie-break.

    Implements "prefer a right-extending near-max segment, else the largest near-max
    slope" WITHOUT folding the two keys into one scalar. Folding the right-extends
    bit and the slope into a single float (an `arctan(slope)/pi + right_available`
    rank) loses the slope bits for near-equal small slopes in float32, so two
    genuinely-distinct slopes round to the same rank and `argmax` falls back to the
    lower index — the wrong branch (round-4 audit F2, second half). Instead: among
    the near-max segments, if ANY extends strictly right, only those compete; else
    all near-max compete. The returned `key` is the raw `slope` for the competing
    segments and `-inf` otherwise, so `argmax(key)` compares slopes at native
    precision. `any_eligible` (per query) is also returned so the blocked scan can
    reconcile the global right-extends priority across blocks lexicographically:
    the dense path sees every segment on one axis and argmaxes the key directly.
    """
    eligible = near_max & right_available
    any_eligible = jnp.any(eligible, axis=1, keepdims=True)
    compete = jnp.where(any_eligible, eligible, near_max)
    key = jnp.where(compete, slope, -jnp.inf)
    return any_eligible[:, 0], key


def _block_query_terms(
    *, block: FloatND, live: BoolND, flat: Float1D
) -> tuple[BoolND, FloatND, FloatND, FloatND, FloatND, FloatND, FloatND]:
    """Bracket-and-interpolate one segment block against every query.

    `block` is one `(block_size, 8)` slice of the stacked link endpoint columns
    and `live` its `(block_size,)` live-flag slice. Returns the
    `(n_query, block_size)` bracket mask; the value, policy, marginal, and
    value-slope interpolated at each query for each link in the block; the
    link's upper endpoint (for the right-continuous tie-break); and the
    operand-scaled interpolation error scale (for the DC-1 tie band) — the same
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
    error_scale = _interp_error_scale(
        left_value[None, :], right_value[None, :], relative
    )
    return (
        brackets,
        value_interp,
        policy_interp,
        marginal_interp,
        slope,
        upper,
        error_scale,
    )


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
      the operand-scaled tie band (`_interp_error_scale`, plus the max
      segment's own error tracked in pass 1) of that (now fixed) envelope value,
      keeps the right-continuous winner (`_tie_break_slope_key`: a right-extending
      near-max segment over one ending at the query, then larger value-slope) — the
      dense path's tie-break. The scan carries the two keys separately —
      `(has_eligible, slope)` — and reconciles them across blocks lexicographically,
      so the right-extends priority is global while the slope stays at native
      precision; the strict cross-block `>` keeps the earliest winner, matching the
      dense `argmax`.

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
        carry: tuple[FloatND, FloatND, BoolND],
        block_and_live: tuple[FloatND, BoolND],
    ) -> tuple[tuple[FloatND, FloatND, BoolND], None]:
        running_max, running_max_scale, any_bracket = carry
        block, block_live = block_and_live
        brackets, value_interp, _, _, _, _, error_scale = _block_query_terms(
            block=block, live=block_live, flat=flat
        )
        block_masked = jnp.where(brackets, value_interp, -jnp.inf)
        block_max = jnp.max(block_masked, axis=1)
        # Track the operand error scale OF the running-max segment (round-4
        # audit F2): the tie band needs the max side's rounding error, and the
        # max segment can live in any block. A cross-block tie keeps the earlier
        # block's scale, matching the dense `argmax` (first max wins).
        block_argmax = jnp.argmax(block_masked, axis=1)[:, None]
        block_max_scale = jnp.take_along_axis(error_scale, block_argmax, axis=1)[:, 0]
        take = block_max > running_max
        return (
            jnp.where(take, block_max, running_max),
            jnp.where(take, block_max_scale, running_max_scale),
            any_bracket | jnp.any(brackets, axis=1),
        ), None

    (running_max, env_max_scale, any_bracket), _ = jax.lax.scan(
        max_step,
        (
            jnp.full((n_query,), -jnp.inf, dtype=dtype),
            jnp.zeros((n_query,), dtype=dtype),
            jnp.zeros((n_query,), dtype=bool),
        ),
        (blocks, live_blocks),
    )
    env_value = jnp.where(any_bracket, running_max, jnp.nan)

    def policy_step(
        carry: tuple[BoolND, FloatND, FloatND, FloatND],
        block_and_live: tuple[FloatND, BoolND],
    ) -> tuple[tuple[BoolND, FloatND, FloatND, FloatND], None]:
        best_has_elig, best_slope, best_policy, best_marginal = carry
        block, block_live = block_and_live
        (
            brackets,
            value_interp,
            policy_interp,
            marginal_interp,
            slope,
            upper,
            error_scale,
        ) = _block_query_terms(block=block, live=block_live, flat=flat)
        # Operand-scaled tie band with the max side's error (round-4 audit F2),
        # identical scale to the dense path so both select the same branch.
        eps = jnp.finfo(value_interp.dtype).eps
        err = _TIE_BAND_ULPS * eps * error_scale
        err_at_max = _TIE_BAND_ULPS * eps * env_max_scale[:, None]
        near_max = brackets & (value_interp >= env_value[:, None] - (err + err_at_max))
        block_has_elig, key = _tie_break_slope_key(
            near_max=near_max, right_available=flat[:, None] < upper, slope=slope
        )
        winner = jnp.argmax(key, axis=1)[:, None]
        block_slope = jnp.take_along_axis(key, winner, axis=1)[:, 0]
        block_policy = jnp.take_along_axis(policy_interp, winner, axis=1)[:, 0]
        block_marginal = jnp.take_along_axis(marginal_interp, winner, axis=1)[:, 0]
        # Cross-block lexicographic on (has_eligible desc, slope desc): a block
        # with a right-extending near-max beats one without; within the same class
        # the larger value-slope wins, the strict `>` keeping the earliest winner
        # to match the dense `argmax`. Comparing the slope directly preserves its
        # native precision (round-4 audit F2, second half).
        take = (block_has_elig & ~best_has_elig) | (
            (block_has_elig == best_has_elig) & (block_slope > best_slope)
        )
        return (
            jnp.where(take, block_has_elig, best_has_elig),
            jnp.where(take, block_slope, best_slope),
            jnp.where(take, block_policy, best_policy),
            jnp.where(take, block_marginal, best_marginal),
        ), None

    (_, _, env_policy_flat, env_marginal_flat), _ = jax.lax.scan(
        policy_step,
        (
            jnp.zeros((n_query,), dtype=bool),
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
