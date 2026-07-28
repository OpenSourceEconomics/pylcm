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
itself the memory wall; `segment_block_size` swaps it for a single-pass blocked
scan over segment blocks, which peaks at `(n_query, block)` instead of
`(n_query, n_segment)` and returns the identical result.

## Winner selection: certified comparison, no tie band (round-5 audit)

Earlier revisions selected the winning branch through a tie band proportional
to the operand magnitude (`64*eps*(|left| + |r|*(|right| + |left|))`). That is
a heuristic on the size of the data, not a bound on the arithmetic actually
performed: when the query coincides with a stored node the candidate value IS a
stored float — zero rounding error — yet the band still scaled with `|value|`.
Consequences (round-5 audit F2): genuinely distinct stored values were declared
tied; adding a common constant to every branch flipped a strict winner
(translation variance); and the published value and policy could come from
DIFFERENT branches. That selector is replaced by a certified comparison:

1. **Node events are exact.** When the query equals a stored endpoint (or the
   segment has zero width) the candidate value/policy/marginal are the stored
   floats themselves — no arithmetic, certified rounding radius zero — and they
   are published and compared exactly, with no tolerance whatsoever.
2. **Interior queries are evaluated compensated.** The interpolant
   `left + (t/w)*d` is evaluated in double-double arithmetic built from
   error-free transforms (TwoSum/TwoDiff and Dekker's TwoProd), yielding a
   value pair `(hi, lo)` with relative accuracy O(eps^2) and a certified
   residual radius `_INTERIOR_RADIUS_ULPS2 * eps^2 * (|left| + |r*d|)` derived
   from the operations performed. The radius is exactly zero at node events and
   second-order small in the interior — it tracks the arithmetic, not the
   operand magnitude.
3. **Unresolved overlaps follow a documented deterministic rule.** The
   double-double evaluation IS the recomputation at higher precision: any two
   candidates whose working-precision values would be ambiguous (within an ULP
   or two) are ranked on their `(hi, lo)` pairs, compared lexicographically and
   exactly. Two certified intervals that still overlap without being bitwise
   equal — a true gap below ~2x the eps^2-level radius, sub-representable in
   the stored data — are resolved by that same deterministic `(hi, lo)` order,
   NEVER by right-continuity. A bracketed query whose candidates cannot be
   ranked at all (non-finite interior arithmetic, e.g. infinite endpoint
   values) fails loud: all three outputs are NaN.
4. **One winner index.** Each query selects a single winning candidate column
   and gathers value, policy, AND marginal from that same column, so a mixed
   A-value/B-policy result is structurally impossible.
5. **Right-continuity applies only to exact ties.** Only candidates whose
   `(hi, lo)` pairs are exactly equal to the maximum enter the right-continuous
   tie-break (prefer a segment extending strictly right of the query, then the
   larger value-slope, then the earliest candidate) — which is that rule's
   actual purpose: choosing among branches that genuinely attain the same
   value, matching the kernel's `side="right"` read.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, FloatND

# ----------------------------------------------------------------------------
# Error-free transforms and double-double arithmetic.
#
# All helpers are branch-free elementwise jnp arithmetic (JIT/vmap-safe, no
# data-dependent control flow) and rely only on IEEE-754 round-to-nearest.
# Dekker's split constant `2**ceil(p/2) + 1` overflows for inputs within a
# factor ~2**ceil(p/2) of the dtype maximum; envelope value/grid data live far
# below that regime.
# ----------------------------------------------------------------------------


def _dekker_split_factor(dtype: jnp.dtype) -> float:
    """Dekker splitting constant `2**ceil(p/2) + 1` for a p-bit significand."""
    return float(2 ** ((jnp.finfo(dtype).nmant + 2) // 2) + 1)


def _two_sum(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """`a + b` as `(fl(a + b), exact residual)` — Knuth's TwoSum."""
    s = a + b
    t = s - a
    return s, (a - (s - t)) + (b - t)


def _fast_two_sum(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """TwoSum for pre-ordered operands (`|a| >= |b|` or `a == 0`)."""
    s = a + b
    return s, b - (s - a)


def _two_diff(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """`a - b` as `(fl(a - b), exact residual)`."""
    d = a - b
    t = d - a
    return d, (a - (d - t)) - (b + t)


def _two_prod(a: FloatND, b: FloatND, split: float) -> tuple[FloatND, FloatND]:
    """`a * b` as `(fl(a * b), exact residual)` via Dekker splitting."""
    p = a * b
    ca = split * a
    a_hi = ca - (ca - a)
    a_lo = a - a_hi
    cb = split * b
    b_hi = cb - (cb - b)
    b_lo = b - b_hi
    return p, ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo


def _dd_add_fp(ah: FloatND, al: FloatND, b: FloatND) -> tuple[FloatND, FloatND]:
    """Double-double plus float, renormalised."""
    sh, sl = _two_sum(ah, b)
    return _fast_two_sum(sh, sl + al)


def _dd_add(
    ah: FloatND, al: FloatND, bh: FloatND, bl: FloatND
) -> tuple[FloatND, FloatND]:
    """Double-double plus double-double (accurate variant)."""
    sh, sl = _two_sum(ah, bh)
    th, tl = _two_sum(al, bl)
    sh, sl = _fast_two_sum(sh, sl + th)
    return _fast_two_sum(sh, sl + tl)


def _dd_mul_fp(
    ah: FloatND, al: FloatND, b: FloatND, split: float
) -> tuple[FloatND, FloatND]:
    """Double-double times float."""
    ph, pl = _two_prod(ah, b, split)
    return _fast_two_sum(ph, pl + al * b)


def _dd_mul(
    ah: FloatND, al: FloatND, bh: FloatND, bl: FloatND, split: float
) -> tuple[FloatND, FloatND]:
    """Double-double times double-double."""
    ph, pl = _two_prod(ah, bh, split)
    return _fast_two_sum(ph, pl + (ah * bl + al * bh))


def _dd_div(
    ah: FloatND, al: FloatND, bh: FloatND, bl: FloatND, split: float
) -> tuple[FloatND, FloatND]:
    """Double-double division: long division with two residual corrections."""
    q1 = ah / bh
    th, tl = _dd_mul_fp(bh, bl, q1, split)
    rh, rl = _dd_add(ah, al, -th, -tl)
    q2 = rh / bh
    th, tl = _dd_mul_fp(bh, bl, q2, split)
    rh, rl = _dd_add(rh, rl, -th, -tl)
    q3 = rh / bh
    qh, ql = _fast_two_sum(q1, q2)
    return _dd_add_fp(qh, ql, q3)


# Certified residual radius of the compensated interior evaluation, in units of
# `eps**2` times the operand scale `|left_value| + |r*d|` of the final
# accumulation. The interior value runs one dd division, one dd multiply, and
# one dd add on exactly-represented dd inputs (TwoDiff residuals are exact);
# each contributes at most a few eps^2 of its operand scale, so 16 is a
# conservative envelope. The radius only certifies the "unresolved" overlap
# class (module docstring, point 3): the selection itself compares the exact
# `(hi, lo)` pairs, and the radius bounds how far the true interpolant can sit
# from them. It is zero — not merely small — at node events.
_INTERIOR_RADIUS_ULPS2 = 16.0


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


class _CandidateTerms(NamedTuple):
    """Per-(query, segment) candidate quantities, each `(n_query, n_block)`.

    `value_hi/value_lo` is the certified double-double candidate value (stored
    floats with `value_lo == 0` at node events, compensated interpolation in the
    interior) and `radius` its certified residual rounding radius (zero at node
    events, O(eps^2) interior). `policy`/`marginal` are the candidate's outputs
    at the query, `slope` its value-slope, and `right_available` whether it
    extends strictly right of the query — the right-continuous tie-break keys.
    """

    brackets: BoolND
    value_hi: FloatND
    value_lo: FloatND
    radius: FloatND
    policy: FloatND
    marginal: FloatND
    slope: FloatND
    right_available: BoolND


def _candidate_terms(*, block: FloatND, live: BoolND, flat: Float1D) -> _CandidateTerms:
    """Evaluate one block of segments against every query, with certification.

    `block` is a `(n_block, 8)` slice of the stacked link endpoint columns and
    `live` its `(n_block,)` live-flag slice. Both the dense reduction (which
    passes the whole link set as one block) and the blocked scan call this, so
    every per-lane quantity is computed by the very same expressions and the
    two paths select from identical candidates.

    Node events (query equal to a stored endpoint, or a zero-width segment) are
    published as the stored floats with zero certified radius — requirement (1)
    of the selection architecture. Interior lanes are evaluated compensated in
    double-double with an eps^2-level certified radius — requirement (2). A
    zero-width segment carrying a value jump publishes its higher end (its
    lower end still competes through that endpoint's zero-width self-bracket),
    matching the host oracle's vertical-edge rule.
    """
    left_grid, right_grid = block[:, 0][None, :], block[:, 1][None, :]
    left_value, right_value = block[:, 2][None, :], block[:, 3][None, :]
    left_policy, right_policy = block[:, 4][None, :], block[:, 5][None, :]
    left_marginal, right_marginal = block[:, 6][None, :], block[:, 7][None, :]
    q = flat[:, None]

    lower = jnp.minimum(left_grid, right_grid)
    upper = jnp.maximum(left_grid, right_grid)
    brackets = live[None, :] & (q >= lower) & (q <= upper)

    # Node events: the candidate value IS stored data; no arithmetic, radius 0.
    at_left = q == left_grid
    at_right = q == right_grid
    node = at_left | at_right
    # A zero-width segment sets both flags; publish its higher end (left on a
    # value tie, matching the oracle's vertical-edge rule).
    use_right = (at_right & ~at_left) | (
        at_left & at_right & (right_value > left_value)
    )

    # Interior: compensated interpolation `left + (t/w)*d` in double-double.
    # TwoDiff makes t, w, d exact dd representations; division and product are
    # dd-accurate, so (hi, lo) carries the interpolant to O(eps^2) relative.
    zero_width = right_grid == left_grid
    split = _dekker_split_factor(block.dtype)
    th, tl = _two_diff(q, left_grid)
    wh, wl = _two_diff(right_grid, left_grid)
    wh = jnp.where(zero_width, jnp.ones_like(wh), wh)
    wl = jnp.where(zero_width, jnp.zeros_like(wl), wl)
    dh, dl = _two_diff(right_value, left_value)
    rh, rl = _dd_div(th, tl, wh, wl, split)
    ph, pl = _dd_mul(rh, rl, dh, dl, split)
    vh, vl = _dd_add_fp(ph, pl, left_value)

    eps = jnp.finfo(block.dtype).eps
    interior_radius = (
        _INTERIOR_RADIUS_ULPS2 * eps * eps * (jnp.abs(left_value) + jnp.abs(ph))
    )

    zero = jnp.zeros_like(vh)
    return _CandidateTerms(
        brackets=brackets,
        value_hi=jnp.where(node, jnp.where(use_right, right_value, left_value), vh),
        value_lo=jnp.where(node, zero, vl),
        radius=jnp.where(node, zero, interior_radius),
        policy=jnp.where(
            node,
            jnp.where(use_right, right_policy, left_policy),
            left_policy + rh * (right_policy - left_policy),
        ),
        marginal=jnp.where(
            node,
            jnp.where(use_right, right_marginal, left_marginal),
            left_marginal + rh * (right_marginal - left_marginal),
        ),
        slope=dh / jnp.where(zero_width, jnp.ones_like(wh), right_grid - left_grid),
        right_available=q < upper,
    )


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
        winning segment's marginal at each query, each shaped like `x_query`
        and all gathered from the same winning segment. A query no live segment
        brackets yields NaN in all three.
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
    # exact-tie break, so multi-point chains and their interpolation are
    # unchanged; a self-bracket wins only where nothing brackets the query from
    # the right.
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
    columns = jnp.stack(
        [
            links.left_grid,
            links.right_grid,
            links.left_value,
            links.right_value,
            links.left_policy,
            links.right_policy,
            links.left_marginal,
            links.right_marginal,
        ],
        axis=1,
    )

    query = jnp.asarray(x_query)
    n_segment = links.live.shape[0]
    if 0 < segment_block_size < n_segment:
        return _envelope_at_query_blocked(
            columns=columns, live=links.live, query=query, block_size=segment_block_size
        )

    flat = query.reshape(-1)
    terms = _candidate_terms(block=columns, live=links.live, flat=flat)

    # Certified lexicographic maximum over the exact (hi, lo) value pairs. The
    # comparisons are exact float comparisons — no tolerance, no band — so a
    # strict represented gap always selects the higher branch and a common
    # translation cannot flip a winner.
    masked_hi = jnp.where(terms.brackets, terms.value_hi, -jnp.inf)
    max_hi = jnp.max(masked_hi, axis=1, keepdims=True)
    hi_tied = terms.brackets & (masked_hi == max_hi)
    masked_lo = jnp.where(hi_tied, terms.value_lo, -jnp.inf)
    max_lo = jnp.max(masked_lo, axis=1, keepdims=True)
    exact_tie = hi_tied & (masked_lo == max_lo)

    # Right-continuous break among the exactly-tied candidates only, then ONE
    # winner index per query; value, policy, and marginal are gathered from that
    # same index (requirements 4 and 5).
    _, tie_key = _tie_break_slope_key(
        tied=exact_tie, right_available=terms.right_available, slope=terms.slope
    )
    best = jnp.argmax(tie_key, axis=1)[:, None]
    any_bracket = jnp.any(terms.brackets, axis=1)
    # A bracketed query whose candidates cannot be ranked — a non-finite
    # certified value pair (infinite endpoint arithmetic), which also empties
    # the exact-tie set — fails loud with NaN (requirement 3). The explicit
    # `poisoned` flag keeps the dense and blocked paths on the same rule.
    resolved = jnp.any(exact_tie, axis=1)
    poisoned = jnp.any(
        terms.brackets & (jnp.isnan(terms.value_hi) | jnp.isnan(terms.value_lo)),
        axis=1,
    )
    ok = any_bracket & resolved & ~poisoned
    env_value = jnp.where(
        ok, jnp.take_along_axis(terms.value_hi, best, axis=1)[:, 0], jnp.nan
    )
    env_policy = jnp.where(
        ok, jnp.take_along_axis(terms.policy, best, axis=1)[:, 0], jnp.nan
    )
    env_marginal = jnp.where(
        ok, jnp.take_along_axis(terms.marginal, best, axis=1)[:, 0], jnp.nan
    )
    return (
        env_value.reshape(query.shape),
        env_policy.reshape(query.shape),
        env_marginal.reshape(query.shape),
    )


def _tie_break_slope_key(
    *, tied: BoolND, right_available: BoolND, slope: FloatND
) -> tuple[BoolND, FloatND]:
    """Per-query eligibility flag + per-segment slope key for the exact-tie break.

    Applies ONLY to the exactly-tied candidates (`tied`: certified-equal
    `(hi, lo)` value pairs — requirement 5). Implements "prefer a right-extending
    tied segment, else the largest tied slope" WITHOUT folding the two keys into
    one scalar. Folding the right-extends bit and the slope into a single float
    (an `arctan(slope)/pi + right_available` rank) loses the slope bits for
    near-equal small slopes in float32, so two genuinely-distinct slopes round to
    the same rank and `argmax` falls back to the lower index — the wrong branch
    (round-4 audit F2, second half). Instead: among the tied segments, if ANY
    extends strictly right, only those compete; else all tied compete. The
    returned `key` is the raw `slope` for the competing segments and `-inf`
    otherwise, so `argmax(key)` compares slopes at native precision. Ties on the
    slope itself resolve to the earliest candidate. `any_eligible` (per query) is
    also returned so the blocked scan can reconcile the global right-extends
    priority across blocks lexicographically; the dense path sees every segment
    on one axis and argmaxes the key directly.
    """
    eligible = tied & right_available
    any_eligible = jnp.any(eligible, axis=1, keepdims=True)
    compete = jnp.where(any_eligible, eligible, tied)
    key = jnp.where(compete, slope, -jnp.inf)
    return any_eligible[:, 0], key


class _BlockedCarry(NamedTuple):
    """Running winner of the blocked scan, one entry per query.

    The full selection rule is one lexicographic maximum over the candidate key
    `(value_hi, value_lo, right_available, slope, earliest)`, so a single scan
    can carry the current winner's key components together with its gathered
    value/policy/marginal. Every candidate is evaluated exactly once, inside
    one compiled scan body; no quantity is ever recomputed in a second program
    and compared for equality (XLA is free to fuse each lowering differently at
    the bit level, so a cross-program exact-equality rendezvous would be
    unsound — the round-5 rewrite's first blocked draft failed exactly there).
    """

    any_bracket: BoolND
    poisoned: BoolND
    hi: FloatND
    lo: FloatND
    right_extending: BoolND
    slope: FloatND
    value: FloatND
    policy: FloatND
    marginal: FloatND


def _envelope_at_query_blocked(
    *, columns: FloatND, live: BoolND, query: FloatND, block_size: int
) -> tuple[FloatND, FloatND, FloatND]:
    """Single-scan blocked equivalent of the dense `(n_query, n_segment)` path.

    Evaluates every candidate through the shared `_candidate_terms` and reduces
    with the same lexicographic rule as the dense path — certified `(hi, lo)`
    value pairs first, then (only on an exact tie) right-extension and value
    slope. Within a block the winner is found exactly as in the dense reduction
    (`_tie_break_slope_key` restricted to the block's exactly-tied candidates)
    and its value/policy/marginal are gathered from the SAME within-block
    index. Across blocks the carried winner is updated by strict lexicographic
    comparison of the carried keys, which keeps the earliest winner and so
    matches the dense `argmax`. Lexicographic maximum is associative, so the
    scan order cannot change the result.

    The links are padded to a multiple of `block_size` with dead segments
    (which never bracket); the scan peaks at `(n_query, block_size)` working
    memory. A bracketed query with a non-finite candidate value pair (infinite
    endpoint arithmetic) is `poisoned` and fails loud with NaN in all three
    outputs, matching the dense path's empty-exact-tie rule.
    """
    flat = query.reshape(-1)
    n_query = flat.shape[0]
    n_segment = live.shape[0]
    pad = (-n_segment) % block_size
    if pad:
        columns = jnp.concatenate(
            [columns, jnp.zeros((pad, columns.shape[1]), dtype=columns.dtype)]
        )
        live = jnp.concatenate([live, jnp.zeros((pad,), dtype=bool)])
    blocks = columns.reshape(-1, block_size, columns.shape[1])
    live_blocks = live.reshape(-1, block_size)
    dtype = columns.dtype

    def step(
        carry: _BlockedCarry,
        block_and_live: tuple[FloatND, BoolND],
    ) -> tuple[_BlockedCarry, None]:
        block, block_live = block_and_live
        t = _candidate_terms(block=block, live=block_live, flat=flat)
        # Within-block winner: identical construction to the dense reduction.
        masked_hi = jnp.where(t.brackets, t.value_hi, -jnp.inf)
        block_hi = jnp.max(masked_hi, axis=1)
        hi_tied = t.brackets & (masked_hi == block_hi[:, None])
        masked_lo = jnp.where(hi_tied, t.value_lo, -jnp.inf)
        block_lo = jnp.max(masked_lo, axis=1)
        tied = hi_tied & (masked_lo == block_lo[:, None])
        block_ra, key = _tie_break_slope_key(
            tied=tied, right_available=t.right_available, slope=t.slope
        )
        winner = jnp.argmax(key, axis=1)[:, None]
        block_slope = jnp.take_along_axis(key, winner, axis=1)[:, 0]
        block_value = jnp.take_along_axis(t.value_hi, winner, axis=1)[:, 0]
        block_policy = jnp.take_along_axis(t.policy, winner, axis=1)[:, 0]
        block_marginal = jnp.take_along_axis(t.marginal, winner, axis=1)[:, 0]
        # Strict lexicographic update on (hi, lo, right-extending, slope); the
        # strict comparisons keep the earliest winner, matching the dense
        # `argmax`, and every comparison is between values computed once inside
        # this same compiled body. NaN keys compare false everywhere, so a
        # poisoned block never takes; `poisoned` masks the query at the end.
        value_tie = (block_hi == carry.hi) & (block_lo == carry.lo)
        take = (
            (block_hi > carry.hi)
            | ((block_hi == carry.hi) & (block_lo > carry.lo))
            | (
                value_tie
                & (
                    (block_ra & ~carry.right_extending)
                    | (
                        (block_ra == carry.right_extending)
                        & (block_slope > carry.slope)
                    )
                )
            )
        )
        return _BlockedCarry(
            any_bracket=carry.any_bracket | jnp.any(t.brackets, axis=1),
            poisoned=carry.poisoned
            | jnp.any(
                t.brackets & (jnp.isnan(t.value_hi) | jnp.isnan(t.value_lo)), axis=1
            ),
            hi=jnp.where(take, block_hi, carry.hi),
            lo=jnp.where(take, block_lo, carry.lo),
            right_extending=jnp.where(take, block_ra, carry.right_extending),
            slope=jnp.where(take, block_slope, carry.slope),
            value=jnp.where(take, block_value, carry.value),
            policy=jnp.where(take, block_policy, carry.policy),
            marginal=jnp.where(take, block_marginal, carry.marginal),
        ), None

    final, _ = jax.lax.scan(
        step,
        _BlockedCarry(
            any_bracket=jnp.zeros((n_query,), dtype=bool),
            poisoned=jnp.zeros((n_query,), dtype=bool),
            hi=jnp.full((n_query,), -jnp.inf, dtype=dtype),
            lo=jnp.full((n_query,), -jnp.inf, dtype=dtype),
            right_extending=jnp.zeros((n_query,), dtype=bool),
            slope=jnp.full((n_query,), -jnp.inf, dtype=dtype),
            value=jnp.full((n_query,), jnp.nan, dtype=dtype),
            policy=jnp.full((n_query,), jnp.nan, dtype=dtype),
            marginal=jnp.full((n_query,), jnp.nan, dtype=dtype),
        ),
        (blocks, live_blocks),
    )
    ok = final.any_bracket & ~final.poisoned
    env_value = jnp.where(ok, final.value, jnp.nan)
    env_policy = jnp.where(ok, final.policy, jnp.nan)
    env_marginal = jnp.where(ok, final.marginal, jnp.nan)
    return (
        env_value.reshape(query.shape),
        env_policy.reshape(query.shape),
        env_marginal.reshape(query.shape),
    )
