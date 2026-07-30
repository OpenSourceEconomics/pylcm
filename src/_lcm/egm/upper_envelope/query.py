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
3. **What the radii cannot separate is decided EXACTLY.** A double-double pair
   is more accurate than a float, but it is not a canonical representation of
   the rational interpolant: two algebraically different segments that attain
   the same exact value at the query need not produce the same low word. So
   ordering candidates lexicographically on `(hi, lo)` reads rounding residue
   as strict order, declares a genuine tie a strict win, and skips the
   right-continuous rule that tie exists to trigger — publishing the wrong
   branch's policy and marginal while the value looks right (round-6 audit F2;
   the round-5 selector did exactly this). The certified radii are therefore
   consumed, not merely computed: candidates whose certified interval falls
   below the leader's are certified losers and discarded, and everything the
   radii leave overlapping is settled by `_exact_compare`, an exact
   cross-multiplied sign test over error-free products. A comparison thus
   returns a certified strict sign or a TRUE equality — never an ordering
   invented by residue. A bracketed query whose candidates cannot be ranked at
   all (non-finite interior arithmetic, e.g. infinite endpoint values) fails
   loud: all three outputs are NaN.
4. **One winner index.** Each query selects a single winning candidate column
   and gathers value, policy, AND marginal from that same column, so a mixed
   A-value/B-policy result is structurally impossible.
5. **Right-continuity applies only to exact ties, and it too is exact.** Only
   candidates whose values are EXACTLY equal to the maximum enter the
   right-continuous tie-break (prefer a segment extending strictly right of the
   query, then the larger value-slope, then the earliest candidate) — that
   rule's actual purpose: choosing among branches that genuinely attain the
   same value, matching the kernel's `side="right"` read. The slope half is
   settled by the same exact cross-multiplied predicate as the value, because
   an exact value tie followed by a ROUNDED slope key rebuilds the defect one
   operation later: two strictly ordered exact slopes can share one float key,
   `argmax` then falls back to candidate order, and permuting the branches
   flips the published policy and marginal (round-7 audit F2). Candidate order
   decides only on EXACT slope equality.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, FloatND, LoopIndex

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

# Screen slack on the certified radii. The overlap test compares a
# double-double difference against the sum of two certified radii, and forming
# that difference is itself two roundings; widening by a small factor keeps the
# screen a guaranteed SUPERSET of the candidates that can still win. Widening
# only ever adds exact comparisons — it can never drop a winner — so this is a
# performance knob, not a correctness tolerance.
_SCREEN_SLACK = 4.0

# Exact-comparison expansion width. A candidate's numerator is four error-free
# products (eight floats) and its denominator one error-free difference (two);
# cross-multiplying the two candidates therefore yields 2 * 8 * 2 * 2 = 64
# exactly-representable terms whose sum has the sign of `V_a - V_b`.
_EXACT_TERMS = 64

# Screen slack on the ROUNDED slope key, in ULPs. The native slope is one
# division, so a gap wider than a few ULP of the operand scale is already
# certified by it; anything closer goes to the exact slope predicate. As with
# the value screen, generous is safe — a superset only costs comparisons.
_SLOPE_SCREEN_ULPS = 8.0


def _one_hot(index: jax.Array, width: int) -> BoolND:
    """Row-wise indicator of `index` over `width` columns."""
    return jnp.arange(width) == index[:, None]


def _node_selection(
    *,
    q: FloatND,
    left_grid: FloatND,
    right_grid: FloatND,
    left_value: FloatND,
    right_value: FloatND,
) -> tuple[BoolND, BoolND]:
    """Node-event flag and which stored endpoint the candidate publishes.

    Shared by the double-double evaluation and the exact comparison so the two
    can never disagree about which lane is a node or which stored float that
    node publishes.
    """
    at_left = q == left_grid
    at_right = q == right_grid
    # A zero-width segment sets both flags; publish its higher end (left on a
    # value tie, matching the oracle's vertical-edge rule).
    use_right = (at_right & ~at_left) | (
        at_left & at_right & (right_value > left_value)
    )
    return at_left | at_right, use_right


class _ExactRatio(NamedTuple):
    """A candidate's value at the query as an exact rational `num / den`.

    `numerator` and `denominator` are unevaluated sums of exactly-representable
    floats — error-free transforms, so no information has been discarded. The
    denominator is canonically positive, which lets a cross-multiplied
    comparison read its sign straight off the numerator difference.
    """

    numerator: FloatND
    denominator: FloatND


def _to_safe_binade(ratio: _ExactRatio) -> _ExactRatio:
    """Rescale one candidate's ratio into a binade where cross products are safe.

    An error-free transform is exact only while its results stay representable.
    `_two_prod` is not: at a small enough operand scale the product underflows
    into the subnormal range and its residual is silently lost, and at a large
    enough scale it overflows to infinity and poisons the expansion with NaN.
    The cross-multiplied predicates below then read a strict sign as `0` or as
    NaN, and the caller falls back to candidate order — so a pure power-of-two
    rescaling of an entire model, which changes nothing mathematically, flips
    the published policy and marginal (round-8 audit F2).

    The repair is to divide the problem out rather than to widen a tolerance.
    Scaling a numerator AND its denominator by one common power of two leaves
    the rational `num / den` **identically** unchanged, so it cannot change any
    ordering. Applied per candidate, it also cancels from the comparison
    itself: with `a` scaled by `2**-i` and `b` by `2**-j`, the two cross
    products `N_a*D_b` and `N_b*D_a` both acquire the same factor `2**-(i+j)`,
    so the sign of their difference survives the transformation exactly.

    Choosing the scale from the largest component puts every component at or
    below one, which makes overflow in the subsequent products impossible and
    is exactly the normalization that undoes a uniform rescaling of the model.
    Powers of two scale a float without rounding, so the transform costs no
    accuracy. Zero and non-finite magnitudes have no binade and are passed
    through untouched: a NaN must stay NaN so it surfaces rather than being
    ordered.
    """
    components = jnp.concatenate([ratio.numerator, ratio.denominator], axis=-1)
    exponent = _binade_exponent(jnp.max(jnp.abs(components), axis=-1))[..., None]
    return _ExactRatio(
        numerator=jnp.ldexp(ratio.numerator, -exponent),
        denominator=jnp.ldexp(ratio.denominator, -exponent),
    )


def _binade_exponent(magnitude: FloatND) -> jax.Array:
    """Exponent `e` with `magnitude` in `[2**(e-1), 2**e)`; `0` where there is none.

    Returned as an exponent rather than a factor so callers scale with `ldexp`.
    Materializing `2**-e` as a float would itself be lossy at the extremes: for
    a float32 magnitude near the top of the range `2**-e` is SUBNORMAL, so the
    factor carries fewer bits than the scaling it is meant to perform exactly.
    `ldexp` adjusts the exponent field directly and has no such failure mode.
    """
    _, exponent = jnp.frexp(magnitude)
    usable = (magnitude > 0) & jnp.isfinite(magnitude)
    return jnp.where(usable, exponent, jnp.zeros_like(exponent))


def _mid_range_target(dtype: jnp.dtype) -> int:
    """Binade to normalize a product's operands into: half the exponent range.

    Normalizing the largest operand to `[0.5, 1)` — the obvious choice, and what
    rounds 8 to 10 used — stops products overflowing but spends the entire lower
    half of the exponent range on nothing. Every operand more than `emax` binades
    below the largest is then pushed into the subnormals, where it loses bits, or
    vanishes outright. A segment whose own grid span covers the dynamic range has
    exactly that shape: normalizing `(x1 - x0)` to below one sends `(q - x0)` to
    zero, and the interpolant collapses onto its left value.

    Landing the largest operand mid-range instead leaves both halves of headroom:
    a product of two such operands still cannot overflow (`2**(2*target)` plus
    room for the sum of the terms stays inside the range), while an operand up to
    `target + emax` binades below the largest stays finite NORMAL — 189 binades
    of slack in float32, 1533 in float64, against zero before.
    """
    # Two operands multiply and up to sixteen such products are summed, so leave
    # four binades of headroom above `2 * target`.
    return (int(jnp.finfo(dtype).maxexp) - 4) // 2


def _scale_exponent(magnitude: FloatND, target: int) -> jax.Array:
    """Shift placing `magnitude` in binade `target`; `0` where there is none.

    Zero and non-finite magnitudes are passed through unscaled: a NaN must stay
    NaN so it surfaces rather than being silently ordered.
    """
    _, exponent = jnp.frexp(magnitude)
    usable = (magnitude > 0) & jnp.isfinite(magnitude)
    return jnp.where(usable, exponent - target, jnp.zeros_like(exponent))


def _group_scale_exponent(*magnitudes: FloatND, target: int) -> jax.Array:
    """One shift for a group of quantities that must remain COMPARABLE.

    Taken from the largest magnitude in the group. A single shared power of two
    is the only scaling allowed on quantities whose ORDER is read across group
    members: it multiplies each of them, and hence every difference between them,
    by the same positive constant, so no ordering can change.
    """
    stacked = jnp.stack([jnp.abs(magnitude) for magnitude in magnitudes], axis=-1)
    return _scale_exponent(jnp.max(stacked, axis=-1), target)


def _exact_cross_sign(*, a: _ExactRatio, b: _ExactRatio, dtype: jnp.dtype) -> FloatND:
    """Exact sign of `a - b` for two rationals with positive denominators.

    `a - b = (N_a*D_b - N_b*D_a) / (D_a*D_b)`, so with both denominators
    canonically positive the sign is that of the cross-multiplied numerator
    difference. Every cross product of two exactly-represented components is
    itself split into an exact pair, so the terms sum to that difference with
    no error and `_exact_sign_of_sum` reads its sign exactly.

    Both operands are moved into a safe binade first — see `_to_safe_binade`.
    That step is what makes "exact" mean exact at every model scale rather than
    only at the scale the tests happened to use.

    This is the single ordering kernel: value selection and the
    right-continuous slope tie-break share it, so the two can never disagree
    about what an exact tie is.
    """
    a = _to_safe_binade(a)
    b = _to_safe_binade(b)
    split = _dekker_split_factor(dtype)
    terms: list[FloatND] = []
    for numerator, denominator, orientation in (
        (a.numerator, b.denominator, 1.0),
        (b.numerator, a.denominator, -1.0),
    ):
        for i in range(numerator.shape[-1]):
            for j in range(denominator.shape[-1]):
                # Scaling by +-1 is exact, so negation preserves the transform.
                head, tail = _two_prod(numerator[..., i], denominator[..., j], split)
                terms.extend([orientation * head, orientation * tail])
    return _exact_sign_of_sum(jnp.stack(terms, axis=-1))


def _exact_ratio(
    *, cols: FloatND, q: FloatND, value_exponent: jax.Array
) -> _ExactRatio:
    """Exact rational value of one candidate per query, from its raw columns.

    `V = (v0*(x1 - q) + v1*(q - x0)) / (x1 - x0)` is the same interpolant
    `_candidate_terms` evaluates, but every factor is kept as an error-free
    pair instead of being rounded, so the pair `(numerator, denominator)`
    represents `V` with no error at all. A node event publishes a STORED float,
    so its exact value is that float over one.

    Returned in units of `2**value_exponent`, which the caller supplies because
    it must be SHARED with the competitor: `V_a - V_b` only keeps its sign while
    both are measured in the same units.

    **The differences are formed FIRST, on the RAW columns.** `_two_diff` is
    exact and the subtraction of two finite floats cannot overflow, so nothing
    is lost and nothing needs rescuing. Scaling the columns first — which is
    what the old `_to_safe_columns` did, with one grid exponent covering `q` and
    both candidates' nodes — can map `q` and a node onto the SAME float, after
    which `q - x0` is zero, the interpolant collapses onto its left value, and
    no amount of downstream exactness recovers the difference (round-10 audit
    F2, the same class as the value path's round-11 repair: scale the
    DIFFERENCES, never the operands).

    The grid scale is then PER CANDIDATE and cancels exactly — it multiplies
    this candidate's numerator and denominator by one power of two, leaving the
    rational identically unchanged — so it needs no coordination at all.
    """
    split = _dekker_split_factor(cols.dtype)
    left_grid, right_grid = cols[..., 0], cols[..., 1]
    left_value, right_value = cols[..., 2], cols[..., 3]

    ah, al = _two_diff(right_grid, q)
    bh, bl = _two_diff(q, left_grid)
    dh, dl = _two_diff(right_grid, left_grid)

    grid_exponent = _group_scale_exponent(
        ah, bh, dh, target=_mid_range_target(cols.dtype)
    )
    ah, al, bh, bl, dh, dl = (
        jnp.ldexp(term, -grid_exponent) for term in (ah, al, bh, bl, dh, dl)
    )
    scaled_left_value = jnp.ldexp(left_value, -value_exponent)
    scaled_right_value = jnp.ldexp(right_value, -value_exponent)

    products = (
        _two_prod(scaled_left_value, ah, split),
        _two_prod(scaled_left_value, al, split),
        _two_prod(scaled_right_value, bh, split),
        _two_prod(scaled_right_value, bl, split),
    )
    numerator = jnp.stack([term for pair in products for term in pair], axis=-1)
    denominator = jnp.stack([dh, dl], axis=-1)

    # Canonical orientation. Endpoints may be stored in either order within a
    # branch; negating both numerator and denominator leaves `V` unchanged.
    flip = (dh < 0)[..., None]
    numerator = jnp.where(flip, -numerator, numerator)
    denominator = jnp.where(flip, -denominator, denominator)

    node, use_right = _node_selection(
        q=q,
        left_grid=left_grid,
        right_grid=right_grid,
        left_value=left_value,
        right_value=right_value,
    )
    # Node detection reads the RAW grid and query — an equality that a shared
    # power of two would have preserved anyway, but which is now never disturbed.
    # The published value goes into the same `2**value_exponent` units as the
    # interpolated branch so the two are directly comparable.
    node_value = jnp.ldexp(
        jnp.where(use_right, right_value, left_value), -value_exponent
    )
    zeros = jnp.zeros_like(node_value)
    is_node = node[..., None]
    return _ExactRatio(
        numerator=jnp.where(
            is_node,
            jnp.stack([node_value, *([zeros] * 7)], axis=-1),
            numerator,
        ),
        denominator=jnp.where(
            is_node, jnp.stack([jnp.ones_like(node_value), zeros], axis=-1), denominator
        ),
    )


def _exact_sign_of_sum(terms: FloatND) -> FloatND:
    """Exact sign of `terms.sum(-1)`, with no rounding anywhere.

    Accumulates the terms into a non-overlapping floating-point expansion by
    Shewchuk's GROW-EXPANSION: every TwoSum is exact, so the expansion's sum
    equals the input's sum bit for bit however catastrophically the terms
    cancel. Because the expansion is non-overlapping and increasing in
    magnitude, its highest-index non-zero component dominates the sum of all
    lower ones and therefore carries the sign.

    Slots at or above the current length are zero and `TwoSum(Q, 0) = (Q, 0)`,
    so the running total falls through them untouched and lands in the first
    free slot — which is why a fixed-width buffer suffices for an expansion
    that grows by one component per term.
    """
    n_term = terms.shape[-1]
    expansion = jnp.zeros_like(terms)

    # `LoopIndex`, not an array-only hint: a `jax.Array` annotation here made every
    # EAGER call raise a beartype violation, which is what broke all 24
    # `test_jitted_solve_matches_the_eager_solve[*]` cases — that test is the only
    # one building its reference under `disable_jit`, so it was the first caller to
    # reach this kernel eagerly. See the alias in `lcm.typing` for why both forms
    # must be admitted.
    def absorb_term(k: LoopIndex, expansion: FloatND) -> FloatND:
        term = jax.lax.dynamic_index_in_dim(terms, k, axis=-1, keepdims=False)

        def grow(total: FloatND, component: FloatND) -> tuple[FloatND, FloatND]:
            return _two_sum(total, component)

        total, residuals = jax.lax.scan(grow, term, jnp.moveaxis(expansion, -1, 0))
        return jax.lax.dynamic_update_index_in_dim(
            jnp.moveaxis(residuals, 0, -1), total, k, axis=-1
        )

    expansion = jax.lax.fori_loop(0, n_term, absorb_term, expansion)
    # NaN compares unequal to zero, so a non-finite input surfaces as a NaN
    # sign rather than a silently-ordered comparison.
    nonzero = expansion != 0
    top = n_term - 1 - jnp.argmax(nonzero[..., ::-1], axis=-1)
    leading = jnp.take_along_axis(expansion, top[..., None], axis=-1)[..., 0]
    return jnp.where(
        jnp.any(nonzero, axis=-1), jnp.sign(leading), jnp.zeros_like(leading)
    )


def _exact_compare(*, cols_a: FloatND, cols_b: FloatND, q: FloatND) -> FloatND:
    """Exact sign of `V_a(q) - V_b(q)`: `+1`, `-1`, or `0` for a TRUE tie.

    `V_a - V_b = (N_a*D_b - N_b*D_a) / (D_a*D_b)` with both denominators
    positive, so the sign is that of the cross-multiplied numerator difference.
    Every cross product of two exactly-represented terms is itself split into
    an exact pair, so the 64 terms sum to that difference with no error and
    `_exact_sign_of_sum` reads its sign exactly. This is the only test that can
    tell a genuine tie from a strict gap finer than the working precision —
    double-double values cannot, because algebraically different segment
    parameterizations of the SAME exact value need not produce the same low
    word (round-6 audit F2).

    Only the VALUE scale is shared, and it is the only thing this function has
    to decide: each candidate normalizes its own grid differences, which cancel
    within its own ratio. The grid columns are handed over RAW so that every
    difference is formed before anything is rescaled (round-10 audit F2).
    """
    value_exponent = _group_scale_exponent(
        cols_a[..., 2],
        cols_a[..., 3],
        cols_b[..., 2],
        cols_b[..., 3],
        target=_mid_range_target(cols_a.dtype),
    )
    return _exact_cross_sign(
        a=_exact_ratio(cols=cols_a, q=q, value_exponent=value_exponent),
        b=_exact_ratio(cols=cols_b, q=q, value_exponent=value_exponent),
        dtype=cols_a.dtype,
    )


def _exact_slope_ratio(
    *, cols: FloatND, value_exponent: jax.Array, grid_exponent: jax.Array
) -> _ExactRatio:
    """A candidate's value slope as an exact rational `(v1-v0)/(x1-x0)`.

    Both differences are error-free pairs, so the ratio carries no rounding at
    all. The denominator is canonically positive so a cross-multiplied
    comparison reads its sign off the numerator difference.

    Returned in units of `2**(grid_exponent - value_exponent)`. Unlike the value
    ratio, a slope's numerator and denominator scales do NOT cancel within one
    candidate — they change the ratio — so BOTH are supplied by the caller and
    shared across the candidates being compared. The shared factor is a positive
    power of two, so it cannot change which slope is larger.

    As in `_exact_ratio`, both differences are formed on the RAW columns before
    any scaling: a scale applied to the operands can merge two distinct floats
    and silently zero the difference (round-10 audit F2).
    """
    left_grid, right_grid = cols[..., 0], cols[..., 1]
    left_value, right_value = cols[..., 2], cols[..., 3]
    nh, nl = _two_diff(right_value, left_value)
    dh, dl = _two_diff(right_grid, left_grid)
    # A zero-width segment has no grid span; the established convention
    # publishes its raw value jump as the slope, i.e. a unit RAW denominator —
    # substituted before scaling so it lands in the same units as every other
    # candidate's width.
    zero_width = right_grid == left_grid
    dh = jnp.where(zero_width, jnp.ones_like(dh), dh)
    dl = jnp.where(zero_width, jnp.zeros_like(dl), dl)
    numerator = jnp.stack(
        [jnp.ldexp(nh, -value_exponent), jnp.ldexp(nl, -value_exponent)], axis=-1
    )
    denominator = jnp.stack(
        [jnp.ldexp(dh, -grid_exponent), jnp.ldexp(dl, -grid_exponent)], axis=-1
    )
    flip = (denominator[..., 0] < 0)[..., None]
    return _ExactRatio(
        numerator=jnp.where(flip, -numerator, numerator),
        denominator=jnp.where(flip, -denominator, denominator),
    )


def _exact_slope_compare(*, cols_a: FloatND, cols_b: FloatND) -> FloatND:
    """Exact sign of `slope_a - slope_b`: `+1`, `-1`, or `0` for a TRUE tie.

    Same cross-multiplied construction as `_exact_compare`, on the slope ratio
    instead of the value: `sign(N_a*D_b - N_b*D_a)` over 16 error-free terms.

    This exists because an exact VALUE tie is only half the rule. The
    right-continuous break then orders slopes, and ordering them on
    `fl((v1-v0)/(x1-x0))` re-introduces the very defect the exact value
    predicate removed one operation earlier: two strictly ordered exact slopes
    can share a single float key, and `argmax` then silently falls back to
    candidate order, so a pure branch permutation flips the published policy
    and marginal (round-7 audit F2).

    Round 8 made that predicate exact but not exponent-safe: it multiplied the
    cross terms at their native scale, so a uniformly rescaled model could
    underflow a strict sign to zero or overflow it to NaN. `_exact_cross_sign`
    normalizes each ratio into a safe binade first, which is a no-op on the
    rational being compared (round-8 audit F2).

    Both scales are shared across the two candidates and both are taken from the
    DIFFERENCES, not the columns — a slope's numerator and denominator scales do
    not cancel within a candidate, so neither may be chosen locally.
    """

    def effective_width(cols: FloatND) -> FloatND:
        # The zero-width convention's unit RAW denominator has to sit inside the
        # group that picks the shared scale, or that scale could push it into the
        # subnormal range and cost the convention its exactness.
        width = cols[..., 1] - cols[..., 0]
        return jnp.where(width == 0, jnp.ones_like(width), width)

    target = _mid_range_target(cols_a.dtype)
    value_exponent = _group_scale_exponent(
        cols_a[..., 3] - cols_a[..., 2],
        cols_b[..., 3] - cols_b[..., 2],
        target=target,
    )
    grid_exponent = _group_scale_exponent(
        effective_width(cols_a), effective_width(cols_b), target=target
    )
    return _exact_cross_sign(
        a=_exact_slope_ratio(
            cols=cols_a, value_exponent=value_exponent, grid_exponent=grid_exponent
        ),
        b=_exact_slope_ratio(
            cols=cols_b, value_exponent=value_exponent, grid_exponent=grid_exponent
        ),
        dtype=cols_a.dtype,
    )


class _SlopeState(NamedTuple):
    """Running state of the exact slope resolution loop, one entry per query."""

    lead_cols: FloatND
    lead_index: jax.Array
    remaining: BoolND


class _ResolveState(NamedTuple):
    """Running state of the exact resolution loop, one entry per query."""

    lead_cols: FloatND
    tied: BoolND
    remaining: BoolND


def _exactly_maximal(
    *,
    terms: _CandidateTerms,
    gather: Callable[[jax.Array], FloatND],
    q: FloatND,
) -> BoolND:
    """Mask of the exactly-maximal bracketing candidates, per query.

    Two stages, and the split is the point of the design:

    1. **Certified screen.** The double-double leader's interval is compared
       against every other candidate's. Anything whose certified interval lies
       strictly below the leader's is certified to lose and is discarded with
       no further work — this is where the radius, previously computed and then
       ignored, actually enters selection. The screen is deliberately generous:
       it must be a superset, and a superset only costs comparisons.
    2. **Exact resolution.** Whatever the screen could not separate is resolved
       by `_exact_compare`, which is exact and therefore recognises a true tie
       as a true tie. Candidates that are *certifiably* tied already — node
       events, whose radius is zero and whose stored `(hi, lo)` pairs are
       bitwise equal — skip it, since exactness has nothing to add there.

    Consequently the loop body runs zero times on the overwhelmingly common
    query: at a node every survivor is a certified exact tie, and in the
    interior the runner-up is certified strictly below. `while_loop` under
    `vmap` executes only as many iterations as some lane still needs, so an
    empty screen costs nothing at all.
    """
    n_segment = terms.brackets.shape[1]
    masked_hi = jnp.where(terms.brackets, terms.value_hi, -jnp.inf)
    max_hi = jnp.max(masked_hi, axis=1, keepdims=True)
    hi_tied = terms.brackets & (masked_hi == max_hi)
    masked_lo = jnp.where(hi_tied, terms.value_lo, -jnp.inf)
    max_lo = jnp.max(masked_lo, axis=1, keepdims=True)
    dd_tied = hi_tied & (masked_lo == max_lo)

    lead = jnp.argmax(dd_tied, axis=1)
    lead_hot = _one_hot(lead, n_segment)
    lead_hi = jnp.take_along_axis(terms.value_hi, lead[:, None], axis=1)
    lead_lo = jnp.take_along_axis(terms.value_lo, lead[:, None], axis=1)
    lead_radius = jnp.take_along_axis(terms.radius, lead[:, None], axis=1)

    # Certified screen: keep candidate i when `V_i + r_i >= V_lead - r_lead`,
    # i.e. when its certified interval still reaches the leader's. Sharing a
    # high word is kept unconditionally — that is exactly the class where the
    # low word is rounding residue rather than order.
    gap_hi, gap_lo = _dd_add(terms.value_hi, terms.value_lo, -lead_hi, -lead_lo)
    reach = _SCREEN_SLACK * (terms.radius + lead_radius)
    contends = terms.brackets & (
        ((gap_hi + gap_lo) >= -reach) | (terms.value_hi == lead_hi)
    )
    # Node events carry a zero radius and a stored value, so bitwise-equal
    # `(hi, lo)` pairs there ARE exactly equal values; nothing to resolve.
    certified_tie = dd_tied & (terms.radius == 0) & (lead_radius == 0)

    def unresolved(state: _ResolveState) -> BoolND:
        return jnp.any(state.remaining)

    def resolve_one(state: _ResolveState) -> _ResolveState:
        active = jnp.any(state.remaining, axis=1)
        index = jnp.argmax(state.remaining, axis=1)
        hot = _one_hot(index, n_segment) & state.remaining
        cols = gather(index)
        sign = _exact_compare(cols_a=cols, cols_b=state.lead_cols, q=q)
        # A NaN sign (non-finite candidate arithmetic) is neither greater nor
        # equal, so it never takes the lead; `poisoned` NaNs the query anyway.
        greater = active & (sign > 0)
        equal = active & (sign == 0)
        return _ResolveState(
            lead_cols=jnp.where(greater[:, None], cols, state.lead_cols),
            # A promotion invalidates every earlier tie: those candidates tied
            # with a leader now known to be strictly smaller.
            tied=jnp.where(
                greater[:, None],
                hot,
                state.tied | (hot & equal[:, None]),
            ),
            remaining=state.remaining & ~hot,
        )

    final = jax.lax.while_loop(
        unresolved,
        resolve_one,
        _ResolveState(
            lead_cols=gather(lead),
            tied=certified_tie | lead_hot,
            remaining=contends & ~certified_tie & ~lead_hot,
        ),
    )
    return final.tied


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
    node, use_right = _node_selection(
        q=q,
        left_grid=left_grid,
        right_grid=right_grid,
        left_value=left_value,
        right_value=right_value,
    )

    # Interior: compensated interpolation `left + (t/w)*d` in double-double.
    # TwoDiff makes t, w, d exact dd representations; division and product are
    # dd-accurate, so (hi, lo) carries the interpolant to O(eps^2) relative.
    #
    # SCALE AFTER DIFFERENCING, AND CARRY THE EXPONENT AS AN INTEGER.
    #
    # Dekker's TwoProd splits an operand by multiplying it by `2**s + 1`, so it
    # needs `|a| < 2**(emax - s)` — `2**115` in float32, `2**996` in float64 —
    # and past that the interior pair goes non-finite (round-8 audit F2). Rounds
    # 9 and 10 bought that headroom by scaling the OPERANDS first: round 9 with
    # one exponent for the whole array, round 10 with one per candidate. Both are
    # lossy for the same reason — scaling `q` and `x0` before subtracting them
    # can map two distinct represented grid points onto the SAME float, and then
    # `t = q - x0` is zero and no downstream exactness can recover it. Round 10's
    # own witness: `x0 = 1`, `q = nextafter(1)`, `x1 = 2**126`, where the
    # candidate exponent 127 makes both `x0` and `q` the same subnormal and a
    # strictly lower constant competitor takes the value, policy and marginal
    # (round-10 audit F2).
    #
    # So the differences are formed FIRST, on the raw stored operands, where
    # `_two_diff` is exact and subtraction of finite floats cannot overflow.
    # Only then is each difference scaled by its OWN binade, and the scale is
    # kept as an integer exponent rather than being applied to the operands:
    #
    #     r = t/w = (t * 2**-et) / (w * 2**-ew) * 2**(et - ew)
    #     p = r*d = [(t*2**-et)/(w*2**-ew) * (d*2**-ed)] * 2**(et - ew + ed)
    #
    # Every significand handed to Dekker is O(1), so splitting cannot overflow at
    # any model scale, and the exponent is applied ONCE, exactly, at the end.
    #
    # No value scale is needed at all. A bracketing candidate has `q` in
    # `[x0, x1]`, hence `|t| <= |w|` and `r` in `[0, 1]`, so `|p| <= |d|` and
    # `v = v0 + p` is bounded by the endpoint values themselves. The unbounded
    # intermediates only ever came from mixing units — a value times a grid
    # difference — which this form never constructs.
    zero_width = right_grid == left_grid
    split = _dekker_split_factor(block.dtype)

    # ... with ONE exception, which is where scaling operands is not merely safe
    # but necessary. "Subtraction of finite floats cannot overflow" is true, but
    # it can UNDERFLOW: near the bottom of the range the difference of two
    # distinct normals is SUBNORMAL, and XLA flushes subnormals to zero. For a
    # segment at the smallest normal scale, `q - x0` came back exactly `0.0` on
    # raw operands and the interpolant collapsed onto its left value — the
    # round-10 defect reappearing at the opposite end of the range (round-9
    # audit MT6, 48/144 mismatches, introduced by this round's own repair).
    #
    # The asymmetry that resolves it: scaling operands DOWN can merge two
    # distinct floats, which is what rounds 9 and 10 got wrong; scaling them UP
    # cannot, because multiplying by a power of two is exact and injective while
    # nothing overflows. So LIFT — never lower — the grid operands into mid-range
    # before differencing, with ONE common shift for `q` and both nodes so every
    # difference and every equality between them is preserved exactly. The lift
    # cancels out of `r = t/w` and so never reaches the published fraction.
    largest_operand = jnp.maximum(
        jnp.abs(q), jnp.maximum(jnp.abs(left_grid), jnp.abs(right_grid))
    )
    lift = jnp.minimum(
        _scale_exponent(largest_operand, _mid_range_target(block.dtype)), 0
    )
    th, tl = _two_diff(jnp.ldexp(q, -lift), jnp.ldexp(left_grid, -lift))
    wh, wl = _two_diff(jnp.ldexp(right_grid, -lift), jnp.ldexp(left_grid, -lift))
    wh = jnp.where(zero_width, jnp.ones_like(wh), wh)
    wl = jnp.where(zero_width, jnp.zeros_like(wl), wl)
    dh, dl = _two_diff(right_value, left_value)

    t_exp = _binade_exponent(jnp.abs(th))
    w_exp = _binade_exponent(jnp.abs(wh))
    d_exp = _binade_exponent(jnp.abs(dh))

    rh, rl = _dd_div(
        jnp.ldexp(th, -t_exp),
        jnp.ldexp(tl, -t_exp),
        jnp.ldexp(wh, -w_exp),
        jnp.ldexp(wl, -w_exp),
        split,
    )
    ph, pl = _dd_mul(rh, rl, jnp.ldexp(dh, -d_exp), jnp.ldexp(dl, -d_exp), split)
    product_exp = t_exp - w_exp + d_exp
    ph, pl = jnp.ldexp(ph, product_exp), jnp.ldexp(pl, product_exp)
    vh, vl = _dd_add_fp(ph, pl, left_value)

    eps = jnp.finfo(block.dtype).eps
    interior_radius = (
        _INTERIOR_RADIUS_ULPS2 * eps * eps * (jnp.abs(left_value) + jnp.abs(ph))
    )

    # `rh` is the ratio's SIGNIFICAND; the carried and interpolated quantities
    # need the ratio itself. `r` is in `[0, 1]` for a bracketing candidate, so
    # this shift cannot overflow.
    fraction = jnp.ldexp(rh, t_exp - w_exp)

    zero = jnp.zeros_like(vh)
    return _CandidateTerms(
        brackets=brackets,
        value_hi=jnp.where(node, jnp.where(use_right, right_value, left_value), vh),
        value_lo=jnp.where(node, zero, vl),
        radius=jnp.where(node, zero, interior_radius),
        policy=jnp.where(
            node,
            jnp.where(use_right, right_policy, left_policy),
            left_policy + fraction * (right_policy - left_policy),
        ),
        marginal=jnp.where(
            node,
            jnp.where(use_right, right_marginal, left_marginal),
            left_marginal + fraction * (right_marginal - left_marginal),
        ),
        # In ORIGINAL units, not the per-candidate binade: this is a screen key
        # compared ACROSS candidates, so every candidate must express it in the
        # same units. It is one subtraction over another, with no Dekker split,
        # so it carries no overflow risk of its own.
        slope=(right_value - left_value)
        / jnp.where(zero_width, jnp.ones_like(right_grid), right_grid - left_grid),
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
    # NO global rescaling happens here, deliberately. Round 9 normalized the
    # whole problem at this point to keep Dekker's splitting in range, and that
    # map is NOT INJECTIVE over a mixed-scale array: one exponent was chosen
    # from the largest candidate anywhere in the input, including candidates
    # that cannot bracket the query, so a remote segment at `2**126` collapsed
    # two adjacent local floats into the same subnormal BEFORE the certified
    # comparator ever saw them. Exact arithmetic downstream cannot rebuild bits
    # that were discarded upstream, and the published value, policy and marginal
    # all changed in response to a segment that is mathematically irrelevant
    # (round-9 audit F2).
    #
    # Exponent safety is therefore established PER CANDIDATE, inside
    # `_candidate_terms`, where the operands of each error-free transform are
    # known and nothing another candidate does can perturb them. Topology and
    # node events stay on the ORIGINAL represented coordinates.
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

    # Certified screen followed by exact resolution: the winner set is the
    # EXACTLY maximal candidates, so right-continuity is applied to genuine
    # ties and never bypassed by a low-word residue (requirement 3).
    exact_tie = _exactly_maximal(
        terms=terms, gather=lambda index: columns[index], q=flat
    )

    # Right-continuous break among the exactly-tied candidates only, then ONE
    # winner index per query; value, policy, and marginal are gathered from that
    # same index (requirements 4 and 5).
    _, winner = _right_continuous_winner(
        tied=exact_tie, terms=terms, gather=lambda index: columns[index]
    )
    best = winner[:, None]
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


def _right_continuous_winner(
    *,
    tied: BoolND,
    terms: _CandidateTerms,
    gather: Callable[[jax.Array], FloatND],
) -> tuple[BoolND, jax.Array]:
    """Per-query right-continuous winner among the EXACTLY value-tied candidates.

    Implements "prefer a segment extending strictly right of the query, then
    the larger value slope, then the earliest candidate" — and implements the
    slope half EXACTLY.

    The rounded key `fl((v1-v0)/(x1-x0))` is used only as a screen. A slope gap
    wider than a few ULP is certified by it; anything closer is settled by
    `_exact_slope_compare`. Ordering on the rounded key alone is unsound: two
    strictly ordered exact slopes can collapse onto one float key, and `argmax`
    then resolves by candidate order, so permuting the branches flips the
    published policy and marginal (round-7 audit F2 — 16 of 16 generated
    collision classes were order-dependent).

    Keeping the two keys separate rather than folding them into one scalar is
    still required: an `arctan(slope)/pi + right_available` rank loses slope
    bits for near-equal small slopes in float32 (round-4 audit F2).

    Returns the per-query right-extension flag (so the blocked scan can
    reconcile that priority across blocks) and ONE winner index.
    """
    n_segment = tied.shape[1]
    eligible = tied & terms.right_available
    any_eligible = jnp.any(eligible, axis=1, keepdims=True)
    compete = jnp.where(any_eligible, eligible, tied)

    key = jnp.where(compete, terms.slope, -jnp.inf)
    lead = jnp.argmax(key, axis=1)
    lead_slope = jnp.take_along_axis(terms.slope, lead[:, None], axis=1)
    eps = jnp.finfo(terms.slope.dtype).eps
    reach = _SLOPE_SCREEN_ULPS * eps * (jnp.abs(terms.slope) + jnp.abs(lead_slope))
    # A screen that cannot discriminate must DEFER to the exact comparison, not
    # decide. `slope` is a rounded key in ORIGINAL units — deliberately, since
    # it is compared across candidates — so it overflows to an infinity when a
    # large value gap meets a small grid width, with every stored input still
    # finite normal. `reach` is then infinite, `lead_slope - reach` is NaN,
    # every `>=` is False, and `contends` would be EMPTY: the exact loop never
    # runs and the winner is whatever `argmax` over the rounded key returned,
    # i.e. candidate order. That is precisely the round-7 F2 failure — level
    # tied either way, only the published policy and marginal wrong — and it is
    # reachable at `|Δvalue| / Δgrid > max_float` (probe: float32 values at
    # `2**100` over a width of `2**-60` flip policy 1.0 <-> 2.0 under a branch
    # permutation). Where the screen's own arithmetic is not finite it admits
    # every competitor and lets `_exact_slope_compare` settle the order.
    screen_usable = jnp.isfinite(reach) & jnp.isfinite(lead_slope)
    contends = compete & (~screen_usable | (terms.slope >= lead_slope - reach))
    lead_hot = _one_hot(lead, n_segment)

    def unresolved(state: _SlopeState) -> BoolND:
        return jnp.any(state.remaining)

    def resolve_one(state: _SlopeState) -> _SlopeState:
        active = jnp.any(state.remaining, axis=1)
        index = jnp.argmax(state.remaining, axis=1)
        hot = _one_hot(index, n_segment) & state.remaining
        cols = gather(index)
        sign = _exact_slope_compare(cols_a=cols, cols_b=state.lead_cols)
        # Candidates are visited in increasing index and a STRICTLY greater
        # exact slope takes the lead, so the winner is the earliest index
        # attaining the exact maximum — the documented earliest-on-tie rule,
        # conditioned on exact rather than rounded equality.
        #
        # The `sign == 0` clause is what makes that unconditional. The loop is
        # seeded from `argmax` over the ROUNDED key, and that key is a doubly
        # rounded quantity: `fl(fl(v1-v0) / fl(x1-x0))`, whereas the exact ratio
        # uses the error-free `(hi, lo)` differences. When a stored difference
        # is not representable the two disagree, so two candidates with EQUAL
        # exact slopes can carry different rounded keys and `argmax` can seed
        # the lead at the later of them. Without this clause the last-resort
        # fallback would then publish the later candidate — the rounded key
        # deciding an order it has no standing to decide, which is the very
        # defect this function exists to remove. Taking the lead on an exact tie
        # from any strictly earlier index closes that without relying on the
        # rounded key being monotone.
        earlier_on_exact_tie = (sign == 0) & (index < state.lead_index)
        greater = active & ((sign > 0) | earlier_on_exact_tie)
        return _SlopeState(
            lead_cols=jnp.where(greater[:, None], cols, state.lead_cols),
            lead_index=jnp.where(greater, index, state.lead_index),
            remaining=state.remaining & ~hot,
        )

    final = jax.lax.while_loop(
        unresolved,
        resolve_one,
        _SlopeState(
            lead_cols=gather(lead),
            lead_index=lead,
            remaining=contends & ~lead_hot,
        ),
    )
    return any_eligible[:, 0], final.lead_index


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
    has_winner: BoolND
    hi: FloatND
    lo: FloatND
    radius: FloatND
    right_extending: BoolND
    slope: FloatND
    value: FloatND
    policy: FloatND
    marginal: FloatND
    cols: FloatND


def _combine_blocked(
    *, carry: _BlockedCarry, block: _BlockedCarry, q: FloatND
) -> _BlockedCarry:
    """Fold a block's winner into the running winner, under the shared rule.

    The cross-block comparison is the SAME certified-screen-then-exact
    resolution the within-block reduction runs, applied to a two-candidate set.
    Routing both through one implementation is what makes dense and blocked
    provably agree: a bespoke lexicographic update here would re-introduce
    exactly the low-word ordering the exact predicate exists to prevent, and
    only on the block boundary, where no within-block test would see it.
    """
    stacked = _CandidateTerms(
        brackets=jnp.stack([carry.has_winner, block.has_winner], axis=1),
        value_hi=jnp.stack([carry.hi, block.hi], axis=1),
        value_lo=jnp.stack([carry.lo, block.lo], axis=1),
        radius=jnp.stack([carry.radius, block.radius], axis=1),
        policy=jnp.stack([carry.policy, block.policy], axis=1),
        marginal=jnp.stack([carry.marginal, block.marginal], axis=1),
        slope=jnp.stack([carry.slope, block.slope], axis=1),
        right_available=jnp.stack(
            [carry.right_extending, block.right_extending], axis=1
        ),
    )
    pair_cols = jnp.stack([carry.cols, block.cols], axis=1)
    tied = _exactly_maximal(
        terms=stacked,
        gather=lambda index: jnp.take_along_axis(
            pair_cols, index[:, None, None], axis=1
        )[:, 0],
        q=q,
    )
    _, winner = _right_continuous_winner(
        tied=tied,
        terms=stacked,
        gather=lambda index: jnp.take_along_axis(
            pair_cols, index[:, None, None], axis=1
        )[:, 0],
    )
    # Column 0 is the carry, so keeping the earlier column on an exact tie is
    # precisely the dense reduction's earliest-candidate rule.
    take = winner == 1
    return _BlockedCarry(
        any_bracket=carry.any_bracket | block.any_bracket,
        poisoned=carry.poisoned | block.poisoned,
        has_winner=carry.has_winner | block.has_winner,
        hi=jnp.where(take, block.hi, carry.hi),
        lo=jnp.where(take, block.lo, carry.lo),
        radius=jnp.where(take, block.radius, carry.radius),
        right_extending=jnp.where(take, block.right_extending, carry.right_extending),
        slope=jnp.where(take, block.slope, carry.slope),
        value=jnp.where(take, block.value, carry.value),
        policy=jnp.where(take, block.policy, carry.policy),
        marginal=jnp.where(take, block.marginal, carry.marginal),
        cols=jnp.where(take[:, None], block.cols, carry.cols),
    )


def _envelope_at_query_blocked(
    *, columns: FloatND, live: BoolND, query: FloatND, block_size: int
) -> tuple[FloatND, FloatND, FloatND]:
    """Single-scan blocked equivalent of the dense `(n_query, n_segment)` path.

    Evaluates every candidate through the shared `_candidate_terms` and reduces
    with the same lexicographic rule as the dense path — certified `(hi, lo)`
    value pairs first, then (only on an exact tie) right-extension and value
    slope. Within a block the winner is found exactly as in the dense reduction
    (`_right_continuous_winner` restricted to the block's exactly-tied candidates)
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
        # Within-block winner: identical construction to the dense reduction,
        # through the very same certified-screen-then-exact resolution.
        tied = _exactly_maximal(terms=t, gather=lambda index: block[index], q=flat)
        block_ra, winner = _right_continuous_winner(
            tied=tied, terms=t, gather=lambda index: block[index]
        )
        column = winner[:, None]
        block_has = jnp.any(t.brackets, axis=1)
        block_carry = _BlockedCarry(
            any_bracket=block_has,
            poisoned=jnp.any(
                t.brackets & (jnp.isnan(t.value_hi) | jnp.isnan(t.value_lo)), axis=1
            ),
            has_winner=block_has,
            hi=jnp.take_along_axis(t.value_hi, column, axis=1)[:, 0],
            lo=jnp.take_along_axis(t.value_lo, column, axis=1)[:, 0],
            radius=jnp.take_along_axis(t.radius, column, axis=1)[:, 0],
            right_extending=block_ra,
            slope=jnp.take_along_axis(t.slope, column, axis=1)[:, 0],
            value=jnp.take_along_axis(t.value_hi, column, axis=1)[:, 0],
            policy=jnp.take_along_axis(t.policy, column, axis=1)[:, 0],
            marginal=jnp.take_along_axis(t.marginal, column, axis=1)[:, 0],
            cols=block[winner],
        )
        return _combine_blocked(carry=carry, block=block_carry, q=flat), None

    final, _ = jax.lax.scan(
        step,
        _BlockedCarry(
            any_bracket=jnp.zeros((n_query,), dtype=bool),
            poisoned=jnp.zeros((n_query,), dtype=bool),
            has_winner=jnp.zeros((n_query,), dtype=bool),
            hi=jnp.full((n_query,), -jnp.inf, dtype=dtype),
            lo=jnp.full((n_query,), -jnp.inf, dtype=dtype),
            radius=jnp.zeros((n_query,), dtype=dtype),
            right_extending=jnp.zeros((n_query,), dtype=bool),
            slope=jnp.full((n_query,), -jnp.inf, dtype=dtype),
            value=jnp.full((n_query,), jnp.nan, dtype=dtype),
            policy=jnp.full((n_query,), jnp.nan, dtype=dtype),
            marginal=jnp.full((n_query,), jnp.nan, dtype=dtype),
            cols=jnp.zeros((n_query, columns.shape[1]), dtype=dtype),
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
