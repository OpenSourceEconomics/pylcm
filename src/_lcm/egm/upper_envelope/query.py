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


class _Dyadic(NamedTuple):
    """Exactly-represented reals `mantissa * 2**exponent`, one per trailing slot.

    Every quantity in the ordering kernel travels this way: a finite float
    mantissa and a SEPARATE `int32` exponent. That split is the whole design.
    An exponent held as an integer never rounds, never overflows and never
    flushes, so a term's magnitude may range over the entire real line while its
    mantissa stays inside one binade — the binade where every error-free
    transform below is exact.

    Nine consecutive repairs to this kernel (rounds 6 to 12) each moved the same
    defect one layer down: a value comparison, a slope tie-break, an exponent, a
    normalization's scope, then its shape, then the last normalization inside
    the "exact" fallback. All nine shared one premise — that a rescaling can
    make a rounded evaluation safe. It cannot: scaling operands DOWN merges
    distinct floats and scaling them UP eventually overflows, and no single
    binade holds a problem whose terms span more binades than the format has.
    Carrying the exponent outside the float removes the premise instead of
    patching its latest consequence.
    """

    mantissa: FloatND
    exponent: jax.Array


class _ExactRatio(NamedTuple):
    """A candidate's value at the query as an exact rational `num / den`.

    `numerator` and `denominator` are unevaluated sums of dyadic terms — error
    free transforms throughout, so no information has been discarded and no
    term's magnitude has been constrained. The denominator is canonically
    positive, which lets a cross-multiplied comparison read its sign straight
    off the numerator difference.
    """

    numerator: _Dyadic
    denominator: _Dyadic


def _dyadic_parts(x: FloatND) -> tuple[FloatND, jax.Array]:
    """`x` as `(m, e)` with `m` in `[0.5, 1)` and `x == m * 2**e` EXACTLY.

    Zero and non-finite inputs carry exponent zero and pass through unchanged,
    so a NaN stays a NaN and surfaces rather than being silently ordered.
    """
    mantissa, exponent = jnp.frexp(x)
    usable = jnp.isfinite(x) & (x != 0)
    return mantissa, jnp.where(usable, exponent, jnp.zeros_like(exponent))


def _binade_exponent(magnitude: FloatND) -> jax.Array:
    """Exponent `e` with `magnitude` in `[2**(e-1), 2**e)`; `0` where there is none.

    Returned as an exponent rather than a factor so callers scale with `ldexp`.
    Materializing `2**-e` as a float would itself be lossy at the extremes: for
    a float32 magnitude near the top of the range `2**-e` is SUBNORMAL, so the
    factor carries fewer bits than the scaling it is meant to perform exactly.
    `ldexp` adjusts the exponent field directly and has no such failure mode.

    Used only by the double-double SCREEN, which may round; the exact ordering
    kernel carries its exponents in `_Dyadic` and normalizes nothing.
    """
    _, exponent = jnp.frexp(magnitude)
    usable = (magnitude > 0) & jnp.isfinite(magnitude)
    return jnp.where(usable, exponent, jnp.zeros_like(exponent))


def _as_dyadic(x: FloatND) -> _Dyadic:
    """A raw float as a one-term dyadic list; its own exponent is implicit."""
    return _Dyadic(
        mantissa=x[..., None],
        exponent=jnp.zeros((*x.shape, 1), dtype=jnp.int32),
    )


def _dyadic_join(*parts: _Dyadic) -> _Dyadic:
    """Concatenate dyadic term lists — the exact sum of the parts."""
    return _Dyadic(
        mantissa=jnp.concatenate([part.mantissa for part in parts], axis=-1),
        exponent=jnp.concatenate([part.exponent for part in parts], axis=-1),
    )


def _dyadic_negate(terms: _Dyadic, flip: BoolND) -> _Dyadic:
    """Negate every term where `flip`; exact, and it leaves the exponents alone."""
    return _Dyadic(
        mantissa=jnp.where(flip[..., None], -terms.mantissa, terms.mantissa),
        exponent=terms.exponent,
    )


def _exact_difference(a: FloatND, b: FloatND) -> _Dyadic:
    """`a - b` as an UNEVALUATED two-term dyadic sum. No subtraction is performed.

    Round 13 formed the difference with `_two_diff` on operands LIFTED into
    mid-range, and justified it with the hinge claim that "subtraction of two
    finite floats cannot overflow". That claim is false, and opposite-signed
    top-binade operands are the counterexample: for `H = 2**127` in float32 or
    `2**1023` in float64 — both finite normals — the lift is zero because the
    operands already exceed the target, and `_two_diff(H, -H)` returns
    `(inf, nan)`. A finite exact envelope was then published as three NaNs
    (round-13 audit F2).

    Rounds 8 to 12 guarded the same operation against UNDERFLOW and round 13
    against nothing else; each repair fixed the direction its witness pointed at
    and left the mirror image open. The fix is to stop performing the operation.

    `a = m_a * 2**e_a` and `b = m_b * 2**e_b` exactly, from `frexp`, so
    `a - b` is exactly the two-term sum `[(m_a, e_a), (-m_b, e_b)]`. Each
    mantissa is already in `[0.5, 1)`, each exponent is an int32, and NOTHING is
    added, subtracted, scaled or rounded to build it. There is no direction left
    in which to be wrong: the construction is total over every pair of finite
    floats, at every relative scale, and `_exact_sign_of_sum` consumes exactly
    such term lists already.

    The two terms are also strictly TIGHTER than the lifted pair they replace —
    they live in the operands' own binades rather than a head one binade above
    and a residual `precision` binades below — so the bound in
    `_accumulator_layout` still holds with room to spare.
    """
    mantissa_a, exponent_a = _dyadic_parts(a)
    mantissa_b, exponent_b = _dyadic_parts(b)
    return _Dyadic(
        mantissa=jnp.stack([mantissa_a, -mantissa_b], axis=-1),
        exponent=jnp.stack([exponent_a, exponent_b], axis=-1),
    )


def _framed_difference(a: FloatND, b: FloatND) -> tuple[FloatND, FloatND, jax.Array]:
    """`(head, tail, exponent)` with `a - b == (head + tail) * 2**exponent`.

    The double-double SCREEN's counterpart to `_exact_difference`, and the answer
    to the same round-13 finding on the public path: `_candidate_terms` formed
    its grid and value differences in the working dtype after a LIFT-ONLY shift,
    so two opposite-signed top-binade operands overflowed there exactly as they
    did in the exact kernel, and the published triple was `(NaN, NaN, NaN)`.

    Both operands are first divided by `2**max(e_a, e_b)` — which is `ldexp` on
    the `frexp` mantissa, hence exact — landing them in `(-1, 1]`. `_two_diff` on
    such a pair has `|head| <= 2` and a residual at most `precision` binades
    below, so it can neither overflow nor lose its tail, whatever the operands'
    own magnitudes. The frame rides along as an INTEGER exponent.

    Where the operands are more than `precision + |minexp|` binades apart the
    smaller flushes to zero in the frame, and the head is then the larger operand
    to within `2**-precision-|minexp|` relative — some 100 binades below the
    certified radius, so the screen stays honest. That is the ONLY approximation
    here, it is bounded, and the exact predicate does not use this function.
    """
    mantissa_a, exponent_a = _dyadic_parts(a)
    mantissa_b, exponent_b = _dyadic_parts(b)
    exponent = jnp.maximum(exponent_a, exponent_b)
    head, tail = _two_diff(
        jnp.ldexp(mantissa_a, exponent_a - exponent),
        jnp.ldexp(mantissa_b, exponent_b - exponent),
    )
    return head, tail, exponent


def _dyadic_product(a: _Dyadic, b: _Dyadic, split: float) -> _Dyadic:
    """Exact outer product of two dyadic term lists: `2 * k_a * k_b` terms.

    Each pair of mantissas is normalized into `[0.5, 1)` BEFORE multiplying, so
    `_two_prod` always runs on operands whose product lands in `[0.25, 1)`: it
    can neither overflow to infinity nor lose its residual to underflow, at any
    input scale whatsoever. Dekker's split constant is likewise safe there,
    which retires the caveat at the top of this module for the exact path.

    The magnitudes ride along in the integer exponents, where nothing rounds.
    This is what the round-12 finding asked for: the decisive product of a
    half-minimum-normal numerator term with a half-minimum-normal denominator
    term used to flush to zero before the exact summation ever saw it.
    """
    b_parts = [_dyadic_parts(b.mantissa[..., j]) for j in range(b.mantissa.shape[-1])]
    mantissas: list[FloatND] = []
    exponents: list[jax.Array] = []
    for i in range(a.mantissa.shape[-1]):
        a_mantissa, a_exponent = _dyadic_parts(a.mantissa[..., i])
        a_exponent = a_exponent + a.exponent[..., i]
        for j, (b_mantissa, b_exponent) in enumerate(b_parts):
            head, tail = _two_prod(a_mantissa, b_mantissa, split)
            exponent = a_exponent + b_exponent + b.exponent[..., j]
            mantissas += [head, tail]
            exponents += [exponent, exponent]
    return _Dyadic(
        mantissa=jnp.stack(mantissas, axis=-1),
        exponent=jnp.stack(exponents, axis=-1),
    )


class _AccumulatorLayout(NamedTuple):
    """Geometry of the exact fixed-point accumulator, derived from the dtype."""

    limb_bits: int
    n_limb: int
    n_digit: int


def _accumulator_layout(dtype: jnp.dtype) -> _AccumulatorLayout:
    """Limb width, limb count and digits per term of the exact accumulator.

    Derived from the format rather than tabulated, so float32 and float64 cannot
    drift apart and a wrong constant cannot hide behind a passing float64 test.

    A limb holds a signed INTEGER in a float of the working dtype. It receives at
    most one digit per term plus the carry from the limb below, so it needs
    `log2(_EXACT_TERMS)` guard bits above `limb_bits` to stay exact.

    The limb count covers the full exponent span a cross-product term can reach.
    That span is what the previous nine repairs kept trying to compress into one
    binade: a value and a grid difference each range over the whole format, and
    their products then range over twice it. Nothing here compresses it — the
    accumulator is simply wide enough to hold it, which is why no input scale can
    make a term vanish before the sign is read.
    """
    info = jnp.finfo(dtype)
    precision = int(info.nmant) + 1
    # `frexp` exponents of the smallest and largest finite magnitudes.
    min_exponent = int(info.minexp) + 1
    max_exponent = int(info.maxexp)

    limb_bits = precision - (_EXACT_TERMS.bit_length() + 2)
    n_digit = -(-precision // limb_bits) + 1

    # A difference contributes its two OPERANDS as terms (`_exact_difference`
    # performs no subtraction), so each sits in the operand's own binade — inside
    # `[min_exponent, max_exponent]`. The wider window kept here is the round-13
    # bound for a rounded head-plus-residual pair; it strictly CONTAINS the
    # current one, so it stays valid, and holding it fixed keeps the layout the
    # one the round-13 review verified directly at 2,000/2,000 sign matches. A
    # product of two normalized mantissas puts its own residual at most
    # `2 * precision` below its head.
    difference_hi = max_exponent + 1
    difference_lo = min_exponent - precision - 1
    numerator_hi = max_exponent + difference_hi
    numerator_lo = min_exponent + difference_lo - 2 * precision
    cross_hi = numerator_hi + difference_hi
    cross_lo = numerator_lo + difference_lo - 2 * precision

    span = cross_hi - cross_lo
    n_limb = -(-(span + n_digit * limb_bits) // limb_bits) + 2
    return _AccumulatorLayout(limb_bits=limb_bits, n_limb=n_limb, n_digit=n_digit)


def _exact_sign_of_sum(terms: _Dyadic) -> FloatND:
    """Exact sign of `sum(mantissa * 2**exponent)`, with no rounding anywhere.

    Every term is deposited into a fixed-point accumulator of `n_limb` limbs,
    each limb an exact integer weighted by `2**(-limb_bits)` relative to the one
    above it and anchored at the largest term exponent present. A term's mantissa
    is sliced into `n_digit` such digits by repeated `trunc`, which is exact, and
    the digits are scattered into consecutive limbs. Nothing is rounded, nothing
    is compared against a tolerance, and no term is ever too small to contribute:
    a term 3000 binades below the leading one simply lands 3000 binades further
    down the accumulator.

    That is the difference from the expansion this replaces. Shewchuk's
    GROW-EXPANSION is exact for the terms it RECEIVES, and the loss was always
    upstream — in forming the terms at all. A fixed-point accumulator has no
    upstream: the exponent selects a limb, and selecting a limb cannot round.

    Carries are then propagated once from the least significant limb upward,
    leaving every limb in `[0, 2**limb_bits)` and the sum equal to
    `carry * 2**(n_limb * limb_bits) + (non-negative remainder)`. So the carry
    out decides the sign, and only an all-zero accumulator with zero carry is a
    TRUE tie — the one thing a rounded evaluation can never certify.
    """
    dtype = terms.mantissa.dtype
    limb_bits, n_limb, n_digit = _accumulator_layout(dtype)
    scale = float(2**limb_bits)

    mantissa, exponent = _dyadic_parts(terms.mantissa)
    exponent = exponent + terms.exponent
    active = mantissa != 0
    finite = jnp.all(jnp.isfinite(terms.mantissa), axis=-1)

    # A sentinel at or below every exponent present, so an all-zero lane picks a
    # harmless anchor instead of an out-of-range one.
    absent = jnp.min(exponent) - 1
    top = jnp.max(jnp.where(active, exponent, absent), axis=-1, keepdims=True)
    shift = jnp.where(active, top - exponent, jnp.zeros_like(exponent))
    limb = shift // limb_bits
    # `|mantissa| < 1` and the intra-limb offset is below `limb_bits`, so the
    # carrier stays inside one limb's range and its lowest bit is `precision`
    # bits below it — which is what `n_digit` is sized to drain exactly.
    carrier = jnp.ldexp(
        jnp.where(active, mantissa, jnp.zeros_like(mantissa)),
        limb_bits - (shift - limb * limb_bits),
    )

    indices: list[jax.Array] = []
    digits: list[FloatND] = []
    residue = carrier
    for offset in range(n_digit):
        digit = jnp.trunc(residue)
        indices.append(limb + offset)
        digits.append(digit)
        residue = jnp.ldexp(residue - digit, limb_bits)

    leading = terms.mantissa.shape[:-1]
    n_lane = 1
    for size in leading:
        n_lane *= size
    flat_index = jnp.concatenate(indices, axis=-1).reshape(n_lane, -1)
    flat_digit = jnp.concatenate(digits, axis=-1).reshape(n_lane, -1)
    accumulator = (
        jnp.zeros((n_lane, n_limb), dtype)
        .at[jnp.arange(n_lane)[:, None], flat_index]
        .add(flat_digit, mode="drop")
    )

    def propagate(carry: FloatND, limb_value: FloatND) -> tuple[FloatND, FloatND]:
        total = limb_value + carry
        quotient = jnp.floor(total * (1.0 / scale))
        return quotient, total - quotient * scale

    carry, residues = jax.lax.scan(
        propagate,
        jnp.zeros((n_lane,), dtype),
        jnp.moveaxis(accumulator, -1, 0)[::-1],
    )
    sign = jnp.where(
        carry < 0,
        -jnp.ones_like(carry),
        jnp.where(
            (carry > 0) | jnp.any(residues != 0, axis=0),
            jnp.ones_like(carry),
            jnp.zeros_like(carry),
        ),
    ).reshape(leading)

    # `_accumulator_layout` proves this cannot fire; it is kept because a silent
    # `mode="drop"` is exactly the failure this rewrite exists to remove, and a
    # loud NaN is recoverable evidence where a dropped digit is not.
    dropped = jnp.any(active & (limb + n_digit > n_limb), axis=-1)
    return jnp.where(finite & ~dropped, sign, jnp.full_like(sign, jnp.nan))


def _exact_cross_sign(*, a: _ExactRatio, b: _ExactRatio, dtype: jnp.dtype) -> FloatND:
    """Exact sign of `a - b` for two rationals with positive denominators.

    `a - b = (N_a*D_b - N_b*D_a) / (D_a*D_b)`, so with both denominators
    canonically positive the sign is that of the cross-multiplied numerator
    difference. Every cross product is formed on normalized mantissas with its
    exponent carried alongside, so the terms sum to that difference with no error
    at ANY model scale — not merely at the scale the tests happened to use.

    This is the single ordering kernel: value selection and the right-continuous
    slope tie-break share it, so the two can never disagree about what an exact
    tie is.
    """
    split = _dekker_split_factor(dtype)
    forward = _dyadic_product(a.numerator, b.denominator, split)
    reverse = _dyadic_product(b.numerator, a.denominator, split)
    return _exact_sign_of_sum(
        _dyadic_join(forward, _Dyadic(-reverse.mantissa, reverse.exponent))
    )


def _exact_ratio(*, cols: FloatND, q: FloatND) -> _ExactRatio:
    """Exact rational value of one candidate per query, from its raw columns.

    `V = (v0*(x1 - q) + v1*(q - x0)) / (x1 - x0)` is the same interpolant
    `_candidate_terms` evaluates, but every factor is kept as a dyadic term list
    instead of being rounded, so the pair `(numerator, denominator)` represents
    `V` with no error at all. A node event publishes a STORED float, so its exact
    value is that float over one — exact by construction, and expressible here
    with a unit denominator and no arithmetic at all.

    **No scale is shared and none is passed in.** Earlier rounds took a value
    exponent from the caller so both candidates' values could be compared in one
    frame; that frame is what underflowed. With the exponent carried per term,
    `V_a` and `V_b` are comparable because they are EXACT, not because they were
    manoeuvred into common units — and the grid differences, which used to need a
    group scale of their own, need none either.
    """
    dtype = cols.dtype
    split = _dekker_split_factor(dtype)
    left_grid, right_grid = cols[..., 0], cols[..., 1]
    left_value, right_value = cols[..., 2], cols[..., 3]

    left_weight = _exact_difference(right_grid, q)
    right_weight = _exact_difference(q, left_grid)
    width = _exact_difference(right_grid, left_grid)

    numerator = _dyadic_join(
        _dyadic_product(_as_dyadic(left_value), left_weight, split),
        _dyadic_product(_as_dyadic(right_value), right_weight, split),
    )

    # Canonical orientation. Endpoints may be stored in either order within a
    # branch; negating both numerator and denominator leaves `V` unchanged.
    #
    # Read off the STORED endpoints, not off a leading term. Under the round-13
    # representation the width's first term was the rounded difference, whose
    # sign was the width's sign; it is now the frexp mantissa of `right_grid`,
    # whose sign is that endpoint's. A comparison of two finite floats is exact
    # and needs no difference to be formed at all.
    flip = right_grid < left_grid
    numerator = _dyadic_negate(numerator, flip)
    denominator = _dyadic_negate(width, flip)

    node, use_right = _node_selection(
        q=q,
        left_grid=left_grid,
        right_grid=right_grid,
        left_value=left_value,
        right_value=right_value,
    )
    node_value = jnp.where(use_right, right_value, left_value)
    zeros = jnp.zeros_like(node_value)
    is_node = node[..., None]
    node_numerator = jnp.stack(
        [node_value, *([zeros] * (numerator.mantissa.shape[-1] - 1))], axis=-1
    )
    node_denominator = jnp.stack([jnp.ones_like(node_value), zeros], axis=-1)
    return _ExactRatio(
        numerator=_Dyadic(
            mantissa=jnp.where(is_node, node_numerator, numerator.mantissa),
            exponent=jnp.where(
                is_node, jnp.zeros_like(numerator.exponent), numerator.exponent
            ),
        ),
        denominator=_Dyadic(
            mantissa=jnp.where(is_node, node_denominator, denominator.mantissa),
            exponent=jnp.where(
                is_node, jnp.zeros_like(denominator.exponent), denominator.exponent
            ),
        ),
    )


def _exact_compare(*, cols_a: FloatND, cols_b: FloatND, q: FloatND) -> FloatND:
    """Exact sign of `V_a(q) - V_b(q)`: `+1`, `-1`, or `0` for a TRUE tie.

    `V_a - V_b = (N_a*D_b - N_b*D_a) / (D_a*D_b)` with both denominators
    positive, so the sign is that of the cross-multiplied numerator difference.
    This is the only test that can tell a genuine tie from a strict gap finer
    than the working precision — double-double values cannot, because
    algebraically different segment parameterizations of the SAME exact value
    need not produce the same low word (round-6 audit F2).

    The columns are handed over RAW. Every difference is formed on them, every
    product carries its own exponent, and the sum is accumulated in fixed point:
    there is no scale to choose, hence no scale that can be chosen wrongly.
    """
    return _exact_cross_sign(
        a=_exact_ratio(cols=cols_a, q=q),
        b=_exact_ratio(cols=cols_b, q=q),
        dtype=cols_a.dtype,
    )


def _exact_slope_ratio(*, cols: FloatND) -> _ExactRatio:
    """A candidate's value slope as an exact rational `(v1-v0)/(x1-x0)`.

    Both differences are dyadic pairs, so the ratio carries no rounding at all.
    The denominator is canonically positive so a cross-multiplied comparison
    reads its sign off the numerator difference.

    A slope's numerator and denominator scales do NOT cancel within one candidate
    — they change the ratio — which used to force BOTH on the caller as shared
    exponents. Carrying the exponent per term removes that coupling: neither
    scale is chosen at all, so neither can be chosen wrongly.
    """
    left_grid, right_grid = cols[..., 0], cols[..., 1]
    left_value, right_value = cols[..., 2], cols[..., 3]

    numerator = _exact_difference(right_value, left_value)
    width = _exact_difference(right_grid, left_grid)

    # A zero-width segment has no grid span; the established convention publishes
    # its raw value jump as the slope, i.e. a unit denominator. Under exponent
    # tagging that unit is literally `1 * 2**0`, so the convention is exact by
    # construction rather than by surviving a scale choice.
    zero_width = (right_grid == left_grid)[..., None]
    ones = jnp.ones_like(width.mantissa[..., :1])
    unit = jnp.concatenate([ones, jnp.zeros_like(width.mantissa[..., 1:])], axis=-1)
    width = _Dyadic(
        mantissa=jnp.where(zero_width, unit, width.mantissa),
        exponent=jnp.where(zero_width, jnp.zeros_like(width.exponent), width.exponent),
    )

    # As in `_exact_ratio`: the orientation is a property of the STORED
    # endpoints, read by an exact float comparison rather than off a term.
    flip = right_grid < left_grid
    return _ExactRatio(
        numerator=_dyadic_negate(numerator, flip),
        denominator=_dyadic_negate(width, flip),
    )


def _exact_slope_compare(*, cols_a: FloatND, cols_b: FloatND) -> FloatND:
    """Exact sign of `slope_a - slope_b`: `+1`, `-1`, or `0` for a TRUE tie.

    Same cross-multiplied construction as `_exact_compare`, on the slope ratio
    instead of the value.

    This exists because an exact VALUE tie is only half the rule. The
    right-continuous break then orders slopes, and ordering them on
    `fl((v1-v0)/(x1-x0))` re-introduces the very defect the exact value predicate
    removed one operation earlier: two strictly ordered exact slopes can share a
    single float key, and `argmax` then silently falls back to candidate order,
    so a pure branch permutation flips the published policy and marginal
    (round-7 audit F2).
    """
    return _exact_cross_sign(
        a=_exact_slope_ratio(cols=cols_a),
        b=_exact_slope_ratio(cols=cols_b),
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
       events, which publish stored data and so are exact BY CONSTRUCTION, with
       bitwise-equal `(hi, lo)` pairs — skip it, since exactness has nothing to
       add there. That certificate is structural on purpose: see `certified_tie`.

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
    # Skipping the exact comparison needs a STRUCTURAL certificate of exactness,
    # never a numerical one. `terms.exact` is set by how the value was produced —
    # a node event publishes stored data and performs no arithmetic — so a
    # bitwise-equal `(hi, lo)` pair between two such candidates IS an exact tie.
    #
    # The previous rule read `radius == 0` as that certificate, and a radius is a
    # float like any other: near the bottom of the range `eps**2 * |v|` underflows
    # to zero for interior lanes whose value pair also collapsed, so two candidates
    # that are NOT tied presented as certifiably tied and `_exact_compare` — which
    # returns the correct strict sign on exactly that input — was never consulted
    # (round-11 audit F2/RT11). "The radius came out zero" is a statement about
    # the arithmetic's dynamic range, not about the candidate's value.
    #
    # The consequence is the invariant the whole selection now rests on: the
    # approximate layer may RESOLVE an ordering, never CERTIFY an equality. It
    # resolves soundly because `value_hi` is correctly rounded and rounding to
    # nearest is monotone, so `value_hi_a > value_hi_b` implies the exact values
    # are strictly ordered the same way. Everything it cannot separate — every
    # tie, however certain it looks — is decided by `_exact_compare`.
    lead_exact = jnp.take_along_axis(terms.exact, lead[:, None], axis=1)
    certified_tie = dd_tied & terms.exact & lead_exact

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
    events, O(eps^2) interior). `exact` records STRUCTURALLY — by how the value
    was produced, not by inspecting it — whether the pair is the exact candidate
    value; see `_exactly_maximal`. `policy`/`marginal` are the candidate's outputs
    at the query, `slope` its value-slope, and `right_available` whether it
    extends strictly right of the query — the right-continuous tie-break keys.
    """

    brackets: BoolND
    value_hi: FloatND
    value_lo: FloatND
    radius: FloatND
    exact: BoolND
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
    # No value scale is needed to keep the intermediates BOUNDED. A bracketing
    # candidate has `q` in `[x0, x1]`, hence `|t| <= |w|` and `r` in `[0, 1]`, so
    # `|p| <= |d|` and `v = v0 + p` is bounded by the endpoint values themselves.
    # The unbounded intermediates only ever came from mixing units — a value
    # times a grid difference — which this form never constructs.
    #
    # Boundedness is not the whole requirement, though, and reading it as if it
    # were is what left the value axis unscaled through round 11 (round-11 audit
    # F2). The value axis needs a scale for the OPPOSITE reason to the grid axis:
    # not because its intermediates can grow, but because they can vanish. See
    # the value lift below.
    zero_width = right_grid == left_grid
    split = _dekker_split_factor(block.dtype)

    # ... and the ONE thing a shift of the OPERANDS can never deliver is safety
    # in both directions at once, which is the round-13 finding. A difference of
    # finite floats can UNDERFLOW — near the bottom of the range the gap between
    # two distinct normals is subnormal and XLA flushes it, so `q - x0` came back
    # exactly `0.0` and the interpolant collapsed onto its left value (round-9
    # audit MT6). Rounds 9 to 13 answered that by LIFTING, never lowering, and
    # asserted that subtraction of finite floats cannot overflow. It can: for
    # `H = 2**127` in float32 the operands already exceed any target, the lift is
    # zero, and `H - (-H)` is `+inf`. A finite exact envelope was published as
    # three NaNs (round-13 audit F2, MT10: 288 of 288 generated cells).
    #
    # No choice of shift closes both ends, because the spread the operands can
    # span is the whole format and no single frame holds it. So the difference is
    # not formed in the working dtype at all: `_framed_difference` divides each
    # PAIR by its own `2**max(e_a, e_b)` — exact, by `ldexp` on the `frexp`
    # mantissa — and returns the pair's exponent alongside the double-double
    # head and tail. Both operands are then in `(-1, 1]`, where `_two_diff`
    # cannot overflow and cannot lose its residual, at any model scale.
    #
    # Each difference carries its OWN frame, so the grid axis needs no common
    # shift for `q` and the two nodes: `r = t/w` recombines them through the
    # integer exponents, where the round-10 hazard of merging two distinct grid
    # points onto one float cannot arise because no operand is ever lowered
    # relative to another it is compared against.
    th, tl, t_frame = _framed_difference(q, left_grid)
    wh, wl, w_frame = _framed_difference(right_grid, left_grid)
    wh = jnp.where(zero_width, jnp.ones_like(wh), wh)
    wl = jnp.where(zero_width, jnp.zeros_like(wl), wl)
    w_frame = jnp.where(zero_width, jnp.zeros_like(w_frame), w_frame)

    # The value axis takes the same treatment, and for BOTH reasons. Downward:
    # with `v0 = tiny` and `v1 = nextafter(tiny)` the endpoint gap `d` used to
    # flush and a strictly lower branch took the policy and the marginal
    # (round-11 audit F2, MT8). Upward: with `v0 = -H` and `v1 = H` the gap `2H`
    # is not representable at all, though every endpoint and the interpolated
    # value are. In the framed form `d` is `(1.0, 0.0)` at exponent `maxexp`, and
    # nothing overflows.
    dh, dl, d_frame = _framed_difference(right_value, left_value)

    # Renormalize each head into `[0.5, 1)` before Dekker sees it, and fold the
    # shift into the integer exponent. This is the round-8 requirement — the
    # split constant `2**s + 1` needs `|a| < 2**(emax - s)` — now applied to a
    # quantity that is already O(1), so it is conditioning rather than rescue.
    t_exp = _binade_exponent(jnp.abs(th)) + t_frame
    w_exp = _binade_exponent(jnp.abs(wh)) + w_frame
    d_exp = _binade_exponent(jnp.abs(dh)) + d_frame
    t_shift = t_frame - t_exp
    w_shift = w_frame - w_exp
    d_shift = d_frame - d_exp

    rh, rl = _dd_div(
        jnp.ldexp(th, t_shift),
        jnp.ldexp(tl, t_shift),
        jnp.ldexp(wh, w_shift),
        jnp.ldexp(wl, w_shift),
        split,
    )
    ph, pl = _dd_mul(rh, rl, jnp.ldexp(dh, d_shift), jnp.ldexp(dl, d_shift), split)
    product_exp = t_exp - w_exp + d_exp

    # `v = v0 + r*d` in a frame that holds BOTH addends. Round 13 applied the
    # product's exponent to `ph` and added the raw `v0`, which overflows whenever
    # the product does even though the sum need not: at `v0 = -H`, `v1 = H`,
    # `r = 3/4` the true value is `H/2` while `r*d` is `1.5 * H`. Anchoring on
    # the larger of the two exponents keeps every addend in `(-2, 2)` and defers
    # the single rounding to publication.
    #
    # For a BRACKETING candidate `|t| <= |w|`, so `r` is in `[0, 1]` and `|v|` is
    # bounded by the endpoint values: `ldexp(vh, value_frame)` therefore lands
    # back inside the binade the endpoints came from and is finite whenever they
    # are. Scaling a normal by a power of two is exact, so `value_hi` is the
    # framed computation's correctly rounded value, rounded ONCE, in ORIGINAL
    # units — comparable across candidates that framed differently.
    #
    # `value_lo` and the radius are scaled back by the same shift and may flush
    # to zero there. That costs resolution, never correctness: a flushed residual
    # says the remainder sits below the representable grid, and under the
    # structural-exactness rule in `_exactly_maximal` a `(hi, lo)` tie is never a
    # certificate — it routes to `_exact_compare` — while a strict difference in
    # a correctly rounded `value_hi` certifies a strict difference in the exact
    # values, because rounding to nearest is monotone.
    _, left_value_exponent = _dyadic_parts(left_value)
    value_frame = jnp.maximum(product_exp, left_value_exponent)
    framed_ph = jnp.ldexp(ph, product_exp - value_frame)
    framed_pl = jnp.ldexp(pl, product_exp - value_frame)
    framed_left_value = jnp.ldexp(left_value, -value_frame)
    vh, vl = _dd_add_fp(framed_ph, framed_pl, framed_left_value)

    eps = jnp.finfo(block.dtype).eps
    interior_radius = (
        _INTERIOR_RADIUS_ULPS2
        * eps
        * eps
        * (jnp.abs(framed_left_value) + jnp.abs(framed_ph))
    )

    vh = jnp.ldexp(vh, value_frame)
    vl = jnp.ldexp(vl, value_frame)
    interior_radius = jnp.ldexp(interior_radius, value_frame)

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
        # A node event's value IS stored data: no arithmetic was performed, so
        # the pair is exact by construction. Nothing else here is.
        exact=node,
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
        # same units.
        #
        # Built from the framed differences rather than from raw subtractions.
        # "One subtraction over another carries no overflow risk of its own" was
        # the same false hinge as in `_exact_difference`: with `v0 = -H`,
        # `v1 = H` the numerator alone is `+inf`, and against an equally
        # overflowing width it becomes NaN — a key the round-13 sentinel repair
        # in `_right_continuous_winner` then has to demote, losing the ordering
        # rather than getting it right. Here the mantissa ratio is O(1) and the
        # magnitude rides in the integer exponent, so the key is finite whenever
        # the mathematical slope is, and infinite only when it genuinely is.
        slope=jnp.ldexp(jnp.ldexp(dh, d_shift) / jnp.ldexp(wh, w_shift), d_exp - w_exp),
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

    # `-inf` is the MASK sentinel here, and it is also a REACHABLE rounded key:
    # `slope` is `fl(v1-v0) / fl(x1-x0)` in original units, so a large value gap
    # over a small grid width overflows to `-inf` with every stored input still
    # finite normal. A plain `argmax` over `key` then cannot tell the only
    # competitor from the candidates the mask excluded, and returns index 0 —
    # a candidate that is not competing at all, which the loop below seeds as
    # the lead and never revisits. The published policy and marginal are then
    # whichever branch happens to sit first, so a pure branch permutation flips
    # them while the value stays right: the round-7 F2 signature, reached
    # through the sentinel rather than through a shared float key.
    #
    # Selecting the first COMPETING candidate that attains the maximum keeps the
    # documented earliest-on-tie rule and makes the sentinel unreachable as an
    # answer. NaN keys are folded onto the sentinel for the same reason — they
    # must not be allowed to win a comparison the screen cannot make — and
    # `contends` below then hands every one of them to the exact predicate.
    key = jnp.where(compete, terms.slope, -jnp.inf)
    rankable = jnp.where(jnp.isnan(key), -jnp.inf, key)
    lead = jnp.argmax(
        compete & (rankable == jnp.max(rankable, axis=1, keepdims=True)), axis=1
    )
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
    # Whether the carried winner's value pair is exact by construction, carried
    # for the same reason the radius is: `_combine_blocked` runs the shared
    # selection rule, and that rule may only skip the exact comparison on a
    # structural certificate. A winner interpolated in a block's interior is not
    # exact, and the fold must not treat it as such just because two blocks
    # produced the same rounded pair.
    exact: BoolND
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
        exact=jnp.stack([carry.exact, block.exact], axis=1),
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
        exact=jnp.where(take, block.exact, carry.exact),
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
            exact=jnp.take_along_axis(t.exact, column, axis=1)[:, 0],
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
            # The seed carries no winner (`has_winner` is False), so it never
            # reaches a comparison; claiming exactness for it would be a lie the
            # fold could only ever act on by mistake.
            exact=jnp.zeros((n_query,), dtype=bool),
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
