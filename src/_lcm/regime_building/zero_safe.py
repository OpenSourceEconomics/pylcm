"""Zero-weight-safe arithmetic on the extended reals for the collective solve core.

On-path ``-inf`` is admissible throughout the collective-regimes extension (a
feasible zero-consumption state, an all-infeasible dissolution cell whose
value is masked to ``-inf`` before being folded/averaged away, ...), and an
exact-zero weight is equally admissible (an inactive regime-transition target,
a zero-probability `MarkovTransition` node, a zero-weight quadrature node, an
on-grid interpolation corner, a zero Pareto weight). Whenever such a weight
multiplies such a value, naive floating-point arithmetic computes
``0.0 * -inf = nan`` (or ``+inf``), which then poisons whatever sum it feeds —
a continuation expectation, a fold reduction, an interpolated reference value,
or a household scalarization — even though the zero-weight term should
contribute exactly nothing.

This module is now down to ``zero_safe_average``, and it performs no multiply of
its own: the weighted TERM lives in `_lcm.zero_safe` and serves the whole engine.
The copy that used to live here is gone, along with the divergence it accumulated
on every upstream wave — it read a represented zero from a comparison rather than
from the bits, and had no subnormal handling at all. What the term does is
documented there; the history below is kept because it is what established the
rule, and because two of its claims were confidently wrong in ways worth not
repeating.

The fix pattern that term applies: replace the VALUE with an
explicit `0.0` wherever the weight is exactly zero AND the value is NON-FINITE,
via ``jnp.where``, BEFORE multiplying. `weight * where(mask, 0, value)`
annihilates a zero-weight ``+-inf`` (the multiply sees ``w * 0 = 0``, never
``0 * -inf``) AND leaves the
multiply as a bare operation feeding the downstream reduction, which XLA CAN fuse
into an FMA — so the all-positive-weight path is bit-identical to the naive
``jnp.average`` / raw corner sum **on the currently pinned jaxlib**. That identity
is NOT guaranteed: it rests on XLA choosing to contract the multiply into the
reduction's FMA identically for both expressions, which JAX's compatibility
policy explicitly does not promise across releases, backends, or jit contexts
(ROUND-4 CAVEAT below). Where the reduction MUST tolerate an exact zero (the
runtime call sites), that is a safety requirement, not a bit-exactness one; the
bit-exactness is a convenient property of the current toolchain, not a contract.

WHY THE MASK IS RESTRICTED TO NON-FINITE VALUES. ``jnp.where`` is a hard select,
so a mask that fires on EVERY zero-weight node also kills ``d/dw`` there: the
branch taken is a constant, whose derivative w.r.t. the weight is ``0`` rather
than ``value``. That is invisible while the weight is a CONSTANT of the
differentiation — a Pareto weight, a transition probability, a quadrature
weight, i.e. every call site this module was written for — and WRONG the moment
the weight is itself a function of the argument being differentiated. An
interpolation corner weight is exactly that: applying the unrestricted mask
inside `map_coordinates` made ``jax.grad`` return ``-grid[c]`` instead of the
segment slope at every on-node coordinate, with the VALUES still correct and so
nothing but a gradient test able to see it. Restricting the mask costs nothing:
for finite ``v``, ``0 * v == 0`` either way, so values are NUMERICALLY EQUAL to
the unrestricted form, and only the genuine ``0 * +-inf`` case still selects,
which has no finite derivative to preserve. See
`tests/regime_building/test_zero_safe_gradients.py`.

SIGNED-ZERO EXCEPTION (round-3 audit H2; this docstring said "bit-identical" and
that was WRONG — the third time a confident identity claim here failed to an
outside check, hence the HISTORY section below). At ``w = +0`` with a NEGATIVE
finite ``v``, the restricted form returns ``-0.0`` where the unrestricted mask
returned ``+0.0``: the mask no longer fires, so the sign of the product is the
sign of ``v``. They compare equal, have the same derivative, and any reduction
consuming them is byte-for-byte unchanged (``sum([-0.0, 3.0])`` and
``sum([+0.0, 3.0])`` agree exactly), so no decision moves. **The claim was wrong
because the CHECK was wrong**: it used `jnp.array_equal`, and ``+0.0 == -0.0`` is
True, so it could not observe the difference it asserted the absence of. If you
claim bit identity, compare BITS (`np.signbit`, `.tobytes()`), not values.

HISTORY (this docstring was wrong twice; both errors are recorded because each
was a confident claim no test could contradict, and an external re-review broke
both by running code). The original guard masked the PRODUCT
(``where(weight==0, 0, weight*value)``). That `select` sits AFTER the multiply
and blocks FMA contraction, so the all-positive path rounds twice where the
naive path rounds once. This docstring then claimed the drift was (a) "~1 ULP",
(b) "not decision-relevant", and (c) unfixable without a global ``lax.cond``.
All three were false, MEASURED (jax 0.10.x/0.9.x, CPU, float32):

- (a) The drift reaches **6 ULP** on a valid all-positive probability vector
  under cancellation, not ~1: with nodes ``[-3.9480734, 2.6238020]`` and
  probabilities ``[0.38403073, 0.61596930]`` (sum exactly 1), the masked-PRODUCT
  average returns bits ``1036831952`` where ``jnp.average`` returns
  ``1036831946`` — six apart, and on the WRONG side of the exact real mean.
- (b) That reverses a **non-tied** discrete action: against a deterministic
  alternative ``0.1``, the exact/naive value is below ``0.1`` (choose the
  alternative) while the masked-product value is above it (choose the stochastic
  action). The same 1-ULP+ drift on the interpolation path reverses an E2
  individual-rationality comparison and hence the dissolution flag ``D`` (which
  is ``~any(final_mask)`` — any finite IR flip flips it). No exact tie is
  required: if the IR margin has positive density near 0, any nonzero
  perturbation flips a positive-probability band.
- (c) Masking the VALUE (this form) restores the naive bits on the all-positive
  path with NO global predicate and NO ``lax.cond`` — MEASURED 0 drift vs
  ``jnp.average`` across K in {2,3,4,7,8,16} x {float32,float64}, and 0 drift
  vs the raw corner sum across 5000 off-grid interpolation coordinates (where
  the masked-product form drifted on 802/5000). Cost is ~6% over ``jnp.average``
  under plain ``vmap`` — versus ~4.5x for the ``lax.map``-batched scalar
  ``lax.cond`` the re-review proposed; that alternative also works but is a far
  heavier hit on a solve-core hot path, so it was not taken.

Statically-known weights were always constant-folded and matched bit-for-bit;
the FOLD reduction remains the one call site that binds `jnp.average` vs
`zero_safe_average` at build time via `max_Q_over_a._select_fold_reducer` (its
weights are the process quadrature marginal, concrete before tracing). Runtime
call sites use `zero_safe_average` unconditionally; they do NOT get guaranteed
bit-exactness — the value-masking order recovers raw bits on SOME toolchains/CPUs
and not others (ROUND-5 CAVEAT: a dynamic-per-cell reversal was reproduced on one
0.10.1 CPU and not another). Read the runtime contract as the HONEST CONTRACT
below, not as bit-identity.

ROUND-4 CAVEAT (external re-review, corrects the (c) claim above). The "0 drift
across K in {2,3,4,7,8,16}" sweep OMITTED K=5, and that is exactly where the
identity can fail: the re-review exhibited a K=5 float32 all-positive vector on
which `jit(vmap(zero_safe_average))` drifts SEVEN ULP from `jit(vmap(jnp.average))`
on CPU jax 0.9.0.1, reversing a constructed non-tied argmax. Reproduce-first on
the currently pinned jaxlib (0.10.1, this CPU) did NOT reproduce it — 0 drift on
the exact K=5 counterexample, the five-target mixture, and a nested 32x32x32 vmap
— so there is no live decision reversal here. But the drift IS reachable on
another supported toolchain, so the bit-identity is a property of the current XLA
lowering, not a guarantee. The durable fix (unnecessary while no live reversal
exists) is a WHOLE-EXPRESSION branch: raw expression on all-positive slices, safe
expression when any exact zero occurs, mirroring `_select_fold_reducer`'s
build-time selection so the all-positive path is exact-to-raw BY CONSTRUCTION,
independent of FMA behavior. Do NOT restore an unconditional "bit-identical"
claim: measure, per (K, dtype, jaxlib, backend), before asserting identity.

ROUND-5 CAVEAT (external re-review, corrects the round-4 "no live reversal here").
The round-4 non-reproduction (and a within-session re-check) exercised SHARED /
closed-over weight vectors, which XLA constant-folds — so they cannot exhibit the
runtime path, where each product-map cell carries its own DYNAMICALLY computed
weight vector. The re-review reproduced a SEVEN-ULP reversal on official jax/jaxlib
0.10.1 CPU by broadcasting the K=5 all-positive vector over a 32x32x8 carrier and
compiling raw vs `zero_safe_average` through nested `vmap`s with the WEIGHTS as
mapped inputs: raw 0.08923107385635376 vs guarded 0.08923112601041794, with the
exact real mean on the raw side and every one of the 8192 cells flipping a non-tied
argmax. HOWEVER, reproduce-first on THIS machine (jax/jaxlib 0.10.1, hmg-office CPU)
with the reviewer's EXACT dynamic-per-cell nested-vmap setup still gives 0 drift:
raw `jnp.average`, the explicit `sum(w*v)/sum(w)`, and `zero_safe_average` ALL
return bits 1035386575 here. So raw and guarded agree on this CPU and disagree on
the reviewer's — the drift is CPU-microarchitecture / XLA-lowering dependent, NOT
merely jaxlib-version or weight-shape dependent. That is the real content of the
finding: `raw jnp.average` is ITSELF not bit-portable across CPUs, so "bit-identical
to raw" was never a portable contract.

ROUND-6 CAVEAT (reproduce-first on a divergence-reproducing GPU; RETIRES the
"durable whole-expression branch" as unachievable and names the real resolution).
The round-4/5 caveats floated a whole-expression branch — raw `sum(w*a)` on all-
positive slices, masked only where an exact zero occurs — as the durable fix that
would be "exact-to-raw BY CONSTRUCTION." It was BUILT and MEASURED on gb10 (aarch64,
jax 0.11.0, GPU backend, which DOES reproduce the divergence: raw `jnp.average` bits
1035386569 vs masked 1035386568 on the 32x32x8-carried K=5 vector). Result: under the
solve's nested `vmap` the branch does NOT recover the raw bits. Mechanism, isolated
by a 4-way probe on the SAME backend: the raw reduction returns 1035386569 ONLY when
it is the sole reduction in the graph (`n=sum(w*a); d=sum(w); n/d`, no mask); the
moment the value-masked reduction is materialised on any co-path — as it must be for
zero-mass safety — XLA co-fuses the two reductions and the raw one COLLAPSES onto the
masked 1035386568. `jax.lax.optimization_barrier` on the reduced numerators does not
isolate them (it independently yields the masked bits); `jax.lax.cond` vmaps to the
same select. So every safety-preserving structure lands on the masked bits under
vmap — confirming the reviewer's list (cond / select-after-both / lax.map /
optimization_barrier all fail) by direct MEASUREMENT, not report.

The same probe fixes the resolution FOR THE AVERAGE HELPER: in float64 the masked
`zero_safe_average` is BIT-IDENTICAL to the exact `jnp.average` (relative diff 0.00e+00,
both 0.089231097529865119) on the very carrier that diverges 1 ULP in float32. The
divergence is therefore a float32 rounding-FLOOR artifact, not an expression-structure
bug — and `raw jnp.average` is itself not bit-portable across CPUs (ROUND-5), so "match
raw in float32" was never a well-posed target. The sound remedy is to solve the
collective core in FLOAT64 (the test conftest already enables x64 — precision is part of
the model spec), which removes the AVERAGE-helper divergence. No expression rewrite is
pursued. (See ROUND-7: float64 does NOT extend this bit-identity to the sequential
regime-MIXTURE accumulation, and "tied at float32 precision" is imprecise — corrected
below.)

ROUND-7 CAVEAT (external re-review of the round-6 disposition; NARROWS the float64
claim from "the collective core is resolved" to "the AVERAGE HELPER is resolved",
and corrects the "tie" wording). Round 6 MEASURED float64 bit-identity for
`zero_safe_average` (a single vectorised `sum(w*a)/sum(w)` reduction) and I then
over-generalised it to "solve the collective core in float64 removes the
discrepancy." Two things are wrong with that generalisation, both confirmed
reproduce-first:

- The runtime collective core has a SECOND, structurally DIFFERENT consumer:
  `Q_and_F` accumulates the regime mixture as a SEQUENTIAL left-fold
  `E = 0; for r in targets: E += zero_safe_weighted_term(p_r, V_r)` — NOT a call to
  `zero_safe_average`. With all-positive `p_r` the mask is the identity, so this is
  pure reduction-ORDER, and float64 does NOT make it correctly-rounded to the exact
  mixture. MEASURED under jit (the real solve path) on hmg-office CPU jax 0.10.1, on
  a valid all-positive 5-target float64 mixture: the fold returns bits
  4583286125422516842 — 9 ULP BELOW the exact real mixture (bits ...851) and on the
  WRONG side of a representable knife-edge alternative (bits ...843), reversing the
  downstream argmax relative to exact. The external re-review reproduced the same
  direction (fold low, wrong side) on jax 0.9.0.1 / 0.10.1 / 0.11.0 CPU. CRUCIALLY,
  consolidating the ALREADY-MULTIPLIED products as `jnp.sum(jnp.stack(products))`
  returns the IDENTICAL wrong-side bits ...842 as the left-fold under jit
  (MEASURED). ROUND-7 wrongly concluded from this that "no source restructuring
  fixes it" — see the ROUND-8 UPDATE below, which corrects it: stacking the
  UNMULTIPLIED OPERANDS (p_r and V_r into two arrays) and doing ONE zero-safe
  contraction DOES land on the exact side, and the real code CAN build that form.
  Beware the trap: an eager (non-jit) run of the same fixture gives bits ...848 for
  every form and hides the reversal — the divergence only appears under jit, so
  validate on the jitted path.
- "An action that flips under a few-ULP perturbation is TIED at float32 precision"
  is imprecise. The exact real average has a UNIQUE correctly-rounded float32 value;
  the flip is an ill-conditioned NEAR-tie whose correctly-rounded resolution a
  backend's reduction error can land on the WRONG side of (reviewer: exact rounds to
  float32 bits 1035386571, the guarded reduction returns 1035386575 — a determinate
  boundary crossed, not an equality). The decision cost is still ULP-level and still
  smaller in float64, but call it a near-tie, not a tie.

ROUND-8 UPDATE (external re-review of the round-7 disposition; SUPERSEDES the "no
source restructuring fixes it" claim above, and corrects the error-magnitude wording).
Two round-7 statements were wrong, both confirmed reproduce-first on hmg-office CPU:

- "NOT fixable by source restructuring" is FALSE. The distinction round-7 missed is
  stacking OPERANDS vs stacking PRODUCTS. The left-fold and `jnp.sum(jnp.stack(
  products))` both accumulate already-multiplied `p_r*V_r` terms and land on the
  wrong side (bits ...842). But stacking the UNMULTIPLIED operands — collect each
  target's `p_r` and `V_r` into two arrays and do ONE `jnp.sum(zero_safe_weighted_
  term(P, V), axis=0)` — lands on the exact-policy side (bits ...858). The real code
  CAN build this: the per-target terms are collected into a list at Python trace time
  and stacked, so a single vectorised contraction replaces the fold. This is now
  `_sum_regime_mixture` in `Q_and_F` (applied at all three mixture sites). It does NOT
  make the result correctly-rounded at every knife-edge, but it (a) crosses to the
  exact side on the pinned fixture and (b) is order-independent (see next point).
- "up to a few ULP of reduction error" understates the worst case. Under CANCELLATION
  (Σ|p_r·V_r| ≫ |Σ p_r·V_r|) the reduction can be HUNDREDS of result-space ULP from
  exact — the honest bound is absolute-plus-relative in the SUMMAND scale Σ|p_r·V_r|,
  not a fixed few result-ULP. And the left-fold's order-dependence meant a mere
  permutation of the target-declaration order could flip a pinned-fixture policy on
  one backend; `_sum_regime_mixture` reduces the per-target contributions in VALUE
  order (round-10; a `jnp.sort` of the zero-safe `p_r*V_r` terms along the target
  axis before the sum), so the result is invariant to declaration order AND to an
  economically-inert alpha-renaming of the regimes. (Round 8 sorted by target NAME,
  which fixed iteration order but still made the float64 bits — and a non-tied argmax
  — a function of the arbitrary regime labels; see `_sum_regime_mixture`.)

HONEST CONTRACT (supersedes every unconditional "bit-identical" statement below).
`zero_safe_average` / `zero_safe_weighted_term` are (i) exact-zero-mass-SAFE
(guaranteed, the load-bearing property) and (ii) in float32, equal to the naive raw
reduction up to a few ULP of reduction error whose sign/magnitude depend on the XLA
lowering and CPU/GPU — the SAME order of non-determinism raw float reductions carry
across backends, NOT removable by any vmap-safe expression restructuring (ROUND-6);
and (iii) in float64, `zero_safe_average` is bit-identical to the exact `jnp.average`
on the cases measured. Guarantee (iii) is scoped to the AVERAGE HELPER — a single
vectorised reduction. The `Q_and_F` regime mixture is now `_sum_regime_mixture`, a
single zero-safe contraction over the STACKED UNMULTIPLIED operands whose per-target
contributions are reduced in VALUE order (ROUND-10: a `jnp.sort` along the target axis
before `jnp.sum`; ROUND-8 sorted by target NAME) — it lands on the exact-policy side of
the pinned knife-edge fixture where the old `E += zero_safe_weighted_term(p_r, V_r)`
left-fold did not, and is invariant to target-declaration order AND to an
economically-inert alpha-renaming of the regimes (the name-sort was not: it left the
float64 bits, and a non-tied argmax, a function of the arbitrary regime labels).
It is still NOT correctly-rounded at every knife-edge: under cancellation
(Σ|p_r·V_r| ≫ |Σ p_r·V_r|) the error is bounded by the
SUMMAND scale, not a fixed few result-ULP, so a float32 (and, at a genuine knife-edge,
even a float64) difference can still flip an ill-conditioned NEAR-tie — NOT an equality;
the exact value has a unique correctly-rounded representative and the reduction error
can land on the wrong side of it — in the downstream argmax / IR comparison. Run the
collective core in float64 to shrink this to a float64-knife-edge event; deterministic
resolution AT a genuine knife-edge would require correctly-rounded/compensated
summation, which is not implemented.
"""

import jax
import jax.numpy as jnp

from _lcm.probability import scaled_down_by_power_of_two
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.typing import FloatND, IntND


def zero_safe_average(
    a: FloatND,
    *,
    weights: FloatND,
    shifts: IntND | None,
    axis: int | None = None,
) -> FloatND:
    """Zero-weight-safe replacement for ``jnp.average(a, weights=weights, axis=axis)``.

    Mirrors `jnp.average`'s own weight-broadcasting contract (`weights` must
    either match `a`'s full shape, when `axis` is `None`, or `a.shape[axis]`,
    when `axis` is a single int) but forms the weighted sum via
    `zero_safe_weighted_term` so a zero-weight node next to an on-path
    `+-inf` value cannot inject a `nan`. Used for the stochastic-node /
    regime-mixture / fold-state weighted averages in the collective solve
    core (`Q_and_F.py`, `max_Q_over_a.py`).

    `shifts` carries the base-two scale each weight is held at — a node's
    probability is ``weights * 2**-shift`` — as `scaled_joint_weight` now
    returns one scale per node rather than one for the whole lottery. Weights
    from different nodes are therefore not comparable as plain floats, and a
    reduction that ignored the scales would weight the nodes by their
    coefficients alone, valuing a near-impossible node as an even chance.

    It has **no default**, following upstream's rule that a coefficient cannot
    be supplied without its scale: a caller states either the scales or `None`,
    and no call can leave the question unasked. `None` means every weight is
    already on one scale, which makes their ratios exact without any of this —
    the fold-state and Pareto-weight reductions, whose weights never passed
    through `scaled_joint_weight` at all.

    `None` is a distinct path rather than a shift array of zeros because the
    two are not numerically equivalent here: scaling by a zero shift still
    costs the FMA contraction, and MEASURED on this fixture it moves the result
    off `jnp.average`'s bits, which
    `test_zero_safe_average_matches_jnp_average_on_the_finite_path` pins. Zeros
    would therefore change the unscaled callers' arithmetic to buy nothing.
    Measured for `jnp.ldexp` and again for `scaled_down_by_power_of_two` when
    the latter replaced it; the two agree, as their bit-for-bit equality on
    non-positive shifts implies.

    The two reductions treat the scale differently, following
    `_expectation_over_stochastic_nodes` in `Q_and_F.py`, against which this is
    tested:

    - the **mass** lowers each weight onto the common scale before summing. A
      live node too far below that scale to register contributes no share of a
      total of order one, which is the right answer for a denominator;
    - the **numerator** forms ``w * a`` first and applies the node's relative
      scale to the product. A tiny probability meeting a large value makes an
      ordinary contribution, and scaling the weight first would flush it before
      the value supplied its binades.

    The common scale is the smallest shift — the largest node — so every
    relative scale is a power of two no greater than one and the lowering
    cannot overflow.

    Unlike `jnp.average`, only `axis=None` or a single `int` `axis` is
    supported — every call site here reduces at most one axis at a time; pass
    a tuple of axes straight to `jnp.average` if a multi-axis weighted
    reduction is ever needed elsewhere.

    Raises:
        ValueError: If `a` and `weights` have inconsistent shapes (same
            contract as `jnp.average`), or if the total weight along `axis`
            is a CONCRETE zero — an average with no supporting mass is
            undefined (`0/0`), not an admissible extended-real value, and
            this is a caller bug (e.g. every node pruned to weight zero), not
            a genuine `+-inf` positive-mass mixture that should be silently
            masked. Under `jax.jit` tracing this check is a no-op (JAX has no
            concrete value to inspect mid-trace); it fires for eager calls
            (tests, the `validate_V` diagnostic path) and is a documented
            best-effort backstop, not a trace-time guarantee.

    """
    a_arr = jnp.asarray(a)
    weights_arr = jnp.asarray(weights)
    shifts_arr = None if shifts is None else jnp.asarray(shifts)

    if a_arr.shape != weights_arr.shape:
        if axis is None:
            msg = "Axis must be specified when shapes of `a` and `weights` differ."
            raise ValueError(msg)
        if weights_arr.shape != (a_arr.shape[axis],):
            msg = (
                "Shape of `weights` must be consistent with shape of `a` "
                "along the specified axis."
            )
            raise ValueError(msg)
        new_shape = tuple(
            a_arr.shape[axis] if i == axis % a_arr.ndim else 1
            for i in range(a_arr.ndim)
        )
        weights_arr = jnp.reshape(weights_arr, new_shape)
        # A scale belongs to its weight, so it is carried through exactly the
        # same reshape rather than broadcast on its own.
        if shifts_arr is not None:
            shifts_arr = jnp.reshape(shifts_arr, new_shape)

    if shifts_arr is None:
        relative_scale = None
        lowered_weights = weights_arr
    else:
        common_shift = (
            jnp.min(shifts_arr)
            if axis is None
            else jnp.min(shifts_arr, axis=axis, keepdims=True)
        )
        # `common_shift` is the SMALLEST shift, so every relative scale is
        # non-positive — which is exactly `scaled_down_by_power_of_two`'s
        # precondition. It agrees with `jnp.ldexp` bit-for-bit on that domain and
        # skips the general `frexp` graph, which here would run twice over the
        # whole value surface.
        relative_scale = (common_shift - shifts_arr).astype(jnp.int32)
        lowered_weights = scaled_down_by_power_of_two(weights_arr, relative_scale)

    total_weight = jnp.sum(lowered_weights, axis=axis)
    _raise_if_concretely_zero(total_weight, context="zero_safe_average")
    # The masked numerator is used unconditionally -- see the ROUND-6 note below and
    # the module CAVEAT. A whole-expression branch that keeps a RAW `sum(w*a)`
    # reduction for all-positive slices was BUILT and MEASURED on a divergence-
    # reproducing backend (gb10 GPU): under the solve's nested vmap it does NOT
    # recover the raw bits. Once the value-masked reduction is materialised on any
    # co-path, XLA co-fuses the two reductions and the raw one collapses onto the
    # masked bits; `optimization_barrier` does not isolate them (it independently
    # yields the masked bits); `lax.cond` vmaps to the same select. The gap from the
    # exact `jnp.average` expression (up to ~6 ULP in float32; bit-IDENTICAL in float64
    # FOR THIS AVERAGE reduction on the measured carriers) is a float32 rounding-floor
    # artifact, not an expression-structure bug, so there is nothing to gain by the
    # extra reductions. NB the float64 bit-identity is a property of THIS single
    # vectorised reduction; the SEQUENTIAL regime-mixture fold in `Q_and_F` is a
    # different reduction order and is not made bit-portable by float64 (ROUND-7).
    terms = zero_safe_weighted_term(
        weight=weights_arr,
        value=a_arr,
        # With scales in hand the weights are `scaled_joint_weight` coefficients,
        # every one of them normal by construction, so the term needs no exponent
        # move. Without them they are whatever the model supplied.
        subnormal_is_accounted_for=shifts_arr is not None,
    )
    if relative_scale is not None:
        # The product, not the weight: see the docstring. `zero_safe_weighted_term`
        # has already made a zero-weight `+-inf` an exact zero, and scaling down
        # returns a zero unchanged.
        terms = scaled_down_by_power_of_two(terms, relative_scale)
    numerator = jnp.sum(terms, axis=axis)
    return numerator / total_weight


def _raise_if_concretely_zero(total_weight: FloatND, *, context: str) -> None:
    """Best-effort eager guard: raise if `total_weight` is a CONCRETE zero.

    Under `jax.jit`, every intermediate touched inside the trace is an
    abstract tracer — JAX's tracing model has no notion of "this value came
    from a closure constant" once inside a `jit` boundary — so attempting to
    convert `total_weight` to a Python bool always raises
    `jax.errors.ConcretizationTypeError` there; caught and ignored, since
    there is nothing to check mid-trace (a `nan` from a genuinely-zero total
    weight would still surface downstream, e.g. via the existing
    `validate_V` NaN diagnostics). Called eagerly (outside `jit`, e.g. from
    a unit test or a non-jitted diagnostic path), this DOES raise.
    """
    try:
        is_zero = bool(jnp.any(total_weight == 0))
    except jax.errors.ConcretizationTypeError:
        return
    if is_zero:
        msg = (
            f"{context}: total weight is exactly zero along the averaged "
            "axis. A weighted average with no supporting mass is undefined "
            "(0/0), not an admissible extended-real value — this indicates "
            "a caller bug (e.g. every stochastic/regime/fold node pruned to "
            "weight zero), not a genuine +-inf positive-mass mixture."
        )
        raise ValueError(msg)
