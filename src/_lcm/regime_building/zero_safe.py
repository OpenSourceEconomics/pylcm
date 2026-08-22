"""Zero-weight-safe arithmetic on the extended reals for the collective solve core.

On-path `-inf` is admissible throughout the collective-regimes extension (a
feasible zero-consumption state, an all-infeasible dissolution cell whose
value is masked to `-inf` before being folded/averaged away, ...), and an
exact-zero weight is equally admissible (an inactive regime-transition target,
a zero-probability `MarkovTransition` node, a zero-weight quadrature node, an
on-grid interpolation corner, a zero Pareto weight). Whenever such a weight
multiplies such a value, naive floating-point arithmetic computes
`0.0 * -inf = nan` (or `+inf`), which then poisons whatever sum it feeds —
a continuation expectation, a fold reduction, an interpolated reference value,
or a household scalarization — even though the zero-weight term should
contribute exactly nothing.

This module holds `zero_safe_average`, and performs no multiply of its own:
the weighted TERM lives in `_lcm.zero_safe` and serves the whole engine, where
what it does is documented.

The fix pattern that term applies: replace the VALUE with an
explicit `0.0` wherever the weight is exactly zero AND the value is NON-FINITE,
via `jnp.where`, BEFORE multiplying. `weight * where(mask, 0, value)`
annihilates a zero-weight `+-inf` (the multiply sees `w * 0 = 0`, never
`0 * -inf`) AND leaves the
multiply as a bare operation feeding the downstream reduction, which XLA CAN fuse
into an FMA — so the all-positive-weight path is bit-identical to the naive
`jnp.average` / raw corner sum **on the currently pinned jaxlib**. That identity
is NOT guaranteed: it rests on XLA choosing to contract the multiply into the
reduction's FMA identically for both expressions, which JAX's compatibility
policy explicitly does not promise across releases, backends, or jit contexts
(see PORTABILITY below). Where the reduction MUST tolerate an exact zero (the
runtime call sites), that is a safety requirement, not a bit-exactness one; the
bit-exactness is a convenient property of the current toolchain, not a contract.

WHY THE MASK IS RESTRICTED TO NON-FINITE VALUES. `jnp.where` is a hard select,
so a mask that fires on EVERY zero-weight node also kills `d/dw` there: the
branch taken is a constant, whose derivative w.r.t. the weight is `0` rather
than `value`. That is invisible while the weight is a CONSTANT of the
differentiation — a Pareto weight, a transition probability, a quadrature
weight, i.e. every call site this module was written for — and WRONG the moment
the weight is itself a function of the argument being differentiated. An
interpolation corner weight is exactly that: applying the unrestricted mask
inside `map_coordinates` made `jax.grad` return `-grid[c]` instead of the
segment slope at every on-node coordinate, with the VALUES still correct and so
nothing but a gradient test able to see it. Restricting the mask costs nothing:
for finite `v`, `0 * v == 0` either way, so values are NUMERICALLY EQUAL to
the unrestricted form, and only the genuine `0 * +-inf` case still selects,
which has no finite derivative to preserve. See
`tests/regime_building/test_zero_safe_gradients.py`.

SIGNED ZERO. At `w = +0` with a NEGATIVE finite `v`, the restricted form
returns `-0.0` where an unrestricted mask returns `+0.0`: the mask does not
fire, so the sign of the product is the sign of `v`. The two compare equal,
have the same derivative, and any reduction consuming them is byte-for-byte
unchanged (`sum([-0.0, 3.0])` and `sum([+0.0, 3.0])` agree exactly), so no
decision moves. This is an exception to bit-identity, not to numerical
equality — and only a check that looks at BITS (`np.signbit`, `.tobytes()`) can
see it at all. `jnp.array_equal` cannot, because `+0.0 == -0.0` is True.

WHY THE MASK GOES ON THE VALUE, NOT THE PRODUCT. Masking the product
(`where(weight == 0, 0, weight * value)`) puts the `select` AFTER the
multiply, which blocks FMA contraction, so the all-positive path rounds twice
where the naive path rounds once. That is decision-relevant, MEASURED (jax
0.9.x/0.10.x, CPU, float32):

- The drift reaches **6 ULP** on a valid all-positive probability vector under
  cancellation: with nodes `[-3.9480734, 2.6238020]` and probabilities
  `[0.38403073, 0.61596930]` (sum exactly 1), the masked-PRODUCT average
  returns bits `1036831952` where `jnp.average` returns `1036831946` —
  six apart, and on the WRONG side of the exact real mean.
- That reverses a **non-tied** discrete action: against a deterministic
  alternative `0.1`, the exact/naive value is below `0.1` (choose the
  alternative) while the masked-product value is above it (choose the
  stochastic action). The same 1-ULP+ drift on the interpolation path reverses
  an individual-rationality comparison and hence the dissolution flag `D`
  (`~any(final_mask)` — any finite IR flip flips it). No exact tie is
  required: if the IR margin has positive density near 0, any nonzero
  perturbation flips a positive-probability band.
- Masking the VALUE (this form) restores the naive bits on the all-positive
  path with NO global predicate and NO `lax.cond` — MEASURED 0 drift vs
  `jnp.average` across K in {2,3,4,7,8,16} x {float32,float64}, and 0 drift
  vs the raw corner sum across 5000 off-grid interpolation coordinates (where
  the masked-product form drifted on 802/5000). Cost is ~6% over
  `jnp.average` under plain `vmap`, against ~4.5x for a `lax.map`-batched
  scalar `lax.cond`, which also works but is far heavier on a solve-core hot
  path.

PORTABILITY. That 0-drift sweep is not a guarantee, and K=5 is where the
identity fails: a K=5 float32 all-positive vector drifts SEVEN ULP between
`jit(vmap(zero_safe_average))` and `jit(vmap(jnp.average))` on CPU jax 0.9.0.1,
reversing a constructed non-tied argmax, and on gb10 (aarch64, jax 0.11.0, GPU)
raw `jnp.average` returns bits 1035386569 against the masked 1035386568 on a
32x32x8-carried K=5 vector. The same vector on hmg-office CPU jax 0.10.1 gives
0 drift across raw, the explicit `sum(w*v)/sum(w)` and `zero_safe_average`
alike (all bits 1035386575). So the divergence is CPU-microarchitecture and
XLA-lowering dependent, not merely a function of jaxlib version or weight
shape, and `raw jnp.average` is ITSELF not bit-portable across CPUs — "match
raw in float32" was never a well-posed target. Measure per (K, dtype, jaxlib,
backend) before asserting any identity. Note also that a SHARED or closed-over
weight vector is constant-folded by XLA and so cannot exhibit the runtime path
at all; a reproduction needs each product-map cell to carry its own DYNAMICALLY
computed weights.

A whole-expression branch — the raw expression on all-positive slices, the safe
one only where an exact zero occurs — does NOT recover the raw bits, and is not
worth building. MEASURED on the divergence-reproducing GPU backend by a 4-way
probe: the raw reduction returns its own bits ONLY when it is the sole
reduction in the graph; the moment the value-masked reduction is materialised
on any co-path — as zero-mass safety requires — XLA co-fuses the two and the
raw one COLLAPSES onto the masked bits. `jax.lax.optimization_barrier` on the
reduced numerators does not isolate them, and `jax.lax.cond` vmaps to the same
select. Every safety-preserving structure lands on the masked bits under vmap.

Statically-known weights are constant-folded and match bit-for-bit; the FOLD
reduction is the one call site that binds `jnp.average` vs `zero_safe_average`
at build time via `max_Q_over_a._select_fold_reducer` (its weights are the
process quadrature marginal, concrete before tracing). Runtime call sites use
`zero_safe_average` unconditionally and do NOT get guaranteed bit-exactness.

THE CONTRACT. `zero_safe_average` / `zero_safe_weighted_term` are:

1. exact-zero-mass-SAFE — guaranteed, and the load-bearing property;
2. in float32, equal to the naive raw reduction up to a few ULP of reduction
   error whose sign and magnitude depend on the XLA lowering and the CPU/GPU —
   the same order of non-determinism raw float reductions carry across
   backends, and not removable by any vmap-safe expression restructuring;
3. in float64, bit-identical to the exact `jnp.average` on the cases measured
   (relative diff 0.00e+00, both 0.089231097529865119, on the very carrier that
   diverges 1 ULP in float32). Scoped to the AVERAGE HELPER — a single
   vectorised reduction — and NOT extending to the regime mixture below.

The `Q_and_F` regime mixture is `_sum_regime_mixture`, a single zero-safe
contraction over the STACKED UNMULTIPLIED operands whose per-target
contributions are reduced in VALUE order (a `jnp.sort` along the target axis
before `jnp.sum`). Two properties follow, both MEASURED under jit on a valid
all-positive 5-target float64 mixture:

- It lands on the exact-policy side of a pinned knife-edge fixture (bits ...858
  against an alternative at ...843, exact at ...851) where accumulating
  ALREADY-MULTIPLIED terms does not — a sequential left-fold
  `E += zero_safe_weighted_term(p_r, V_r)` and `jnp.sum(jnp.stack(products))`
  both return the identical wrong-side bits ...842. Stacking OPERANDS rather
  than PRODUCTS is the whole distinction. Beware the trap: an eager (non-jit)
  run of the same fixture returns bits ...848 for every form and hides the
  reversal, so validate on the jitted path.
- It is invariant to target-declaration order AND to an economically-inert
  alpha-renaming of the regimes. Sorting by target NAME instead fixes iteration
  order but leaves the float64 bits — and a non-tied argmax — a function of the
  arbitrary regime labels.

It is still NOT correctly-rounded at every knife-edge: under cancellation the
error is bounded by the SUMMAND scale, not by a fixed few result-ULP, so a
float32 — and, at a genuine knife-edge, even a float64 — difference can still
flip an ill-conditioned NEAR-tie in the downstream argmax / IR comparison. That
is a near-tie, NOT an equality: the exact value has a unique correctly-rounded
representative, and the reduction error can land on the wrong side of it. Run
the collective core in float64 to shrink this to a float64-knife-edge event;
deterministic resolution AT a genuine knife-edge would require
correctly-rounded/compensated summation, which is not implemented.
"""

import jax
import jax.numpy as jnp

from _lcm.probability import scaled_down_by_power_of_two
from _lcm.zero_safe import (
    relative_scales,
    scaled_weighted_terms,
    zero_safe_weighted_term,
)
from lcm.typing import FloatND, IntND


def zero_safe_average(
    a: FloatND,
    *,
    weights: FloatND,
    shifts: IntND | None,
    axis: int | None = None,
) -> FloatND:
    """Zero-weight-safe replacement for `jnp.average(a, weights=weights, axis=axis)`.

    Mirrors `jnp.average`'s own weight-broadcasting contract (`weights` must
    either match `a`'s full shape, when `axis` is `None`, or `a.shape[axis]`,
    when `axis` is a single int) but forms the weighted sum via
    `zero_safe_weighted_term` so a zero-weight node next to an on-path
    `+-inf` value cannot inject a `nan`. Used for the stochastic-node /
    regime-mixture / fold-state weighted averages in the collective solve
    core (`Q_and_F.py`, `max_Q_over_a.py`).

    `shifts` carries the base-two scale each weight is held at — a node's
    probability is `weights * 2**-shift` — as `scaled_joint_weight` now
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

    The two reductions treat the scale differently:

    - the **mass** lowers each weight onto the common scale before summing. A
      live node too far below that scale to register contributes no share of a
      total of order one, which is the right answer for a denominator;
    - the **numerator** goes through `scaled_weighted_terms`, the engine's one
      scaled contraction, which splits the relative scale between the
      coefficient and the product. All of it on the coefficient would flush a
      tiny probability before the value supplied its binades; all of it on the
      product would overflow an ordinary coefficient meeting a value near the
      top of the range, which no later scaling recovers.

    Both read the relative scale from `relative_scales`, so numerator and mass
    are lowered onto the same one and their ratio is the mean. The common scale
    is the smallest shift — the largest node — so every relative scale is a
    power of two no greater than one and the lowering cannot overflow.

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
        lowered_weights = weights_arr
    else:
        # `relative_scales` returns the SMALLEST shift's offset, so every entry
        # is non-positive — which is exactly `scaled_down_by_power_of_two`'s
        # precondition. It agrees with `jnp.ldexp` bit-for-bit on that domain and
        # skips the general `frexp` graph, which here would run twice over the
        # whole value surface.
        lowered_weights = scaled_down_by_power_of_two(
            weights_arr, relative_scales(shifts=shifts_arr, axis=axis)
        )

    total_weight = jnp.sum(lowered_weights, axis=axis)
    _raise_if_concretely_zero(total_weight, context="zero_safe_average")
    # The masked numerator is used unconditionally -- see the module docstring's
    # PORTABILITY section. A whole-expression branch that keeps a RAW `sum(w*a)`
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
    # different reduction order and is not made bit-portable by float64.
    if shifts_arr is None:
        # Weights on one scale already: whatever size the model supplied them
        # at, and nothing upstream has accounted for a small one.
        terms = zero_safe_weighted_term(
            weight=weights_arr, value=a_arr, subnormal_is_accounted_for=False
        )
    else:
        # Scaled weights share their contraction with every other scaled
        # weighted sum in the engine, which is what splits the downscale
        # between the coefficient and the product so that neither a rare node
        # against a large value nor an ordinary node against a near-max value
        # leaves the format on the way.
        terms = scaled_weighted_terms(
            coefficients=weights_arr, shifts=shifts_arr, values=a_arr, axis=axis
        )
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
