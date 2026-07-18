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

Both helpers below apply the same fix pattern: replace the VALUE with an
explicit `0.0` wherever the weight is exactly zero, via ``jnp.where``, BEFORE
multiplying. `weight * where(weight==0, 0, value)` annihilates a zero-weight
``+-inf`` (the multiply sees ``w * 0 = 0``, never ``0 * -inf``) AND leaves the
multiply as a bare operation feeding the downstream reduction, which XLA CAN fuse
into an FMA — so the all-positive-weight path is bit-identical to the naive
``jnp.average`` / raw corner sum **on the currently pinned jaxlib**. That identity
is NOT guaranteed: it rests on XLA choosing to contract the multiply into the
reduction's FMA identically for both expressions, which JAX's compatibility
policy explicitly does not promise across releases, backends, or jit contexts
(ROUND-4 CAVEAT below). Where the reduction MUST tolerate an exact zero (the
runtime call sites), that is a safety requirement, not a bit-exactness one; the
bit-exactness is a convenient property of the current toolchain, not a contract.

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
call sites use `zero_safe_average` unconditionally and get bit-exactness from the
value-masking order ON THE PINNED TOOLCHAIN — see the round-4 caveat.

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
"""

import jax
import jax.numpy as jnp

from lcm.typing import FloatND, ScalarFloat


def zero_safe_weighted_term(
    weight: FloatND | ScalarFloat | float, value: FloatND | ScalarFloat | float
) -> FloatND:
    """``weight * value``, exactly ``0.0`` wherever ``weight == 0`` — never ``nan``.

    Standard floating-point multiplication computes ``0.0 * (+-inf) = nan``; on
    the extended reals, a zero weight should annihilate ANY value at that
    node, including an admissible on-path ``-inf``. `jnp.where` replaces the
    VALUE with `0.0` at a zero-weight node BEFORE the multiply, so the product
    is ``weight * 0 = 0`` there and never a `nan`. Because the `select` is on
    the value (an operand of the multiply) rather than on the product, the
    multiply stays FMA-contractible into a downstream reduction, so the
    all-positive-weight path is **bit-identical** to plain ``weight * value``
    (see the module docstring for why the previous product-masking order was
    not, and the decision reversals that cost).

    Args:
        weight: The weight (Pareto weight, regime-transition probability,
            interpolation corner weight, quadrature weight, ...).
        value: The value being weighted (may be `+-inf` on an admissible
            zero-weight node).

    Returns:
        The zero-safe elementwise product, broadcast like ``weight * value``.

    """
    weight_arr = jnp.asarray(weight)
    value_arr = jnp.asarray(value)
    # Mask the VALUE before the multiply, not the PRODUCT after it. Both
    # neutralize a zero-weight `+-inf` (`where` replaces the `+-inf` with 0 at a
    # zero-weight node, so the multiply sees `w * 0 = 0`, never `0 * -inf`), but
    # only this ordering keeps the multiply FMA-contractible into a downstream
    # reduction: a `select` sitting AFTER the multiply (the previous form,
    # `where(w==0, 0, w*v)`) forces the product to round once on its own before
    # the sum rounds again, so an all-positive-weight reduction drifts from the
    # naive `jnp.average` / raw corner sum -- MEASURED up to 6 ULP, enough to
    # REVERSE a non-tied action or an individual-rationality / dissolution flag
    # (see the regression tests). Masking the value leaves `w * safe_value` as a
    # bare multiply feeding the reduce, which XLA fuses, so the all-positive path
    # is bit-identical to the naive computation while the zero-weight path stays
    # `+-inf`-safe.
    safe_value = jnp.where(weight_arr == 0, jnp.zeros((), value_arr.dtype), value_arr)
    return weight_arr * safe_value


def zero_safe_average(
    a: FloatND, *, weights: FloatND, axis: int | None = None
) -> FloatND:
    """Zero-weight-safe replacement for ``jnp.average(a, weights=weights, axis=axis)``.

    Mirrors `jnp.average`'s own weight-broadcasting contract (`weights` must
    either match `a`'s full shape, when `axis` is `None`, or `a.shape[axis]`,
    when `axis` is a single int) but forms the weighted sum via
    `zero_safe_weighted_term` so a zero-weight node next to an on-path
    `+-inf` value cannot inject a `nan`. Used for the stochastic-node /
    regime-mixture / fold-state weighted averages in the collective solve
    core (`Q_and_F.py`, `max_Q_over_a.py`).

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

    total_weight = jnp.sum(weights_arr, axis=axis)
    _raise_if_concretely_zero(total_weight, context="zero_safe_average")
    numerator = jnp.sum(zero_safe_weighted_term(weights_arr, a_arr), axis=axis)
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
