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

Both helpers below apply the same fix pattern: replace the elementwise
product with an explicit `0.0` wherever the weight is exactly zero, via
``jnp.where``, BEFORE summing.

On the all-positive-weight path the result is MATHEMATICALLY equal to the naive
computation, but it is **not byte-identical** when the weights are traced: the
inserted `select` blocks XLA from contracting `multiply` into the reduction's
FMA, so the naive path rounds once where this one rounds twice, and the two can
differ by ~1 ULP. MEASURED (jax 0.10.1, CPU, float32): ``jnp.average`` returns
bits ``3171324718`` where `zero_safe_average` returns ``3171324720``; across
20k random fractional interpolation coordinates 24.4% of cells differ, by at
most 4.77e-07 (float32 eps scale). Statically-known weights are constant-folded
and DO match bit-for-bit. Note the direction: the naive path is the marginally
more accurate one (one fewer rounding) — the guard buys extended-real
correctness at the price of a free FMA.

This is deliberate and the drift is not decision-relevant, because the two
properties live on disjoint paths: ties arise STRUCTURALLY only where a weight
is exactly zero (an on-grid interpolation corner, a degenerate ``p = [1, 0]``
mixture), and on exactly those cells both kernels are exact and agree
bit-for-bit — while the drift occurs only off-grid/all-positive, where exact
ties do not arise (measured: 0 exact ties in 200k random off-grid pairs). A
`jax.lax.cond` on ``any(weights == 0)`` would restore the naive bits on the
all-positive arm at no measured cost, but its predicate is GLOBAL: a single
on-grid cell anywhere in a batch would flip the whole batch to the safe arm, so
it would deliver bit-compatibility precisely in the batches that do not need it
and withhold it from the ones that motivated the guard. The per-element
`jnp.where` is the honest primitive.

That argument is about weights that are only known at RUNTIME (an
interpolation corner, a mixture probability) — the case every call site here
faces but one. The FOLD reduction is the exception: a fold axis's weights are
the process's own quadrature marginal, and `_validate_fold_declarations`
rejects a runtime-parameterized folded process, so they are concrete before
the kernel is ever traced. `max_Q_over_a._select_fold_reducer` therefore
picks between `jnp.average` and `zero_safe_average` per axis with a plain
Python ``if``, at kernel-build time — no traced predicate, no `lax.cond`, and
no global/per-axis conflict. Callers whose weights ARE runtime values cannot
do this and should keep using `zero_safe_average` unconditionally.
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
    node, including an admissible on-path ``-inf``. `jnp.where` selects the
    literal `0.0` there instead of the (already-computed, possibly `nan`)
    product, so the returned array never carries a `nan` a zero-weight term
    would otherwise have contributed. Mathematically equal to plain
    ``weight * value`` whenever `weight` is nowhere exactly zero — but not
    byte-identical for traced weights, since the `select` blocks XLA's FMA
    contraction (~1 ULP; see the module docstring).

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
    product = weight_arr * value_arr
    zero = jnp.zeros((), dtype=product.dtype)
    return jnp.where(weight_arr == 0, zero, product)


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
