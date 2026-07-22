"""Extended-real (0 * -inf -> nan) regression tests for the collective solve core.

On-path `-inf` is admissible throughout the collective-regimes extension (a
feasible zero-consumption action, a stakeholder excluded via a zero Pareto
weight, ...), and an exact-zero weight is equally admissible (a zero Pareto
weight, a zero-probability regime-transition target, a zero-weight quadrature
node). Naive floating-point arithmetic computes `0.0 * -inf = nan`, which then
poisons the household scalarization, the argmax comparison, or a weighted
average — even though the zero-weight term should contribute exactly nothing.

These tests target `_lcm.regime_building.zero_safe` (the centralized helper)
and its call sites in `_lcm.regime_building.collective` (F4) directly, plus
the collective-regime construction validation in
`_lcm.user_regime_validation` (J1). Before the fix, every test in this file
that exercises an on-path `-inf` next to a zero weight either raises (a bare
`nan` propagating into a boolean comparison silently returns `False`
everywhere, which here manifests as a WRONG argmax, not an exception) or
asserts a value that is `nan` pre-fix.
"""

import contextlib
import itertools
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building import ndimage
from _lcm.regime_building.collective import (
    _weighted_sum,
    collective_argmax_and_readout,
    collective_readout,
)
from _lcm.regime_building.ndimage import (
    _compute_indices_and_weights,
    _multiply_all,
    _sum_all,
)
from _lcm.regime_building.Q_and_F import _sum_regime_mixture
from _lcm.regime_building.zero_safe import zero_safe_average, zero_safe_weighted_term
from lcm import DiscreteGrid, LinSpacedGrid, categorical
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime
from lcm.typing import DiscreteAction, FloatND, ScalarInt

# ----------------------------------------------------------------------------------
# Helpers for the FMA / bit-exactness regression tests.
#
# The fix under test masks the VALUE before the weight-multiply
# (`w * where(w==0, 0, v)`) instead of masking the PRODUCT after it
# (`where(w==0, 0, w*v)`). Both neutralize a zero-weight `+-inf`, but only the
# value-masking form leaves the multiply FMA-contractible into the downstream
# reduction, so the all-positive-weight path is BIT-IDENTICAL to the naive
# `jnp.average` / raw corner sum. The pre-fix product-masking form drifts (up to
# 6 ULP measured), enough to reverse a non-tied action or an IR/dissolution flag.
# `_old_weighted_average*` replay that pre-fix recipe in-process so each
# bit-identity test can PROVE it would have failed against the old code without
# reverting `src/`.
# ----------------------------------------------------------------------------------


@contextlib.contextmanager
def _x64(*, enabled: bool):
    """Scope `jax_enable_x64` and restore it (x64 is OFF by default in this env)."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _bits(x: object) -> object:
    """Raw IEEE-754 bit pattern(s) as unsigned ints, for exact bit comparison."""
    arr = np.asarray(x)
    view = np.uint32 if arr.dtype == np.float32 else np.uint64
    if arr.ndim == 0:
        return int(arr.reshape(()).view(view))
    return arr.view(view)


def _old_weighted_average(a: FloatND, w: FloatND) -> FloatND:
    """The PRE-FIX recipe: mask the PRODUCT after the multiply (blocks FMA fusion)."""
    return jnp.sum(jnp.where(w == 0, jnp.zeros((), a.dtype), w * a)) / jnp.sum(w)


def _old_weighted_average_axis(a: FloatND, w: FloatND) -> FloatND:
    """Pre-fix product-masking recipe, reduced along `axis=1` (per row)."""
    w2 = jnp.reshape(w, (1, -1))
    numerator = jnp.sum(jnp.where(w2 == 0, jnp.zeros((), a.dtype), w2 * a), axis=1)
    return numerator / jnp.sum(w2, axis=1)


def _raw_corner_sum_interpolator(term_fn: object) -> object:
    """Reimplement `map_coordinates`'s 1-D corner sum with a pluggable `w*v` term.

    Uses the SAME internals as `_lcm.regime_building.ndimage.map_coordinates`
    (`_compute_indices_and_weights`, `_multiply_all`, `_sum_all`); only the
    per-corner weight*value term is swapped, so any bit difference between two
    instances is attributable solely to the term's FMA behavior.
    """

    @jax.jit
    def interpolate(array: FloatND, coordinates: FloatND) -> FloatND:
        interpolation_data = [_compute_indices_and_weights(coordinates, array.shape[0])]
        contributions = []
        for indices_and_weights in itertools.product(*interpolation_data):
            indices, weights = zip(*indices_and_weights, strict=True)
            weight_product = _multiply_all(weights)
            contributions.append(term_fn(weight_product, array[indices]))
        return _sum_all(contributions)

    return interpolate


# ----------------------------------------------------------------------------------
# `zero_safe_weighted_term` / `zero_safe_average` — the centralized primitives
# ----------------------------------------------------------------------------------


def test_zero_safe_weighted_term_annihilates_minus_inf_at_zero_weight():
    weight = jnp.array([0.0, 1.0, 0.5])
    value = jnp.array([-jnp.inf, 3.0, 4.0])
    result = zero_safe_weighted_term(weight, value)
    assert bool(jnp.all(jnp.isfinite(result)))
    np.testing.assert_allclose(np.asarray(result), [0.0, 3.0, 2.0])


def test_zero_safe_weighted_term_matches_naive_product_when_no_zero_weight():
    # No weight is exactly zero -> byte-identical to the naive product.
    weight = jnp.array([0.2, 1.0, 0.5])
    value = jnp.array([-jnp.inf, 3.0, jnp.inf])
    result = zero_safe_weighted_term(weight, value)
    naive = weight * value
    np.testing.assert_array_equal(np.asarray(result), np.asarray(naive))


def test_zero_safe_average_ignores_a_zero_weight_minus_inf_node():
    values = jnp.array([-jnp.inf, 3.0, 5.0])
    weights = jnp.array([0.0, 0.5, 0.5])
    result = zero_safe_average(values, weights=weights)
    assert bool(jnp.isfinite(result))
    np.testing.assert_allclose(float(result), 4.0)


def test_zero_safe_average_matches_jnp_average_on_the_finite_path():
    # No zero weight, no +-inf value -> BYTE-IDENTICAL to jnp.average now.
    #
    # After the value-masking fix the multiply stays FMA-contractible into the
    # reduction, so the all-positive path matches jnp.average bit-for-bit -- not
    # merely within a ULP. (The earlier retracted claim was the reverse: that the
    # product-masking guard blocked FMA and drifted ~1 ULP. That was the BUG.)
    # This particular fixture rounds identically under both the old and new forms,
    # so it is a plain regression pin; the fail-pre/pass-post proof that the OLD
    # recipe drifts lives in the counterexample tests below.
    values = jnp.array([1.0, 3.0, 5.0])
    weights = jnp.array([0.2, 0.3, 0.5])
    result = jax.jit(lambda a, w: zero_safe_average(a, weights=w))(values, weights)
    expected = jax.jit(lambda a, w: jnp.average(a, weights=w))(values, weights)
    assert _bits(result) == _bits(expected)


@pytest.mark.parametrize("use_x64", [False, True], ids=["float32", "float64"])
def test_zero_safe_average_is_bit_identical_to_jnp_average_on_the_positive_path(
    use_x64,
):
    """The all-positive path is now BIT-IDENTICAL to `jnp.average`, not within a ULP.

    This replaces `test_...within_one_ulp...`, whose `<= 2` assertion encoded the
    now-retracted drift claim. On the reviewer-supplied counterexample both weights
    are strictly positive, so the zero guard never fires. Pre-fix (product masking)
    the guarded path rounded twice and drifted from `jnp.average`; post-fix (value
    masking) the multiply stays FMA-contractible and the result matches bit-for-bit.

    Fail-pre/pass-post is PROVEN in-process (no `src/` revert): the `guarded ==
    naive` assertion itself would have failed against the pre-fix code in float32,
    and we additionally replay the OLD product-masking recipe and show IT drifts
    while the real `zero_safe_average` does not. In float64 this fixture happens to
    round identically under both forms, so there the bit-identity is a regression
    pin rather than a fail-pre proof (noted, not forced).
    """
    with _x64(enabled=use_x64):
        dtype = jnp.float64 if use_x64 else jnp.float32
        values = jnp.array([-0.3096868097782135, 0.3673213720321655], dtype=dtype)
        weights = jnp.array([0.5910956263542175, 0.40890437364578247], dtype=dtype)
        # Guard the guard: an all-positive fixture is the whole point; a zero
        # weight would make the old and new forms agree and prove nothing.
        assert bool(jnp.all(weights > 0))

        naive = jax.jit(lambda a, w: jnp.average(a, weights=w))(values, weights)
        guarded = jax.jit(lambda a, w: zero_safe_average(a, weights=w))(values, weights)
        old = jax.jit(_old_weighted_average)(values, weights)

        naive_bits = _bits(naive)
        guarded_bits = _bits(guarded)
        old_bits = _bits(old)

        # Core contract: bit-for-bit identical to jnp.average on the positive path.
        assert guarded_bits == naive_bits, (
            f"zero_safe_average is not bit-identical to jnp.average on an "
            f"all-positive input (guarded={guarded_bits}, naive={naive_bits})"
        )

        if dtype == jnp.float32:
            # Fail-pre proof: the pre-fix product-masking recipe (what
            # zero_safe_average USED to compute) drifts from jnp.average here,
            # so the `guarded == naive` assertion above would have FAILED pre-fix.
            assert old_bits != naive_bits, (
                "the OLD product-masking recipe no longer drifts from jnp.average "
                "on this fixture -- it then fails to exercise the FMA divergence "
                "and the fail-pre proof is vacuous"
            )
            assert guarded_bits != old_bits


def test_zero_safe_average_is_exact_where_ties_actually_arise():
    """A degenerate p=[1, 0] mixture must be EXACT, not merely close.

    This is the path that matters for the argmax: exact ties arise structurally
    where a weight IS zero (a degenerate mixture, an on-grid interpolation
    corner), and there the guard must reproduce the surviving value bit-for-bit
    -- otherwise a tie-break really could flip. Off-grid, where the ~1 ULP drift
    lives, exact ties do not arise.
    """
    values = jnp.array([2.5, -jnp.inf], dtype=jnp.float32)
    weights = jnp.array([1.0, 0.0], dtype=jnp.float32)

    result = jax.jit(lambda a, w: zero_safe_average(a, weights=w))(values, weights)

    assert np.asarray(result).view(np.uint32) == np.float32(2.5).view(np.uint32)


def test_zero_safe_average_axis_reduction_matches_jnp_average_on_the_finite_path():
    """The axis reduction is now BYTE-IDENTICAL to `jnp.average` on the positive path.

    Was documented "mathematically equal, not byte-identical" -- that described the
    pre-fix product-masking form. With the value-masking fix each per-row weighted
    sum stays FMA-contractible, so it matches `jnp.average` bit-for-bit. Exercised on
    a cancellation-prone float32 fixture whose first row is the F1 counterexample, so
    the FMA actually bites: the OLD product-masking recipe drifts on that row
    (fail-pre proof) while the real axis reduction is exact.
    """
    values = jnp.array(
        [[-3.9480734, 2.623802], [5.5, -1.25], [0.38403073, -7.1]],
        dtype=jnp.float32,
    )
    weights = jnp.array([0.38403073, 0.6159693], dtype=jnp.float32)
    # Guard the guard: strictly-positive weights, so the FMA divergence is live.
    assert bool(jnp.all(weights > 0))

    guarded = jax.jit(lambda a, w: zero_safe_average(a, axis=1, weights=w))(
        values, weights
    )
    naive = jax.jit(lambda a, w: jnp.average(a, axis=1, weights=w))(values, weights)
    old = jax.jit(_old_weighted_average_axis)(values, weights)

    # Byte-identical on the positive path.
    np.testing.assert_array_equal(_bits(guarded), _bits(naive))
    # Fail-pre: the pre-fix product-masking recipe drifts on at least the F1 row.
    assert int(np.sum(_bits(old) != _bits(naive))) > 0


def test_zero_safe_average_raises_eagerly_on_concretely_zero_total_weight():
    values = jnp.array([1.0, 2.0])
    weights = jnp.array([0.0, 0.0])
    with pytest.raises(ValueError, match="total weight is exactly zero"):
        zero_safe_average(values, weights=weights)


def test_zero_safe_average_does_not_reverse_a_nontied_action():
    """F1: the ULP drift must not flip a NON-TIED discrete-action choice.

    The concrete reversal the fix prevents. With nodes `[-3.9480734, 2.623802]`
    and probabilities `[0.38403073, 0.61596930]` (both strictly positive, sum
    exactly 1 in float32), the exact stochastic value is ~0.0999998 -- just BELOW
    a deterministic alternative of 0.1, so the household picks the alternative.
    `jnp.average` and the fixed `zero_safe_average` both land below 0.1 (same
    choice). The pre-fix product-masking recipe rounds up to ~0.1000000, ABOVE
    0.1, and would pick the stochastic action instead -- a reversed, non-tied
    choice. No exact tie is required; this is the decision-relevance of the drift.
    """
    nodes = jnp.array([-3.9480734, 2.623802], dtype=jnp.float32)
    probabilities = jnp.array([0.38403073, 0.6159693], dtype=jnp.float32)
    alternative = np.float32(0.1)

    # Guard the guard: all-positive probabilities summing to exactly 1.0 in
    # float32. A zero weight would make old and new forms agree and defang F1.
    assert bool(jnp.all(probabilities > 0))
    assert float(jnp.sum(probabilities)) == 1.0

    naive = jax.jit(lambda a, w: jnp.average(a, weights=w))(nodes, probabilities)
    guarded = jax.jit(lambda a, w: zero_safe_average(a, weights=w))(
        nodes, probabilities
    )
    old = jax.jit(_old_weighted_average)(nodes, probabilities)

    naive_below = bool(naive < alternative)
    guarded_below = bool(guarded < alternative)
    old_below = bool(old < alternative)

    # Fixed function picks the SAME side of the alternative as jnp.average.
    assert guarded_below == naive_below
    assert naive_below is True  # exact value is below 0.1 -> choose alternative
    # Fail-pre proof: the pre-fix recipe lands on the OPPOSITE side (>= 0.1),
    # i.e. it would reverse the action to the stochastic node.
    assert old_below is False
    assert guarded_below != old_below


def _old_left_fold_mixture(terms: list[FloatND]) -> FloatND:
    """The PRE-round-8 recipe: a Python left-fold over already-multiplied terms.

    `Q_and_F` used to accumulate `E = 0; for r: E += zero_safe_weighted_term(p_r, V_r)`.
    `_sum_regime_mixture` replaced it with a deferred vectorised zero-safe contraction
    over the UNMULTIPLIED operands (round-8). This replays the old order in-process so
    the regression can PROVE the pre-round-8 recipe lands on the wrong knife-edge side
    without reverting `src/`.
    """
    total = jnp.zeros_like(terms[0])
    for term in terms:
        total = total + term
    return total


def _deferred_mixture(w: FloatND, v: FloatND, order: tuple[int, ...]) -> FloatND:
    """Run `_sum_regime_mixture` from traced arrays: names are STATIC, arrays traced.

    `order` fixes the list order the terms are appended in; each value keeps its own
    canonical name `r{i}`, so a permuted `order` must not change the sorted result.
    """
    terms = [(f"r{i}", w[i], v[i]) for i in order]
    return _sum_regime_mixture(terms, like=v[0])


def test_sum_regime_mixture_is_zero_mass_safe():
    """The load-bearing GUARANTEE: a zero-prob target with a -inf continuation -> 0.

    An unreached regime-transition target carries probability exactly 0; its
    continuation may be an admissible on-path -inf. The mixture reduction must
    annihilate that term (contribute exactly 0), never inject a nan into E_next_V.
    """
    values = jnp.array([1.5, -jnp.inf, 2.0, 0.5], dtype=jnp.float32)
    probs = jnp.array([0.5, 0.0, 0.3, 0.2], dtype=jnp.float32)
    result = jax.jit(lambda w, v: _deferred_mixture(w, v, (0, 1, 2, 3)))(probs, values)
    assert jnp.isfinite(result)
    # exact mixture over the positive-mass terms: .5*1.5 + .3*2 + .2*.5 = 1.45
    assert float(result) == pytest.approx(1.45, abs=1e-6)


def test_sum_regime_mixture_lands_on_the_exact_side_where_the_left_fold_did_not():
    """F1 (round-8): the deferred vectorised reduction crosses to the exact-policy side.

    On the round-7 pinned 5-target float64 fixture the exact mixture is bits ...851,
    above a representable knife-edge alternative at bits ...843. `_sum_regime_mixture`
    (stack the UNMULTIPLIED operands, one zero-safe contraction) lands on the exact side
    (> alternative), while the pre-round-8 left fold lands at bits ...842, BELOW the
    alternative -- the opposite action. Proves the round-7 "no source restructuring
    fixes this" disposition was wrong. (Stacking the already-MULTIPLIED products, by
    contrast, reproduces the left fold's wrong-side bits -- the operand-vs-product
    distinction is the point.)
    """
    vals = [
        0.812941999835589,
        1.1378181379219148,
        -0.5779549019050126,
        -2.64240682258276,
        1.2829525381652913,
    ]
    probs = [
        0.12272144807325755,
        0.2780493197350539,
        0.08032169107399144,
        0.2570410094999844,
        0.2618665316177127,
    ]
    order = (0, 1, 2, 3, 4)
    alternative = np.int64(4583286125422516843).view(np.float64).item()
    with _x64(enabled=True):
        v = jnp.asarray(vals, dtype=jnp.float64)
        w = jnp.asarray(probs, dtype=jnp.float64)
        deferred = float(jax.jit(lambda w, v: _deferred_mixture(w, v, order))(w, v))
        left_fold = float(
            jax.jit(
                lambda w, v: _old_left_fold_mixture(
                    [zero_safe_weighted_term(w[i], v[i]) for i in order]
                )
            )(w, v)
        )
    assert deferred > alternative  # post-fix: exact-policy side
    assert left_fold < alternative  # fail-pre: wrong side
    assert (deferred > alternative) != (left_fold > alternative)


def test_sum_regime_mixture_is_independent_of_target_declaration_order():
    """F2 (round-8): the sorted reduction is invariant to target permutation.

    The left fold changed a pinned-fixture policy under a mere target permutation on the
    same backend. `_sum_regime_mixture` sorts by target name before stacking, so any
    permutation of the same (name, prob, value) terms yields bit-identical results.
    """
    vals = [0.81, 1.14, -0.58, -2.64, 1.28]
    probs = [0.1227, 0.2780, 0.0803, 0.2570, 0.2619]
    with _x64(enabled=True):
        v = jnp.asarray(vals, dtype=jnp.float64)
        w = jnp.asarray(probs, dtype=jnp.float64)
        base = _bits(
            jax.jit(lambda w, v: _deferred_mixture(w, v, (0, 1, 2, 3, 4)))(w, v)
        )
        for perm in [(4, 0, 3, 1, 2), (2, 1, 0, 4, 3)]:
            got = _bits(
                jax.jit(lambda w, v, perm=perm: _deferred_mixture(w, v, perm))(w, v)
            )
            assert got == base


def test_sum_regime_mixture_accuracy_scales_with_summand_magnitude_not_result_ulp():
    """F2 (round-8): the error bound is summand-scale, NOT a fixed few result-ULP.

    Under cancellation (sum|p_r*V_r| >> |sum p_r*V_r|) the reduction is hundreds of
    result-space ULP from exact, so a fixed few-ULP contract is false. The valid bound
    is absolute-plus-relative in the sum of absolute contributions.
    """
    vals = [
        -6.744894126570187,
        -9.669040336801100,
        4.023434514395978,
        0.244618606567219,
        15.911066759940047,
    ]
    probs = [
        0.17226549255821572,
        0.33944387307107820,
        0.06951303254907440,
        0.15995998262955247,
        0.25881761919207920,
    ]
    exact = float(
        sum(Fraction(p) * Fraction(v) for p, v in zip(probs, vals, strict=True))
    )
    summand_scale = float(sum(abs(p * v) for p, v in zip(probs, vals, strict=True)))
    with _x64(enabled=True):
        v = jnp.asarray(vals, dtype=jnp.float64)
        w = jnp.asarray(probs, dtype=jnp.float64)
        got = float(
            jax.jit(lambda w, v: _deferred_mixture(w, v, (0, 1, 2, 3, 4)))(w, v)
        )
    # Cancellation: the result-ULP gap is large, but the SUMMAND-scale bound holds.
    result_ulp = abs(got - exact) / np.spacing(abs(exact))
    assert result_ulp > 50  # a fixed "few ULP" contract would be false here
    assert abs(got - exact) <= 1e-15 + 1e-14 * summand_scale


def test_sum_regime_mixture_weights_the_target_axis_not_the_stakeholder_axis():
    """F3 (round-9): at the COLLECTIVE site each per-target continuation is a
    STAKEHOLDER vector, so stacking gives values (K, S) while the scalar regime
    probabilities stack to (K,). The reduction must weight the TARGET axis (axis 0)
    and hold the weight constant across the trailing stakeholder axis. Without the
    rank-align, trailing-axis broadcasting weights the stakeholder axis instead:
    K=S=2 with p=[0.25, 0.75] and values=[[0, 4], [4, 0]] returns [1, 3] (fail-pre)
    rather than the correct [3, 1] -- silently reversing the household action.
    """
    terms = [
        ("r0", jnp.asarray(0.25), jnp.asarray([0.0, 4.0])),
        ("r1", jnp.asarray(0.75), jnp.asarray([4.0, 0.0])),
    ]
    out = _sum_regime_mixture(terms, like=jnp.zeros(2))
    assert [float(x) for x in out] == pytest.approx([3.0, 1.0])


def test_sum_regime_mixture_is_zero_mass_safe_on_the_stakeholder_axis():
    """F3 (round-9): a zero-probability target with an admissible -inf stakeholder
    vector must contribute exactly 0 across ALL stakeholders, not leak -inf through
    a misaligned broadcast (fail-pre returned [-inf, 0])."""
    terms = [
        ("r0", jnp.asarray(1.0), jnp.asarray([1.0, 2.0])),
        ("r1", jnp.asarray(0.0), jnp.asarray([-jnp.inf, -jnp.inf])),
    ]
    out = _sum_regime_mixture(terms, like=jnp.zeros(2))
    assert bool(jnp.all(jnp.isfinite(out)))
    assert [float(x) for x in out] == pytest.approx([1.0, 2.0])


def test_sum_regime_mixture_collective_allows_unequal_target_and_stakeholder_counts():
    """F3 (round-9): K != S must not raise -- the misaligned trailing-axis broadcast
    crashed with a ValueError when K=3, S=2."""
    terms = [
        ("r0", jnp.asarray(0.2), jnp.asarray([1.0, 1.0])),
        ("r1", jnp.asarray(0.3), jnp.asarray([2.0, 2.0])),
        ("r2", jnp.asarray(0.5), jnp.asarray([3.0, 3.0])),
    ]
    out = _sum_regime_mixture(terms, like=jnp.zeros(2))
    assert [float(x) for x in out] == pytest.approx([2.3, 2.3])


# The round-10 counterexample (external re-review): a valid strictly-positive
# 5-target float64 mixture (probs sum to exactly 1.0) on which the round-8 name-sort
# made the float64 bits — and a NON-TIED household argmax — a function of the
# arbitrary regime LABELS. A pure alpha-renaming (same probabilities, same
# continuations, only the dict keys change) reorders the non-associative name-sorted
# sum: across the 120 name bijections the name-sort produces 37 distinct outputs,
# 20 of which choose the action OPPOSITE to exact arithmetic. `_sum_regime_mixture`
# now reduces the per-target contributions in VALUE order, provably invariant to
# alpha-renaming. See `_sum_regime_mixture`.
_ALPHA_RENAME_PROBS = [
    0.17226549255821572,
    0.33944387307107820,
    0.06951303254907440,
    0.15995998262955247,
    0.25881761919207920,
]
_ALPHA_RENAME_VALS = [
    -6.744894126570187,
    -9.669040336801100,
    4.023434514395978,
    0.244618606567219,
    15.911066759940047,
]
# A representable competing action strictly between the exact mixture and the
# name-sorted variants: some relabelings pick it, others the stochastic action.
_ALPHA_RENAME_COMPETING = -0.007134269741330662


def _old_name_sorted_mixture(names: list[str], w: FloatND, v: FloatND) -> FloatND:
    """PRE-round-10 recipe: sort `(name, p, V)` by NAME, stack, one zero-safe sum.

    Replays the label-dependent reduction in-process so the regression can PROVE the
    name-sort flips the bits (and a non-tied argmax) under a pure alpha-renaming
    without reverting `src/`. The ONLY difference from `_sum_regime_mixture` is the
    missing value-sort of the zero-safe contributions before `jnp.sum`.
    """
    order = sorted(range(len(names)), key=lambda i: names[i])
    probs = jnp.stack([w[i] for i in order], axis=0)
    values = jnp.stack([v[i] for i in order], axis=0)
    return jnp.sum(zero_safe_weighted_term(probs, values), axis=0)


def _new_value_sorted_mixture(names: list[str], w: FloatND, v: FloatND) -> FloatND:
    """Drive `_sum_regime_mixture` (the code under test) under an alpha-renaming.

    Each economic term `i` keeps its own `(prob, value)`; only its NAME (`names[i]`)
    changes across relabelings.
    """
    terms = [(names[i], w[i], v[i]) for i in range(len(names))]
    return _sum_regime_mixture(terms, like=v[0])


def _alpha_rename_mixture(reducer: object, names: list[str]) -> FloatND:
    """Broadcast the round-10 mixture over an 8x8 carrier through two nested `vmap`s
    inside `jit` — exactly the collective site's structure — and reduce it with
    `reducer` under the alpha-renaming `names`, returning one carrier cell."""

    def core(w: FloatND, v: FloatND) -> FloatND:
        return reducer(names, w, v)

    carrier = jnp.ones((8, 8))
    w = jnp.asarray(_ALPHA_RENAME_PROBS)[:, None, None] * carrier
    v = jnp.asarray(_ALPHA_RENAME_VALS)[:, None, None] * carrier
    f = jax.jit(jax.vmap(jax.vmap(core, in_axes=(1, 1)), in_axes=(2, 2)))
    return f(w, v)[0, 0]


def test_sum_regime_mixture_is_invariant_to_alpha_renaming_of_the_regimes():
    """F1 (round-10): the value-ordered reduction is BIT-invariant to a pure
    alpha-renaming of the regimes, where the round-8 name-sort was not.

    A pure alpha-renaming is economically inert (same probabilities, same
    continuations, only the dict keys change), so the household argmax must not
    depend on it. `_sum_regime_mixture` now reduces the per-target contributions in
    VALUE order (`jnp.sort` of the zero-safe `p_r*V_r` along the target axis before
    `jnp.sum`), which is a deterministic function of the contribution MULTISET and
    hence provably invariant to relabeling. This asserts bit-identity AND a single
    policy across ALL 120 name bijections, and PROVES the pre-round-10 name-sort
    (`_old_name_sorted_mixture`, replayed in-process) produced many distinct bit
    patterns AND reversed the non-tied argmax.
    """
    exact = float(
        sum(
            Fraction(p) * Fraction(v)
            for p, v in zip(_ALPHA_RENAME_PROBS, _ALPHA_RENAME_VALS, strict=True)
        )
    )
    exact_side = exact > _ALPHA_RENAME_COMPETING

    new_bits: set[object] = set()
    new_policy: set[bool] = set()
    old_bits: set[object] = set()
    old_policy: set[bool] = set()
    with _x64(enabled=True):
        for perm in itertools.permutations(range(5)):
            names = [str(p) for p in perm]
            new_val = _alpha_rename_mixture(_new_value_sorted_mixture, names)
            old_val = _alpha_rename_mixture(_old_name_sorted_mixture, names)
            new_bits.add(_bits(new_val))
            new_policy.add(bool(float(new_val) > _ALPHA_RENAME_COMPETING))
            old_bits.add(_bits(old_val))
            old_policy.add(bool(float(old_val) > _ALPHA_RENAME_COMPETING))

    # pass-post: bit-identical across ALL 120 relabelings -> ONE label-independent
    # policy, and it is the exact-arithmetic decision.
    assert len(new_bits) == 1
    assert new_policy == {exact_side}
    # fail-pre proof: the name-sort's float64 bits AND its non-tied argmax both
    # depend on the arbitrary regime labels.
    assert len(old_bits) > 1
    assert old_policy == {True, False}


def test_map_coordinates_is_bit_identical_to_the_raw_corner_sum_off_grid():
    """F2: the real interpolation path is bit-exact vs the raw `w*v` corner sum.

    `map_coordinates` weights each corner via `zero_safe_weighted_term`; off-grid,
    with both corner weights strictly positive, that must be bit-identical to the
    naive `w*v` corner sum (same internals, plain term). The pre-fix
    product-masking corner term drifts on a nonzero fraction of coordinates
    (~14-40% here, exact count platform/jax-version dependent), enough to reverse
    an IR / dissolution comparison. Asserted as `== 0` for the real path and `> 0`
    for the old path rather than a hard-coded count.
    """
    array = jnp.array([-3.9480734, 2.623802], dtype=jnp.float32)
    rng = np.random.default_rng(0)
    # ~2000 strictly-interior off-grid coordinates: both corner weights (1-c, c)
    # are strictly positive, so the zero guard never fires and the FMA is live.
    coordinates = jnp.asarray(
        rng.uniform(1e-4, 1.0 - 1e-4, size=2000), dtype=jnp.float32
    )
    # Guard the guard: strictly inside (0, 1) -> no on-grid (zero-weight) corner.
    assert bool(jnp.all((coordinates > 0.0) & (coordinates < 1.0)))

    real = ndimage.map_coordinates(array, [coordinates])
    plain_reference = _raw_corner_sum_interpolator(lambda w, v: w * v)(
        array, coordinates
    )
    old_interpolator = _raw_corner_sum_interpolator(
        lambda w, v: jnp.where(w == 0, jnp.zeros((), v.dtype), w * v)
    )(array, coordinates)

    real_bits = _bits(real)
    reference_bits = _bits(plain_reference)
    old_bits = _bits(old_interpolator)

    # Real interpolation path == naive corner sum, bit-for-bit, everywhere.
    real_diffs = int(np.sum(real_bits != reference_bits))
    assert real_diffs == 0, (
        f"map_coordinates drifted from the raw w*v corner sum on {real_diffs} "
        f"of {coordinates.size} off-grid coordinates"
    )
    # Fail-pre proof: the pre-fix product-masking corner term drifts on some.
    old_diffs = int(np.sum(old_bits != reference_bits))
    assert old_diffs > 0, (
        "the OLD product-masking corner term no longer drifts from the naive "
        "corner sum -- the fixture stopped exercising the FMA divergence"
    )


# ----------------------------------------------------------------------------------
# F4: `_weighted_sum` — the household Pareto scalarization
# ----------------------------------------------------------------------------------


def test_weighted_sum_zero_weight_minus_inf_stakeholder_stays_finite():
    # Stakeholder "f" is excluded (weight 0); its Q is -inf (an admissible
    # on-path value, e.g. a feasible zero-consumption action). The
    # scalarization must equal m's Q alone, not nan.
    stakeholder_Q = {
        "f": jnp.array([-jnp.inf, 0.0, 0.0]),
        "m": jnp.array([1.0, 5.0, 3.0]),
    }
    weights = {"f": 0.0, "m": 1.0}
    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)
    assert bool(jnp.all(jnp.isfinite(objective)))
    np.testing.assert_allclose(np.asarray(objective), [1.0, 5.0, 3.0])


def test_zero_pareto_weight_with_minus_inf_does_not_flip_the_argmax():
    """A zero-weighted stakeholder's `-inf` must not corrupt the household argmax.

    Pre-fix repro: `objective = 0.0 * Q_f + 1.0 * Q_m`. At action 0, where
    `Q_f = -inf`, `0.0 * -inf = nan`, so `objective[0] = nan`. `jnp.maximum`
    propagates `nan`, so the masked max over all three (feasible) actions
    becomes `nan` too; `a == nan` is `False` everywhere, so `argmax` of an
    all-`False` mask silently returns index 0 — the WRONG action (the true
    optimum, by `m`'s Q alone since `f` is excluded, is action 1). Read off
    at the wrong action, `f`'s value would incorrectly be `-inf` and `m`'s
    would incorrectly be `1.0` instead of `5.0`.
    """
    stakeholder_Q = {
        "f": jnp.array([-jnp.inf, 0.0, 0.0]),
        "m": jnp.array([1.0, 5.0, 3.0]),
    }
    feasibility = jnp.array([True, True, True])
    weights = {"f": 0.0, "m": 1.0}

    argmax_flat, values, dissolution = collective_argmax_and_readout(
        stakeholder_Q=stakeholder_Q,
        feasibility=feasibility,
        weights=weights,
        action_axes=(0,),
    )

    assert int(argmax_flat) == 1
    assert bool(dissolution) is False
    assert values["m"] == pytest.approx(5.0)
    assert values["f"] == pytest.approx(0.0)


def test_zero_pareto_weight_minus_inf_batched_over_states():
    # Same repro as above, but batched over two state cells with a different
    # true optimum in each, to guard against an axis-handling regression.
    q_f = jnp.array([[-jnp.inf, 0.0, 0.0], [0.0, -jnp.inf, 0.0]])
    q_m = jnp.array([[1.0, 5.0, 3.0], [7.0, 2.0, 1.0]])
    feasibility = jnp.ones((2, 3), dtype=bool)
    weights = {"f": 0.0, "m": 1.0}

    argmax_flat, values, dissolution = collective_argmax_and_readout(
        stakeholder_Q={"f": q_f, "m": q_m},
        feasibility=feasibility,
        weights=weights,
        action_axes=(1,),
    )

    np.testing.assert_array_equal(np.asarray(argmax_flat), [1, 0])
    np.testing.assert_array_equal(np.asarray(dissolution), [False, False])
    np.testing.assert_allclose(np.asarray(values["m"]), [5.0, 7.0])


# ----------------------------------------------------------------------------------
# Dissolution flag D: an on-path -inf must not be confused with the
# all-infeasible (empty-mask) marker.
# ----------------------------------------------------------------------------------


def test_onpath_minus_inf_with_a_feasible_action_leaves_dissolution_false():
    stakeholder_Q = {
        "f": jnp.array([-jnp.inf, 0.0, 0.0]),
        "m": jnp.array([1.0, 5.0, 3.0]),
    }
    feasibility = jnp.array([True, True, True])
    values, dissolution = collective_readout(
        stakeholder_Q=stakeholder_Q,
        feasibility=feasibility,
        weights={"f": 0.5, "m": 0.5},
        action_axes=(0,),
    )
    assert bool(dissolution) is False
    assert bool(jnp.isfinite(values["m"]))


def test_empty_feasible_mask_sets_dissolution_true():
    stakeholder_Q = {
        "f": jnp.array([-jnp.inf, 0.0, 0.0]),
        "m": jnp.array([1.0, 5.0, 3.0]),
    }
    feasibility = jnp.array([False, False, False])
    _values, dissolution = collective_readout(
        stakeholder_Q=stakeholder_Q,
        feasibility=feasibility,
        weights={"f": 0.5, "m": 0.5},
        action_axes=(0,),
    )
    assert bool(dissolution) is True


# ----------------------------------------------------------------------------------
# J1 (minor): collective weight / stakeholder validation at `Regime` construction.
# ----------------------------------------------------------------------------------

_WEALTH = LinSpacedGrid(start=1, stop=10, n_points=5)


@categorical(ordered=True)
class LaborSupply:
    do_not_work: ScalarInt
    work: ScalarInt


def _utility_f(labor_supply_f: DiscreteAction) -> FloatND:
    return -0.3 * (labor_supply_f == LaborSupply.work)


def _utility_m(labor_supply_f: DiscreteAction) -> FloatND:
    return -0.5 * (labor_supply_f == LaborSupply.work)


def _build_terminal_regime(**kwargs: object) -> Regime:
    base = {
        "transition": None,
        "stakeholders": ("f", "m"),
        "states": {"wealth": _WEALTH},
        "actions": {"labor_supply_f": DiscreteGrid(LaborSupply)},
        "functions": {"utility_f": _utility_f, "utility_m": _utility_m},
    }
    base.update(kwargs)
    return Regime(**base)  # type: ignore[arg-type]


def test_empty_stakeholders_tuple_is_rejected():
    with pytest.raises(RegimeInitializationError, match="non-empty"):
        _build_terminal_regime(stakeholders=())


def test_duplicate_stakeholders_are_rejected():
    with pytest.raises(RegimeInitializationError, match="duplicate"):
        Regime(
            transition=None,
            stakeholders=("f", "f"),
            states={"wealth": _WEALTH},
            actions={"labor_supply_f": DiscreteGrid(LaborSupply)},
            functions={"utility_f": _utility_f},
        )


def test_non_finite_weight_is_rejected():
    with pytest.raises(RegimeInitializationError, match="finite"):
        _build_terminal_regime(weights={"f": float("nan"), "m": 0.5})


def test_negative_weight_is_rejected():
    with pytest.raises(RegimeInitializationError, match="non-negative"):
        _build_terminal_regime(weights={"f": -0.1, "m": 1.1})


def test_all_zero_weights_are_rejected():
    with pytest.raises(RegimeInitializationError, match="positive total"):
        _build_terminal_regime(weights={"f": 0.0, "m": 0.0})


def test_a_single_zero_weight_with_a_positive_total_is_allowed():
    # A zero weight is a deliberate exclusion, not an error, as long as the
    # total remains positive.
    regime = _build_terminal_regime(weights={"f": 0.0, "m": 1.0})
    assert regime.weights == {"f": 0.0, "m": 1.0}
