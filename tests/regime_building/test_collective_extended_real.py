"""Extended-real (0 * -inf -> nan) regression tests for the collective solve core.

On-path `-inf` is admissible throughout the collective-regimes extension (a
feasible zero-consumption action, a stakeholder excluded via a zero Pareto
weight, ...), and an exact-zero weight is equally admissible (a zero Pareto
weight, a zero-probability regime-transition target, a zero-weight quadrature
node). Naive floating-point arithmetic computes `0.0 * -inf = nan`, which then
poisons the household scalarization, the argmax comparison, or a weighted
average — even though the zero-weight term should contribute exactly nothing.

These tests target `_lcm.regime_building.zero_safe` (the centralized helper)
and its call sites in `_lcm.regime_building.collective` directly, plus
the collective-regime construction validation in
`_lcm.user_regime_validation`. Without the zero-safe arithmetic, every test in
this file that exercises an on-path `-inf` next to a zero weight either raises
(a bare `nan` propagating into a boolean comparison silently returns `False`
everywhere, which here manifests as a WRONG argmax, not an exception) or
asserts a value that comes out `nan`.
"""

import contextlib
import inspect
import itertools
from collections.abc import Callable
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
from _lcm.regime_building.zero_safe import zero_safe_average
from _lcm.zero_safe import zero_safe_weighted_term
from lcm import (
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    ParetoObjective,
    categorical,
)
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime
from lcm.typing import DiscreteAction, FloatND, ScalarInt

# Helpers for the FMA / bit-exactness regression tests.
#
# The production arithmetic masks the VALUE before the weight-multiply
# (`w * where(w==0, 0, v)`) rather than the PRODUCT after it
# (`where(w==0, 0, w*v)`). Both neutralize a zero-weight `+-inf`, but only the
# value-masking form leaves the multiply FMA-contractible into the downstream
# reduction, so the all-positive-weight path is BIT-IDENTICAL to the naive
# `jnp.average` / raw corner sum. The product-masking form drifts (up to 6 ULP
# measured), enough to reverse a non-tied action or an IR/dissolution flag.
# `_product_masked_average*` run that recipe in-process so each bit-identity
# test can PROVE the difference is live on its fixture, without touching `src/`.


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


def _product_masked_average(*, a: FloatND, w: FloatND) -> FloatND:
    """Mask the PRODUCT after the multiply, which blocks FMA fusion."""
    return jnp.sum(jnp.where(w == 0, jnp.zeros((), a.dtype), w * a)) / jnp.sum(w)


def _product_masked_average_axis(*, a: FloatND, w: FloatND) -> FloatND:
    """Product-masking recipe, reduced along `axis=1` (per row)."""
    w2 = jnp.reshape(w, (1, -1))
    numerator = jnp.sum(jnp.where(w2 == 0, jnp.zeros((), a.dtype), w2 * a), axis=1)
    return numerator / jnp.sum(w2, axis=1)


def _raw_corner_sum_interpolator(
    term_fn: Callable[[FloatND, FloatND], FloatND],
) -> Callable[..., FloatND]:
    """Reimplement `map_coordinates`'s 1-D corner sum with a pluggable `w*v` term.

    Uses the SAME internals as `_lcm.regime_building.ndimage.map_coordinates`
    (`_compute_indices_and_weights`, `_multiply_all`, `_sum_all`); only the
    per-corner weight*value term is swapped, so any bit difference between two
    instances is attributable solely to the term's FMA behavior.
    """

    @jax.jit
    def interpolate(*, array: FloatND, coordinates: FloatND) -> FloatND:
        interpolation_data = [
            _compute_indices_and_weights(
                coordinate=coordinates, input_size=array.shape[0]
            )
        ]
        contributions = []
        for indices_and_weights in itertools.product(*interpolation_data):
            indices, weights = zip(*indices_and_weights, strict=True)
            weight_product = _multiply_all(weights)
            contributions.append(term_fn(weight_product, array[indices]))
        return _sum_all(contributions)

    return interpolate


# `zero_safe_weighted_term` / `zero_safe_average` — the centralized primitives


def test_zero_safe_weighted_term_annihilates_minus_inf_at_zero_weight():
    weight = jnp.array([0.0, 1.0, 0.5])
    value = jnp.array([-jnp.inf, 3.0, 4.0])
    result = zero_safe_weighted_term(
        weight=weight, value=value, subnormal_is_accounted_for=False
    )
    assert bool(jnp.all(jnp.isfinite(result)))
    np.testing.assert_allclose(np.asarray(result), [0.0, 3.0, 2.0])


def test_zero_safe_weighted_term_matches_naive_product_when_no_zero_weight():
    # No weight is exactly zero -> byte-identical to the naive product.
    weight = jnp.array([0.2, 1.0, 0.5])
    value = jnp.array([-jnp.inf, 3.0, jnp.inf])
    result = zero_safe_weighted_term(
        weight=weight, value=value, subnormal_is_accounted_for=False
    )
    naive = weight * value
    np.testing.assert_array_equal(np.asarray(result), np.asarray(naive))


def test_zero_safe_average_ignores_a_zero_weight_minus_inf_node():
    values = jnp.array([-jnp.inf, 3.0, 5.0])
    weights = jnp.array([0.0, 0.5, 0.5])
    result = zero_safe_average(a=values, weights=weights, shifts=None)
    assert bool(jnp.isfinite(result))
    np.testing.assert_allclose(float(result), 4.0)


def test_a_coefficient_cannot_be_supplied_to_the_average_without_its_scale():
    """`shifts` is required, so no call can leave the scale question unasked.

    A weight that came from `scaled_joint_weight` means nothing without the
    scale it is held at, so the signature offers no way to omit it: a caller
    states the scales, or states `None` to say its weights are already on one.
    """
    parameter = inspect.signature(zero_safe_average).parameters["shifts"]

    assert parameter.default is inspect.Parameter.empty


def test_zero_safe_average_weighs_nodes_by_probability_not_by_coefficient():
    """Nodes on different scales are weighed by `coefficient * 2**-shift`.

    `scaled_joint_weight` returns one scale per NODE, so two coefficients say
    nothing about their relative probability until both are on one scale. Here
    the second node is held one binade down: equal coefficients, but half the
    probability. Averaging the coefficients alone would return `4.0` — the
    near-impossible node valued as an even chance — where the lottery the
    weights actually describe averages to `10/3`.
    """
    values = jnp.array([2.0, 6.0])
    weights = jnp.array([1.0, 1.0])
    shifts = jnp.array([0, 1], dtype=jnp.int32)

    result = zero_safe_average(a=values, weights=weights, shifts=shifts)

    # p = [1.0, 0.5] -> (1.0*2 + 0.5*6) / 1.5
    np.testing.assert_allclose(float(result), 10.0 / 3.0)
    assert float(result) != 4.0


def test_zero_safe_average_matches_jnp_average_on_the_finite_path():
    # No zero weight, no +-inf value -> BYTE-IDENTICAL to jnp.average.
    #
    # Value masking leaves the multiply FMA-contractible into the reduction, so
    # the all-positive path matches jnp.average bit-for-bit -- not merely within
    # a ULP. This particular fixture rounds identically under value masking and
    # product masking alike, so it is a plain regression pin; the counterexample
    # tests below carry the proof that the product-masking recipe drifts.
    values = jnp.array([1.0, 3.0, 5.0])
    weights = jnp.array([0.2, 0.3, 0.5])
    result = jax.jit(lambda a, w: zero_safe_average(a=a, weights=w, shifts=None))(
        values, weights
    )
    expected = jax.jit(lambda a, w: jnp.average(a, weights=w))(values, weights)
    assert _bits(result) == _bits(expected)


@pytest.mark.parametrize("use_x64", [False, True], ids=["float32", "float64"])
def test_zero_safe_average_is_bit_identical_to_jnp_average_on_the_positive_path(
    use_x64,
):
    """The all-positive path is BIT-IDENTICAL to `jnp.average`, not within a ULP.

    On this counterexample both weights are strictly positive, so the zero guard
    never fires. Product masking would round twice and drift from `jnp.average`;
    value masking keeps the multiply FMA-contractible and the result matches
    bit-for-bit.

    The difference is shown live in-process: the product-masking recipe is run
    alongside and shown to drift where the real `zero_safe_average` does not. In
    float64 this fixture happens to round identically under both forms, so there
    the bit-identity is a regression pin rather than a discriminating check.
    """
    with _x64(enabled=use_x64):
        dtype = jnp.float64 if use_x64 else jnp.float32
        values = jnp.array([-0.3096868097782135, 0.3673213720321655], dtype=dtype)
        weights = jnp.array([0.5910956263542175, 0.40890437364578247], dtype=dtype)
        # Guard the guard: an all-positive fixture is the whole point; a zero
        # weight would make the two forms agree and prove nothing.
        assert bool(jnp.all(weights > 0))

        naive = jax.jit(lambda a, w: jnp.average(a, weights=w))(values, weights)
        guarded = jax.jit(lambda a, w: zero_safe_average(a=a, weights=w, shifts=None))(
            values, weights
        )
        old = jax.jit(_product_masked_average)(a=values, w=weights)

        naive_bits = _bits(naive)
        guarded_bits = _bits(guarded)
        old_bits = _bits(old)

        # Core contract: bit-for-bit identical to jnp.average on the positive path.
        assert guarded_bits == naive_bits, (
            f"zero_safe_average is not bit-identical to jnp.average on an "
            f"all-positive input (guarded={guarded_bits}, naive={naive_bits})"
        )

        if dtype == jnp.float32:
            # Guard the guard: the product-masking recipe drifts from
            # jnp.average here, so the `guarded == naive` assertion above is
            # discriminating rather than vacuous.
            assert old_bits != naive_bits, (
                "the product-masking recipe no longer drifts from jnp.average "
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

    result = jax.jit(lambda a, w: zero_safe_average(a=a, weights=w, shifts=None))(
        values, weights
    )

    assert np.asarray(result).view(np.uint32) == np.float32(2.5).view(np.uint32)


def test_zero_safe_average_axis_reduction_matches_jnp_average_on_the_finite_path():
    """The axis reduction is now BYTE-IDENTICAL to `jnp.average` on the positive path.

    "Mathematically equal, not byte-identical" describes the product-masking
    form. Under value masking each per-row weighted sum stays FMA-contractible,
    so it matches `jnp.average` bit-for-bit. Exercised on a cancellation-prone
    float32 fixture whose first row is the drift counterexample, so the FMA
    actually bites: the product-masking recipe drifts on that row while the real
    axis reduction is exact.
    """
    values = jnp.array(
        [[-3.9480734, 2.623802], [5.5, -1.25], [0.38403073, -7.1]],
        dtype=jnp.float32,
    )
    weights = jnp.array([0.38403073, 0.6159693], dtype=jnp.float32)
    # Guard the guard: strictly-positive weights, so the FMA divergence is live.
    assert bool(jnp.all(weights > 0))

    guarded = jax.jit(
        lambda a, w: zero_safe_average(a=a, axis=1, weights=w, shifts=None)
    )(values, weights)
    naive = jax.jit(lambda a, w: jnp.average(a, axis=1, weights=w))(values, weights)
    old = jax.jit(_product_masked_average_axis)(a=values, w=weights)

    # Byte-identical on the positive path.
    np.testing.assert_array_equal(_bits(guarded), _bits(naive))
    # The product-masking recipe drifts on at least the counterexample row.
    assert int(np.sum(_bits(old) != _bits(naive))) > 0


def test_zero_safe_average_raises_eagerly_on_concretely_zero_total_weight():
    values = jnp.array([1.0, 2.0])
    weights = jnp.array([0.0, 0.0])
    with pytest.raises(ValueError, match="total weight is exactly zero"):
        zero_safe_average(a=values, weights=weights, shifts=None)


def test_zero_safe_average_does_not_reverse_a_nontied_action():
    """The ULP drift must not flip a NON-TIED discrete-action choice.

    The concrete reversal at stake. With nodes `[-3.9480734, 2.623802]`
    and probabilities `[0.38403073, 0.61596930]` (both strictly positive, sum
    exactly 1 in float32), the exact stochastic value is ~0.0999998 -- just BELOW
    a deterministic alternative of 0.1, so the household picks the alternative.
    `jnp.average` and `zero_safe_average` both land below 0.1 (same choice).
    The product-masking recipe rounds up to ~0.1000000, ABOVE
    0.1, and would pick the stochastic action instead -- a reversed, non-tied
    choice. No exact tie is required; this is the decision-relevance of the drift.
    """
    nodes = jnp.array([-3.9480734, 2.623802], dtype=jnp.float32)
    probabilities = jnp.array([0.38403073, 0.6159693], dtype=jnp.float32)
    alternative = np.float32(0.1)

    # Guard the guard: all-positive probabilities summing to exactly 1.0 in
    # float32. A zero weight would make the two forms agree and defang the test.
    assert bool(jnp.all(probabilities > 0))
    assert float(jnp.sum(probabilities)) == 1.0

    naive = jax.jit(lambda a, w: jnp.average(a, weights=w))(nodes, probabilities)
    guarded = jax.jit(lambda a, w: zero_safe_average(a=a, weights=w, shifts=None))(
        nodes, probabilities
    )
    old = jax.jit(_product_masked_average)(a=nodes, w=probabilities)

    naive_below = bool(naive < alternative)
    guarded_below = bool(guarded < alternative)
    old_below = bool(old < alternative)

    # Fixed function picks the SAME side of the alternative as jnp.average.
    assert guarded_below == naive_below
    assert naive_below is True  # exact value is below 0.1 -> choose alternative
    # The product-masking recipe lands on the OPPOSITE side (>= 0.1), i.e. it
    # would reverse the action to the stochastic node.
    assert old_below is False
    assert guarded_below != old_below


def _product_left_fold_mixture(terms: list[FloatND]) -> FloatND:
    """A Python left-fold over already-multiplied terms.

    The declaration-ordered accumulation against which `_sum_regime_mixture`'s
    value-ordered reduction is compared. Run in-process so the regression can
    show why canonical contribution ordering is load-bearing without touching
    `src/`.
    """
    total = jnp.zeros_like(terms[0])
    for term in terms:
        total = total + term
    return total


def _value_ordered_mixture(
    *, w: FloatND, v: FloatND, order: tuple[int, ...]
) -> FloatND:
    """Run `_sum_regime_mixture` from traced arrays: names are static, arrays traced.

    `order` fixes the list order the terms are appended in; each value keeps its own
    canonical name `r{i}`, so a permuted `order` must not change the sorted result.
    """
    terms = [(f"r{i}", w[i], v[i]) for i in order]
    return _sum_regime_mixture(mixture_terms=terms, like=v[0])


def test_sum_regime_mixture_is_zero_mass_safe():
    """The load-bearing GUARANTEE: a zero-prob target with a -inf continuation -> 0.

    An unreached regime-transition target carries probability exactly 0; its
    continuation may be an admissible on-path -inf. The mixture reduction must
    annihilate that term (contribute exactly 0), never inject a nan into E_next_V.
    """
    values = jnp.array([1.5, -jnp.inf, 2.0, 0.5], dtype=jnp.float32)
    probs = jnp.array([0.5, 0.0, 0.3, 0.2], dtype=jnp.float32)
    result = jax.jit(lambda w, v: _value_ordered_mixture(w=w, v=v, order=(0, 1, 2, 3)))(
        probs, values
    )
    assert jnp.isfinite(result)
    # exact mixture over the positive-mass terms: .5*1.5 + .3*2 + .2*.5 = 1.45
    assert float(result) == pytest.approx(1.45, abs=1e-6)


def test_value_ordered_mixture_lands_on_the_exact_side_where_left_fold_did_not():
    """Canonical contribution ordering crosses to the exact-policy side.

    On a pinned 5-target float64 fixture the exact mixture is above a representable
    competing action. `_sum_regime_mixture` orders the separately formed zero-safe
    contributions by value and lands above the alternative, while declaration-order
    accumulation lands below it. The witness pins the supported reduction order; it
    does not require a target-axis stack or one vectorized multiplication.
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
        value_ordered = float(
            jax.jit(lambda w, v: _value_ordered_mixture(w=w, v=v, order=order))(w, v)
        )
        left_fold = float(
            jax.jit(
                lambda w, v: _product_left_fold_mixture(
                    [
                        zero_safe_weighted_term(
                            weight=w[i], value=v[i], subnormal_is_accounted_for=False
                        )
                        for i in order
                    ]
                )
            )(w, v)
        )
    assert value_ordered > alternative  # exact-policy side
    assert left_fold < alternative  # wrong side
    assert (value_ordered > alternative) != (left_fold > alternative)


def test_sum_regime_mixture_is_independent_of_target_declaration_order():
    """The sorted reduction is invariant to target permutation.

    A left fold changes a pinned-fixture policy under a mere target permutation on
    the same backend. `_sum_regime_mixture` orders by contribution value, so any
    permutation of the same `(name, probability, value)` terms is bit-identical.
    """
    vals = [0.81, 1.14, -0.58, -2.64, 1.28]
    probs = [0.1227, 0.2780, 0.0803, 0.2570, 0.2619]
    with _x64(enabled=True):
        v = jnp.asarray(vals, dtype=jnp.float64)
        w = jnp.asarray(probs, dtype=jnp.float64)
        base = _bits(
            jax.jit(
                lambda w, v: _value_ordered_mixture(w=w, v=v, order=(0, 1, 2, 3, 4))
            )(w, v)
        )
        for perm in [(4, 0, 3, 1, 2), (2, 1, 0, 4, 3)]:
            got = _bits(
                jax.jit(
                    lambda w, v, perm=perm: _value_ordered_mixture(w=w, v=v, order=perm)
                )(w, v)
            )
            assert got == base


def test_sum_regime_mixture_accuracy_scales_with_summand_magnitude_not_result_ulp():
    """The error bound is summand-scale, NOT a fixed few result-ULP.

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
            jax.jit(
                lambda w, v: _value_ordered_mixture(w=w, v=v, order=(0, 1, 2, 3, 4))
            )(w, v)
        )
    # Cancellation: the result-ULP gap is large, but the SUMMAND-scale bound holds.
    result_ulp = abs(got - exact) / np.spacing(abs(exact))
    assert result_ulp > 50  # a fixed "few ULP" contract would be false here
    assert abs(got - exact) <= 1e-15 + 1e-14 * summand_scale


def test_sum_regime_mixture_weights_the_target_axis_not_the_stakeholder_axis():
    """At the COLLECTIVE site each per-target continuation is a STAKEHOLDER vector,
    so stacking gives values (K, S) while the scalar regime probabilities stack to
    (K,). The reduction must weight the TARGET axis (axis 0)
    and hold the weight constant across the trailing stakeholder axis. Without the
    rank-align, trailing-axis broadcasting weights the stakeholder axis instead:
    K=S=2 with p=[0.25, 0.75] and values=[[0, 4], [4, 0]] returns [1, 3] rather
    than the correct [3, 1] -- silently reversing the household action.
    """
    terms = [
        ("r0", jnp.asarray(0.25), jnp.asarray([0.0, 4.0])),
        ("r1", jnp.asarray(0.75), jnp.asarray([4.0, 0.0])),
    ]
    out = _sum_regime_mixture(mixture_terms=terms, like=jnp.zeros(2))
    assert [float(x) for x in out] == pytest.approx([3.0, 1.0])


def test_sum_regime_mixture_is_zero_mass_safe_on_the_stakeholder_axis():
    """A zero-probability target with an admissible -inf stakeholder vector
    contributes exactly 0 across ALL stakeholders, and does not leak -inf through
    a misaligned broadcast (which would return [-inf, 0])."""
    terms = [
        ("r0", jnp.asarray(1.0), jnp.asarray([1.0, 2.0])),
        ("r1", jnp.asarray(0.0), jnp.asarray([-jnp.inf, -jnp.inf])),
    ]
    out = _sum_regime_mixture(mixture_terms=terms, like=jnp.zeros(2))
    assert bool(jnp.all(jnp.isfinite(out)))
    assert [float(x) for x in out] == pytest.approx([1.0, 2.0])


def test_sum_regime_mixture_collective_allows_unequal_target_and_stakeholder_counts():
    """K != S must not raise -- a misaligned trailing-axis broadcast crashes with
    a ValueError when K=3, S=2."""
    terms = [
        ("r0", jnp.asarray(0.2), jnp.asarray([1.0, 1.0])),
        ("r1", jnp.asarray(0.3), jnp.asarray([2.0, 2.0])),
        ("r2", jnp.asarray(0.5), jnp.asarray([3.0, 3.0])),
    ]
    out = _sum_regime_mixture(mixture_terms=terms, like=jnp.zeros(2))
    assert [float(x) for x in out] == pytest.approx([2.3, 2.3])


# The alpha-renaming counterexample: a valid strictly-positive 5-target float64
# mixture (probs sum to exactly 1.0) on which a name-sort makes the float64 bits
# — and a NON-TIED household argmax — a function of the arbitrary regime LABELS. A
# pure alpha-renaming (same probabilities, same continuations, only the dict keys
# change) reorders the non-associative name-sorted
# sum: across the 120 name bijections the name-sort produces 37 distinct outputs,
# 20 of which choose the action OPPOSITE to exact arithmetic. `_sum_regime_mixture`
# reduces the per-target contributions in VALUE order, provably invariant to
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


def _name_sorted_mixture(*, names: list[str], w: FloatND, v: FloatND) -> FloatND:
    """Sort `(name, p, V)` by NAME, stack, one zero-safe sum.

    The label-dependent reduction, run in-process so the regression can show the
    name-sort flips the bits (and a non-tied argmax) under a pure alpha-renaming,
    without touching `src/`. The ONLY difference from `_sum_regime_mixture` is the
    missing value-sort of the zero-safe contributions before `jnp.sum`.
    """
    order = sorted(range(len(names)), key=lambda i: names[i])
    probs = jnp.stack([w[i] for i in order], axis=0)
    values = jnp.stack([v[i] for i in order], axis=0)
    return jnp.sum(
        zero_safe_weighted_term(
            weight=probs, value=values, subnormal_is_accounted_for=False
        ),
        axis=0,
    )


def _value_sorted_mixture(*, names: list[str], w: FloatND, v: FloatND) -> FloatND:
    """Drive `_sum_regime_mixture` (the code under test) under an alpha-renaming.

    Each economic term `i` keeps its own `(prob, value)`; only its NAME (`names[i]`)
    changes across relabelings.
    """
    terms = [(names[i], w[i], v[i]) for i in range(len(names))]
    return _sum_regime_mixture(mixture_terms=terms, like=v[0])


def _alpha_rename_mixture(
    *, reducer: Callable[..., FloatND], names: list[str]
) -> FloatND:
    """Broadcast the alpha-renaming mixture over an 8x8 carrier through two nested
    `vmap`s inside `jit` — exactly the collective site's structure — and reduce it with
    `reducer` under the alpha-renaming `names`, returning one carrier cell."""

    # keyword-only-exempt: library-callback=jax.vmap
    def core(w: FloatND, v: FloatND) -> FloatND:
        return reducer(names=names, w=w, v=v)

    carrier = jnp.ones((8, 8))
    w = jnp.asarray(_ALPHA_RENAME_PROBS)[:, None, None] * carrier
    v = jnp.asarray(_ALPHA_RENAME_VALS)[:, None, None] * carrier
    f = jax.jit(jax.vmap(jax.vmap(core, in_axes=(1, 1)), in_axes=(2, 2)))
    return f(w, v)[0, 0]


def test_sum_regime_mixture_is_invariant_to_alpha_renaming_of_the_regimes():
    """The value-ordered reduction is BIT-invariant to a pure alpha-renaming of
    the regimes, where a name-sort is not.

    A pure alpha-renaming is economically inert (same probabilities, same
    continuations, only the dict keys change), so the household argmax must not
    depend on it. `_sum_regime_mixture` reduces the separate zero-safe per-target
    contributions in VALUE order, which is a deterministic function of the
    contribution MULTISET and
    hence provably invariant to relabeling. This asserts bit-identity AND a single
    policy across ALL 120 name bijections, and shows the name-sort
    (`_name_sorted_mixture`, run in-process) produces many distinct bit patterns
    AND reverses the non-tied argmax.
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
            new_val = _alpha_rename_mixture(reducer=_value_sorted_mixture, names=names)
            old_val = _alpha_rename_mixture(reducer=_name_sorted_mixture, names=names)
            new_bits.add(_bits(new_val))
            new_policy.add(bool(float(new_val) > _ALPHA_RENAME_COMPETING))
            old_bits.add(_bits(old_val))
            old_policy.add(bool(float(old_val) > _ALPHA_RENAME_COMPETING))

    # The alpha-renaming reducer: bit-identical across ALL 120 relabelings -> ONE
    # label-independent policy, and it is the exact-arithmetic decision.
    assert len(new_bits) == 1
    assert new_policy == {exact_side}
    # The name-sort reducer, for contrast: its float64 bits AND its non-tied
    # argmax both depend on the arbitrary regime labels.
    assert len(old_bits) > 1
    assert old_policy == {True, False}


def test_map_coordinates_is_bit_identical_to_the_raw_corner_sum_off_grid():
    """The real interpolation path is bit-exact vs the raw `w*v` corner sum.

    `map_coordinates` weights each corner via `zero_safe_weighted_term`; off-grid,
    with both corner weights strictly positive, that must be bit-identical to the
    naive `w*v` corner sum (same internals, plain term). A product-masking corner
    term drifts on a nonzero fraction of coordinates (~14-40% here, exact count
    platform/jax-version dependent), enough to reverse an IR / dissolution
    comparison. Asserted as `== 0` for the real path and `> 0` for the
    product-masking path rather than a hard-coded count.
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

    real = ndimage.map_coordinates(input=array, coordinates=[coordinates])
    plain_reference = _raw_corner_sum_interpolator(lambda w, v: w * v)(
        array=array, coordinates=coordinates
    )
    old_interpolator = _raw_corner_sum_interpolator(
        lambda w, v: jnp.where(w == 0, jnp.zeros((), v.dtype), w * v)
    )(array=array, coordinates=coordinates)

    real_bits = _bits(real)
    reference_bits = _bits(plain_reference)
    old_bits = _bits(old_interpolator)

    # Real interpolation path == naive corner sum, bit-for-bit, everywhere.
    real_diffs = int(np.sum(real_bits != reference_bits))
    assert real_diffs == 0, (
        f"map_coordinates drifted from the raw w*v corner sum on {real_diffs} "
        f"of {coordinates.size} off-grid coordinates"
    )
    # Guard the guard: the product-masking corner term drifts on some.
    old_diffs = int(np.sum(old_bits != reference_bits))
    assert old_diffs > 0, (
        "the product-masking corner term no longer drifts from the naive "
        "corner sum -- the fixture stopped exercising the FMA divergence"
    )


# `_weighted_sum` — the household Pareto scalarization


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

    Without the guard, `objective = 0.0 * Q_f + 1.0 * Q_m`. At action 0, where
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


# Dissolution flag D: an on-path -inf must not be confused with the
# all-infeasible (empty-mask) marker.


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


# Collective weight / stakeholder validation at `Regime` construction.

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
    """Build the two-stakeholder terminal regime, overriding one slot by keyword.

    `utilities` and `objective` reach the `CollectiveUtility` the regime
    declares; every other keyword is a `Regime` slot.
    """
    utilities = kwargs.pop("utilities", {"f": _utility_f, "m": _utility_m})
    objective = kwargs.pop("objective", None)
    base = {
        "transition": None,
        "states": {"wealth": _WEALTH},
        "actions": {"labor_supply_f": DiscreteGrid(category_class=LaborSupply)},
        "functions": {
            "utility": CollectiveUtility(
                utilities=utilities,  # ty: ignore[invalid-argument-type]
                objective=objective,  # ty: ignore[invalid-argument-type]
            )
        },
    }
    base.update(kwargs)
    return Regime(**base)  # ty: ignore[invalid-argument-type]


def test_a_household_with_no_stakeholders_is_rejected():
    """A `CollectiveUtility` naming nobody is not a household."""
    with pytest.raises(RegimeInitializationError, match="at least one stakeholder"):
        _build_terminal_regime(utilities={})


def test_non_finite_weight_is_rejected():
    with pytest.raises(RegimeInitializationError, match="finite"):
        _build_terminal_regime(
            objective=ParetoObjective(weights={"f": float("nan"), "m": 0.5})
        )


def test_negative_weight_is_rejected():
    with pytest.raises(RegimeInitializationError, match="non-negative"):
        _build_terminal_regime(objective=ParetoObjective(weights={"f": -0.1, "m": 1.1}))


def test_all_zero_weights_are_rejected():
    with pytest.raises(RegimeInitializationError, match="positive total"):
        _build_terminal_regime(objective=ParetoObjective(weights={"f": 0.0, "m": 0.0}))


def test_a_single_zero_weight_with_a_positive_total_is_allowed():
    # A zero weight is a deliberate exclusion, not an error, as long as the
    # total remains positive.
    regime = _build_terminal_regime(
        objective=ParetoObjective(weights={"f": 0.0, "m": 1.0})
    )
    objective = regime.pareto_objective
    assert objective is not None
    assert objective.weights == {"f": 0.0, "m": 1.0}
