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
def _x64(enabled: bool):
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
    with _x64(use_x64):
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


def _sequential_regime_mixture(terms: list[FloatND]) -> FloatND:
    """Replay `Q_and_F`'s regime-mixture accumulation: a Python left-fold `E += term`.

    `Q_and_F` accumulates `E_next_V = 0; for r: E += zero_safe_weighted_term(p_r, V_r)`.
    This is a distinct reduction from the vectorised `zero_safe_average` and is what the
    zero_safe.py ROUND-7 note characterizes: accurate to a few ULP of the exact mixture
    but NOT correctly-rounded. (A `jnp.sum(jnp.stack(terms))` returns the identical bits
    under jit, so consolidating the fold changes nothing — it is not implemented.)
    """
    total = jnp.zeros_like(terms[0])
    for term in terms:
        total = total + term
    return total


def test_sequential_regime_mixture_is_zero_mass_safe():
    """The load-bearing GUARANTEE: a zero-prob target with a -inf continuation -> 0.

    An unreached regime-transition target carries probability exactly 0; its
    continuation may be an admissible on-path -inf. The mixture fold must annihilate
    that term (contribute exactly 0), never inject a nan into E_next_V. This is the
    property no backend/dtype can take away — unlike the last-few-ULP reduction order,
    which is inherently non-portable (see the companion accuracy test).
    """
    values = jnp.array([1.5, -jnp.inf, 2.0, 0.5], dtype=jnp.float32)
    probs = jnp.array([0.5, 0.0, 0.3, 0.2], dtype=jnp.float32)
    terms = [zero_safe_weighted_term(p, v) for p, v in zip(probs, values, strict=True)]
    result = jax.jit(_sequential_regime_mixture)(terms)
    assert jnp.isfinite(result)
    # exact mixture over the positive-mass terms: .5*1.5 + .3*2 + .2*.5 = 1.45
    assert float(result) == pytest.approx(1.45, abs=1e-6)


def test_sequential_regime_mixture_is_accurate_but_not_correctly_rounded():
    """F1 (round-7): the mixture fold is a few ULP from exact, NOT correctly-rounded.

    On a valid all-positive 5-target float64 mixture the exact real value is
    0.026468077778441356. `Q_and_F`'s left-fold lands within a few ULP of it but not on
    it (measured bits ...842, 9 ULP below exact, under jit on hmg-office CPU jax 0.10.1)
    -- which is why at a knife-edge the downstream argmax can go either way and float64
    only shrinks, never removes, the event (zero_safe.py ROUND-7). The accuracy BOUND
    below is durable across backends; the exact side of a specific knife-edge is NOT, so
    it is deliberately not asserted. This is a characterization, not a fixable defect:
    no source restructuring makes a 5-term float sum correctly-rounded.
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
    exact = float(
        sum(Fraction(p) * Fraction(v) for p, v in zip(probs, vals, strict=True))
    )
    with _x64(enabled=True):
        v = jnp.asarray(vals, dtype=jnp.float64)
        w = jnp.asarray(probs, dtype=jnp.float64)
        terms = [zero_safe_weighted_term(w[i], v[i]) for i in range(v.shape[0])]
        fold = float(jax.jit(_sequential_regime_mixture)(terms))
    # Durable accuracy bound: within ~16 ULP of the exact real mixture (~3.5e-16).
    assert abs(fold - exact) <= 16 * np.spacing(abs(exact))


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
