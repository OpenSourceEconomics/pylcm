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

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.collective import (
    _weighted_sum,
    collective_argmax_and_readout,
    collective_readout,
)
from _lcm.regime_building.zero_safe import zero_safe_average, zero_safe_weighted_term
from lcm import DiscreteGrid, LinSpacedGrid, categorical
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime
from lcm.typing import DiscreteAction, FloatND, ScalarInt

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
    # No zero weight, no +-inf value -> byte-identical to jnp.average.
    values = jnp.array([1.0, 3.0, 5.0])
    weights = jnp.array([0.2, 0.3, 0.5])
    result = zero_safe_average(values, weights=weights)
    expected = jnp.average(values, weights=weights)
    np.testing.assert_allclose(float(result), float(expected))


def test_zero_safe_average_axis_reduction_matches_jnp_average_on_the_finite_path():
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    weights = jnp.array([0.25, 0.75])
    result = zero_safe_average(values, axis=1, weights=weights)
    expected = jnp.average(values, axis=1, weights=weights)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected))


def test_zero_safe_average_raises_eagerly_on_concretely_zero_total_weight():
    values = jnp.array([1.0, 2.0])
    weights = jnp.array([0.0, 0.0])
    with pytest.raises(ValueError, match="total weight is exactly zero"):
        zero_safe_average(values, weights=weights)


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
