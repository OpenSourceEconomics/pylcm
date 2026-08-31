"""Fixed-cost aggregation inside the continuous collapse.

The keeper/adjuster fold under `UniformObservedFixedCost` must reproduce the
closed form's expectation cell by cell, degrade exactly to the deterministic
fold at `B = 0`, keep the degeneracy vocabulary, and publish an envelope-
consistent expected marginal (a central finite difference of the collapsed
carry value across the liquid axis reproduces the probability-weighted
marginal blend).
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.carry import EGMCarry
from _lcm.egm.outer_candidates import (
    OuterCandidateBank,
    OuterCandidateResult,
    build_outer_candidate_bank,
)
from _lcm.egm.outer_carry import collapse_continuous_candidate_bank
from _lcm.egm.outer_search import AdaptiveOuterMesh
from lcm import LinSpacedGrid
from lcm.typing import FloatND

_M = jnp.linspace(0.0, 1.0, 101)  # liquid axis
_NODES = jnp.linspace(0.0, 1.0, 33)  # outer mesh
_CONFIG = AdaptiveOuterMesh(
    initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=33),
    golden_iterations=40,
)
_SUPPORT = (0.0, 1.0)


def _a(m: FloatND) -> FloatND:
    return 0.2 + 0.5 * m


def _conditional_value(f: FloatND, m: FloatND) -> FloatND:
    return -((f - _a(m)) ** 2) + jnp.log1p(m)


def _conditional_marginal(f: FloatND, m: FloatND) -> FloatND:
    return (f - _a(m)) + 1.0 / (1.0 + m)


def _keeper_value(m: FloatND) -> FloatND:
    # Crosses the adjuster optimum log1p(m): better at low m, worse at high m,
    # so the fixed-cost cutoff is interior somewhere on the axis.
    return jnp.log1p(m) + 0.15 - 0.4 * m


def _keeper_marginal(m: FloatND) -> FloatND:
    return 1.0 / (1.0 + m) - 0.4


def _carry(value: FloatND, marginal: FloatND) -> EGMCarry:
    return EGMCarry(
        endog_grid=jnp.broadcast_to(_M, value.shape),
        value=value,
        marginal_utility=marginal,
        taste_shock_scale=jnp.asarray(0.0),
    )


def _analytic_bank() -> OuterCandidateBank:
    results = [
        OuterCandidateResult(
            outer_node=node,
            V_arr=_conditional_value(node, _M),
            carry=_carry(_conditional_value(node, _M), _conditional_marginal(node, _M)),
            sim_policy=None,
        )
        for node in _NODES
    ]
    return build_outer_candidate_bank(outer_nodes=_NODES, results=results)


def _collapse(scale: float | None):
    bank = _analytic_bank()
    keeper = _carry(_keeper_value(_M), _keeper_marginal(_M))
    return collapse_continuous_candidate_bank(
        keeper_v_arr=_keeper_value(_M),
        keeper_carry=keeper,
        bank=bank,
        config=_CONFIG,
        fixed_cost_scale=None if scale is None else jnp.asarray(scale),
        fixed_cost_support=None if scale is None else _SUPPORT,
    )


def _expected_by_quadrature(
    keeper: np.ndarray, adjuster: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(_SUPPORT[0], _SUPPORT[1], 100_001)
    chi = 0.5 * (edges[:-1] + edges[1:])
    net = adjuster[:, None] - scale * chi[None, :]
    values = np.maximum(keeper[:, None], net)
    return values.mean(axis=1), (net > keeper[:, None]).mean(axis=1)


def test_zero_scale_reproduces_the_deterministic_fold() -> None:
    deterministic = _collapse(None)
    zero_scale = _collapse(0.0)
    np.testing.assert_array_equal(
        np.asarray(zero_scale.V_arr), np.asarray(deterministic.V_arr)
    )
    np.testing.assert_array_equal(
        np.asarray(zero_scale.carry.value), np.asarray(deterministic.carry.value)
    )
    np.testing.assert_array_equal(
        np.asarray(zero_scale.carry.marginal_utility),
        np.asarray(deterministic.carry.marginal_utility),
    )
    assert deterministic.adjustment_probability is None
    assert zero_scale.adjustment_probability is not None


def test_expected_value_matches_dense_quadrature() -> None:
    scale = 0.35
    collapse = _collapse(scale)
    keeper = np.asarray(_keeper_value(_M))
    adjuster = np.asarray(collapse.value_search.value)
    expected, probability = _expected_by_quadrature(keeper, adjuster, scale)
    np.testing.assert_allclose(np.asarray(collapse.V_arr), expected, atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(collapse.adjustment_probability), probability, atol=1e-4
    )
    p = np.asarray(collapse.adjustment_probability)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.any((p > 0.0) & (p < 1.0)), "no interior cutoff cell exercised"


def test_expectation_is_bounded_by_keeper_and_hard_maximum() -> None:
    scale = 0.35
    collapse = _collapse(scale)
    deterministic = _collapse(None)
    value = np.asarray(collapse.V_arr)
    assert np.all(value <= np.asarray(deterministic.V_arr) + 1e-12)
    assert np.all(value >= np.asarray(_keeper_value(_M)) - 1e-12)
    assert np.any(value < np.asarray(deterministic.V_arr) - 1e-6)


def test_collapsed_marginal_is_the_derivative_of_the_collapsed_value() -> None:
    """Envelope consistency under the expectation fold.

    d/dm E[max(v_K(m), v_A(m) - B chi)] = p v_A'(m) + (1 - p) v_K'(m); the
    published marginal must match a central finite difference of the
    published value away from the cutoff-kink cells (where the FD straddles
    the probability's own kink, second-order effects enter).
    """
    scale = 0.35
    collapse = _collapse(scale)
    value = np.asarray(collapse.carry.value)
    marginal = np.asarray(collapse.carry.marginal_utility)
    m = np.asarray(_M)
    fd = (value[2:] - value[:-2]) / (m[2:] - m[:-2])
    err = np.abs(fd - marginal[1:-1])
    assert float(np.quantile(err, 0.9)) < 5e-3
    assert float(np.median(err)) < 1e-3


def test_degeneracies_keep_the_fold_vocabulary() -> None:
    """NaN-dead sides act as -inf; both-dead keeps the keeper entry."""
    nodes = jnp.asarray([0.0, 1.0])
    live = jnp.asarray([1.0, 1.0, -jnp.inf, jnp.nan])
    keeper_value = jnp.asarray([2.0, -jnp.inf, -jnp.inf, jnp.nan])

    def small_carry(value: FloatND) -> EGMCarry:
        return EGMCarry(
            endog_grid=jnp.linspace(0.0, 1.0, value.shape[0]),
            value=value,
            marginal_utility=jnp.zeros_like(value),
            taste_shock_scale=jnp.asarray(0.0),
        )

    results = [
        OuterCandidateResult(
            outer_node=node,
            V_arr=live,
            carry=small_carry(live),
            sim_policy=None,
        )
        for node in nodes
    ]
    bank = build_outer_candidate_bank(outer_nodes=nodes, results=results)
    collapse = collapse_continuous_candidate_bank(
        keeper_v_arr=keeper_value,
        keeper_carry=small_carry(keeper_value),
        bank=bank,
        config=AdaptiveOuterMesh(
            initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
            golden_iterations=8,
        ),
        fixed_cost_scale=jnp.asarray(2.0),
        fixed_cost_support=_SUPPORT,
    )
    value = np.asarray(collapse.V_arr)
    p = np.asarray(collapse.adjustment_probability)
    # keeper 2 vs adjuster 1 - 2 chi: keeper always wins.
    np.testing.assert_allclose(value[0], 2.0)
    np.testing.assert_allclose(p[0], 0.0)
    # dead keeper, live adjuster: forced adjustment, E[1 - 2 chi] = 0.
    np.testing.assert_allclose(value[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(p[1], 1.0)
    # live keeper (-inf) vs dead adjuster stays -inf with p = 0... both -inf:
    assert value[2] == -np.inf
    np.testing.assert_allclose(p[2], 0.0)
    # both NaN-dead: the keeper's NaN entry rides through.
    assert np.isnan(value[3])
    np.testing.assert_allclose(p[3], 0.0)
