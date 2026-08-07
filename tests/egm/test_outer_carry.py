"""The continuous collapse of an outer candidate bank into the keeper.

Analytic-surface unit tests: the collapsed value matches the true optimized
value, the collapsed marginal is the *conditional* marginal at the selected
outer action (envelope theorem), a central finite difference of the
collapsed value across the liquid axis reproduces the collapsed marginal
(the PR-5 envelope-consistency gate), the keeper wins where it is better and
on exact ties, and NaN padding / all-infeasible columns keep the finite
fold's semantics.
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


def _a(m: FloatND) -> FloatND:
    """The analytic interior optimizer of the outer action."""
    return 0.2 + 0.5 * m


def _conditional_value(f: FloatND, m: FloatND) -> FloatND:
    return -((f - _a(m)) ** 2) + jnp.log1p(m)


def _conditional_marginal(f: FloatND, m: FloatND) -> FloatND:
    """d/dm of the conditional value at fixed outer action f."""
    return (f - _a(m)) + 1.0 / (1.0 + m)


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


def test_adjuster_optimum_matches_the_analytic_envelope() -> None:
    """Collapsed value = optimized value; marginal = envelope marginal."""
    bank = _analytic_bank()
    keeper_value = jnp.log1p(_M) - 1.0  # strictly worse everywhere
    collapse = collapse_continuous_candidate_bank(
        keeper_v_arr=keeper_value,
        keeper_carry=_carry(keeper_value, jnp.full_like(_M, 0.123)),
        bank=bank,
        config=_CONFIG,
    )
    np.testing.assert_allclose(
        np.asarray(collapse.carry.value), np.asarray(jnp.log1p(_M)), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(collapse.carry.marginal_utility),
        np.asarray(1.0 / (1.0 + _M)),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(collapse.V_arr), np.asarray(jnp.log1p(_M)), atol=1e-6
    )


def test_envelope_consistency_fd_of_value_matches_marginal() -> None:
    """Central FD of the collapsed value across the liquid axis reproduces
    the collapsed marginal to 1e-4 relative error away from the edges."""
    bank = _analytic_bank()
    keeper_value = jnp.log1p(_M) - 1.0
    collapse = collapse_continuous_candidate_bank(
        keeper_v_arr=keeper_value,
        keeper_carry=_carry(keeper_value, jnp.zeros_like(_M)),
        bank=bank,
        config=_CONFIG,
    )
    value = np.asarray(collapse.carry.value)
    marginal = np.asarray(collapse.carry.marginal_utility)
    dm = float(_M[1] - _M[0])
    fd = (value[2:] - value[:-2]) / (2.0 * dm)
    relative_error = np.abs(fd - marginal[1:-1]) / np.abs(marginal[1:-1])
    assert float(relative_error.max()) < 1e-4


def test_keeper_wins_where_better_and_on_exact_ties() -> None:
    bank = _analytic_bank()
    # Keeper beats the adjuster optimum log1p(m) by 0.5 everywhere.
    keeper_value = jnp.log1p(_M) + 0.5
    keeper_marginal = jnp.full_like(_M, 7.0)
    collapse = collapse_continuous_candidate_bank(
        keeper_v_arr=keeper_value,
        keeper_carry=_carry(keeper_value, keeper_marginal),
        bank=bank,
        config=_CONFIG,
    )
    np.testing.assert_array_equal(
        np.asarray(collapse.carry.value), np.asarray(keeper_value)
    )
    np.testing.assert_array_equal(
        np.asarray(collapse.carry.marginal_utility), np.asarray(keeper_marginal)
    )
    np.testing.assert_array_equal(np.asarray(collapse.V_arr), np.asarray(keeper_value))


def test_nan_padding_rides_through_untouched() -> None:
    """Aligned NaN tails (endogenous-grid padding) stay NaN after collapse."""
    tail = jnp.array([jnp.nan, jnp.nan])
    padded_grid = jnp.concatenate([_M, tail])

    def padded_carry(value: FloatND, marginal: FloatND) -> EGMCarry:
        return EGMCarry(
            endog_grid=padded_grid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(0.0),
        )

    results = [
        OuterCandidateResult(
            outer_node=node,
            V_arr=jnp.concatenate([_conditional_value(node, _M), tail]),
            carry=padded_carry(
                jnp.concatenate([_conditional_value(node, _M), tail]),
                jnp.concatenate([_conditional_marginal(node, _M), tail]),
            ),
            sim_policy=None,
        )
        for node in _NODES
    ]
    bank = build_outer_candidate_bank(outer_nodes=_NODES, results=results)
    keeper_value = jnp.concatenate([jnp.log1p(_M) - 1.0, tail])
    collapse = collapse_continuous_candidate_bank(
        keeper_v_arr=keeper_value,
        keeper_carry=padded_carry(keeper_value, jnp.zeros_like(keeper_value)),
        bank=bank,
        config=_CONFIG,
    )
    assert bool(jnp.all(jnp.isnan(collapse.carry.value[-2:])))
    assert bool(jnp.all(jnp.isnan(collapse.V_arr[-2:])))


def test_all_infeasible_adjuster_column_keeps_finite_fold_semantics() -> None:
    """All-(-inf) adjuster cells lose to a finite keeper but take over a
    NaN-dead keeper cell as -inf value / 0.0 marginal — exactly `fmax`."""
    neg_inf_row = jnp.full_like(_M, -jnp.inf)
    results = [
        OuterCandidateResult(
            outer_node=node,
            V_arr=neg_inf_row,
            carry=_carry(neg_inf_row, jnp.zeros_like(_M)),
            sim_policy=None,
        )
        for node in _NODES
    ]
    bank = build_outer_candidate_bank(outer_nodes=_NODES, results=results)
    keeper_value = jnp.where(_M < 0.5, jnp.log1p(_M), jnp.nan)
    keeper_marginal = jnp.where(_M < 0.5, 1.0 / (1.0 + _M), 0.3)
    collapse = collapse_continuous_candidate_bank(
        keeper_v_arr=keeper_value,
        keeper_carry=_carry(keeper_value, keeper_marginal),
        bank=bank,
        config=_CONFIG,
    )
    finite_region = np.asarray(_M) < 0.5
    np.testing.assert_array_equal(
        np.asarray(collapse.carry.value)[finite_region],
        np.asarray(keeper_value)[finite_region],
    )
    assert bool(jnp.all(collapse.carry.value[~finite_region] == -jnp.inf))
    assert bool(jnp.all(collapse.carry.marginal_utility[~finite_region] == 0.0))
    assert bool(jnp.all(collapse.V_arr[~finite_region] == -jnp.inf))
