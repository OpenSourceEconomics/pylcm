"""Candidate banks retain numeric metadata beside their conditional value rows."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.nbegm_step import _make_nbegm_query_data
from _lcm.egm.outer_candidates import (
    OuterCandidateResult,
    build_outer_candidate_bank,
)
from _lcm.egm.published_policy import EGMSimPolicy


@pytest.mark.parametrize(
    "retain_query_data", [False, True], ids=["literal", "query-data"]
)
def test_candidate_bank_preserves_each_nodes_typed_channels(
    *,
    retain_query_data: bool,
) -> None:
    """Each candidate retains floating rows and integer/Boolean query metadata."""
    nodes = jnp.asarray([0.25, 0.75])
    results = []
    for index in range(2):
        offset = 10.0 * index
        grid = jnp.asarray([2.0, 4.0])
        value = jnp.asarray([3.0, 7.0]) + offset
        marginal = jnp.asarray([5.0, 11.0]) + offset
        policy = jnp.asarray([1.0, 2.0]) + offset
        data = (
            None
            if not retain_query_data
            else _make_nbegm_query_data(
                value=value,
                marginal=marginal,
                policy=policy,
                savings=jnp.asarray([0.5, 1.0]),
                floor_used=jnp.asarray(0.25),
                liquid_grid=grid,
                breakpoints=jnp.asarray([] if index == 0 else [3.0]),
                corner_continuation=jnp.asarray([13.0 + offset]),
            )
        )
        carry = EGMCarry(
            endog_grid=grid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(0.0),
            policy=policy,
            query_data=data,
        )
        results.append(
            OuterCandidateResult(
                outer_node=nodes[index],
                V_arr=value,
                carry=carry,
                sim_policy=EGMSimPolicy(
                    endog_grid=grid,
                    policy=policy,
                    value=value,
                    marginal_utility=marginal,
                ),
            )
        )
    bank = build_outer_candidate_bank(outer_nodes=nodes, results=results)
    assert bank.n_candidates == 2
    np.testing.assert_array_equal(bank.outer_nodes, [0.25, 0.75])
    np.testing.assert_array_equal(bank.V_arr, [[3.0, 7.0], [13.0, 17.0]])
    np.testing.assert_array_equal(
        bank.carry.marginal_utility, [[5.0, 11.0], [15.0, 21.0]]
    )
    assert bank.sim_policy is not None
    np.testing.assert_array_equal(bank.sim_policy.policy, [[1.0, 2.0], [11.0, 12.0]])
    np.testing.assert_array_equal(bank.candidate_valid, [True, True])
    for index, result in enumerate(results):
        restored = bank.candidate_carry(index)
        assert jax.tree.structure(restored) == jax.tree.structure(result.carry)
        for actual, expected in zip(
            jax.tree.leaves(restored),
            jax.tree.leaves(result.carry),
            strict=True,
        ):
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            np.testing.assert_array_equal(actual, expected)
    if retain_query_data:
        assert bank.carry.query_data is not None
        np.testing.assert_array_equal(bank.carry.query_data.owner_id, [[0, 0], [0, 1]])
        np.testing.assert_array_equal(
            bank.carry.query_data.link_valid, [[True], [False]]
        )
        assert bank.carry.query_data.owner_id.dtype == jnp.int32
        assert bank.carry.query_data.link_valid.dtype == jnp.bool_
    else:
        assert bank.carry.query_data is None
