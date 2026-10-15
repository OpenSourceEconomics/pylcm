"""Outer continuation stacks keep replay rows out of their legacy layout."""

from dataclasses import replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import (
    EGMCarry,
    build_template_egm_query_data,
    egm_carry_role_tree,
    egm_query_role_tree,
)
from _lcm.egm.outer_envelope import stack_candidate_carries
from _lcm.execution.output_layout import StateAxesLeading
from _lcm.solution.negm import _stack_carry_template


def _candidate(*, offset: float, feature: bool) -> EGMCarry:
    """Give each row a concrete numeric identity and an inner replay policy."""
    query = (
        replace(
            build_template_egm_query_data(
                state_shape=(2,), action_shape=(), row_capacity=2, interval_capacity=1
            ),
            savings=jnp.asarray([[0.25, 0.5], [0.5, 0.75]]) + offset,
            floor_used=jnp.asarray([0.25, 0.5]),
            corner_continuation=jnp.asarray([[10.0], [20.0]]) + offset,
        )
        if feature
        else None
    )
    return EGMCarry(
        endog_grid=jnp.asarray([[2.0, 4.0], [6.0, 8.0]]) + offset,
        value=jnp.asarray([[10.0, 12.0], [14.0, 16.0]]) + offset,
        marginal_utility=jnp.asarray([[0.5, 0.25], [0.125, 0.0625]]) + offset,
        taste_shock_scale=jnp.asarray(0.0),
        policy=jnp.asarray([[1.0, 2.0], [3.0, 4.0]]) + offset,
        query_data=query,
    )


@pytest.mark.parametrize("feature", [False, True], ids=("legacy", "query"))
@pytest.mark.parametrize("surface", ["runtime", "template"])
def test_outer_stack_layout_matches_its_declared_optional_rows(
    *, surface: Literal["runtime", "template"], feature: bool
) -> None:
    """Legacy outer carries have four leaves; query transport retains its own rows."""
    first = _candidate(offset=0.0, feature=feature)
    second = _candidate(offset=1.0 if surface == "runtime" else 0.0, feature=feature)
    assert first.policy is not None
    assert first.breakpoints is None
    assert (first.query_data_for_transport() is not None) is feature
    assert len(jax.tree.leaves(first)) == (15 if feature else 5)
    result = (
        stack_candidate_carries(candidates=(first, second), nan_is_infeasible=True)
        if surface == "runtime"
        else _stack_carry_template(template=first, n_candidates=2)
    )
    assert result is not None
    row = StateAxesLeading(state_names=("z",), shape=(2, 2, 2))
    scalar = StateAxesLeading(state_names=(), shape=())
    query_template = (
        build_template_egm_query_data(
            state_shape=(2,),
            action_shape=(),
            outer_capacity=2,
            row_capacity=2,
            interval_capacity=1,
        )
        if feature
        else None
    )
    roles = egm_carry_role_tree(
        row=row,
        scalar=scalar,
        breakpoints=None,
        policy=row if feature else None,
        query_data=None
        if query_template is None
        else egm_query_role_tree(query_data=query_template, state_names=("z",)),
    )
    assert len(jax.tree.leaves(result)) == (15 if feature else 4)
    assert jax.tree.structure(result) == jax.tree.structure(roles)
    assert result.breakpoints is None
    offset = 1.0 if surface == "runtime" else 0.0
    expected_grid = np.array(
        [
            [[2.0, 4.0], [2.0 + offset, 4.0 + offset]],
            [[6.0, 8.0], [6.0 + offset, 8.0 + offset]],
        ]
    )
    np.testing.assert_array_equal(result.endog_grid, expected_grid)
    np.testing.assert_array_equal(result.value, expected_grid + 8.0)
    np.testing.assert_array_equal(
        result.marginal_utility,
        [
            [[0.5, 0.25], [0.5 + offset, 0.25 + offset]],
            [[0.125, 0.0625], [0.125 + offset, 0.0625 + offset]],
        ],
    )
    assert result.taste_shock_scale is first.taste_shock_scale
    if feature:
        data = result.query_data_for_transport()
        first_data = first.query_data_for_transport()
        assert data is not None
        assert first_data is not None
        assert data.floor_used is first_data.floor_used
        np.testing.assert_array_equal(
            result.policy,
            [
                [[1.0, 2.0], [1.0 + offset, 2.0 + offset]],
                [[3.0, 4.0], [3.0 + offset, 4.0 + offset]],
            ],
        )
        np.testing.assert_array_equal(
            data.savings,
            [
                [[0.25, 0.5], [0.25 + offset, 0.5 + offset]],
                [[0.5, 0.75], [0.5 + offset, 0.75 + offset]],
            ],
        )
        np.testing.assert_array_equal(
            data.corner_continuation,
            [[[10.0], [10.0 + offset]], [[20.0], [20.0 + offset]]],
        )
    else:
        assert result.policy is None
        assert result.query_data_for_transport() is None
