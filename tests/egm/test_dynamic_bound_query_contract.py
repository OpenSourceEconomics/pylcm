"""Current feasibility and bounded summaries preserve complete state meaning."""

import itertools
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import carry as carry_module
from _lcm.egm.outer_envelope import stack_candidate_carries
from _lcm.execution.output_layout import StateAxesLeading
from _lcm.solution import feasibility
from _lcm.solution.negm import _stack_carry_template
from _lcm.utils.counts import COUNT_CAPACITY
from lcm.exceptions import UnsupportedOperationError
from lcm.solver_api import EGM_ENDOGENOUS_COORDINATE


@pytest.mark.parametrize("flags", tuple(itertools.product((False, True), repeat=6)))
def test_joint_branch_feasibility_counts_surviving_actions(
    *, flags: tuple[bool, ...]
) -> None:
    """A state is empty exactly when no alternative meets every requirement."""
    helper = getattr(feasibility, "count_current_feasibility", None)
    assert helper is not None, "Current feasibility needs a joint branch reducer."
    budgets = flags[:2]
    predicates = (flags[2:4], flags[4:6])
    expected = int(
        not any(
            budgets[action] and all(predicate[action] for predicate in predicates)
            for action in range(2)
        )
    )
    result = jax.jit(helper)(
        budget_feasible=jnp.asarray([budgets]),
        predicate_feasible=jnp.asarray([predicates]).transpose(0, 2, 1),
        value=jnp.asarray([1.0]),
        active_states=jnp.asarray([True]),
    )
    np.testing.assert_array_equal(jnp.stack(result), [expected, 0])


@pytest.mark.parametrize("active", [False, True])
def test_feasible_infinite_continuation_is_other_infinite(*, active: bool) -> None:
    """An active feasible state with impossible continuation is never empty."""
    helper = getattr(feasibility, "count_current_feasibility", None)
    assert helper is not None, "Current feasibility needs a joint branch reducer."
    result = helper(
        budget_feasible=jnp.asarray([[True, False]]),
        predicate_feasible=jnp.asarray([[[True], [True]]]),
        value=jnp.asarray([-jnp.inf]),
        active_states=jnp.asarray([active]),
    )
    np.testing.assert_array_equal(jnp.stack(result), [0, int(active)])


def test_feasibility_count_addition_saturates() -> None:
    """Large state populations retain a positive bounded count."""
    result = feasibility.add_feasibility_counts(
        total=(jnp.asarray(COUNT_CAPACITY, jnp.int32), jnp.asarray(5, jnp.int32)),
        block=(jnp.asarray(1, jnp.int32), jnp.asarray(3, jnp.int32)),
    )
    np.testing.assert_array_equal(jnp.stack(result), [COUNT_CAPACITY, 8])


@pytest.mark.parametrize("corner", ["compact", "liquid"])
def test_query_payload_roundtrip_preserves_named_numeric_leaves(*, corner: str) -> None:
    """Every query leaf retains its original buffer and field path through JAX."""
    builder = getattr(carry_module, "build_template_egm_query_data", None)
    assert builder is not None, "EGM carries need optional typed query transport."
    query = builder(
        state_shape=(2,),
        action_shape=(3,),
        row_capacity=5,
        interval_capacity=1,
        liquid_corner=corner == "liquid",
    )
    template = carry_module.build_template_egm_carry(
        n_rows=5,
        leading_shape=(2, 3),
        query_data=query,
    )
    leaves, tree = jax.tree_util.tree_flatten(template)
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    original = template.leaves()
    observed = restored.leaves()
    assert tuple(
        (path, observed[path] is value) for path, value in original.items()
    ) == (tuple((path, True) for path in original))


@pytest.mark.parametrize("feature", [False, True])
def test_whole_query_record_transport_preserves_identity(*, feature: bool) -> None:
    """Whole-record transport retains the optional record and numeric leaf schema."""
    query = (
        carry_module.build_template_egm_query_data(
            state_shape=(),
            action_shape=(),
            row_capacity=2,
            interval_capacity=1,
        )
        if feature
        else None
    )
    carry = carry_module.build_template_egm_carry(n_rows=2, query_data=query)
    accessor = getattr(carry, "query_data_for_transport", None)
    assert accessor is not None, "Whole query records need a typed transport accessor."
    assert accessor() is query
    assert ("query_data",) not in carry.leaves()
    assert all(isinstance(leaf, jax.Array) for leaf in carry.leaves().values())
    expected_paths = (
        {("query_data", *path) for path in query.leaves()}
        if query is not None
        else set()
    )
    assert {
        path for path in carry.leaves() if path[:1] == ("query_data",)
    } == expected_paths


@pytest.mark.parametrize("corner", ["compact", "liquid"])
def test_query_templates_have_independent_status_and_floor_axes(*, corner: str) -> None:
    """Inert status, fixed-cell floors and corner links have declared shapes."""
    builder = getattr(carry_module, "build_template_egm_query_data", None)
    assert builder is not None, "EGM carries need optional typed query transport."
    query = builder(
        state_shape=(2,),
        action_shape=(3,),
        row_capacity=5,
        interval_capacity=1,
        liquid_corner=corner == "liquid",
    )
    actual = {
        "floor": (query.floor_used.shape, str(query.floor_used.dtype)),
        "status": (query.node_kind.shape, str(query.node_kind.dtype)),
        "links": (query.link_valid.shape, str(query.link_valid.dtype)),
        "corner": (
            query.liquid_corner_data.node_kind.shape
            if corner == "liquid"
            else query.corner_kind.shape
        ),
    }
    assert actual == {
        "floor": ((2,), str(query.savings.dtype)),
        "status": ((2, 3, 5), "int32"),
        "links": ((2, 3, 4), "bool"),
        "corner": (2, 3, 5, 1) if corner == "liquid" else (2, 3, 1),
    }


@pytest.mark.parametrize("method", ["value_at", "marginal_at"])
def test_feature_rows_require_the_model_bound_query_adapter(*, method: str) -> None:
    """A query coordinate alone cannot authorize current bound or predicate reads."""
    template = carry_module.build_template_egm_carry(
        n_rows=5,
        query_data=carry_module.build_template_egm_query_data(
            state_shape=(),
            action_shape=(),
            row_capacity=5,
            interval_capacity=1,
        ),
    )
    arguments = {"query": jnp.asarray(0.5)}
    if method == "marginal_at":
        arguments["state"] = EGM_ENDOGENOUS_COORDINATE
    with pytest.raises(UnsupportedOperationError, match="model-bound"):
        getattr(template, method)(**arguments)


@pytest.mark.parametrize("liquid_corner", [False, True])
@pytest.mark.parametrize("template_only", [False, True])
def test_outer_stack_preserves_conditional_query_buffers(
    *, liquid_corner: bool, template_only: bool
) -> None:
    """Outer stacking preserves every query leaf with its own declared axes."""
    query = carry_module.build_template_egm_query_data(
        state_shape=(2,),
        action_shape=(3,),
        row_capacity=5,
        interval_capacity=2,
        liquid_corner=liquid_corner,
    )
    carry = carry_module.build_template_egm_carry(
        n_rows=5,
        leading_shape=(2, 3),
        query_data=query,
    )
    result = (
        _stack_carry_template(template=carry, n_candidates=2)
        if template_only
        else stack_candidate_carries(candidates=(carry, carry))
    )
    assert result is not None
    assert result.query_data is not None, "Outer stacking must retain query transport."
    for path, leaf in query.leaves().items():
        actual = result.query_data.leaves()[path]
        if path == ("floor_used",):
            assert actual is leaf
        else:
            axis = -3 if path[0] == "liquid_corner_data" else -2
            np.testing.assert_array_equal(actual, jnp.stack((leaf, leaf), axis=axis))


def test_outer_stack_rejects_partial_query_transport() -> None:
    """Every outer candidate contributes the same declared query structure."""
    query = carry_module.build_template_egm_query_data(
        state_shape=(2,),
        action_shape=(),
        row_capacity=5,
        interval_capacity=1,
    )
    carry = carry_module.build_template_egm_carry(
        n_rows=5,
        leading_shape=(2,),
        query_data=query,
    )
    other = replace(carry, query_data=None)
    with pytest.raises(TypeError, match="query"):
        stack_candidate_carries(candidates=(carry, other))


@pytest.mark.parametrize("liquid_corner", [False, True])
def test_query_output_roles_follow_each_leafs_own_axes(*, liquid_corner: bool) -> None:
    """Every query leaf names the fixed states and keeps its own shape and dtype."""
    builder = getattr(carry_module, "egm_query_role_tree", None)
    assert builder is not None, "Each query buffer needs an explicit output role."
    query = carry_module.build_template_egm_query_data(
        state_shape=(2,),
        action_shape=(3,),
        outer_capacity=4,
        row_capacity=5,
        interval_capacity=2,
        liquid_corner=liquid_corner,
    )
    roles = builder(query_data=query, state_names=("income",))
    role_leaves = {
        tuple(key.name for key in path): role
        for path, role in jax.tree_util.tree_leaves_with_path(roles)
    }
    for path, leaf in query.leaves().items():
        role = role_leaves[path]
        assert type(role) is StateAxesLeading
        assert (role.state_names, role.n_free_leading_axes, role.shape, role.dtype) == (
            ("income",),
            0,
            leaf.shape,
            leaf.dtype,
        )


@pytest.mark.parametrize("feature", [False, True])
def test_outer_stack_nan_normalization_preserves_feature_poison(
    *, feature: bool
) -> None:
    """Feature support preserves invalid arithmetic while legacy rows normalize."""
    query = (
        carry_module.build_template_egm_query_data(
            state_shape=(),
            action_shape=(),
            row_capacity=2,
            interval_capacity=1,
        )
        if feature
        else None
    )
    if query is not None:
        query = replace(query, node_kind=jnp.asarray([1, 1], dtype=jnp.int32))
    carry = carry_module.build_template_egm_carry(n_rows=2, query_data=query)
    carry = replace(
        carry,
        value=jnp.asarray([jnp.nan, 2.0]),
        marginal_utility=jnp.asarray([jnp.nan, 3.0]),
    )
    result = stack_candidate_carries(candidates=(carry, carry), nan_is_infeasible=True)
    expected_value = [jnp.nan if feature else -jnp.inf, 2.0]
    expected_marginal = [jnp.nan if feature else 0.0, 3.0]
    np.testing.assert_array_equal(result.value, [expected_value, expected_value])
    np.testing.assert_array_equal(
        result.marginal_utility, [expected_marginal, expected_marginal]
    )
    if feature:
        assert result.query_data is not None
        np.testing.assert_array_equal(result.query_data.node_kind, [[1, 1], [1, 1]])
