"""Model-authoritative preflight for built-in simulation-policy artifacts."""

from collections.abc import Mapping, Sequence

import numpy as np
from jax import Array

from _lcm.egm.nested_published_policy import NestedEGMSimPolicy, OuterPolicyBank
from _lcm.egm.outer_inversion import (
    DeclaredOuterInverse,
    coefficient_is_exactly_invertible,
)
from _lcm.egm.outer_replay_capability import OuterReplayCapability
from _lcm.egm.published_policy import EGMSimPolicy, NNBEGMSimPolicy
from _lcm.engine import EGMPolicyRead, NNBEGMPolicyRead


def _same_exactly_typed(*, actual: object, expected: object) -> bool:
    """Compare replay metadata without admitting equal values of another type."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple):
            return False
        return len(actual) == len(expected) and all(
            _same_exactly_typed(actual=actual_item, expected=expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def validate_egm_sim_policy(
    *,
    policy: EGMSimPolicy,
    policy_read: EGMPolicyRead,
    period: int,
    expected_node_count: int,
) -> str | None:
    """Return a defect against the model-owned flat-EGM route, if any."""
    return _validate_egm_payload(
        policy=policy,
        expected_discrete_state_names=policy_read.row_discrete_state_names,
        expected_passive_state_names=policy_read.row_passive_state_names,
        expected_discrete_action_names=policy_read.row_discrete_action_names,
        expected_axis_lengths=_axis_lengths_for_period(
            lengths_by_period=policy_read.row_axis_lengths_by_period,
            period=period,
            label="EGMSimPolicy",
        ),
        expected_node_count=expected_node_count,
        expected_float_dtype=policy_read.float_dtype,
        extra_leading_axes=0,
        label="EGMSimPolicy",
    )


def validate_nnbegm_sim_policy(
    *,
    policy: NNBEGMSimPolicy,
    policy_read: NNBEGMPolicyRead,
    period: int,
    expected_replay_capability: OuterReplayCapability,
) -> str | None:
    """Return a defect against the model-owned finite NNBEGM route, if any."""
    if policy_read.replay_policy_is_nested:
        return "NNBEGMSimPolicy does not match the model's nested replay route"
    if not policy_read.policy_applicable:
        return "NNBEGMSimPolicy is not applicable to the model's replay route"

    defect = _validate_nnbegm_names(policy=policy, policy_read=policy_read)
    if defect is not None:
        return defect
    defect = _validate_nnbegm_array_layout(
        policy=policy,
        policy_read=policy_read,
        period=period,
    )
    if defect is not None:
        return defect
    for defect in (
        _validate_nnbegm_candidate_counts(
            policy=policy,
            policy_read=policy_read,
            period=period,
            expected_replay_capability=expected_replay_capability,
        ),
        _validate_nnbegm_discrete_metadata(
            policy=policy,
            policy_read=policy_read,
        ),
    ):
        if defect is not None:
            return defect
    return None


def validate_nested_egm_sim_policy(  # noqa: PLR0911
    *,
    policy: NestedEGMSimPolicy,
    policy_read: NNBEGMPolicyRead,
    period: int,
    expected_node_count: int,
    expected_outer_nodes: tuple[float, ...],
    expected_replay_capability: OuterReplayCapability,
) -> str | None:
    """Return a defect against the model-owned nested NNBEGM route, if any."""
    if not policy_read.replay_policy_is_nested:
        return "NestedEGMSimPolicy does not match the model's finite replay route"
    if not policy_read.policy_applicable:
        return "NestedEGMSimPolicy is not applicable to the model's replay route"
    if type(policy.keeper) is not EGMSimPolicy:
        return "NestedEGMSimPolicy keeper has the wrong exact payload type"
    if type(policy.adjuster) is not OuterPolicyBank:
        return "NestedEGMSimPolicy adjuster has the wrong exact payload type"
    if type(policy.adjuster.policies) is not EGMSimPolicy:
        return "NestedEGMSimPolicy adjuster policies have the wrong exact payload type"

    expected_axis_lengths = _axis_lengths_for_period(
        lengths_by_period=policy_read.row_axis_lengths_by_period,
        period=period,
        label="NestedEGMSimPolicy",
    )
    defect = _validate_egm_payload(
        policy=policy.keeper,
        expected_discrete_state_names=policy_read.row_discrete_state_names,
        expected_passive_state_names=policy_read.row_passive_state_names,
        expected_discrete_action_names=(),
        expected_axis_lengths=expected_axis_lengths,
        expected_node_count=expected_node_count,
        expected_float_dtype=policy_read.float_dtype,
        extra_leading_axes=0,
        label="keeper EGMSimPolicy",
    )
    if defect is not None:
        return f"keeper: {defect}"

    defect = _validate_nested_adjuster(
        policy=policy,
        policy_read=policy_read,
        expected_axis_lengths=expected_axis_lengths,
        expected_node_count=expected_node_count,
        expected_outer_nodes=expected_outer_nodes,
    )
    if defect is not None:
        return defect
    defect = _validate_nested_static_fields(
        policy=policy,
        policy_read=policy_read,
        period=period,
        expected_replay_capability=expected_replay_capability,
    )
    if defect is not None:
        return defect
    return None


def _validate_egm_payload(
    *,
    policy: EGMSimPolicy,
    expected_discrete_state_names: tuple[str, ...],
    expected_passive_state_names: tuple[str, ...],
    expected_discrete_action_names: tuple[str, ...],
    expected_axis_lengths: tuple[int, ...],
    expected_node_count: int,
    expected_float_dtype: str,
    extra_leading_axes: int,
    label: str,
) -> str | None:
    """Check one EGM policy against model-owned names, lengths, and dtype."""
    arrays = (
        policy.endog_grid,
        policy.policy,
        policy.value,
        policy.marginal_utility,
    )
    defect = _same_shape_and_dtype(arrays=arrays, label=f"{label} arrays")
    if defect is not None:
        return defect
    defect = _validate_float_array_dtypes(
        arrays=arrays,
        expected=expected_float_dtype,
        label=f"{label} arrays",
    )
    if defect is not None:
        return defect
    if arrays[0].ndim == 0 or arrays[0].shape[-1] != expected_node_count:
        actual_node_count = arrays[0].shape[-1] if arrays[0].ndim else 0
        return (
            f"{label} trailing node axis has length {actual_node_count}, "
            f"expected {expected_node_count} from model authority"
        )

    expected_name_groups = (
        expected_discrete_state_names,
        expected_passive_state_names,
        expected_discrete_action_names,
    )
    actual_name_groups = (
        policy.row_discrete_state_names,
        policy.row_passive_state_names,
        policy.row_discrete_action_names,
    )
    if not _same_exactly_typed(
        actual=actual_name_groups, expected=expected_name_groups
    ):
        return (
            f"{label} row roles {actual_name_groups!r} do not match the "
            f"model-owned roles {expected_name_groups!r}"
        )

    expected_rank = extra_leading_axes + len(expected_axis_lengths) + 1
    if arrays[0].ndim != expected_rank:
        return (
            f"{label} rank {arrays[0].ndim} does not match the model-owned "
            f"row layout (expected {expected_rank})"
        )
    return _validate_axis_lengths(
        shape=arrays[0].shape,
        expected=expected_axis_lengths,
        axis_offset=extra_leading_axes,
        label=label,
    )


def _validate_nnbegm_array_layout(
    *,
    policy: NNBEGMSimPolicy,
    policy_read: NNBEGMPolicyRead,
    period: int,
) -> str | None:
    """Check finite candidate arrays against the model-owned layout."""
    arrays = (
        policy.candidate_inner_action,
        policy.candidate_outer_target,
        policy.candidate_value,
    )
    defect = _same_shape_and_dtype(
        arrays=arrays, label="NNBEGMSimPolicy candidate arrays"
    )
    if defect is not None:
        return defect
    defect = _validate_float_array_dtypes(
        arrays=arrays,
        expected=policy_read.float_dtype,
        label="NNBEGMSimPolicy candidate arrays",
    )
    if defect is not None:
        return defect
    if arrays[0].ndim == 0 or arrays[0].shape[0] == 0:
        return "NNBEGMSimPolicy candidate axis must be non-empty"
    if (
        policy_read.candidate_count is None
        or arrays[0].shape[0] != policy_read.candidate_count
    ):
        return (
            "NNBEGMSimPolicy candidate count does not match model authority: "
            f"got {arrays[0].shape[0]}, expected {policy_read.candidate_count!r}"
        )
    expected_axis_lengths = _axis_lengths_for_period(
        lengths_by_period=policy_read.state_axis_lengths_by_period,
        period=period,
        label="NNBEGMSimPolicy",
    )
    expected_rank = len(expected_axis_lengths) + 1
    if arrays[0].ndim != expected_rank:
        return (
            f"NNBEGMSimPolicy rank {arrays[0].ndim} does not match the "
            f"model-owned state layout (expected {expected_rank})"
        )
    return _validate_axis_lengths(
        shape=arrays[0].shape,
        expected=expected_axis_lengths,
        axis_offset=1,
        label="NNBEGMSimPolicy",
    )


def _validate_nnbegm_names(
    *, policy: NNBEGMSimPolicy, policy_read: NNBEGMPolicyRead
) -> str | None:
    """Require exact state axes and action roles from the model descriptor."""
    actual = (
        policy.state_names,
        policy.inner_action_name,
        policy.outer_action_name,
        policy.discrete_action_names,
    )
    expected = (
        policy_read.state_names,
        policy_read.inner_action_name,
        policy_read.outer_action_name,
        policy_read.discrete_action_names,
    )
    return (
        "NNBEGMSimPolicy state/action roles do not match model authority: "
        f"got {actual!r}, expected {expected!r}"
        if not _same_exactly_typed(actual=actual, expected=expected)
        else None
    )


def _validate_nnbegm_candidate_counts(  # noqa: PLR0911
    *,
    policy: NNBEGMSimPolicy,
    policy_read: NNBEGMPolicyRead,
    period: int,
    expected_replay_capability: OuterReplayCapability,
) -> str | None:
    """Check finite search nodes, counts, and capability against model authority."""
    outer_grid = policy.outer_grid_values
    if not isinstance(outer_grid, Array):
        return "NNBEGMSimPolicy outer_grid_values has the wrong array type"
    if outer_grid.ndim != 1 or outer_grid.shape[0] == 0:
        return "NNBEGMSimPolicy outer_grid_values must be a non-empty vector"
    defect = _validate_float_array_dtypes(
        arrays=(outer_grid,),
        expected=policy_read.float_dtype,
        label="NNBEGMSimPolicy outer grid",
    )
    if defect is not None:
        return defect
    if policy_read.outer_grid_values is None:
        return "NNBEGMSimPolicy has no model-owned finite outer grid"
    expected_grid = np.asarray(
        policy_read.outer_grid_values,
        dtype=np.dtype(policy_read.float_dtype),
    )
    if not np.array_equal(np.asarray(outer_grid), expected_grid):
        return "NNBEGMSimPolicy outer_grid_values differ from model authority"
    if not _same_exactly_typed(
        actual=policy.n_keeper_candidates,
        expected=policy_read.n_keeper_candidates,
    ):
        return (
            "NNBEGMSimPolicy n_keeper_candidates differs from model authority: "
            f"got {policy.n_keeper_candidates}, "
            f"expected {policy_read.n_keeper_candidates}"
        )
    n_candidates = policy.candidate_value.shape[0]
    if not 0 < policy.n_keeper_candidates < n_candidates:
        return "NNBEGMSimPolicy n_keeper_candidates is outside the candidate bank"
    n_adjuster_candidates = n_candidates - policy.n_keeper_candidates
    if (
        n_adjuster_candidates < outer_grid.shape[0]
        or n_adjuster_candidates % outer_grid.shape[0] != 0
    ):
        return "NNBEGMSimPolicy adjuster candidates do not divide over its outer grid"
    return _validate_replay_capability(
        capability=policy.replay_capability,
        policy_read=policy_read,
        period=period,
        expected=expected_replay_capability,
        require_continuous_support=False,
        label="NNBEGMSimPolicy",
    )


def _validate_nnbegm_discrete_metadata(  # noqa: PLR0911
    *, policy: NNBEGMSimPolicy, policy_read: NNBEGMPolicyRead
) -> str | None:
    """Check exact int32 categorical columns and outer-tiled code rows."""
    codes = policy.candidate_discrete_actions
    names = policy_read.discrete_action_names
    if not names:
        return (
            "NNBEGMSimPolicy has codes without model-owned discrete action names"
            if codes is not None
            else None
        )
    if codes is None:
        return "NNBEGMSimPolicy discrete action metadata is absent"
    if not isinstance(codes, Array):
        return "NNBEGMSimPolicy discrete action metadata has the wrong array type"
    candidate_count = policy_read.candidate_count
    if candidate_count is None:
        return "NNBEGMSimPolicy has no model-owned finite candidate count"
    expected_shape = (candidate_count, len(names))
    if codes.shape != expected_shape:
        return (
            "NNBEGMSimPolicy discrete action metadata has the wrong shape: "
            f"got {codes.shape!r}, expected {expected_shape!r}"
        )
    actual_dtype = str(np.dtype(codes.dtype))
    if actual_dtype != policy_read.integer_dtype:
        return (
            "NNBEGMSimPolicy discrete action metadata has non-canonical dtype: "
            f"got {actual_dtype!r}, expected {policy_read.integer_dtype!r}"
        )
    actual = np.asarray(codes)
    for position, name in enumerate(names):
        domain = policy_read.discrete_action_code_domains[name]
        if not np.isin(actual[:, position], np.asarray(domain)).all():
            return (
                "NNBEGMSimPolicy discrete action metadata contains codes outside "
                f"the model-owned domain for {name!r}: {domain!r}"
            )
    expected = np.asarray(
        policy_read.candidate_discrete_action_codes,
        dtype=np.dtype(policy_read.integer_dtype),
    )
    if not np.array_equal(actual, expected):
        return (
            "NNBEGMSimPolicy discrete action code rows differ from the "
            "model-owned Cartesian-product order"
        )
    return None


def _validate_nested_adjuster(  # noqa: PLR0911
    *,
    policy: NestedEGMSimPolicy,
    policy_read: NNBEGMPolicyRead,
    expected_axis_lengths: tuple[int, ...],
    expected_node_count: int,
    expected_outer_nodes: tuple[float, ...],
) -> str | None:
    """Check the nested adjuster bank against model-owned rows and dtype."""
    outer_nodes = policy.adjuster.outer_nodes
    if not isinstance(outer_nodes, Array):
        return "adjuster outer_nodes has the wrong array type"
    if outer_nodes.ndim != 1 or outer_nodes.shape[0] == 0:
        return "adjuster outer_nodes must be a non-empty vector"
    expected_nodes = np.asarray(
        expected_outer_nodes,
        dtype=np.dtype(policy_read.float_dtype),
    )
    if not np.array_equal(np.asarray(outer_nodes), expected_nodes):
        return "adjuster outer nodes differ from the exact generated model authority"
    defect = _validate_float_array_dtypes(
        arrays=(outer_nodes,),
        expected=policy_read.float_dtype,
        label="adjuster outer_nodes",
    )
    if defect is not None:
        return defect
    nodes = np.asarray(outer_nodes)
    if not np.isfinite(nodes).all() or not np.all(np.diff(nodes) > 0):
        return "adjuster outer_nodes must be finite and strictly increasing"

    defect = _validate_egm_payload(
        policy=policy.adjuster.policies,
        expected_discrete_state_names=policy_read.row_discrete_state_names,
        expected_passive_state_names=policy_read.row_passive_state_names,
        expected_discrete_action_names=(),
        expected_axis_lengths=expected_axis_lengths,
        expected_node_count=expected_node_count,
        expected_float_dtype=policy_read.float_dtype,
        extra_leading_axes=1,
        label="adjuster EGMSimPolicy",
    )
    if defect is not None:
        return f"adjuster: {defect}"
    adjuster_arrays = (
        policy.adjuster.policies.endog_grid,
        policy.adjuster.policies.policy,
        policy.adjuster.policies.value,
        policy.adjuster.policies.marginal_utility,
    )
    if any(array.shape[0] != outer_nodes.shape[0] for array in adjuster_arrays):
        return "adjuster policy candidate count does not match outer_nodes"
    return _validate_nested_row_agreement(
        keeper=policy.keeper,
        adjuster=policy.adjuster.policies,
        adjuster_arrays=adjuster_arrays,
    )


def _validate_nested_static_fields(
    *,
    policy: NestedEGMSimPolicy,
    policy_read: NNBEGMPolicyRead,
    period: int,
    expected_replay_capability: OuterReplayCapability,
) -> str | None:
    """Require every payload-carried route setting to equal model authority."""
    actual = (
        policy.outer_action_name,
        policy.outer_state_name,
        policy.outer_post_decision_name,
        policy.inner_action_name,
        policy.liquid_state_name,
        policy.outer_no_adjustment_name,
        policy.resources_target_name,
        policy.savings_lower_bound,
        policy.golden_iterations,
        policy.value_atol,
        policy.value_rtol,
    )
    expected = (
        policy_read.outer_action_name,
        policy_read.outer_state_name,
        policy_read.outer_post_decision,
        policy_read.inner_action_name,
        policy_read.liquid_state_name,
        policy_read.outer_no_adjustment_target,
        policy_read.resources_target,
        policy_read.savings_lower_bound,
        policy_read.golden_iterations,
        policy_read.value_atol,
        policy_read.value_rtol,
    )
    if not _same_exactly_typed(actual=actual, expected=expected):
        return (
            "NestedEGMSimPolicy static route settings do not match model "
            f"authority: got {actual!r}, expected {expected!r}"
        )
    return _validate_replay_capability(
        capability=policy.replay_capability,
        policy_read=policy_read,
        period=period,
        expected=expected_replay_capability,
        require_continuous_support=True,
        label="NestedEGMSimPolicy",
    )


def _validate_replay_capability(  # noqa: PLR0911
    *,
    capability: object,
    policy_read: NNBEGMPolicyRead,
    period: int,
    expected: OuterReplayCapability,
    require_continuous_support: bool,
    label: str,
) -> str | None:
    """Validate capability type/domain and the route's structural support gate."""
    if type(capability) is not OuterReplayCapability:
        return f"{label} replay_capability has the wrong type"
    inverse = capability.inverse
    if type(inverse) is not DeclaredOuterInverse:
        return f"{label} replay capability inverse has the wrong type"
    expected_inverse = expected.inverse
    actual_fields = (
        inverse.coefficient,
        inverse.low,
        inverse.high,
        capability.undeclared_functions,
        capability.unbindable_functions,
        capability.unavailable_keeper_states,
        capability.unaddressable_passive_states,
        capability.unaddressable_discrete_actions,
    )
    expected_fields = (
        expected_inverse.coefficient,
        expected_inverse.low,
        expected_inverse.high,
        expected.undeclared_functions,
        expected.unbindable_functions,
        expected.unavailable_keeper_states,
        expected.unaddressable_passive_states,
        expected.unaddressable_discrete_actions,
    )
    if not _same_exactly_typed(actual=actual_fields, expected=expected_fields):
        return f"{label} replay_capability differs from model authority"
    if not coefficient_is_exactly_invertible(inverse.coefficient):
        return f"{label} replay capability coefficient is not exactly invertible"
    expected_domain = policy_read.outer_state_domain_by_period.get(period)
    if expected_domain is None:
        return f"{label} has no model-owned outer-state domain for period {period}"
    if not _same_exactly_typed(
        actual=(inverse.low, inverse.high), expected=expected_domain
    ):
        return (
            f"{label} replay capability domain {(inverse.low, inverse.high)!r} "
            f"does not match model authority {expected_domain!r}"
        )
    if require_continuous_support and not capability.continuous_replay_is_supported:
        return f"{label} replay_capability is not supported"
    return None


def _same_shape_and_dtype(*, arrays: Sequence[Array], label: str) -> str | None:
    """Return a reason when array leaves do not share shape and dtype."""
    if any(not isinstance(array, Array) for array in arrays):
        return f"{label} contain a non-JAX array leaf"
    first = arrays[0]
    if any(array.shape != first.shape for array in arrays[1:]):
        return f"{label} do not share a shape"
    if any(array.dtype != first.dtype for array in arrays[1:]):
        return f"{label} do not share a dtype"
    return None


def _validate_float_array_dtypes(
    *, arrays: Sequence[Array], expected: str, label: str
) -> str | None:
    """Require every floating payload leaf to use the canonical model dtype."""
    if any(not isinstance(array, Array) for array in arrays):
        return f"{label} contain a non-JAX array leaf"
    actual = tuple(str(np.dtype(array.dtype)) for array in arrays)
    if any(dtype != expected for dtype in actual):
        return f"{label} dtypes {actual!r} do not match model authority {expected!r}"
    return None


def _axis_lengths_for_period(
    *,
    lengths_by_period: Mapping[int, tuple[int, ...]],
    period: int,
    label: str,
) -> tuple[int, ...]:
    """Read an active period's immutable axis lengths."""
    try:
        return lengths_by_period[period]
    except KeyError as error:
        raise ValueError(
            f"{label} has no model-owned axis descriptor for period {period}"
        ) from error


def _validate_axis_lengths(
    *,
    shape: tuple[int, ...],
    expected: tuple[int, ...],
    axis_offset: int,
    label: str,
) -> str | None:
    """Check named payload axes against model-owned lengths in exact order."""
    actual = tuple(shape[axis_offset : axis_offset + len(expected)])
    return (
        f"{label} named-axis lengths {actual!r} do not match model authority "
        f"{expected!r}"
        if actual != expected
        else None
    )


def _egm_row_names(*, policy: EGMSimPolicy) -> tuple[str, ...]:
    """Return flat EGM row axes in their declared storage order."""
    return (
        *policy.row_discrete_state_names,
        *policy.row_passive_state_names,
        *policy.row_discrete_action_names,
    )


def _validate_nested_row_agreement(
    *,
    keeper: EGMSimPolicy,
    adjuster: EGMSimPolicy,
    adjuster_arrays: Sequence[Array],
) -> str | None:
    """Check keeper and adjuster rows describe the same conditional grid."""
    if _egm_row_names(policy=keeper) != _egm_row_names(policy=adjuster):
        return "keeper and adjuster row-name metadata differ"
    keeper_arrays = (
        keeper.endog_grid,
        keeper.policy,
        keeper.value,
        keeper.marginal_utility,
    )
    if any(
        keeper_array.shape != adjuster_array.shape[1:]
        for keeper_array, adjuster_array in zip(
            keeper_arrays, adjuster_arrays, strict=True
        )
    ):
        return "keeper and adjuster row shapes differ"
    return None


__all__ = [
    "validate_egm_sim_policy",
    "validate_nested_egm_sim_policy",
    "validate_nnbegm_sim_policy",
]
