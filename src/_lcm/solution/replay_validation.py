"""Structural preflight for built-in simulation-policy artifacts."""

from collections.abc import Sequence

import numpy as np
from jax import Array

from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.egm.outer_replay_capability import OuterReplayCapability
from _lcm.egm.published_policy import EGMSimPolicy, NNBEGMSimPolicy
from _lcm.engine import NNBEGMPolicyRead, Regime


def validate_egm_sim_policy(
    *, policy: EGMSimPolicy, regime: Regime, extra_leading_axes: int = 0
) -> str | None:
    """Return a structural defect in a flat policy, or ``None`` when valid."""
    arrays = (
        policy.endog_grid,
        policy.policy,
        policy.value,
        policy.marginal_utility,
    )
    defect = _same_shape_and_dtype(arrays=arrays, label="EGMSimPolicy arrays")
    if defect is not None:
        return defect
    if not _is_floating_dtype(arrays[0].dtype):
        return "EGMSimPolicy arrays must have a floating dtype"
    if arrays[0].ndim == 0 or arrays[0].shape[-1] == 0:
        return "EGMSimPolicy trailing node axis must be non-empty"

    defect = _validate_egm_row_names(policy=policy, regime=regime)
    if defect is not None:
        return defect
    row_names = _egm_row_names(policy=policy)
    expected_rank = extra_leading_axes + len(row_names) + 1
    if arrays[0].ndim != expected_rank:
        return (
            f"EGMSimPolicy rank {arrays[0].ndim} does not match its row names "
            f"(expected {expected_rank})"
        )
    return _validate_named_axis_lengths(
        shape=arrays[0].shape,
        names=row_names,
        axis_offset=extra_leading_axes,
        regime=regime,
        label="EGMSimPolicy",
    )


def validate_nnbegm_sim_policy(
    *, policy: NNBEGMSimPolicy, regime: Regime
) -> str | None:
    """Return a structural defect in a finite NNBEGM bank, if any."""
    candidate_arrays = (
        policy.candidate_inner_action,
        policy.candidate_outer_target,
        policy.candidate_value,
    )
    defect = _validate_nnbegm_names(policy=policy, regime=regime)
    if defect is not None:
        return defect
    defect = _validate_nnbegm_array_layout(
        policy=policy,
        arrays=candidate_arrays,
        regime=regime,
    )
    if defect is not None:
        return defect
    for defect in (
        _validate_nnbegm_candidate_counts(policy=policy),
        _validate_nnbegm_discrete_metadata(policy=policy),
    ):
        if defect is not None:
            return defect
    return None


def validate_nested_egm_sim_policy(
    *, policy: NestedEGMSimPolicy, regime: Regime, policy_read: NNBEGMPolicyRead
) -> str | None:
    """Return a structural defect in a nested continuous-outer payload, if any."""
    defect = validate_egm_sim_policy(policy=policy.keeper, regime=regime)
    if defect is not None:
        return f"keeper: {defect}"

    defect = _validate_nested_adjuster(policy=policy, regime=regime)
    if defect is not None:
        return defect
    defect = _validate_nested_names(
        policy=policy, regime=regime, policy_read=policy_read
    )
    if defect is not None:
        return defect
    return None


def _validate_nnbegm_array_layout(
    *, policy: NNBEGMSimPolicy, arrays: Sequence[Array], regime: Regime
) -> str | None:
    """Check the shared candidate-array layout and declared state rank."""
    defect = _same_shape_and_dtype(
        arrays=arrays, label="NNBEGMSimPolicy candidate arrays"
    )
    if defect is not None:
        return defect
    if not _is_floating_dtype(arrays[0].dtype):
        return "NNBEGMSimPolicy candidate arrays must have a floating dtype"
    if arrays[0].ndim == 0 or arrays[0].shape[0] == 0:
        return "NNBEGMSimPolicy candidate axis must be non-empty"
    candidate_rank = arrays[0].ndim
    if candidate_rank != len(policy.state_names) + 1:
        return (
            f"NNBEGMSimPolicy rank {candidate_rank} does not match "
            f"state_names {policy.state_names!r}"
        )
    return _validate_named_axis_lengths(
        shape=arrays[0].shape,
        names=policy.state_names,
        axis_offset=1,
        regime=regime,
        label="NNBEGMSimPolicy",
    )


def _validate_nnbegm_names(*, policy: NNBEGMSimPolicy, regime: Regime) -> str | None:
    """Check uniqueness and model membership of finite-policy names."""
    if len(set(policy.state_names)) != len(policy.state_names):
        return "NNBEGMSimPolicy state_names are not unique"
    unknown_states = tuple(
        name for name in policy.state_names if name not in regime.simulation.state_names
    )
    if unknown_states:
        return f"NNBEGMSimPolicy has unknown state names {unknown_states!r}"
    action_names = (
        policy.inner_action_name,
        policy.outer_action_name,
        *policy.discrete_action_names,
    )
    if len(set(action_names)) != len(action_names):
        return "NNBEGMSimPolicy action names are not unique"
    unknown_actions = tuple(
        name for name in action_names if name not in regime.simulation.action_names
    )
    wrongly_discrete = tuple(
        name
        for name in (policy.inner_action_name, policy.outer_action_name)
        if name in regime.simulation.discrete_grids
    )
    wrongly_continuous = tuple(
        name
        for name in policy.discrete_action_names
        if name not in regime.simulation.discrete_grids
    )
    if wrongly_discrete or wrongly_continuous:
        return (
            "NNBEGMSimPolicy action topology is inconsistent: "
            f"continuous={wrongly_discrete!r}, discrete={wrongly_continuous!r}"
        )
    return (
        f"NNBEGMSimPolicy has unknown action names {unknown_actions!r}"
        if unknown_actions
        else None
    )


def _validate_nnbegm_candidate_counts(*, policy: NNBEGMSimPolicy) -> str | None:
    """Check keeper/adjuster counts against the declared outer grid."""
    n_candidates = policy.candidate_value.shape[0]
    outer_grid = policy.outer_grid_values
    if outer_grid.ndim != 1 or outer_grid.shape[0] == 0:
        return "NNBEGMSimPolicy outer_grid_values must be a non-empty vector"
    if outer_grid.dtype != policy.candidate_value.dtype:
        return "NNBEGMSimPolicy outer grid and candidate dtypes differ"
    if not isinstance(policy.replay_capability, OuterReplayCapability):
        return "NNBEGMSimPolicy replay_capability has the wrong type"
    if not 0 < policy.n_keeper_candidates < n_candidates:
        return "NNBEGMSimPolicy n_keeper_candidates is outside the candidate bank"
    n_adjuster_candidates = n_candidates - policy.n_keeper_candidates
    n_outer_nodes = outer_grid.shape[0]
    if (
        n_adjuster_candidates < n_outer_nodes
        or n_adjuster_candidates % n_outer_nodes != 0
    ):
        return "NNBEGMSimPolicy adjuster candidates do not divide over its outer grid"
    return None


def _validate_nnbegm_discrete_metadata(*, policy: NNBEGMSimPolicy) -> str | None:
    """Check optional discrete-code columns against their declared names."""
    codes = policy.candidate_discrete_actions
    names = policy.discrete_action_names
    if not names:
        return (
            "NNBEGMSimPolicy has codes without discrete action names"
            if codes is not None
            else None
        )
    if codes is None:
        return "NNBEGMSimPolicy discrete action metadata is absent"
    if codes.shape != (policy.candidate_value.shape[0], len(names)):
        return "NNBEGMSimPolicy discrete action metadata has the wrong shape"
    if np.dtype(codes.dtype).kind not in "iu":
        return "NNBEGMSimPolicy discrete action metadata is not integer"
    return None


def _validate_nested_adjuster(
    *, policy: NestedEGMSimPolicy, regime: Regime
) -> str | None:
    """Check the candidate axis and row agreement of a nested adjuster bank."""
    outer_nodes = policy.adjuster.outer_nodes
    if outer_nodes.ndim != 1 or outer_nodes.shape[0] == 0:
        return "adjuster outer_nodes must be a non-empty vector"
    if not _is_floating_dtype(outer_nodes.dtype):
        return "adjuster outer_nodes must have a floating dtype"
    defect = validate_egm_sim_policy(
        policy=policy.adjuster.policies,
        regime=regime,
        extra_leading_axes=1,
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
    if outer_nodes.dtype != adjuster_arrays[0].dtype:
        return "adjuster outer-node and policy dtypes differ"
    return _validate_nested_row_agreement(
        keeper=policy.keeper,
        adjuster=policy.adjuster.policies,
        adjuster_arrays=adjuster_arrays,
    )


def _validate_nested_names(
    *, policy: NestedEGMSimPolicy, regime: Regime, policy_read: NNBEGMPolicyRead
) -> str | None:
    """Check nested payload state/action names and static search settings."""
    validators = (
        _validate_nested_model_names,
        _validate_nested_reader_binding,
        _validate_nested_static_settings,
    )
    for validator in validators:
        defect = validator(policy=policy, regime=regime, policy_read=policy_read)
        if defect is not None:
            return defect
    return None


def _validate_nested_model_names(
    *,
    policy: NestedEGMSimPolicy,
    regime: Regime,
    policy_read: NNBEGMPolicyRead,  # noqa: ARG001
) -> str | None:
    """Check nested state/action uniqueness, membership, and grid topology."""
    state_names = (policy.outer_state_name, policy.liquid_state_name)
    action_names = (policy.outer_action_name, policy.inner_action_name)
    if len(set(state_names)) != len(state_names):
        return "NestedEGMSimPolicy state names are not unique"
    if len(set(action_names)) != len(action_names):
        return "NestedEGMSimPolicy action names are not unique"
    unknown_states = tuple(
        name for name in state_names if name not in regime.simulation.state_names
    )
    if unknown_states:
        return f"NestedEGMSimPolicy has unknown state names {unknown_states!r}"
    unknown_actions = tuple(
        name for name in action_names if name not in regime.simulation.action_names
    )
    if unknown_actions:
        return f"NestedEGMSimPolicy has unknown action names {unknown_actions!r}"
    wrongly_discrete_states = tuple(
        name for name in state_names if name in regime.simulation.discrete_state_names
    )
    wrongly_discrete_actions = tuple(
        name for name in action_names if name in regime.simulation.discrete_grids
    )
    if wrongly_discrete_states or wrongly_discrete_actions:
        return (
            "NestedEGMSimPolicy continuous-name topology is inconsistent: "
            f"states={wrongly_discrete_states!r}, "
            f"actions={wrongly_discrete_actions!r}"
        )
    return None


def _validate_nested_reader_binding(
    *, policy: NestedEGMSimPolicy, regime: Regime, policy_read: NNBEGMPolicyRead
) -> str | None:
    """Bind nested static names to the active reader and model functions."""
    reader_names = (
        (policy.outer_state_name, policy_read.outer_state_name),
        (policy.outer_post_decision_name, policy_read.outer_post_decision),
        (policy.outer_no_adjustment_name, policy_read.outer_no_adjustment_target),
    )
    if any(payload_name != reader_name for payload_name, reader_name in reader_names):
        return "NestedEGMSimPolicy names do not match the active NNBEGMPolicyRead"
    required_functions = (
        policy.outer_post_decision_name,
        policy.resources_target_name,
        *(
            (policy.outer_no_adjustment_name,)
            if policy.outer_no_adjustment_name is not None
            else ()
        ),
    )
    missing_functions = tuple(
        name for name in required_functions if name not in regime.simulation.functions
    )
    if missing_functions:
        return f"NestedEGMSimPolicy references unknown functions {missing_functions!r}"
    return None


def _validate_nested_static_settings(
    *,
    policy: NestedEGMSimPolicy,
    regime: Regime,  # noqa: ARG001
    policy_read: NNBEGMPolicyRead,  # noqa: ARG001
) -> str | None:
    """Check nested capability type/support and static search budget."""
    if not isinstance(policy.replay_capability, OuterReplayCapability):
        return "NestedEGMSimPolicy replay_capability has the wrong type"
    if not policy.replay_capability.continuous_replay_is_supported:
        return "NestedEGMSimPolicy replay_capability is not supported"
    if policy.golden_iterations < 0:
        return "NestedEGMSimPolicy golden_iterations must be non-negative"
    return None


def _same_shape_and_dtype(*, arrays: Sequence[Array], label: str) -> str | None:
    """Return a reason when array leaves do not share shape and dtype."""
    first = arrays[0]
    if any(array.shape != first.shape for array in arrays[1:]):
        return f"{label} do not share a shape"
    if any(array.dtype != first.dtype for array in arrays[1:]):
        return f"{label} do not share a dtype"
    return None


def _egm_row_names(*, policy: EGMSimPolicy) -> tuple[str, ...]:
    """Return flat EGM row axes in their declared storage order."""
    return (
        *policy.row_discrete_state_names,
        *policy.row_passive_state_names,
        *policy.row_discrete_action_names,
    )


def _validate_egm_row_names(*, policy: EGMSimPolicy, regime: Regime) -> str | None:
    """Check flat EGM row-name uniqueness, membership, and topology."""
    row_names = _egm_row_names(policy=policy)
    if len(set(row_names)) != len(row_names):
        return "EGMSimPolicy row names are not unique"
    wrong_discrete_states = tuple(
        name
        for name in policy.row_discrete_state_names
        if name not in regime.simulation.discrete_state_names
    )
    wrong_passive_states = tuple(
        name
        for name in policy.row_passive_state_names
        if name not in regime.simulation.state_names
        or name in regime.simulation.discrete_state_names
    )
    wrong_discrete_actions = tuple(
        name
        for name in policy.row_discrete_action_names
        if name not in regime.simulation.action_names
        or name not in regime.simulation.discrete_grids
    )
    defects = (
        ("discrete states", wrong_discrete_states),
        ("passive states", wrong_passive_states),
        ("discrete actions", wrong_discrete_actions),
    )
    invalid = tuple((kind, names) for kind, names in defects if names)
    return (
        f"EGMSimPolicy row topology is inconsistent: {invalid!r}" if invalid else None
    )


def _validate_named_axis_lengths(
    *,
    shape: tuple[int, ...],
    names: tuple[str, ...],
    axis_offset: int,
    regime: Regime,
    label: str,
) -> str | None:
    """Check every declared grid-valued axis against its canonical grid."""
    mismatches: list[tuple[str, int, int]] = []
    for position, name in enumerate(names):
        actual = shape[axis_offset + position]
        expected = _grid_length(regime=regime, name=name)
        if actual != expected:
            mismatches.append((name, actual, expected))
    return (
        f"{label} named-axis lengths are inconsistent: {tuple(mismatches)!r}"
        if mismatches
        else None
    )


def _grid_length(*, regime: Regime, name: str) -> int:
    """Return a canonical simulation grid's static coordinate count."""
    return int(regime.simulation.grids[name].to_jax().shape[0])


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


def _is_floating_dtype(dtype: object) -> bool:
    """Whether a JAX array dtype is floating-point."""
    return np.issubdtype(np.dtype(dtype), np.floating)


__all__ = [
    "validate_egm_sim_policy",
    "validate_nested_egm_sim_policy",
    "validate_nnbegm_sim_policy",
]
