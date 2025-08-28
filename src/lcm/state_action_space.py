"""Create a state space for a given model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from lcm.grids import ContinuousGrid, DiscreteGrid
from lcm.interfaces import InternalModel, StateActionSpace, StateSpaceInfo

if TYPE_CHECKING:
    from jax import Array


def create_state_action_space(
    model: InternalModel,
    *,
    states: dict[str, Array] | None = None,
    is_last_period: bool = False,
    multi_device_support: bool = False,
) -> StateActionSpace:
    """Create a state-action-space.

    Creates the state-action-space for the solution and simulation of a model. For the
    simulation, states must be provided.

    Args:
        model: A processed model.
        states: A dictionary of states. If None, the grids as specified in the model
            are used.
        is_last_period: Whether the state-action-space is created for the last period,
            in which case auxiliary variables are not included.
        multi_device_support: Whether to use sharded arrays to distribute the
            computation on multiple devices.

    Returns:
        A state-action-space. Contains the grids of the discrete and continuous actions,
        the states, and the names of the state and action variables in the order they
        appear in the variable info table.

    """
    vi = model.variable_info
    if is_last_period:
        vi = vi.query("enters_concurrent_valuation")

    if states is None:
        _states = {sn: model.grids[sn] for sn in vi.query("is_state").index}
        if multi_device_support:
            device_count = jax.device_count()
            sucess = False
            for state in _states:
                if (_states[state].shape[0] % device_count) == 0:
                    mesh = jax.make_mesh((device_count,), ("x"))
                    sharding = jax.sharding.NamedSharding(
                        mesh, jax.sharding.PartitionSpec("x")
                    )
                    _states[state] = jax.device_put(_states[state], device=sharding)
                    sucess = True
                    break
            if not sucess:
                raise ValueError(
                    "If you want to use multiple devices, at least one state variable has to"
                    f" have a number of gridpoints divisible by the number of available devices.\n"
                    f"Available devices: {device_count}",
                )
    else:
        _validate_all_states_present(
            provided_states=states,
            required_states_names=set(vi.query("is_state").index),
        )
        _states = states

    discrete_actions = {
        name: model.grids[name] for name in vi.query("is_action & is_discrete").index
    }
    continuous_actions = {
        name: model.grids[name] for name in vi.query("is_action & is_continuous").index
    }
    ordered_var_names = tuple(vi.query("is_state | is_discrete").index)

    return StateActionSpace(
        states=_states,
        discrete_actions=discrete_actions,
        continuous_actions=continuous_actions,
        states_and_discrete_actions_names=ordered_var_names,
    )


def create_state_space_info(
    model: InternalModel,
    *,
    is_last_period: bool,
) -> StateSpaceInfo:
    """Collect information on the state space for the model solution.

    A state-space information is a compressed representation of all feasible states.

    Args:
        model: A processed model.
        is_last_period: Whether the function is created for the last period.

    Returns:
        The state-space information.

    """
    vi = model.variable_info
    if is_last_period:
        vi = vi.query("enters_concurrent_valuation")

    state_names = vi.query("is_state").index.tolist()

    discrete_states = {
        name: grid_spec
        for name, grid_spec in model.gridspecs.items()
        if name in state_names and isinstance(grid_spec, DiscreteGrid)
    }

    continuous_states = {
        name: grid_spec
        for name, grid_spec in model.gridspecs.items()
        if name in state_names and isinstance(grid_spec, ContinuousGrid)
    }

    return StateSpaceInfo(
        states_names=tuple(state_names),
        discrete_states=discrete_states,
        continuous_states=continuous_states,
    )


def _validate_all_states_present(
    provided_states: dict[str, Array], required_states_names: set[str]
) -> None:
    """Check that all states are present in the provided states."""
    provided_states_names = set(provided_states)

    if required_states_names != provided_states_names:
        missing = required_states_names - provided_states_names
        too_many = provided_states_names - required_states_names
        raise ValueError(
            "You need to provide an initial array for each state variable in the model."
            f"\n\nMissing initial states: {missing}\n",
            f"Provided variables that are not states: {too_many}",
        )
