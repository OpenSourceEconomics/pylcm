from types import MappingProxyType

import pandas as pd
from jax import Array

from lcm.grids import ContinuousGrid, DiscreteGrid
from lcm.input_processing.util import get_gridspecs, get_variable_info
from lcm.interfaces import StateActionSpace, StateSpaceInfo
from lcm.regime import Regime
from lcm.shocks import _ShockGrid


def create_state_action_space(
    variable_info: pd.DataFrame,
    grids: MappingProxyType[str, Array],
    *,
    states: dict[str, Array] | None = None,
) -> StateActionSpace:
    """Create a state-action-space.

    Creates the state-action-space for the solution and simulation of a regime. For the
    simulation, states must be provided.

    Args:
        variable_info: The variable info table as returned by get_variable_info.
        grids: A dictionary of grids as returned by get_grids.
        states: A dictionary of states. If None, the grids as specified in the regime
            are used.

    Returns:
        A state-action-space. Contains the grids of the discrete and continuous actions,
        the states, and the names of the state and action variables in the order they
        appear in the variable info table.

    """
    vi = variable_info.copy()

    if states is None:
        _states = {sn: grids[sn] for sn in vi.query("is_state").index}
    else:
        _validate_all_states_present(
            provided_states=states,
            required_states_names=set(vi.query("is_state").index),
        )
        _states = states

    discrete_actions = {
        name: grids[name] for name in vi.query("is_action & is_discrete").index
    }
    continuous_actions = {
        name: grids[name] for name in vi.query("is_action & is_continuous").index
    }
    ordered_var_names = tuple(vi.query("is_state | is_discrete").index)

    return StateActionSpace(
        states=MappingProxyType(_states),
        discrete_actions=MappingProxyType(discrete_actions),
        continuous_actions=MappingProxyType(continuous_actions),
        states_and_discrete_actions_names=ordered_var_names,
    )


def create_state_space_info(regime: Regime) -> StateSpaceInfo:
    """Collect information on the state space for the regime solution.

    A state-space information is a compressed representation of all feasible states.

    Args:
        regime: Regime instance.

    Returns:
        The state-space information.

    """
    vi = get_variable_info(regime)
    gridspecs = get_gridspecs(regime)

    if regime.terminal:
        vi = vi.query("enters_concurrent_valuation")

    state_names = vi.query("is_state").index.tolist()

    discrete_states = {
        name: grid_spec
        for name, grid_spec in gridspecs.items()
        if (name in state_names and isinstance(grid_spec, DiscreteGrid))
        or isinstance(grid_spec, _ShockGrid)
    }

    continuous_states = {
        name: grid_spec
        for name, grid_spec in gridspecs.items()
        if name in state_names
        and isinstance(grid_spec, ContinuousGrid)
        and not isinstance(grid_spec, _ShockGrid)
    }

    return StateSpaceInfo(
        states_names=tuple(state_names),
        discrete_states=MappingProxyType(discrete_states),
        continuous_states=MappingProxyType(continuous_states),
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
            "You need to provide an initial array for each state variable in the "
            f"regime.\n\nMissing initial states: {missing}\n",
            f"Provided variables that are not states: {too_many}",
        )
