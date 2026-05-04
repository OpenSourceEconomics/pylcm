from types import MappingProxyType

import jax.numpy as jnp
import pandas as pd
from jax import Array

from lcm.grids import Grid, IrregSpacedGrid
from lcm.interfaces import StateActionSpace
from lcm.typing import StateName, StateOrActionName


def create_state_action_space(
    *,
    variable_info: pd.DataFrame,
    grids: MappingProxyType[StateOrActionName, Grid],
    states: dict[StateName, Array] | None = None,
) -> StateActionSpace:
    """Create a state-action-space.

    Creates the state-action-space for the solution and simulation of a regime. For the
    simulation, states must be provided.

    Args:
        variable_info: The variable info table as returned by get_variable_info.
        grids: Immutable mapping of variable names to Grid spec objects.
        states: A dictionary of states. If None, the grids as specified in the regime
            are used.

    Returns:
        A state-action-space. Contains the grids of the discrete and continuous actions,
        the states, and the names of the state and action variables in the order they
        appear in the variable info table.

    """
    if states is None:
        _states = {
            sn: _grid_to_jax_or_placeholder(grids[sn])
            for sn in variable_info.query("is_state").index
        }
    else:
        _validate_all_states_present(
            provided_states=states,
            required_state_names=set(variable_info.query("is_state").index),
        )
        _states = states

    discrete_actions = {
        name: grids[name].to_jax()
        for name in variable_info.query("is_action & is_discrete").index
    }
    continuous_actions = {
        name: _grid_to_jax_or_placeholder(grids[name])
        for name in variable_info.query("is_action & is_continuous").index
    }
    state_and_discrete_action_names = tuple(
        variable_info.query("is_state | is_discrete").index
    )

    return StateActionSpace(
        states=MappingProxyType(_states),
        discrete_actions=MappingProxyType(discrete_actions),
        continuous_actions=MappingProxyType(continuous_actions),
        state_and_discrete_action_names=state_and_discrete_action_names,
    )


def _grid_to_jax_or_placeholder(grid: Grid) -> Array:
    """Return the grid's points, or a NaN placeholder for runtime-supplied grids.

    `IrregSpacedGrid.to_jax()` raises when its points haven't been supplied — that
    is the right behaviour everywhere except here: the base state-action space
    needs a *shape-correct* array to wire through pytree structures and AOT
    tracing before runtime substitution by
    `InternalRegime.state_action_space(regime_params=...)`. NaN (rather than
    zero) makes any accidental computation against the placeholder fail loudly.
    """
    if isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime:
        return jnp.full(grid.n_points, jnp.nan)
    return grid.to_jax()


def _validate_all_states_present(
    *, provided_states: dict[StateName, Array], required_state_names: set[StateName]
) -> None:
    """Check that all states are present in the provided states."""
    provided_state_names = set(provided_states)

    if required_state_names != provided_state_names:
        missing = required_state_names - provided_state_names
        too_many = provided_state_names - required_state_names
        raise ValueError(
            "You need to provide an initial array for each state variable in the "
            f"regime.\n\nMissing initial states: {missing}\n",
            f"Provided variables that are not states: {too_many}",
        )
