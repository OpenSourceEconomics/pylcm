from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from jax import Array, vmap
from numpy.testing import assert_array_equal

from lcm.dispatchers import simulation_spacemap, vmap_1d
from lcm.input_processing import process_model
from lcm.interfaces import StateActionSpace, Target
from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a, get_max_Q_over_a
from lcm.max_Q_over_c import get_argmax_and_max_Q_over_c, get_max_Q_over_c
from lcm.max_Qc_over_d import get_argmax_and_max_Qc_over_d, get_max_Qc_over_d
from lcm.next_state import get_next_state_function
from lcm.Q_and_F import get_Q_and_F
from lcm.state_action_space import create_state_action_space, create_state_space_info
from tests.test_models import (
    get_model_config,
)

if TYPE_CHECKING:
    from lcm.typing import (
        IntND,
    )


@pytest.fixture
def model_input():
    model_config = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=1)
    actions = model_config.actions
    actions["consumption"] = actions["consumption"].replace(stop=20)  # type: ignore[attr-defined]
    model_config = model_config.replace(actions=actions)
    model = process_model(model_config)

    state_space_info = create_state_space_info(
        model=model,
        is_last_period=False,
    )
    state_action_space = create_state_action_space(
        model=model,
        is_last_period=False,
    )
    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
        },
    }
    return {
        "model": model,
        "state_action_space": state_action_space,
        "state_space_info": state_space_info,
        "next_state": get_next_state_function(model, target=Target.SOLVE),
        "params": params,
    }


def test_max_Q_over_a_equal(model_input):
    params = model_input["params"]
    state_space_info = model_input["state_space_info"]
    state_action_space = model_input["state_action_space"]
    model = model_input["model"]

    Q_and_F = get_Q_and_F(
        model=model,
        next_state_space_info=state_space_info,
        period=0,
    )
    max_Q_over_a = get_max_Q_over_a(
        Q_and_F=Q_and_F,
        actions_names=(
            *state_action_space.discrete_actions,
            *state_action_space.continuous_actions,
        ),
        states_names=(*state_action_space.states,),
    )

    max_Q_over_c = get_max_Q_over_c(
        Q_and_F=Q_and_F,
        continuous_actions_names=(*state_action_space.continuous_actions,),
        states_and_discrete_actions_names=(
            *state_action_space.discrete_actions,
            *state_action_space.states,
        ),
    )
    max_Qc_over_d = get_max_Qc_over_d(
        random_utility_shock_type=model.random_utility_shocks,
        variable_info=model.variable_info,
        is_last_period=False,
    )

    V_arr_a = max_Q_over_a(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        next_V_arr=jnp.zeros((2, 2)),
        params=params,
    )

    Qc_arr = max_Q_over_c(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        next_V_arr=jnp.zeros((2, 2)),
        params=params,
    )
    V_arr_c_d = max_Qc_over_d(Qc_arr, params=params)

    assert_array_equal(V_arr_a, V_arr_c_d)


def test_argmax_Q_over_a_equal(model_input):
    params = model_input["params"]
    state_space_info = model_input["state_space_info"]
    state_action_space = model_input["state_action_space"]
    model = model_input["model"]
    Q_and_F = get_Q_and_F(
        model=model,
        next_state_space_info=state_space_info,
        period=0,
    )
    argmax_and_max_Q_over_a_func = get_argmax_and_max_Q_over_a(
        Q_and_F=Q_and_F,
        actions_names=(
            *state_action_space.discrete_actions,
            *state_action_space.continuous_actions,
        ),
    )
    argmax_and_max_Q_over_a = simulation_spacemap(
        argmax_and_max_Q_over_a_func,
        actions_names=(),
        states_names=tuple(state_action_space.states),
    )
    argmax_and_max_Q_over_c_func = get_argmax_and_max_Q_over_c(
        Q_and_F=Q_and_F,
        continuous_actions_names=(*state_action_space.continuous_actions,),
    )
    argmax_and_max_Q_over_c = simulation_spacemap(
        argmax_and_max_Q_over_c_func,
        actions_names=tuple(state_action_space.discrete_actions),
        states_names=tuple(state_action_space.states),
    )
    argmax_and_max_Qc_over_d = get_argmax_and_max_Qc_over_d(
        variable_info=model.variable_info,
    )

    indices_optimal_actions, V_arr = argmax_and_max_Q_over_a(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        next_V_arr=jnp.zeros((2, 2)),
        params=params,
    )

    indices_argmax_Q_over_c, Qc_arr = argmax_and_max_Q_over_c(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        next_V_arr=jnp.zeros((2, 2)),
        params=params,
    )
    indices_optimal_discrete_actions, V_arr = argmax_and_max_Qc_over_d(
        Qc_arr, params=params
    )
    discrete_actions_grid_shape = tuple(
        len(grid) for grid in state_action_space.discrete_actions.values()
    )
    continuous_actions_grid_shape = tuple(
        len(grid) for grid in state_action_space.continuous_actions.values()
    )
    actions_grid_shape = discrete_actions_grid_shape + continuous_actions_grid_shape
    indices_optimal_continuous_actions = _lookup_optimal_continuous_actions(
        indices_argmax_Q_over_c=indices_argmax_Q_over_c,
        discrete_argmax=indices_optimal_discrete_actions,
        discrete_actions_grid_shape=discrete_actions_grid_shape,
    )
    optimal_actions_c_d = _lookup_actions_from_indices(
        indices_optimal_discrete_actions=indices_optimal_discrete_actions,
        indices_optimal_continuous_actions=indices_optimal_continuous_actions,
        discrete_actions_grid_shape=discrete_actions_grid_shape,
        continuous_actions_grid_shape=continuous_actions_grid_shape,
        state_action_space=state_action_space,
    )

    optimal_actions_a = _lookup_actions_from_indices_2(
        indices_optimal_actions=indices_optimal_actions,
        actions_grid_shape=actions_grid_shape,
        state_action_space=state_action_space,
    )

    assert_array_equal(
        optimal_actions_a["retirement"], optimal_actions_c_d["retirement"]
    )
    assert_array_equal(
        optimal_actions_a["consumption"], optimal_actions_c_d["consumption"]
    )


@partial(vmap_1d, variables=("indices_argmax_Q_over_c", "discrete_argmax"))
def _lookup_optimal_continuous_actions(
    indices_argmax_Q_over_c: IntND,
    discrete_argmax: IntND,
    discrete_actions_grid_shape: tuple[int, ...],
) -> IntND:
    """Look up the optimal continuous action index given index of discrete action.

    Args:
        indices_argmax_Q_over_c: Index array of optimal continous actions conditional on
            discrete actions and states.
        discrete_argmax: Index array of optimal discrete actions.
        discrete_actions_grid_shape: Shape of the discrete actions grid.

    Returns:
        Index array of optimal continuous actions.

    """
    indices = jnp.unravel_index(discrete_argmax, shape=discrete_actions_grid_shape)
    return indices_argmax_Q_over_c[indices]


def _lookup_actions_from_indices(
    indices_optimal_discrete_actions: IntND,
    indices_optimal_continuous_actions: IntND,
    discrete_actions_grid_shape: tuple[int, ...],
    continuous_actions_grid_shape: tuple[int, ...],
    state_action_space: StateActionSpace,
) -> dict[str, Array]:
    """Lookup optimal actions from indices.

    Args:
        indices_optimal_discrete_actions: Indices of optimal discrete actions.
        indices_optimal_continuous_actions: Indices of optimal continuous actions.
        discrete_actions_grid_shape: Shape of the discrete actions grid.
        continuous_actions_grid_shape: Shape of the continuous actions grid.
        state_action_space: StateActionSpace instance.

    Returns:
        Dictionary of optimal actions.

    """
    optimal_discrete_actions = _lookup_values_from_indices(
        flat_indices=indices_optimal_discrete_actions,
        grids=state_action_space.discrete_actions,
        grids_shapes=discrete_actions_grid_shape,
    )

    optimal_continuous_actions = _lookup_values_from_indices(
        flat_indices=indices_optimal_continuous_actions,
        grids=state_action_space.continuous_actions,
        grids_shapes=continuous_actions_grid_shape,
    )

    return optimal_discrete_actions | optimal_continuous_actions


def _lookup_actions_from_indices_2(
    indices_optimal_actions: IntND,
    actions_grid_shape: tuple[int, ...],
    state_action_space: StateActionSpace,
) -> dict[str, Array]:
    """Lookup optimal actions from indices.

    Args:
        indices_optimal_actions: Indices of optimal actions.
        actions_grid_shape: Shape of the actions grid.
        state_action_space: StateActionSpace instance.

    Returns:
        Dictionary of optimal actions.

    """
    return _lookup_values_from_indices(
        flat_indices=indices_optimal_actions,
        grids=state_action_space.discrete_actions
        | state_action_space.continuous_actions,
        grids_shapes=actions_grid_shape,
    )


def _lookup_values_from_indices(
    flat_indices: IntND,
    grids: dict[str, Array],
    grids_shapes: tuple[int, ...],
) -> dict[str, Array]:
    """Retrieve values from indices.

    Args:
        flat_indices: General indices. Represents the index of the flattened grid.
        grids: Dictionary of grid values.
        grids_shapes: Shape of the grids. Is used to unravel the index.

    Returns:
        Dictionary of values.

    """
    nd_indices = vmapped_unravel_index(flat_indices, grids_shapes)
    return {
        name: grid[index]
        for (name, grid), index in zip(grids.items(), nd_indices, strict=True)
    }


# vmap jnp.unravel_index over the first axis of the `indices` argument, while holding
# the `shape` argument constant (in_axes = (0, None)).
vmapped_unravel_index = vmap(jnp.unravel_index, in_axes=(0, None))
