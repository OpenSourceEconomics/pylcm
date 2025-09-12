from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from jax import Array
from numpy.testing import assert_array_equal

from lcm.dispatchers import simulation_spacemap
from lcm.input_processing import process_model
from lcm.interfaces import StateActionSpace, Target
from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a, get_max_Q_over_a
from lcm.max_Q_over_c import get_argmax_and_max_Q_over_c, get_max_Q_over_c
from lcm.max_Qc_over_d import get_argmax_and_max_Qc_over_d, get_max_Qc_over_d
from lcm.next_state import get_next_state_function
from lcm.Q_and_F import get_Q_and_F
from lcm.simulation.simulate import (
    _lookup_actions_from_indices,
    _lookup_optimal_continuous_actions,
    _lookup_values_from_indices,
)
from lcm.state_action_space import create_state_action_space, create_state_space_info
from tests.test_models import (
    get_model,
)

if TYPE_CHECKING:
    from lcm.typing import (
        IntND,
    )


@pytest.fixture
def model_input():
    _model = get_model("iskhakov_et_al_2017_stripped_down", n_periods=1)
    # Modify the model to have a coarser continuous action space for testing
    actions = _model.actions
    actions["consumption"] = actions["consumption"].replace(stop=20)  # type: ignore[attr-defined]
    model = _model.replace(actions=actions)
    model = process_model(model)

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
        "next_state": get_next_state_function(
            model=model, next_states=("wealth",), target=Target.SOLVE
        ),
        "params": params,
    }


def test_max_Q_over_a_equal(model_input):
    """Test max_Q_over_a is equivalent to max_Qc_over_d (max_Q_over_c).

    In this test we check that taking the maximum of Q over all actions
    (max_Q_over_a) is equivalent to taking the maximum of Q over continuous actions
    (max_Q_over_c), conditional on the discrete actions, and then over discrete actions
    (max_Qc_over_d); since these operations should be mathematically equivalent.

    """
    params = model_input["params"]
    state_space_info = model_input["state_space_info"]
    state_action_space = model_input["state_action_space"]
    model = model_input["model"]

    Q_and_F = get_Q_and_F(
        model=model,
        next_state_space_info=state_space_info,
        period=0,
    )
    next_V_arr = jnp.zeros((2, 2))

    # ----------------------------------------------------------------------------------
    # Maximum over all actions directly
    # ----------------------------------------------------------------------------------
    max_Q_over_a = get_max_Q_over_a(
        Q_and_F=Q_and_F,
        actions_names=(
            *state_action_space.discrete_actions,
            *state_action_space.continuous_actions,
        ),
        states_names=(*state_action_space.states,),
    )

    V_arr_a = max_Q_over_a(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        next_V_arr=next_V_arr,
        params=params,
    )

    # ----------------------------------------------------------------------------------
    # Maximum over continuous actions and then over discrete actions
    # ----------------------------------------------------------------------------------
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
    Qc_arr = max_Q_over_c(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        next_V_arr=next_V_arr,
        params=params,
    )
    V_arr_c_d = max_Qc_over_d(Qc_arr, params=params)

    # ----------------------------------------------------------------------------------
    # Assert equality
    # ----------------------------------------------------------------------------------
    assert_array_equal(V_arr_a, V_arr_c_d)


def test_argmax_Q_over_a_equal(model_input):
    """Test argmax_Q_over_a is equivalent to argmax_Qc_over_d (argmax_Q_over_c).

    In this test we check that taking the argmax of Q over all actions
    (argmax_Q_over_a) is equivalent to taking the argmax of Q over continuous actions
    (argmax_Q_over_c), conditional on the discrete actions, and then over discrete
    actions (argmax_Qc_over_d); since these operations should be mathematically
    equivalent.

    """
    params = model_input["params"]
    state_space_info = model_input["state_space_info"]
    state_action_space = model_input["state_action_space"]
    model = model_input["model"]

    Q_and_F = get_Q_and_F(
        model=model,
        next_state_space_info=state_space_info,
        period=0,
    )
    next_V_arr = jnp.zeros((2, 2))

    discrete_actions_grid_shape = tuple(
        len(grid) for grid in state_action_space.discrete_actions.values()
    )
    continuous_actions_grid_shape = tuple(
        len(grid) for grid in state_action_space.continuous_actions.values()
    )
    actions_grid_shape = discrete_actions_grid_shape + continuous_actions_grid_shape

    # ----------------------------------------------------------------------------------
    # Argmax over all actions directly
    # ----------------------------------------------------------------------------------
    _argmax_and_max_Q_over_a_func = get_argmax_and_max_Q_over_a(
        Q_and_F=Q_and_F,
        actions_names=(
            *state_action_space.discrete_actions,
            *state_action_space.continuous_actions,
        ),
    )
    argmax_and_max_Q_over_a = simulation_spacemap(
        _argmax_and_max_Q_over_a_func,
        actions_names=(),
        states_names=tuple(state_action_space.states),
    )
    indices_optimal_actions, V_arr_a = argmax_and_max_Q_over_a(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        next_V_arr=next_V_arr,
        params=params,
    )
    optimal_actions_a = _lookup_actions_from_indices(
        indices_optimal_actions=indices_optimal_actions,
        actions_grid_shape=actions_grid_shape,
        state_action_space=state_action_space,
    )

    # ----------------------------------------------------------------------------------
    # Argmax over continuous actions and then over discrete actions
    # ----------------------------------------------------------------------------------
    _argmax_and_max_Q_over_c_func = get_argmax_and_max_Q_over_c(
        Q_and_F=Q_and_F,
        continuous_actions_names=(*state_action_space.continuous_actions,),
    )
    argmax_and_max_Q_over_c = simulation_spacemap(
        _argmax_and_max_Q_over_c_func,
        actions_names=tuple(state_action_space.discrete_actions),
        states_names=tuple(state_action_space.states),
    )
    argmax_and_max_Qc_over_d = get_argmax_and_max_Qc_over_d(
        variable_info=model.variable_info,
    )

    indices_argmax_Q_over_c, Qc_arr = argmax_and_max_Q_over_c(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        next_V_arr=jnp.zeros((2, 2)),
        params=params,
    )
    indices_optimal_discrete_actions, V_arr_c_d = argmax_and_max_Qc_over_d(
        Qc_arr, params=params
    )
    indices_optimal_continuous_actions = _lookup_optimal_continuous_actions(
        indices_argmax_Q_over_c=indices_argmax_Q_over_c,
        discrete_argmax=indices_optimal_discrete_actions,
        discrete_actions_grid_shape=discrete_actions_grid_shape,
    )
    optimal_actions_c_d = _lookup_actions_from_indices_c_d(
        indices_optimal_discrete_actions=indices_optimal_discrete_actions,
        indices_optimal_continuous_actions=indices_optimal_continuous_actions,
        discrete_actions_grid_shape=discrete_actions_grid_shape,
        continuous_actions_grid_shape=continuous_actions_grid_shape,
        state_action_space=state_action_space,
    )

    # ----------------------------------------------------------------------------------
    # Assert equality
    # ----------------------------------------------------------------------------------
    assert_array_equal(
        optimal_actions_a["retirement"], optimal_actions_c_d["retirement"]
    )
    assert_array_equal(
        optimal_actions_a["consumption"], optimal_actions_c_d["consumption"]
    )
    assert_array_equal(V_arr_a, V_arr_c_d)


def _lookup_actions_from_indices_c_d(
    indices_optimal_discrete_actions: IntND,
    indices_optimal_continuous_actions: IntND,
    discrete_actions_grid_shape: tuple[int, ...],
    continuous_actions_grid_shape: tuple[int, ...],
    state_action_space: StateActionSpace,
) -> dict[str, Array]:
    """Lookup optimal actions from indices of discrete and continuous actions.

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
