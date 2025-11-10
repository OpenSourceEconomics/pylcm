from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from jax import Array
from numpy.testing import assert_array_equal

from lcm.dispatchers import simulation_spacemap, vmap_1d
from lcm.input_processing import process_regimes
from lcm.interfaces import StateActionSpace, Target
from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a, get_max_Q_over_a
from lcm.max_Q_over_c import get_argmax_and_max_Q_over_c, get_max_Q_over_c
from lcm.max_Qc_over_d import get_argmax_and_max_Qc_over_d, get_max_Qc_over_d
from lcm.next_state import get_next_state_function
from lcm.Q_and_F import get_Q_and_F
from lcm.simulation.simulate import (
    _lookup_values_from_indices,
)
from lcm.state_action_space import create_state_action_space, create_state_space_info
from tests.test_models.utils import get_regime

if TYPE_CHECKING:
    from lcm.typing import (
        IntND,
    )


@pytest.fixture
def regime_input():
    regime = get_regime("iskhakov_et_al_2017_stripped_down")
    # Modify the regime to have a coarser continuous action space for testing
    actions = regime.actions
    actions["consumption"] = actions["consumption"].replace(stop=20)  # type: ignore[attr-defined]
    regime = regime.replace(actions=actions)
    internal_regime = process_regimes([regime], n_periods=3, enable_jit=True)[
        "iskhakov_et_al_2017_stripped_down"
    ]

    state_space_info = create_state_space_info(
        regime=regime,
        is_last_period=False,
    )
    state_action_space = create_state_action_space(
        variable_info=internal_regime.variable_info,
        grids=internal_regime.grids,
        is_last_period=False,
    )
    params = {
        "iskhakov_et_al_2017_stripped_down": {
            "beta": 1.0,
            "utility": {"disutility_of_work": 1.0},
            "next_wealth": {
                "interest_rate": 0.05,
            },
        }
    }
    return {
        "regime": regime,
        "internal_regime": internal_regime,
        "state_action_space": state_action_space,
        "state_space_info": state_space_info,
        "next_state": get_next_state_function(
            transitions=internal_regime.internal_functions.transitions[
                "iskhakov_et_al_2017_stripped_down"
            ],
            functions=internal_regime.internal_functions.functions,
            grids=internal_regime.grids,
            target=Target.SOLVE,
        ),
        "params": params,
        "grids": internal_regime.grids,
    }


def test_max_Q_over_a_equal(regime_input):
    """Test max_Q_over_a is equivalent to max_Qc_over_d (max_Q_over_c).

    In this test we check that taking the maximum of Q over all actions
    (max_Q_over_a) is equivalent to taking the maximum of Q over continuous actions
    (max_Q_over_c), conditional on the discrete actions, and then over discrete actions
    (max_Qc_over_d); since these operations should be mathematically equivalent.

    """
    params = regime_input["params"]
    grids = regime_input["grids"]
    state_space_infos = regime_input["state_space_info"]
    state_action_space = regime_input["state_action_space"]
    regime = regime_input["regime"]
    internal_regime = regime_input["internal_regime"]

    Q_and_F = get_Q_and_F(
        regime=regime,
        internal_functions=internal_regime.internal_functions,
        next_state_space_infos=state_space_infos,
        is_last_period=True,
        grids=grids,
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
        period=0,
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
        random_utility_shock_type=internal_regime.random_utility_shocks,
        variable_info=internal_regime.variable_info,
        is_last_period=False,
    )
    Qc_arr = max_Q_over_c(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        period=0,
        next_V_arr=next_V_arr,
        params=params,
    )
    V_arr_c_d = max_Qc_over_d(Qc_arr, params=params)

    # ----------------------------------------------------------------------------------
    # Assert equality
    # ----------------------------------------------------------------------------------
    assert_array_equal(V_arr_a, V_arr_c_d)


def test_argmax_Q_over_a_equal(regime_input):
    """Test argmax_Q_over_a is equivalent to argmax_Qc_over_d (argmax_Q_over_c).

    In this test we check that taking the argmax of Q over all actions
    (argmax_Q_over_a) is equivalent to taking the argmax of Q over continuous actions
    (argmax_Q_over_c), conditional on the discrete actions, and then over discrete
    actions (argmax_Qc_over_d); since these operations should be mathematically
    equivalent.

    """
    params = regime_input["params"]
    grids = regime_input["grids"]
    state_space_infos = regime_input["state_space_info"]
    state_action_space = regime_input["state_action_space"]
    regime = regime_input["regime"]
    internal_regime = regime_input["internal_regime"]

    Q_and_F = get_Q_and_F(
        regime=regime,
        internal_functions=internal_regime.internal_functions,
        next_state_space_infos=state_space_infos,
        is_last_period=True,
        grids=grids,
    )
    next_V_arr = jnp.zeros((2, 2))

    discrete_actions_grid_shape = tuple(
        len(grid) for grid in state_action_space.discrete_actions.values()
    )

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
        period=0,
        next_V_arr=next_V_arr,
        params=params,
    )
    optimal_actions_a = _lookup_values_from_indices(
        flat_indices=indices_optimal_actions,
        grids=state_action_space.actions,
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
        variable_info=internal_regime.variable_info,
    )

    indices_argmax_Q_over_c, Qc_arr = argmax_and_max_Q_over_c(
        **state_action_space.states,
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        next_V_arr=jnp.zeros((2, 2)),
        period=0,
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
    )

    optimal_continuous_actions = _lookup_values_from_indices(
        flat_indices=indices_optimal_continuous_actions,
        grids=state_action_space.continuous_actions,
    )

    return optimal_discrete_actions | optimal_continuous_actions


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
