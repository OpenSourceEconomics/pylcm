from __future__ import annotations

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from lcm.input_processing import process_regime
from lcm.interfaces import StateActionSpace, StateSpaceInfo
from lcm.state_action_space import (
    create_state_action_space,
    create_state_space_info,
)
from tests.test_models.utils import get_regime


def test_create_state_action_space_solution():
    regime = get_regime("iskhakov_et_al_2017_stripped_down")
    internal_regime = process_regime(regime, n_periods=3, enable_jit=True)

    state_action_space = create_state_action_space(
        variable_info=internal_regime.variable_info,
        grids=internal_regime.grids,
        is_last_period=False,
    )

    assert isinstance(state_action_space, StateActionSpace)
    assert jnp.array_equal(
        state_action_space.discrete_actions["retirement"],
        regime.actions["retirement"].to_jax(),
    )
    assert jnp.array_equal(
        state_action_space.states["wealth"], regime.states["wealth"].to_jax()
    )


def test_create_state_action_space_simulation():
    regime = get_regime("iskhakov_et_al_2017")
    internal_regime = process_regime(regime, n_periods=3, enable_jit=True)
    got_space = create_state_action_space(
        variable_info=internal_regime.variable_info,
        grids=internal_regime.grids,
        states={
            "wealth": jnp.array([10.0, 20.0]),
            "lagged_retirement": jnp.array([0, 1]),
        },
    )
    assert_array_equal(got_space.discrete_actions["retirement"], jnp.array([0, 1]))
    assert_array_equal(got_space.states["wealth"], jnp.array([10.0, 20.0]))
    assert_array_equal(got_space.states["lagged_retirement"], jnp.array([0, 1]))


def test_create_state_space_info():
    regime = get_regime("iskhakov_et_al_2017_stripped_down")

    state_space_info = create_state_space_info(
        regime=regime,
        is_last_period=False,
    )

    assert isinstance(state_space_info, StateSpaceInfo)
    assert state_space_info.states_names == ("wealth",)
    assert state_space_info.discrete_states == {}
    assert state_space_info.continuous_states == regime.states


def test_create_state_action_space_replace():
    regime = get_regime("iskhakov_et_al_2017")
    internal_regime = process_regime(regime, n_periods=3, enable_jit=True)
    space = create_state_action_space(
        variable_info=internal_regime.variable_info,
        grids=internal_regime.grids,
        states={
            "wealth": jnp.array([10.0, 20.0]),
            "lagged_retirement": jnp.array([0, 1]),
        },
    )
    new_space = space.replace(
        states={"wealth": jnp.array([10.0, 30.0])},
    )
    assert_array_equal(new_space.states["wealth"], jnp.array([10.0, 30.0]))
