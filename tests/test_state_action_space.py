from dataclasses import dataclass
from types import MappingProxyType

import jax.numpy as jnp
import pandas as pd
from numpy.testing import assert_array_equal

from lcm.grids import DiscreteGrid, LinSpacedGrid
from lcm.interfaces import StateActionSpace, StateSpaceInfo
from lcm.regime import Regime
from lcm.state_action_space import (
    create_state_action_space,
    create_state_space_info,
)


def _create_variable_info(
    discrete_states: list[str],
    continuous_states: list[str],
    discrete_actions: list[str],
    continuous_actions: list[str],
) -> pd.DataFrame:
    ordered_vars = (
        discrete_states + discrete_actions + continuous_states + continuous_actions
    )
    info = pd.DataFrame(index=pd.Index(ordered_vars))
    info["is_state"] = info.index.isin(discrete_states + continuous_states)
    info["is_action"] = ~info["is_state"]
    info["is_discrete"] = info.index.isin(discrete_states + discrete_actions)
    info["is_continuous"] = ~info["is_discrete"]
    info["enters_concurrent_valuation"] = True
    info["enters_transition"] = True
    return info


def test_create_state_action_space_solution_discrete_action_continuous_state():
    variable_info = _create_variable_info(
        continuous_states=["wealth"],
        discrete_actions=["work"],
        discrete_states=[],
        continuous_actions=[],
    )
    grids = MappingProxyType(
        {
            "wealth": jnp.array([0.0, 50.0, 100.0]),
            "work": jnp.array([0, 1]),
        }
    )

    space = create_state_action_space(
        variable_info=variable_info,
        grids=grids,
    )

    assert isinstance(space, StateActionSpace)
    assert_array_equal(space.states["wealth"], grids["wealth"])
    assert_array_equal(space.discrete_actions["work"], grids["work"])
    assert space.continuous_actions == {}
    assert space.states_and_discrete_actions_names == ("work", "wealth")


def test_create_state_action_space_solution_continuous_action():
    variable_info = _create_variable_info(
        continuous_states=["wealth"],
        continuous_actions=["consumption"],
        discrete_states=[],
        discrete_actions=[],
    )
    grids = MappingProxyType(
        {
            "wealth": jnp.array([0.0, 50.0, 100.0]),
            "consumption": jnp.array([0.0, 25.0, 50.0]),
        }
    )

    space = create_state_action_space(
        variable_info=variable_info,
        grids=grids,
    )

    assert isinstance(space, StateActionSpace)
    assert_array_equal(space.states["wealth"], grids["wealth"])
    assert space.discrete_actions == {}
    assert_array_equal(space.continuous_actions["consumption"], grids["consumption"])
    assert space.states_and_discrete_actions_names == ("wealth",)


def test_state_action_space_replace_method():
    variable_info = _create_variable_info(
        continuous_states=["wealth"],
        discrete_actions=["work"],
        discrete_states=[],
        continuous_actions=[],
    )
    grids = MappingProxyType(
        {
            "wealth": jnp.array([0.0, 50.0, 100.0]),
            "work": jnp.array([0, 1]),
        }
    )

    space = create_state_action_space(
        variable_info=variable_info,
        grids=grids,
        states={"wealth": jnp.array([10.0, 20.0])},
    )

    new_space = space.replace(
        states=MappingProxyType({"wealth": jnp.array([30.0, 40.0])})
    )

    assert_array_equal(new_space.states["wealth"], jnp.array([30.0, 40.0]))


def test_create_state_space_info():
    @dataclass
    class HealthStatus:
        good: int = 0
        bad: int = 1

    regime = Regime(
        functions={"utility": lambda wealth: wealth},
        states={
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
            "health": DiscreteGrid(HealthStatus),
        },
        transitions={
            "next_wealth": lambda wealth: wealth,
            "next_health": lambda health: health,
        },
        terminal=False,
        active=lambda age: age < 5,
    )

    state_space_info = create_state_space_info(regime)

    assert isinstance(state_space_info, StateSpaceInfo)
    assert set(state_space_info.states_names) == {"wealth", "health"}
    assert "health" in state_space_info.discrete_states
    assert "wealth" in state_space_info.continuous_states
