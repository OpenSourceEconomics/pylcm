from types import MappingProxyType

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from _lcm.engine import StateActionSpace, VariableInfo, Variables
from _lcm.grids import DiscreteGrid, IrregSpacedGrid, LinSpacedGrid, categorical
from _lcm.regime_building.V import VInterpolationInfo, create_v_interpolation_info
from _lcm.state_action_space import create_state_action_space
from lcm.regime import Regime as UserRegime
from lcm.typing import ScalarInt


def _create_variables(
    discrete_states: list[str],
    continuous_states: list[str],
    discrete_actions: list[str],
    continuous_actions: list[str],
) -> Variables:
    info: dict[str, VariableInfo] = {}
    # Order matches Variables.from_regime ordering: discrete states, continuous
    # states, then actions (discrete then continuous within actions in original
    # declaration order).
    for name in discrete_states:
        info[name] = VariableInfo(kind="state", topology="discrete", is_process=False)
    for name in continuous_states:
        info[name] = VariableInfo(kind="state", topology="continuous", is_process=False)
    for name in discrete_actions:
        info[name] = VariableInfo(kind="action", topology="discrete", is_process=False)
    for name in continuous_actions:
        info[name] = VariableInfo(
            kind="action", topology="continuous", is_process=False
        )
    return Variables(info=MappingProxyType(info))


def test_create_state_action_space_solution_discrete_action_continuous_state():
    @categorical(ordered=False)
    class WorkChoice:
        no_work: ScalarInt
        work: ScalarInt

    variables = _create_variables(
        continuous_states=["wealth"],
        discrete_actions=["work"],
        discrete_states=[],
        continuous_actions=[],
    )
    grids = MappingProxyType(
        {
            "wealth": IrregSpacedGrid(points=[0.0, 50.0, 100.0]),
            "work": DiscreteGrid(WorkChoice),
        }
    )

    space = create_state_action_space(
        variables=variables,
        grids=grids,
    )

    assert isinstance(space, StateActionSpace)
    assert_array_equal(space.states["wealth"], grids["wealth"].to_jax())
    assert_array_equal(space.discrete_actions["work"], grids["work"].to_jax())
    assert space.continuous_actions == {}
    assert space.state_and_discrete_action_names == ("wealth", "work")


def test_create_state_action_space_solution_continuous_action():
    variables = _create_variables(
        continuous_states=["wealth"],
        continuous_actions=["consumption"],
        discrete_states=[],
        discrete_actions=[],
    )
    grids = MappingProxyType(
        {
            "wealth": IrregSpacedGrid(points=[0.0, 50.0, 100.0]),
            "consumption": IrregSpacedGrid(points=[0.0, 25.0, 50.0]),
        }
    )

    space = create_state_action_space(
        variables=variables,
        grids=grids,
    )

    assert isinstance(space, StateActionSpace)
    assert_array_equal(space.states["wealth"], grids["wealth"].to_jax())
    assert space.discrete_actions == {}
    assert_array_equal(
        space.continuous_actions["consumption"], grids["consumption"].to_jax()
    )
    assert space.state_and_discrete_action_names == ("wealth",)


def test_state_action_space_replace_method():
    @categorical(ordered=False)
    class WorkChoice:
        no_work: ScalarInt
        work: ScalarInt

    variables = _create_variables(
        continuous_states=["wealth"],
        discrete_actions=["work"],
        discrete_states=[],
        continuous_actions=[],
    )
    grids = MappingProxyType(
        {
            "wealth": IrregSpacedGrid(points=[0.0, 50.0, 100.0]),
            "work": DiscreteGrid(WorkChoice),
        }
    )

    space = create_state_action_space(
        variables=variables,
        grids=grids,
        states={"wealth": jnp.array([10.0, 20.0])},
    )

    new_space = space.replace(
        states=MappingProxyType({"wealth": jnp.array([30.0, 40.0])})
    )

    assert_array_equal(new_space.states["wealth"], jnp.array([30.0, 40.0]))


def test_create_v_interpolation_info():
    @categorical(ordered=False)
    class Health:
        good: ScalarInt
        bad: ScalarInt

    regime = UserRegime(
        transition=lambda: 0,  # non-terminal
        functions={"utility": lambda wealth: wealth},
        states={
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
            "health": DiscreteGrid(Health),
        },
        state_transitions={
            "wealth": lambda wealth: wealth,
            "health": lambda health: health,
        },
        active=lambda age: age < 5,
    )

    v_interpolation_info = create_v_interpolation_info(regime)

    assert isinstance(v_interpolation_info, VInterpolationInfo)
    assert set(v_interpolation_info.state_names) == {"wealth", "health"}
    assert "health" in v_interpolation_info.discrete_states
    assert "wealth" in v_interpolation_info.continuous_states
