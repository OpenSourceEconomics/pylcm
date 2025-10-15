from __future__ import annotations

import dataclasses
from enum import Enum
from typing import TYPE_CHECKING

from lcm.utils import first_non_none

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    from jax import Array

    from lcm.grids import ContinuousGrid, DiscreteGrid, Grid
    from lcm.typing import (
        ContinuousAction,
        ContinuousState,
        DiscreteAction,
        DiscreteState,
        Float1D,
        Int1D,
        InternalUserFunction,
        ParamsDict,
    )


@dataclasses.dataclass(frozen=True)
class StateActionSpace:
    """The state-action space.

    When used for the model solution:
    ---------------------------------

    The state-action space becomes the full Cartesian product of the state variables and
    the action variables.

    When used for the simulation:
    ----------------------------

    The state-action space becomes the product of state-combinations with the full
    Cartesian product of the action variables.

    In both cases, infeasible state-action combinations will be masked.

    Note:
    -----
    We store discrete and continuous actions separately since these are handled during
    different stages of the solution and simulation processes.

    Attributes:
        states: Dictionary containing the values of the state variables.
        discrete_actions: Dictionary containing the values of the discrete action
            variables.
        continuous_actions: Dictionary containing the values of the continuous action
            variables.
        states_and_discrete_actions_names: Tuple with names of states and discrete
            action variables in the order they appear in the variable info table.

    """

    states: dict[str, ContinuousState | DiscreteState]
    discrete_actions: dict[str, DiscreteAction]
    continuous_actions: dict[str, ContinuousAction]
    states_and_discrete_actions_names: tuple[str, ...]

    def replace(
        self,
        states: dict[str, ContinuousState | DiscreteState] | None = None,
        discrete_actions: dict[str, DiscreteAction] | None = None,
        continuous_actions: dict[str, ContinuousAction] | None = None,
    ) -> StateActionSpace:
        """Replace the states or actions in the state-action space.

        Args:
            states: Dictionary with new states. If None, the existing states are used.
            discrete_actions: Dictionary with new discrete actions. If None, the
                existing discrete actions are used.
            continuous_actions: Dictionary with new continuous actions. If None, the
                existing continuous actions are used.

        Returns:
            New state-action space with the replaced states or actions.

        """
        states = first_non_none(states, self.states)
        discrete_actions = first_non_none(discrete_actions, self.discrete_actions)
        continuous_actions = first_non_none(continuous_actions, self.continuous_actions)
        return dataclasses.replace(
            self,
            states=states,
            discrete_actions=discrete_actions,
            continuous_actions=continuous_actions,
        )


@dataclasses.dataclass(frozen=True)
class StateSpaceInfo:
    """Information to work with the output of a function evaluated on a state space.

    An example is the value function array, which is the output of the value function
    evaluated on the state space.

    Attributes:
        var_names: Tuple with names of state variables.
        discrete_vars: Dictionary with grids of discrete state variables.
        continuous_vars: Dictionary with grids of continuous state variables.

    """

    states_names: tuple[str, ...]
    discrete_states: Mapping[str, DiscreteGrid]
    continuous_states: Mapping[str, ContinuousGrid]


class ShockType(Enum):
    """Type of shocks."""

    EXTREME_VALUE = "extreme_value"
    NONE = None


@dataclasses.dataclass(frozen=True)
class InternalModel:
    """Internal representation of a user model.

    Attributes:
        grids: Dictionary that maps names of model variables to grids of feasible values
            for that variable.
        gridspecs: Dictionary that maps names of model variables to specifications from
            which grids of feasible values can be built.
        variable_info: A table with information about all variables in the model. The
            index contains the name of a model variable. The columns are booleans that
            are True if the variable has the corresponding property. The columns are:
            is_state, is_action, is_continuous, is_discrete.
        utility: The utility function of the model.
        transitions: Dictionary that maps transition functions to state names.
        constraints: Dictionary that maps constraint names to constraint functions.
        functions: Dictionary that maps names of functions to functions. The functions
            differ from the user functions in that they take `params` as a keyword
            argument. Two cases:
            - If the original function depended on model parameters, those are
              automatically extracted from `params` and passed to the original
              function.
            - Otherwise, the `params` argument is simply ignored.
        params: Dict of model parameters.
        n_periods: Number of periods.
        random_utility_shocks: Type of random utility shocks.

    """

    grids: dict[str, Float1D | Int1D]
    gridspecs: dict[str, Grid]
    variable_info: pd.DataFrame
    utility: InternalUserFunction
    constraints: dict[str, InternalUserFunction]
    transitions: dict[str, InternalUserFunction]
    functions: dict[str, InternalUserFunction]
    params: ParamsDict
    n_periods: int
    # Not properly processed yet
    random_utility_shocks: ShockType


@dataclasses.dataclass(frozen=True)
class InternalSimulationPeriodResults:
    """The results of a simulation for one period."""

    value: Array
    actions: dict[str, Array]
    states: dict[str, Array]


class Target(Enum):
    """Target of the function."""

    SOLVE = "solve"
    SIMULATE = "simulate"
