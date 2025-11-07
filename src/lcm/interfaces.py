from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from dags.tree import flatten_to_qnames

from lcm.utils import first_non_none

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    from jax import Array

    from lcm.grids import ContinuousGrid, DiscreteGrid, Grid
    from lcm.typing import (
        ArgmaxQOverAFunction,
        ContinuousAction,
        ContinuousState,
        DiscreteAction,
        DiscreteState,
        Float1D,
        Int1D,
        InternalUserFunction,
        MaxQOverAFunction,
        NextStateSimulationFunction,
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

    @property
    def states_names(self) -> tuple[str, ...]:
        """Tuple with names of all state variables."""
        return tuple(self.states)

    @property
    def actions_names(self) -> tuple[str, ...]:
        """Tuple with names of all action variables."""
        return tuple(self.discrete_actions) + tuple(self.continuous_actions)

    @property
    def actions(self) -> dict[str, DiscreteAction | ContinuousAction]:
        """Dictionary with all action variables."""
        return self.discrete_actions | self.continuous_actions

    @property
    def actions_grid_shapes(self) -> tuple[int, ...]:
        """Dictionary with all action variables."""
        return tuple(len(grid) for grid in self.actions.values())

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


class PeriodVariantContainer[T]:
    """Container for objects that vary by period relative to the terminal period.

    Attributes:
        terminal: Object for the terminal period.
        non_terminal: Object for all non-terminal periods, except the one before the
            terminal period if provided.
        before_terminal: Object for the period just before the terminal period. If None,
            defaults to the non-terminal object.

    """

    __slots__ = ("before_terminal", "non_terminal", "terminal")

    def __init__(
        self, terminal: T, non_terminal: T, before_terminal: T | None = None
    ) -> None:
        self.terminal = terminal
        self.non_terminal = non_terminal
        self.before_terminal = (
            non_terminal if before_terminal is None else before_terminal
        )

    def __call__(self, period: int, *, n_periods: int) -> T:
        """Return object given period relative to the terminal period."""
        if period == n_periods - 1:
            return self.terminal
        if period == n_periods - 2:
            return self.before_terminal
        return self.non_terminal


@dataclasses.dataclass(frozen=True)
class InternalRegime:
    """Internal representation of a user regime.

    MUST BE UPDATED.

    """

    name: str
    grids: dict[str, Float1D | Int1D]
    gridspecs: dict[str, Grid]
    variable_info: pd.DataFrame
    utility: InternalUserFunction
    constraints: dict[str, InternalUserFunction]
    transitions: dict[str, InternalUserFunction]
    functions: dict[str, InternalUserFunction]
    regime_transition_probs: Callable[..., dict[str, float]]
    internal_functions: InternalFunctions
    params_template: ParamsDict
    state_action_spaces: PeriodVariantContainer[StateActionSpace]
    state_space_infos: PeriodVariantContainer[StateSpaceInfo]
    max_Q_over_a_functions: PeriodVariantContainer[MaxQOverAFunction]
    argmax_and_max_Q_over_a_functions: PeriodVariantContainer[ArgmaxQOverAFunction]
    next_state_simulation_function: NextStateSimulationFunction
    # Not properly processed yet
    random_utility_shocks: ShockType

    def get_all_functions(self) -> dict[str, InternalUserFunction]:
        """Get all regime functions including utility, constraints, and transitions.

        Returns:
            Dictionary that maps names of all regime functions to the functions.

        """
        return (
            self.functions
            | {"utility": self.utility}
            | self.constraints
            | self.transitions
            | {"regime_transition_probs": self.regime_transition_probs}
        )


@dataclasses.dataclass(frozen=True)
class SimulationResults:
    """The results of a simulation for one period and one regime."""

    V_arr: Array
    actions: dict[str, Array]
    states: dict[str, Array]
    subject_ids: Array


class Target(Enum):
    """Target of the function."""

    SOLVE = "solve"
    SIMULATE = "simulate"


@dataclass(frozen=True)
class InternalFunctions:
    """All functions that are used in the regime."""

    functions: dict[str, InternalUserFunction]
    utility: InternalUserFunction
    constraints: dict[str, InternalUserFunction]
    transitions: dict[str, InternalUserFunction]
    regime_transition_probs: Callable[..., dict[str, float]]

    def get_all_functions(self) -> dict[str, InternalUserFunction]:
        """Get all regime functions including utility, constraints, and transitions.

        Returns:
            Dictionary that maps names of all regime functions to the functions.

        """
        return flatten_to_qnames(
            self.functions
            | {"utility": self.utility}
            | self.constraints
            | self.transitions
            | {"regime_transition_probs": self.regime_transition_probs}
        )
