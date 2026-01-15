from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from lcm.utils import first_non_none, flatten_regime_namespace

if TYPE_CHECKING:
    import pandas as pd
    from jax import Array

    from lcm.grids import ContinuousGrid, DiscreteGrid, Grid, ShockGrid
    from lcm.typing import (
        ArgmaxQOverAFunction,
        Bool1D,
        ContinuousAction,
        ContinuousState,
        DiscreteAction,
        DiscreteState,
        InternalUserFunction,
        MaxQOverAFunction,
        NextStateSimulationFunction,
        ParamsDict,
        RegimeTransitionFunction,
        TransitionFunctionsDict,
        VmappedRegimeTransitionFunction,
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
    discrete_states: dict[str, DiscreteGrid | ShockGrid]
    continuous_states: dict[str, ContinuousGrid]


class ShockType(Enum):
    """Type of shocks."""

    EXTREME_VALUE = "extreme_value"
    NONE = None


class PhaseVariantContainer[S, T]:
    """Container for objects that vary whether we are in the solve or simulate phase.

    Attributes:
        solve: Object for the solve phase.
        simulate: Object for the simulate phase.

    """

    __slots__ = ("simulate", "solve")

    def __init__(self, solve: S, simulate: T) -> None:
        self.solve = solve
        self.simulate = simulate


@dataclasses.dataclass(frozen=False)
class InternalRegime:
    """Internal representation of a user regime.

    MUST BE UPDATED.

    """

    name: str
    terminal: bool
    grids: dict[str, Array]
    gridspecs: dict[str, Grid]
    variable_info: pd.DataFrame
    transition_info: pd.DataFrame
    utility: InternalUserFunction
    constraints: dict[str, InternalUserFunction]
    transitions: TransitionFunctionsDict
    functions: dict[str, InternalUserFunction]
    active_periods: list[int]
    regime_transition_probs: (
        PhaseVariantContainer[RegimeTransitionFunction, VmappedRegimeTransitionFunction]
        | None
    )
    internal_functions: InternalFunctions
    params_template: ParamsDict
    state_action_space: StateActionSpace
    state_space_info: StateSpaceInfo
    max_Q_over_a_functions: dict[int, MaxQOverAFunction]
    argmax_and_max_Q_over_a_functions: dict[int, ArgmaxQOverAFunction]
    next_state_simulation_function: NextStateSimulationFunction
    # Not properly processed yet
    random_utility_shocks: ShockType


@dataclasses.dataclass(frozen=True)
class PeriodRegimeSimulationData:
    """Raw simulation data for one period in one regime.

    Attributes:
        V_arr: Value function array of a regime for all subjects at this period.
        actions: Dict mapping action names to optimal action arrays for all subjects.
        states: Dict mapping state names to state value arrays for all subjects.
        in_regime: Boolean mask indicating which subjects are in this regime at this
            period. True means the subject is in this regime; False means they are in
            a different regime (and the corresponding values in V_arr, actions, and
            states should be ignored for that subject).

    """

    V_arr: Array
    actions: dict[str, Array]
    states: dict[str, Array]
    in_regime: Bool1D


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
    transitions: TransitionFunctionsDict
    regime_transition_probs: (
        PhaseVariantContainer[RegimeTransitionFunction, VmappedRegimeTransitionFunction]
        | None
    )

    def get_all_functions(self) -> dict[str, InternalUserFunction]:
        """Get all regime functions including utility, constraints, and transitions.

        Returns:
            Dictionary that maps names of all regime functions to the functions.

        """
        functions_pool = self.functions | {
            "utility": self.utility,
            **self.constraints,
            **self.transitions,
        }
        if self.regime_transition_probs is not None:
            functions_pool["regime_transition_probs_solve"] = (
                self.regime_transition_probs.solve
            )
            functions_pool["regime_transition_probs_simulate"] = (
                self.regime_transition_probs.simulate
            )
        return flatten_regime_namespace(functions_pool)
