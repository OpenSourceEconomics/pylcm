import dataclasses
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import pandas as pd
from jax import Array

from lcm.grids import ContinuousGrid, DiscreteGrid, Grid, IrregSpacedGrid
from lcm.shocks import _ShockGrid
from lcm.typing import (
    ArgmaxQOverAFunction,
    Bool1D,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FlatRegimeParams,
    InternalUserFunction,
    MaxQOverAFunction,
    NextStateSimulationFunction,
    RegimeParamsTemplate,
    RegimeTransitionFunction,
    TransitionFunctionsMapping,
    VmappedRegimeTransitionFunction,
)
from lcm.utils import first_non_none, flatten_regime_namespace


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

    """

    states: MappingProxyType[str, ContinuousState | DiscreteState]
    """Immutable mapping of state variable names to their values."""

    discrete_actions: MappingProxyType[str, DiscreteAction]
    """Immutable mapping of discrete action variable names to their values."""

    continuous_actions: MappingProxyType[str, ContinuousAction]
    """Immutable mapping of continuous action variable names to their values."""

    state_and_discrete_action_names: tuple[str, ...]
    """Names of states and discrete actions in variable info table order."""

    @property
    def state_names(self) -> tuple[str, ...]:
        """Tuple with names of all state variables."""
        return tuple(self.states)

    @property
    def action_names(self) -> tuple[str, ...]:
        """Tuple with names of all action variables."""
        return tuple(self.discrete_actions) + tuple(self.continuous_actions)

    @property
    def actions(self) -> MappingProxyType[str, DiscreteAction | ContinuousAction]:
        """Read-only mapping with all action variables."""
        return MappingProxyType(
            dict(self.discrete_actions) | dict(self.continuous_actions)
        )

    @property
    def actions_grid_shapes(self) -> tuple[int, ...]:
        """Tuple of action grid sizes."""
        return tuple(len(grid) for grid in self.actions.values())

    def replace(
        self,
        states: MappingProxyType[str, ContinuousState | DiscreteState] | None = None,
        discrete_actions: MappingProxyType[str, DiscreteAction] | None = None,
        continuous_actions: MappingProxyType[str, ContinuousAction] | None = None,
    ) -> StateActionSpace:
        """Replace the states or actions in the state-action space.

        Args:
            states: Read-only mapping with new states. If None, the existing states are
                used.
            discrete_actions: Read-only mapping with new discrete actions. If None, the
                existing discrete actions are used.
            continuous_actions: Read-only mapping with new continuous actions. If None,
                the existing continuous actions are used.

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class StateSpaceInfo:
    """Information to work with the output of a function evaluated on a state space.

    An example is the value function array, which is the output of the value function
    evaluated on the state space.

    """

    state_names: tuple[str, ...]
    """Tuple of state variable names."""

    discrete_states: MappingProxyType[str, DiscreteGrid | _ShockGrid]
    """Immutable mapping of discrete state names to their grids."""

    continuous_states: MappingProxyType[str, ContinuousGrid]
    """Immutable mapping of continuous state names to their grids."""


class ShockType(Enum):
    """Type of shocks."""

    EXTREME_VALUE = "extreme_value"
    NONE = None


class PhaseVariantContainer[S, T]:
    """Container for objects that vary whether we are in the solve or simulate phase."""

    __slots__ = ("simulate", "solve")

    def __init__(self, *, solve: S, simulate: T) -> None:
        self.solve = solve
        self.simulate = simulate


@dataclasses.dataclass(frozen=True, kw_only=True)
class InternalRegime:
    """Internal representation of a user regime."""

    name: str
    """Regime name (key in the regimes dict)."""

    terminal: bool
    """Whether this is a terminal regime."""

    grids: MappingProxyType[str, Array]
    """Immutable mapping of variable names to materialized JAX grid arrays."""

    gridspecs: MappingProxyType[str, Grid]
    """Immutable mapping of variable names to grid specification objects."""

    variable_info: pd.DataFrame
    """DataFrame with variable metadata (is_state, is_action, etc.)."""

    constraints: MappingProxyType[str, InternalUserFunction]
    """Immutable mapping of constraint names to compiled constraint functions."""

    transitions: TransitionFunctionsMapping
    """Immutable mapping of state transition names to compiled transition functions."""

    functions: MappingProxyType[str, InternalUserFunction]
    """Immutable mapping of function names to compiled user functions."""

    active_periods: tuple[int, ...]
    """Period indices during which this regime is active."""

    regime_transition_probs: (
        PhaseVariantContainer[RegimeTransitionFunction, VmappedRegimeTransitionFunction]
        | None
    )
    """Regime transition probability functions for solve and simulate, or `None`."""

    internal_functions: InternalFunctions
    """All compiled functions for this regime.

    Includes user functions, constraints, and transitions.
    """

    regime_params_template: RegimeParamsTemplate
    """Template for the parameter structure expected by this regime."""

    state_space_info: StateSpaceInfo
    """Metadata for working with function outputs on the state space."""

    max_Q_over_a_functions: MappingProxyType[int, MaxQOverAFunction]
    """Immutable mapping of period to max-Q-over-actions functions for solving."""

    argmax_and_max_Q_over_a_functions: MappingProxyType[int, ArgmaxQOverAFunction]
    """Immutable mapping of period to argmax-and-max-Q functions for simulation."""

    next_state_simulation_function: NextStateSimulationFunction
    """Compiled function to compute next-period states during simulation."""

    # Not properly processed yet
    random_utility_shocks: ShockType
    """Type of random utility shocks (extreme value or none)."""

    _base_state_action_space: StateActionSpace = dataclasses.field(repr=False)
    """Base state-action space before runtime grid substitution."""

    # Resolved fixed params (flat) for this regime, used by to_dataframe targets
    resolved_fixed_params: FlatRegimeParams = MappingProxyType({})
    """Flat resolved fixed params for this regime, used by to_dataframe targets."""

    def state_action_space(self, regime_params: FlatRegimeParams) -> StateActionSpace:
        """Return the state-action space with runtime state grids filled in.

        For IrregSpacedGrid with runtime-supplied points, the grid points come from
        params as `{state_name}__points`. For _ShockGrid with runtime-supplied params,
        the grid points are computed from shock params in the params dict or
        resolved_fixed_params.

        Args:
            regime_params: Flat regime parameters supplied at runtime.

        Returns:
            Completed state-action space.

        """
        all_params = {**self.resolved_fixed_params, **regime_params}
        replacements: dict[str, object] = {}
        for state_name, spec in self.gridspecs.items():
            if state_name not in self._base_state_action_space.states:
                continue
            if isinstance(spec, IrregSpacedGrid) and spec.pass_points_at_runtime:
                points_key = f"{state_name}__points"
                if points_key not in all_params:
                    continue
                replacements[state_name] = all_params[points_key]
            elif isinstance(spec, _ShockGrid) and spec.params_to_pass_at_runtime:
                all_present = all(
                    f"{state_name}__{p}" in all_params
                    for p in spec.params_to_pass_at_runtime
                )
                if not all_present:
                    continue
                shock_kw: dict[str, bool | float | Array] = dict(spec.params)
                for p in spec.params_to_pass_at_runtime:
                    shock_kw[p] = all_params[f"{state_name}__{p}"]
                replacements[state_name] = spec.compute_gridpoints(
                    spec.n_points,
                    **shock_kw,  # ty: ignore[invalid-argument-type]
                )

        if not replacements:
            return self._base_state_action_space

        new_states = dict(self._base_state_action_space.states) | replacements
        return self._base_state_action_space.replace(
            states=MappingProxyType(new_states)
        )


@dataclasses.dataclass(frozen=True)
class PeriodRegimeSimulationData:
    """Raw simulation data for one period in one regime."""

    V_arr: Array
    """Value function array for all subjects at this period."""

    actions: MappingProxyType[str, Array]
    """Immutable mapping of action names to optimal action arrays for all subjects."""

    states: MappingProxyType[str, Array]
    """Immutable mapping of state names to state value arrays for all subjects."""

    in_regime: Bool1D
    """Boolean mask indicating which subjects are in this regime at this period."""


class Target(Enum):
    """Target of the function."""

    SOLVE = "solve"
    SIMULATE = "simulate"


@dataclass(frozen=True)
class InternalFunctions:
    """All functions that are used in the regime."""

    functions: MappingProxyType[str, InternalUserFunction]
    """Immutable mapping of function names to internal user functions."""

    constraints: MappingProxyType[str, InternalUserFunction]
    """Immutable mapping of constraint names to internal user functions."""

    transitions: TransitionFunctionsMapping
    """Immutable mapping of transition names to transition functions."""

    regime_transition_probs: (
        PhaseVariantContainer[RegimeTransitionFunction, VmappedRegimeTransitionFunction]
        | None
    )
    """Regime transition probability functions, or None for terminal regimes."""

    def get_all_functions(self) -> MappingProxyType[str, InternalUserFunction]:
        """Get all regime functions including utility, constraints, and transitions.

        Returns:
            Read-only mapping of all regime functions to the functions.

        """
        functions_pool = {
            **self.functions,
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
        return MappingProxyType(flatten_regime_namespace(functions_pool))
