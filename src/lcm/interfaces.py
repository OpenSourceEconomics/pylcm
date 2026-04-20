import dataclasses
from collections.abc import Callable
from types import MappingProxyType
from typing import cast

import pandas as pd
from jax import Array

from lcm.grids import DiscreteGrid, Grid, IrregSpacedGrid
from lcm.shocks import _ShockGrid
from lcm.typing import (
    ArgmaxQOverAFunction,
    Bool1D,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FlatRegimeParams,
    FunctionsMapping,
    MaxQOverAFunction,
    NextStateSimulationFunction,
    RegimeParamsTemplate,
    RegimeTransitionFunction,
    TransitionFunctionsMapping,
    VmappedRegimeTransitionFunction,
)
from lcm.utils.containers import first_non_none


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


class SolveSimulateFunctionPair[S, T]:
    """Container for phase-specific function variants.

    Use this to provide different implementations of a function for the solve
    and simulate phases.  For example, naive beta-delta discounting uses
    exponential discounting during backward induction (solve) but present-biased
    discounting for action selection (simulate).

    Variants may have different parameter signatures.  The params template is
    the union of both variants' parameters; each variant receives only the
    kwargs it expects.

    """

    __slots__ = ("simulate", "solve")

    def __init__(self, *, solve: S, simulate: T) -> None:
        self.solve = solve
        self.simulate = simulate


@dataclasses.dataclass(frozen=True, kw_only=True)
class SolveFunctions:
    """Compiled functions for the backward-induction (solve) phase."""

    functions: FunctionsMapping
    """Immutable mapping of function names to internal user functions."""

    constraints: FunctionsMapping
    """Immutable mapping of constraint names to internal user functions."""

    transitions: TransitionFunctionsMapping
    """Immutable mapping of transition names to transition functions."""

    stochastic_transition_names: frozenset[str]
    """Frozenset of stochastic transition function names."""

    compute_regime_transition_probs: RegimeTransitionFunction | None
    """Regime transition probability function for solve, or `None`."""

    max_Q_over_a: MappingProxyType[int, MaxQOverAFunction]
    """Immutable mapping of period to max-Q-over-actions functions."""

    compute_intermediates: MappingProxyType[int, Callable]
    """Immutable mapping of period to intermediate-computation closures.

    Productmap-wrapped and fused with on-device reductions inside a single
    `jax.jit`; invoked only in the error path when `validate_V` detects
    NaN. Each closure returns a flat dict of reductions — scalar
    `{U_nan,E_nan,Q_nan,F_feasible}_overall` entries, per-dimension
    `{...}_by_{name}` vectors, and `regime_probs` as a dict of per-target
    scalar means — so full-shape U/F/E/Q arrays never materialise in
    host-visible memory.
    """


@dataclasses.dataclass(frozen=True, kw_only=True)
class SimulateFunctions:
    """Compiled functions for the forward-simulation phase."""

    functions: FunctionsMapping
    """Immutable mapping of function names to internal user functions."""

    constraints: FunctionsMapping
    """Immutable mapping of constraint names to internal user functions."""

    transitions: TransitionFunctionsMapping
    """Immutable mapping of transition names to transition functions."""

    stochastic_transition_names: frozenset[str]
    """Frozenset of stochastic transition function names."""

    compute_regime_transition_probs: VmappedRegimeTransitionFunction | None
    """Regime transition probability function for simulate, or `None`."""

    argmax_and_max_Q_over_a: MappingProxyType[int, ArgmaxQOverAFunction]
    """Immutable mapping of period to argmax-and-max-Q functions."""

    next_state: NextStateSimulationFunction
    """Compiled function to compute next-period states."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class InternalRegime:
    """Internal representation of a user regime."""

    name: str
    """Regime name (key in the regimes dict)."""

    terminal: bool
    """Whether this is a terminal regime."""

    grids: MappingProxyType[str, Grid]
    """Immutable mapping of variable names to grid objects."""

    variable_info: pd.DataFrame
    """DataFrame with variable metadata (is_state, is_action, etc.)."""

    active_periods: tuple[int, ...]
    """Period indices during which this regime is active."""

    regime_params_template: RegimeParamsTemplate
    """Template for the parameter structure expected by this regime."""

    solve_functions: SolveFunctions
    """Compiled functions for the backward-induction (solve) phase."""

    simulate_functions: SimulateFunctions
    """Compiled functions for the forward-simulation phase."""

    _base_state_action_space: StateActionSpace = dataclasses.field(repr=False)
    """Base state-action space before runtime grid substitution."""

    resolved_fixed_params: FlatRegimeParams = MappingProxyType({})
    """Flat resolved fixed params for this regime, used by to_dataframe targets."""

    partitions: MappingProxyType[str, DiscreteGrid] = MappingProxyType({})
    """Immutable mapping of partition-dimension names to their discrete grids.

    Partitions are states declared via `state_transitions[name] = None`:
    because their value never changes along its own axis in the Bellman
    equation, they are lifted out of `grids` / `variable_info` at
    regime-processing time. Solve and simulate iterate over the product
    of partition grids once per point, compile-once / run-N-times, instead
    of vectorising over a partition axis. Empty when the user declared no
    `None` transitions (default — no behavioural change).
    """

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
        replacements: dict[str, ContinuousState | DiscreteState] = {}
        for state_name, spec in self.grids.items():
            if state_name not in self._base_state_action_space.states:
                continue
            if isinstance(spec, IrregSpacedGrid) and spec.pass_points_at_runtime:
                points_key = f"{state_name}__points"
                if points_key not in all_params:
                    continue
                replacements[state_name] = cast(
                    "ContinuousState", all_params[points_key]
                )
            elif isinstance(spec, _ShockGrid) and spec.params_to_pass_at_runtime:
                all_present = all(
                    f"{state_name}__{p}" in all_params
                    for p in spec.params_to_pass_at_runtime
                )
                if not all_present:
                    continue
                shock_kw: dict[str, float] = dict(spec.params)
                for p in spec.params_to_pass_at_runtime:
                    shock_kw[p] = cast("float", all_params[f"{state_name}__{p}"])
                replacements[state_name] = spec.compute_gridpoints(**shock_kw)

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

    subject_ids: Array
    """Global subject-id array aligned with `in_regime` / `V_arr` / states / actions.

    Threaded explicitly (rather than recomputed as `jnp.arange(n_subjects)`) so
    that downstream concatenation across partition-dispatch groups preserves
    the caller's subject ordering.
    """
