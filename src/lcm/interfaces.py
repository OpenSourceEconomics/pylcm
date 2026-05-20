import dataclasses
from collections.abc import Callable
from math import prod as math_prod
from types import MappingProxyType
from typing import cast

import jax
from jax import Array

from lcm.exceptions import PyLCMError
from lcm.grids import Grid, IrregSpacedGrid
from lcm.shocks import _ShockGrid
from lcm.typing import (
    ActionName,
    ArgmaxQOverAFunction,
    Bool1D,
    ConstraintFunctionsMapping,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    EconFunctionsMapping,
    FlatRegimeParams,
    Float1D,
    FloatND,
    IntND,
    MaxQOverAFunction,
    NextStateSimulationFunction,
    RegimeName,
    RegimeParamsTemplate,
    RegimeTransitionFunction,
    ScalarFloat,
    ScalarInt,
    StateName,
    StateOrActionName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
    VmappedRegimeTransitionFunction,
)
from lcm.utils.containers import first_non_none
from lcm.variables import Variables


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

    states: MappingProxyType[StateName, ContinuousState | DiscreteState]
    """Immutable mapping of state variable names to their values."""

    discrete_actions: MappingProxyType[ActionName, DiscreteAction]
    """Immutable mapping of discrete action variable names to their values."""

    continuous_actions: MappingProxyType[ActionName, ContinuousAction]
    """Immutable mapping of continuous action variable names to their values."""

    state_and_discrete_action_names: tuple[StateOrActionName, ...]
    """Names of states and discrete actions in variable info table order."""

    @property
    def state_names(self) -> tuple[StateName, ...]:
        """Tuple with names of all state variables."""
        return tuple(self.states)

    @property
    def action_names(self) -> tuple[ActionName, ...]:
        """Tuple with names of all action variables."""
        return tuple(self.discrete_actions) + tuple(self.continuous_actions)

    @property
    def actions(
        self,
    ) -> MappingProxyType[ActionName, DiscreteAction | ContinuousAction]:
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
        states: MappingProxyType[StateName, ContinuousState | DiscreteState]
        | None = None,
        discrete_actions: MappingProxyType[ActionName, DiscreteAction] | None = None,
        continuous_actions: MappingProxyType[ActionName, ContinuousAction]
        | None = None,
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

    functions: EconFunctionsMapping
    """Immutable mapping of function names to internal user functions."""

    constraints: ConstraintFunctionsMapping
    """Immutable mapping of constraint names to feasibility predicates."""

    transitions: TransitionFunctionsMapping
    """Immutable mapping of transition names to transition functions."""

    stochastic_transition_names: frozenset[TransitionFunctionName]
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

    functions: EconFunctionsMapping
    """Immutable mapping of function names to internal user functions."""

    constraints: ConstraintFunctionsMapping
    """Immutable mapping of constraint names to feasibility predicates."""

    transitions: TransitionFunctionsMapping
    """Immutable mapping of transition names to transition functions."""

    stochastic_transition_names: frozenset[TransitionFunctionName]
    """Frozenset of stochastic transition function names."""

    compute_regime_transition_probs: VmappedRegimeTransitionFunction | None
    """Regime transition probability function for simulate, or `None`."""

    argmax_and_max_Q_over_a: MappingProxyType[int, ArgmaxQOverAFunction]
    """Immutable mapping of period to argmax-and-max-Q functions."""

    next_state: NextStateSimulationFunction
    """Compiled function to compute next-period states."""


@dataclasses.dataclass(frozen=True)
class _StochasticStateTransition:
    """Metadata for a stochastic state transition, used by automatic validation.

    One entry exists for every `MarkovTransition` state — and for each target
    of a per-target dict. The pre-solve state-transition validator consumes
    these to evaluate the function on the regime's grid Cartesian product and
    check that the output has the expected outcome-axis size, lies in [0, 1],
    and has rows summing to 1.
    """

    func: Callable[..., FloatND]
    """The `MarkovTransition`'s wrapped function."""

    state_name: StateName
    """Name of the state being transitioned."""

    target_regime_name: RegimeName | None
    """Target regime for per-target dicts; `None` for a plain `MarkovTransition`."""

    n_outcomes: int
    """Size of the outcome axis (always the last axis of the function output)."""

    indexing_params: tuple[str, ...]
    """Parameters used to index `probs_array`, in subscript order.

    Derived statically at process time from the function's AST. Empty
    when the function doesn't use the `probs_array[...]` pattern, in
    which case the AST subscript-order check is permissively skipped.
    """


@dataclasses.dataclass(frozen=True, kw_only=True)
class Regime:
    """Canonical regime produced by `process_regimes` from a user-facing `Regime`.

    Threaded through the solver and simulator as the engine-side representation.
    The user-facing counterpart with the same name lives in `lcm.user_regime`.
    """

    name: RegimeName
    """Regime name (key in the regimes dict)."""

    terminal: bool
    """Whether this is a terminal regime."""

    grids: MappingProxyType[StateOrActionName, Grid]
    """Immutable mapping of variable names to grid objects."""

    variables: Variables
    """States and actions of the regime, with kind/topology/shock tags."""

    active_periods: tuple[int, ...]
    """Period indices during which this regime is active."""

    regime_params_template: RegimeParamsTemplate
    """Template for the parameter structure expected by this regime."""

    solve_functions: SolveFunctions
    """Compiled functions for the backward-induction (solve) phase."""

    simulate_functions: SimulateFunctions
    """Compiled functions for the forward-simulation phase."""

    stochastic_state_transitions: MappingProxyType[
        TransitionFunctionName, _StochasticStateTransition
    ]
    """Immutable mapping of qualified transition name to validation metadata.

    Populated for every `MarkovTransition` state transition. Per-target
    dict entries appear under qualified names like `next_health__working`.
    Empty for terminal regimes and for regimes whose state transitions
    are all deterministic.
    """

    _base_state_action_space: StateActionSpace = dataclasses.field(repr=False)
    """Base state-action space before runtime grid substitution."""

    resolved_fixed_params: FlatRegimeParams = MappingProxyType({})
    """Flat resolved fixed params for this regime, used by to_dataframe targets."""

    def state_action_space(self, regime_params: FlatRegimeParams) -> StateActionSpace:
        """Return the state-action space with runtime grids filled in.

        For IrregSpacedGrid (state or continuous action) with runtime-supplied
        points, the grid points come from params as `{name}__points`. For
        `_ShockGrid` with runtime-supplied params, the grid points are computed
        from shock params in the params dict or `resolved_fixed_params`.

        Args:
            regime_params: Flat regime parameters supplied at runtime.

        Returns:
            Completed state-action space.

        """
        all_params = {**self.resolved_fixed_params, **regime_params}
        state_replacements: dict[str, ContinuousState | DiscreteState] = {}
        action_replacements: dict[str, ContinuousAction] = {}
        for name, spec in self.grids.items():
            in_states = name in self._base_state_action_space.states
            in_continuous_actions = (
                name in self._base_state_action_space.continuous_actions
            )
            if not (in_states or in_continuous_actions):
                continue
            if isinstance(spec, IrregSpacedGrid) and spec.pass_points_at_runtime:
                points_key = f"{name}__points"
                if points_key not in all_params:
                    continue
                # Runtime grid-point params are flat JAX arrays — never a
                # `MappingLeaf` / `SequenceLeaf` — so narrow via `cast`.
                points = cast("Array", all_params[points_key])
                if in_states:
                    state_replacements[name] = points
                else:
                    action_replacements[name] = points
            # `_ShockGrid` is state-only by construction (intrinsic
            # transitions, forbidden as actions per AGENTS.md). The
            # `in_states` gate makes that invariant explicit — a
            # `_ShockGrid` reaching the action branch would be a model
            # bug, not something this method should silently substitute.
            elif (
                in_states
                and isinstance(spec, _ShockGrid)
                and spec.params_to_pass_at_runtime
            ):
                all_present = all(
                    f"{name}__{p}" in all_params for p in spec.params_to_pass_at_runtime
                )
                if not all_present:
                    continue
                shock_kw: dict[str, ScalarFloat | ScalarInt] = dict(spec.params)
                for p in spec.params_to_pass_at_runtime:
                    # Runtime shock-grid params are flat JAX scalars — never
                    # a `MappingLeaf` / `SequenceLeaf` — so narrow via `cast`.
                    shock_kw[p] = cast(
                        "ScalarFloat | ScalarInt", all_params[f"{name}__{p}"]
                    )
                state_replacements[name] = spec.compute_gridpoints(**shock_kw)

        new_states = (
            dict(self._base_state_action_space.states) | state_replacements
            if state_replacements
            else dict(self._base_state_action_space.states)
        )
        new_continuous_actions = (
            dict(self._base_state_action_space.continuous_actions) | action_replacements
            if action_replacements
            else dict(self._base_state_action_space.continuous_actions)
        )
        distributed_states = _distribute_states_to_devices(
            states=MappingProxyType(new_states), grids=self.grids
        )
        return self._base_state_action_space.replace(
            states=distributed_states,
            continuous_actions=MappingProxyType(new_continuous_actions),
        )


@dataclasses.dataclass(frozen=True)
class _RegimeSharding:
    """Per-regime device-sharding plan for state and value-function arrays.

    The mesh has one axis per distributed state, named after the state.
    `state_sharding` produces the 1-D sharding for a single state grid (or
    array of subjects); `V_arr_sharding` produces the multi-axis sharding
    for the V-array given the order of states in the state-action space.
    """

    mesh: jax.sharding.Mesh
    """Device mesh whose axes are named after the distributed states."""

    distributed_state_names: tuple[StateName, ...]
    """Names of states whose axes appear in `mesh`."""

    def state_sharding(self, state_name: StateName) -> jax.NamedSharding:
        """Return the sharding for a single state's 1-D grid array."""
        return jax.NamedSharding(mesh=self.mesh, spec=jax.P(state_name))

    def V_arr_sharding(self, state_order: tuple[StateName, ...]) -> jax.NamedSharding:
        """Return the sharding for a V-array whose axes are `state_order`."""
        spec = jax.P(
            *(
                name if name in self.distributed_state_names else None
                for name in state_order
            )
        )
        return jax.NamedSharding(mesh=self.mesh, spec=spec)


def _build_regime_sharding(
    *,
    grids: MappingProxyType[StateOrActionName, Grid],
    n_devices: int,
) -> _RegimeSharding | None:
    """Build a `_RegimeSharding` covering this regime's distributed grids.

    Returns `None` when no grid is distributed. Action grids are rejected at
    user-facing `Regime` construction (see `regime_building.validation`); the
    helper assumes any grid with `distributed=True` is a state grid.

    Sharding policy depends on the number of distributed grids:
    - exactly one: build a 1-axis mesh with shape `(n_devices,)`, axis name
      equal to the state name; the grid's axis is split into `n_devices`
      chunks. Requires `n_points % n_devices == 0`.
    - more than one: build a multi-axis mesh whose axes are the grid sizes
      in iteration order, axis names equal to the state names; each state's
      axis is scattered one element per device. Requires
      `prod(grid_sizes) == n_devices` so every device is used exactly once.

    Args:
        grids: Immutable mapping of state and action names to their grids.
        n_devices: Number of available devices.

    Returns:
        The regime's sharding plan, or `None` if no grid is distributed.

    """
    distributed_grids = {name: grid for name, grid in grids.items() if grid.distributed}
    if not distributed_grids:
        return None

    state_names = tuple(distributed_grids.keys())
    grid_sizes = tuple(grid.to_jax().shape[0] for grid in distributed_grids.values())

    if len(distributed_grids) == 1:
        n_points = grid_sizes[0]
        if n_points % n_devices != 0:
            raise PyLCMError(
                "When distributing over one grid, the number of points must be "
                "a multiple of the available devices. "
                f"Gridpoints: {n_points} Available devices: {n_devices}"
            )
        mesh = jax.make_mesh(
            (n_devices,),
            state_names,
            axis_types=(jax.sharding.AxisType.Auto,),
            devices=jax.devices(),
        )
    else:
        product = math_prod(grid_sizes)
        if product != n_devices:
            raise PyLCMError(
                "When distributing over multiple grids, the product of the "
                "number of points in the grids must equal the number of "
                f"available devices. Gridpoints product: {product} "
                f"Available devices: {n_devices}"
            )
        mesh = jax.make_mesh(
            grid_sizes,
            state_names,
            axis_types=tuple(jax.sharding.AxisType.Auto for _ in distributed_grids),
            devices=jax.devices(),
        )

    return _RegimeSharding(mesh=mesh, distributed_state_names=state_names)


def _distribute_states_to_devices(
    *,
    states: MappingProxyType[StateName, FloatND | IntND],
    grids: MappingProxyType[StateOrActionName, Grid],
) -> MappingProxyType[StateName, FloatND | IntND]:
    """Place each distributed state's array on its device mesh.

    States whose grid carries `distributed=True` are placed via
    `jax.device_put` onto the per-regime mesh; other states pass through
    unchanged. The input mapping is treated as immutable.

    Args:
        states: Immutable mapping of state names to their 1-D arrays.
        grids: Immutable mapping of state and action names to their grids.

    Returns:
        Immutable mapping with distributed states placed on the mesh and
        every other state untouched.

    """
    sharding_plan = _build_regime_sharding(grids=grids, n_devices=len(jax.devices()))
    if sharding_plan is None:
        return states
    placed = dict(states)
    for state_name in sharding_plan.distributed_state_names:
        placed[state_name] = jax.device_put(
            states[state_name],
            sharding_plan.state_sharding(state_name),
        )
    return MappingProxyType(placed)


@dataclasses.dataclass(frozen=True)
class PeriodRegimeSimulationData:
    """Raw simulation data for one period in one regime."""

    V_arr: Float1D
    """Value function array for all subjects at this period."""

    actions: MappingProxyType[ActionName, FloatND | IntND]
    """Immutable mapping of action names to optimal action arrays for all subjects."""

    states: MappingProxyType[StateName, FloatND | IntND]
    """Immutable mapping of state names to state value arrays for all subjects."""

    in_regime: Bool1D
    """Boolean mask indicating which subjects are in this regime at this period."""
