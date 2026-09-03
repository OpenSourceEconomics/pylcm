import dataclasses
from collections.abc import Callable, Iterator, Mapping
from math import prod as math_prod
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import jax
from jax import Array

from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.continuation import ContinuationPayload, EGMContinuationSpec
from _lcm.grids import DiscreteGrid, Grid, IrregSpacedGrid
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.reachability import PhaseReachability
from _lcm.regime_building.collective import ParetoWeights
from _lcm.transition_plans import TargetTransitionPlans
from _lcm.typing import (
    ActionName,
    ArgmaxQOverAFunction,
    ConstraintFunctionsMapping,
    EconFunctionsMapping,
    FlatRegimeParams,
    FunctionName,
    NextStateSimulationFunction,
    QAndFFunction,
    RegimeName,
    RegimeParamsTemplate,
    RegimeTransitionFunction,
    StateName,
    StateOrActionName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
    VmappedRegimeTransitionFunction,
)
from _lcm.utils.containers import first_non_none
from lcm.exceptions import PyLCMError
from lcm.typing import (
    Bool1D,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    Int1D,
    IntND,
    ScalarFloat,
    ScalarInt,
)

if TYPE_CHECKING:
    from _lcm.regime_building.gated_edges import ResolvedGatedEdge
    from _lcm.solution.contract import PeriodKernel

    # The contract module imports this one at runtime, so `PeriodKernel` is
    # reachable only under `TYPE_CHECKING`. ty reads the precise element type;
    # the beartype claw checks only the outer `Mapping` container at runtime
    # (see the runtime alias below).
    PeriodKernelsMapping: TypeAlias = Mapping[int, PeriodKernel]  # noqa: UP040
else:
    PeriodKernelsMapping = Mapping

# A precondition a solver can only check once parameter *values* exist. Kernels
# are built at `Model` construction, before any params are supplied, so a check
# that must evaluate or differentiate the model's DAG on real schedules, tables,
# and coefficients cannot run there. A solver publishes such a check with its
# kernels. The engine calls every published check as `check(flat_params=...)` on
# every parameter draw; each check owns its evaluation schedule and any cached
# verdict. The check reports by raising; its return value is ignored. Defined
# here rather than in `_lcm.solution.contract` — which re-exports it — for the
# same reason as `ContinuationPayload`: the engine stays a leaf of the contract,
# not a peer in a cycle.
type ParamCheck = Callable[..., None]


@dataclasses.dataclass(frozen=True)
class VariableInfo:
    """Kind/topology/process tags for one state or action variable."""

    kind: Literal["state", "action"]
    """Whether the variable is a state or an action."""

    topology: Literal["continuous", "discrete"]
    """Topology as treated by pylcm's solve/simulate machinery.

    Stochastic processes have topology `"discrete"` because their value
    space is approximated by a finite grid of nodes, even though the
    underlying random variable is mathematically continuous. Combine with
    `is_process` when the distinction matters.

    """

    is_process: bool
    """Whether the variable is a stochastic process (always a state)."""


@dataclasses.dataclass(frozen=True)
class Variables(Mapping[StateOrActionName, VariableInfo]):
    """States + actions of a regime, with pre-computed name-tuple views.

    Mapping access by variable name returns the per-variable `VariableInfo`.
    Named accessors return tuples of names in iteration order. Use
    `_lcm.variables.from_regime` to construct from a regime; pass `info`
    directly only when names are already in the desired order.

    """

    info: MappingProxyType[StateOrActionName, VariableInfo]
    """Immutable mapping of variable name to its `VariableInfo`."""

    state_names: tuple[StateOrActionName, ...] = dataclasses.field(init=False)
    """Names of variables with kind='state'."""

    action_names: tuple[StateOrActionName, ...] = dataclasses.field(init=False)
    """Names of variables with kind='action'."""

    discrete_state_names: tuple[StateOrActionName, ...] = dataclasses.field(init=False)
    """Names of states with topology='discrete' (includes stochastic processes)."""

    continuous_state_names: tuple[StateOrActionName, ...] = dataclasses.field(
        init=False
    )
    """Names of states with topology='continuous'."""

    discrete_action_names: tuple[StateOrActionName, ...] = dataclasses.field(init=False)
    """Names of actions with topology='discrete'."""

    continuous_action_names: tuple[StateOrActionName, ...] = dataclasses.field(
        init=False
    )
    """Names of actions with topology='continuous'."""

    state_and_discrete_action_names: tuple[StateOrActionName, ...] = dataclasses.field(
        init=False
    )
    """Every state plus every discrete action — the gridded variable set."""

    process_names: tuple[StateOrActionName, ...] = dataclasses.field(init=False)
    """Names of variables with `is_process=True`."""

    def __post_init__(self) -> None:
        items = tuple(self.info.items())
        object.__setattr__(
            self,
            "state_names",
            tuple(name for name, info in items if info.kind == "state"),
        )
        object.__setattr__(
            self,
            "action_names",
            tuple(name for name, info in items if info.kind == "action"),
        )
        object.__setattr__(
            self,
            "discrete_state_names",
            tuple(
                name
                for name, info in items
                if info.kind == "state" and info.topology == "discrete"
            ),
        )
        object.__setattr__(
            self,
            "continuous_state_names",
            tuple(
                name
                for name, info in items
                if info.kind == "state" and info.topology == "continuous"
            ),
        )
        object.__setattr__(
            self,
            "discrete_action_names",
            tuple(
                name
                for name, info in items
                if info.kind == "action" and info.topology == "discrete"
            ),
        )
        object.__setattr__(
            self,
            "continuous_action_names",
            tuple(
                name
                for name, info in items
                if info.kind == "action" and info.topology == "continuous"
            ),
        )
        object.__setattr__(
            self,
            "state_and_discrete_action_names",
            tuple(
                name
                for name, info in items
                if info.kind == "state" or info.topology == "discrete"
            ),
        )
        object.__setattr__(
            self,
            "process_names",
            tuple(name for name, info in items if info.is_process),
        )

    def __getitem__(self, key: StateOrActionName) -> VariableInfo:
        return self.info[key]

    def __iter__(self) -> Iterator[StateOrActionName]:
        return iter(self.info)

    def __len__(self) -> int:
        return len(self.info)


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
        *,
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class SolutionPhase:
    """Solve-phase view of a canonical regime.

    Owns everything backward induction reads: the solve variables and grids
    (a carried state contributes no axis here — its name is a
    derived function), the compiled function sets, and the state-action
    space. Reading phase-dependent data through this namespace makes the
    phase explicit at every call site.
    """

    _variables: Variables
    """Solve states and actions, with kind/topology/process tags.

    Private bundle behind the name-tuple properties (`state_names`,
    `action_names`, `discrete_state_names`); read those instead.
    """

    grids: MappingProxyType[StateOrActionName, Grid]
    """Immutable mapping of variable names to grid objects (productmap order)."""

    functions: EconFunctionsMapping
    """Immutable mapping of function names to internal user functions.

    An age-specialized function is resolved here at the regime's first active
    age, which stands in for the regime as a whole. Its consumers — feasibility
    checks and the additional targets a simulation reports — need only a
    concrete function, not the one belonging to a particular period. Pricing a
    continuation does need the period's own age; that reads
    `continuation_functions` instead.
    """

    _continuation_functions: EconFunctionsMapping | None = None
    """Solve-phase pool with age-specialized functions left unresolved.

    A simulated agent prices its continuation under the law it solved with, and
    an age-specialized belief is a different function at every age. Leaving the
    functions unresolved here lets each period resolve them at its own age.
    Resolving them once, at the regime's first active age, would impose one
    age's belief on the whole regime and can reverse the simulated choice.
    `None` — the regime has nothing age-specialized — falls back to `functions`.
    """

    constraints: ConstraintFunctionsMapping
    """Immutable mapping of constraint names to feasibility predicates."""

    has_compiled_constraint_boundaries: bool = False
    """Whether the solve route compiles any constraint boundary into its kernel."""

    transitions: TransitionFunctionsMapping
    """Immutable mapping of transition names to transition functions."""

    transition_plans: TargetTransitionPlans
    """Immutable mapping of target regime names to their transition laws."""

    reachability: PhaseReachability
    """Construction-time solve graph shared by every canonical regime."""

    compute_regime_transition_probs: RegimeTransitionFunction | None
    """Regime transition probability function for solve, or `None`."""

    period_kernels: PeriodKernelsMapping
    """Immutable mapping of period to the regime's uniform period adapter.

    Every regime — grid search or DC-EGM — exposes one adapter per period; the
    solve loop invokes them the same way and reads each `KernelResult` without
    branching on solver type. A grid-search adapter for a terminal regime in a
    model with a DC-EGM regime is wrapped by an engine-owned output decorator so
    it additionally publishes the regime's closed-form continuation carry.
    """

    continuation_spec: EGMContinuationSpec | None = None
    """Concrete EGM continuation template bundled with its static layout."""

    @property
    def continuation_template(self) -> ContinuationPayload | None:
        """Return the opaque payload template used by generic engine code."""
        return (
            None if self.continuation_spec is None else self.continuation_spec.template
        )

    validation_regime_transition_probs: RegimeTransitionFunction | None
    """Probability function retaining declared cells for runtime validation."""

    compute_intermediates: MappingProxyType[int, Callable]
    """Immutable mapping of period to intermediate-computation closures.

    Productmap-wrapped and fused with on-device reductions inside a single
    `jax.jit`; invoked only in the error path when `validate_V` detects
    NaN. Each closure returns a flat dict of reductions — scalar
    `{U_nan,CE_nan,Q_nan,F_feasible}_overall` entries, per-dimension
    `{...}_by_{name}` vectors, and `regime_probs` as a dict of per-target
    scalar means — so full-shape U/F/CE/Q arrays never materialise in
    host-visible memory.
    """

    param_checks: tuple[ParamCheck, ...] = ()
    """The regime solver's preconditions that need real parameter values.

    `check_solver_params` passes every parameter draw to every published check;
    each check owns its evaluation schedule and cache. Empty for a solver whose
    scope is decided by structure alone.
    """

    pareto_weights: ParetoWeights | None = None
    """The household's Pareto weight evaluator, or `None` for a singleton regime.

    Kept here so the params-bound preflight can read the weights the solve will
    actually use, rather than re-deriving them from the user declaration.
    """

    resolved_fixed_params: FlatRegimeParams = MappingProxyType({})
    """Flat resolved fixed params, consulted for runtime grid substitution."""

    _base_state_action_space: StateActionSpace = dataclasses.field(repr=False)
    """Base state-action space before runtime grid substitution."""

    period_state_axes: (
        MappingProxyType[int, MappingProxyType[StateOrActionName, object]] | None
    ) = None
    """Per-period node arrays for age-varying (`AgeSpecializedGrid`) states.

    `{period: {state_name: nodes}}` — the current period's grid nodes for each
    age-varying continuous state, used by backward induction to override the
    (representative) base axis so period `t`'s value function is tabulated on
    period `t`'s grid. `None` for age-invariant regimes (the base axis is used
    unchanged)."""

    @property
    def solves_from_continuation(self) -> bool:
        """Whether this regime's V is built from interpolated continuations.

        True exactly for a non-terminal regime that publishes a continuation
        payload — such a regime's kernels solve by reading its targets'
        continuations rather than by the compiled Q-and-F grid program. A
        grid-search regime publishes no continuation, and a terminal
        carry-producing regime publishes one without reading any (no
        regime-transition probs). Downstream consumers ask this capability —
        the brute U/F/E/Q breakdown cannot reproduce such a regime's failure
        rows, and its inversion-internal functions are not simulate-readable
        targets — instead of asking which solver produced the regime.
        """
        return (
            self.compute_regime_transition_probs is not None
            and self.continuation_template is not None
        )

    @property
    def continuation_functions(self) -> EconFunctionsMapping:
        """Solve-phase pool a simulated agent prices its continuation with.

        Falls back to `functions` when the regime has no age-specialized
        function to resolve. Read this rather than `_continuation_functions`,
        so the fallback is applied once.
        """
        return self._continuation_functions or self.functions

    @property
    def state_names(self) -> tuple[StateOrActionName, ...]:
        """Solve-phase state names in canonical (productmap) order."""
        return self._variables.state_names

    @property
    def action_names(self) -> tuple[StateOrActionName, ...]:
        """Solve-phase action names in canonical (productmap) order."""
        return self._variables.action_names

    @property
    def discrete_state_names(self) -> tuple[StateOrActionName, ...]:
        """Solve-phase discrete state names (includes stochastic processes)."""
        return self._variables.discrete_state_names

    @property
    def discrete_grids(self) -> MappingProxyType[StateOrActionName, DiscreteGrid]:
        """Discrete grids (states and actions), for label/code mapping."""
        return MappingProxyType(
            {
                name: grid
                for name, grid in self.grids.items()
                if isinstance(grid, DiscreteGrid)
            }
        )

    def state_action_space(self, regime_params: FlatRegimeParams) -> StateActionSpace:
        """Return the state-action space with runtime grids filled in.

        For IrregSpacedGrid (state or continuous action) with runtime-supplied
        points, the grid points come from params as `{name}__points`. For
        `_ContinuousStochasticProcess` with runtime-supplied params, the grid
        points are computed from process params in the params dict or
        `resolved_fixed_params`.

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
            # `_ContinuousStochasticProcess` is state-only by construction (intrinsic
            # transitions, forbidden as actions per AGENTS.md). The
            # `in_states` gate makes that invariant explicit — a
            # `_ContinuousStochasticProcess` reaching the action branch would be a model
            # bug, not something this method should silently substitute.
            elif (
                in_states
                and isinstance(spec, _ContinuousStochasticProcess)
                and spec.params_to_pass_at_runtime
            ):
                all_present = all(
                    f"{name}__{p}" in all_params for p in spec.params_to_pass_at_runtime
                )
                if not all_present:
                    continue
                process_kw: dict[str, ScalarFloat | ScalarInt] = dict(spec.params)
                for p in spec.params_to_pass_at_runtime:
                    # Runtime process-grid params are flat JAX scalars — never
                    # a `MappingLeaf` / `SequenceLeaf` — so narrow via `cast`.
                    process_kw[p] = cast(
                        "ScalarFloat | ScalarInt", all_params[f"{name}__{p}"]
                    )
                state_replacements[name] = spec.compute_gridpoints(**process_kw)

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class EGMPolicyRead:
    """Names for the off-grid read of a published `EGMSimPolicy` in simulate.

    The stored policy maps the endogenous resources value to the optimal
    continuous action; simulation interpolates it at each subject's resources
    instead of argmaxing over the action grid.
    """

    action_name: ActionName
    """The EGM continuous action the interpolated policy value replaces."""

    resources_target: FunctionName
    """DAG function computing the endogenous resources the policy is read at."""

    savings_lower_bound: float
    """Lower bound of the solver's savings grid — the borrowing limit the
    post-read feasibility check enforces (`action <= resources - bound`)."""

    row_discrete_state_names: tuple[StateName, ...]
    """Exact producer-owned discrete-state row axes, in storage order."""

    row_passive_state_names: tuple[StateName, ...]
    """Exact producer-owned passive-state row axes, after discrete states."""

    row_discrete_action_names: tuple[ActionName, ...]
    """Exact producer-owned discrete-action row axes, after state axes."""

    row_axis_lengths_by_period: MappingProxyType[int, tuple[int, ...]]
    """Exact lengths of the ordered row axes in every active period."""

    float_dtype: str
    """Canonical dtype of every numeric array leaf in the payload."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class NNBEGMPolicyRead:
    """Realized-state adapter for the NNBEGM joint replay payload."""

    outer_target_function_by_period: MappingProxyType[
        int, Callable[..., Mapping[str, FloatND]]
    ]
    """Resolved simulate-phase target DAG for each active period."""

    outer_post_decision: FunctionName
    """Target whose retained identity is inverted during replay."""

    outer_no_adjustment_target: FunctionName | None
    """Custom keeper target, or ``None`` for the outer-state identity."""

    outer_state_name: StateName
    """State whose realized value is the default keeper target."""

    inner_action_name: ActionName
    """Exact continuous inner-action role published by the solver."""

    outer_action_name: ActionName
    """Exact continuous outer-action role published by the solver."""

    state_names: tuple[StateName, ...]
    """Exact producer-owned state axes of a finite replay bank."""

    state_axis_lengths_by_period: MappingProxyType[int, tuple[int, ...]]
    """Exact lengths of the ordered finite-bank state axes by period."""

    row_discrete_state_names: tuple[StateName, ...]
    """Exact discrete-state row axes of each nested inner policy."""

    row_passive_state_names: tuple[StateName, ...]
    """Exact passive-state row axes of each nested inner policy."""

    row_axis_lengths_by_period: MappingProxyType[int, tuple[int, ...]]
    """Exact lengths of the ordered nested-policy row axes by period."""

    discrete_action_names: tuple[ActionName, ...]
    """Exact finite-bank categorical columns, in producer order."""

    discrete_action_code_domains: MappingProxyType[ActionName, tuple[int, ...]]
    """Exact allowed categorical codes for every finite-bank column."""

    candidate_discrete_action_codes: tuple[tuple[int, ...], ...]
    """Exact outer-tiled categorical rows of a finite candidate bank."""

    candidate_count: int | None
    """Exact finite-bank candidate count, or ``None`` for a nested route."""

    float_dtype: str
    """Canonical dtype of every floating array leaf in the payload."""

    integer_dtype: str
    """Canonical dtype of categorical-code leaves in the payload."""

    outer_grid_values: tuple[float, ...] | None = None
    """Exact finite outer search nodes, or ``None`` for a nested route."""

    n_keeper_candidates: int | None = None
    """Exact finite-bank keeper count, or ``None`` for a nested route."""

    liquid_state_name: StateName | None = None
    """Nested route's inner Euler-state role, if applicable."""

    resources_target: FunctionName | None = None
    """Nested route's intrinsic-budget resources target, if applicable."""

    savings_lower_bound: float | None = None
    """Nested route's model-owned intrinsic borrowing limit, if applicable."""

    golden_iterations: int | None = None
    """Nested route's exact outer-refinement iteration budget."""

    value_atol: float | None = None
    """Nested route's exact canonical-Q absolute agreement band."""

    value_rtol: float | None = None
    """Nested route's exact canonical-Q relative agreement band."""

    outer_state_domain_by_period: MappingProxyType[int, tuple[float, float]] = (
        MappingProxyType({})
    )
    """Exact outer-state domain endpoints in every active period."""

    policy_applicable: bool = True
    """Whether this route structurally publishes a simulation policy."""

    policy_required: bool = True
    """Whether every successful solve must retain a payload for this route."""

    fixed_cost_simulation_unsupported: bool = False
    """Whether solution analytically integrates an observed fixed cost whose
    realized keeper/adjuster branch simulation cannot yet replay."""

    replay_policy_is_nested: bool = False
    """Whether the configured outer search publishes the nested continuous-outer
    payload (`NestedEGMSimPolicy`) rather than the finite candidate bank
    (`NNBEGMSimPolicy`). Set by the adaptive mesh, unset by the finite grid; it
    is what lets caller-supplied replay policies be checked against the type the
    solve actually returns."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class SimulationPhase:
    """Simulate-phase view of a canonical regime.

    Owns everything forward simulation reads: the per-subject state set
    (solve states plus carried-only states), the per-subject grids, and the
    compiled function sets. Reading phase-dependent data through this
    namespace makes the phase explicit at every call site.
    """

    _variables: Variables
    """Simulate states (solve states plus carried-only states, appended) and
    actions.

    NOT a productmap order — carried-only states are appended after the solve
    states, so this ordering carries no dispatch meaning; it only fixes column
    order in simulation output. Private bundle behind the name-tuple
    properties (`state_names`, `action_names`, `discrete_state_names`); read
    those instead.
    """

    grids: MappingProxyType[StateOrActionName, Grid]
    """Solve grids plus each carried-only state's simulate-phase grid."""

    carried_only_state_names: frozenset[StateName]
    """States carried only in simulation: derived functions (no grid axis)
    during backward induction, genuine seeded-and-evolved states here."""

    functions: EconFunctionsMapping
    """Immutable mapping of function names to internal user functions."""

    constraints: ConstraintFunctionsMapping
    """Immutable mapping of constraint names to feasibility predicates."""

    transitions: TransitionFunctionsMapping
    """Immutable mapping of transition names to transition functions."""

    transition_plans: TargetTransitionPlans
    """Immutable mapping of target regime names to their transition laws."""

    reachability: PhaseReachability
    """Construction-time simulate graph shared by every canonical regime."""

    compute_regime_transition_probs: VmappedRegimeTransitionFunction | None
    """Regime transition probability function for simulate, or `None`."""

    argmax_and_max_Q_over_a: MappingProxyType[int, ArgmaxQOverAFunction]
    """Immutable mapping of period to argmax-and-max-Q functions."""

    edge_reference_regimes_by_period: MappingProxyType[int, tuple[RegimeName, ...]] = (
        MappingProxyType({})
    )
    """Per period, the regimes whose landing values the decision program reads.

    A gate reference and a leg fallback are read inside the source's own
    decision function — but only where that function carries the gated
    continuation. A regime whose gate is applied by the simulate router
    instead never names them, a period whose edge targets are all inactive has
    no gated continuation to read, and a period where only some of the
    regime's edges land in an active target reads only those edges'
    references: the others name regimes the landing period never solved. The
    two sites that hand the channel over, AOT lowering and the runtime call,
    both consult this mapping, so the compiled program's pytree and the
    call's arguments cannot disagree.
    """

    Q_and_F: MappingProxyType[int, QAndFFunction]
    """Immutable mapping of period to pointwise state-action value functions.

    Evaluates the canonical `Q` and its feasibility at one action value per
    subject, rather than maximizing over the action grid. It shares the model
    DAG, transitions, constraints, aggregators, params, and next-period value
    arrays with `argmax_and_max_Q_over_a`, so a value it reports for an off-grid
    candidate action is directly comparable with the grid winner's value.
    """

    next_state: MappingProxyType[int, NextStateSimulationFunction]
    """Immutable mapping of period to next-period-state functions."""

    age_specialized_function_names: frozenset[FunctionName] = frozenset()
    """Function names that were `AgeSpecializedFunction` in the user regime.

    The published `functions` hold these resolved at the regime's representative
    age only — the per-period programs (`argmax_and_max_Q_over_a`, `next_state`)
    carry the true per-age closures. Consumers computing period-specific outputs
    from `functions` (e.g. `additional_targets`) must reject targets that depend
    on these names."""

    egm_policy_read: EGMPolicyRead | NNBEGMPolicyRead | None = None
    """Off-grid read of the published EGM simulation policy, or `None`.

    Present only where replaying the solve-phase policy is valid:
    - the regime is solved by an EGM kernel that publishes `EGMSimPolicy`;
    - the Koopmans aggregator `W` is phase-invariant (a phase-variant `W`
      changes the simulate-phase FOC, so the stored policy is wrong there);
    - the regime declares no taste shocks.
    `None` keeps the grid-argmax decision path for the continuous action.
    """

    @property
    def state_names(self) -> tuple[StateOrActionName, ...]:
        """States carried per subject: solve states plus carried-only states."""
        return self._variables.state_names

    @property
    def action_names(self) -> tuple[StateOrActionName, ...]:
        """Simulate-phase action names."""
        return self._variables.action_names

    @property
    def discrete_state_names(self) -> tuple[StateOrActionName, ...]:
        """Per-subject discrete state names (includes stochastic processes)."""
        return self._variables.discrete_state_names

    @property
    def discrete_grids(self) -> MappingProxyType[StateOrActionName, DiscreteGrid]:
        """Discrete grids (states and actions), for label/code mapping."""
        return MappingProxyType(
            {
                name: grid
                for name, grid in self.grids.items()
                if isinstance(grid, DiscreteGrid)
            }
        )

    @property
    def carried_grids(self) -> MappingProxyType[StateName, Grid]:
        """Grids of the carried-only states (the simulate-phase domains)."""
        return MappingProxyType(
            {
                name: self.grids[name]
                for name in self.state_names
                if name in self.carried_only_state_names
            }
        )


@dataclasses.dataclass(frozen=True)
class _StochasticStateTransition:
    """Metadata for a stochastic state transition, used by automatic validation.

    One entry exists for every `MarkovTransition` state — for each target of a
    per-target dict, and for each phase variant of a `Phased` law. The pre-solve
    state-transition validator consumes these to evaluate the function on the
    regime's grid Cartesian product and check that the output has the expected
    outcome-axis size, lies in [0, 1], and has rows summing to 1.

    A `Phased` law contributes one entry per phase, and both are kept. They are
    different kernels doing different jobs: the perceived one prices every action
    in backward induction, the realized one governs the draw the simulation
    takes. Either is fatal if malformed, so keeping one key for both would leave
    whichever was collected second the only one ever checked.
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

    phase: Literal["solve", "simulate"] | None = None
    """Phase this kernel belongs to; `None` for a phase-invariant law.

    Carried so that a failure names the offending phase — a phase-invariant law
    and the two variants of a `Phased` pair are otherwise validated identically.
    """


@dataclasses.dataclass(frozen=True, kw_only=True)
class Regime:
    """Canonical regime produced by `process_regimes` from a user-facing `Regime`.

    Threaded through the solver and simulator as the engine-side representation.
    The user-facing counterpart with the same name lives in `lcm.regime`.
    """

    name: RegimeName
    """Regime name (key in the regimes dict)."""

    terminal: bool
    """Whether this is a terminal regime."""

    active_periods: tuple[int, ...]
    """Period indices during which this regime is active."""

    regime_params_template: RegimeParamsTemplate
    """Template for the parameter structure expected by this regime."""

    solution: SolutionPhase
    """Solve-phase view: variables, grids, compiled functions, state-action space."""

    simulation: SimulationPhase
    """Simulate-phase view: carried states (incl. pairs), grids, compiled functions."""

    stochastic_state_transitions: MappingProxyType[
        TransitionFunctionName, _StochasticStateTransition
    ]
    """Immutable mapping of qualified transition name to validation metadata.

    Populated for every `MarkovTransition` state transition. Per-target
    dict entries appear under qualified names like `next_health__working`.
    Empty for terminal regimes and for regimes whose state transitions
    are all deterministic.
    """

    has_taste_shocks: bool = False
    """Whether the regime declares EV1 taste shocks on its discrete actions."""

    fold_state_names: tuple[StateName, ...] = ()
    """IID-process states declared `fold=True`, or empty (the default).

    A folded state is integrated out of the stored value by quadrature at
    solve time, so it is NOT an axis of the regime's stored `V`-array: the
    backward-induction V topology (`_get_regime_V_shapes_and_shardings`)
    excludes it from the shape/sharding it computes for this regime. Empty
    keeps the default path byte-identical.
    """

    certainty_equivalent: CertaintyEquivalent | None
    """Nonlinear certainty equivalent declared by the regime, if any."""

    resolved_fixed_params: FlatRegimeParams = MappingProxyType({})
    """Flat resolved fixed params for this regime, used by to_dataframe targets."""

    granular_param_expansions: MappingProxyType[FunctionName, tuple[str, ...]] = (
        MappingProxyType({})
    )
    """Immutable mapping of coarse-template law keys to granular qname prefixes.

    A state law whose params the template keys coarsely (`next_<state>`)
    binds granularly in the engine (`<target>__next_<state>`); each entry
    lists every such prefix across both phases so canonical flat params can
    materialize one shared leaf per target. Empty when every law's params
    are user-granular or absent.
    """

    stakeholders: tuple[str, ...] | None = None
    """Ordered stakeholder names for a collective regime, or `None` (singleton).

    When set, the regime's value-function array carries a
    trailing length-`len(stakeholders)` axis: the backward-induction V topology
    appends it so the zero template and the roll match the collective kernel's
    stakeholder-valued output.
    """

    stakeholder_names_to_ids: MappingProxyType[str, int] = MappingProxyType({})
    """The model's role vocabulary: every regime's stakeholder names, as codes.

    One vocabulary for the whole model, carried on every regime, so a role
    survives the move between regimes that name their stakeholders differently
    and a simulated row can carry it as an integer. `NO_ROLE` is what a row in
    a singleton regime carries — it occupies no household's role.
    """

    edge_reference_regimes: tuple[RegimeName, ...] = ()
    """Regimes a gated edge reads a projected value from, or empty.

    A gate reference and a leg fallback both name another regime's value at
    coordinates a projection produces. Neither is tabulated on the target's
    grid, so both are read where the source lands — inside the source's own
    kernel — at the value of the period the source lands in. The rolled V
    mapping already carries that array; these names are what pick it out and
    thread each reference regime's OWN grid params beside it.
    """

    same_period_ref_regimes: tuple[RegimeName, ...] = ()
    """Regimes whose SAME-period V this regime's solve kernel reads, or empty.

    Non-empty only for a collective regime declaring
    `same_period_refs`. The backward-induction loop orders each period's active
    regimes topologically by these edges (references solved first) and passes
    the referenced regimes' freshly solved V arrays into this regime's kernel
    call. Empty for every other regime — the default path is unchanged.
    """

    gated_edges: MappingProxyType[RegimeName, ResolvedGatedEdge] = MappingProxyType({})
    """This regime's gated edges keyed by TARGET regime name, or empty.

    Non-empty only for a source regime declaring
    `gated_edges`: each entry folds a gated continuation object `Wbar` on the
    target regime's grid at each period's end, which this regime's continuation
    reads in place of the raw target V. Empty for every other regime.

    Each entry carries its own compiled callables — the `Wbar` fold, the
    simulate-side gate evaluator, and one fallback state projector per leg —
    so a consumer reads them off the edge it is already holding rather than
    re-pairing parallel mappings by target name or by leg position.
    """


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


#: Highest rank a simulated value array can carry: the subject axis, plus at
#: most one trailing stakeholder axis for a collective regime.
_MAX_SIMULATED_V_RANK = 2


@dataclasses.dataclass(frozen=True)
class PeriodRegimeSimulationData:
    """Raw simulation data for one period in one regime."""

    V_arr: FloatND
    """Value function array for all subjects at this period.

    Shape `(n_subjects,)` for a singleton regime; `(n_subjects,
    n_stakeholders)` for a collective regime — each stakeholder's own
    value at the household's shared argmax, mirroring the solve-side V's
    trailing stakeholder axis.

    The grid-argmax value: where the off-grid policy read replaces the
    continuous action, this value belongs to the pre-replacement gridded
    action combination, not to the recorded action — the pair is not a
    consistent (action, value) evaluation there.
    """

    actions: MappingProxyType[ActionName, FloatND | IntND]
    """Immutable mapping of action names to optimal action arrays for all subjects."""

    states: MappingProxyType[StateName, FloatND | IntND]
    """Immutable mapping of state names to state value arrays for all subjects."""

    in_regime: Bool1D
    """Boolean mask indicating which subjects are in this regime at this period."""

    own_stakeholder: Int1D
    """The role each subject occupies here, as a code in the model's role
    vocabulary (`Regime.stakeholder_names_to_ids`).

    A row in a collective regime carries the stakeholder it IS — the wife's row
    her role, the husband's his — which is what decides the leg it follows when
    the household ends. A row in a singleton regime occupies no role and carries
    `NO_ROLE`.
    """

    nested_policy_fallback: Bool1D
    """Per-subject flag that the continuous-outer (nested) off-grid policy read
    was refused and fell back to the grid-argmax action pair at this period.

    True only for subjects in a regime whose simulation published a nested
    (continuous-outer) policy where the runtime replay could not certify the
    off-grid read: the branch read left its live row support, a non-affine outer
    transition surfaced at the recovered action, or the inner action came out
    non-finite, non-positive, or over budget.
    All-False for every other path: no nested payload, the flat single-EGM
    read, passive rows, or the grid path. Inference on the continuous-outer
    path must refuse whenever any entry is True — the recorded action there is
    the gridded fallback, not the model's off-grid optimum.
    """

    def __post_init__(self) -> None:
        """Check `V_arr` against the per-subject axis every other field carries.

        `V_arr` is rank-polymorphic so a collective regime can publish one value
        per stakeholder, which means its annotation admits any rank and cannot
        state the agreement the record depends on: a leading subject axis
        matching `in_regime`, and at most one trailing role axis. A `V_arr` that
        disagrees misaligns every value in the simulated frame against the
        actions and states beside it, and nothing downstream re-derives the
        pairing.

        Raises:
            ValueError: `V_arr` has no subject axis, has more than one axis
                beyond it, or its leading axis is not the subject axis.
        """
        # Reached with a non-array leaf whenever a pytree traversal rebuilds the
        # record from something other than arrays — a shape-dtype struct still
        # answers, an arbitrary placeholder does not. Topology is only defined
        # for a leaf that carries a shape.
        V_shape = getattr(self.V_arr, "shape", None)
        mask_shape = getattr(self.in_regime, "shape", None)
        if V_shape is None or mask_shape is None:
            return
        if not 1 <= len(V_shape) <= _MAX_SIMULATED_V_RANK:
            msg = (
                f"V_arr must be a per-subject value array, optionally with a "
                f"trailing stakeholder axis for a collective regime, so its "
                f"rank is 1 or 2; got shape {V_shape}."
            )
            raise ValueError(msg)
        if V_shape[0] != mask_shape[0]:
            msg = (
                f"V_arr's leading axis is the subject axis and must match the "
                f"per-subject `in_regime` mask: V_arr has {V_shape[0]} rows, "
                f"`in_regime` has {mask_shape[0]}."
            )
            raise ValueError(msg)


# Register as a JAX pytree so traversals like `jax.block_until_ready` and
# `jax.tree.map` recurse into the fields instead of treating the dataclass
# as an opaque leaf. Without registration, an outer drain over a
# `dict[regime][period] -> PeriodRegimeSimulationData` skips the inner
# `V_arr` / `in_regime` / `actions` / `states` — the per-subject lazy
# compute graphs build up across periods and only fire at access time,
# whose materialisation workspace dwarfs the per-period output.
jax.tree_util.register_dataclass(
    PeriodRegimeSimulationData,
    data_fields=(
        "V_arr",
        "actions",
        "states",
        "in_regime",
        "own_stakeholder",
        "nested_policy_fallback",
    ),
    meta_fields=(),
)
