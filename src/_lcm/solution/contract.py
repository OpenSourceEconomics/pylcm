"""The solver contract: what every regime solver provides to the engine.

A regime's `solver` field selects its backward-induction algorithm. The engine
dispatches polymorphically on the solver instance. Model finalization calls
`solver.validate_model(context)`; the processed build calls
`solver.validate_build(context)` and then
`solver.build_period_kernels(context)`, with no switch on solver type. Add a
solver by subclassing `Solver` and implementing `build_period_kernels`; override
either validation hook for the stage whose information it needs (both default
to no-ops).

Each entry of `SolutionKernels.period_kernels` is a `PeriodKernel`: a single
non-jitted period adapter that wraps the solver's compiled programs, calls them
with the solver's own argument layout, and returns a `KernelOutput` outside JIT:
the value array plus the artifacts the kernel publishes on its keyed channels.
The solve loop invokes the same adapter for every solver and reads each channel
by the artifact's declared key, never by solver type.

This module is an engine leaf. Resolving finalized user-regime or
`VInterpolationInfo` types at runtime would close an import
cycle through the
`lcm.solvers` façade, which re-exports `Solver` from here. They are therefore
referenced through two-form aliases: precise element types for ty under
`TYPE_CHECKING`, a bare container for the beartype claw at runtime. The
remaining engine types (`StateActionSpace`, `EGMCarry`, `EGMSimPolicy`) live in
sibling leaves with no path back to `lcm.solvers`, so they import normally and
beartype checks them precisely. The widened runtime aliases are required because
the claw beartypes each dataclass `__init__`, and under PEP 649 that forces the
field annotations to resolve to real objects when an instance is constructed.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast, runtime_checkable

from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.constraints.processed import ProcessedConstraintsMapping
from _lcm.constraints.routes import (
    ConstraintPlan,
    ConstraintRoute,
    ConstraintRouteKey,
    ConstraintSite,
    StructuralProof,
)
from _lcm.continuation import (
    ContinuationPayload,
    ContinuationSpec,
    EGMContinuationLayout,
)
from _lcm.egm.branch_aggregation import OuterBranchAggregator
from _lcm.engine import ParamCheck, StateActionSpace, Variables
from _lcm.grids import Grid
from _lcm.reachability import PhaseReachability
from _lcm.regime_building.collective import ParetoWeights
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from _lcm.transition_plans import TargetTransitionPlans
from _lcm.typing import (
    ActionName,
    ConstraintFunctionsMapping,
    EconFunction,
    EconFunctionsMapping,
    FlatParams,
    FunctionName,
    PeriodToRegimeToDissolutionFlags,
    PeriodToRegimeToSimulationPolicy,
    PeriodToRegimeToVArr,
    QAndFFunction,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    StateOrActionName,
    TransitionFunctionsMapping,
)
from lcm.ages import AgeGrid
from lcm.solver_api import (
    DISSOLUTION_FLAG,
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    SOLVER_API_VERSION,
    SOLVER_DIAGNOSTICS,
    ArtifactAuthority,
    ArtifactChannel,
    ArtifactKey,
    ArtifactStore,
    ExecutableReplayRoute,
    KernelOutput,
    SolverIdentity,
)
from lcm.typing import Float1D, FloatND, UserFunction

# The continuation channel is defined once in `_lcm.continuation`. Backward
# induction treats `ContinuationPayload` opaquely; EGM solvers additionally
# publish one `EGMContinuationSpec` bundling the template and layout metadata.

# The published off-grid simulation-policy artifact. The rule here is a **fixed
# read set** rather than opacity, because simulation genuinely interpolates the
# rows: it reads the row arrays (`endog_grid` to locate a query, `value` or
# `policy` as the field asked for, `marginal_utility` for the node slopes the
# value convention needs) and the row-axis name tuples that say which regime
# variable each leading axis stands for. Those two groups are the contract, and
# a second implementation has to publish both. Any *further* field the engine
# reaches for is the signal to introduce a protocol rather than to grow this
# one — the seam is a stated read set, not permission to read whatever is there.
#
# Which payload a regime publishes is declared, not discovered: the regime's
# `SimulationPhase.replay_route` names the exact `payload_type` and the
# `ReplayMode` under which simulation reads it, and both the model authority's
# preflight and the simulation loop dispatch on that route. `SimulationPolicy`
# enumerates the payload classes the shipped routes declare.

if TYPE_CHECKING:
    from _lcm.regime_building.finalize import FinalizedUserRegime
    from _lcm.regime_building.V import VInterpolationInfo

    UserRegimesMapping: TypeAlias = Mapping[  # noqa: UP040
        RegimeName, FinalizedUserRegime
    ]
    RegimeToVInterpolationInfo: TypeAlias = MappingProxyType[  # noqa: UP040
        RegimeName, VInterpolationInfo
    ]
else:
    # Resolving the element types closes a cycle via the `lcm.solvers` façade,
    # which re-exports `Solver` from this module. ty reads the precise types
    # above; the beartype claw checks only the outer container at runtime.
    UserRegimesMapping = Mapping
    RegimeToVInterpolationInfo = MappingProxyType


@dataclass(frozen=True, kw_only=True)
class SolverModelContext:
    """Finalized user-level information available to solver validation."""

    regime_name: RegimeName
    """Name of the regime whose solver is being validated."""

    user_regimes: UserRegimesMapping
    """Mapping of every finalized user regime in the model."""

    solve_functions: MappingProxyType[FunctionName, UserFunction]
    """Normalized solve-phase declarations for this regime.

    Model construction owns phase normalization. Solver validation receives the
    resulting pool instead of importing declaration-topology builders.
    """

    phase_variation_paths: tuple[str, ...]
    """Public declaration paths whose solve/simulate objects differ."""

    solution_reachability: PhaseReachability | None = None
    """Static solution graph when available; `None` during early validation."""


@dataclass(frozen=True, kw_only=True)
class ConstraintRouteContext:
    """What a solver may read to declare the routes it walks in one phase.

    Deliberately smaller than the build context: a route is a statement about
    the solver's own candidate production, so declaring one must not need the
    compiled kernels it will later be checked against. Built once per phase,
    against that phase's own function pool.
    """

    regime_name: RegimeName
    """Name of the regime whose routes are being declared."""

    phase: Literal["solve", "simulate"]
    """Phase whose candidates these routes produce."""

    functions: EconFunctionsMapping
    """The phase's processed functions, before any solver-specific rewrite.

    A solver that rewrites its pool going into a branch rewrites *this* and
    hands the result to the site, so a constraint's leaves are resolved through
    the scope the site actually has.
    """

    variables: Variables
    """The phase's states and actions, with kind and topology tags."""

    flat_param_names: frozenset[str]
    """Names supplied as parameters rather than computed."""

    active_periods: tuple[int, ...]
    """The regime's active periods, in ascending order.

    Not a partition. A solver whose build resolves the pool differently per age
    partitions this itself and declares one route per group of its own; one
    that does not declares a single route whose `period_group` is `None`.
    Handing over a partition the engine chose would make every solver look
    grouped whether or not it rebuilds.
    """


@dataclass(frozen=True, kw_only=True)
class SolverBuildContext:
    """Everything a solver may read to build one regime's kernels.

    Bundled so the solver method signature stays stable as solvers with
    different needs are added; each solver reads only the fields it uses.
    """

    regime_name: RegimeName
    """Name of the regime the kernels are built for."""

    ages: AgeGrid
    """The model's lifecycle age grid.

    A solver whose kernels or preconditions depend on the age a period
    prices — an age-varying schedule, a per-period cost scale — reads it
    here, so a precondition it defers to solve time can close over the
    ages rather than take them through the `ParamCheck` call.
    """

    user_regimes: UserRegimesMapping
    """Mapping of regime names to user-provided `Regime` instances."""

    solve_functions: MappingProxyType[FunctionName, UserFunction]
    """Normalized, unprocessed solve-phase declarations for this regime."""

    phase_variation_paths: tuple[str, ...]
    """Public declaration paths whose solve/simulate objects differ."""

    state_action_space: StateActionSpace
    """The regime's state-action space."""

    solution_reachability: PhaseReachability
    """Static solution graph used to derive solver layouts."""

    Q_and_F_functions: MappingProxyType[int, QAndFFunction]
    """Immutable mapping of period to Q-and-F closures."""

    grids: MappingProxyType[StateOrActionName, Grid]
    """Immutable mapping of the regime's variable names to grid objects.

    Age-invariant: for an `AgeSpecializedGrid` state this holds the representative
    age's grid. Read it for a grid's *shape traits* — kind, `n_points`, dtype,
    `batch_size` — which are invariant across ages by contract. For a grid's *node
    values* in a particular period, read `period_to_state_nodes`.
    """

    period_to_state_nodes: (
        MappingProxyType[int, MappingProxyType[StateName, Float1D]] | None
    ) = None
    """Immutable mapping of period to that period's age-specialized state nodes.

    `None` when the regime has no age-specialized state, in which case `grids` is
    already the whole story. A solver that lifts a state's nodes into a numerical
    computation must consult this per period: capturing one array outside its
    per-period loop silently pins every period to the representative age.
    """

    functions: EconFunctionsMapping
    """The regime's processed functions (params renamed to qualified names)."""

    koopmans_aggregator: EconFunction | None
    """The regime's processed Koopmans aggregator, or `None` in a terminal regime.

    Processed like every other function, so its parameters carry their qualified
    names — a solver that reads them off the runtime pool must use this object,
    not the user-facing `LinearAggregator`.
    """

    constraints: ConstraintFunctionsMapping
    """Constraint callables the solver evaluates in its numerical kernel."""

    constraint_functions: ConstraintFunctionsMapping
    """Every declared constraint callable, including compiled and proved ones."""

    processed_constraints: ProcessedConstraintsMapping
    """Every declared constraint in the form a solver can reason about.

    Read this to ask what a constraint *says* — whether it bounds a state, which
    names it reads — and `constraint_functions` to obtain its callable. The
    callables are built from these, so the two cannot disagree about what was
    declared."""

    constraint_plan: ConstraintPlan | None
    """Complete route ledger, or `None` when the solver declares no routes."""

    transitions: TransitionFunctionsMapping
    """Immutable mapping of target regime names to transition functions."""

    transition_plans: TargetTransitionPlans
    """Immutable mapping of target regime names to their transition laws."""

    compute_regime_transition_probs: RegimeTransitionFunction | None
    """Regime transition probability function, or `None` for terminal regimes."""

    regime_to_v_interpolation_info: RegimeToVInterpolationInfo
    """Immutable mapping of regime names to V-interpolation info."""

    period_to_regime_v_interp: (
        MappingProxyType[int, RegimeToVInterpolationInfo] | None
    ) = None
    """Immutable mapping of period to that period's V-interpolation info per regime.

    A period-`t` kernel reads its continuation on the *target's* period-`t+1` grid,
    so a solver that interpolates `V_{t+1}` must look the target's info up under
    `period + 1` rather than reuse `regime_to_v_interpolation_info`, which carries
    the representative age. `None` when no regime has an age-specialized state.
    """

    period_to_regime_grid_signature: (
        MappingProxyType[int, MappingProxyType[RegimeName, Hashable]] | None
    ) = None
    """Immutable mapping of period to each regime's age-specialized grid signature.

    The user's own `AgeSpecializedGrid.signature(age)` values, so a solver that
    groups periods into shared compiled programs can fold its targets' signatures
    at `period + 1` into the group key. Periods whose continuation grids differ
    then never share a trace. `None` when no regime has an age-specialized state.
    """

    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]]
    """Immutable mapping of regime names to their active period tuples."""

    flat_param_names: frozenset[str]
    """Frozenset of flat parameter names for the regime."""

    regime_to_flat_param_names: MappingProxyType[RegimeName, frozenset[str]]
    """Immutable mapping of every regime name to its flat parameter names.

    A DC-EGM source carrying into a different target regime reads the target's
    params in its per-asset-node solve, so the kernel build admits and binds
    the union of the source and its reachable carry targets' params.
    """

    enable_jit: bool
    """Whether to JIT-compile the kernels."""

    has_taste_shocks: bool
    """Whether the regime declares EV1 taste shocks on its discrete actions."""

    certainty_equivalent: CertaintyEquivalent | None
    """Nonlinear certainty equivalent declared by the regime, if any.

    `GridSearch` consumes it via the compiled Q-and-F closures; solvers
    that exploit the linear-expectation structure of the continuation
    (e.g. Euler-inversion EGM) must reject regimes that declare one.
    """

    co_map_state_names: tuple[StateName, ...] = ()
    """Fixed, distributed state names co-mapped with the continuation V.

    Their axes are the leading axes of the value-function array; a solver that reads
    the continuation V slices each off and reads only the device-local slice, so no
    all-gather is inserted. Empty when no state qualifies.
    """

    co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...] = ()
    """Per-co-map-state `in_axes` for the continuation-V mapping, aligned with
    `co_map_state_names`.

    Each entry maps a regime name to `0` (its value-function leaf carries that state —
    slice it) or `None` (the state is pruned from that regime — pass the leaf through).
    """

    stakeholders: tuple[str, ...] | None = None
    """Ordered stakeholder names for a collective regime, or `None` (singleton).

    When set, the grid-search kernel reads off each
    stakeholder's own value at the shared household argmax of the Pareto-weighted
    scalarization, and the regime's value-function array gains a trailing
    stakeholder axis.
    """

    pareto_weights: ParetoWeights | None = None
    """The household's Pareto weight evaluator; set together with `stakeholders`."""

    edge_target_regimes: tuple[RegimeName, ...] = ()
    """Target regimes this regime reaches through a gated edge, or empty.

    Non-empty only for a source regime declaring
    `gated_edges`. The grid-search kernel then substitutes each such target's
    gated continuation object `Wbar` (supplied by the solve loop under
    `edge_regime_to_V_arr`) for the raw target V in the `next_regime_to_V_arr`
    mapping it reads and lowers against. Empty for every other regime.
    """

    fold_state_names: tuple[StateName, ...] = ()
    """IID-process states declared `fold=True`, or empty (the default).

    Only `GridSearch` consumes this: the grid-search kernel weighted-averages
    each named state's axis out of the stored value immediately after the
    max-over-actions / collective readout, using the process's own
    quadrature weights. Empty keeps the default path byte-identical.
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
    """Reference regimes whose SAME-period V this regime's kernels read.

    Non-empty only for a collective regime declaring
    `same_period_refs`. The grid-search kernel then accepts the extra call
    argument `same_period_regime_to_V_arr` (the mapping of these regimes to
    their current-period V arrays, supplied by the solve loop after solving
    them earlier in the same period) and includes matching zero templates in
    its lowering arguments. Empty for every other regime, whose kernel
    signatures are unchanged.
    """


@dataclass(frozen=True, kw_only=True)
class GeneratedReplayAuthority:
    """Trusted dynamic replay facts emitted beside, never read from, a payload."""

    adaptive_outer_nodes: tuple[float, ...]
    """Exact generated outer mesh, in candidate-axis order."""


# Engine-private auxiliary key under which a kernel emits its replay authority.
# The payload is a `GeneratedReplayAuthority`; the solve loop reads it beside the
# replay-channel policy it belongs to and binds it into the model-owned solution
# authority. It is never retained into a `SolutionResult`.
GENERATED_REPLAY_AUTHORITY = ArtifactKey(
    type_id="pylcm.engine.generated_replay_authority", schema_version=1
)

_ENGINE_ARTIFACT_TYPE_IDS = frozenset(
    {
        DISSOLUTION_FLAG.type_id,
        EGM_CONTINUATION.type_id,
        GENERATED_REPLAY_AUTHORITY.type_id,
        SIMULATION_POLICY.type_id,
        SOLVER_DIAGNOSTICS.type_id,
    }
)


@dataclass(frozen=True, kw_only=True)
class BackwardInductionResult:
    """The generic outputs of one backward-induction run.

    Internal to the engine: the public `Model.solve` packages its values and
    addressed artifacts into a `SolutionResult`.
    """

    value_functions: PeriodToRegimeToVArr
    """Immutable mapping of period to each regime's value-function array."""

    simulation_policies: PeriodToRegimeToSimulationPolicy
    """Immutable mapping of period to each regime's published simulation policy.

    Sparse over regimes: only kernels that publish a policy contribute entries.
    """

    generated_replay_authorities: MappingProxyType[
        int, MappingProxyType[RegimeName, GeneratedReplayAuthority]
    ] = MappingProxyType({})
    """Trusted dynamic facts kept outside the public replay payload store."""

    dissolution_flags: PeriodToRegimeToDissolutionFlags = MappingProxyType({})
    """Immutable mapping of period to each COLLECTIVE regime's dissolution flag `D`.

    `True` on the state cells whose action mask is empty
    (distinct from a numeric `-inf` value); empty inner mappings for models
    without collective regimes, so the default (singleton) path is unchanged.

    Carries arrays only where something reads them: a gate declaring the
    `D_target` operand, which forward simulation recomputes from the flag, or a
    caller asking backward induction to retain them. A collective model whose
    gates read only value operands gets the same period keys with empty inner
    mappings, and its flags are freed period by period rather than held for the
    whole induction.
    """

    diagnostics: MappingProxyType[
        int, MappingProxyType[RegimeName, SolverDiagnostics]
    ] = MappingProxyType({})
    """Solver-published diagnostics retained under the existing ``log_level``.

    Sparse over periods and regimes and empty at ``log_level="off"``. This is
    distinct from the engine's value-validation logging: it transports the
    numerical self-report a period kernel already published instead of dropping
    it at the period boundary.
    """

    retained_continuations: ArtifactStore = field(default_factory=ArtifactStore)
    """Addressed continuations retained only under all-persistable mode."""

    replay_artifacts: ArtifactStore = field(default_factory=ArtifactStore)
    """Addressed replay artifacts, including plugin-defined payloads."""

    auxiliary_artifacts: ArtifactStore = field(default_factory=ArtifactStore)
    """Addressed solver-defined inspection/persistence artifacts."""


@runtime_checkable
class PeriodKernel(Protocol):
    """One regime's per-period solve adapter — the loop's uniform call target.

    The runtime interface is intentionally smaller than the execution declaration.
    A kernel invokes the named executable mapping and assembles its output outside JIT.
    Native and legacy execution declarations both cross
    :func:`_lcm.execution.core_program.core_program_graph`; the solve loop never reads
    their distinct declaration methods through this protocol.
    """

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> PeriodKernel:
        """Return a copy with the regime's fixed params bound into the core(s).

        The adapter owns its solver's binding rule — which fixed params reach
        the core (and any inline closure it wraps) — so the engine binds fixed
        params without a solver-type switch.
        """
        ...

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
        logger: logging.Logger,
    ) -> KernelOutput:
        """Invoke the compiled program(s) and assemble the period's output.

        A one-program kernel reads `compiled_cores["main"]`; a multi-program
        kernel reads each of its own program keys. The return is the public
        :class:`~lcm.solver_api.KernelOutput`: the value array and the
        artifacts the kernel publishes, each on the channel of its retention
        semantics under its declared `ArtifactKey`.

        `logger` carries the run's validation policy. A kernel that can detect a
        defect only by reading a device value back reads it in raise mode alone
        (`validation_raises`), so a healthy solve never pays a host transfer for
        a check it would not act on.
        """
        ...


@dataclass(frozen=True)
class _BoundLiquidMargin:
    """Resolved DAG role names carried privately by an EGM-family solver."""

    state: StateName
    """Name of the continuous state the liquid margin is solved over."""

    action: ActionName
    """Name of the continuous action the Euler equation is inverted for."""

    resources: FunctionName
    """Name of the function giving resources available before the action."""

    post_decision_state: FunctionName
    """Name of the function giving the post-decision state the grid spans."""

    before_cost: FunctionName | None = None
    """Gross-resources function of a composed `NetOfAdjustmentCost`, else `None`."""

    cost: FunctionName | None = None
    """Cost function of a composed `NetOfAdjustmentCost`, else `None`."""


@dataclass(frozen=True)
class _BoundOuterContinuousMargin:
    """Resolved outer-margin DAG role names carried privately by a solver.

    `no_adjustment` is `None` where the regime declares the identity map with
    `lcm.outer_unchanged`, which is the form every consumer branches on.
    """

    state: StateName
    """Name of the continuous state the outer margin searches over."""

    action: ActionName
    """Name of the continuous action setting the outer margin."""

    post_decision_state: FunctionName
    """Name of the function giving the outer post-decision state."""

    no_adjustment: FunctionName | None
    """Name of the no-adjustment map, or `None` for the identity map."""

    adjustment_cost: OuterBranchAggregator | None = None
    """Declared adjustment-cost structure, or `None` for the deterministic
    maximum. A solver reads it to select its keeper/adjuster fold and to refuse
    a declaration its kernels cannot aggregate."""


@dataclass(frozen=True, kw_only=True)
class SolutionKernels:
    """Per-period solve adapters produced by a solver."""

    period_kernels: Mapping[int, PeriodKernel]
    """Immutable mapping of period to the regime's uniform period adapter."""

    continuation_spec: ContinuationSpec | None = None
    """Template and identity of the continuation this solver's kernels publish."""

    param_checks: tuple[ParamCheck, ...] = ()
    """Preconditions the engine passes every parameter draw against real params.

    A solver whose scope condition is a property of the *evaluated* model — the
    budget's affinity in the liquid state, a carried law's constancy between
    breakpoints — cannot check it in `validate_build`, which runs before any
    parameter value exists. It publishes the check here instead, and the engine
    calls each entry in order as `check(flat_params=...)` for every draw. Each
    check owns its evaluation schedule and any cached verdict, so different
    solvers can choose different schedules in the same model. Empty for a solver
    whose scope is decided by structure alone.
    """

    replay_route: ExecutableReplayRoute | None = None
    """External executable replay route, or ``None`` for an engine adapter."""

    artifact_authorities: Mapping[ArtifactKey, ArtifactAuthority] = MappingProxyType({})
    """Model-derived contracts for custom retained artifacts."""

    def __post_init__(self) -> None:
        """Snapshot mappings and reject contradictory artifact identities."""
        object.__setattr__(
            self, "period_kernels", MappingProxyType(dict(self.period_kernels))
        )
        authorities = dict(self.artifact_authorities)
        if any(
            key != authority.descriptor.key for key, authority in authorities.items()
        ):
            raise ValueError(
                "SolutionKernels.artifact_authorities keys must equal each "
                "authority descriptor's ArtifactKey."
            )
        if any(
            authority.descriptor.channel is ArtifactChannel.DIAGNOSTIC
            for authority in authorities.values()
        ):
            raise TypeError(
                "SolutionKernels cannot declare custom DIAGNOSTIC authorities; "
                "KernelOutput exposes no corresponding extension channel."
            )
        reserved_type_ids = tuple(
            sorted(
                key.type_id
                for key in authorities
                if key.type_id in _ENGINE_ARTIFACT_TYPE_IDS
            )
        )
        if reserved_type_ids:
            raise ValueError(
                "SolutionKernels cannot declare custom authorities under "
                f"engine-owned artifact type ids: {reserved_type_ids!r}."
            )
        object.__setattr__(self, "artifact_authorities", MappingProxyType(authorities))

    @property
    def continuation_template(self) -> ContinuationPayload | None:
        """Return the template payload for generic rolling and lowering code."""
        return (
            None if self.continuation_spec is None else self.continuation_spec.template
        )


def simulation_route(
    *,
    context: ConstraintRouteContext,
    solver_path: tuple[str, ...],
    structural_proofs: tuple[StructuralProof, ...] = (),
) -> ConstraintRoute:
    """Build the simulate-phase route a solver walks over whole candidates.

    Simulation is not a solver's pipeline. It walks the regime's DAG on each
    subject's realized states and its realized action, so every name is
    computable and the feasibility check sees a complete candidate — which is
    true of every solver shipped today, whatever it does when solving. One
    unrestricted site, at the simulation stage.

    Built here rather than spelled out by each solver on purpose. Six copies of
    one declaration agree by convention until one of them does not, and the
    disagreement would be a field nobody compared. A solver whose simulate
    phase genuinely differs writes its own route instead of calling this, which
    then reads as the deliberate departure it would be.

    Args:
        context: What the solver may read about the regime and the phase.
        solver_path: The nest of solvers, so the route still says whose it is.
        structural_proofs: Route-specific proofs consulted at the site. Most
            simulation routes need none because their complete candidates can
            be evaluated directly.

    Returns:
        The route.

    """
    from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
        resolve_periodized_nodes,
    )

    functions = cast(
        "EconFunctionsMapping",
        (
            resolve_periodized_nodes(
                mapping=context.functions, period=context.active_periods[0]
            )
            if context.active_periods
            else context.functions
        ),
    )
    return ConstraintRoute(
        key=ConstraintRouteKey(
            phase="simulate", period_group=None, solver_path=solver_path
        ),
        sites=(
            ConstraintSite(
                stage="simulation",
                function_pool=functions,
                available_names=None,
                structural_proofs=structural_proofs,
            ),
        ),
    )


class Solver(ABC):
    """Base class for regime solvers — the polymorphic dispatch target.

    The engine calls `validate_model`, then later `validate_build` and
    `build_period_kernels` on the instance, matching the engine's own
    polymorphism (`Grid(ABC)`, the stochastic processes). Subclasses are frozen
    dataclasses carrying the solver's configuration.
    """

    @abstractmethod
    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build the regime's per-period solve adapters."""

    @property
    def identity(self) -> SolverIdentity:
        """Return this solver implementation's exact public compatibility identity.

        Built-in and legacy extension solvers receive a deterministic version-1
        identity. Supported external plugins should override this property with a
        package-owned stable identifier and release version.
        """
        solver_type = type(self)
        return SolverIdentity(
            plugin_id=f"{solver_type.__module__}.{solver_type.__qualname__}",
            plugin_version="1",
            solver_api_version=SOLVER_API_VERSION,
        )

    @property
    def egm_continuation_layout(self) -> EGMContinuationLayout:
        """Declare how a reading parent interprets this solver's EGM carry.

        The default is an endogenous-grid row per discrete-action combination,
        with row-specific abscissae and no stacked outer-candidate axis. Solvers
        whose representation differs override this one bundled declaration.
        """
        return EGMContinuationLayout()

    @property
    def publishes_simulation_policy(self) -> bool:
        """Whether this solver's kernels publish a simulation policy.

        The canonical consuming-route signal is `SimulationPhase.egm_policy_read`,
        and a solve retains a published policy only through that route. This
        declaration serves the result ledger: a cell whose solver publishes a
        policy that no route consumes records the omission as not applicable,
        while a solver that never publishes one records nothing.
        """
        return False

    @property
    def publishes_one_sided_jump_reads(self) -> bool:
        """Whether the published carry duplicates abscissae across a value jump.

        A solver that resolves a declared jump one-sidedly publishes two rows
        at (essentially) the same abscissa, one per side of the jump. Those
        abscissae move with any state entering the jump's variable or its
        threshold, so a parent must not fold its stochastic node axes into a
        single read across states that move them.
        """
        return False

    def validate_model(self, *, context: SolverModelContext) -> None:  # noqa: B027
        """Validate finalized user declarations. Default: no-op."""

    def validate_build(self, *, context: SolverBuildContext) -> None:  # noqa: B027
        """Validate processed period/build context. Default: no-op."""

    def build_constraint_routes(
        self,
        *,
        context: ConstraintRouteContext,  # noqa: ARG002
    ) -> tuple[ConstraintRoute, ...] | None:
        """Declare the candidate-production routes this solver walks in one phase.

        `None` — the default — says the solver has not declared its routes, and
        nothing is planned for it. That is not the same as declaring an
        unrestricted route: a permissive default would claim, of every solver
        nobody has written down, the opposite of the truth for any that in fact
        evaluates no constraint at all. Undeclared also leaves the attributed
        refusal a custom solver already gets exactly where it is, rather than
        replacing it with a generic failure.

        Not abstract, for the same reason. Making it abstract would turn every
        existing custom solver into a construction-time `TypeError` naming a
        method its author has never heard of.

        Args:
            context: What the solver may read about the regime and the phase.

        Returns:
            One route per pipeline the solver walks in this phase, or `None`
            when it has not declared them.

        """
        return None

    @property
    def supports_transition_local_lotteries(self) -> bool:
        """Whether this solver consumes transition-local lottery axes.

        A ``JointTransition`` is enumerated inside the source action value.  The
        grid-search Q kernel implements that dataflow.  Continuation-based
        solvers must opt in only after their own child-read representation also
        enumerates the canonical ``TargetTransitionPlan`` lotteries; accepting
        the declaration without doing so would defer a semantic mismatch to a
        runtime missing-node failure.
        """
        return False

    @property
    def required_continuation_keys(self) -> frozenset[ArtifactKey]:
        """Continuation artifacts this solver reads from its target regimes.

        An endogenous-grid solver inverts the Euler equation against its
        target regimes' value *and marginal* on a continuation grid, so each
        target — including a terminal one — must publish a continuation under
        one of these keys, which the engine rolls alongside
        `next_regime_to_V_arr`. Grid search reads only the value array, so its
        set is empty. Model building matches each key against what every
        reachable target publishes, so a version the targets do not carry is
        refused before any solve, without forking on the solver type.
        """
        return frozenset()

    @property
    def supports_nonlinear_certainty_equivalent(self) -> bool:
        """Whether this solver's continuation step implements the EZ recursion.

        Reading a continuation and assuming expected utility are separate
        properties. An Euler inversion written against `E[V']` is only valid
        under `LinearExpectation`, so by default a solver that reads a
        continuation refuses a nonlinear certainty equivalent rather than
        solving a recursion the regime does not declare. A solver whose step
        inverts the recursive Euler equation instead — carrying the certainty
        equivalent's transform through the marginal — overrides this to `True`
        and is admitted. Grid search never consults it: reading only the value
        array, it aggregates any certainty equivalent in concrete values.
        """
        return False


class OneMarginSolver(Solver):
    """Marker base for solvers consuming one explicit liquid margin.

    Future NB-EGM implementations can join this family without changing the
    regime interface; the only shared operation is immutable role binding.
    """

    @abstractmethod
    def _with_liquid_margin(self, margin: _BoundLiquidMargin) -> OneMarginSolver:
        """Return an immutable solver copy bound to one liquid margin."""


class TwoMarginSolver(Solver):
    """Marker base for solvers consuming liquid and outer continuous margins."""

    @abstractmethod
    def _with_margins(
        self,
        *,
        liquid: _BoundLiquidMargin,
        outer: _BoundOuterContinuousMargin,
    ) -> TwoMarginSolver:
        """Return an immutable solver copy bound to both margins."""


_BOUND_TYPES: dict[tuple[type, type], type] = {}


def bind_roles(*, solver: Solver, role_type: type, **roles: object) -> Solver:
    """Return a copy of `solver` carrying `roles`, keeping its own type.

    The regime resolves a solver's DAG role names and hands them back to the
    solver, which has to answer with an object that is still the one the user
    constructed: a subclass keeps the fields it added and the methods it
    overrides, so a custom solver reaches the engine as itself.

    `role_type` declares the role fields for the stock solver and is returned
    directly for it. For any other concrete type, the bound class is derived
    from that type and `role_type` together, so the subclass sits ahead of the
    stock solver in the method resolution order.

    Args:
        solver: The solver whose roles are being bound.
        role_type: The bound dataclass declaring the role fields, itself a
            subclass of the stock solver.
        **roles: The resolved role names, one per field `role_type` declares.
            An entry naming a field the solver already carries replaces it, which
            is how a nest hands on its *bound* inner solver rather than the
            public one it was declared with.

    Returns:
        A frozen solver of `solver`'s own type carrying the resolved roles.

    """
    solver_type = type(solver)
    stock_type = role_type.__bases__[0]
    values = {
        field.name: getattr(solver, field.name)
        for field in fields(solver_type)  # ty: ignore[invalid-argument-type]
    }
    if solver_type is stock_type:
        bound_type = role_type
    elif issubclass(solver_type, role_type):
        # Already bound: a regime that replaces one of its own fields rebinds
        # the solver it is already carrying, and its type is the answer.
        bound_type = solver_type
    else:
        key = (solver_type, role_type)
        if key not in _BOUND_TYPES:
            derived = type(
                f"_Bound{solver_type.__name__}", (solver_type, role_type), {}
            )
            _BOUND_TYPES[key] = dataclass(frozen=True, kw_only=True)(derived)
        bound_type = _BOUND_TYPES[key]
    return bound_type(**(values | roles))
