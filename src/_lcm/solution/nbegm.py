"""The NB-EGM solver (non-convex-budget endogenous grid method) and its kernels.

`NBEGM` solves a 1-D consumption-savings problem whose budget may carry
declared breakpoints (kinks and cliffs): the budget is partitioned into
affine-in-liquid intervals, each case piece is solved by EGM, and the pieces
are merged by the query-side upper envelope. `build_period_kernels` returns one
`PeriodKernel` per period — a non-jitted adapter that wraps the solver's
shared jitted core (deduped by function identity, so periods sharing a core
reuse one compiled program) and assembles a `KernelOutput` (value array,
continuation carry, published simulation policy) outside JIT.

The kernel-building imports are function-local so the public `lcm.solvers`
façade stays a thin re-export that pulls in no numerical engine modules.
"""

import ast
import functools
import inspect
import itertools
import logging
import math
import textwrap
import warnings
from collections.abc import Callable, Hashable, Iterator, Mapping
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    NotRequired,
    Protocol,
    TypeAliasType,
    TypedDict,
    cast,
    overload,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
from beartype import beartype
from dags import concatenate_functions
from jax.typing import DTypeLike

import lcm.typing as lcm_typing
from _lcm.axis_boundaries import (
    AxisBoundary,
    ResolvedAxisPartition,
    canonical_boundary_order,
    partition_effect_for_schedule_kind,
    resolve_axis_partition,
)
from _lcm.beartype_conf import REGIME_CONF
from _lcm.certainty_equivalent import CertaintyEquivalent, LinearExpectation
from _lcm.constraints.dispositions import CompileBoundary
from _lcm.constraints.ir import Compare, Const, Ref
from _lcm.constraints.routes import ConstraintPlan, ConstraintRoute
from _lcm.continuation import EGMContinuationLayout, EGMContinuationSpec
from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import (
    EGMCarry,
    build_template_egm_carry,
    egm_carry_role_tree,
    egm_query_role_tree,
    shard_carry_template,
)
from _lcm.egm.continuation_grids import (
    continuation_grid_signature,
    continuation_v_interpolation_info,
)
from _lcm.egm.dcegm_query import DCEGMQuery
from _lcm.egm.declared_law import build_declared_liquid_law
from _lcm.egm.nbegm import NBEGMRegistry
from _lcm.egm.nbegm_breakpoints import (
    interval_midpoints,
    interval_segment_coefficients,
)
from _lcm.egm.nbegm_constraint_boundaries import (
    NBEGMFeasibilityBoundaryProgram,
    feasibility_axis_boundaries,
)
from _lcm.egm.nbegm_probe_stages import (
    evaluate_probe_stage,
    probe_between,
    probe_clamped_breakpoints,
    probe_distinct,
    probe_greater,
    probe_grid_scalar,
    probe_int_fill,
    probe_law_samples,
    probe_midpoint,
    probe_nextafter,
    probe_points,
    probe_raw_breakpoints,
    probe_scalar,
    probe_selected_threshold,
    probe_sorted,
    probe_unstack,
)
from _lcm.egm.nbegm_query import NBEGMQuery
from _lcm.egm.nbegm_query_model import NBEGMIntervalRecipe, NBEGMQueryModel
from _lcm.egm.nbegm_step import (
    JumpPosition,
    _NBEGMDiscreteQueryStep,
    _NBEGMIntervalRows,
    _NBEGMQueryStep,
    _OwnedNBEGMQueryStep,
    nbegm_multi_interval_step,
    nbegm_one_asset_step,
    nbegm_recurring_jump_step,
    nbegm_unified_step,
)
from _lcm.egm.nested_floor import (
    NESTED_FLOOR_INPUT,
    PreparedNestedFloor,
    nested_floor_at_cell,
)
from _lcm.egm.one_asset_egm_step import _EGMSavingsSupport
from _lcm.egm.preferences import Preferences
from _lcm.egm.published_policy import NBEGMGridPolicy
from _lcm.egm.query_data import (
    EGMQueryData,
    QueryNodeKind,
    build_template_egm_query_data,
)
from _lcm.egm.upper_envelope.query import ComparisonArithmetic
from _lcm.engine import StateActionSpace, placed_devices_for_ids
from _lcm.execution.core_program import (
    CoreArgumentBuilder,
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    ProgramScope,
    ReducedAxis,
    TiledOutputAxis,
    _CheckedCoreVariant,
    bind_core_program_fixed_params,
)
from _lcm.execution.output_layout import VALUE, OutputRoleTree, StateAxesLeading
from _lcm.execution.reductions import (
    INTERVAL_ENVELOPE_REDUCTION,
    WEIGHTED_EXPECTATION_REDUCTION,
)
from _lcm.grids import ContinuousGrid, DiscreteGrid
from _lcm.grids.base import Grid
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.params.edges import EDGES, flat_namespaces, regime_kernel_params
from _lcm.params.mapping_leaf import MappingLeaf, UserMappingLeaf
from _lcm.post_decision_bound import PostDecisionBoundSpec, grid_origin_bound_spec
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION
from _lcm.solution.continuation_arguments import (
    MARGINAL_ARGUMENT,
    MarginalLeafArguments,
    MarginalLeafCore,
    marginal_leaf_reads,
)
from _lcm.solution.continuation_reads import (
    continuation_leaf_reads,
    published_continuation_template,
    with_continuation_leaf_reads,
)
from _lcm.solution.continuation_target import (
    period_to_continuation_target,
    target_period_grid,
    union_fixed_params,
    union_free_params,
)
from _lcm.solution.contract import (
    ConstraintRouteContext,
    ContinuationPayload,
    OneMarginSolver,
    ParamCheck,
    PeriodKernel,
    SolutionKernels,
    SolverBuildContext,
    SolverModelContext,
    _BoundLiquidMargin,
)
from _lcm.solution.dcegm import (
    CELL_AXIS,
    STOCHASTIC_NODE_AXIS,
    _carry_subset,
    _fail_if_exact_affine_kernel_unavailable,
)
from _lcm.solution.egm import (
    OneRowCore,
    _build_egm_period_kernel,
    _EGMArgumentBuilder,
    _EGMPeriodKernel,
    _scalar_egm_output_role,
    declare_egm_carry_reads,
    guard_regime_mass,
)
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    ContinuationReadCounts,
    FeasibilityCounts,
    FeasibilityKernelOutput,
    QueryInputStatus,
    _unflatten_continuation_read_counts,
    merge_continuation_read_counts,
    merge_query_input_status,
    reduce_bound_input_status,
    reduce_continuation_read_counts,
    reduce_query_input_status,
    shift_savings_grid,
)
from _lcm.solution.periodization import (
    resolve_solver_build_context,
    restrict_solver_build_context_to_period_group,
    solver_period_group_key,
)
from _lcm.time import TimeAxis
from _lcm.transition_plans import SupportOrigin
from _lcm.typing import (
    AnnotationForm,
    ArrayTree,
    EconFunction,
    EconFunctionArg,
    EconFunctionKwargs,
    EconFunctionsMapping,
    FlatParams,
    ParamsLeaf,
    PytreeValue,
    QualifiedName,
    RegimeName,
    ShapeDtypePytree,
    TransitionFunction,
    TransitionFunctionsMapping,
)
from _lcm.utils.counts import sum_counts
from _lcm.utils.dispatchers import map_over_leading_axis
from _lcm.utils.functools import array_result, is_user_function
from lcm._solver_api.capabilities import SolverExecutionCapabilities
from lcm.case_piece import CaseBoundary, EqualityOwner, PiecewiseAffineMeta
from lcm.collective import CollectiveUtility
from lcm.exceptions import (
    ExecutionPlanningError,
    RegimeInitializationError,
    UnsupportedOperationError,
)
from lcm.fixed_forms import cash_on_hand_with_subsidy
from lcm.phased import Phased
from lcm.solver_api import (
    EGM_CONTINUATION,
    EGM_ENDOGENOUS_COORDINATE,
    SIMULATION_POLICY,
    ArtifactKey,
    ContinuationCapabilities,
    KernelOutput,
)
from lcm.typing import (
    ActionName,
    BoolND,
    ContinuousState,
    DiscreteState,
    Float1D,
    FloatND,
    FunctionName,
    Int1D,
    IntND,
    ParameterName,
    ReferenceName,
    ScalarFloat,
    StateName,
    StateOrActionName,
    TransitionFunctionName,
    UserFunction,
    ValueND,
)

if TYPE_CHECKING:
    from _lcm.egm.continuation import ContinuationPlan, _ChildRead
else:
    # `_lcm.egm.continuation` imports the `lcm.solvers` facade, which imports
    # this module, so it is not importable here at runtime.
    type ContinuationPlan = object
    type _ChildRead = object

# A regime's function slot as the probe classifiers read it. Solver runtime
# modules never import `lcm.regime`, so the slot's union is spelled out here.
type _FunctionEntry = UserFunction | Phased | CollectiveUtility | None

# Key under which ride-along periods share one compiled core: the continuation
# targets either side of a `"|"` separator, followed by those targets'
# age-specialized grid signatures at `period + 1`.
type _RideAlongGroupKey = tuple[RegimeName | Hashable, ...]

# A law the continuation reads: a target's next-state or Euler-state function, or
# the regime transition, each returning its value tree.
type _ContinuationLaw = Callable[..., ArrayTree]

# A compiled liquid-derivative program: the derivative of a budget, or of a
# continuation law, at a batch of liquid points, with every other argument of the
# differentiated function supplied per call.
type _BudgetProbeProgram = Callable[[FloatND, EconFunctionKwargs], FloatND]
type _LawProbeProgram = Callable[[FloatND, EconFunctionKwargs], ArrayTree]

# A structural argument a deferred probe binds at model build: the composed
# function or continuation plan it probes, the liquid grid and its samples, the
# declared breakpoints, the derivative programs, a name, or a flag.
type _ProbeBinding = (
    Callable[..., FloatND]
    | ContinuationPlan
    | Float1D
    | tuple[_NBEGMSource, ...]
    | MappingProxyType[int, _BudgetProbeProgram]
    | MappingProxyType[_ContinuationLaw, _LawProbeProgram]
    | str
    | bool
)

# One ride-along state's coordinate in a cell.
type _RideCoordinate = FloatND | IntND

# Every discrete action the envelope branches over, paired with its grid codes.
# The branch axis is the product of these code sets, so the pairing has to keep
# each action's name next to its own codes.
type DiscreteActionCodes = tuple[tuple[ActionName, tuple[int, ...]], ...]

# Raw branch cores publish two economic channels before the current-state owner
# adds its acceptance counts and eligibility record.
type _RawNBEGMOutput = (
    tuple[FloatND, EGMCarry] | BoundedResult[tuple[FloatND, EGMCarry]]
)
type _RawNBEGMCore = Callable[..., _RawNBEGMOutput]
type _NBEGMStepChannels = tuple[Float1D, Float1D, Float1D] | _NBEGMQueryStep
# Branch mapping adds optional leading action axes to each liquid-query channel.
type _NBEGMMappedStepChannels = tuple[FloatND, FloatND, FloatND] | _NBEGMQueryStep
type _NBEGMReadPair = tuple[FloatND, FloatND]
type _NBEGMReadResult = _NBEGMReadPair | BoundedResult[_NBEGMReadPair]
type _NBEGMContinuationRows = (
    tuple[FloatND, ...]
    | BoundedResult[tuple[FloatND, ...]]
    | BoundedResult[_NBEGMReadPair]
)
type _NBEGMCellRead = _NBEGMContinuationRows | _NBEGMIntervalContinuation
type _NBEGMPreparedRead = _NBEGMCellRead | _NBEGMPreparedCell
type _NBEGMCellNumeric = tuple[FloatND, ...] | _NBEGMCellQueryPublication
type _NBEGMCellOutput = _NBEGMCellNumeric | BoundedResult[_NBEGMCellNumeric]
type _NBEGMRideReplayOutput = (
    tuple[FloatND, EGMCarry, FloatND]
    | tuple[FloatND, EGMCarry, FloatND, FloatND, FloatND]
)
type _NBEGMRideOutput = tuple[FloatND, EGMCarry] | _NBEGMRideReplayOutput
_NBEGM_RIDE_MAIN_ARITY = 2
_NBEGM_RIDE_POLICY_ARITY = 3
_PREPARED_NBEGM_KEEPER_ARITY = 4
type _NBEGMBoundedRideOutput = BoundedResult[
    tuple[FloatND, EGMCarry, tuple[IntND, IntND]]
    | tuple[FloatND, EGMCarry, FloatND, tuple[IntND, IntND]]
    | tuple[FloatND, EGMCarry, FloatND, FloatND, FloatND, tuple[IntND, IntND]]
]
type _NBEGMNestedRideOutput = BoundedResult[
    tuple[_NBEGMRideOutput, tuple[IntND, IntND], BoolND]
]


type _NBEGMSingleAxisCore = (
    _NBEGMCaseCore
    | _NBEGMContinuousCore
    | _NBEGMDiscreteCore
    | _NBEGMScheduleDiscreteCore
)

# Planner axis for the continuation-dependent liquid intervals.
INTERVAL_AXIS = "interval"

# Planner axis for the Cartesian product of discrete-action branches.
BRANCH_AXIS = "branch"

# A leading reserved separator cannot start user names or qualified parameters.
# Trailing separators keep explicit width keywords free of class name-mangling.
_INTERVAL_COORDINATE = "__lcm_interval_indices"
_WIDTH_KEYWORDS = (
    "__lcm_cell_width__",
    "__lcm_interval_width__",
    "__lcm_branch_width__",
    "__lcm_stochastic_node_width__",
)


@dataclass(frozen=True)
class _NBEGMSource:
    """One breakpoint of one schedule, in solver-facing form.

    A regime may declare several piecewise-affine schedules, each bracketing on
    its own monotone income variable; every threshold of every schedule becomes
    one source. The solver maps each source to its per-ride-along-cell asset
    preimage in its own variable and merges all sources into one sorted partition.
    """

    variable: str
    """Name of the monotone schedule variable this breakpoint brackets on."""
    threshold_param_name: QualifiedName
    """Qualified parameter name of this breakpoint's threshold."""
    kind: str
    """Discontinuity kind: `continuous_kink`, `jump`, or `hard_constraint`."""
    equality_owner: Literal["below", "above"]
    """Side of the schedule coordinate containing the exact threshold."""
    derived_of_liquid_dag: Callable[..., FloatND] | None
    """Composed schedule variable as a function of the liquid state, or `None`
    when the schedule varies in the liquid state directly (no preimage needed)."""
    derived_param_names: tuple[ParameterName, ...]
    """Unqualified parameter names the schedule variable reads (non-state args)."""
    derived_state_names: tuple[StateName, ...] = ()
    """Ride-along state names the schedule variable reads, so the per-cell call
    passes only the cell entries the derived DAG accepts."""
    threshold_index_state: StateName | None = None
    """Ride-along state indexing this breakpoint's threshold table, or `None` for a
    scalar threshold. When set, the threshold is read per cell as
    `threshold[cell_state, static_index]`."""
    threshold_static_index: int | None = None
    """Static column index into the threshold table, applied after the ride-along
    row index. `None` leaves the row-indexed value as-is."""
    threshold_subkey: str | None = None
    """Entry to select inside a `MappingLeaf` threshold param (`leaf.data[subkey]`),
    resolved before the ride-along row index and static column index. `None` when
    the threshold param is a bare array."""


@dataclass(frozen=True)
class _NBEGMScheduleSpec:
    """Build-time statics for a continuous piecewise-affine schedule regime."""

    coh_of_liquid_dag: Callable[..., FloatND]
    """Composed `coh` as a function of the liquid state and qualified params."""
    coh_param_names: tuple[QualifiedName, ...]
    """Qualified parameter names `coh` reads (everything but the state axes)."""
    utility_dag: Callable[..., FloatND]
    """Composed period utility as a function of the consumption action, the
    ride-along states it reads, and qualified utility params. The ride-along core
    binds it per cell to invert the Euler equation and evaluate the period value."""
    consumption_action_name: ActionName
    """Name of the continuous consumption action the period utility reads."""
    liquid_state_name: StateName
    """Name of the liquid state the schedule and budget vary in."""
    ride_along_state_names: tuple[StateName, ...]
    """State axes other than the liquid axis (the budget varies per ride-along cell)."""
    liquid_axis_pos: int
    """Index of the liquid axis in the canonical productmap state order. The
    ride-along core solves in working layout (ride axes leading the liquid axis)
    and moves the liquid axis to this position so the published value array follows
    the productmap order — a no-op when every ride-along axis is a discrete state
    sorting ahead of the liquid axis, a genuine transpose for a continuous co-state
    declared after it."""
    threshold_param_names: tuple[QualifiedName, ...]
    """Qualified parameter names of the schedule's thresholds."""
    breakpoint_kinds: tuple[str, ...]
    """Discontinuity kind per threshold, in the schedule's declared order."""
    sources: tuple[_NBEGMSource, ...] = ()
    """Every breakpoint across all declared schedules, merged on the liquid axis.
    The ride-along core maps each source to its own per-cell asset preimage."""
    discount_factor_dag: Callable[..., FloatND] | None = None
    """Composed `discount_factor` as a function of its ride-along state arguments and
    qualified params, or `None` when the regime uses pylcm's flat
    `koopmans_aggregator__discount_factor` parameter. When set, the ride-along
    core resolves the discount factor per cell."""
    discrete_actions: DiscreteActionCodes = ()
    """Each discrete action the budget shifts, paired with its grid codes, empty
    when the regime carries no discrete action. Excluded from `coh_param_names` —
    the envelope core binds them per branch."""
    param_checks: tuple[ParamCheck, ...] = ()
    """Preconditions on the composed budget, evaluated as their schedules request.

    Collected here because the budget is composed while the spec is, and checking
    its affinity in the liquid state needs each draw's schedule parameters."""

    @property
    def branch_bindings(self) -> tuple[MappingProxyType[ActionName, int], ...]:
        """One binding of every declared discrete action, per envelope branch."""
        return _branch_bindings(self.discrete_actions)


@dataclass(frozen=True)
class _NBEGMScheduleDiscreteSpec:
    """Build-time statics for a discrete action over a cliffed single-liquid budget.

    Each discrete-action value shifts cash-on-hand and the budget also carries a
    declared schedule (kinks/jumps) on the liquid state. Per action value the
    continuous subproblem is solved by the multi-interval EGM step honouring the
    schedule, and the discrete choice is taken by the upper envelope over the
    branch values.
    """

    coh_of_liquid_action_dag: Callable[..., FloatND]
    """Composed budget node as a function of the liquid state, the discrete actions,
    and qualified params."""
    coh_param_names: tuple[QualifiedName, ...]
    """Qualified parameter names the budget reads (excluding the liquid state and the
    discrete actions)."""
    coh_action_names: tuple[ActionName, ...]
    """Discrete action names read by the composed resource function."""
    liquid_state_name: StateName
    """Name of the liquid (Euler) state the budget varies in."""
    discrete_actions: DiscreteActionCodes
    """Each discrete action enveloped over, paired with its grid codes."""
    threshold_param_names: tuple[QualifiedName, ...]
    """Qualified parameter names of the schedule's thresholds (liquid breakpoints)."""
    breakpoint_kinds: tuple[str, ...]
    """Discontinuity kind per threshold, in the schedule's declared order."""
    param_checks: tuple[ParamCheck, ...] = ()
    """Parameter-dependent preconditions for every schedule-action branch."""

    @property
    def branch_bindings(self) -> tuple[MappingProxyType[ActionName, int], ...]:
        """One binding of every declared discrete action, per envelope branch."""
        return _branch_bindings(self.discrete_actions)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class NBEGM(OneMarginSolver):
    """Case-piece endogenous-grid solver for a 1-D consumption--saving regime.

    A regime whose budget is split by case boundaries on the liquid state (e.g. a
    Medicaid asset test) is smooth within each case. NBEGM solves each case by
    ordinary 1-D EGM, masks each case's candidates to the region where its
    predicate is consistent with the recovered state, and merges the cases on the
    liquid grid with the branch-aware upper envelope. The strict/non-strict
    consistency split gives the boundary point to the side that owns equality.
    The step carries the marginal value of liquid backward, like the plain 1-D
    EGM, so this regime both reads and publishes a continuation carry.

    The regime's declarations select the kernel:

    - Case-piece split (`lcm.case_boundary` / `lcm.piece`): the binary jump step
      on the two masked subsidy cases. The case-piece route splits exactly
      one additive cash-on-hand contribution across one binary predicate.
    - Piecewise-affine schedule (`lcm.piecewise_affine`): the breakpoint
      kinds pick the step — kinks/floors only, jumps only, or mixed —
      solved by `coh` inversion per continuous run and masked across the jumps.
    - Schedule with ride-along co-states: one tile-local core per period that
      reads each cell block's transition-aware continuation and runs the block's
      envelope solve in savings space before the next block, so the expected
      continuation over every cell never exists as one array.
    - Discrete action over a smooth budget: one continuous subproblem per
      discrete-action value, merged by the discrete upper envelope.
    """

    @property
    def transition_local_lottery_origins(self) -> frozenset[SupportOrigin]:
        """The child read enumerates an edge's local draws as node axes."""
        return frozenset({SupportOrigin.SOURCE_PROCESS})

    @property
    def capabilities(self) -> SolverExecutionCapabilities:
        """Describe this configured solver without building numerical kernels."""
        return SolverExecutionCapabilities(
            required_declaration="ConsumptionSavingsRegime with one LiquidMargin",
            problem_shape=(
                "Supported declared kinks, jumps, hard boundaries or smooth discrete "
                "branches"
            ),
            prerequisites=(
                "Supported case-piece or piecewise-affine declaration; proven "
                "constraint routes; nonlinear CE only on eligible ride-along routes; "
                "no EV1 taste shocks; marginal donation only on an unsharded "
                "self-carry main program with eligible ownership"
            ),
            main_tradeoff=(
                "Preserves declared topology; structural probes and candidate "
                "geometry add cost"
            ),
            reduced_axes=("stochastic_node", "interval", "branch"),
            tiled_axes=("cell",),
            host_axes=(),
            host_driven_programs=(),
            donation_candidates=("main",),
            supports_ev1_taste_shocks=False,
            supports_nonlinear_certainty_equivalent=True,
        )

    savings_grid: ContinuousGrid
    """Exogenous post-decision savings grid `s = coh - consumption` (>= 0)."""
    jump_read: Literal["one_sided", "bridged"] = "one_sided"
    """How the parent's continuation interpolation treats the child value's cliffs.

    The within-period solve is jump-aware in both modes: cases are masked and the
    declared equality owner receives the boundary. The mode changes only the rows
    published for a parent to interpolate:

    - `"one_sided"` publishes every jump preimage twice, once with each one-sided
      value and marginal limit, so a parent approaching the cliff reads the matching
      side.
    - `"bridged"` publishes one value per point on the ordinary liquid grid. A parent
      may then interpolate across a cliff, accepting finite-grid cliff error in return
      for the simpler representation.

    In both modes, the solver interpolates each stochastic row before applying the
    parent expectation; `"bridged"` does not expose an unfurled stochastic-row
    representation for a later fold.
    """
    envelope_arithmetic: ComparisonArithmetic = "ordinary"
    """Which arithmetic decides ownership in the merged upper envelope.

    Every case, corner, and node candidate is read at each liquid query point and
    the largest owns it. How that comparison is made is this knob:

    - `"ordinary"` (default) takes the largest read in the working format. It
      decides every bracketed query at a fraction of the certified read's cost. Near
      a crossing whose candidates differ by less than the format's resolution it can
      pick a different owner or move the crossing slightly; the published value is
      still read from one candidate together with its policy and marginal.
    - `"certified"` compares the exact affine values represented by the stored
      floating-point operands using fixed-width integer arithmetic. Exact value is
      ordered first, followed by deterministic geometric and stable-index tie-breaks.
      NaNs and other non-finite or invalid geometry are refused rather than ordered as
      ordinary candidates. This mode requires pylcm's installed exact-affine payload
      for the active backend and fails before returning a certified result when that
      capability is unavailable. Select it when exact ownership is itself the object
      of a check.
    """
    probe_failure: Literal["reject", "assume_declared"] = "reject"
    """What to do when a derivative probe cannot evaluate the model.

    The affine-budget, interval-constancy, and single-power-flow probes
    differentiate the model's DAG functions on each draw selected by
    `probe_schedule`, reading that draw's parameter values and synthesizing the
    states and actions they sweep. A DAG that cannot be differentiated that way
    leaves the precondition unverified:

    - `"reject"` — refuse to solve; the per-interval EGM preconditions must be
      machine-verified.
    - `"assume_declared"` — warn and solve; the model author asserts the budget's
      within-interval affinity, every liquid-reading law's interval-constancy,
      and the consumption flow's single-power form, to be validated empirically.
    """
    probe_schedule: Literal["first_solve", "every_solve", "never"] = "never"
    """How often the parameter-dependent preconditions run.

    The affine-budget, interval-constancy, and single-power-flow probes
    differentiate the model's DAG against real parameter values, so they run at
    solve rather than at build. What they cost is therefore charged per solve,
    which an estimation loop pays on every criterion evaluation:

    - `"never"` (default) — skip all three, so a solve pays no probe cost; the
      model author asserts each precondition. A model that violates one is solved
      under a false structural assumption rather than refused.
    - `"first_solve"` — check all three once per model and reuse their verdicts.
      Use when parameter movement cannot change any precondition over the
      supported parameter domain.
    - `"every_solve"` — re-check every draw. Use while developing a model, or when
      a parameter can move a budget from affine to curved within an interval,
      switch a law from piecewise-constant to smoothly varying, or turn a
      single-power flow into another functional form.
    """

    def _with_liquid_margin(self, margin: _BoundLiquidMargin) -> _BoundNBEGM:
        """Bind regime-owned DAG names without exposing them on public `NBEGM`."""
        kwargs = {field.name: getattr(self, field.name) for field in fields(NBEGM)}
        return _BoundNBEGM(
            **kwargs,
            continuous_state=margin.state,
            continuous_action=margin.action,
            budget_target=margin.resources,
            post_decision_function=margin.post_decision_state,
            post_decision_bound=grid_origin_bound_spec(
                post_decision=margin.post_decision_state,
                liquid_state=margin.state,
                action=margin.action,
                resources_target=margin.resources,
                grid=self.savings_grid,
            ),
        )

    @property
    def required_continuation_keys(self) -> frozenset[ArtifactKey]:
        """The case-piece EGM step reads its continuation's marginal value."""
        return frozenset({EGM_CONTINUATION})

    @property
    def required_continuation_capabilities(self) -> ContinuationCapabilities:
        """The EGM step reads the target's value and its marginal in resources."""
        return ContinuationCapabilities(
            value=True, marginal_states=frozenset({EGM_ENDOGENOUS_COORDINATE})
        )

    @property
    def supports_nonlinear_certainty_equivalent(self) -> bool:
        """The case-piece step inverts the recursive Euler equation."""
        return True

    @property
    def egm_continuation_layout(self) -> EGMContinuationLayout:
        """The carry is maxed over the continuous action, on the liquid grid."""
        return EGMContinuationLayout(
            retains_discrete_action_rows=False,
            rows_share_state_grid=True,
        )

    @property
    def publishes_one_sided_jump_reads(self) -> bool:
        """One-sided jump resolution duplicates abscissae across each jump."""
        return self.jump_read == "one_sided"

    def build_constraint_routes(
        self, *, context: ConstraintRouteContext
    ) -> tuple[ConstraintRoute, ...]:
        """Declare savings proofs/boundaries and qualified current-node predicates.

        Only canonical model-owned predicate names may defer the boundary
        compiler. Their transitive leaves must fit the actual states, discrete
        actions, parameters and clock; continuous actions remain unavailable.
        """
        from _lcm.egm.nbegm_constraint_boundaries import (  # noqa: PLC0415
            build_nbegm_feasibility_boundary_compiler,
        )
        from _lcm.egm.nbegm_routes import case_piece_routes  # noqa: PLC0415

        bound = cast("_BoundNBEGM", self)
        evaluated_predicate_inputs = (
            (
                frozenset(context.variables.discrete_state_names)
                | frozenset(context.variables.continuous_state_names)
                | frozenset(context.variables.discrete_action_names)
                | context.flat_param_names
                | {"period", "age"}
            )
            if bound._evaluated_predicate_names  # noqa: SLF001
            else frozenset()
        )
        boundary_compilers = (
            ()
            if context.phase == "simulate"
            else (
                build_nbegm_feasibility_boundary_compiler(
                    liquid_state=bound.continuous_state,
                    evaluated_predicate_names=bound._evaluated_predicate_names,  # noqa: SLF001
                    evaluated_predicate_inputs=evaluated_predicate_inputs,
                ),
            )
        )
        return case_piece_routes(
            context=context,
            post_decision_function=proved_post_decision_of(solver=self),
            bound_spec=bound.post_decision_bound,
            solver_path=("nbegm",),
            boundary_compilers=boundary_compilers,
            evaluated_predicate_inputs=evaluated_predicate_inputs,
        )

    def validate_model(self, *, context: SolverModelContext) -> None:
        """Validate grids the kernel reads and the declared borrowing limit."""
        from _lcm.egm.validation import (  # noqa: PLC0415
            fail_if_declared_lower_bound_disagrees_with_the_grid,
            fail_if_kernel_grids_withhold_their_points,
        )

        user_regime = context.user_regimes[context.regime_name]
        bound = cast("_BoundNBEGM", self)
        liquid = bound.continuous_state
        fail_if_kernel_grids_withhold_their_points(
            grids={
                "savings grid": bound.savings_grid,
                f"grid of the liquid state '{liquid}'": cast(
                    "Grid", user_regime.states[liquid]
                ),
            },
            regime_name=context.regime_name,
            solver_name="NBEGM",
        )
        fail_if_declared_lower_bound_disagrees_with_the_grid(
            regime_name=context.regime_name,
            user_regime=user_regime,
            solver=bound,
            solver_name="NBEGM",
        )
        _validate_nbegm_case_piece_declarations(context=context, solver=self)

    def validate_build(self, *, context: SolverBuildContext) -> None:
        """Check build-only capabilities after declaration validation."""
        if self.envelope_arithmetic == "certified":
            _fail_if_exact_affine_kernel_unavailable(
                regime_name=context.regime_name,
                selection="certified NBEGM envelope arithmetic",
            )
        fail_if_taste_shocks_declared(context=context)

    def declare_continuation_reads(
        self, *, kernels: SolutionKernels, context: SolverBuildContext
    ) -> SolutionKernels:
        """Declare actual leaves on the single-liquid and adapted self-carry routes."""
        kernels = declare_egm_carry_reads(kernels=kernels, context=context)
        adapted: dict[
            tuple[int, int], tuple[CoreArgumentBuilder, MarginalLeafCore]
        ] = {}
        return replace(
            kernels,
            period_kernels=MappingProxyType(
                {
                    period: _with_nbegm_scalar_carry_reads(
                        kernel=_with_ride_marginal_reads(
                            kernel=kernel,
                            context=context,
                            period=period,
                            adapted=adapted,
                        ),
                        context=context,
                        period=period,
                    )
                    for period, kernel in kernels.period_kernels.items()
                }
            ),
        )

    def build_period_kernels(
        self,
        *,
        context: SolverBuildContext,
        materializer: DatedGenerationMaterializer | None = None,
    ) -> SolutionKernels:
        """Build one case-piece EGM period adapter per active period."""
        from _lcm.egm.nbegm import collect_nbegm_metadata  # noqa: PLC0415

        bound = cast("_BoundNBEGM", self)
        representative_period = context.regimes_to_active_periods[context.regime_name][
            0
        ]
        representative_context = resolve_solver_build_context(
            context=context, period=representative_period
        )
        feasibility_constraints = _consume_nbegm_feasibility_constraints(
            context=context,
            solver_path=("nbegm",),
        )
        savings_grid = (
            self.savings_grid.to_jax()
            if materializer is None
            else materializer.materialize_grid(spec=self.savings_grid)
        )

        functions = context.solve_functions
        registry = collect_nbegm_metadata(functions=functions)
        has_discrete = bool(context.state_action_space.discrete_actions)
        has_ride_along = self._schedule_has_ride_along(context=context)
        _fail_if_feasibility_composition_is_unsupported(
            constraints=feasibility_constraints,
            registry=registry,
            jump_read=self.jump_read,
            has_discrete=has_discrete,
            regime_name=context.regime_name,
        )
        # No declared case pieces routes to the schedule path. With no declared
        # piecewise-affine schedules either, the partition is empty — a single
        # interval covering the whole liquid axis, solved as plain EGM — so a
        # declaration-free budget is in scope. A declaration-free regime with a
        # discrete action takes the dedicated single-liquid discrete path below,
        # unless it also carries a ride-along co-state: only the schedule path's
        # kernels represent a second state axis, and the partition being
        # degenerate does not change which axes the branch envelope must span.
        has_schedule = not registry.piece_sets and (
            bool(registry.piecewise_affine_schedules)
            or not has_discrete
            or has_ride_along
        )
        # A discrete action over a cliffed single-liquid budget composes the
        # discrete upper envelope with the schedule's per-branch intervals.
        # Alongside a ride-along co-state the branch envelope instead runs per
        # ride cell, which is the ride-along route's own composition — admitted
        # subject to the guard it applies below.
        is_schedule_discrete = has_schedule and has_discrete and not has_ride_along
        is_schedule = has_schedule and not is_schedule_discrete
        is_discrete = not has_schedule and not registry.piece_sets and has_discrete
        schedule_discrete_spec = (
            _collect_nbegm_schedule_discrete_spec(
                context=representative_context,
                budget_target=bound.budget_target,
                continuous_state=bound.continuous_state,
                post_decision_function=bound.post_decision_function,
                probe_failure=self.probe_failure,
                probe_schedule=self.probe_schedule,
            )
            if is_schedule_discrete
            else None
        )
        schedule_spec = (
            _collect_nbegm_schedule_spec(
                context=representative_context,
                budget_target=bound.budget_target,
                continuous_state=bound.continuous_state,
                consumption_action_name=bound.continuous_action,
                probe_failure=self.probe_failure,
                probe_schedule=self.probe_schedule,
                materializer=materializer,
            )
            if is_schedule
            else None
        )
        if schedule_spec is not None and schedule_spec.ride_along_state_names:
            if context.state_action_space.discrete_actions:
                self._fail_if_unsupported_ride_discrete(
                    context=context, schedule_spec=schedule_spec
                )
            return self._build_ride_along_kernels(
                context=context,
                savings_grid=savings_grid,
                schedule_spec=schedule_spec,
                feasibility_constraints=feasibility_constraints,
                materializer=materializer,
            )

        if materializer is not None:
            raise UnsupportedOperationError(
                "Owned nested NB requires its original ride-along schedule."
            )
        return self._build_single_liquid_kernels(
            context=context,
            representative_context=representative_context,
            feasibility_constraints=feasibility_constraints,
            savings_grid=savings_grid,
            schedule_spec=schedule_spec,
            schedule_discrete_spec=schedule_discrete_spec,
            is_schedule=is_schedule,
            is_discrete=is_discrete,
        )

    def _build_single_liquid_kernels(
        self,
        *,
        context: SolverBuildContext,
        representative_context: SolverBuildContext,
        feasibility_constraints: tuple[_NBEGMFeasibilityConstraint, ...],
        savings_grid: Float1D,
        schedule_spec: _NBEGMScheduleSpec | None,
        schedule_discrete_spec: _NBEGMScheduleDiscreteSpec | None,
        is_schedule: bool,
        is_discrete: bool,
    ) -> SolutionKernels:
        """Build the native case/discrete/schedule kernels on one liquid state."""
        from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
            periodized_tree_signature,
        )

        bound = cast("_BoundNBEGM", self)
        # Every route below solves the additive expected-utility step; only the
        # ride-along route above carries the Epstein-Zin kernels. Reject a
        # declared certainty equivalent here rather than silently solving the
        # additive recursion the regime did not declare.
        if _aggregates_nonlinearly(context.certainty_equivalent):
            msg = (
                f"Regime {context.regime_name!r} declares a "
                "`certainty_equivalent` but has no ride-along state, so NBEGM "
                "would route it to the additive expected-utility step. The "
                "Epstein-Zin kernels run on the ride-along route only; use "
                "GridSearch() for a single-liquid-state recursive regime."
            )
            raise RegimeInitializationError(msg)

        # Every remaining route (case pieces, discrete envelope, schedule+discrete)
        # carries the liquid axis alone, so the resolver also refuses a second state
        # rather than letting it surface as a missing-parameter lookup inside the
        # traced core.
        liquid_state_name = (
            schedule_spec.liquid_state_name
            if schedule_spec is not None
            else _single_liquid_state_name(
                context=context,
                declared=bound.continuous_state,
                path="single-liquid-state kernels",
            )
        )
        post_decision_name = bound.post_decision_function
        fail_if_liquid_law_is_not_written_through_savings(
            context=representative_context,
            liquid_state_name=liquid_state_name,
            post_decision_name=post_decision_name,
        )
        liquid_grid = context.grids[liquid_state_name].to_jax()
        discrete_spec = (
            _collect_nbegm_discrete_spec(
                context=representative_context,
                budget_target=bound.budget_target,
                post_decision_function=bound.post_decision_function,
                continuous_state=bound.continuous_state,
                probe_failure=self.probe_failure,
                probe_schedule=self.probe_schedule,
            )
            if is_discrete
            else None
        )
        case_spec = (
            _collect_nbegm_case_spec(
                context=representative_context,
                continuous_state=bound.continuous_state,
            )
            if not is_schedule and not is_discrete and schedule_discrete_spec is None
            else None
        )
        _fail_if_budget_node_differs_from_kernel_cash_on_hand(
            context=context,
            routes_to_case_piece_core=(
                case_spec is not None
                and schedule_discrete_spec is None
                and schedule_spec is None
                and discrete_spec is None
            ),
            budget_target=bound.budget_target,
            liquid_state_name=liquid_state_name,
        )

        period_to_target = period_to_continuation_target(context=context)
        current_model = context.nbegm_source_query_model or (
            context.nbegm_query_models or {}
        ).get(context.regime_name)
        plans = {
            period: _build_nbegm_continuation_plan(
                context=context, period=period, post_decision_name=post_decision_name
            )
            for period, target in period_to_target.items()
            if current_model is not None
            or target in (context.dcegm_query_models or {})
            or target in (context.nbegm_query_models or {})
        }
        query_slots_by_period = {}
        reader_capabilities = {}
        query_template = (
            None
            if current_model is None
            else _build_nbegm_query_carry_template(
                context=context, model=current_model, retain_policy=False
            )
        )
        cores: dict[Hashable, OneRowCore] = {}
        checked_cores: dict[Hashable, OneRowCore] = {}
        laws: dict[Hashable, Callable[..., tuple[Float1D, Float1D]]] = {}
        period_kernels: dict[int, PeriodKernel] = {}
        period_group_keys = {
            period: (
                solver_period_group_key(
                    context=context,
                    period=period,
                    continuation_targets=(target,),
                    solver_path=("nbegm",),
                ),
                *(
                    (
                        (
                            "query_targets",
                            tuple(
                                (
                                    child,
                                    periodized_tree_signature(
                                        tree=_required_nbegm_child_query(
                                            plan=plans[period], target=child
                                        ).model.functions,
                                        period=period + 1,
                                    ),
                                    periodized_tree_signature(
                                        tree={
                                            "koopmans_aggregator": (
                                                _nbegm_child_koopmans(
                                                    plan=plans[period], target=child
                                                )
                                            )
                                        },
                                        period=period + 1,
                                    ),
                                )
                                for child in plans[period].query_targets
                            ),
                        ),
                    )
                    if period in plans and plans[period].query_targets
                    else ()
                ),
            )
            if period in plans and plans[period].query_targets
            else solver_period_group_key(
                context=context,
                period=period,
                continuation_targets=(target,),
                solver_path=("nbegm",),
            )
            for period, target in period_to_target.items()
        }
        variable_names = (
            frozenset(context.state_action_space.states)
            | frozenset(context.state_action_space.continuous_actions)
            | frozenset(context.state_action_space.discrete_actions)
        )
        route: Literal["schedule_discrete", "schedule", "discrete", "case"] = (
            "schedule_discrete"
            if schedule_discrete_spec is not None
            else "schedule"
            if schedule_spec is not None
            else "discrete"
            if discrete_spec is not None
            else "case"
        )
        grouped_param_checks: list[ParamCheck] = []
        n_boundaries = sum(
            len(constraint.program.surfaces) for constraint in feasibility_constraints
        )
        for period, target in period_to_target.items():
            plan = plans.get(period)
            child_slots = _record_nbegm_scalar_child_slots(
                plan=plan,
                period=period,
                query_slots_by_period=query_slots_by_period,
                reader_capabilities=reader_capabilities,
            )
            group_key = period_group_keys[period]
            if group_key not in cores:
                resolved = restrict_solver_build_context_to_period_group(
                    context=resolve_solver_build_context(
                        context=context, period=period
                    ),
                    periods=tuple(
                        candidate
                        for candidate, key in period_group_keys.items()
                        if key == group_key
                    ),
                )
                core, law, checks = _build_nbegm_owned_single_axis_group(
                    context=resolved,
                    solver=bound,
                    route=route,
                    target=target,
                    liquid_state_name=liquid_state_name,
                    variable_names=variable_names,
                    savings_grid=savings_grid,
                    feasibility_constraints=feasibility_constraints,
                    current_model=current_model,
                    plan=plan,
                    has_child_slots=bool(child_slots),
                    period=period,
                )
                laws[group_key] = law
                grouped_param_checks.extend(checks)
                if child_slots:
                    if not isinstance(core, _NBEGMOwnedSingleAxisCore):
                        raise TypeError(
                            "An NB checked child reader "
                            "requires its genuine owning core."
                        )
                    checked_cores[group_key] = guard_regime_mass(
                        core=replace(core, collect_continuation_read_counts=True),
                        enable_jit=context.enable_jit,
                    )
                cores[group_key] = guard_regime_mass(
                    core=core, enable_jit=context.enable_jit
                )
            current_floor = _build_nbegm_source_floor(
                context=resolve_solver_build_context(context=context, period=period),
                bound=bound.post_decision_bound,
                savings_grid=savings_grid,
            )
            kernel = _build_egm_period_kernel(
                core=cores[group_key],
                declared_law=laws[group_key],
                compute_regime_transition_probs=context.compute_regime_transition_probs,
                savings_grid=savings_grid,
                regime_name=context.regime_name,
                continuation_target=target,
                liquid_state=liquid_state_name,
                transition_target_names=tuple(context.transitions),
                next_liquid_grid=target_period_grid(
                    context=context,
                    period=period,
                    target=target,
                    target_state_name=liquid_state_name,
                ),
                publishes_breakpoints=n_boundaries > 0,
                query_template=None
                if query_template is None
                else query_template.query_data,
                own_bound_specs=_nbegm_own_bound_specs(
                    floor=current_floor, bound=bound.post_decision_bound
                ),
                source_period=period if current_floor is not None or child_slots else 0,
                query_input_authorities=child_slots or None,
                **(
                    {"checked_core": checked_cores[group_key]}
                    if group_key in checked_cores
                    else {}
                ),
            )
            if current_floor is not None or child_slots:
                kernel = _with_nbegm_source_arguments(
                    kernel=kernel,
                    current_floor=current_floor,
                    owned_child_reads=bool(child_slots),
                )
            if current_floor is not None or child_slots:
                kernel = _with_nbegm_scalar_output_roles(
                    kernel=kernel,
                    liquid_grid=liquid_grid,
                    carry_template=query_template
                    if query_template is not None
                    else _build_nbegm_feasibility_carry_template(
                        liquid_grid=liquid_grid, n_boundaries=n_boundaries
                    ),
                    own_bound_count=int(current_floor is not None),
                    query_status_count=len(child_slots),
                )
            period_kernels[period] = kernel
        return SolutionKernels(
            period_kernels=MappingProxyType(period_kernels),
            period_group_keys=MappingProxyType(period_group_keys),
            continuation_spec=EGMContinuationSpec(
                template=_build_nbegm_feasibility_carry_template(
                    liquid_grid=liquid_grid,
                    n_boundaries=n_boundaries,
                )
                if query_template is None
                else query_template,
                layout=replace(
                    self.egm_continuation_layout,
                    rows_share_state_grid=n_boundaries == 0 and current_model is None,
                    query_spec=None
                    if current_model is None
                    else current_model.authority,
                    retains_discrete_action_rows=False
                    if current_model is None
                    else bool(current_model.authority.action_names),
                ),
            ),
            query_input_authorities=query_slots_by_period or None,
            continuation_reader_capabilities=reader_capabilities or None,
            param_checks=tuple(grouped_param_checks),
        )

    def _fail_if_unsupported_ride_discrete(
        self, *, context: SolverBuildContext, schedule_spec: _NBEGMScheduleSpec
    ) -> None:
        """Reject a ride-along discrete action the envelope path cannot handle.

        The ride-along discrete envelope solves the continuous subproblem per
        discrete branch — with the action bound into the budget, period utility,
        discount factor, continuation (co-state laws, off-budget liquid law,
        regime transition), and the breakpoint partition — and takes the upper
        envelope. Outside that contract, model build refuses an action entering
        a *jumped* schedule variable under the one-sided cliff read (branches
        would not share the published parent query grid).

        Several discrete actions are supported: the branch axis is the product of
        their grids, and every branch binds a code for each. A jump breakpoint is
        likewise supported — each branch publishes its one-sided cliff limits and
        the envelope takes the max over branches.
        """
        bound = cast("_BoundNBEGM", self)
        action_names = tuple(context.state_action_space.discrete_actions)
        for action_name in action_names:
            # The envelope binds each action into every branch's period utility and
            # discount factor, so an action either reads is supported (a
            # leisure/effort-like term, a branch-specific patience).
            _fail_if_discrete_action_feeds_continuation(
                context=context,
                action_name=action_name,
                liquid_state_name=schedule_spec.liquid_state_name,
                budget_target=bound.budget_target,
                post_decision_function=bound.post_decision_function,
                allow_continuation_feed=True,
            )
        # An action entering a schedule variable gives each branch its own
        # breakpoint partition (its own asset preimage of every threshold), which
        # the envelope solves per branch. Publishing a jump is the exception: the
        # one-sided cliff limits ride a jump-augmented query grid built once,
        # outside the per-branch loop, so *any* source whose variable the action
        # shifts would move that grid's extra abscissae per branch and the
        # branches' rows would no longer share a grid to take the discrete max
        # over. The kink sources are not spared: they are augmented on the same
        # grid, and the cell-breakpoint DAG the augmentation evaluates is called
        # without an action binding there.
        publishes_jumps = self.jump_read == "one_sided" and any(
            source.kind == "jump" for source in schedule_spec.sources
        )
        if not publishes_jumps:
            return
        for source in schedule_spec.sources:
            dag = source.derived_of_liquid_dag
            if dag is None:
                continue
            source_args = frozenset(inspect.signature(dag).parameters)
            shifting = tuple(name for name in action_names if name in source_args)
            if not shifting:
                continue
            msg = (
                "NBEGM's schedule+ride-along discrete envelope publishes the jump "
                f"breakpoint on a shared query grid, so the actions {shifting} "
                f"must not enter any schedule variable — regime "
                f"{context.regime_name!r} reads it in {source.variable!r} (a "
                f"{source.kind} source), whose breakpoints would then sit at a "
                "different liquid per branch. Use `jump_read='bridged'`, or "
                "declare the schedule on a variable the action does not shift."
            )
            raise RegimeInitializationError(msg)

    def _schedule_has_ride_along(self, *, context: SolverBuildContext) -> bool:
        """Whether the schedule regime carries a ride-along co-state.

        A ride-along axis is any state other than the liquid (Euler) axis. The
        Euler axis is `continuous_state` when named, else the regime's single
        continuous state; discrete actions are not states and never ride along.
        """
        space = context.state_action_space
        liquid_state_name = cast("_BoundNBEGM", self).continuous_state
        return any(name != liquid_state_name for name in space.state_names)

    def _build_ride_along_kernels(  # noqa: PLR0915 - original ride constructor and transient admission
        self,
        *,
        context: SolverBuildContext,
        savings_grid: Float1D,
        schedule_spec: _NBEGMScheduleSpec,
        feasibility_constraints: tuple[_NBEGMFeasibilityConstraint, ...] = (),
        materializer: DatedGenerationMaterializer | None = None,
    ) -> SolutionKernels:
        """Build the case-piece kernels for a regime carrying a ride-along co-state.

        The continuation is read through the transition-aware reader, so each
        period's plan depends on its reachable carry/scalar target split; cores
        are deduplicated by that split. Each period's kernel publishes a native
        two-program graph; both programs run the tile-local core that reads a
        cell block's continuation and runs its 1-D liquid solve before the next
        block.
        """
        from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
            periodized_tree_signature,
        )

        bound = cast("_BoundNBEGM", self)

        liquid_grid = (
            context.grids[schedule_spec.liquid_state_name].to_jax()
            if materializer is None
            else materializer.materialize_grid(
                spec=context.grids[schedule_spec.liquid_state_name]
            )
        )
        ride_shape = tuple(
            int(
                (
                    context.grids[name].to_jax()
                    if materializer is None
                    else materializer.materialize_grid(spec=context.grids[name])
                ).shape[0]
            )
            for name in schedule_spec.ride_along_state_names
        )
        query_model = context.nbegm_source_query_model or (
            context.nbegm_query_models or {}
        ).get(context.regime_name)
        if materializer is not None and query_model is not None:
            raise UnsupportedOperationError(
                "Owned nested NB requires original query-free rows."
            )
        query_template = (
            None
            if query_model is None
            else _build_nbegm_query_carry_template(context=context, model=query_model)
        )
        query_slots_by_period = {}
        reader_capabilities = {}
        param_checks: list[ParamCheck] = []
        transition_target_names = tuple(context.transitions)

        # The ride-along kernel takes the continuation as a probability-weighted
        # blend over the full reachable target set (`bind_continuation` sums the
        # per-target carries by `compute_regime_transition_probs`), so it admits a
        # stochastic multi-target lifecycle transition. Enumerate the regime's
        # active periods directly rather than resolving a single target per period.
        active_periods = sorted(context.regimes_to_active_periods[context.regime_name])
        programs_by_key: dict[
            _RideAlongGroupKey, MappingProxyType[str, CoreProgram]
        ] = {}
        statics_by_key: dict[_RideAlongGroupKey, _NBEGMRideAlongStatics] = {}
        cliff_candidates_by_key: dict[_RideAlongGroupKey, bool] = {}
        period_kernels: dict[int, PeriodKernel] = {}
        contexts = {
            period: resolve_solver_build_context(context=context, period=period)
            for period in active_periods
        }
        plans = {
            period: _build_nbegm_continuation_plan(
                context=resolved,
                period=period,
                post_decision_name=bound.post_decision_function,
            )
            for period, resolved in contexts.items()
        }
        # Probes and compiled cores share exactly the same resolved period group.
        period_group_keys: dict[int, _RideAlongGroupKey] = {
            period: (
                *plan.stateful_targets,
                "|",
                *plan.scalar_targets,
                continuation_grid_signature(
                    period=period,
                    targets=plan.stateful_targets + plan.scalar_targets,
                    period_to_regime_grid_signature=(
                        context.period_to_regime_grid_signature
                    ),
                ),
                solver_period_group_key(
                    context=context,
                    period=period,
                    continuation_targets=(plan.stateful_targets + plan.scalar_targets),
                    solver_path=("nbegm", "ride_along"),
                ),
                *(
                    (
                        (
                            "query_targets",
                            tuple(
                                (
                                    target,
                                    periodized_tree_signature(
                                        tree=_required_nbegm_child_query(
                                            plan=plan, target=target
                                        ).model.functions,
                                        period=period + 1,
                                    ),
                                    periodized_tree_signature(
                                        tree={
                                            "koopmans_aggregator": (
                                                _nbegm_child_koopmans(
                                                    plan=plan, target=target
                                                )
                                            )
                                        },
                                        period=period + 1,
                                    ),
                                )
                                for target in plan.query_targets
                            ),
                        ),
                    )
                    if plan.query_targets
                    else ()
                ),
            )
            for period, plan in plans.items()
        }
        for period in active_periods:
            key = period_group_keys[period]
            resolved = restrict_solver_build_context_to_period_group(
                context=contexts[period],
                periods=tuple(
                    candidate
                    for candidate, candidate_key in period_group_keys.items()
                    if candidate_key == key
                ),
            )
            plan = plans[period]
            child_slots = tuple(
                (
                    target,
                    period + 1,
                    _required_nbegm_child_query(
                        plan=plan, target=target
                    ).model.authority.bound,
                )
                for target in plan.query_targets
            )
            if child_slots:
                query_slots_by_period[period] = child_slots
                reader_capabilities[period] = MappingProxyType(
                    {
                        target: _required_nbegm_child_query(
                            plan=plan, target=target
                        ).capabilities
                        for target in plan.query_targets
                    }
                )
            current_floor = _build_nbegm_source_floor(
                context=resolved,
                bound=bound.post_decision_bound,
                savings_grid=savings_grid,
            )
            group_spec = _collect_nbegm_schedule_spec(
                context=resolved,
                budget_target=bound.budget_target,
                continuous_state=bound.continuous_state,
                consumption_action_name=bound.continuous_action,
                probe_failure=self.probe_failure,
                probe_schedule=self.probe_schedule,
                materializer=materializer,
            )
            group_constraints = tuple(
                replace(
                    constraint,
                    predicate=resolved.constraint_functions[
                        constraint.program.constraint_name
                    ],
                )
                for constraint in feasibility_constraints
            )
            if key not in programs_by_key:
                probe_arguments = _probe_arguments(context=resolved)
                param_checks.extend(group_spec.param_checks)
                if _aggregates_nonlinearly(resolved.certainty_equivalent):
                    param_checks.append(
                        _deferred_probe(
                            probe=_fail_if_flow_not_single_power,
                            regime_name=context.regime_name,
                            probe_arguments=probe_arguments,
                            utility_dag=group_spec.utility_dag,
                            consumption_action_name=bound.continuous_action,
                            probe_failure=self.probe_failure,
                            probe_schedule=self.probe_schedule,
                        )
                    )
                param_checks.append(
                    _deferred_probe(
                        probe=_fail_if_liquid_reading_next_state_varies_within_interval,
                        regime_name=context.regime_name,
                        probe_arguments=probe_arguments,
                        continuation_plan=plan,
                        liquid_name=group_spec.liquid_state_name,
                        probe_failure=self.probe_failure,
                        probe_schedule=self.probe_schedule,
                    )
                )
                statics = _nbegm_ride_along_statics(
                    savings_grid=savings_grid,
                    schedule_spec=group_spec,
                    continuation_plan=plan,
                    envelope_arithmetic=self.envelope_arithmetic,
                    publish_jump_topology=self.jump_read == "one_sided",
                    co_map_state_names=resolved.co_map_state_names,
                    materializer=materializer,
                )
                # The Epstein-Zin kernels cover smooth and pure-kink budgets;
                # the unified jump-and-kink candidate step is additive. Reject
                # the combination here, at model build, rather than midway
                # through a traced solve.
                if (
                    _aggregates_nonlinearly(resolved.certainty_equivalent)
                    and statics.has_jump
                ):
                    msg = (
                        f"Regime {context.regime_name!r} declares a "
                        "`certainty_equivalent` and a current-period jump "
                        "breakpoint. Epstein-Zin NBEGM covers smooth and "
                        "pure-kink budgets only; use a kink-only schedule or "
                        "GridSearch() for this regime."
                    )
                    raise RegimeInitializationError(msg)
                # The per-interval candidate step evaluates every interior and
                # corner candidate with the additive expected-utility recursion;
                # combining it with a certainty equivalent would compare
                # candidates under the wrong objective. Reject at model build.
                if (
                    _aggregates_nonlinearly(resolved.certainty_equivalent)
                    and statics.continuation_reads_liquid
                ):
                    msg = (
                        f"Regime {context.regime_name!r} declares a "
                        "`certainty_equivalent` while its continuation depends "
                        "on the current liquid state (a next-state law or the "
                        "regime-transition probabilities read it). The "
                        "per-interval candidate step is additive; Epstein-Zin "
                        "NBEGM requires a continuation that is independent of "
                        "the current liquid state. Keep the laws of motion and "
                        "transition probabilities free of the liquid state, or "
                        "use GridSearch() for this regime."
                    )
                    raise RegimeInitializationError(msg)
                # Save-to-cliff candidates need the regime's own carry read
                # (the cliffs are the self-schedule's); a period whose targets
                # exclude the regime itself solves without them.
                cliff_candidates = (
                    statics.n_published_jumps > 0
                    and context.regime_name in plan.child_reads
                )
                envelope_build = _build_nbegm_envelope_core(
                    savings_grid=savings_grid,
                    schedule_spec=group_spec
                    if materializer is None
                    else replace(group_spec, param_checks=()),
                    statics=statics,
                    materializer=materializer,
                    is_epstein_zin=_aggregates_nonlinearly(
                        context.certainty_equivalent
                    ),
                    feasibility_constraints=group_constraints,
                    query_model=query_model,
                    current_floor=current_floor,
                    current_predicates=(
                        _nbegm_current_predicates(
                            context=resolved, bound=bound.post_decision_bound
                        )
                        if current_floor is not None or plan.query_targets
                        else None
                    ),
                )
                programs_by_key[key] = _ride_along_core_programs(
                    savings_grid=savings_grid,
                    continuation_plan=plan,
                    statics=statics,
                    regime_name=context.regime_name,
                    cliff_candidates=cliff_candidates,
                    schedule_spec=group_spec
                    if materializer is None
                    else replace(group_spec, param_checks=()),
                    envelope_build=envelope_build,
                    state_action_space=resolved.state_action_space,
                    argument_builder=_RideAlongArgumentBuilder(
                        regime_name=context.regime_name,
                        transition_target_names=transition_target_names,
                        stateful_targets=frozenset(plan.stateful_targets),
                        co_map_state_names=statics.co_map_state_names,
                        n_intervals=(
                            statics.n_intervals
                            if statics.continuation_reads_liquid
                            else 0
                        ),
                    ),
                    enable_jit=context.enable_jit,
                    query_template=query_template,
                    materializer=materializer,
                )
                statics_by_key[key] = statics
                cliff_candidates_by_key[key] = cliff_candidates
            period_kernels[period] = _RideAlongNBEGMPeriodKernel(
                _core_programs=programs_by_key[key],
                statics=statics_by_key[key],
                cliff_candidates=cliff_candidates_by_key[key],
                regime_name=context.regime_name,
                stateful_targets=frozenset(plan.stateful_targets),
                transition_target_names=transition_target_names,
                schedule_spec=group_spec,
                continuation_plan=plan,
                savings_grid=savings_grid,
                own_bound_specs=_nbegm_own_bound_specs(
                    floor=current_floor, bound=bound.post_decision_bound
                ),
                query_input_authorities=child_slots or None,
                source_period=period if current_floor is not None or child_slots else 0,
                retain_nested_feasibility=query_model is not None
                and query_model.retain_nested_feasibility,
            )
        n_carry_breakpoints = (
            next(iter(statics_by_key.values())).n_published_jumps
            if statics_by_key
            else 0
        ) + sum(
            len(constraint.program.surfaces) for constraint in feasibility_constraints
        )
        return SolutionKernels(
            continuation_reader_capabilities=reader_capabilities or None,
            query_input_authorities=query_slots_by_period or None,
            period_kernels=MappingProxyType(period_kernels),
            period_group_keys=MappingProxyType(period_group_keys),
            continuation_spec=EGMContinuationSpec(
                template=query_template
                if query_template is not None
                else _shard_ride_carry_template(
                    template=_build_ride_along_carry_template(
                        materializer=materializer,
                        liquid_grid=liquid_grid,
                        ride_shape=ride_shape,
                        n_breakpoints=n_carry_breakpoints,
                        # Match `_assemble_ride_carry`'s carry_policy predicate:
                        # continuous-only (no ride discrete action) and jump-free.
                        carry_policy=(
                            not schedule_spec.discrete_actions
                            and n_carry_breakpoints == 0
                        ),
                    ),
                    grids=context.grids,
                    ride_along_state_names=schedule_spec.ride_along_state_names,
                    sharded_state_names=context.sharded_state_names,
                    submesh_device_ids=context.submesh_device_ids,
                ),
                layout=replace(
                    self.egm_continuation_layout,
                    rows_share_state_grid=n_carry_breakpoints == 0
                    if query_model is None
                    else False,
                    retains_discrete_action_rows=bool(
                        query_model.authority.action_names
                    )
                    if query_model is not None
                    else False,
                    query_spec=None if query_model is None else query_model.authority,
                ),
            ),
            param_checks=tuple(param_checks),
        )


@dataclass(frozen=True)
class _NBEGMFeasibilityConstraint:
    """One compiled boundary paired with its phase-processed predicate."""

    program: NBEGMFeasibilityBoundaryProgram
    """Solver-facing boundary geometry."""

    predicate: Callable[..., BoolND]
    """Executable predicate generated from the same declaration."""


def _compiled_boundaries_for_inner(
    *, plan: ConstraintPlan, solver_path: tuple[str, ...]
) -> tuple[CompileBoundary, ...]:
    """Select a direct route or an already-filtered rewritten inner plan."""
    selected = plan.for_solver_path(solver_path=solver_path).compiled_boundaries
    if selected:
        return selected

    route_paths = {entry.route.solver_path for entry in plan.entries}
    return plan.compiled_boundaries if len(route_paths) == 1 else ()


def _consume_nbegm_feasibility_constraints(
    *,
    context: SolverBuildContext,
    solver_path: tuple[str, ...],
) -> tuple[_NBEGMFeasibilityConstraint, ...]:
    """Pair every compiled program for one builder with its predicate exactly once."""
    plan = context.constraint_plan
    if plan is None:
        return ()

    selected = _compiled_boundaries_for_inner(plan=plan, solver_path=solver_path)
    all_compiled = plan.compiled_boundaries
    if len(selected) != len(all_compiled):
        msg = (
            f"NBEGM for regime {context.regime_name!r} received compiled "
            "constraint boundaries belonging to another solver route; filter the "
            "constraint plan before constructing the inner solver context."
        )
        raise RegimeInitializationError(msg)

    names = tuple(disposition.constraint.name for disposition in selected)
    if len(names) != len(set(names)):
        msg = (
            f"NBEGM for regime {context.regime_name!r} would consume a compiled "
            f"constraint more than once; got route entries {names}."
        )
        raise RegimeInitializationError(msg)

    consumed: list[_NBEGMFeasibilityConstraint] = []
    for disposition in selected:
        if not isinstance(disposition, CompileBoundary):
            msg = "Internal error: selected a non-boundary constraint disposition."
            raise TypeError(msg)
        payload = disposition.program.payload
        if not isinstance(payload, NBEGMFeasibilityBoundaryProgram):
            msg = (
                f"NBEGM for regime {context.regime_name!r} received an unsupported "
                f"compiled payload for constraint {disposition.constraint.name!r}."
            )
            raise RegimeInitializationError(msg)
        if payload.constraint_name != disposition.constraint.name:
            msg = (
                f"NBEGM for regime {context.regime_name!r} received a boundary "
                f"program named {payload.constraint_name!r} for constraint "
                f"{disposition.constraint.name!r}."
            )
            raise RegimeInitializationError(msg)
        predicate = context.constraint_functions.get(payload.constraint_name)
        if predicate is None:
            msg = (
                f"NBEGM for regime {context.regime_name!r} received compiled "
                f"constraint {payload.constraint_name!r} without its executable "
                "phase-processed predicate."
            )
            raise RegimeInitializationError(msg)
        consumed.append(
            _NBEGMFeasibilityConstraint(program=payload, predicate=predicate)
        )

    return tuple(consumed)


def _fail_if_feasibility_composition_is_unsupported(
    *,
    constraints: tuple[_NBEGMFeasibilityConstraint, ...],
    registry: NBEGMRegistry,
    jump_read: Literal["one_sided", "bridged"],
    has_discrete: bool,
    regime_name: RegimeName,
) -> None:
    """Reject boundary combinations that need another one-sided topology."""
    if not constraints:
        return
    if jump_read == "bridged":
        msg = (
            f"Regime {regime_name!r} compiles a feasibility boundary, which "
            "requires NBEGM's one-sided carry; `jump_read='bridged'` would "
            "interpolate across it."
        )
        raise RegimeInitializationError(msg)
    if registry.piece_sets:
        msg = (
            f"Regime {regime_name!r} composes a compiled feasibility boundary "
            "with binary case pieces. NBEGM does not yet combine those two "
            "one-sided boundary topologies."
        )
        raise RegimeInitializationError(msg)

    schedule_kinds = tuple(
        breakpoint_meta.kind
        for schedule in registry.piecewise_affine_schedules
        for breakpoint_meta in schedule.breakpoints
    )
    if "jump" in schedule_kinds:
        msg = (
            f"Regime {regime_name!r} composes a compiled feasibility boundary "
            "with a finite-value schedule jump. NBEGM supports feasibility with "
            "smooth budgets, continuous kinks, and flat-budget floors only."
        )
        raise RegimeInitializationError(msg)
    if has_discrete:
        msg = (
            f"Regime {regime_name!r} composes a compiled feasibility boundary "
            "with a discrete action. NBEGM's feasibility topology supports the "
            "liquid axis, including ride-along cells, but not a further discrete "
            "branch envelope."
        )
        raise RegimeInitializationError(msg)


@dataclass(frozen=True)
class _ResolvedNBEGMFeasibility:
    """Runtime feasibility geometry shared by the solve and its carry."""

    partition: ResolvedAxisPartition
    """Combined schedule and feasibility partition on the liquid axis."""

    feasible_interval_mask: BoolND
    """Whether every interval satisfies every compiled predicate."""

    boundary_values: Float1D
    """Topology-relevant feasibility thresholds, sorted and NaN-padded."""
    carry_boundary_values: Float1D
    """Distinct topology-relevant thresholds published to the parent."""

    boundary_owner_is_right: BoolND
    """Equality ownership aligned with `boundary_values`."""


def _resolve_nbegm_feasibility(
    *,
    constraints: tuple[_NBEGMFeasibilityConstraint, ...],
    liquid: Float1D,
    schedule_breakpoints: Float1D,
    schedule_kinds: tuple[str, ...],
    params: Mapping[str, FloatND | IntND | BoolND],
) -> _ResolvedNBEGMFeasibility:
    """Resolve one owner-aware partition and evaluate every interval predicate."""
    schedule_sources = tuple(
        AxisBoundary(
            value=value,
            owner="right",
            effect=partition_effect_for_schedule_kind(
                cast(
                    "Literal['continuous_kink', 'jump', 'hard_constraint']",
                    kind,
                )
            ),
        )
        for value, kind in zip(schedule_breakpoints, schedule_kinds, strict=True)
    )
    feasibility_sources = feasibility_axis_boundaries(
        programs=tuple(constraint.program for constraint in constraints),
        params=params,
    )
    partition = resolve_axis_partition(
        start=liquid[0],
        stop=liquid[-1],
        boundaries=(*schedule_sources, *feasibility_sources),
    )

    representatives = 0.5 * partition.effective_starts + 0.5 * partition.effective_stops
    feasible = jnp.ones_like(representatives, dtype=jnp.bool_)
    for constraint in constraints:
        arguments = {
            name: (
                representatives
                if name == constraint.program.liquid_state
                else params[name]
            )
            for name in inspect.signature(constraint.predicate).parameters
        }
        feasible = feasible & jnp.asarray(
            constraint.predicate(**arguments),
            dtype=jnp.bool_,
        )

    if feasibility_sources:
        unsorted_values = jnp.stack(
            [
                jnp.asarray(source.value, dtype=liquid.dtype)
                for source in feasibility_sources
            ]
        )
        unsorted_owner_is_right = jnp.asarray(
            [source.owner == "right" for source in feasibility_sources],
            dtype=jnp.bool_,
        )
        order = canonical_boundary_order(
            values=unsorted_values,
            owner_is_right=unsorted_owner_is_right,
        )
        boundary_values = unsorted_values[order]
        owner_is_right = unsorted_owner_is_right[order]
        open_side = jnp.where(
            owner_is_right,
            jnp.nextafter(boundary_values, -jnp.inf),
            jnp.nextafter(boundary_values, jnp.inf),
        )
        boundary_in_domain = (boundary_values >= liquid[0]) & (
            boundary_values <= liquid[-1]
        )
        open_side_in_domain = (open_side >= liquid[0]) & (open_side <= liquid[-1])
        topology_relevant = boundary_in_domain & open_side_in_domain
        previous_is_relevant_at_same_location = jnp.concatenate(
            (
                jnp.zeros((1,), dtype=jnp.bool_),
                topology_relevant[:-1] & (boundary_values[1:] == boundary_values[:-1]),
            )
        )
        first_relevant_at_location = (
            topology_relevant & ~previous_is_relevant_at_same_location
        )
        boundary_values = jnp.where(topology_relevant, boundary_values, jnp.nan)
        carry_boundary_values = jnp.where(
            first_relevant_at_location, boundary_values, jnp.nan
        )
    else:
        boundary_values = jnp.zeros((0,), dtype=liquid.dtype)
        carry_boundary_values = jnp.zeros((0,), dtype=liquid.dtype)
        owner_is_right = jnp.zeros((0,), dtype=jnp.bool_)

    return _ResolvedNBEGMFeasibility(
        partition=partition,
        feasible_interval_mask=feasible,
        boundary_values=boundary_values,
        carry_boundary_values=carry_boundary_values,
        boundary_owner_is_right=owner_is_right,
    )


def _augment_liquid_for_feasibility(
    *,
    liquid: Float1D,
    feasibility: _ResolvedNBEGMFeasibility,
) -> tuple[Float1D, IntND]:
    """Add the exact owner and the adjacent open-side point at each boundary.

    The solve sees a weakly ascending query grid containing both representable
    sides of every topology-relevant feasibility threshold. A threshold is
    relevant when both its exact value and its ownership-selected open-side
    neighbour lie on the closed state domain. Other declarations occupy
    fixed-shape padding slots at the upper grid endpoint; their published
    breakpoint entry is NaN.

    Returns:
        The augmented query grid and the permutation that restores the original
        concatenation order, whose first `liquid.shape[0]` entries are the
        model's state-grid rows.

    """
    boundary = feasibility.boundary_values
    finite = jnp.isfinite(boundary)
    first_at_location = jnp.concatenate(
        (
            jnp.ones((1,), dtype=jnp.bool_),
            boundary[1:] != boundary[:-1],
        )
    )
    already_on_grid = jnp.any(boundary[:, None] == liquid[None, :], axis=1)
    exact_slot = jnp.where(
        finite & first_at_location & ~already_on_grid,
        boundary,
        liquid[-1],
    )
    open_base = jnp.where(finite, boundary, liquid[-1])
    open_side = jnp.where(
        feasibility.boundary_owner_is_right,
        jnp.nextafter(open_base, -jnp.inf),
        jnp.nextafter(open_base, jnp.inf),
    )
    open_side = jnp.where(finite, open_side, liquid[-1])
    concatenated = jnp.concatenate((liquid, exact_slot, open_side))
    order = jnp.argsort(concatenated, stable=True)
    return concatenated[order], jnp.argsort(order).astype(jnp.int32)


def _build_nbegm_feasibility_carry_template(
    *, liquid_grid: Float1D, n_boundaries: int
) -> EGMCarry:
    """Build the fixed-shape carry used by one-sided feasibility boundaries."""
    row = jnp.concatenate(
        (
            liquid_grid,
            jnp.repeat(liquid_grid[-1:], repeats=2 * n_boundaries),
        )
    )
    return EGMCarry(
        endog_grid=row,
        value=jnp.zeros_like(row),
        marginal_utility=jnp.zeros_like(row),
        taste_shock_scale=jnp.asarray(0.0, dtype=liquid_grid.dtype),
        breakpoints=(
            jnp.zeros((n_boundaries,), dtype=liquid_grid.dtype)
            if n_boundaries
            else None
        ),
    )


@dataclass(frozen=True, kw_only=True)
class _BoundNBEGM(NBEGM):
    """Internal NB-EGM configuration with regime-resolved DAG role names."""

    continuous_state: StateName
    continuous_action: ActionName
    budget_target: FunctionName
    post_decision_function: FunctionName
    post_decision_bound: PostDecisionBoundSpec | None = None
    """Literal grid floor installed as a construction capability."""
    _evaluated_predicate_names: tuple[FunctionName, ...] = field(
        default=(), metadata={"fingerprint_omit_if_default": True}
    )
    """Canonical state-node capability installed from this regime's authority."""


@dataclass(frozen=True, kw_only=True)
class _NBEGMFloor:
    """One processed source-cell floor and its shared saving offsets."""

    lower: Callable[..., ScalarFloat] | None
    """Canonical processed floor DAG, including qualified parameter leaves."""
    lower_inputs: tuple[ReferenceName, ...]
    """Actual leaves of the processed DAG."""
    offsets: Float1D
    """One unexpanded saving-offset vector shared by source cells."""
    literal_nodes: Float1D | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Original physical saving nodes for a genuine literal feature."""
    literal_value: float | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Canonical literal bound; no executable lower DAG is invented."""
    nested_axes: tuple[tuple[StateName, Float1D], ...] | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )

    def evaluate_floor(
        self,
        *,
        pool: EconFunctionKwargs,
        prepared_source_floor: PreparedNestedFloor | None = None,
    ) -> tuple[ScalarFloat, Float1D, BoundInputStatus]:
        """Evaluate once and preserve invalid-input status before safe substitution."""
        if self.nested_axes is not None:
            if type(prepared_source_floor) is not BoundedResult:
                raise TypeError(
                    "A nested source floor requires the genuine keeper bank."
                )
            return nested_floor_at_cell(
                prepared=prepared_source_floor,
                offsets=self.offsets,
                axes=self.nested_axes,
                pool=pool,
            )
        if self.lower is None:
            nodes = self.literal_nodes
            if nodes is None or self.literal_value is None:
                raise TypeError(
                    "A literal NB source feature requires its original physical grid."
                )
            floor = jnp.asarray(self.literal_value, dtype=nodes.dtype)
            return (
                floor,
                nodes,
                BoundInputStatus(
                    nonfinite_floor=~jnp.isfinite(floor),
                    invalid_shifted_grid=(
                        ~jnp.all(jnp.isfinite(nodes)) | jnp.any(nodes[1:] <= nodes[:-1])
                    ),
                ),
            )
        floor = self.lower(**{name: pool[name] for name in self.lower_inputs})
        nodes, status = shift_savings_grid(floor=floor, offsets=self.offsets)
        invalid = status.nonfinite_floor | status.invalid_shifted_grid
        return (
            jnp.where(invalid, 0.0, floor),
            jnp.where(invalid, self.offsets, nodes),
            status,
        )


@dataclass(frozen=True, kw_only=True)
class _NBEGMArgumentBuilder(_EGMArgumentBuilder):
    """Tabulate an NB law and its original boundary candidates at one source floor."""

    current_floor: _NBEGMFloor | None = None
    """Installed source evaluator; absent on ordinary feature-free adapters."""

    owned_child_reads: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Pass canonical numeric child carries to the installed NB consumer."""

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        """Retain the existing law preparation and add only the actual carry channel."""
        arguments = super().__call__(context)
        if not self.owned_child_reads:
            return arguments
        return MappingProxyType(
            {
                **arguments,
                "next_regime_to_continuation": MappingProxyType(
                    {
                        self.continuation_target: cast(
                            "EGMCarry",
                            context.next_regime_to_continuation[
                                self.continuation_target
                            ],
                        ),
                    }
                ),
            }
        )

    def _law_readings(
        self,
        *,
        flat_params: FlatParams,
        next_breakpoints: FloatND | None,
        period: int,
        age: ParamsLeaf,
    ) -> tuple[
        Float1D,
        Float1D,
        Float1D,
        Float1D,
        Float1D,
        BoundInputStatus | None,
        _EGMSavingsSupport | None,
    ]:
        """Share one evaluated floor across source nodes and child-boundary targets."""
        if self.current_floor is None:
            return super()._law_readings(
                flat_params=flat_params,
                next_breakpoints=next_breakpoints,
                period=period,
                age=age,
            )
        pool = {
            **self.bound_params,
            **union_free_params(
                flat_params=flat_params,
                regime_name=self.regime_name,
                transition_target_names=self.transition_target_names,
            ),
            "period": jnp.int32(period),
            "age": age,
        }
        _floor, nodes, status = self.current_floor.evaluate_floor(pool=pool)
        prepared = replace(
            self, savings_grid=nodes, current_floor=None, current_query=None
        )
        readings = super(_NBEGMArgumentBuilder, prepared)._law_readings(
            flat_params=flat_params,
            next_breakpoints=next_breakpoints,
            period=period,
            age=age,
        )
        return (*readings[:5], status, readings[6])


def _build_nbegm_source_floor(
    *,
    context: SolverBuildContext,
    bound: PostDecisionBoundSpec | None,
    savings_grid: Float1D,
) -> _NBEGMFloor | None:
    """Bind the canonical source-period floor without evaluating numerical params."""
    if bound is None:
        return None
    model = context.nbegm_source_query_model or (context.nbegm_query_models or {}).get(
        context.regime_name
    )
    if bound.kind == "literal":
        if model is None:
            return None
        if model.literal_nodes is None or bound.literal_value is None:
            raise TypeError("A literal NB feature lacks its canonical original nodes.")
        return _NBEGMFloor(
            lower=None,
            lower_inputs=(),
            offsets=savings_grid - savings_grid[0],
            literal_nodes=savings_grid,
            literal_value=bound.literal_value,
        )
    if bound.lower_target is None:
        raise TypeError("An NB callable bound lacks its canonical target.")
    from _lcm.egm.preferences import concatenate_regime_function  # noqa: PLC0415

    lower = concatenate_regime_function(
        functions=context.functions, target=bound.lower_target
    )
    nested_axes = None
    if model is not None and model.retain_nested_feasibility:
        nested_axes = tuple(
            (name, context.grids[name].to_jax()) for name in model.authority.state_names
        )
    return _NBEGMFloor(
        lower=lower,
        lower_inputs=tuple(inspect.signature(lower).parameters),
        offsets=savings_grid - savings_grid[0],
        nested_axes=nested_axes,
    )


def proved_post_decision_of(*, solver: NBEGM) -> FunctionName | None:
    """Name the post-decision state a case-piece solver's savings grid spans.

    `None` before the solver is bound to a regime's liquid margin: the grid
    exists, but the state it spans is not yet named, so nothing can be proved
    against it. Deciding that on the solver's type rather than on whether an
    attribute happens to be there keeps a rename a failure at this read — a
    defaulted lookup would answer `None` for a bound solver too, and the caller
    cannot tell that apart from an honestly unbound one.

    Args:
        solver: The solver a regime declared.

    Returns:
        The post-decision state's name, or `None` if the solver is not a bound
        case-piece kernel.

    """
    return solver.post_decision_function if isinstance(solver, _BoundNBEGM) else None


@dataclass(frozen=True, kw_only=True)
class _RideAlongArgumentBuilder:
    """Build the ride-along programs' arguments for lowering and execution.

    Both programs take the state grids, the carries of this period's stateful
    targets placed on the co-map mesh, the free params of the regime and of its
    transition targets, and the period and age. No target value array is read: the
    continuation is interpolated from the carries alone.
    """

    regime_name: RegimeName
    """Name of the regime whose flat params the programs project."""

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    stateful_targets: frozenset[RegimeName]
    """Carry keys the programs read; the rolling carry is filtered to these."""

    co_map_state_names: tuple[StateName, ...]
    """Distributed ride states co-mapped with the child carry (a leading prefix of
    the ride axes), or empty when no ride state is distributed."""

    n_intervals: int
    """Number of declared continuation intervals, or zero on a smooth route."""

    def __call__(
        self, context: CoreBuildContext
    ) -> MappingProxyType[ReferenceName, PytreeValue | ShapeDtypePytree]:
        """Return the exact kwargs shared by lowering and the runtime call."""
        state_action_space = cast("StateActionSpace", context.state_action_space)
        flat_params = cast("FlatParams", context.flat_params)
        ages = cast("TimeAxis", context.ages)
        states = dict(state_action_space.states)
        interval_arguments = (
            {_INTERVAL_COORDINATE: jnp.arange(self.n_intervals, dtype=jnp.int32)}
            if self.n_intervals > 1
            else {}
        )
        return MappingProxyType(
            {
                **states,
                **state_action_space.discrete_actions,
                **interval_arguments,
                "next_regime_to_continuation": self._co_map_carry(
                    states=states,
                    next_regime_to_continuation=context.next_regime_to_continuation,
                ),
                **union_free_params(
                    flat_params=flat_params,
                    regime_name=self.regime_name,
                    transition_target_names=self.transition_target_names,
                ),
                "period": jnp.int32(context.period),
                "age": ages.values[context.period],
            }
        )

    def _co_map_carry(
        self,
        *,
        states: Mapping[StateName, ContinuousState | DiscreteState],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
    ) -> MappingProxyType[RegimeName, EGMCarry]:
        """Filter the carry to this period's targets and place it on the co-map mesh.

        With a distributed ride state the programs run on the co-map mesh, and every
        committed argument has to live there. A target whose carry lacks the
        co-mapped axis — a terminal carry that does not depend on the distributed
        state — is produced on a single device, so it is replicated onto the mesh
        (lowering template and runtime value alike, so the compiled-for input
        sharding and the argument agree). Leaves already on the mesh, sharded per
        slice or replicated, pass through; without a distributed ride state the
        filtered carry is returned as is.
        """
        selected = _carry_subset(
            next_regime_to_continuation=next_regime_to_continuation,
            stateful_targets=self.stateful_targets,
        )
        ordinary: dict[RegimeName, EGMCarry] = {}
        for name, payload in selected.items():
            if not isinstance(payload, EGMCarry):
                raise UnsupportedOperationError(
                    "NB ride-along projected child readers are not installed."
                )
            ordinary[name] = payload
        carry = MappingProxyType(ordinary)
        co_map_sharding = self._co_map_sharding(states=states)
        if co_map_sharding is None:
            return carry
        replicated = jax.NamedSharding(mesh=co_map_sharding.mesh, spec=jax.P())
        return jax.tree.map(
            functools.partial(
                _place_on_co_map_mesh, mesh=co_map_sharding.mesh, replicated=replicated
            ),
            carry,
        )

    def _co_map_sharding(
        self, *, states: Mapping[StateName, ContinuousState | DiscreteState]
    ) -> jax.NamedSharding | None:
        """Sharding of the flattened ride-cell axis over the co-mapped states.

        The co-mapped ride states are a leading prefix of the ride axes, so a
        per-cell array is sharded along the flattened leading ride-cell axis (one
        block per device). Returns `None` when no ride state is distributed.
        """
        co_map_state_names = self.co_map_state_names
        if not co_map_state_names:
            return None
        leading_sharding = getattr(states[co_map_state_names[0]], "sharding", None)
        if not isinstance(leading_sharding, jax.NamedSharding):
            return None
        axes: str | tuple[str, ...] = (
            co_map_state_names[0]
            if len(co_map_state_names) == 1
            else co_map_state_names
        )
        return jax.NamedSharding(mesh=leading_sharding.mesh, spec=jax.P(axes))


# keyword-only-exempt: library-callback=jax.tree.map
def _place_on_co_map_mesh(
    leaf: PytreeValue,
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    replicated: jax.NamedSharding,
) -> PytreeValue:
    """Replicate a carry leaf onto the co-map mesh unless it already lives there."""
    leaf_sharding = getattr(leaf, "sharding", None)
    if isinstance(leaf_sharding, jax.NamedSharding) and leaf_sharding.mesh == mesh:
        return leaf
    return jax.device_put(leaf, replicated)


@dataclass(frozen=True, kw_only=True)
class _NBEGMNestedKernelOutput(FeasibilityKernelOutput):
    """Actual current D feasibility consumed only by the enclosing outer fold."""

    current_feasibility: BoolND
    prepared_source_floor: PreparedNestedFloor | None = None


@dataclass(frozen=True, kw_only=True)
class _RideAlongNBEGMPeriodKernel:
    """The case-piece EGM period kernel for a regime carrying a ride-along co-state.

    The kernel's native program graph is its sole core authority. Both programs
    run the same tile-local body: per cell block it reads
    `next_regime_to_continuation` through the transition-aware reader — the
    probability-weighted expected value and marginal over the savings grid — and
    solves the block's 1-D liquid step against those rows inside the same
    compiled body.

    - `main` (`VALUES_ONLY`) publishes the value array and the ride-axes-leading
      continuation carry a parent interpolates.
    - `replay` (`REPLAY`) additionally publishes the consumption policy on the
      value layout and, for a discrete ride action, the conditional branch value
      and consumption banks with the branch axis leading.

    The solve loop dispatches exactly one of them per solve, chosen by the result
    retention; calling the kernel runs whichever program was compiled.
    """

    _core_programs: MappingProxyType[str, CoreProgram]
    """The immutable two-node program graph, `main` then `replay`."""

    statics: _NBEGMRideAlongStatics
    """Build-time config: the cell block width and the per-cell parameter splits."""

    cliff_candidates: bool
    """Whether this period's programs read save-to-cliff candidate columns.

    True only when the carry publishes jump topology and the period's targets
    include the regime itself (the cliffs are the self-schedule's).
    """

    regime_name: RegimeName
    """Name of the regime whose flat params this kernel projects."""

    stateful_targets: frozenset[RegimeName]
    """Carry keys this period's programs read; the rolling carry is filtered to
    these."""

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    schedule_spec: _NBEGMScheduleSpec
    """The composed budget, utility, and discount declarations the programs bind
    per cell, published so an independent reader can evaluate the same
    declarations."""

    continuation_plan: ContinuationPlan
    """The `ContinuationPlan` (regime transition, child reads, stochastic weights)
    the programs read the continuation through, published for the same reason."""

    savings_grid: Float1D
    """The post-decision savings grid the programs close over."""

    own_bound_specs: tuple[PostDecisionBoundSpec, ...] = field(
        default=(), metadata={"fingerprint_omit_if_default": True}
    )
    """Canonical source-cell bound slots, separate from child slots."""
    query_input_authorities: (
        tuple[tuple[RegimeName, int, PostDecisionBoundSpec], ...] | None
    ) = field(default=None, metadata={"fingerprint_omit_if_default": True})
    """Installed actual-child diagnostic ownership in canonical plan order."""
    source_period: int = field(
        default=0, metadata={"fingerprint_omit_if_default": True}
    )
    """Computational source period for feature ownership validation."""
    retain_nested_feasibility: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Expose actual pre-D current feasibility to an enclosing nested solver."""

    def __post_init__(self) -> None:
        """Snapshot the graph and require exactly the two scoped programs."""
        programs = MappingProxyType(dict(self._core_programs))
        if tuple(programs) != ("main", "replay"):
            msg = (
                "A ride-along NB-EGM kernel publishes exactly the programs 'main' "
                f"and 'replay'; got {tuple(programs)}."
            )
            raise ValueError(msg)
        object.__setattr__(self, "_core_programs", programs)

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Return the native graph used by eager, JIT, AOT, and replay paths."""
        return self._core_programs

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> _RideAlongNBEGMPeriodKernel:
        """Bind the regime's and its carry targets' fixed params into every program.

        Each program's checked variant receives the same binding, so a solve
        that collects continuation read counts lowers the same inputs.
        """
        bound = union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=self.regime_name,
            transition_target_names=self.transition_target_names,
        )
        if not bound:
            return self
        return replace(
            self,
            _core_programs=MappingProxyType(
                {
                    name: bind_core_program_fixed_params(program=program, params=bound)
                    for name, program in self._core_programs.items()
                }
            ),
        )

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable[..., PytreeValue]],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: TimeAxis,
        logger: logging.Logger,  # noqa: ARG002
        prepared_source_floor: PreparedNestedFloor | None = None,
    ) -> KernelOutput:
        """Run the compiled program the retention selected and assemble the output.

        `replay` is run whenever it was compiled; a values-only solve compiles
        `main` alone and publishes no simulation policy.
        """
        name = "replay" if "replay" in compiled_cores else "main"
        program = self._core_programs[name]
        arguments = program.argument_builder(
            CoreBuildContext(
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                next_regime_to_continuation=next_regime_to_continuation,
                flat_params=flat_params,
                period=period,
                ages=ages,
            )
        )
        if prepared_source_floor is not None:
            if not self.retain_nested_feasibility:
                raise TypeError(
                    "Only a canonical nested inner accepts a prepared floor."
                )
            arguments = {**arguments, NESTED_FLOOR_INPUT: prepared_source_floor}
        outputs = compiled_cores[name](**arguments)
        return self._consume_output(
            name=name,
            outputs=outputs,
            state_action_space=state_action_space,
            period=period,
            prepared_source_floor=prepared_source_floor,
        )

    def _consume_output(
        self,
        *,
        name: str,
        outputs: PytreeValue,
        state_action_space: StateActionSpace,
        period: int,
        prepared_source_floor: PreparedNestedFloor | None,
    ) -> KernelOutput:
        """Consume actual retained-program channels under canonical integrity slots."""
        summary = None
        current_feasibility = None
        if self.own_bound_specs or self.query_input_authorities:
            if period != self.source_period or type(outputs) is not BoundedResult:
                raise TypeError(
                    "NB query publication requires its declared period "
                    "and integrity records."
                )
            _validate_nbegm_integrity_slot_counts(
                outputs=outputs,
                own_bound_specs=self.own_bound_specs,
                query_input_authorities=self.query_input_authorities,
            )
            summary = outputs
            if self.retain_nested_feasibility:
                if len(outputs.output) == _PREPARED_NBEGM_KEEPER_ARITY:
                    prepared_source_floor = outputs.output[-1]
                    outputs = replace(outputs, output=outputs.output[:-1])
                outputs, counts, current_feasibility = outputs.output
            else:
                outputs, counts = outputs.output[:-1], outputs.output[-1]
        elif isinstance(outputs, BoundedResult):
            raise TypeError(
                "NB integrity records require model-owned diagnostic slots."
            )
        if name == "main":
            V_arr, carry = cast("tuple[FloatND, EGMCarry]", outputs)
            if summary is not None:
                return _bounded_nbegm_kernel_output(
                    value=V_arr,
                    carry=carry,
                    counts=counts,
                    summary=summary,
                    current_feasibility=current_feasibility,
                    prepared_source_floor=prepared_source_floor,
                )
            return KernelOutput(value=V_arr, continuations={EGM_CONTINUATION: carry})
        if self.statics.n_action_branches:
            V_arr, carry, inner_action, branch_value, branch_inner_action = cast(
                "tuple[FloatND, EGMCarry, FloatND, FloatND, FloatND]", outputs
            )
            branch_discrete_actions = jnp.asarray(
                self.statics.discrete_action_codes, dtype=jnp.int32
            )
        else:
            V_arr, carry, inner_action = cast(
                "tuple[FloatND, EGMCarry, FloatND]", outputs
            )
            branch_value = None
            branch_inner_action = None
            branch_discrete_actions = None
        replay = {
            SIMULATION_POLICY: NBEGMGridPolicy(
                action=inner_action,
                state_names=tuple(state_action_space.states),
                branch_inner_action=branch_inner_action,
                branch_value=branch_value,
                branch_discrete_actions=branch_discrete_actions,
                discrete_action_names=self.statics.discrete_action_names,
            )
        }
        if summary is not None:
            return _bounded_nbegm_kernel_output(
                value=V_arr,
                carry=carry,
                counts=counts,
                summary=summary,
                current_feasibility=current_feasibility,
                prepared_source_floor=prepared_source_floor,
                replay=replay,
            )
        return KernelOutput(
            value=V_arr,
            continuations={EGM_CONTINUATION: carry},
            replay=replay,
        )


@dataclass(frozen=True, kw_only=True)
class _LiteralCaseThreshold:
    """A case threshold declared as a literal number."""

    literal: float | int | bool
    """The declared boundary value."""

    def __call__(self, _params: EconFunctionKwargs) -> FloatND:
        """Return the literal as an array; the params play no part."""
        return jnp.asarray(self.literal)


@dataclass(frozen=True, kw_only=True)
class _ParameterCaseThreshold:
    """A case threshold read from one qualified parameter."""

    qualified_name: QualifiedName
    """The `<predicate>__<threshold>` key the boundary value is read under."""

    def __call__(self, params: EconFunctionKwargs) -> FloatND | float:
        """Read the threshold from the regime's flat params."""
        return _numeric_nbegm_argument(value=params[self.qualified_name])


@dataclass(frozen=True, eq=False, kw_only=True)
class _ComputedCaseThreshold:
    """A case threshold composed from the regime's own functions."""

    threshold_dag: Callable[..., FloatND]
    """The composed function evaluating the threshold from its params."""

    argument_names: tuple[str, ...]
    """The parameter names the composed function reads, all state-independent."""

    def __call__(self, params: EconFunctionKwargs) -> FloatND:
        """Evaluate the composed threshold on the regime's flat params."""
        return jnp.asarray(
            self.threshold_dag(**{name: params[name] for name in self.argument_names})
        )


@dataclass(frozen=True, kw_only=True)
class _NBEGMCaseSpec:
    """Build-time statics describing one binary case split."""

    below_callable: UserFunction
    """Piece contribution on the lower side of the liquid boundary."""
    above_callable: UserFunction
    """Piece contribution on the upper side of the liquid boundary."""
    below_func: FunctionName
    """Qualified-name prefix of the lower piece's params."""
    above_func: FunctionName
    """Qualified-name prefix of the upper piece's params."""
    below_param_names: tuple[ReferenceName, ...]
    """Parameter names of the lower piece."""
    above_param_names: tuple[ReferenceName, ...]
    """Parameter names of the upper piece."""
    threshold_resolver: (
        _LiteralCaseThreshold | _ParameterCaseThreshold | _ComputedCaseThreshold
    )
    """Resolve the literal, parameter, or computed boundary value."""
    equality_owner: EqualityOwner
    """Coordinate side owning equality (`when` is below, `otherwise` above)."""


# The only split output the case-piece route knows how to route — an additive
# cash-on-hand shift.
_NBEGM_SPLIT_OUTPUT = "subsidy"


def fail_if_taste_shocks_declared(*, context: SolverBuildContext) -> None:
    """Reject EV1 taste shocks on a regime solved by the case-piece kernels.

    Every NB-EGM envelope takes a hard maximum over branches and every carry it
    publishes pins the taste-shock scale to zero, while the simulate phase draws
    the declared shocks and applies the smoothed choice probabilities. Solving
    the hard-max problem and simulating the smoothed one yields an inconsistent
    policy and a biased simulated distribution, so the declaration is refused
    rather than ignored.

    Args:
        context: The regime's solver build context.

    Raises:
        RegimeInitializationError: If the regime declares taste shocks.

    """
    if context.has_taste_shocks:
        msg = (
            f"Regime '{context.regime_name}' declares EV1 taste shocks, but "
            "NB-EGM does not implement taste shocks: its envelopes take a hard "
            "maximum over the discrete branches and the carry it publishes "
            "fixes the taste-shock scale at zero, while the simulate phase "
            "would apply the declared shocks. Remove the taste shocks, or use "
            "`GridSearch` or `DCEGM` for this regime."
        )
        raise RegimeInitializationError(msg)


def _validate_nbegm_case_piece_declarations(
    *,
    context: SolverModelContext,
    solver: NBEGM,
) -> None:
    """Check case coverage and reject hidden branching in a regime's pieces.

    Shared by every solver that runs the case-piece EGM kernels on a liquid
    margin, so a nested solver applies the same gate to its inner margin as a
    bare `NBEGM` would.

    Args:
        context: Finalized declarations for the regime and its model.
        solver: Bound inner NB-EGM configuration whose liquid role and tracing
            policy govern the declarations.

    Raises:
        NBEGMCaseError: A split output is not fully covered, a boundary declares
            no surface, a piece reaches outside the flat params, or a piece hides
            branching the Euler inversion cannot absorb.

    """
    import inspect  # noqa: PLC0415

    import jax.numpy as jnp  # noqa: PLC0415

    from _lcm.egm.nbegm import collect_nbegm_metadata  # noqa: PLC0415
    from _lcm.egm.nbegm_validation import (  # noqa: PLC0415
        find_ast_violations,
        find_jaxpr_violations,
        is_smooth_helper,
    )

    user_regime = context.user_regimes[context.regime_name]
    bound = cast("_BoundNBEGM", solver)
    functions = context.solve_functions
    registry = collect_nbegm_metadata(functions=functions)
    discrete_actions = frozenset(
        name
        for name, grid in user_regime.actions.items()
        if isinstance(grid, DiscreteGrid)
    )
    if registry.piece_sets and discrete_actions:
        msg = (
            f"Regime '{context.regime_name}' declares case pieces alongside the "
            f"discrete action(s) {sorted(discrete_actions)}. The "
            "case-piece kernels solve a one-asset consumption-saving problem "
            "and maximize over the continuous action only, so the discrete "
            "choice would never enter the published value and simulation would "
            "draw its policy from a value function that never saw it. Declare "
            "the discrete choice with a `lcm.piecewise_affine` schedule, or use "
            "`GridSearch` for this regime."
        )
        raise RegimeInitializationError(msg)
    _validate_nbegm_boundary_scope(
        registry=registry,
        functions=functions,
        liquid_state_name=bound.continuous_state,
        reserved_names=frozenset(user_regime.states) | frozenset(user_regime.actions),
    )
    violations: list[str] = []
    for piece_set in registry.piece_sets:
        for piece_name in (piece_set.when_func, piece_set.otherwise_func):
            declared_piece = functions[piece_name]
            if is_smooth_helper(declared_piece):
                continue
            piece = inspect.unwrap(declared_piece)
            violations += find_ast_violations(func=piece, mode="smooth_user")
            n_params = len(inspect.signature(piece).parameters)
            abstract_args = tuple(jnp.asarray(1.0) for _ in range(n_params))
            violations += find_jaxpr_violations(
                func=piece,
                abstract_args=abstract_args,
                mode="smooth_user",
                probe_failure=solver.probe_failure,
            )
    if violations:
        from lcm.exceptions import NBEGMCaseError  # noqa: PLC0415

        msg = "NBEGM smoothness gate failed:\n" + "\n".join(violations)
        raise NBEGMCaseError(msg)


def _validate_nbegm_boundary_scope(
    *,
    registry: NBEGMRegistry,
    functions: Mapping[FunctionName, UserFunction],
    liquid_state_name: StateName,
    reserved_names: frozenset[str],
) -> None:
    """Reject case-piece declarations the case-piece kernels cannot solve.

    The case-piece route splits exactly one additive cash-on-hand `subsidy`
    across one jump boundary on the liquid state, with pieces that read only the
    flat params (not states or actions). The comparison operator and operand
    direction jointly determine which formula lies below the boundary and which
    side owns equality.
    A budget outside that shape is declarable as a `lcm.piecewise_affine`
    schedule, which carries kinks, jumps, and floors together.
    """
    import inspect  # noqa: PLC0415

    from lcm.exceptions import NBEGMCaseError  # noqa: PLC0415

    for piece_set in registry.piece_sets:
        if piece_set.output != _NBEGM_SPLIT_OUTPUT:
            msg = (
                f"NBEGM case pieces split exactly one additive cash-on-hand "
                f"{_NBEGM_SPLIT_OUTPUT!r} output; the regime splits "
                f"{piece_set.output!r}."
            )
            raise NBEGMCaseError(msg)
        for piece_name in (piece_set.when_func, piece_set.otherwise_func):
            params = inspect.signature(functions[piece_name]).parameters
            state_action_deps = sorted(set(params) & reserved_names)
            if state_action_deps:
                msg = (
                    f"NBEGM case pieces read only the flat params; piece "
                    f"{piece_name!r} depends on the state/action "
                    f"{state_action_deps!r}."
                )
                raise NBEGMCaseError(msg)
    for predicate_name, boundary in registry.boundaries.items():
        _normalize_case_boundary(
            boundary=boundary,
            predicate_name=predicate_name,
            liquid_state_name=liquid_state_name,
            reserved_names=reserved_names,
        )


def _normalize_case_boundary(
    *,
    boundary: CaseBoundary,
    predicate_name: FunctionName,
    liquid_state_name: StateName,
    reserved_names: frozenset[str],
) -> tuple[Const | Ref, bool, EqualityOwner]:
    """Normalize either operand direction to lower/upper coordinate ownership."""
    from lcm.exceptions import NBEGMCaseError  # noqa: PLC0415

    if boundary.kind != "jump":
        msg = (
            "NBEGM case boundaries declare `kind='jump'`; "
            f"{predicate_name!r} declares {boundary.kind!r}."
        )
        raise NBEGMCaseError(msg)
    expression = boundary.expression
    if not isinstance(expression, Compare):
        msg = f"NBEGM case boundary {predicate_name!r} is not one comparison."
        raise NBEGMCaseError(msg)
    left_is_liquid = (
        isinstance(expression.left, Ref) and expression.left.name == liquid_state_name
    )
    right_is_liquid = (
        isinstance(expression.right, Ref) and expression.right.name == liquid_state_name
    )
    if left_is_liquid == right_is_liquid:
        msg = (
            f"NBEGM case boundary {predicate_name!r} must compare the liquid "
            f"state {liquid_state_name!r} with one threshold."
        )
        raise NBEGMCaseError(msg)
    threshold = expression.right if left_is_liquid else expression.left
    if not isinstance(threshold, (Const, Ref)):
        msg = f"NBEGM case boundary {predicate_name!r} has an invalid threshold."
        raise NBEGMCaseError(msg)
    if isinstance(threshold, Ref) and threshold.name in reserved_names:
        msg = (
            f"NBEGM case boundary {predicate_name!r} uses state/action "
            f"{threshold.name!r} as its threshold. The case-piece boundary must "
            "be a literal, parameter, or state-independent computed value."
        )
        raise NBEGMCaseError(msg)
    if isinstance(threshold, Const) and isinstance(threshold.value, bool):
        msg = (
            f"NBEGM case boundary {predicate_name!r} uses a Boolean threshold; "
            "the liquid boundary must be numerical."
        )
        raise NBEGMCaseError(msg)
    when_is_below = (left_is_liquid and expression.op in {"<", "<="}) or (
        right_is_liquid and expression.op in {">", ">="}
    )
    below_owns_equality = (
        expression.admits_equality if when_is_below else not expression.admits_equality
    )
    equality_owner: EqualityOwner = "when" if below_owns_equality else "otherwise"
    return threshold, when_is_below, equality_owner


_KERNEL_LIQUID_STATE = "liquid"
_KERNEL_DEFAULT_SAVINGS_NAME = "savings"


def fail_if_liquid_law_is_not_written_through_savings(
    *,
    context: SolverBuildContext,
    liquid_state_name: StateName,
    post_decision_name: FunctionName,
) -> None:
    """Check the regime states its liquid law as a function of savings.

    The kernels read the declared law rather than assuming its shape, so the law
    may carry any term the modeller writes. What it may not do is depend on the
    consumption choice by any route other than post-decision savings: the Euler
    inversion runs on a grid of savings and reads the continuation off the
    landing points that grid reaches, so a law whose landing point still moves
    with cash-on-hand at fixed savings has no single continuation to read. A law
    stated as `next_liquid(resources, consumption, ...)` is written through that
    second route even when it happens to depend on the difference alone.

    The route also binds the Euler grid to a state named `liquid`, because the
    case-piece kernels form cash-on-hand as `liquid + subsidy` themselves.

    Args:
        context: The regime's solver build context.
        liquid_state_name: The resolved Euler axis.
        post_decision_name: Name of the function computing post-decision savings.

    Raises:
        RegimeInitializationError: If the liquid state is named differently, if
            the regime declares no post-decision savings function, or if a liquid
            law does not read it.

    """
    regime_name = context.regime_name
    if liquid_state_name != _KERNEL_LIQUID_STATE:
        msg = (
            f"NBEGM's single-liquid kernels bind the Euler grid to a state named "
            f"{_KERNEL_LIQUID_STATE!r}; regime {regime_name!r} names it "
            f"{liquid_state_name!r}. Rename the state, or declare a "
            "`lcm.piecewise_affine` schedule with a `post_decision_function` so "
            "the budget is composed from the DAG instead."
        )
        raise RegimeInitializationError(msg)
    if post_decision_name not in context.functions:
        msg = (
            f"NBEGM inverts the Euler equation on a grid of post-decision "
            f"savings, so regime {regime_name!r} declares the function computing "
            f"them; none is named {post_decision_name!r}. Add it — "
            f"`def {post_decision_name}(resources, consumption): return resources "
            f"- consumption` is the usual one — or name the regime's own with "
            "`LiquidMargin(post_decision_state=...)`."
        )
        raise RegimeInitializationError(msg)
    liquid_law_name = f"next_{_KERNEL_LIQUID_STATE}"
    for target, target_transitions in context.transitions.items():
        liquid_law = target_transitions.get(liquid_law_name)
        if not callable(liquid_law):
            continue
        if post_decision_name not in _parameter_names(liquid_law):
            msg = (
                f"NBEGM reads the declared law at each level of post-decision "
                f"savings, so the law states where savings land: regime "
                f"{regime_name!r}'s transition to {target!r} declares "
                f"{liquid_law_name!r} as "
                f"{getattr(liquid_law, '__name__', liquid_law)!r}, which does not "
                f"read {post_decision_name!r}. Rewrite it through savings — "
                f"`lcm.liquid_law_from_savings` is the conventional form — or use "
                "`GridSearch`, which maximizes over the action grid and needs no "
                "post-decision variable."
            )
            raise RegimeInitializationError(msg)


def _fail_if_budget_node_differs_from_kernel_cash_on_hand(
    *,
    context: SolverBuildContext,
    routes_to_case_piece_core: bool,
    budget_target: FunctionName,
    liquid_state_name: StateName,
) -> None:
    """Check the regime declares the cash-on-hand the case-piece kernels form.

    The case-piece core evaluates each piece to a scalar and adds it to the liquid
    state itself, so the declared budget node is never called. An arbitrary
    callable reading exactly the liquid state and the split output can still
    combine them any way at all, and no finite check separates the ones that
    agree: a global rescaling reproduces the sum at every sampled point and still
    moves every state's value. So the route accepts `lcm.cash_on_hand_with_subsidy`
    — the form the kernels implement, recognized by identity — and refuses
    everything else. Every other route composes the budget node from the DAG and
    may read whatever it declares, so only that core's regimes are checked.

    Args:
        context: The regime's solver build context.
        routes_to_case_piece_core: Whether the regime's kernels come from the
            case-piece core, the one route that forms cash-on-hand itself.
        budget_target: Name of the budget node in the regime's function pool.
        liquid_state_name: Name of the liquid (Euler) state.

    Raises:
        RegimeInitializationError: If the node splits by phase, or is anything
            other than the declared fixed form.

    """
    regime_name = context.regime_name
    split_output = _NBEGM_SPLIT_OUTPUT
    user_regime = context.user_regimes[regime_name]
    raw_budget_node = user_regime.functions.get(budget_target)
    if not routes_to_case_piece_core or raw_budget_node is None:
        return
    budget_path = f"functions[{budget_target!r}]"
    if budget_path in context.phase_variation_paths:
        msg = (
            f"NBEGM's case-piece kernels form cash-on-hand as "
            f"`{liquid_state_name} + {split_output}` in both phases, so regime "
            f"{regime_name!r} cannot give {budget_target!r} a phase-dependent "
            "form. Declare a `lcm.piecewise_affine` schedule with a "
            "`post_decision_function` so the budget is composed from the DAG, or "
            "use `GridSearch` for this regime."
        )
        raise RegimeInitializationError(msg)
    budget_node = context.solve_functions.get(budget_target)
    if budget_node is not cash_on_hand_with_subsidy:
        msg = (
            f"NBEGM's case-piece kernels add the split output to the liquid state "
            f"themselves rather than calling the declared budget node, so this "
            f"route takes the node pylcm supplies and not an arbitrary callable: "
            f"regime {regime_name!r} declares {budget_target!r} as "
            f"{getattr(budget_node, '__name__', budget_node)!r}. Declare "
            "`lcm.cash_on_hand_with_subsidy`, whose identity settles the contract "
            "by construction. A budget that form cannot express does not belong "
            "on this route — declare a `lcm.piecewise_affine` schedule with a "
            "`post_decision_function` so the budget is composed from the DAG, or "
            "use `GridSearch` for this regime."
        )
        raise RegimeInitializationError(msg)


def _parameter_names(func: TransitionFunction) -> frozenset[str]:
    """Return a function's parameter names, empty when it has no signature."""
    import inspect  # noqa: PLC0415

    try:
        return frozenset(inspect.signature(func).parameters)
    except TypeError, ValueError:
        return frozenset()


def resolve_liquid_state_name(
    *, context: SolverBuildContext, declared: StateName
) -> StateName:
    """Validate and return a regime's declared liquid (Euler) axis.

    The canonical variable order leads with *discrete* states, so a regime's
    first state is its liquid one only when it declares nothing else — the
    axis is the state named by the regime's `LiquidMargin`.

    Args:
        context: The regime's solver build context.
        declared: The regime-bound liquid-state name.

    Returns:
        Name of the liquid (Euler) state.

    Raises:
        RegimeInitializationError: If the declared state is not one of the
            regime's.

    """
    continuous_states = tuple(
        name
        for name in context.state_action_space.state_names
        if isinstance(context.grids[name], ContinuousGrid)
    )
    if declared not in continuous_states:
        msg = (
            f"NBEGM's LiquidMargin.state {declared!r} is not a continuous state "
            f"of regime {context.regime_name!r}; its continuous states are "
            f"{continuous_states}."
        )
        raise RegimeInitializationError(msg)
    return declared


def _single_liquid_state_name(
    *, context: SolverBuildContext, declared: StateName, path: str
) -> StateName:
    """Resolve the liquid axis of a regime the single-axis kernels solve.

    These kernels carry one grid axis and read every other name as a flat
    param, so a second state is refused here rather than surfacing as a
    missing-parameter lookup inside the traced core.

    Args:
        context: The regime's solver build context.
        declared: The regime-bound liquid-state name.
        path: Name of the kernel path, for the message.

    Returns:
        Name of the liquid (Euler) state.

    Raises:
        RegimeInitializationError: If the regime carries any state besides the
            liquid one, or the axis cannot be resolved.

    """
    liquid_state_name = resolve_liquid_state_name(context=context, declared=declared)
    others = tuple(
        name
        for name in context.state_action_space.state_names
        if name != liquid_state_name
    )
    if others:
        msg = (
            f"NBEGM's {path} carries the liquid axis alone; regime "
            f"{context.regime_name!r} also declares {others}. A ride-along "
            "co-state needs the schedule path (declare a "
            "`lcm.piecewise_affine` schedule and a `post_decision_function`), "
            "or use `GridSearch` for this regime."
        )
        raise RegimeInitializationError(msg)
    return liquid_state_name


def _collect_nbegm_case_spec(
    *, context: SolverBuildContext, continuous_state: StateName
) -> _NBEGMCaseSpec:
    """Collect the single binary case split from the regime's user functions."""
    import inspect  # noqa: PLC0415

    from _lcm.egm.nbegm import collect_nbegm_metadata  # noqa: PLC0415

    functions = context.solve_functions
    registry = collect_nbegm_metadata(functions=functions)
    if len(registry.piece_sets) != 1:
        msg = (
            "NBEGM case pieces split exactly one output; the regime declares "
            f"{len(registry.piece_sets)}."
        )
        raise RegimeInitializationError(msg)
    piece_set = registry.piece_sets[0]
    space = context.state_action_space
    liquid_state_name = _single_liquid_state_name(
        context=context, declared=continuous_state, path="case-piece path"
    )
    _validate_nbegm_boundary_scope(
        registry=registry,
        functions=functions,
        liquid_state_name=liquid_state_name,
        reserved_names=frozenset(space.state_names) | frozenset(space.action_names),
    )
    threshold, when_is_below, equality_owner = _normalize_case_boundary(
        boundary=registry.boundaries[piece_set.predicate_name],
        predicate_name=piece_set.predicate_name,
        liquid_state_name=liquid_state_name,
        reserved_names=frozenset(space.state_names) | frozenset(space.action_names),
    )
    when_func = piece_set.when_func
    otherwise_func = piece_set.otherwise_func
    below_func, above_func = (
        (when_func, otherwise_func) if when_is_below else (otherwise_func, when_func)
    )
    below_callable = functions[below_func]
    above_callable = functions[above_func]
    return _NBEGMCaseSpec(
        below_callable=below_callable,
        above_callable=above_callable,
        below_func=below_func,
        above_func=above_func,
        below_param_names=tuple(inspect.signature(below_callable).parameters),
        above_param_names=tuple(inspect.signature(above_callable).parameters),
        threshold_resolver=_case_threshold_resolver(
            context=context,
            predicate_name=piece_set.predicate_name,
            threshold=threshold,
        ),
        equality_owner=equality_owner,
    )


def _case_threshold_resolver(
    *,
    context: SolverBuildContext,
    predicate_name: FunctionName,
    threshold: Const | Ref,
) -> _LiteralCaseThreshold | _ParameterCaseThreshold | _ComputedCaseThreshold:
    """Compile a case threshold without inventing a second predicate."""
    import inspect  # noqa: PLC0415

    from lcm.exceptions import NBEGMCaseError  # noqa: PLC0415

    if isinstance(threshold, Const):
        return _LiteralCaseThreshold(literal=threshold.value)

    threshold_name = threshold.name
    if threshold_name not in context.functions:
        return _ParameterCaseThreshold(
            qualified_name=f"{predicate_name}__{threshold_name}"
        )

    threshold_dag = concatenate_functions(
        dict(context.functions), targets=threshold_name
    )
    argument_names = tuple(inspect.signature(threshold_dag).parameters)
    state_action_names = frozenset(context.state_action_space.state_names) | frozenset(
        context.state_action_space.action_names
    )
    dynamic_names = sorted(set(argument_names) & state_action_names)
    if dynamic_names:
        msg = (
            f"NBEGM case threshold {threshold_name!r} depends on the "
            f"state/action names {dynamic_names}; the case-piece kernel requires "
            "one state-independent boundary value per solve."
        )
        raise NBEGMCaseError(msg)

    return _ComputedCaseThreshold(
        threshold_dag=threshold_dag, argument_names=argument_names
    )


def _build_nbegm_core(
    *,
    savings_grid: Float1D,
    functions: EconFunctionsMapping,
    consumption_action: ActionName,
    case_spec: _NBEGMCaseSpec,
    envelope_arithmetic: ComparisonArithmetic = "ordinary",
) -> _NBEGMCaseCore:
    """Build the jittable case-piece EGM core closing over the case split.

    The core evaluates each piece's additive contribution and the boundary
    threshold from the regime's flat params, runs the two-case EGM merge, and
    returns the value array and the marginal-value carry on the liquid grid.
    """
    from _lcm.egm.preferences import (  # noqa: PLC0415
        NEWTON_ACTION_FLOOR,
        get_preferences_builder,
        newton_action_ceiling,
    )

    return _NBEGMCaseCore(
        savings_grid=savings_grid,
        build_preferences=get_preferences_builder(
            functions=functions,
            action_name=consumption_action,
            action_lower=NEWTON_ACTION_FLOOR,
            action_upper=newton_action_ceiling(savings_grid),
        ),
        case_spec=case_spec,
        envelope_arithmetic=envelope_arithmetic,
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class _NBEGMCaseCore:
    """The jittable case-piece EGM core of one solver group.

    Evaluates each piece's additive contribution and the boundary threshold from
    the regime's flat params, runs the two-case EGM merge, and returns the value
    array and the marginal-value carry on the liquid grid.
    """

    savings_grid: Float1D
    """Exogenous post-decision savings grid the step inverts on."""

    build_preferences: Callable[[EconFunctionKwargs], Preferences]
    """Bind the regime's flat params into its utility, marginal, and inverse maps."""

    case_spec: _NBEGMCaseSpec
    """The case split: the two pieces, the threshold, and the equality owner."""

    envelope_arithmetic: ComparisonArithmetic
    """Arithmetic the upper envelope merges the cases with."""

    retain_query: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Keep independently declared conditional original-saving rows."""

    def __call__(
        self,
        *,
        liquid: Float1D,
        next_liquid_grid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        next_liquid: Float1D,
        marginal_return: Float1D,
        effective_savings_grid: Float1D | None = None,
        source_bound_status: BoundInputStatus | None = None,
        continuation_read: Callable[[FloatND, BoolND], tuple[FloatND, FloatND]]
        | None = None,
        **params: ValueND,
    ) -> _RawNBEGMOutput:
        """Solve one period of the case-piece regime on the liquid grid."""
        from _lcm.egm.nbegm_step import nbegm_one_asset_step  # noqa: PLC0415

        case_spec = self.case_spec
        preferences = self.build_preferences(params)
        subsidy_below = case_spec.below_callable(
            **{
                p: params[p if p == "period" else f"{case_spec.below_func}__{p}"]
                for p in case_spec.below_param_names
            }
        )
        subsidy_above = case_spec.above_callable(
            **{
                p: params[p if p == "period" else f"{case_spec.above_func}__{p}"]
                for p in case_spec.above_param_names
            }
        )
        asset_limit = case_spec.threshold_resolver(params)
        step = nbegm_one_asset_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            next_liquid_grid=next_liquid_grid,
            savings_grid=self.savings_grid
            if effective_savings_grid is None
            else effective_savings_grid,
            discount_factor=params["koopmans_aggregator__discount_factor"],
            preferences=preferences,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            subsidy_when=array_result(subsidy_below),
            subsidy_otherwise=array_result(subsidy_above),
            asset_limit=asset_limit,
            equality_owner=case_spec.equality_owner,
            arithmetic=self.envelope_arithmetic,
            retain_query=self.retain_query,
            continuation_read=continuation_read,
        )
        value, marginal, _policy = step[:3]
        carry = EGMCarry(
            endog_grid=liquid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(0.0, dtype=value.dtype),
            query_data=_required_nbegm_query_data(step=step)
            if self.retain_query
            else None,
        )
        if source_bound_status is not None:
            return BoundedResult(
                output=(value, carry), bound_status=(source_bound_status,)
            )
        return value, carry


def _fail_if_discrete_action_feeds_continuation(
    *,
    context: SolverBuildContext,
    action_name: ActionName,
    liquid_state_name: StateName,
    budget_target: FunctionName,
    post_decision_function: FunctionName | None,
    allow_continuation_feed: bool = False,
) -> None:
    """Reject a discrete action that shifts the continuation, not just the budget.

    The discrete envelope solves every branch against one shared next-period
    continuation, valid only when the action enters the current budget and utility
    alone. Two channels make the continuation branch-dependent and are refused:

    - the regime transition or a non-liquid state's law of motion reads the action
      (each branch would evolve to different targets / co-states);
    - the liquid law reads the action through anything other than the budget — e.g.
      an out-of-pocket cost that lands directly on next assets — so the branches
      reach different next liquid at the same savings.

    Feeding cash-on-hand is the intended budget channel: for the liquid law the
    budget nodes (the budget target and the post-decision savings) are cut to free
    leaves, so the action reaching next liquid only through the budget — whether the
    law reads `coh` directly or a post-decision `savings` — is exempt, while any
    off-budget path is still caught.

    `allow_continuation_feed` exempts every continuation channel on the ride-along
    path: it carries a leading branch axis on the continuation, and `bind_continuation`
    reads the regime-transition probabilities and every next-state from the same
    per-branch `combo_pool`, so a co-state law, next liquid off the budget, or the
    regime transition all become branch-dependent for free. Each branch then reads its
    own next-state coordinate and its own alive-vs-target weighting.
    """
    import inspect  # noqa: PLC0415

    transition_probs = context.compute_regime_transition_probs
    if (
        not allow_continuation_feed
        and transition_probs is not None
        and action_name in inspect.signature(transition_probs).parameters
    ):
        _reject_continuation_feed(
            where="the regime transition",
            action_name=action_name,
            regime_name=context.regime_name,
        )

    regime = context.user_regimes[context.regime_name]
    funcs: dict[FunctionName, UserFunction] = {
        name: func
        for name, func in regime.decomposed_functions.items()
        if is_user_function(func)
    }
    budget_nodes = frozenset({budget_target, post_decision_function})

    # The branch-indexed continuation reads each branch's own next-state
    # coordinate, so on the ride-along path every state-law feed is supported.
    if allow_continuation_feed:
        return

    for state_name, func in _state_laws(transitions=context.transitions):
        is_liquid = state_name == liquid_state_name
        if _law_reads_action(
            law=func,
            cut_budget=is_liquid,
            funcs=funcs,
            budget_nodes=budget_nodes,
            action_name=action_name,
        ):
            where = (
                f"the law of motion for {state_name!r} off the budget channel"
                if is_liquid
                else f"the law of motion for {state_name!r}"
            )
            _reject_continuation_feed(
                where=where, action_name=action_name, regime_name=context.regime_name
            )


def _reject_continuation_feed(
    *, where: str, action_name: ActionName, regime_name: RegimeName
) -> None:
    """Raise for a discrete action reaching the continuation through `where`."""
    msg = (
        f"NBEGM's discrete envelope shares one continuation across the "
        f"branches of {action_name!r}, so the action may shift only the "
        f"current budget and utility; regime {regime_name!r} reads it "
        f"in {where}. Fix the action there, or use a solver that carries a "
        "branch-specific continuation."
    )
    raise RegimeInitializationError(msg)


def _law_reads_action(
    *,
    law: Callable[..., FloatND | IntND],
    cut_budget: bool,
    funcs: Mapping[FunctionName, UserFunction],
    budget_nodes: frozenset[str | None],
    action_name: ActionName,
) -> bool:
    """Whether `law`, composed over the regime's functions, reads the action.

    For the liquid law the budget nodes are dropped, so the action reaches the law
    only through an off-budget path (an out-of-pocket cost on next assets).
    """
    pool = {
        name: func
        for name, func in funcs.items()
        if not (cut_budget and name in budget_nodes)
    }
    try:
        combined = concatenate_functions(
            {**pool, "__continuation_target__": law},
            targets="__continuation_target__",
        )
    except Exception:  # noqa: BLE001  # unanalysable law: leave to other gates
        return False
    return action_name in inspect.signature(combined).parameters


def _state_laws(
    *, transitions: TransitionFunctionsMapping
) -> Iterator[tuple[StateName, Callable[..., FloatND | IntND]]]:
    """Yield each canonical state law with the state it evolves.

    Reads the canonical per-target laws rather than a regime's declared
    `state_transitions`: which targets a regime reaches is the graph's to say,
    and the canonical form has already broadcast bare laws over exactly those
    targets. The regime transition and the stochastic weight laws carry no
    state coordinate, so neither is yielded.
    """
    from lcm.transition import StochasticTransition  # noqa: PLC0415

    for target_laws in transitions.values():
        for law_name, candidate in target_laws.items():
            if law_name.startswith("weight_"):
                continue
            state_name = law_name.removeprefix("next_")
            if state_name == "regime":
                continue
            func = (
                candidate.func
                if isinstance(candidate, StochasticTransition)
                else candidate
            )
            if callable(func):
                yield state_name, func


def _continuation_action_names(
    *,
    regime_transition: _ContinuationLaw,
    target_laws: Mapping[RegimeName, tuple[_ContinuationLaw, ...]],
    target_weight_laws: Mapping[RegimeName, _ContinuationLaw | None],
    target_resources_arg_names: Mapping[RegimeName, frozenset[str]],
    discount_factor_dag: Callable[..., FloatND] | None,
    interval_schedule_dags: tuple[Callable[..., FloatND] | None, ...],
    action_names: tuple[ActionName, ...],
) -> tuple[ActionName, ...]:
    """Name the discrete actions the per-cell continuation read consumes.

    The ride-along continuation is evaluated from a combo pool the branch's action
    codes ride in, so one branch's rows differ from another's only when an action
    is an argument of a channel the read consumes:

    - `regime_transition`: the regime transition probabilities;
    - `target_laws`: each stateful target's next-state functions, including the
      regime's own law that the save-to-cliff targets invert;
    - `target_weight_laws`: each stateful target's intrinsic stochastic-state
      weights, which may depend on the source branch even though a Markov state's
      support function is branch-independent;
    - `target_resources_arg_names`: each stateful target's resources leaves;
    - `discount_factor_dag`: the per-cell discount factor, when the regime
      composes one;
    - `interval_schedule_dags`: the derived schedule variables whose breakpoints
      partition a per-interval read — empty when the continuation does not read
      the liquid state, since the breakpoints then never enter the read.

    Actions reaching none of them leave every branch's rows identical, so the core
    reads the continuation once per class of branches agreeing on the named
    actions. Returned in declared order.
    """
    consumed: set[str] = set(_declared_parameter_names(regime_transition))
    consumed |= _declared_parameter_names(discount_factor_dag)
    for dag in interval_schedule_dags:
        consumed |= _declared_parameter_names(dag)
    for target, laws in target_laws.items():
        for law in laws:
            consumed |= _declared_parameter_names(law)
        consumed |= _declared_parameter_names(target_weight_laws[target])
        consumed |= set(target_resources_arg_names[target])
    return tuple(name for name in action_names if name in consumed)


def _declared_parameter_names(func: _ContinuationLaw | None) -> frozenset[str]:
    """Parameter names of a callable, empty for an absent one."""
    if func is None:
        return frozenset()
    return frozenset(inspect.signature(func).parameters)


def _continuation_branch_classes(
    *,
    branch_bindings: tuple[MappingProxyType[ActionName, int], ...],
    continuation_action_names: tuple[ActionName, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Partition the envelope branches by their continuation actions' codes.

    Returns the class index of every branch and, per class, the first branch
    carrying it — the branch whose continuation rows the class reads. Branches
    keyed on no action at all form one class.
    """
    keys = [
        tuple(binding[name] for name in continuation_action_names)
        for binding in branch_bindings
    ]
    distinct = list(dict.fromkeys(keys))
    class_of_branch = tuple(distinct.index(key) for key in keys)
    representatives = tuple(keys.index(key) for key in distinct)
    return class_of_branch, representatives


def _ride_discrete_actions(*, context: SolverBuildContext) -> DiscreteActionCodes:
    """Collect every budget-shifting discrete action with its grid codes.

    Returns an empty tuple when the regime carries no discrete action.
    """
    return _discrete_actions_of(space=context.state_action_space)


def _discrete_actions_of(*, space: StateActionSpace) -> DiscreteActionCodes:
    """Pair each declared discrete action with its grid codes, in declared order."""
    return tuple(
        (name, tuple(int(code) for code in codes))
        for name, codes in space.discrete_actions.items()
    )


def _stacked_branch_codes(
    *,
    branch_bindings: tuple[MappingProxyType[ActionName, int], ...],
    action_names: tuple[ActionName, ...],
) -> IntND:
    """Stack the branch bindings into a `(n_branches, n_actions)` code array.

    The branch axis is streamed by `lax.map`, so every branch has to present the
    same pytree: one row of codes, ordered by `action_names`, rather than a
    per-action entry whose presence varies.
    """
    return jnp.asarray(
        [[binding[name] for name in action_names] for binding in branch_bindings],
        dtype=jnp.int32,
    )


def _branch_inputs(
    *,
    codes: IntND,
    cont_value: FloatND | None,
    cont_marginal: FloatND | None,
    extra_cont_value: FloatND | None,
    cliff_savings: FloatND | None,
) -> MappingProxyType[str, FloatND | IntND]:
    """Assemble the pytree `lax.map` streams over the branch axis.

    The optional continuations enter only where the regime supplies them, so a
    regime without child cliffs or an extra continuation maps a narrower tree
    rather than one padded with sentinels that every branch would then carry.
    """
    inputs: dict[str, FloatND | IntND] = {"codes": codes}
    if cont_value is not None:
        inputs["cont_value"] = cont_value
    if cont_marginal is not None:
        inputs["cont_marginal"] = cont_marginal
    if extra_cont_value is not None:
        inputs["extra_cont_value"] = extra_cont_value
    if cliff_savings is not None:
        inputs["cliff_savings"] = cliff_savings
    return MappingProxyType(inputs)


def _branch_bindings(
    discrete_actions: DiscreteActionCodes,
) -> tuple[MappingProxyType[ActionName, int], ...]:
    """Bind every combination of the declared discrete actions, in envelope order.

    The envelope's branch axis is the *product* of the declared grids: a regime
    declaring a binary and a five-valued action carries ten branches, each a
    complete assignment of a code to every action. A regime declaring none
    carries a single empty binding, so the branch axis is never degenerate and
    callers need no separate no-action case.

    The order is the contract between the continuation read and the envelope
    solve: the continuation stacks one slice per branch and the envelope reads
    slice `pos` for branch `pos`. Both derive it from the same `discrete_actions`
    tuple through this function, so position means the same thing on both sides
    as long as neither builds the product itself.
    """
    names = tuple(name for name, _ in discrete_actions)
    code_sets = tuple(codes for _, codes in discrete_actions)
    return tuple(
        MappingProxyType(dict(zip(names, combination, strict=True)))
        for combination in itertools.product(*code_sets)
    )


def _liquid_affinity_samples(
    *, context: SolverBuildContext, liquid_state_name: StateName
) -> Float1D:
    """Return broad liquid-grid nodes and cell interiors for affinity checks."""
    grid = context.grids[liquid_state_name].to_jax()
    return _liquid_affinity_grid_samples(grid=grid)


def _liquid_affinity_grid_samples(*, grid: Float1D) -> Float1D:
    """Return the original separately evaluated grid/midpoint affinity samples."""
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    return jnp.sort(a=jnp.concatenate([grid, midpoints]))


def _budget_affinity_check(
    *,
    context: SolverBuildContext,
    coh_dag: Callable[..., FloatND],
    liquid_state_name: StateName,
    require_unit_slope: bool,
    probe_failure: Literal["reject", "assume_declared"],
    probe_schedule: Literal["first_solve", "every_solve", "never"] = "never",
    breakpoint_sources: tuple[_NBEGMSource, ...] = (),
    materializer: DatedGenerationMaterializer | None = None,
) -> ParamCheck:
    """Configure the shared all-branch, resolved-interval affinity preflight."""
    liquid_grid = (
        context.grids[liquid_state_name].to_jax()
        if materializer is None
        else materializer.materialize_grid(spec=context.grids[liquid_state_name])
    )
    liquid_samples = (
        _liquid_affinity_samples(context=context, liquid_state_name=liquid_state_name)
        if materializer is None
        else materializer.evaluate_validation(
            function=_liquid_affinity_grid_samples, arguments={"grid": liquid_grid}
        )
    )
    return _deferred_probe(
        probe=_fail_if_budget_nonaffine_in_liquid,
        regime_name=context.regime_name,
        probe_arguments=_probe_arguments(context=context),
        probe_schedule=probe_schedule,
        coh_dag=coh_dag,
        liquid_name=liquid_state_name,
        liquid_grid=liquid_grid,
        liquid_samples=liquid_samples,
        breakpoint_sources=breakpoint_sources,
        require_unit_slope=require_unit_slope,
        probe_failure=probe_failure,
    )


def _fail_if_budget_nonaffine_in_liquid(
    *,
    coh_dag: Callable[..., FloatND],
    liquid_name: str,
    require_unit_slope: bool,
    regime_name: RegimeName,
    probe_arguments: "_ProbeArguments",  # noqa: UP037 - original callback annotations resolve before the later class is bound
    liquid_grid: Float1D,
    liquid_samples: Float1D,
    breakpoint_sources: tuple[_NBEGMSource, ...] = (),
    probe_failure: Literal["reject", "assume_declared"] = "reject",
    derivative_programs: MappingProxyType[int, _BudgetProbeProgram] | None = None,
    materializer: DatedGenerationMaterializer | None = None,
) -> None:
    """Reject budgets violating affinity or required slope on any live interval.

    NBEGM reconstructs each declared economic interval from the cash-on-hand value
    and derivative at an interior point. The pure all-jump path reconstructs only
    intercepts and therefore additionally requires derivative one. A fixed set of
    liquid-grid probes cannot certify those assumptions when a parameter-dependent
    breakpoint creates a branch between or beyond the sampled points.

    For every fill/rank rung and integer-code override already used by the preflight,
    this check resolves the declaration's liquid-space breakpoint selection values
    with the current parameter draw, intersects their sorted partition with the
    closed liquid domain, and differentiates at one representable strict interior of
    every live cell. The previous grid-node/midpoint samples remain as broad
    diagnostics for undeclared curvature. Breakpoint work is host-side pre-solve
    validation only; no solve or simulation kernel changes.
    """
    arg_names = tuple(inspect.signature(coh_dag).parameters)
    if liquid_name not in arg_names:
        return

    dtype = canonical_float_dtype()
    tol = max(1e-6, 64.0 * float(jnp.finfo(dtype).eps))
    probe = _LiquidAffinityProbe(
        coh_dag=coh_dag,
        arg_names=arg_names,
        liquid_name=liquid_name,
        probe_arguments=probe_arguments,
        liquid_grid=liquid_grid,
        liquid_samples=liquid_samples,
        breakpoint_sources=breakpoint_sources,
        dtype=dtype,
        int_sweeps=_int_code_sweeps(
            arg_names=arg_names, int_arg_values=probe_arguments.int_arg_values
        ),
        derivative_programs=derivative_programs,
        materializer=materializer,
    )
    unprobeable = functools.partial(
        _fail_unprobeable,
        regime_name=regime_name,
        liquid_name=liquid_name,
        probe_failure=probe_failure,
    )

    try:
        second_derivatives = _evaluate_probe_fill_rungs(
            evaluate=functools.partial(
                _second_liquid_derivative_magnitudes, probe=probe
            ),
            materializer=materializer,
        )
    except Exception as probe_error:
        if materializer is not None and isinstance(probe_error, ExecutionPlanningError):
            raise
        unprobeable(probe_error=probe_error)
        worst_second = None
    else:
        worst_second = max(second_derivatives)
    if worst_second is not None and worst_second > tol:
        msg = (
            f"NBEGM's budget must be affine in the liquid state {liquid_name!r} "
            f"within each resolved interval, but regime {regime_name!r} has a "
            "nonzero second derivative there — a smooth nonlinear budget is not "
            "recovered by the per-interval affine segment. Declare every economic "
            "boundary or keep the budget affine per interval."
        )
        raise RegimeInitializationError(msg)

    slopes: tuple[float, ...] | None = None
    if require_unit_slope:
        try:
            slopes = _evaluate_probe_fill_rungs(
                evaluate=functools.partial(_liquid_slopes, probe=probe),
                materializer=materializer,
            )
        except Exception as probe_error:
            if materializer is not None and isinstance(
                probe_error, ExecutionPlanningError
            ):
                raise
            unprobeable(probe_error=probe_error)
    offending = (
        next(
            (
                slope
                for slope in slopes
                if abs(slope - 1.0) > tol * max(1.0, abs(slope))
            ),
            None,
        )
        if slopes is not None
        else None
    )
    if offending is not None:
        msg = (
            f"NBEGM's all-jump path solves the budget from per-interval intercepts "
            f"assuming unit slope in the liquid state, but regime {regime_name!r} has "
            f"a budget slope of {offending:.4g} in {liquid_name!r}. Declare a "
            "coincident `continuous_kink` so the non-unit affine slope routes to the "
            "mixed step, or keep the jump-only budget additive (unit slope)."
        )
        raise RegimeInitializationError(msg)


@dataclass(frozen=True)
class _FlowProbeReadings:
    """One fill rung's readings of the period flow in the consumption action."""

    flows: tuple[float, ...]
    """Flow level at each probed consumption, over the discrete-code sweep."""
    marginals: tuple[float, ...]
    """Flow derivative in consumption, aligned with `flows`."""
    elasticities: tuple[float, ...]
    """`c q'(c) / q(c)`, aligned with `flows`."""


def _fail_if_flow_not_single_power(
    *,
    utility_dag: Callable[..., FloatND],
    consumption_action_name: ActionName,
    regime_name: RegimeName,
    probe_arguments: _ProbeArguments,
    probe_failure: Literal["reject", "assume_declared"],
) -> None:
    """Probe the flow's consumption elasticity for the single-power contract.

    The Epstein-Zin Euler inversion is closed-form only for a flow whose
    marginal is a single power of consumption, `q = A c^phi` with `phi > 0`.
    Reading `q(1)` and `q'(1)` alone identifies a local scale and elasticity —
    it cannot certify the global structure (for `q = e^c` the locally fitted
    power solves a different first-order condition). The probe evaluates the
    flow, its marginal, and the elasticity `c q'(c)/q(c)` at several
    consumption values, with every declared parameter set to its own value and
    each integer-coded argument swept over its actual grid codes. Rejected on any
    draw selected by the probe schedule:

    - a nonpositive flow or nonpositive marginal at any probed point (the
      recursion takes fractional powers of the flow; `q = -c` carries a
      constant positive elasticity, so signs are checked directly),
    - a varying elasticity (the closed-form inversion needs one global power),
    - a nonpositive elasticity.
    """
    arg_names = tuple(inspect.signature(utility_dag).parameters)
    if consumption_action_name not in arg_names:
        return

    try:
        readings = _evaluate_on_first_workable_fill(
            functools.partial(
                _flow_probe_readings,
                flow_of_consumption=functools.partial(
                    _flow_of_consumption,
                    utility_dag=utility_dag,
                    arg_names=arg_names,
                    consumption_action_name=consumption_action_name,
                    probe_arguments=probe_arguments,
                    fill=_FLOW_PROBE_FILL,
                ),
                sweep_names=tuple(
                    name for name in arg_names if name in probe_arguments.int_arg_names
                ),
                probe_arguments=probe_arguments,
            )
        )
        flows = list(readings.flows)
        marginals = list(readings.marginals)
        elasticities = list(readings.elasticities)
    except Exception as probe_error:
        msg = (
            f"NBEGM could not verify that regime {regime_name!r}'s period flow "
            f"is a single power of {consumption_action_name!r}: the "
            "elasticity probe failed to evaluate the flow on scalar inputs "
            f"({type(probe_error).__name__}: {probe_error})."
        )
        if probe_failure == "assume_declared":
            warnings.warn(
                msg + " Solving anyway (`probe_failure='assume_declared'`): "
                "the model author asserts the single-power flow; validate the "
                "solve against an independent reference.",
                stacklevel=2,
            )
            return
        raise RegimeInitializationError(
            msg + " Restructure the flow so it evaluates on scalar inputs, set "
            "`probe_failure='assume_declared'` to assert the structure "
            "yourself, or use GridSearch() for this regime."
        ) from probe_error
    # A negative flow can carry a constant *positive* elasticity (`q = -c` has
    # elasticity one everywhere), so the sign of the flow and of its marginal
    # must be checked directly — the recursion takes fractional powers of the
    # flow, and the Euler inversion assumes an increasing one.
    if min(flows) <= 0.0 or min(marginals) <= 0.0:
        msg = (
            f"Regime {regime_name!r} declares a `certainty_equivalent`, but "
            "its period flow is not strictly positive and increasing in "
            f"{consumption_action_name!r} at the probed points (flow range "
            f"[{min(flows):.6g}, {max(flows):.6g}], marginal range "
            f"[{min(marginals):.6g}, {max(marginals):.6g}]). The Epstein-Zin "
            "recursion requires `q = A c^phi` with `A > 0` and `phi > 0`; "
            "restructure the flow or use GridSearch() for this regime."
        )
        raise RegimeInitializationError(msg)
    # The probe's elasticities come out of `jax.grad` at the active float
    # dtype, so their roundoff scatter scales with that dtype's precision:
    # sqrt(eps) covers the accumulated error of the flow/marginal quotient
    # in both float64 and float32 while staying far below any genuine
    # elasticity variation.
    tol = math.sqrt(float(jnp.finfo(canonical_float_dtype()).eps))
    spread = max(elasticities) - min(elasticities)
    scale = max(1.0, abs(elasticities[0]))
    if spread > tol * scale or min(elasticities) <= 0.0:
        msg = (
            f"Regime {regime_name!r} declares a `certainty_equivalent`, but "
            "its period flow is not a single power of "
            f"{consumption_action_name!r} with a positive exponent: the probed "
            f"consumption elasticities range over "
            f"[{min(elasticities):.6g}, {max(elasticities):.6g}]. The "
            "Epstein-Zin Euler inversion is closed-form only for "
            "`q = A c^phi` with `phi > 0`; restructure the flow or use "
            "GridSearch() for this regime."
        )
        raise RegimeInitializationError(msg)


class _ConstantKeyMapping(Mapping[str, FloatND]):
    """Mapping answering every key with the same value.

    Which keys a grouped param carries is a property of the params, which arrive
    long after the kernels are built, so the probe cannot enumerate them and
    answers whatever the model's own code asks for. It reports itself as empty,
    since there is no key set to iterate.
    """

    def __init__(self, value: FloatND) -> None:
        self._value = value

    def __getitem__(self, key: str) -> FloatND:
        return self._value

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


class _ProbeMappingLeaf(MappingLeaf):
    """A `MappingLeaf` whose every entry is the same probe fill.

    Satisfies a grouped param's declared type so the DAG evaluates, at the cost
    of flattening the group: a schedule probed this way has one bracket with one
    rate, so the probe sees the budget's structure between breakpoints and not
    the schedule's own shape. That is the region the affinity and constancy
    preconditions are about — a genuine nonlinearity in the liquid state (a
    square, a reciprocal) still shows up — but a defect that needs two distinct
    entries to appear does not.
    """

    __slots__ = ("value",)

    def __init__(self, value: FloatND) -> None:
        self.value = value
        self.data = _ConstantKeyMapping(value)


# Registered in its own right: JAX matches a pytree node by exact type, so
# inheriting `MappingLeaf`'s registration gains nothing and the fill would be
# rejected as not-an-array by the first compiled DAG a probe reaches. One child,
# the fill itself, so a traced round trip returns a leaf that still answers every
# key.
jax.tree_util.register_pytree_node(
    _ProbeMappingLeaf,
    lambda leaf: ((leaf.value,), None),
    lambda _aux, children: _ProbeMappingLeaf(children[0]),
)


# The names a stringified parameter annotation is resolved against: the public
# classes and type aliases plus the grouped-param leaf types — the vocabulary
# model functions annotate their parameters in, and so the whole of what a
# composed function's stringified annotations can name.
_PROBE_ANNOTATION_VOCABULARY: MappingProxyType[str, AnnotationForm] = MappingProxyType(
    {
        **{
            name: value
            for name, value in vars(lcm_typing).items()
            if not name.startswith("_") and isinstance(value, type | TypeAliasType)
        },
        "MappingLeaf": MappingLeaf,
        "UserMappingLeaf": UserMappingLeaf,
    }
)


# Fill level at and above which a boolean-annotated argument reads `True`. The
# probes evaluate at constant fills of 1.0 and 3.0 (and ramps spanning both), so
# a gate on a declared flag is probed on each of its branches.
_PROBE_FILL_HIGH = 2.0


@dataclass(frozen=True, kw_only=True)
class _ProbeArguments:
    """How the probes build every argument of the DAG they differentiate.

    Assembled at model build from the regime's grids and its functions'
    annotations, then completed with `with_params` for each selected draw.
    Splitting it that way lets the probes read the model's real schedules and
    tables: the classification is a property of the source, the values are not
    known until the user supplies params.
    """

    int_arg_values: MappingProxyType[str, tuple[int, ...]] = MappingProxyType({})
    """Grid codes per integer-coded argument, swept one code at a time."""
    array_float_arg_names: frozenset[ReferenceName] = frozenset()
    """Arguments whose consumers annotate them as float arrays."""
    array_arg_ranks: MappingProxyType[str, int] = MappingProxyType({})
    """Axis count per array argument, read off how its consumers subscript it."""
    annotated_int_arg_names: frozenset[ReferenceName] = frozenset()
    """Arguments whose consumers annotate them with an integer dtype."""
    bool_arg_names: frozenset[ReferenceName] = frozenset()
    """Arguments whose consumers annotate them with a boolean dtype."""
    mapping_leaf_arg_names: frozenset[ReferenceName] = frozenset()
    """Arguments whose consumers annotate them as grouped params."""
    param_values: MappingProxyType[QualifiedName, ParamsLeaf] = MappingProxyType({})
    """The model's own parameter values, empty until `with_params` runs."""

    @property
    def int_arg_names(self) -> frozenset[str]:
        """Every argument that must receive an integer fill."""
        return frozenset(self.int_arg_values) | self.annotated_int_arg_names

    def with_params(
        self, *, flat_params: FlatParams, regime_name: RegimeName
    ) -> _ProbeArguments:
        """Return a copy answering every declared parameter with its own value.

        The regime's own flat params take precedence; the other regimes' fill in
        underneath, because a probe may differentiate a law that carries into a
        target regime and reads that target's params.
        """
        merged: dict[QualifiedName, ParamsLeaf] = {}
        for path, regime_params in flat_namespaces(flat_params):
            if path != (regime_name,) and path[0] != EDGES:
                merged.update(regime_params)
        merged.update(regime_kernel_params(flat_params, regime_name=regime_name))
        return replace(self, param_values=MappingProxyType(merged))

    def fill(
        self,
        *,
        name: str,
        fill: float,
        array_floats: bool = False,
        array_rank: int = 1,
        leaf_rank: int = 1,
        materializer: DatedGenerationMaterializer | None = None,
    ) -> ParamsLeaf:
        """Build one argument at the given fill level and rung."""
        function = functools.partial(
            _probe_fill,
            name=name,
            fill=fill,
            int_arg_names=self.int_arg_names,
            array_float_arg_names=self.array_float_arg_names,
            array_arg_ranks=self.array_arg_ranks,
            bool_arg_names=self.bool_arg_names,
            mapping_leaf_arg_names=self.mapping_leaf_arg_names,
            array_floats=array_floats,
            array_rank=array_rank,
            leaf_rank=leaf_rank,
        )
        return evaluate_probe_stage(
            function=function,
            arguments={"param_values": self.param_values},
            materializer=materializer,
        )


@dataclass(frozen=True, kw_only=True)
class _ProbeFillRung:
    """One assignment of a probe's synthetic fills.

    A rung fixes the fill level, the array shapes the fill ladder is trying, and
    the integer-coded arguments pinned to one of their grid codes, so every
    argument a probed function reads other than the one being varied is constant
    across the rung.
    """

    fill: float
    """Fill level of every float argument."""
    array_floats: bool
    """Whether every float argument receives an array fill."""
    array_rank: int
    """Floor on the axis count of array-typed arguments."""
    leaf_rank: int
    """Axis count of a grouped param's probe entries."""
    int_overrides: MappingProxyType[str, int]
    """Integer-coded arguments pinned to one grid code each."""

    def argument(
        self,
        *,
        name: str,
        probe_arguments: _ProbeArguments,
        materializer: DatedGenerationMaterializer | None = None,
    ) -> ParamsLeaf:
        """Build one argument of the probed function at this rung."""
        if name in self.int_overrides:
            return evaluate_probe_stage(
                function=functools.partial(
                    probe_int_fill, value=self.int_overrides[name]
                ),
                arguments={},
                materializer=materializer,
            )
        return probe_arguments.fill(
            name=name,
            fill=self.fill,
            array_floats=self.array_floats,
            array_rank=self.array_rank,
            leaf_rank=self.leaf_rank,
            materializer=materializer,
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class _FilledScalarFunction:
    """A probed function as a scalar function of one argument.

    Every other argument is the rung's constant fill, so the callable is
    differentiable in its one argument and a whole rung's probe points go
    through a single vectorized call.
    """

    func: Callable[..., FloatND]
    """The composed function being probed."""
    arg_names: tuple[ReferenceName, ...]
    """Its parameter names, in signature order."""
    varied_name: str
    """The one argument the call varies."""
    probe_arguments: _ProbeArguments
    """How the probe builds every other argument."""
    rung: _ProbeFillRung
    """The fills bound for every other argument."""

    def __call__(self, value: FloatND) -> FloatND:
        """Evaluate the function at `value`, reshaped to a scalar."""
        kwargs = {
            name: (
                value
                if name == self.varied_name
                else self.rung.argument(name=name, probe_arguments=self.probe_arguments)
            )
            for name in self.arg_names
        }
        return jnp.asarray(self.func(**kwargs)).reshape(())


@dataclass(frozen=True, eq=False, kw_only=True)
class _LiquidAffinityProbe:
    """The resolved-interval affinity probe of one composed budget.

    Resolves the declared breakpoints at each rung with the current draw,
    intersects their sorted partition with the closed liquid domain, and
    differentiates the budget at one representable strict interior of every live
    cell — and, for the curvature diagnostic, at the broad grid samples too.
    """

    coh_dag: Callable[..., FloatND]
    """The composed cash-on-hand as a function of the liquid state and params."""
    arg_names: tuple[ReferenceName, ...]
    """The composed budget's parameter names."""
    liquid_name: str
    """Name of the liquid state the budget is differentiated in."""
    probe_arguments: _ProbeArguments
    """How the probe builds every argument other than the liquid state."""
    liquid_grid: Float1D
    """The liquid grid whose closed span bounds the live intervals."""
    liquid_samples: Float1D
    """Broad grid-node and midpoint samples for the curvature diagnostic."""
    breakpoint_sources: tuple[_NBEGMSource, ...]
    """The declared breakpoints, resolved per rung into liquid-space values."""
    dtype: DTypeLike
    """The canonical float dtype every probe value is cast to."""
    int_sweeps: tuple[MappingProxyType[str, int], ...]
    """The one-at-a-time integer-code overrides every fill is swept over."""
    derivative_programs: MappingProxyType[int, _BudgetProbeProgram] | None = None
    """Model-owned derivatives with current filled arguments supplied dynamically."""
    materializer: DatedGenerationMaterializer | None = None
    """Call-local validation owner, never retained by a source core or check."""

    def budget_of_liquid(self, *, rung: _ProbeFillRung) -> _FilledScalarFunction:
        """The budget as a scalar function of the liquid state at one rung."""
        return _FilledScalarFunction(
            func=self.coh_dag,
            arg_names=self.arg_names,
            varied_name=self.liquid_name,
            probe_arguments=self.probe_arguments,
            rung=rung,
        )

    def resolved_breakpoint(
        self, *, source: _NBEGMSource, rung: _ProbeFillRung
    ) -> FloatND:
        """Resolve one declared breakpoint into its liquid-space value."""
        from _lcm.egm.nbegm_breakpoints import (  # noqa: PLC0415
            linear_asset_selection_preimage,
        )

        cell_names = frozenset(
            name
            for name in (
                *source.derived_state_names,
                source.threshold_index_state,
            )
            if name is not None
        )
        cell = {
            name: rung.argument(
                name=name,
                probe_arguments=self.probe_arguments,
                materializer=self.materializer,
            )
            for name in cell_names
        }
        table = rung.argument(
            name=source.threshold_param_name,
            probe_arguments=self.probe_arguments,
            materializer=self.materializer,
        )
        threshold_value = evaluate_probe_stage(
            function=functools.partial(
                _indexed_threshold_value,
                subkey=source.threshold_subkey,
                index_state=source.threshold_index_state,
                static_index=source.threshold_static_index,
            ),
            arguments={"table": table, "cell": cell},
            materializer=self.materializer,
        )
        threshold = evaluate_probe_stage(
            function=functools.partial(probe_scalar, dtype=self.dtype),
            arguments={"value": threshold_value},
            materializer=self.materializer,
        )
        if source.derived_of_liquid_dag is None:
            return evaluate_probe_stage(
                function=functools.partial(
                    probe_selected_threshold,
                    equality_above=source.equality_owner == "above",
                    dtype=self.dtype,
                ),
                arguments={"threshold": threshold},
                materializer=self.materializer,
            )
        if self.materializer is not None:
            raise UnsupportedOperationError(
                "Cold NB derived-schedule validation is not installed."
            )

        dag = source.derived_of_liquid_dag
        return linear_asset_selection_preimage(
            z_of_liquid=_FilledScalarFunction(
                func=dag,
                arg_names=tuple(inspect.signature(dag).parameters),
                varied_name=self.liquid_name,
                probe_arguments=self.probe_arguments,
                rung=rung,
            ),
            threshold=threshold,
            equality_owner=source.equality_owner,
        )

    def strict_interval_samples(self, *, rung: _ProbeFillRung) -> tuple[FloatND, ...]:
        """One representable strict interior per live resolved interval."""
        dtype = self.dtype
        raw = evaluate_probe_stage(
            function=functools.partial(probe_raw_breakpoints, dtype=dtype),
            arguments={
                "values": tuple(
                    self.resolved_breakpoint(source=source, rung=rung)
                    for source in self.breakpoint_sources
                )
            },
            materializer=self.materializer,
        )
        # The original raw stage already owns the explicit-dtype empty vector.
        # With no declared boundaries, clamping cannot change any element; keep
        # domain endpoint reads and every strict-interior check below unchanged.
        clamped = (
            evaluate_probe_stage(
                function=functools.partial(probe_clamped_breakpoints, dtype=dtype),
                arguments={"raw": raw, "liquid_grid": self.liquid_grid},
                materializer=self.materializer,
            )
            if self.breakpoint_sources
            else raw
        )
        domain_start = evaluate_probe_stage(
            function=functools.partial(probe_grid_scalar, index=0, dtype=dtype),
            arguments={"grid": self.liquid_grid},
            materializer=self.materializer,
        )
        domain_stop = evaluate_probe_stage(
            function=functools.partial(probe_grid_scalar, index=-1, dtype=dtype),
            arguments={"grid": self.liquid_grid},
            materializer=self.materializer,
        )

        internal: list[FloatND] = []
        sorted_values = evaluate_probe_stage(
            function=probe_sorted,
            arguments={"values": clamped},
            materializer=self.materializer,
        )
        boundaries = (
            evaluate_probe_stage(
                function=probe_unstack,
                arguments={"values": sorted_values},
                materializer=self.materializer,
            )
            if sorted_values.shape[0]
            else ()
        )
        for raw_boundary in boundaries:
            boundary = evaluate_probe_stage(
                function=functools.partial(probe_scalar, dtype=dtype),
                arguments={"value": raw_boundary},
                materializer=self.materializer,
            )
            if bool(
                evaluate_probe_stage(
                    function=probe_between,
                    arguments={
                        "value": boundary,
                        "lower": domain_start,
                        "upper": domain_stop,
                    },
                    materializer=self.materializer,
                )
            ) and (
                not internal
                or bool(
                    evaluate_probe_stage(
                        function=probe_distinct,
                        arguments={"value": boundary, "previous": internal[-1]},
                        materializer=self.materializer,
                    )
                )
            ):
                internal.append(boundary)

        edges = (domain_start, *internal, domain_stop)
        representatives: list[FloatND] = []
        for lower, upper in itertools.pairwise(edges):
            if not bool(
                evaluate_probe_stage(
                    function=probe_greater,
                    arguments={"upper": upper, "lower": lower},
                    materializer=self.materializer,
                )
            ):
                continue
            midpoint = evaluate_probe_stage(
                function=probe_midpoint,
                arguments={"lower": lower, "upper": upper},
                materializer=self.materializer,
            )
            inside = functools.partial(
                evaluate_probe_stage,
                function=probe_between,
                materializer=self.materializer,
            )
            if not bool(
                inside(arguments={"value": midpoint, "lower": lower, "upper": upper})
            ):
                midpoint = evaluate_probe_stage(
                    function=probe_nextafter,
                    arguments={"lower": lower, "upper": upper},
                    materializer=self.materializer,
                )
            if not bool(
                inside(arguments={"value": midpoint, "lower": lower, "upper": upper})
            ):
                msg = (
                    "resolved economic interval has positive width but no "
                    "representable strict interior in the active float dtype: "
                    f"[{float(lower):.17g}, {float(upper):.17g}]"
                )
                raise ValueError(msg)
            representatives.append(midpoint)
        return tuple(representatives)

    def probe_points(
        self, *, interval_samples: tuple[FloatND, ...], with_grid_samples: bool
    ) -> FloatND:
        """Stack one rung's probe points onto a single batched axis."""
        return evaluate_probe_stage(
            function=functools.partial(
                probe_points, with_grid_samples=with_grid_samples, dtype=self.dtype
            ),
            arguments={
                "liquid_samples": self.liquid_samples,
                "interval_samples": interval_samples,
            },
            materializer=self.materializer,
        )

    def liquid_derivative(
        self, *, order: int, points: FloatND, rung: _ProbeFillRung
    ) -> FloatND:
        """Evaluate the order-th liquid derivative at every probe point at once.

        Everything the budget reads other than the liquid state is constant for a
        given rung, so the whole rung is one compiled, vectorized call rather than
        one traced call per point.
        """
        if self.derivative_programs is not None:
            arguments = {
                name: rung.argument(
                    name=name,
                    probe_arguments=self.probe_arguments,
                    materializer=self.materializer,
                )
                for name in self.arg_names
                if name != self.liquid_name
            }
            return evaluate_probe_stage(
                function=functools.partial(
                    _evaluate_liquid_probe_program,
                    program=self.derivative_programs[order],
                ),
                arguments={"points": points, "arguments": arguments},
                materializer=self.materializer,
            )
        if self.materializer is not None:
            raise UnsupportedOperationError(
                "Owned NB probes require the original prepared derivative programs."
            )
        derivative: Callable[[FloatND], FloatND] = self.budget_of_liquid(rung=rung)
        for _ in range(order):
            derivative = jax.grad(derivative)
        return jax.jit(jax.vmap(derivative))(points)


def _fail_unprobeable(
    *,
    probe_error: Exception,
    regime_name: RegimeName,
    liquid_name: str,
    probe_failure: Literal["reject", "assume_declared"],
) -> None:
    """Refuse, or warn and continue, when the affinity probe cannot evaluate."""
    msg = (
        f"NBEGM could not verify that regime {regime_name!r}'s budget is affine "
        f"in the liquid state {liquid_name!r}: the resolved-interval probe "
        "failed on scalar inputs "
        f"({type(probe_error).__name__}: {probe_error}). The per-interval EGM "
        "inversion is exact only for an affine within-interval budget."
    )
    if probe_failure == "assume_declared":
        warnings.warn(
            msg + " Solving anyway (`probe_failure='assume_declared'`): the "
            "model author asserts within-interval affinity; validate the solve "
            "against an independent reference.",
            stacklevel=2,
        )
        return
    raise RegimeInitializationError(
        msg + " Restructure the budget so it evaluates on scalar inputs (use "
        "`jnp.where` instead of Python branches), set "
        "`probe_failure='assume_declared'` to assert affinity yourself, or use "
        "the brute-force solver for this regime."
    ) from probe_error


def _second_liquid_derivative_magnitudes(
    *,
    array_floats: bool,
    array_rank: int,
    leaf_rank: int,
    probe: _LiquidAffinityProbe,
) -> tuple[float, ...]:
    """Absolute second liquid derivatives at every probe point of one fill rung.

    Sweeps the constant fills and the integer-code overrides, probing the grid
    samples together with each resolved interval's strict interior.
    """
    values: list[float] = []
    for fill in (1.0, 3.0):
        for overrides in probe.int_sweeps:
            rung = _ProbeFillRung(
                fill=fill,
                array_floats=array_floats,
                array_rank=array_rank,
                leaf_rank=leaf_rank,
                int_overrides=overrides,
            )
            interval_samples = probe.strict_interval_samples(rung=rung)
            points = probe.probe_points(
                interval_samples=interval_samples, with_grid_samples=True
            )
            if points.size == 0:
                continue
            second = probe.liquid_derivative(order=2, points=points, rung=rung)
            for value in jax.device_get(second).tolist():
                if not math.isfinite(value):
                    raise ValueError("non-finite second derivative")
                values.append(abs(value))
    return tuple(values)


def _liquid_slopes(
    *,
    array_floats: bool,
    array_rank: int,
    leaf_rank: int,
    probe: _LiquidAffinityProbe,
) -> tuple[float, ...]:
    """Liquid slopes at every resolved interval's strict interior of one rung."""
    slopes: list[float] = []
    for fill in (1.0, 3.0):
        for overrides in probe.int_sweeps:
            rung = _ProbeFillRung(
                fill=fill,
                array_floats=array_floats,
                array_rank=array_rank,
                leaf_rank=leaf_rank,
                int_overrides=overrides,
            )
            interval_samples = probe.strict_interval_samples(rung=rung)
            points = probe.probe_points(
                interval_samples=interval_samples, with_grid_samples=False
            )
            if points.size == 0:
                continue
            first = probe.liquid_derivative(order=1, points=points, rung=rung)
            for slope in jax.device_get(first).tolist():
                if not math.isfinite(slope):
                    raise ValueError("non-finite first derivative")
                slopes.append(slope)
    return tuple(slopes)


# Consumption values the single-power probe reads the flow at.
_FLOW_PROBE_CONSUMPTIONS: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)

# Fill level of every other argument of the probed flow.
_FLOW_PROBE_FILL = 1.7


# keyword-only-exempt: library-callback=jax.grad
def _flow_of_consumption(
    consumption: ScalarFloat,
    int_overrides: Mapping[str, int],
    *,
    array_floats: bool,
    array_rank: int,
    leaf_rank: int,
    utility_dag: Callable[..., FloatND],
    arg_names: tuple[ReferenceName, ...],
    consumption_action_name: ActionName,
    probe_arguments: _ProbeArguments,
    fill: float,
) -> ScalarFloat:
    """The period flow at one consumption, every other argument filled."""
    kwargs = {
        name: (
            consumption
            if name == consumption_action_name
            else jnp.asarray(int_overrides[name], dtype=jnp.int32)
            if name in int_overrides
            else probe_arguments.fill(
                name=name,
                fill=fill,
                array_floats=array_floats,
                array_rank=array_rank,
                leaf_rank=leaf_rank,
            )
        )
        for name in arg_names
    }
    return jnp.asarray(utility_dag(**kwargs)).reshape(())


def _flow_probe_readings(
    *,
    array_floats: bool,
    array_rank: int,
    leaf_rank: int,
    flow_of_consumption: Callable[..., ScalarFloat],
    sweep_names: tuple[str, ...],
    probe_arguments: _ProbeArguments,
) -> _FlowProbeReadings:
    """Read the flow, its marginal, and its elasticity over one fill rung."""
    flows: list[float] = []
    marginals: list[float] = []
    elasticities: list[float] = []
    for overrides in _int_code_sweeps(
        arg_names=sweep_names, int_arg_values=probe_arguments.int_arg_values
    ):
        for probe_c in _FLOW_PROBE_CONSUMPTIONS:
            args = (jnp.asarray(probe_c), overrides)
            rung = {
                "array_floats": array_floats,
                "array_rank": array_rank,
                "leaf_rank": leaf_rank,
            }
            flow = float(flow_of_consumption(*args, **rung))
            marginal = float(jax.grad(flow_of_consumption)(*args, **rung))
            flows.append(flow)
            marginals.append(marginal)
            elasticities.append(probe_c * marginal / flow)
    return _FlowProbeReadings(
        flows=tuple(flows),
        marginals=tuple(marginals),
        elasticities=tuple(elasticities),
    )


def _deferred_probe(
    *,
    probe: Callable[..., None],
    regime_name: RegimeName,
    probe_arguments: _ProbeArguments,
    probe_schedule: Literal["first_solve", "every_solve", "never"] = "never",
    **bound: _ProbeBinding,
) -> ParamCheck:
    """Configure a probe at model build and run it as its schedule asks.

    The probe's target — the composed budget, the utility DAG, the continuation
    plan — is fixed by the model's structure and bound here. Its arguments are
    not: a budget reading tax schedules cannot be differentiated until those
    schedules have values, so the fills are completed from the params the engine
    supplies at solve.

    The engine hands every published check every draw. Which of those draws this
    probe actually evaluates is `probe_schedule`'s decision, kept here rather than
    in the engine so a solver that needs each draw re-checked and one that does
    not can sit in the same model.
    """
    if probe is _fail_if_budget_nonaffine_in_liquid:
        bound["derivative_programs"] = MappingProxyType(
            {
                order: _make_liquid_probe_program(
                    func=cast("Callable[..., FloatND]", bound["coh_dag"]),
                    liquid_name=cast("str", bound["liquid_name"]),
                    order=order,
                    scalar=True,
                )
                for order in ((1, 2) if bound["require_unit_slope"] else (2,))
            }
        )
    elif probe is _fail_if_liquid_reading_next_state_varies_within_interval:
        plan = cast("ContinuationPlan", bound["continuation_plan"])
        funcs = (
            *(
                func
                for target in plan.stateful_targets
                for func in (
                    plan.child_reads[target].next_state_func,
                    plan.child_reads[target].euler_state_func,
                )
            ),
            plan.compute_regime_transition_probs,
        )
        liquid_name = cast("str", bound["liquid_name"])
        law_programs: dict[_ContinuationLaw, _LawProbeProgram] = {
            func: _make_liquid_probe_program(
                func=func, liquid_name=liquid_name, order=1, scalar=False
            )
            for func in funcs
            if liquid_name in inspect.signature(func).parameters
        }
        bound["derivative_programs"] = MappingProxyType(law_programs)
    return _DeferredProbe(
        probe=probe,
        regime_name=regime_name,
        probe_arguments=probe_arguments,
        probe_schedule=probe_schedule,
        bound=MappingProxyType(bound),
    )


def _make_liquid_probe_program[T: ArrayTree](
    *, func: Callable[..., T], liquid_name: str, order: int, scalar: bool
) -> Callable[[FloatND, EconFunctionKwargs], T]:
    """Construct a derivative program closing over model structure only."""

    # Build transforms inside the JIT so its model closure has no JAX sentinels.
    # keyword-only-exempt: library-callback=jax.jit
    def evaluate(points: FloatND, arguments: EconFunctionKwargs) -> T:
        derivative = _DynamicLiquidProbe(
            func=func, liquid_name=liquid_name, scalar=scalar
        )
        for _ in range(order):
            derivative = jax.grad(derivative) if scalar else jax.jacfwd(derivative)
        return cast("T", jax.vmap(derivative, in_axes=(0, None))(points, arguments))

    return jax.jit(evaluate)


@dataclass(frozen=True, eq=False, kw_only=True)
class _DynamicLiquidProbe:
    """Bind model structure while receiving every current probe fill dynamically."""

    func: Callable[..., ArrayTree]
    """The model function differentiated with respect to its liquid argument."""
    liquid_name: str
    """Name of the liquid argument replaced by each probe point."""
    scalar: bool
    """Whether to normalize the budget output to a scalar before differentiation."""

    # keyword-only-exempt: library-callback=jax.grad
    def __call__(self, value: FloatND, arguments: EconFunctionKwargs) -> ArrayTree:
        result = self.func(**{**arguments, self.liquid_name: value})
        return jnp.asarray(result).reshape(()) if self.scalar else result


def _evaluate_liquid_probe_program[T](
    *,
    program: Callable[[FloatND, EconFunctionKwargs], T],
    points: FloatND,
    arguments: EconFunctionKwargs,
) -> T:
    """Evaluate current fills and release failed abstract specializations.

    The result is the budget's derivative at every point, or a continuation law's
    Jacobian tree.
    """
    try:
        return program(points, arguments)
    except Exception:
        if isinstance(program, _CacheClearing):
            program.clear_cache()
        raise


@runtime_checkable
class _CacheClearing(Protocol):
    """A compiled program that can drop the specializations it has cached."""

    def clear_cache(self) -> None:
        """Drop every cached specialization."""
        ...


@dataclass(eq=False, kw_only=True)
class _DeferredProbe:
    """A configured probe that decides per draw whether to evaluate itself.

    The engine calls it as `check(flat_params=...)` on every draw; the schedule
    decides which draws reach the probe. `checked` is the one mutable field —
    a `first_solve` schedule stops after the first evaluated draw.
    """

    probe: Callable[..., None]
    """The probe, called with the regime name, completed fills, and `bound`."""

    regime_name: RegimeName
    """Name of the regime whose params complete the fills."""

    probe_arguments: _ProbeArguments
    """The probe's argument classification, completed per draw."""

    probe_schedule: Literal["first_solve", "every_solve", "never"]
    """Which draws the probe evaluates."""

    bound: MappingProxyType[str, _ProbeBinding]
    """The probe's structural arguments, fixed at model build."""

    checked: bool = False
    """Whether a draw has been evaluated yet."""

    def __call__(self, *, flat_params: FlatParams) -> None:
        """Evaluate the probe on this draw when the schedule asks for it."""
        self.evaluate(flat_params=flat_params, materializer=None)

    def evaluate(
        self,
        *,
        flat_params: FlatParams,
        materializer: DatedGenerationMaterializer | None,
    ) -> None:
        """Keep the same schedule/check identity with an optional call-local owner."""
        if self.probe_schedule == "never" or (
            self.probe_schedule == "first_solve" and self.checked
        ):
            return
        if (
            materializer is not None
            and self.probe is not _fail_if_budget_nonaffine_in_liquid
            and self.probe
            is not _fail_if_liquid_reading_next_state_varies_within_interval
        ):
            raise UnsupportedOperationError(
                "Owned NB validation requires the original scalar affinity probe."
            )
        self.probe(
            regime_name=self.regime_name,
            probe_arguments=self.probe_arguments.with_params(
                flat_params=flat_params, regime_name=self.regime_name
            ),
            **self.bound,
            **({} if materializer is None else {"materializer": materializer}),
        )
        self.checked = True


def _probe_arguments(*, context: SolverBuildContext) -> _ProbeArguments:
    """Classify a regime's probe arguments from its grids and annotations."""
    annotation_sources = _probe_annotation_sources(context=context)
    return _ProbeArguments(
        int_arg_values=MappingProxyType(
            {
                **_int_probe_arg_values(context.grids),
                "period": context.regimes_to_active_periods[context.regime_name],
            }
        ),
        array_float_arg_names=_array_float_arg_names(functions=annotation_sources),
        array_arg_ranks=_indexed_arg_ranks(functions=annotation_sources),
        annotated_int_arg_names=_annotated_int_arg_names(functions=annotation_sources),
        bool_arg_names=_annotated_bool_arg_names(functions=annotation_sources),
        mapping_leaf_arg_names=_annotated_mapping_leaf_arg_names(
            functions=annotation_sources
        ),
    )


def _probe_fill(
    *,
    name: str,
    fill: float,
    int_arg_names: frozenset[ReferenceName],
    array_float_arg_names: frozenset[ReferenceName] = frozenset(),
    array_arg_ranks: Mapping[str, int] = MappingProxyType({}),
    bool_arg_names: frozenset[ReferenceName] = frozenset(),
    mapping_leaf_arg_names: frozenset[ReferenceName] = frozenset(),
    param_values: Mapping[QualifiedName, ParamsLeaf] = MappingProxyType({}),
    array_floats: bool = False,
    array_rank: int = 1,
    leaf_rank: int = 1,
) -> ParamsLeaf:
    """Build a probe argument, preferring the model's own parameter value.

    A parameter the model declares answers with its own value: the probes run only
    once the tax schedules, interpolation tables, and coefficients the budget reads
    are in hand. That is both simpler and stricter than any synthetic stand-in — the
    probe differentiates the real bracket structure rather than a fabricated
    one-bracket schedule.

    Everything else is synthesized from what its consumers declare. The
    remaining arguments are the states and actions the probe is sweeping, plus
    any DAG intermediate the pruned budget leaves unbound.

    Integer-coded arguments — discrete states/actions (their grids are
    `DiscreteGrid`s), the period index, and anything its consumers annotate with
    an integer dtype — receive an int32 scalar fill so runtime type contracts
    accept the probe. A boolean-annotated argument receives a boolean fill, taken
    from whether `fill` is at the probes' high level, so the sweep the callers
    already run puts every declared gate on both of its branches. Every other
    argument receives the float fill, shaped by what its annotations declare:

    - a 0-d scalar by default (a scalar parameter such as a rate);
    - an array of unit-length axes when `name` is in `array_float_arg_names` (an
      array-valued schedule parameter: JAX clamps a scalar index into a
      unit-length axis, and equal-length interpolation rows stay consistent).

    An array annotation says the argument is an array, not how many axes it has —
    the rank-polymorphic aliases cover a schedule read with one index and a table
    read with two alike. The axis count is therefore taken from
    `array_arg_ranks`, which reads it off how the argument's consumers subscript
    it, and raised to `array_rank` for an argument that mapping does not reach.

    A grouped param's *contents* take `leaf_rank` instead, and escalate on their
    own: what a group holds is reached through a local binding rather than
    through the parameter's own name, so no subscript names it and nothing infers
    its depth. Tying it to `array_rank` would over-rank the plain array
    parameters, some of which are strict about their own — an interpolation table
    has to stay 1-D.

    `array_floats` forces the array fill on every float argument — the coarse
    whole-DAG fallback kept for a DAG whose per-argument annotations do not
    resolve, at the cost of violating any genuinely 0-d parameter.
    """
    if name in param_values:
        return param_values[name]
    if name in mapping_leaf_arg_names:
        return _ProbeMappingLeaf(jnp.full((1,) * leaf_rank, fill))
    axes = (1,) * max(array_rank, array_arg_ranks.get(name, 1))
    if name in bool_arg_names:
        return jnp.asarray(fill >= _PROBE_FILL_HIGH, dtype=jnp.bool_)
    if name in int_arg_names or name == "period":
        return jnp.asarray(round(fill), dtype=jnp.int32)
    if array_floats or name in array_float_arg_names:
        return jnp.full(axes, fill)
    return jnp.asarray(fill)


def _bind_all_but_liquid(
    *, positional: _PositionalCall, liquid_pos: int, bound: list[ParamsLeaf]
) -> _LiquidBoundLaw:
    """Return the law as a function of the liquid argument alone.

    Every other argument is a probe fill that is constant for the rung, so binding
    them here leaves one differentiable scalar argument and lets a whole rung's
    probe points go through a single vectorized call.
    """
    return _LiquidBoundLaw(
        positional=positional, liquid_pos=liquid_pos, bound=tuple(bound)
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class _LiquidBoundLaw:
    """A positional law with every argument but the liquid one bound."""

    positional: _PositionalCall
    """The law, taking its arguments positionally in signature order."""

    liquid_pos: int
    """Position of the liquid argument."""

    bound: tuple[ParamsLeaf, ...]
    """One value per argument; the liquid slot is replaced at each call."""

    def __call__(self, liquid_value: FloatND) -> ArrayTree:
        """Evaluate the law at `liquid_value` with the other arguments bound."""
        called = list(self.bound)
        called[self.liquid_pos] = liquid_value
        return self.positional(*called)


# Interior liquid values the constancy probe differentiates at. Chosen away from
# grid nodes and from one another so a dependence that vanishes at a node, or is
# constant across a coincidence of two points, still registers.
_LIQUID_PROBE_POINTS: tuple[float, ...] = (0.37, 1.63, 2.71)


# `(array_floats, array_rank, leaf_rank)` triples the probes try, in order.
_PROBE_FILL_RUNGS: tuple[tuple[bool, int, int], ...] = (
    (False, 1, 1),
    (False, 1, 2),
    (False, 1, 3),
    (False, 2, 2),
    (False, 3, 3),
    (True, 1, 1),
)


def _evaluate_on_first_workable_fill[T](evaluate: Callable[..., T]) -> T:
    """Evaluate the original unbudgeted fill ladder with its existing API."""
    return _evaluate_probe_fill_rungs(evaluate=evaluate, materializer=None)


def _evaluate_probe_fill_rungs[T](
    *,
    evaluate: Callable[..., T],
    materializer: DatedGenerationMaterializer | None = None,
) -> T:
    """Evaluate a probe on the first fill shape its DAG accepts.

    An array-typed parameter's *axis count* is not declared anywhere — the
    rank-polymorphic aliases cover a schedule read with one index and a table
    read with two alike — so the probe reads it off the consumers' subscripts
    and keeps a ladder for whatever that misses:

    - the per-argument inferred ranks, floored at one unit axis, then at two,
      then at three, each rung leaving scalar-annotated arguments 0-d;
    - a final coarse rung that drops the per-argument classification and gives
      every float argument an array fill, which evaluates a DAG whose
      annotations do not resolve at the cost of violating any genuinely 0-d
      parameter.

    Raise the *first* rung's error when no rung evaluates: the coarse rung
    violates every 0-d parameter by construction, so its error names an argument
    the model declares correctly and hides the failure that actually blocks the
    probe.
    """
    errors: list[Exception] = []
    for array_floats, array_rank, leaf_rank in _PROBE_FILL_RUNGS:
        try:
            return evaluate(
                array_floats=array_floats,
                array_rank=array_rank,
                leaf_rank=leaf_rank,
            )
        except Exception as rung_error:
            if materializer is not None and isinstance(
                rung_error, ExecutionPlanningError
            ):
                raise
            errors.append(rung_error)
    raise errors[0]


def _probe_annotation_sources(
    *,
    context: SolverBuildContext,
) -> MappingProxyType[str, UserFunction]:
    """The functions whose signatures classify this regime's probe fills."""
    return _annotation_source_functions(
        functions=context.functions,
        transitions=context.transitions,
        user_regimes=context.user_regimes,
        compute_regime_transition_probs=context.compute_regime_transition_probs,
    )


def _indexed_arg_ranks(
    *,
    functions: Mapping[str, UserFunction],
) -> MappingProxyType[str, int]:
    """Axis count per leaf parameter, read off how its consumers subscript it.

    An array annotation states that a parameter is an array, not how many axes it
    has: the rank-polymorphic aliases cover a schedule row read with one index
    and an age-by-status table read with two alike. Nothing else at build time
    knows either — the values arrive with the params, long after the kernels are
    compiled — so the probe reads the depth from the one place it is written
    down, the subscripts in the consumers' own source.

    A parameter read at several depths takes the deepest, which the shallower
    reads survive: indexing a 2-D fill once yields a row. A parameter no consumer
    subscripts is absent, leaving it to the caller's default.

    Keys are signature names, so a parameter processing has qualified is matched
    to the body's own spelling of it by suffix. A function whose source is
    unavailable contributes nothing rather than failing the build: the fill-rank
    ladder is what covers the gap.
    """
    ranks: dict[str, int] = {}
    for func_name, func in functions.items():
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        except OSError, TypeError, SyntaxError:
            continue
        param_names = tuple(inspect.signature(func).parameters)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            if not isinstance(node.value, ast.Name):
                continue
            read = node.value.id
            n_indices = len(node.slice.elts) if isinstance(node.slice, ast.Tuple) else 1
            for param in param_names:
                if param != read and not param.endswith(f"__{read}"):
                    continue
                for spelling in _probe_arg_spellings(
                    func_name=func_name, arg_name=param
                ):
                    ranks[spelling] = max(ranks.get(spelling, 1), n_indices)
    return MappingProxyType(ranks)


def _resolved_annotation(annotation: AnnotationForm) -> AnnotationForm | None:
    """The annotation object a probe can read a fill contract off, or `None`.

    A function composed for the DAG carries its parameters' types as names rather
    than objects, and those parameters are exactly the leaves a probe fills — so
    a name is looked up in the vocabulary the annotations are written in.

    `None` means the annotation decides nothing: the parameter is unannotated, or
    names something outside that vocabulary. Callers skip it rather than read it
    as evidence of a different type, which would withdraw the classification
    every other consumer of that parameter agrees on.
    """
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        named = _PROBE_ANNOTATION_VOCABULARY.get(annotation)
        if named is None:
            return None
        annotation = named
    return getattr(annotation, "__value__", annotation)


def _probe_arg_spellings(
    *, func_name: FunctionName, arg_name: ReferenceName
) -> tuple[str, str]:
    """Both names a classified parameter can reach the probe under.

    Processing qualifies a function's parameters with the function's own name,
    so a classification read off an unprocessed function has to answer to the
    qualified spelling too — the probe fills the composed DAG's leaves, which
    carry whichever of the two that function contributed.
    """
    return (arg_name, f"{func_name}__{arg_name}")


@runtime_checkable
class _DeclaresFunctions(Protocol):
    """A regime as the probe classifiers read it: through its declared functions."""

    @property
    def functions(self) -> Mapping[FunctionName, _FunctionEntry]:
        """The regime's functions, keyed by name."""
        ...


def _annotation_source_functions(
    *,
    functions: Mapping[str, UserFunction],
    transitions: Mapping[RegimeName, Mapping[TransitionFunctionName, UserFunction]],
    user_regimes: Mapping[RegimeName, _DeclaresFunctions] = MappingProxyType({}),
    compute_regime_transition_probs: UserFunction | None = None,
) -> MappingProxyType[str, UserFunction]:
    """Every function whose signature classifies a probe fill.

    A probe differentiates a composed DAG, and `concatenate_functions` carries
    neither its leaves' annotations nor their bodies onto the composed signature
    — so the fill contract has to be read off the functions the DAG was built
    from. Three sources beyond the regime's own functions are needed, each
    because a probe composes something the econ functions alone do not cover:

    - the laws of motion and the regime transition, which the constancy probe
      differentiates directly;
    - the reachable target regimes' functions, which those laws are composed
      against — a continuation reads the child regime's own DAG, so a schedule
      or flag only the child declares still reaches this regime's probe.

    A regime the transition cannot reach contributes nothing: its declarations
    never appear in this continuation, and folding them in would let an unrelated
    name collide with one that does. Entries are keyed so that no source
    displaces another.
    """
    sources: dict[FunctionName | QualifiedName, UserFunction] = dict(functions)
    for target, target_laws in transitions.items():
        for law_name, candidate in target_laws.items():
            func = getattr(candidate, "func", candidate)
            if callable(func):
                sources[f"{target}__{law_name}"] = func
        target_functions = getattr(user_regimes.get(target), "functions", {})
        for func_name, func in target_functions.items():
            if callable(func) and func_name not in sources:
                sources[func_name] = func
    if compute_regime_transition_probs is not None:
        sources["compute_regime_transition_probs"] = compute_regime_transition_probs
    return MappingProxyType(sources)


def _array_float_arg_names(
    *,
    functions: Mapping[str, UserFunction],
) -> frozenset[str]:
    """Leaf parameter names that must be filled as unit-1D arrays, from annotations.

    A budget DAG mixes 0-d scalar parameters with array-valued schedule tables.
    The probe reads each leaf function's parameter annotations — jaxtyping array
    aliases carry a shape (`dim_str`) that strips to empty for a 0-d scalar
    (`ScalarFloat`) and to a non-empty spec for an array (`Float1D`, `FloatND`,
    `ContinuousState`). A parameter any consumer annotates as a 0-d scalar stays
    scalar-filled; otherwise an array-typed parameter is filled unit-1D so a
    scalar index clamps into its table. A parameter whose annotation carries no
    resolvable shape is left to the scalar default.
    """
    scalar_args: set[ReferenceName] = set()
    array_args: set[ReferenceName] = set()
    for func_name, func in functions.items():
        for arg_name, param in inspect.signature(func).parameters.items():
            resolved = _resolved_annotation(param.annotation)
            dim_str = getattr(resolved, "dim_str", None)
            if dim_str is None:
                continue
            spellings = _probe_arg_spellings(func_name=func_name, arg_name=arg_name)
            (scalar_args if dim_str.strip() == "" else array_args).update(spellings)
    return frozenset(array_args - scalar_args)


def _annotated_int_arg_names(
    *,
    functions: Mapping[str, UserFunction],
) -> frozenset[str]:
    """Leaf parameter names that must be filled as integers, from annotations.

    Backing a `DiscreteGrid` is one way for an argument to be integer-coded, not
    the only one: a DAG intermediate can compute an integer code, and a flat
    parameter can be an integer threshold such as an age. Neither has a grid, so
    both are classified from the jaxtyping dtype their consumers annotate.
    """
    return _annotated_dtype_arg_names(functions=functions, dtype_prefix="int")


def _annotated_bool_arg_names(
    *,
    functions: Mapping[str, UserFunction],
) -> frozenset[str]:
    """Leaf parameter names that must be filled as booleans, from annotations.

    A predicate computed inside the DAG — whether a threshold is crossed, whether
    a means test binds — reaches the probe as an ordinary leaf. JAX would accept a
    numeric fill for it, but the declared annotation does not, so the flag is
    classified and filled as a boolean.
    """
    return _annotated_dtype_arg_names(functions=functions, dtype_prefix="bool")


def _annotated_mapping_leaf_arg_names(
    *,
    functions: Mapping[str, UserFunction],
) -> frozenset[str]:
    """Leaf parameter names declared as grouped param mappings, from annotations.

    A schedule the model author groups under one name — a tax table's brackets,
    rates, and intercepts — reaches the DAG as a single `MappingLeaf` argument
    rather than as separate arrays, so no array fill satisfies it.
    """
    leaf_args: set[ReferenceName] = set()
    other_args: set[ReferenceName] = set()
    for func_name, func in functions.items():
        for arg_name, param in inspect.signature(func).parameters.items():
            resolved = _resolved_annotation(param.annotation)
            if resolved is None:
                continue
            declares = isinstance(resolved, type) and issubclass(
                resolved, UserMappingLeaf
            )
            spellings = _probe_arg_spellings(func_name=func_name, arg_name=arg_name)
            (leaf_args if declares else other_args).update(spellings)
    return frozenset(leaf_args - other_args)


def _annotated_dtype_arg_names(
    *,
    functions: Mapping[str, UserFunction],
    dtype_prefix: str,
) -> frozenset[str]:
    """Leaf parameter names every consumer annotates with the given dtype family.

    The jaxtyping alias on each consumer's parameter is the same declaration the
    runtime type check enforces, so it is what the fill has to satisfy. A name
    whose annotations disagree is excluded and keeps the float default: any fill
    would violate one of its consumers, and the disagreement is the model
    author's to resolve rather than the probe's to guess.
    """
    matching: set[str] = set()
    other: set[str] = set()
    for func_name, func in functions.items():
        for arg_name, param in inspect.signature(func).parameters.items():
            resolved = _resolved_annotation(param.annotation)
            dtypes = getattr(resolved, "dtypes", None)
            if dtypes is None:
                continue
            declares = all(dtype.startswith(dtype_prefix) for dtype in dtypes)
            spellings = _probe_arg_spellings(func_name=func_name, arg_name=arg_name)
            (matching if declares else other).update(spellings)
    return frozenset(matching - other)


def _int_code_sweeps(
    *,
    arg_names: tuple[ReferenceName, ...],
    int_arg_values: Mapping[str, tuple[int, ...]],
) -> tuple[MappingProxyType[str, int], ...]:
    """One-at-a-time overrides sweeping each discrete argument's actual codes.

    Each assignment pins one discrete argument while the others keep their fills.
    Time-reading probes repeat these assignments at every active source period;
    synthetic time fills could skip an active row or inspect an inactive one.
    """
    assignments: list[MappingProxyType[str, int]] = [MappingProxyType({})]
    for name in arg_names:
        if name == "period":
            continue
        codes = int_arg_values.get(name, ())
        assignments.extend(MappingProxyType({name: code}) for code in codes)
    if "period" in arg_names and "period" in int_arg_values:
        return tuple(
            MappingProxyType({**assignment, "period": period})
            for period in int_arg_values["period"]
            for assignment in assignments
        )
    return tuple(assignments)


def _int_probe_arg_values(
    grids: Mapping[StateOrActionName, Grid],
) -> MappingProxyType[str, tuple[int, ...]]:
    """Actual grid codes of the regime's integer-coded states and actions.

    The probes sweep each integer-coded argument over these codes one at a time
    (holding the other fills fixed), so a dependence that is dead at the synthetic
    fill values but live at another valid code is still detected.
    """
    return MappingProxyType(
        {
            name: tuple(int(code) for code in grid.to_jax())
            for name, grid in grids.items()
            if isinstance(grid, DiscreteGrid)
        }
    )


def _fail_if_liquid_reading_next_state_varies_within_interval(
    *,
    continuation_plan: ContinuationPlan,
    liquid_name: str,
    regime_name: RegimeName,
    probe_arguments: _ProbeArguments,
    probe_failure: Literal["reject", "assume_declared"] = "reject",
    derivative_programs: Mapping[_ContinuationLaw, _LawProbeProgram] | None = None,
    materializer: DatedGenerationMaterializer | None = None,
) -> None:
    """Reject a carried-state law that varies smoothly in the liquid state.

    When a carried state's law of motion reads the current liquid (Euler) state,
    NBEGM binds the liquid state to each interval's node and reuses that
    continuation row across the interval — exact only when the law's liquid
    dependence is piecewise-constant (a level switched at a declared cliff, so its
    derivative in the liquid state is zero between breakpoints). A smooth (affine or
    curved) dependence makes the midpoint-bound row wrong for the interval's other
    liquid points, so it is rejected on any draw selected by the probe schedule.

    The probe evaluates each liquid-reading law's first liquid-derivative at a few
    interior points, with every declared parameter set to its own value and the
    remaining arguments filled by several constant and ramped assignments (the ramps
    activate monotone binary gates like an age cutoff that a symmetric fill would
    leave on its zero branch); each integer-coded argument is additionally swept over
    its grid's actual codes one at a time, so a dependence gated on a specific
    discrete cell is still sampled. The probe is a finite diagnostic, not a
    certificate — a smooth dependence vanishing at every probed point passes
    undetected. A law the probe cannot differentiate is refused: the interval path's
    constancy precondition would otherwise go unverified.
    """
    tol = 1e-6

    for target, next_state_func in (
        (name, func)
        for name in continuation_plan.stateful_targets
        for func in (
            continuation_plan.child_reads[name].next_state_func,
            continuation_plan.child_reads[name].euler_state_func,
        )
    ):
        worst = _max_abs_first_liquid_derivative(
            func=next_state_func,
            liquid_name=liquid_name,
            regime_name=regime_name,
            probe_arguments=probe_arguments,
            probe_failure=probe_failure,
            derivative_programs=derivative_programs,
            materializer=materializer,
        )
        if worst is not None and worst > tol:
            msg = (
                "NBEGM binds the liquid state to each interval's node when a carried "
                "state's law reads it, exact only if that dependence is "
                "piecewise-constant (switched at a declared cliff). In regime "
                f"{regime_name!r} the law of motion for {target!r} varies smoothly in "
                f"the liquid state {liquid_name!r} (nonzero derivative between "
                "breakpoints), so the midpoint-bound continuation row is wrong within "
                f"the interval. Declare the switch as a breakpoint, or keep {target!r} "
                "independent of the current liquid state."
            )
            raise RegimeInitializationError(msg)

    # The regime-transition probabilities enter the continuation the same way: the
    # target blend must be constant within each declared interval for the
    # midpoint-bound row to be exact across the interval.
    worst = _max_abs_first_liquid_derivative(
        func=continuation_plan.compute_regime_transition_probs,
        liquid_name=liquid_name,
        regime_name=regime_name,
        probe_arguments=probe_arguments,
        probe_failure=probe_failure,
        derivative_programs=derivative_programs,
        materializer=materializer,
    )
    if worst is not None and worst > tol:
        msg = (
            "NBEGM binds the liquid state to each interval's node when the regime-"
            "transition probabilities read it, exact only if that dependence is "
            "piecewise-constant (switched at a declared cliff). In regime "
            f"{regime_name!r} the regime-transition probabilities vary smoothly in "
            f"the liquid state {liquid_name!r} (nonzero derivative between "
            "breakpoints), so the midpoint-bound continuation row is wrong within "
            "the interval. Declare the switch as a breakpoint, or keep the regime "
            "transition independent of the current liquid state."
        )
        raise RegimeInitializationError(msg)


def _max_abs_first_liquid_derivative(
    *,
    func: _ContinuationLaw,
    liquid_name: str,
    regime_name: RegimeName,
    probe_arguments: _ProbeArguments,
    probe_failure: Literal["reject", "assume_declared"],
    derivative_programs: Mapping[_ContinuationLaw, _LawProbeProgram] | None = None,
    materializer: DatedGenerationMaterializer | None = None,
) -> float | None:
    """Largest absolute liquid derivative of `func` over the probe's fills.

    The composed law returns the child's whole carried-state vector, so the
    Jacobian in the liquid argument is probed and the max taken over all outputs.
    `None` when the law does not read the liquid state, or when it cannot be
    differentiated and the model author asserts constancy.
    """
    arg_names = tuple(inspect.signature(func).parameters)
    if liquid_name not in arg_names:
        return None

    try:
        return _evaluate_probe_fill_rungs(
            evaluate=functools.partial(
                _worst_liquid_jacobian,
                positional=_PositionalCall(func=func, arg_names=arg_names),
                arg_names=arg_names,
                liquid_pos=arg_names.index(liquid_name),
                probe_arguments=probe_arguments,
                program=(
                    derivative_programs[func]
                    if derivative_programs is not None
                    else None
                ),
                materializer=materializer,
            ),
            materializer=materializer,
        )
    except Exception as probe_error:
        if materializer is not None and isinstance(probe_error, ExecutionPlanningError):
            raise
        msg = (
            f"NBEGM could not verify that a liquid-reading law in regime "
            f"{regime_name!r} is piecewise-constant in the liquid state "
            f"{liquid_name!r}: the constancy probe failed to "
            "differentiate it on scalar inputs "
            f"({type(probe_error).__name__}: {probe_error}). The interval "
            "path binds one continuation row per interval, which is exact "
            "only for an interval-constant law."
        )
        if probe_failure == "assume_declared":
            warnings.warn(
                msg + " Solving anyway (`probe_failure='assume_declared'`): "
                "the model author asserts interval-constancy; validate the "
                "solve against an independent reference.",
                stacklevel=2,
            )
            return None
        raise RegimeInitializationError(
            msg + " Restructure the law so it evaluates on scalar inputs "
            "(use `jnp.where` instead of Python branches), set "
            "`probe_failure='assume_declared'` to assert constancy yourself, "
            "or use the brute-force solver for this regime."
        ) from probe_error


@dataclass(frozen=True, eq=False, kw_only=True)
class _PositionalCall:
    """A keyword-bound function called with its arguments in signature order.

    The arguments are the probe's own fills (parameter leaves, integer codes) and
    the varied liquid value.
    """

    func: _ContinuationLaw
    """The function, bound by parameter name."""

    arg_names: tuple[ReferenceName, ...]
    """Its parameter names, in signature order."""

    def __call__(self, *args: ParamsLeaf) -> ArrayTree:
        """Call `func` with `args` bound to its parameters by position."""
        return self.func(**dict(zip(self.arg_names, args, strict=True)))


def _worst_liquid_jacobian(
    *,
    array_floats: bool,
    array_rank: int,
    leaf_rank: int,
    positional: _PositionalCall,
    arg_names: tuple[ReferenceName, ...],
    liquid_pos: int,
    probe_arguments: _ProbeArguments,
    program: _LawProbeProgram | None = None,
    materializer: DatedGenerationMaterializer | None = None,
) -> float:
    """Largest absolute liquid-Jacobian entry over one fill rung's assignments."""
    if materializer is not None and program is None:
        raise UnsupportedOperationError(
            "Owned NB probes require the original prepared derivative programs."
        )
    worst = 0.0
    samples = evaluate_probe_stage(
        function=functools.partial(probe_law_samples, values=_LIQUID_PROBE_POINTS),
        arguments={},
        materializer=materializer,
    )
    sweeps = tuple(
        _int_code_sweeps(
            arg_names=arg_names,
            int_arg_values=probe_arguments.int_arg_values,
        )
    )
    for fills in _fill_assignments(len(arg_names)):
        for overrides in sweeps:
            args: list[ParamsLeaf] = [
                probe_arguments.fill(
                    name=name,
                    fill=fill,
                    array_floats=array_floats,
                    array_rank=array_rank,
                    leaf_rank=leaf_rank,
                    materializer=materializer,
                )
                if name not in overrides
                else evaluate_probe_stage(
                    function=functools.partial(probe_int_fill, value=overrides[name]),
                    arguments={},
                    materializer=materializer,
                )
                for name, fill in zip(arg_names, fills, strict=True)
            ]
            # Every argument but the liquid one is constant for this rung,
            # so all probe points go through one compiled, vectorized call.
            jac = (
                evaluate_probe_stage(
                    function=functools.partial(
                        _evaluate_liquid_probe_program, program=program
                    ),
                    arguments={
                        "points": samples,
                        "arguments": {
                            name: arg
                            for position, (name, arg) in enumerate(
                                zip(arg_names, args, strict=True)
                            )
                            if position != liquid_pos
                        },
                    },
                    materializer=materializer,
                )
                if program is not None
                else jax.jit(
                    jax.vmap(
                        jax.jacfwd(
                            _bind_all_but_liquid(
                                positional=positional,
                                liquid_pos=liquid_pos,
                                bound=args,
                            )
                        )
                    )
                )(samples)
            )
            leaves = jax.tree_util.tree_leaves(jac)
            worst = max(
                [
                    worst,
                    *(
                        float(jnp.max(jnp.abs(leaf)))
                        if materializer is None
                        # Owned leaves are read on the host; |.| and max are exact.
                        else max(abs(v) for v in jax.device_get(leaf).ravel().tolist())
                        for leaf in leaves
                    ),
                ]
            )
    return worst


def _fill_assignments(n_args: int) -> tuple[tuple[float, ...], ...]:
    """Constant and ramped fill vectors the constancy probe assigns to a law.

    The ramps activate monotone binary gates like an age cutoff that a symmetric
    fill would leave on its zero branch.
    """
    constant_1 = tuple(1.0 for _ in range(n_args))
    constant_3 = tuple(3.0 for _ in range(n_args))
    ramp_up = tuple(1.0 + 2.0 * position for position in range(n_args))
    ramp_down = tuple(reversed(ramp_up))
    return (constant_1, constant_3, ramp_up, ramp_down)


def _collect_nbegm_schedule_spec(
    *,
    context: SolverBuildContext,
    budget_target: FunctionName,
    continuous_state: StateName,
    consumption_action_name: ActionName,
    probe_failure: Literal["reject", "assume_declared"] = "reject",
    probe_schedule: Literal["first_solve", "every_solve", "never"] = "never",
    materializer: DatedGenerationMaterializer | None = None,
) -> _NBEGMScheduleSpec:
    """Collect a regime's piecewise-affine schedules into one breakpoint partition.

    A regime may declare several schedules, each bracketing on its own monotone
    income variable (taxable income, MAGI, …); every threshold becomes a
    breakpoint source. Each source maps to its per-ride-along-cell asset preimage
    in its own variable, and the sources merge into one sorted liquid partition.
    The budget node (`budget_target`) is composed once as a function of the liquid
    state, read per interval to recover the active affine segment.
    """
    import inspect  # noqa: PLC0415

    from _lcm.egm.nbegm import collect_nbegm_metadata  # noqa: PLC0415

    user_functions = context.solve_functions
    registry = collect_nbegm_metadata(functions=user_functions)
    # Zero declared schedules produce an empty breakpoint partition: one
    # interval covering the whole liquid axis, solved as plain EGM.
    schedules = registry.piecewise_affine_schedules
    state_names = context.state_action_space.state_names
    # The Euler axis is one continuous state, not the first state axis: the
    # canonical order leads with discrete states, so a ride-along co-state sorts
    # ahead of the liquid axis. The remaining continuous states — a co-state (AIME)
    # or stochastic processes — ride along, integrated by the continuation reader.
    # When the regime carries more than one continuous state the Euler axis is named
    # via the solver's `continuous_state`; a single continuous state is the liquid
    # axis unambiguously. A schedule on the liquid state varies in it directly; a
    # schedule on a derived monotone quantity (gross income, MAGI) maps each
    # threshold to a per-ride-along-cell asset preimage.
    liquid_state_name = resolve_liquid_state_name(
        context=context, declared=continuous_state
    )
    ride_along_state_names = tuple(
        name for name in state_names if name != liquid_state_name
    )
    has_derived = any(schedule.variable != liquid_state_name for schedule in schedules)
    if has_derived and not ride_along_state_names:
        derived_vars = tuple(
            schedule.variable
            for schedule in schedules
            if schedule.variable != liquid_state_name
        )
        msg = (
            f"NBEGM schedule varies in the derived quantity/quantities "
            f"{derived_vars} but the regime has no ride-along co-state; a derived "
            "schedule maps thresholds to per-cell asset preimages and is only "
            "wired on the ride-along path."
        )
        raise RegimeInitializationError(msg)

    _fail_if_single_liquid_schedules_unsupported(
        schedules=schedules,
        ride_along_state_names=ride_along_state_names,
        regime_name=context.regime_name,
    )

    # Cache the composed derived-variable DAG per variable across its breakpoints.
    derived_dags: dict[
        str, tuple[Callable[..., FloatND], tuple[str, ...], tuple[str, ...]]
    ] = {}

    sources: list[_NBEGMSource] = []
    for schedule in schedules:
        is_liquid_direct = schedule.variable == liquid_state_name
        if not is_liquid_direct and schedule.variable not in derived_dags:
            derived_dags[schedule.variable] = _derived_variable_dag(
                variable=schedule.variable,
                context=context,
                state_names=state_names,
                ride_along_state_names=ride_along_state_names,
            )
        dag, params, states_read = (
            (None, (), ()) if is_liquid_direct else derived_dags[schedule.variable]
        )
        sources.extend(
            _NBEGMSource(
                variable=schedule.variable,
                threshold_param_name=f"{schedule.output}__{bracket.threshold}",
                kind=bracket.kind,
                equality_owner=bracket.equality_owner,
                derived_of_liquid_dag=dag,
                derived_param_names=params,
                derived_state_names=states_read,
                threshold_index_state=bracket.indexed_by,
                threshold_static_index=bracket.static_index,
                threshold_subkey=bracket.threshold_subkey,
            )
            for bracket in schedule.breakpoints
        )

    # A single discrete action shifting the budget is enveloped over per ride
    # cell; it is neither a state nor a coh param, so exclude it from
    # `coh_param_names` (the envelope core binds it per branch).
    discrete_actions = _ride_discrete_actions(context=context)
    discrete_action_names = frozenset(name for name, _ in discrete_actions)
    coh_dag = concatenate_functions(dict(context.functions), targets=budget_target)
    coh_args = tuple(inspect.signature(coh_dag).parameters)
    coh_param_names = tuple(
        name
        for name in coh_args
        if name not in state_names and name not in discrete_action_names
    )
    utility_dag = concatenate_functions(dict(context.functions), targets="utility")
    # A regime whose discount factor is a DAG function (e.g. a per-preference-type
    # beta indexed by a ride-along state) exposes it as a target; absent that, the
    # default flat `koopmans_aggregator__discount_factor` param drives discounting.
    discount_factor_dag = (
        concatenate_functions(dict(context.functions), targets="discount_factor")
        if "discount_factor" in context.functions
        else None
    )
    # `threshold_param_names` / `breakpoint_kinds` mirror the first schedule and
    # drive the non-ride-along continuous core, which is reached only for a
    # regime with no ride-along axis (a single liquid-direct schedule). With no
    # declared schedules both are empty — the partition is one interval.
    first_breakpoints = schedules[0].breakpoints if schedules else ()
    threshold_param_names = tuple(
        f"{schedules[0].output}__{bp.threshold}" for bp in first_breakpoints
    )
    breakpoint_kinds = tuple(bp.kind for bp in first_breakpoints)
    all_kinds = tuple(bp.kind for schedule in schedules for bp in schedule.breakpoints)
    affinity_check = _budget_affinity_check(
        context=context,
        coh_dag=coh_dag,
        liquid_state_name=liquid_state_name,
        breakpoint_sources=tuple(sources),
        # The pure-jump step reads only intercepts and assumes unit slope.
        require_unit_slope=(
            not ride_along_state_names
            and not has_derived
            and bool(all_kinds)
            and all(kind == "jump" for kind in all_kinds)
        ),
        probe_failure=probe_failure,
        probe_schedule=probe_schedule,
        materializer=materializer,
    )
    return _NBEGMScheduleSpec(
        param_checks=(affinity_check,),
        coh_of_liquid_dag=coh_dag,
        coh_param_names=coh_param_names,
        utility_dag=utility_dag,
        consumption_action_name=consumption_action_name,
        liquid_state_name=liquid_state_name,
        ride_along_state_names=ride_along_state_names,
        liquid_axis_pos=state_names.index(liquid_state_name),
        threshold_param_names=threshold_param_names,
        breakpoint_kinds=breakpoint_kinds,
        sources=tuple(sources),
        discount_factor_dag=discount_factor_dag,
        discrete_actions=discrete_actions,
    )


def _derived_variable_dag(
    *,
    variable: str,
    context: SolverBuildContext,
    state_names: tuple[StateName, ...],
    ride_along_state_names: tuple[StateName, ...],
) -> tuple[Callable[..., FloatND], tuple[str, ...], tuple[str, ...]]:
    """Compose a derived schedule variable and name its params and states read.

    A discrete action the schedule variable reads is bound per branch by the
    envelope (it shifts the breakpoint partition), not read from kwargs, so it
    is neither a state nor a param here.
    """
    dag = concatenate_functions(dict(context.functions), targets=variable)
    dag_params = tuple(inspect.signature(dag).parameters)
    discrete_action_names = frozenset(context.state_action_space.discrete_actions)
    params = tuple(
        name
        for name in dag_params
        if name not in state_names and name not in discrete_action_names
    )
    states_read = tuple(name for name in dag_params if name in ride_along_state_names)
    return dag, params, states_read


def _sorted_thresholds(*, raw: Float1D, order_sensitive: bool) -> Float1D:
    """Sort declared thresholds, poisoning a set that arrives out of order.

    The interval partition is built from the sorted thresholds while the step's
    `jump_mask` and `flat_mask` are Python statics built from the *declared*
    breakpoint order, so the two describe the same partition only while the
    declaration ascends. Thresholds are free parameters, so an estimator draw
    can swap them and leave the jump mask pointing at a kink — the unified step
    would then bridge the real cliff and split a continuous point, with no
    error anywhere. The misaligned partition is made unrepresentable instead:
    the thresholds are NaN-poisoned and the solve's NaN check reports it.

    A schedule whose breakpoints are all of one kind has an order-independent
    mask, so its thresholds sort without the check.

    Args:
        raw: The declared thresholds in declaration order.
        order_sensitive: Whether the step's masks depend on that order.

    Returns:
        The ascending thresholds, or NaN where the declared order is violated.

    """
    if not order_sensitive:
        return jnp.sort(a=raw)
    ascending = jnp.all(jnp.diff(raw) > 0.0)
    return jnp.where(ascending, jnp.sort(a=raw), jnp.nan)


def _fail_if_single_liquid_schedules_unsupported(
    *,
    schedules: tuple[PiecewiseAffineMeta, ...],
    ride_along_state_names: tuple[StateName, ...],
    regime_name: RegimeName,
) -> None:
    """Reject schedule declarations the single-liquid cores cannot represent.

    Without a ride-along axis the interval partition and the step dispatch both
    come from one schedule's breakpoints, so two declarations fall outside what
    the cores solve:

    - **more than one liquid-direct schedule** — only the first schedule's
      thresholds enter the partition, so an interval straddling a second
      schedule's discontinuity is tangented as if the budget were smooth
      across it;
    - **a hard constraint declared alongside a jump** — the mixed kinds route
      to the unified step, which takes no flat-interval mask, so the floor's
      constant-budget plateau reaches an inversion that assumes a strictly
      increasing cash-on-hand.

    Args:
        schedules: The regime's declared piecewise-affine schedules.
        ride_along_state_names: The regime's ride-along co-states; a non-empty
            tuple means the ride path handles the schedules instead.
        regime_name: Name of the regime, for the message.

    Raises:
        RegimeInitializationError: On either unsupported declaration.

    """
    if ride_along_state_names:
        return
    if len(schedules) > 1:
        outputs = tuple(schedule.output for schedule in schedules)
        msg = (
            f"Regime '{regime_name}' declares {len(schedules)} piecewise-affine "
            f"schedules {outputs} on the liquid state. NBEGM's single-liquid "
            "partition is built from one liquid-direct schedule's thresholds, so "
            "the others' breakpoints would not split the budget and an interval "
            "straddling them would be solved as if it were smooth. Merge them "
            "into one schedule, or declare a ride-along co-state."
        )
        raise RegimeInitializationError(msg)
    kinds = tuple(bp.kind for schedule in schedules for bp in schedule.breakpoints)
    if "hard_constraint" in kinds and "jump" in kinds:
        msg = (
            f"Regime '{regime_name}' declares a hard-constraint (floor) "
            f"breakpoint alongside a jump; got breakpoint kinds {kinds}. The "
            "mixed jump-and-kink step carries no flat-interval mask, so the "
            "floor's constant-budget plateau would reach an inversion that "
            "assumes a strictly increasing cash-on-hand. Declare the floor "
            "without a jump, or model the cliff as a continuous kink."
        )
        raise RegimeInitializationError(msg)


def _schedule_kind_flags(
    kinds: tuple[str, ...],
) -> tuple[bool, bool, bool, tuple[bool, ...], tuple[bool, ...] | None]:
    """Classify a schedule's breakpoint kinds into the step-dispatch flags.

    Returns `(is_single_jump, is_multi_jump, is_mixed, jump_mask, flat_mask)`:

    - `is_single_jump` — one jump, the binary recurring case.
    - `is_multi_jump` — every breakpoint a jump, the N-cliff recurring case.
    - `is_mixed` — jumps and kinks together, solved by the unified step.
    - `jump_mask` — per breakpoint, whether it is a jump (for the unified step).
    - `flat_mask` — per interval (N+1), whether a hard-constraint floors it, or
      `None` when no breakpoint is a hard constraint.
    """
    is_single_jump = kinds == ("jump",)
    is_multi_jump = len(kinds) > 1 and all(kind == "jump" for kind in kinds)
    is_mixed = "jump" in kinds and not all(kind == "jump" for kind in kinds)
    jump_mask = tuple(kind == "jump" for kind in kinds)
    has_floor = "hard_constraint" in kinds
    flat_mask = (
        tuple(
            j < len(kinds) and kinds[j] == "hard_constraint"
            for j in range(len(kinds) + 1)
        )
        if has_floor
        else None
    )
    return is_single_jump, is_multi_jump, is_mixed, jump_mask, flat_mask


def _solve_cliffed_budget(
    *,
    next_value: Float1D,
    next_marginal: Float1D,
    liquid: Float1D,
    next_liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: FloatND,
    preferences: Preferences,
    next_liquid: Float1D,
    marginal_return: Float1D,
    coh_slopes: Float1D,
    coh_intercepts: Float1D,
    breakpoints: Float1D,
    is_single_jump: bool,
    is_multi_jump: bool,
    is_mixed: bool,
    jump_mask: tuple[bool, ...],
    flat_mask: tuple[bool, ...] | None,
    feasibility_partition: ResolvedAxisPartition | None = None,
    feasible_interval_mask: BoolND | None = None,
    boundary_savings_targets: Float1D | None = None,
    boundary_next_liquid: Float1D | None = None,
    arithmetic: ComparisonArithmetic = "certified",
    retain_query: bool = False,
    return_owner: bool = False,
    source_query: Float1D | None = None,
    continuation_read: Callable[[FloatND, BoolND], tuple[FloatND, FloatND]]
    | None = None,
) -> _NBEGMStepChannels | _OwnedNBEGMQueryStep:
    """Solve one period of a cliffed single-liquid budget, dispatching on kind.

    Reads the continuation jump-aware at every jump (no bridging), so the solve
    is exact through recurring jumps, not only at a terminal-adjacent period.
    The kind flags come from `_schedule_kind_flags`. The declared law reaches the
    kernels as its landing points on `savings_grid` and their derivative there.
    Returns this period's value, marginal value of liquid, and consumption policy
    on `liquid`.
    """
    if is_single_jump:
        if return_owner:
            raise UnsupportedOperationError(
                "NB source-point native owners require a continuous schedule."
            )
        # A single jump in cash-on-hand is the binary case the case-piece step solves
        # exactly, including its recurring jumped continuation: each interval's
        # affine segment has slope 1, so its intercept is the additive cash-on-hand
        # level on that side of the cliff.
        return nbegm_one_asset_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            next_liquid_grid=next_liquid_grid,
            savings_grid=savings_grid,
            discount_factor=discount_factor,
            preferences=preferences,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            subsidy_when=coh_intercepts[0],
            subsidy_otherwise=coh_intercepts[1],
            asset_limit=breakpoints[0],
            equality_owner="otherwise",
            arithmetic=arithmetic,
            retain_query=retain_query,
            continuation_read=continuation_read,
        )
    if is_multi_jump:
        if return_owner:
            raise UnsupportedOperationError(
                "NB source-point native owners require a continuous schedule."
            )
        # N cliffs: each affine segment has slope 1, so its intercept is the additive
        # cash-on-hand level on that side, and the recurring step resolves every jump
        # (boundary-targeting + jump-aware continuation).
        return nbegm_recurring_jump_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            next_liquid_grid=next_liquid_grid,
            savings_grid=savings_grid,
            discount_factor=discount_factor,
            preferences=preferences,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            subsidy_levels=coh_intercepts,
            jump_breakpoints=breakpoints,
            equality_owner="otherwise",
            arithmetic=arithmetic,
            retain_query=retain_query,
            continuation_read=continuation_read,
        )
    if is_mixed:
        if return_owner:
            raise UnsupportedOperationError(
                "NB source-point native owners require a continuous schedule."
            )
        # Jumps and kinks together: the unified step solves each continuous case by
        # coh inversion and masks across the jumps. The jump_mask is aligned with the
        # sorted breakpoints (the schedule declares its thresholds ascending).
        return nbegm_unified_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            next_liquid_grid=next_liquid_grid,
            savings_grid=savings_grid,
            discount_factor=discount_factor,
            preferences=preferences,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            coh_slopes=coh_slopes,
            coh_intercepts=coh_intercepts,
            breakpoints=breakpoints,
            jump_mask=jump_mask,
            arithmetic=arithmetic,
            retain_query=retain_query,
            continuation_read=continuation_read,
        )
    return nbegm_multi_interval_step(
        next_value=next_value,
        next_marginal=next_marginal,
        liquid_grid=liquid,
        next_liquid_grid=next_liquid_grid,
        savings_grid=savings_grid,
        discount_factor=discount_factor,
        preferences=preferences,
        next_liquid=next_liquid,
        marginal_return=marginal_return,
        coh_slopes=coh_slopes,
        coh_intercepts=coh_intercepts,
        breakpoints=breakpoints,
        flat_interval_mask=flat_mask,
        arithmetic=arithmetic,
        retain_query=retain_query,
        return_owner=return_owner,
        query_grid=source_query,
        continuation_read=continuation_read,
        feasibility_partition=feasibility_partition,
        feasible_interval_mask=feasible_interval_mask,
        boundary_savings_targets=boundary_savings_targets,
        boundary_next_liquid=boundary_next_liquid,
    )


class _NBEGMSourcePoint(NamedTuple):
    """Original source result beside its separate native envelope owner."""

    value: FloatND
    carry: EGMCarry
    native_owner: IntND | None


@dataclass(frozen=True, eq=False, kw_only=True)
class _NBEGMContinuousCore:
    """The jittable continuous-schedule EGM core of one solver group.

    Reads the schedule's thresholds as liquid breakpoints, recovers the active
    affine cash-on-hand segment per interval by differentiating the composed
    `coh` at each interval's representative, and runs the kind-appropriate EGM
    step.
    """

    savings_grid: Float1D
    """Exogenous post-decision savings grid the step inverts on."""

    build_preferences: Callable[[EconFunctionKwargs], Preferences]
    """Bind the regime's flat params into its utility, marginal, and inverse maps."""

    schedule_spec: _NBEGMScheduleSpec
    """The composed budget, its thresholds, and their breakpoint kinds."""

    feasibility_constraints: tuple[_NBEGMFeasibilityConstraint, ...]
    """Declared feasibility boundaries resolved against each draw's schedule."""

    envelope_arithmetic: ComparisonArithmetic
    """Arithmetic the upper envelope merges the intervals with."""

    retain_query: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Keep independently declared conditional original-saving rows."""

    retain_policy: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Keep the original selected consumption for current-shock reconstruction."""

    interval_recipe: NBEGMIntervalRecipe | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Canonical physical selection cuts for this target-period source recipe."""

    solve_budget: Callable[..., _NBEGMStepChannels | _OwnedNBEGMQueryStep] = field(
        default=_solve_cliffed_budget,
        metadata={"fingerprint_omit_if_default": True},
    )
    """Actual original step callable selected by the owning source preparation."""

    _evaluate_preimages: (
        Callable[[EconFunctionKwargs], tuple[Float1D, BoolND]] | None
    ) = field(default=None, init=False, metadata={"fingerprint_omit_if_default": True})
    """Exact recipe callback, selected before numerical capture without a surrogate."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_evaluate_preimages",
            None
            if self.interval_recipe is None
            else self.interval_recipe.evaluate_preimages,
        )

    def construct_point(
        self,
        *,
        liquid: Float1D,
        next_liquid_grid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        next_liquid: Float1D,
        marginal_return: Float1D,
        effective_savings_grid: Float1D,
        boundary_savings_targets: Float1D,
        boundary_next_liquid: Float1D,
        source_bound_status: BoundInputStatus | None = None,
        continuation_read: Callable[[FloatND, BoolND], tuple[FloatND, FloatND]]
        | None = None,
        source_query: Float1D | None = None,
        **params: ValueND,
    ) -> _NBEGMSourcePoint | BoundedResult[_NBEGMSourcePoint]:
        """Solve one period of the schedule regime on the liquid grid."""
        kinds = self.schedule_spec.breakpoint_kinds
        is_single_jump, is_multi_jump, is_mixed, jump_mask, flat_mask = (
            _schedule_kind_flags(kinds)
        )
        order_sensitive = len(set(kinds)) > 1
        preferences = self.build_preferences(params)
        coh_of_liquid = _BoundScalarFunction(
            func=self.schedule_spec.coh_of_liquid_dag,
            argument_name=self.schedule_spec.liquid_state_name,
            bound={name: params[name] for name in self.schedule_spec.coh_param_names},
        )

        # Zero declared breakpoints ⇒ an empty partition: one interval covering
        # the whole liquid axis, solved as plain EGM.
        breakpoints = (
            _sorted_thresholds(
                raw=jnp.stack(
                    [params[name] for name in self.schedule_spec.threshold_param_names]
                ),
                order_sensitive=order_sensitive,
            )
            if self.schedule_spec.threshold_param_names
            else jnp.zeros((0,), dtype=liquid.dtype)
        )
        if self._evaluate_preimages is not None:
            cuts, _orientation = self._evaluate_preimages(params)
            breakpoints = _sorted_thresholds(raw=cuts, order_sensitive=order_sensitive)
        midpoints = interval_midpoints(liquid_grid=liquid, breakpoints=breakpoints)

        feasibility = _resolve_nbegm_feasibility(
            constraints=self.feasibility_constraints,
            liquid=liquid,
            schedule_breakpoints=breakpoints,
            schedule_kinds=kinds,
            params=params,
        )
        query_liquid, unsort = _augment_liquid_for_feasibility(
            liquid=liquid,
            feasibility=feasibility,
        )
        coh_slopes, coh_intercepts = interval_segment_coefficients(
            schedule=coh_of_liquid,
            interval_midpoints=midpoints,
        )
        step = self.solve_budget(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid=query_liquid,
            next_liquid_grid=next_liquid_grid,
            savings_grid=effective_savings_grid,
            discount_factor=params["koopmans_aggregator__discount_factor"],
            preferences=preferences,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            coh_slopes=coh_slopes,
            coh_intercepts=coh_intercepts,
            breakpoints=breakpoints,
            is_single_jump=is_single_jump,
            is_multi_jump=is_multi_jump,
            is_mixed=is_mixed,
            jump_mask=jump_mask,
            flat_mask=flat_mask,
            feasibility_partition=feasibility.partition,
            feasible_interval_mask=feasibility.feasible_interval_mask,
            boundary_savings_targets=boundary_savings_targets,
            boundary_next_liquid=boundary_next_liquid,
            arithmetic=self.envelope_arithmetic,
            retain_query=self.retain_query,
            return_owner=self.retain_policy,
            source_query=source_query,
            continuation_read=continuation_read,
        )
        value, marginal, policy = step[:3]
        value_at_liquid = (
            value[unsort][: liquid.shape[0]] if source_query is None else value
        )
        carry = EGMCarry(
            endog_grid=query_liquid if source_query is None else source_query,
            value=value,
            breakpoints=(
                feasibility.carry_boundary_values
                if self.feasibility_constraints
                else None
            ),
            marginal_utility=marginal,
            policy=policy if self.retain_policy else None,
            taste_shock_scale=jnp.asarray(0.0, dtype=value.dtype),
            query_data=_required_nbegm_query_data(step=step)
            if self.retain_query
            else None,
        )
        owner = step.owner if isinstance(step, _OwnedNBEGMQueryStep) else None
        result = _NBEGMSourcePoint(value_at_liquid, carry, owner)
        if source_bound_status is not None:
            return BoundedResult(output=result, bound_status=(source_bound_status,))
        return result

    def __call__(
        self,
        *,
        liquid: Float1D,
        next_liquid_grid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        next_liquid: Float1D,
        marginal_return: Float1D,
        effective_savings_grid: Float1D,
        boundary_savings_targets: Float1D,
        boundary_next_liquid: Float1D,
        source_bound_status: BoundInputStatus | None = None,
        continuation_read: Callable[[FloatND, BoolND], tuple[FloatND, FloatND]]
        | None = None,
        **params: ValueND,
    ) -> _RawNBEGMOutput:
        result = self.construct_point(
            liquid=liquid,
            next_liquid_grid=next_liquid_grid,
            next_value=next_value,
            next_marginal=next_marginal,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            effective_savings_grid=effective_savings_grid,
            boundary_savings_targets=boundary_savings_targets,
            boundary_next_liquid=boundary_next_liquid,
            source_bound_status=source_bound_status,
            continuation_read=continuation_read,
            **params,
        )
        if isinstance(result, BoundedResult):
            return BoundedResult(
                output=(result.output.value, result.output.carry),
                bound_status=result.bound_status,
                query_status=result.query_status,
                continuation_read_counts=result.continuation_read_counts,
            )
        return result.value, result.carry


def _build_nbegm_continuous_core(
    *,
    savings_grid: Float1D,
    functions: EconFunctionsMapping,
    consumption_action: ActionName,
    schedule_spec: _NBEGMScheduleSpec,
    feasibility_constraints: tuple[_NBEGMFeasibilityConstraint, ...] = (),
    envelope_arithmetic: ComparisonArithmetic = "ordinary",
) -> _NBEGMContinuousCore:
    """Build the jittable continuous-schedule EGM core for one continuation target.

    The core reads the schedule's thresholds as liquid breakpoints, recovers the
    active affine cash-on-hand segment per interval by differentiating the composed
    `coh` at each interval's representative, and runs the kind-appropriate EGM step.
    """
    from _lcm.egm.preferences import (  # noqa: PLC0415
        NEWTON_ACTION_FLOOR,
        get_preferences_builder,
        newton_action_ceiling,
    )

    action_upper = newton_action_ceiling(savings_grid)
    return _NBEGMContinuousCore(
        savings_grid=savings_grid,
        build_preferences=get_preferences_builder(
            functions=functions,
            action_name=consumption_action,
            action_lower=NEWTON_ACTION_FLOOR,
            action_upper=action_upper,
        ),
        # Host preconditions remain on the separately returned original checks
        # tuple. They are not called by this numerical core and must not become
        # abstract numerical callback dependencies merely through the spec.
        schedule_spec=replace(schedule_spec, param_checks=()),
        feasibility_constraints=feasibility_constraints,
        envelope_arithmetic=envelope_arithmetic,
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class _BoundScalarFunction:
    """A composed function as a function of one named argument.

    Every other argument is bound at construction — a cell's ride-state
    coordinates, the draw's params, a branch's action codes — so the callable
    is differentiable and vectorizable in the one argument that varies.
    """

    func: Callable[..., FloatND]
    """The composed function, bound by parameter name."""

    argument_name: str
    """Name of the argument each call supplies."""

    bound: EconFunctionKwargs
    """The other arguments, by parameter name."""

    def __call__(self, value: FloatND) -> FloatND:
        """Evaluate the function at `value` with the other arguments bound."""
        return self.func(**{self.argument_name: value}, **self.bound)


def _build_nbegm_continuation_plan(
    *,
    context: SolverBuildContext,
    period: int,
    post_decision_name: FunctionName,
) -> ContinuationPlan:
    """Assemble the period's continuation plan for the ride-along case-piece core."""
    from _lcm.egm.continuation import (  # noqa: PLC0415
        build_continuation_plan,
        get_egm_continuation_targets,
    )

    # A regime running the case-piece solver is non-terminal, so it always has a
    # regime transition; narrow the optional for the continuation reader.
    compute_regime_transition_probs = context.compute_regime_transition_probs
    if compute_regime_transition_probs is None:
        msg = (
            f"NBEGM regime {context.regime_name!r} has no regime transition; the "
            "case-piece solver is for non-terminal regimes only."
        )
        raise RegimeInitializationError(msg)
    # A period-`t` kernel reads its targets' `V_{t+1}` over the ride-along co-states,
    # so with an age-specialized co-state the nodes it interpolates on are period
    # `t + 1`'s, not the representative age's.
    v_interpolation_info = continuation_v_interpolation_info(
        period=period,
        regime_to_v_interpolation_info=context.regime_to_v_interpolation_info,
        period_to_regime_v_interp=context.period_to_regime_v_interp,
    )
    solution_reachability = context.solution_reachability
    targets = (
        ()
        if period == solution_reachability.n_periods - 1
        else solution_reachability.targets(period=period, source=context.regime_name)
    )
    stateful_targets, scalar_targets = get_egm_continuation_targets(
        targets=targets,
        regime_to_v_interpolation_info=v_interpolation_info,
    )
    # A nonlinear certainty equivalent switches the child stochastic-node
    # expectation to the Epstein-Zin power mean. Its risk-aversion coefficient is
    # the flat param `certainty_equivalent__risk_aversion` (the only nonlinear CE,
    # `PowerMean`, takes that single argument); `None` keeps the linear read.
    risk_aversion_param_name = (
        "certainty_equivalent__risk_aversion"
        if _aggregates_nonlinearly(context.certainty_equivalent)
        else None
    )
    # A target's resources map is the child's own, read at its period `t + 1`.
    return build_continuation_plan(
        user_regimes=context.user_regimes_at(period=period + 1),
        laws=context.laws,
        functions=context.functions,
        transitions=context.transitions,
        transition_plans=context.transition_plans,
        stateful_targets=stateful_targets,
        scalar_targets=scalar_targets,
        compute_regime_transition_probs=compute_regime_transition_probs,
        post_decision_name=post_decision_name,
        regime_to_v_interpolation_info=v_interpolation_info,
        age_values=context.ages.values,
        risk_aversion_param_name=risk_aversion_param_name,
        dcegm_query_models=context.dcegm_query_models,
        nbegm_query_models=context.nbegm_query_models,
        source_period=period,
    )


def _solve_ride_along_cell_step(
    *,
    has_jump: bool,
    jump_positions: tuple[JumpPosition, ...],
    cont_value: Float1D,
    cont_marginal: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: FloatND,
    preferences: Preferences,
    coh_slopes: Float1D,
    coh_intercepts: Float1D,
    breakpoints: Float1D,
    extra_savings: Float1D | None = None,
    extra_cont_value: Float1D | None = None,
    inverse_eis: FloatND | None = None,
    arithmetic: ComparisonArithmetic = "certified",
    feasibility_partition: ResolvedAxisPartition | None = None,
    feasible_interval_mask: BoolND | None = None,
    retain_query: bool = False,
    source_query: Float1D | None = None,
    return_owner: bool = False,
) -> _NBEGMStepChannels | _OwnedNBEGMQueryStep:
    """Run one ride-along cell's 1-D case-piece step against savings continuation.

    A pure-kink schedule uses the continuous multi-interval step; a schedule with a
    jump breakpoint uses the unified jump-and-kink step, both reading the expected
    value and marginal already evaluated on the savings grid. The Euler inversion,
    the period value, and the marginal value of liquid all read the regime's own
    utility through the `preferences` bundle (bound to this
    cell). The jump positions locate the jump breakpoints in the sorted partition —
    static for a single variable, a per-cell traced tuple when several variables
    reorder per cell.
    """
    from _lcm.egm.nbegm_step import (  # noqa: PLC0415
        nbegm_multi_interval_step_savings,
        nbegm_unified_step_savings,
    )

    if has_jump and (source_query is not None or return_owner):
        raise UnsupportedOperationError(
            "Owned NB ride point reads require the original continuous savings bank."
        )
    if has_jump:
        if inverse_eis is not None:
            msg = (
                "Epstein-Zin NBEGM does not yet support jump breakpoints; the "
                "unified jump-and-kink step still assumes the additive aggregator. "
                "Use a kink-only schedule (or GridSearch) for this regime."
            )
            raise NotImplementedError(msg)
        return nbegm_unified_step_savings(
            cont_value=cont_value,
            cont_marginal=cont_marginal,
            liquid_grid=liquid_grid,
            savings_grid=savings_grid,
            discount_factor=discount_factor,
            preferences=preferences,
            coh_slopes=coh_slopes,
            coh_intercepts=coh_intercepts,
            breakpoints=breakpoints,
            jump_positions=jump_positions,
            extra_savings=extra_savings,
            extra_cont_value=extra_cont_value,
            arithmetic=arithmetic,
            retain_query=retain_query,
        )
    return nbegm_multi_interval_step_savings(
        cont_value=cont_value,
        cont_marginal=cont_marginal,
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=discount_factor,
        preferences=preferences,
        coh_slopes=coh_slopes,
        coh_intercepts=coh_intercepts,
        breakpoints=breakpoints,
        inverse_eis=inverse_eis,
        arithmetic=arithmetic,
        feasibility_partition=feasibility_partition,
        feasible_interval_mask=feasible_interval_mask,
        retain_query=retain_query,
        query_grid=source_query,
        return_owner=return_owner,
    )


def _ride_along_jump_config(
    *,
    kinds: tuple[str, ...],
    materializer: DatedGenerationMaterializer | None = None,
) -> tuple[BoolND, int, bool, tuple[int, ...], bool]:
    """Derive the merged partition's jump statics from the declared breakpoint kinds.

    Returns the per-breakpoint jump flags, the static jump count, whether any jump
    is present, the declared-order jump positions, and whether the jump positions
    must be recovered per cell.

    The positions are recovered per cell whenever jumps and kinks are mixed, on
    however many variables. A single schedule is enough to reorder them: the
    threshold-to-asset preimage divides by a slope of either sign, so a schedule
    on a decreasing derived variable (a remaining allowance, a distance to a cap)
    maps ascending thresholds to *descending* asset preimages, and the sorted
    order then reverses the declared one.
    """
    jump_flags = tuple(kind == "jump" for kind in kinds)
    n_jumps = sum(jump_flags)
    static_jump_positions = tuple(
        index for index, is_jump in enumerate(jump_flags) if is_jump
    )
    dynamic_jumps = 0 < n_jumps < len(kinds)
    return (
        # dtype pinned so the zero-breakpoint (empty) case stays boolean.
        (
            jnp.asarray(jump_flags, dtype=bool)
            if materializer is None
            else materializer.evaluate_validation(
                function=functools.partial(jnp.asarray, jump_flags, dtype=bool),
                arguments={},
            )
        ),
        n_jumps,
        n_jumps > 0,
        static_jump_positions,
        dynamic_jumps,
    )


def _partition_jumps(
    *,
    preimages: Float1D,
    dynamic_jumps: bool,
    jump_flags: BoolND,
    n_jumps: int,
    static_jump_positions: tuple[int, ...],
) -> tuple[Float1D, tuple[JumpPosition, ...]]:
    """Sort a cell's breakpoint preimages and locate the jumps in the sorted order.

    With fixed jump positions the declared-order positions carry over; when the
    jumps reorder per cell the sorted-order jump indices are recovered from the
    permutation that sorts the preimages.
    """
    if dynamic_jumps:
        order = jnp.argsort(preimages)
        sorted_jumps = jnp.nonzero(jump_flags[order], size=n_jumps)[0]
        return preimages[order], tuple(sorted_jumps[k] for k in range(n_jumps))
    return jnp.sort(a=preimages), static_jump_positions


def _indexed_threshold_value(
    *,
    table: EconFunctionArg,
    subkey: str | None,
    index_state: StateName | None,
    static_index: int | None,
    cell: Mapping[StateName, EconFunctionArg],
) -> EconFunctionArg:
    """Read a breakpoint threshold from its param for one ride-along cell.

    The param is resolved to a value in this order:
    - `subkey` selects an entry inside a `MappingLeaf` (`leaf.data[subkey]`).
    - `index_state` reads the row at that ride-along state's code in this cell.
    - `static_index` selects a column (e.g. a bracket edge).

    A scalar threshold leaves every step disabled and passes through unchanged.

    Raises:
        TypeError: If `subkey` is set on a param that is not a `MappingLeaf`, or
            an index is set on a value that is not an array.

    """
    value = table
    if subkey is not None:
        if not isinstance(value, MappingLeaf):
            msg = (
                f"A threshold entry {subkey!r} is read from a grouped param; got "
                f"{type(value).__name__}."
            )
            raise TypeError(msg)
        value = value.data[subkey]
    if index_state is None and static_index is None:
        return value
    if not isinstance(value, jax.Array):
        msg = f"An indexed threshold is read from an array; got {type(value).__name__}."
        raise TypeError(msg)
    if index_state is not None:
        value = value[cell[index_state]]
    if static_index is not None:
        value = value[static_index]
    return value


@dataclass(frozen=True)
class _NBEGMRideAlongStatics:
    """Build-time config of the ride-along tile-local core.

    The core rebuilds each ride-along cell's breakpoint partition, budget schedule,
    discount factor, and utility off this config and reads the regime transition
    through `bind_continuation`. Every field is a Python-level static derived once
    from the schedule spec and continuation plan.
    """

    sources: tuple[_NBEGMSource, ...]
    """Every declared breakpoint, merged on the liquid axis."""
    jump_flags_arr: BoolND
    """Per-source jump indicator in declared order."""
    n_jumps: int
    """Number of jump breakpoints across all sources."""
    publish_jump_topology: bool
    """Whether the carry publishes jump preimages as duplicated row abscissae.

    `False` (`NBEGM.jump_read == "bridged"`) keeps the within-period case solve
    jump-aware but carries plain liquid-grid rows with no breakpoints, so
    parents interpolate across the cliffs and the stochastic-dim fold stays
    available.
    """
    has_jump: bool
    """Whether any declared breakpoint is a jump (vs. a continuous kink)."""
    static_jump_positions: tuple[int, ...]
    """Jump indices in the sorted partition when a single variable fixes the order."""
    dynamic_jumps: bool
    """Whether the sorted-order jump indices must be recovered per cell."""
    liquid_name: str
    """Name of the liquid (Euler) state."""
    ride_names: tuple[str, ...]
    """Ride-along state axes (the budget varies per cell over these)."""
    state_names: tuple[StateName, ...]
    """Liquid plus ride-along state names — the kwargs that are state grids."""
    continuation_reads_liquid: bool
    """Whether the continuation reads the current liquid state — through a carry
    target's next-state law or the regime-transition probabilities — so the
    continuation is piecewise-constant across declared intervals and the per-interval
    path applies."""
    interval_width: int
    """Batch size for the per-interval continuation read: `0` evaluates all
    intervals in one vectorized pass, a positive size runs sequential chunks of
    that many intervals."""
    branch_width: int
    """Block size for the discrete-action branch axis: `0` runs the
    whole axis in one vectorized pass, a positive size scans it in blocks of that
    many branches."""
    stochastic_node_width: int | None
    """Block width of the streamed child stochastic-node expectation; `None`
    folds the whole node mesh in one block."""
    consumption_action_name: ActionName
    """Name of the continuous consumption action the period utility reads."""
    utility_param_names: tuple[QualifiedName, ...]
    """Qualified utility params (excluding the consumption action and states)."""
    utility_state_names: tuple[StateName, ...]
    """Ride-along states the period utility reads, bound per cell."""
    coh_state_names: tuple[StateName, ...]
    """Ride-along states the cash-on-hand schedule reads, bound per cell."""
    discount_param_names: tuple[QualifiedName, ...]
    """Qualified params the discount-factor DAG reads, or empty for flat discount."""
    discount_state_names: tuple[StateName, ...]
    """Ride-along states the discount-factor DAG reads, or empty for flat discount."""
    discount_action_names: tuple[ActionName, ...]
    """Discrete actions the discount-factor DAG reads, bound per branch."""
    n_intervals: int
    """Number of liquid intervals the breakpoints split each cell into (N + 1)."""
    n_savings: int
    """Length of the post-decision savings grid."""
    envelope_arithmetic: ComparisonArithmetic
    """Which arithmetic decides envelope ownership (see
    `NBEGM.envelope_arithmetic`)."""
    cell_width: int
    """Batch width of the ride-cell blocks; `0` vmaps the whole cell mesh."""
    n_action_branches: int
    """Number of discrete-action branches the continuation carries a leading axis
    over; `0` when the regime carries no discrete action (no branch axis). A branch
    reads its own next-state continuation, so a co-state-feeding action gets a
    distinct row per branch and a budget-only action gets identical rows."""
    discrete_action_names: tuple[ActionName, ...]
    """Declared discrete actions in the branch product's column order."""
    discrete_action_codes: tuple[tuple[int, ...], ...]
    """Exact branch-code rows in the inner envelope's product order."""
    continuation_action_names: tuple[ActionName, ...] = ()
    """Declared discrete actions the continuation read consumes, in declared
    order; the branch classes key on their codes alone."""
    continuation_class_of_branch: tuple[int, ...] = ()
    """Continuation class index per envelope branch, empty without discrete
    actions."""
    continuation_representatives: tuple[int, ...] = ()
    """Per continuation class, the first branch carrying it: the branch whose
    rows the class reads. Empty without discrete actions."""
    co_map_state_names: tuple[StateName, ...] = ()
    """Fixed, distributed ride-along states co-mapped with the child carry.

    A leading prefix of `ride_names`: each is distributed (sharded one block per
    device) and never transitions, so a ride cell's continuation depends only on
    its own slice of the next-period carry. The tile-local core `vmap`s over
    these axes, co-slicing the carry, so each device interpolates only its slice
    and XLA inserts no all-gather. Empty when no ride state qualifies."""

    collect_continuation_read_counts: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )

    @property
    def n_published_jumps(self) -> int:
        """Number of jump preimages the carry publishes per row."""
        return self.n_jumps if self.publish_jump_topology else 0

    @property
    def n_continuation_classes(self) -> int:
        """Number of continuation reads per ride cell across the branches."""
        return len(self.continuation_representatives)


def _nbegm_ride_along_statics(
    *,
    savings_grid: Float1D,
    schedule_spec: _NBEGMScheduleSpec,
    continuation_plan: ContinuationPlan,
    envelope_arithmetic: ComparisonArithmetic = "ordinary",
    cell_width: int = 0,
    interval_width: int = 0,
    branch_width: int = 0,
    stochastic_node_width: int | None = None,
    publish_jump_topology: bool = True,
    co_map_state_names: tuple[StateName, ...] = (),
    materializer: DatedGenerationMaterializer | None = None,
) -> _NBEGMRideAlongStatics:
    """Derive the static config of the ride-along tile-local core.

    Partitions the schedule's breakpoints, classifies the jump structure, and reads
    each component DAG's argument names (utility, cash-on-hand, discount factor) into
    the per-cell parameter and state splits the core applies.
    """
    import inspect  # noqa: PLC0415

    sources = schedule_spec.sources
    kinds = tuple(source.kind for source in sources)
    if "hard_constraint" in kinds:
        msg = (
            "NBEGM ride-along path supports continuous-kink and jump schedules; "
            f"got breakpoint kinds {kinds}. A hard-constraint (floor) breakpoint "
            "with a ride-along co-state is a later slice."
        )
        raise RegimeInitializationError(msg)
    jump_flags_arr, n_jumps, has_jump, static_jump_positions, dynamic_jumps = (
        _ride_along_jump_config(kinds=kinds, materializer=materializer)
    )

    liquid_name = schedule_spec.liquid_state_name
    ride_names = schedule_spec.ride_along_state_names
    state_names = (liquid_name, *ride_names)

    # The co-mapped ride states must be a leading prefix of the ride axes: the
    # tile-local core's outer `vmap` peels them off the front of both the ride
    # mesh and each carry leaf, so they have to be the carry's leading axes in order.
    co_map_ride_names = tuple(name for name in ride_names if name in co_map_state_names)
    if co_map_ride_names != ride_names[: len(co_map_ride_names)]:
        msg = (
            "Co-mapped ride states must be the leading ride axes, in order. Got "
            f"co_map_state_names={co_map_ride_names} but the leading ride_names are "
            f"{ride_names[: len(co_map_ride_names)]}."
        )
        raise RegimeInitializationError(msg)

    # The continuation is constant only within each declared interval when the
    # current liquid (Euler) state enters it through either channel:
    # - a carry target's next-state law reads liquid — a current-asset boundary in
    #   `next_<liquid>` (e.g. a Medicaid transfer or pension adjustment that switches
    #   at a declared cliff)
    # - the regime-transition probabilities read liquid — the target blend then
    #   differs across intervals (e.g. survival switched at an asset test)
    # Detect it once: the per-interval path then binds the liquid state to each
    # interval's node and solves interval by interval.
    transition_probs_read_liquid = (
        liquid_name
        in inspect.signature(
            continuation_plan.compute_regime_transition_probs
        ).parameters
    )
    continuation_reads_liquid = transition_probs_read_liquid or any(
        liquid_name in inspect.signature(func).parameters
        for target in continuation_plan.stateful_targets
        for func in (
            continuation_plan.child_reads[target].next_state_func,
            continuation_plan.child_reads[target].euler_state_func,
        )
    )

    # The period utility reads the consumption action, the ride-along states it
    # depends on (bound per cell), and qualified utility params (bound from kwargs).
    consumption_action_name = schedule_spec.consumption_action_name
    utility_arg_names = tuple(inspect.signature(schedule_spec.utility_dag).parameters)
    utility_param_names = tuple(
        name
        for name in utility_arg_names
        if name not in state_names
        and name != consumption_action_name
        and name not in {name for name, _ in schedule_spec.discrete_actions}
    )
    utility_state_names = tuple(
        name for name in ride_names if name in utility_arg_names
    )
    # The cash-on-hand schedule reads the liquid state plus whichever ride-along states
    # and params enter its DAG; bind exactly those per cell so unread ride-along states
    # (e.g. a preference type the budget ignores) are not forwarded to the DAG.
    coh_arg_names = tuple(inspect.signature(schedule_spec.coh_of_liquid_dag).parameters)
    coh_state_names = tuple(name for name in ride_names if name in coh_arg_names)
    discrete_action_names = tuple(
        name for name, _codes in schedule_spec.discrete_actions
    )
    # The discount factor is either pylcm's flat
    # `koopmans_aggregator__discount_factor` param or, when the regime supplies a
    # `discount_factor` DAG function (e.g. a per-preference-type beta read off a
    # ride-along state, or a branch-specific patience read off a discrete
    # action), resolved per branch from that function's qualified params,
    # ride-along state arguments, and the branch's action codes.
    discount_factor_dag = schedule_spec.discount_factor_dag
    if discount_factor_dag is None:
        discount_param_names: tuple[QualifiedName, ...] = ()
        discount_state_names: tuple[StateName, ...] = ()
        discount_action_names: tuple[ActionName, ...] = ()
    else:
        discount_arg_names = tuple(inspect.signature(discount_factor_dag).parameters)
        discount_param_names = tuple(
            name
            for name in discount_arg_names
            if name not in state_names and name not in discrete_action_names
        )
        discount_state_names = tuple(
            name for name in ride_names if name in discount_arg_names
        )
        discount_action_names = tuple(
            name for name in discrete_action_names if name in discount_arg_names
        )
    continuation_action_names = _continuation_action_names(
        regime_transition=continuation_plan.compute_regime_transition_probs,
        target_laws={
            target: (
                continuation_plan.child_reads[target].next_state_func,
                continuation_plan.child_reads[target].euler_state_func,
            )
            for target in continuation_plan.stateful_targets
        },
        target_weight_laws={
            target: continuation_plan.child_reads[target].weights_func
            for target in continuation_plan.stateful_targets
        },
        target_resources_arg_names={
            target: frozenset(continuation_plan.child_reads[target].resources_arg_names)
            | (
                _required_nbegm_child_query(
                    plan=continuation_plan, target=target
                ).input_names
                if continuation_plan.child_reads[target].query is not None
                else frozenset()
            )
            for target in continuation_plan.stateful_targets
        },
        discount_factor_dag=schedule_spec.discount_factor_dag,
        interval_schedule_dags=(
            tuple(source.derived_of_liquid_dag for source in sources)
            if continuation_reads_liquid
            else ()
        ),
        action_names=discrete_action_names,
    )
    continuation_class_of_branch: tuple[int, ...] = ()
    continuation_representatives: tuple[int, ...] = ()
    if schedule_spec.discrete_actions:
        continuation_class_of_branch, continuation_representatives = (
            _continuation_branch_classes(
                branch_bindings=schedule_spec.branch_bindings,
                continuation_action_names=continuation_action_names,
            )
        )
    return _NBEGMRideAlongStatics(
        sources=sources,
        jump_flags_arr=jump_flags_arr,
        n_jumps=n_jumps,
        publish_jump_topology=publish_jump_topology,
        has_jump=has_jump,
        static_jump_positions=static_jump_positions,
        dynamic_jumps=dynamic_jumps,
        liquid_name=liquid_name,
        ride_names=ride_names,
        state_names=state_names,
        continuation_reads_liquid=continuation_reads_liquid,
        consumption_action_name=consumption_action_name,
        utility_param_names=utility_param_names,
        utility_state_names=utility_state_names,
        coh_state_names=coh_state_names,
        discount_param_names=discount_param_names,
        discount_action_names=discount_action_names,
        discount_state_names=discount_state_names,
        n_intervals=len(sources) + 1,
        n_savings=int(savings_grid.shape[0]),
        envelope_arithmetic=envelope_arithmetic,
        cell_width=cell_width,
        interval_width=interval_width,
        branch_width=branch_width,
        stochastic_node_width=stochastic_node_width,
        n_action_branches=(
            0
            if not schedule_spec.discrete_actions
            else len(schedule_spec.branch_bindings)
        ),
        discrete_action_names=discrete_action_names,
        discrete_action_codes=tuple(
            tuple(binding[name] for name in discrete_action_names)
            for binding in schedule_spec.branch_bindings
        ),
        continuation_action_names=continuation_action_names,
        continuation_class_of_branch=continuation_class_of_branch,
        continuation_representatives=continuation_representatives,
        co_map_state_names=co_map_ride_names,
    )


def _nbegm_cell_breakpoints(
    *,
    statics: _NBEGMRideAlongStatics,
    kwargs: EconFunctionKwargs,
    cell: Mapping[StateName, EconFunctionArg],
    liquid_grid: Float1D,
    dtype: DTypeLike,
    action_binding: Mapping[ActionName, EconFunctionArg] = MappingProxyType({}),
) -> tuple[Float1D, tuple[JumpPosition, ...]]:
    """Build one ride-along cell's sorted liquid breakpoints and jump positions.

    Each declared schedule's threshold maps to its asset value in its own variable
    (directly for a liquid-state schedule, via the per-cell affine preimage for a
    derived-variable schedule), and the sources merge into one sorted partition. A
    degenerate boundary — a derived variable with (near-)zero asset slope in this cell,
    so the threshold is never crossed — has a non-finite preimage; clamping to a margin
    just outside the grid collapses it to an empty edge interval instead of poisoning a
    live interval's affine segment.
    """
    from _lcm.egm.nbegm_breakpoints import (  # noqa: PLC0415
        clamp_breakpoints_to_grid,
    )

    # Zero declared breakpoints ⇒ an empty partition (one interval per cell).
    preimages = (
        clamp_breakpoints_to_grid(
            breakpoints=jnp.stack(
                [
                    _cell_breakpoint(
                        source=source,
                        liquid_name=statics.liquid_name,
                        kwargs=kwargs,
                        cell=cell,
                        dtype=dtype,
                        action_binding=action_binding,
                    )
                    for source in statics.sources
                ]
            ),
            liquid_grid=liquid_grid,
        )
        if statics.sources
        else jnp.zeros((0,), dtype=dtype)
    )
    return _partition_jumps(
        preimages=preimages,
        dynamic_jumps=statics.dynamic_jumps,
        jump_flags=statics.jump_flags_arr,
        n_jumps=statics.n_jumps,
        static_jump_positions=statics.static_jump_positions,
    )


def _cell_breakpoint(
    *,
    source: _NBEGMSource,
    liquid_name: str,
    kwargs: EconFunctionKwargs,
    cell: Mapping[StateName, EconFunctionArg],
    dtype: DTypeLike,
    action_binding: Mapping[ActionName, EconFunctionArg],
) -> FloatND:
    """One source's threshold as the liquid value where the cell crosses it.

    A liquid-state schedule's threshold is its own asset value, with the exact
    boundary point given to the side that owns equality; a derived-variable
    schedule's threshold is mapped through the per-cell affine preimage.
    """
    from _lcm.egm.nbegm_breakpoints import (  # noqa: PLC0415
        linear_asset_selection_preimage,
    )

    threshold_value = _indexed_threshold_value(
        table=kwargs[source.threshold_param_name],
        subkey=source.threshold_subkey,
        index_state=source.threshold_index_state,
        static_index=source.threshold_static_index,
        cell=cell,
    )
    threshold = jnp.asarray(threshold_value, dtype=dtype)
    if source.derived_of_liquid_dag is None:
        return jnp.where(
            source.equality_owner == "above",
            threshold,
            jnp.nextafter(threshold, jnp.inf),
        )
    dag = source.derived_of_liquid_dag
    dag_arg_names = frozenset(inspect.signature(dag).parameters)
    return linear_asset_selection_preimage(
        z_of_liquid=_BoundScalarFunction(
            func=dag,
            argument_name=liquid_name,
            bound={
                **{name: cell[name] for name in source.derived_state_names},
                **{name: kwargs[name] for name in source.derived_param_names},
                **{
                    name: value
                    for name, value in action_binding.items()
                    if name in dag_arg_names
                },
            },
        ),
        threshold=threshold,
        equality_owner=source.equality_owner,
    )


# How many units of the liquid law's rounding error to step past a cliff, and the
# largest share of the distance to a neighbouring cliff's preimage that step may
# consume.
_CLIFF_MARGIN_ROUNDINGS = 4.0
_CLIFF_MARGIN_GAP_SHARE = 0.25


def cliff_target_margin(
    *,
    s_star: FloatND,
    slope: FloatND,
    intercept: FloatND,
    dtype: DTypeLike,
) -> FloatND:
    """Return the savings displacement that lands just past each cliff preimage.

    The target reaches the child's liquid axis through the affine savings law
    `next_liquid = slope * s + intercept`, whose evaluation rounds by about
    `eps * (|slope * s| + |intercept|)` in liquid units. Stepping a fixed number
    of those roundings, converted back to savings units by dividing by `|slope|`,
    clears the cliff on the intended side at every scale and in every precision —
    a margin scaled by `|s_star|` alone is both wrong in scale (it ignores the
    intercept) and precision-dependent in relative size.

    The step is also capped at a share of the distance to the nearest other
    preimage, so a nudge can never overshoot a neighbouring cliff.

    Args:
        s_star: Savings preimage of each jump, shape `(n_jumps,)`.
        slope: Slope of the savings-form liquid law.
        intercept: Intercept of the savings-form liquid law.
        dtype: Floating dtype the solve runs in.

    Returns:
        Per-jump displacement, same shape as `s_star`.

    """
    rounding = jnp.finfo(dtype).eps * (jnp.abs(slope * s_star) + jnp.abs(intercept))
    margin = _CLIFF_MARGIN_ROUNDINGS * rounding / jnp.abs(slope)
    n_jumps = s_star.shape[0]
    separation = jnp.abs(s_star[:, None] - s_star[None, :])
    separation = jnp.where(jnp.eye(n_jumps, dtype=bool), jnp.inf, separation)
    return jnp.minimum(margin, _CLIFF_MARGIN_GAP_SHARE * jnp.min(separation, axis=-1))


def _cliff_savings_targets(
    *,
    continuation_plan: ContinuationPlan,
    regime_name: RegimeName,
    statics: _NBEGMRideAlongStatics,
    child_carry: EGMCarry,
    combo_pool: EconFunctionKwargs,
    savings_grid: Float1D,
    dtype: DTypeLike,
    midpoints: Float1D | None = None,
) -> FloatND:
    """Map the self-read child's value cliffs to one-sided savings targets.

    A child value jump creates a legitimate one-sided optimum — save to just
    inside the cliff's owning side — that generically falls strictly between
    savings nodes. The cliffs are the jump preimages the child carry publishes
    per row, so they are the child period's own: evaluated at the child's age
    and period, with its functions and its state values. Each target inverts the
    affine savings-form liquid law against one row's jumps and sits a few float
    margins inside each side of every jump (`2 * n_jumps` entries per row).
    Targets outside the savings grid's span, or under a non-increasing liquid
    law, are NaN — the envelope's point-candidate family treats NaN entries as
    dead.

    The child row is the one the continuation reads, axis by axis:
    - a deterministic discrete state selects the row of its next-state code;
    - a stochastic state whose node the jumps or the liquid law read contributes
      one row per node; one whose node neither reads selects its first node,
      whose jumps every node shares;
    - a passive continuous state the jumps read contributes both rows of the
      grid segment its next value is blended from, since the blended child value
      jumps at each row's cliffs; one the jumps do not read selects the lower;
    - a co-mapped state is already sliced from the carry.

    A local draw the liquid law reads contributes one target set per node.
    """
    read = continuation_plan.child_reads[regime_name]
    if child_carry.breakpoints is None:
        return jnp.zeros((0,), dtype=dtype)
    targets_for_pool = functools.partial(
        _cliff_targets_for_pool,
        read=read,
        breakpoints=child_carry.breakpoints,
        jump_states=_jump_breakpoint_states(statics=statics),
        co_map_state_names=frozenset(statics.co_map_state_names),
        post_decision_name=continuation_plan.post_decision_name,
        savings_grid=savings_grid,
        dtype=dtype,
    )

    if midpoints is None:
        return targets_for_pool(pool=combo_pool)
    # An interval-bound liquid law: the savings-to-liquid map (and so each
    # cliff's savings preimage) is specific to the interval whose node the
    # liquid state is bound to — one target row per interval.
    return jax.vmap(
        functools.partial(
            _cliff_targets_at_midpoint,
            targets_for_pool=targets_for_pool,
            combo_pool=combo_pool,
            liquid_name=statics.liquid_name,
        )
    )(midpoints)


# keyword-only-exempt: library-callback=jax.vmap
def _cliff_targets_at_midpoint(
    midpoint: FloatND,
    *,
    targets_for_pool: Callable[..., FloatND],
    combo_pool: EconFunctionKwargs,
    liquid_name: str,
) -> FloatND:
    """The cliff targets with the liquid state bound to one interval's node."""
    return targets_for_pool(pool={**combo_pool, liquid_name: midpoint})


def _jump_breakpoint_states(*, statics: _NBEGMRideAlongStatics) -> frozenset[str]:
    """The ride-along states a jump threshold is indexed by or its variable reads."""
    return frozenset(
        name
        for source in statics.sources
        if source.kind == "jump"
        for name in (source.threshold_index_state, *source.derived_state_names)
        if name is not None
    )


def _cliff_targets_for_pool(
    *,
    pool: EconFunctionKwargs,
    read: _ChildRead,
    breakpoints: FloatND,
    jump_states: frozenset[str],
    co_map_state_names: frozenset[StateName],
    post_decision_name: str,
    savings_grid: Float1D,
    dtype: DTypeLike,
) -> FloatND:
    """One-sided savings targets of every child row's jumps under one combo pool.

    Every next state but the liquid one is savings-independent, so the child's
    row indices are read off the draw-free laws at savings zero. Each node axis
    (see `_cliff_savings_targets`) is enumerated jointly; at each node the liquid
    law, with the draws it reads bound, is inverted against the jumps of the row
    the node reaches, and the targets of all nodes are concatenated.

    Raises:
        ValueError: If the child carry's breakpoint rows are not indexed by the
            child's state axes followed by one jump axis.

    """
    from _lcm.egm.continuation import euler_draw_nodes  # noqa: PLC0415
    from _lcm.egm.interp import locate_on_grid  # noqa: PLC0415

    next_states = read.next_state_func(
        **pool, **{post_decision_name: jnp.asarray(0.0, dtype=dtype)}
    )
    law_draws = euler_draw_nodes(read=read, combo_pool=pool)
    # Per carry axis: a fixed row index, or the name of the node axis supplying it.
    row: list[IntND | str] = []
    node_axes: dict[str, IntND] = {}
    stochastic_positions = dict(
        zip(
            read.stochastic_state_names,
            range(len(read.stochastic_state_names)),
            strict=True,
        )
    )
    for name, is_stochastic in zip(
        read.discrete_state_names, read.stochastic_flags, strict=True
    ):
        if name in co_map_state_names:
            continue
        if not is_stochastic:
            row.append(jnp.asarray(next_states[f"next_{name}"], dtype=jnp.int32))
            continue
        draw = f"next_{name}"
        if name in jump_states or draw in law_draws:
            n_nodes = read.stochastic_node_values[stochastic_positions[name]].shape[0]
            node_axes[draw] = jnp.arange(n_nodes, dtype=jnp.int32)
            row.append(draw)
        else:
            row.append(jnp.int32(0))
    for name, grid in zip(read.passive_state_names, read.passive_grids, strict=True):
        key = f"next_{name}"
        value = next_states[key] if key in next_states else cast("FloatND", pool[key])
        lower, upper, _ = locate_on_grid(x_query=value, grid=grid)
        if name in jump_states:
            node_axes[f"segment_{name}"] = jnp.stack([lower, upper])
            row.append(f"segment_{name}")
        else:
            row.append(lower)
    local_draws = {
        name: values for name, values in law_draws.items() if name not in node_axes
    }
    if breakpoints.ndim != len(row) + 1:
        msg = (
            f"The child carry publishes breakpoints of shape {breakpoints.shape}, "
            f"but its rows are indexed by {len(row)} state axes."
        )
        raise ValueError(msg)

    def targets_at(node: Mapping[str, FloatND | IntND]) -> FloatND:
        draws = {
            name: law_draws[name][node[name]] if name in node_axes else node[name]
            for name in law_draws
        }
        index = tuple(node[axis] if isinstance(axis, str) else axis for axis in row)
        return _cliff_targets_at_node(
            pool={**pool, **draws},
            next_state_func=read.euler_state_func,
            next_state_key=read.next_state_key,
            post_decision_name=post_decision_name,
            jumps=breakpoints[index],
            savings_grid=savings_grid,
            dtype=dtype,
        )

    axes = {**node_axes, **local_draws}
    if not axes:
        return targets_at({})
    mesh = jnp.meshgrid(*axes.values(), indexing="ij")
    flat = {name: grid.ravel() for name, grid in zip(axes, mesh, strict=True)}
    return jax.vmap(targets_at)(flat).reshape(-1)


def _cliff_targets_at_node(
    *,
    pool: EconFunctionKwargs,
    next_state_func: Callable[..., Mapping[TransitionFunctionName, FloatND | IntND]],
    next_state_key: str,
    post_decision_name: str,
    jumps: Float1D,
    savings_grid: Float1D,
    dtype: DTypeLike,
) -> FloatND:
    """One-sided savings targets of every jump with every input in `pool` fixed."""
    intercept = _next_euler_state(
        savings_value=jnp.asarray(0.0, dtype=dtype),
        pool=pool,
        next_state_func=next_state_func,
        next_state_key=next_state_key,
        post_decision_name=post_decision_name,
        dtype=dtype,
    )
    slope = (
        _next_euler_state(
            savings_value=jnp.asarray(1.0, dtype=dtype),
            pool=pool,
            next_state_func=next_state_func,
            next_state_key=next_state_key,
            post_decision_name=post_decision_name,
            dtype=dtype,
        )
        - intercept
    )
    s_star = (jumps - intercept) / slope
    margin = cliff_target_margin(
        s_star=s_star, slope=slope, intercept=intercept, dtype=dtype
    )
    candidates = jnp.stack([s_star - margin, s_star + margin], axis=-1).reshape(-1)
    valid = (
        (candidates >= savings_grid[0])
        & (candidates <= savings_grid[-1])
        & (slope > 0.0)
    )
    return jnp.where(valid, candidates, jnp.nan)


def _next_euler_state(
    *,
    savings_value: FloatND,
    pool: EconFunctionKwargs,
    next_state_func: Callable[..., Mapping[TransitionFunctionName, FloatND | IntND]],
    next_state_key: str,
    post_decision_name: str,
    dtype: DTypeLike,
) -> FloatND:
    """The child's liquid (Euler) state reached from `savings_value` under `pool`."""
    next_states = next_state_func(**pool, **{post_decision_name: savings_value})
    return jnp.asarray(next_states[next_state_key], dtype=dtype)


def _carry_comap_in_axes(
    *,
    carry: MappingProxyType[RegimeName, EGMCarry],
    slice_targets: frozenset[RegimeName],
) -> MappingProxyType[RegimeName, EGMCarry]:
    """Build the `vmap` `in_axes` pytree slicing each sliced target's leading axis.

    A target in `slice_targets` carries the co-mapped state as its leading axis,
    so its array leaves map over axis `0` and its scalar taste-shock leaf passes
    through (`None`). Every other target — a scalar target, an unread carry, or a
    carry target that does not carry this co-mapped state (e.g. a terminal target
    whose value is kind-independent) — passes through whole.
    """
    result: dict[RegimeName, EGMCarry] = {}
    for target, target_carry in carry.items():
        axis = 0 if target in slice_targets else None
        result[target] = jax.tree_util.tree_map(
            functools.partial(_leading_axis_or_none, axis=axis), target_carry
        )
    return MappingProxyType(result)


# keyword-only-exempt: library-callback=jax.tree_util.tree_map
def _leading_axis_or_none(leaf: FloatND, *, axis: int | None) -> int | None:
    """`axis` for an array leaf, `None` for a scalar leaf that passes through."""
    return axis if jnp.ndim(leaf) > 0 else None


def _ride_along_core_programs(
    *,
    savings_grid: Float1D,
    continuation_plan: ContinuationPlan,
    statics: _NBEGMRideAlongStatics,
    regime_name: RegimeName,
    cliff_candidates: bool,
    schedule_spec: _NBEGMScheduleSpec,
    envelope_build: _NBEGMEnvelopeBuild,
    state_action_space: StateActionSpace,
    argument_builder: _RideAlongArgumentBuilder,
    enable_jit: bool,
    query_template: EGMCarry | None = None,
    materializer: DatedGenerationMaterializer | None = None,
) -> MappingProxyType[str, CoreProgram]:
    """Declare the ride-along kernel's `main` and `replay` programs.

    Both declare the independent inner ride-cell mesh. Co-mapped state axes
    retain their shared carry placement outside this mesh. The continuation is
    read from the carries, so no target value access is declared. `main` is
    dispatched by a values-only solve and `replay` by a solve retaining replay
    artifacts.
    """
    inner_names = tuple(
        name for name in statics.ride_names if name not in statics.co_map_state_names
    )
    n_cells = math.prod(
        int(
            (
                jnp.asarray(state_action_space.states[name])
                if materializer is None
                else state_action_space.states[name]
            ).shape[0]
        )
        for name in inner_names
    )
    interval_axes = (
        (
            ReducedAxis(
                name=INTERVAL_AXIS,
                coordinate_names=(_INTERVAL_COORDINATE,),
                coordinate_extents=(statics.n_intervals,),
                canonical_order="c",
                reduction=INTERVAL_ENVELOPE_REDUCTION,
                width_keyword="__lcm_interval_width__",
            ),
        )
        if statics.continuation_reads_liquid and statics.n_intervals > 1
        else ()
    )
    branch_axes = (
        (
            ReducedAxis(
                name=BRANCH_AXIS,
                coordinate_names=statics.discrete_action_names,
                coordinate_extents=tuple(
                    len(state_action_space.discrete_actions[name])
                    for name in statics.discrete_action_names
                ),
                canonical_order="c",
                reduction=HARD_MAX_REDUCTION,
                width_keyword="__lcm_branch_width__",
            ),
        )
        if statics.n_action_branches > 1
        else ()
    )
    requirements = CoreExecutionRequirements(
        reduced_axes=(
            _ride_along_stochastic_axes(
                continuation_plan=continuation_plan,
                state_action_space=state_action_space,
            )
            + interval_axes
            + branch_axes
        ),
        tiled_axes=(
            (
                TiledOutputAxis(
                    name=CELL_AXIS,
                    state_names=inner_names,
                    extent=n_cells,
                    width_keyword="__lcm_cell_width__",
                ),
            )
            if n_cells > 1
            else ()
        ),
    )
    programs: dict[str, CoreProgram] = {}
    for name, scope, publish_replay in (
        ("main", ProgramScope.VALUES_ONLY, False),
        ("replay", ProgramScope.REPLAY, True),
    ):
        core = _build_nbegm_tiled_core(
            savings_grid=savings_grid,
            continuation_plan=continuation_plan,
            statics=statics,
            regime_name=regime_name,
            cliff_candidates=cliff_candidates,
            schedule_spec=schedule_spec,
            envelope_build=envelope_build,
            publish_replay=publish_replay,
        )
        checked = None
        if (
            envelope_build.bind_cell.current_floor is not None
            or continuation_plan.query_targets
        ):
            checked_statics = replace(statics, collect_continuation_read_counts=True)
            checked_core = _build_nbegm_tiled_core(
                savings_grid=savings_grid,
                continuation_plan=continuation_plan,
                statics=checked_statics,
                regime_name=regime_name,
                cliff_candidates=cliff_candidates,
                schedule_spec=schedule_spec,
                envelope_build=envelope_build,
                publish_replay=publish_replay,
            )
            checked_roles = _ride_along_output_roles(
                statics=checked_statics,
                schedule_spec=schedule_spec,
                n_published_boundaries=envelope_build.n_published_boundaries,
                publish_replay=publish_replay,
                query_template=query_template,
                query_model=envelope_build.bind_cell.query_model,
                own_bound_count=int(envelope_build.bind_cell.current_floor is not None),
                query_status_count=len(continuation_plan.query_targets),
                retain_nested_feasibility=envelope_build.bind_cell.query_model
                is not None
                and envelope_build.bind_cell.query_model.retain_nested_feasibility,
            )
            if type(checked_roles) is not BoundedResult:
                raise TypeError(
                    "Counted NB specialization requires its native bounded roles."
                )
            checked_roles = replace(
                checked_roles,
                continuation_read_counts=_unflatten_continuation_read_counts(
                    _aux=None,
                    children=tuple(
                        StateAxesLeading(state_names=(), shape=(), dtype=jnp.int32)
                        for _ in range(2)
                    ),
                ),
            )
            checked = _CheckedCoreVariant(
                function=jax.jit(checked_core, static_argnames=_WIDTH_KEYWORDS)
                if enable_jit
                else checked_core,
                output_roles=checked_roles,
            )
        programs[name] = CoreProgram(
            name=name,
            function=(
                jax.jit(core, static_argnames=_WIDTH_KEYWORDS) if enable_jit else core
            ),
            argument_builder=argument_builder,
            requirements=requirements,
            output_roles=_ride_along_output_roles(
                statics=statics,
                schedule_spec=schedule_spec,
                n_published_boundaries=envelope_build.n_published_boundaries,
                publish_replay=publish_replay,
                query_template=query_template,
                query_model=envelope_build.bind_cell.query_model,
                own_bound_count=int(envelope_build.bind_cell.current_floor is not None),
                query_status_count=len(continuation_plan.query_targets),
                retain_nested_feasibility=envelope_build.bind_cell.query_model
                is not None
                and envelope_build.bind_cell.query_model.retain_nested_feasibility,
            ),
            disposition=CoreExecutionDisposition.PLANNED,
            donation_candidates=(),
            checked_variant=checked,
            scope=scope,
            retained_artifact_keys=(SIMULATION_POLICY,) if publish_replay else (),
            retained_artifact_payload_types=(
                {SIMULATION_POLICY: NBEGMGridPolicy} if publish_replay else {}
            ),
            replaces_program="main" if publish_replay else None,
        )
    return MappingProxyType(programs)


def _with_ride_marginal_reads(
    *,
    kernel: PeriodKernel,
    context: SolverBuildContext,
    period: int,
    adapted: dict[tuple[int, int], tuple[CoreArgumentBuilder, MarginalLeafCore]],
) -> PeriodKernel:
    """Install the adapter and its complete reads together on the top-level route.

    Composite solvers call NBEGM's builder but own a different declaration hook.
    Their argument trees and read inventories therefore remain unchanged here.
    The call-local memo shares wrappers across grouped periods, retaining only
    existing callables and target metadata, never concrete inputs.
    """
    if not isinstance(kernel, _RideAlongNBEGMPeriodKernel):
        return kernel
    if (
        context.sharded_state_names
        or kernel.statics.co_map_state_names
        or kernel.stateful_targets != frozenset({kernel.regime_name})
    ):
        return kernel
    template = published_continuation_template(
        continuation_specs=context.continuation_specs, target=kernel.regime_name
    )
    if not isinstance(template, EGMCarry):
        raise TypeError(
            "A marginal-leaf program requires its published EGMCarry template."
        )
    programs: dict[str, CoreProgram] = {}
    for name, program in kernel.core_programs().items():
        key = (id(program.function), id(program.argument_builder))
        if key not in adapted:
            adapted[key] = (
                MarginalLeafArguments(
                    inner=program.argument_builder, target=kernel.regime_name
                ),
                MarginalLeafCore(core=program.function, target=kernel.regime_name),
            )
        builder, function = adapted[key]
        checked_variant = program.checked_variant
        if checked_variant is not None:
            checked_key = (id(checked_variant.function), id(program.argument_builder))
            if checked_key not in adapted:
                adapted[checked_key] = (
                    builder,
                    MarginalLeafCore(
                        core=checked_variant.function, target=kernel.regime_name
                    ),
                )
            checked_variant = replace(checked_variant, function=adapted[checked_key][1])
        programs[name] = replace(
            program,
            function=function,
            argument_builder=builder,
            checked_variant=checked_variant,
            donation_candidates=(MARGINAL_ARGUMENT,) if name == "main" else (),
        )
    return replace(
        kernel,
        _core_programs=with_continuation_leaf_reads(
            programs=programs,
            reads_by_core_key={
                name: marginal_leaf_reads(
                    continuation_leaf_reads(
                        template=template,
                        artifact_key=EGM_CONTINUATION,
                        target=kernel.regime_name,
                        source_regime=kernel.regime_name,
                        source_period=period,
                        core_key=name,
                    )
                )
                for name in kernel.core_programs()
            },
        ),
    )


def _ride_along_stochastic_axes(
    *,
    continuation_plan: ContinuationPlan,
    state_action_space: StateActionSpace,
) -> tuple[ReducedAxis, ...]:
    """Declare only a nontrivial child node mesh present in the program's inputs.

    As in DC-EGM, absent, singleton, and cross-grid meshes have no partitionable
    coordinate argument and keep their complete continuation fold.
    """
    from _lcm.egm.continuation import stochastic_node_axes  # noqa: PLC0415

    mesh = stochastic_node_axes(plan=continuation_plan)
    grids = state_action_space.states
    if math.prod(count for _, count in mesh) <= 1 or not all(
        name in grids and len(grids[name]) == count for name, count in mesh
    ):
        return ()
    return (
        ReducedAxis(
            name=STOCHASTIC_NODE_AXIS,
            coordinate_names=tuple(name for name, _ in mesh),
            coordinate_extents=tuple(count for _, count in mesh),
            canonical_order="c",
            reduction=WEIGHTED_EXPECTATION_REDUCTION,
            width_keyword="__lcm_stochastic_node_width__",
        ),
    )


def _ride_along_output_roles(
    *,
    statics: _NBEGMRideAlongStatics,
    schedule_spec: _NBEGMScheduleSpec,
    n_published_boundaries: int,
    publish_replay: bool,
    query_template: EGMCarry | None = None,
    query_model: NBEGMQueryModel | None = None,
    own_bound_count: int = 0,
    query_status_count: int = 0,
    retain_nested_feasibility: bool = False,
) -> OutputRoleTree:
    """Describe the tile-local core's outputs by the state axes that lead them.

    - The value array is the regime's value on the productmap state order.
    - Every carry row leads with the ride axes and ends with the row axis; the
      taste-shock scale is a replicated scalar; the breakpoints row exists only
      when the carry publishes jump topology and the exact policy row only on the
      continuous-only, jump-free path (`_assemble_ride_carry`).
    - The consumption policy sits on the value layout; each branch bank leads with
      the branch axis and then follows the value layout.
    """
    ride = StateAxesLeading(state_names=statics.ride_names)
    has_branches = statics.n_action_branches > 0
    carry_roles = egm_carry_role_tree(
        row=ride,
        scalar=StateAxesLeading(state_names=(), shape=()),
        breakpoints=ride if n_published_boundaries else None,
        policy=ride if not has_branches and n_published_boundaries == 0 else None,
    )
    if query_model is not None:
        if query_template is None or query_template.query_data is None:
            raise TypeError(
                "NB conditional roles require the independent numeric template."
            )
        row = StateAxesLeading(state_names=query_model.authority.state_names)
        carry_roles = egm_carry_role_tree(
            row=row,
            scalar=StateAxesLeading(state_names=(), shape=()),
            breakpoints=row if query_template.breakpoints is not None else None,
            policy=row,
            query_data=egm_query_role_tree(
                query_data=query_template.query_data,
                state_names=query_model.authority.state_names,
            ),
        )
    if retain_nested_feasibility:
        state_names = list(schedule_spec.ride_along_state_names)
        state_names.insert(
            schedule_spec.liquid_axis_pos, schedule_spec.liquid_state_name
        )
        mask = StateAxesLeading(state_names=tuple(state_names), dtype=jnp.bool_)
        ordinary = (VALUE, carry_roles)
        if publish_replay:
            policy = StateAxesLeading(state_names=tuple(state_names))
            bank = StateAxesLeading(
                state_names=tuple(state_names), n_free_leading_axes=1
            )
            ordinary = (
                (VALUE, carry_roles, policy, bank, bank)
                if has_branches
                else (VALUE, carry_roles, policy)
            )
        base = _nbegm_bounded_output_roles(
            ordinary=ordinary,
            own_bound_count=own_bound_count,
            query_status_count=query_status_count,
        )
        if not isinstance(base, BoundedResult):
            raise TypeError("Nested NB feasibility requires mandatory bound summaries.")
        return replace(base, output=(base.output[:-1], base.output[-1], mask))
    if not publish_replay:
        return _nbegm_bounded_output_roles(
            ordinary=(VALUE, carry_roles),
            own_bound_count=own_bound_count,
            query_status_count=query_status_count,
        )
    value_order = list(schedule_spec.ride_along_state_names)
    value_order.insert(schedule_spec.liquid_axis_pos, schedule_spec.liquid_state_name)
    policy = StateAxesLeading(state_names=tuple(value_order))
    if not has_branches:
        return _nbegm_bounded_output_roles(
            ordinary=(VALUE, carry_roles, policy),
            own_bound_count=own_bound_count,
            query_status_count=query_status_count,
        )
    bank = StateAxesLeading(state_names=tuple(value_order), n_free_leading_axes=1)
    return _nbegm_bounded_output_roles(
        ordinary=(VALUE, carry_roles, policy, bank, bank),
        own_bound_count=own_bound_count,
        query_status_count=query_status_count,
    )


@dataclass(frozen=True, kw_only=True)
class _NBEGMIntervalContinuation:
    """Bind a discrete branch to a fixed-width continuation block reader."""

    bind: Callable[..., Callable[[IntND], _NBEGMIntervalRows]]
    """Return ``read(interval_indices)`` for one cell and action binding."""
    query_targets: tuple[RegimeName, ...] = ()
    collect_continuation_read_counts: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Canonical ordered slots reduced by the consuming interval scan."""


@dataclass(frozen=True, kw_only=True)
class _NBEGMPreparedCell:
    """One source-cell floor shared by its continuation reads and branch solves."""

    continuation: _NBEGMCellRead
    """Already read rows, or an interval reader bound to the prepared nodes."""
    savings_grid: Float1D
    """The exact once-prepared physical nodes used by every branch."""
    bound_status: BoundInputStatus
    """Unmasked source integrity, reduced only by the owning cell map."""


class _NBEGMCellQueryPublication(NamedTuple):
    """A cell's ordinary outputs beside its complete conditional native carry."""

    ordinary: tuple[FloatND, ...]
    """Existing collapsed value/policy and optional replay outputs."""
    conditional: EGMCarry | None
    """Full liquid rows, native M, original A and compact interval W before max."""
    feasible: BoolND
    """Whether each original published liquid node has a jointly legal action."""
    conditional_feasible: BoolND | None = None
    """Actual pre-collapse D feasibility, retained only for an enclosing fold."""


@dataclass(frozen=True, eq=False, kw_only=True)
class _NBEGMIntervalBlockReader:
    """Read one branch's continuation rows for a requested block of intervals."""

    midpoints: Float1D
    """Each interval's representative liquid node."""

    cliff_targets: FloatND | None
    """Per-interval save-to-cliff savings targets, or `None` without them."""

    interval_rows: Callable[[tuple[FloatND, ...]], _NBEGMReadResult]
    """The per-interval row read, bound to the branch's combo pool."""

    def __call__(self, interval_indices: IntND) -> _NBEGMIntervalRows:
        """Read the rows of the intervals `interval_indices` name."""
        midpoints = self.midpoints
        selected_midpoints = midpoints[interval_indices]
        selected_targets = (
            None if self.cliff_targets is None else self.cliff_targets[interval_indices]
        )
        inputs = (
            (selected_midpoints,)
            if selected_targets is None
            else (selected_midpoints, selected_targets)
        )
        rows = jax.vmap(self.interval_rows)(inputs)
        if selected_targets is None:
            if isinstance(rows, BoundedResult):
                numeric: tuple[FloatND, ...] = rows.output
                return BoundedResult(
                    output=numeric,
                    bound_status=rows.bound_status,
                    query_status=rows.query_status,
                    continuation_read_counts=rows.continuation_read_counts,
                )
            return rows
        if isinstance(rows, BoundedResult):
            published_rows: tuple[FloatND, ...] = (*rows.output, selected_targets)
            return BoundedResult(
                output=published_rows,
                bound_status=rows.bound_status,
                query_status=rows.query_status,
                continuation_read_counts=rows.continuation_read_counts,
            )
        return (*rows, selected_targets)


def _build_nbegm_tiled_core(
    *,
    savings_grid: Float1D,
    continuation_plan: ContinuationPlan,
    statics: _NBEGMRideAlongStatics,
    regime_name: RegimeName,
    cliff_candidates: bool,
    schedule_spec: _NBEGMScheduleSpec,
    envelope_build: _NBEGMEnvelopeBuild,
    publish_replay: bool,
) -> _NBEGMTiledCore:
    """Build the tile-local core of the ride-along solve.

    One compiled body walks the ride cells in blocks of `statics.cell_width`.
    For each block it reads the continuation rows — the complete expectation over
    every reachable target and stochastic node, on the savings grid — and solves
    the block's envelope step against them before the next block starts, so the
    expected-continuation stacks over every cell are never a complete array, let
    alone an argument. The core takes what the continuation read takes (states,
    filtered carries, params) and returns the value array and the carry; with
    `publish_replay` it also returns the consumption policy and, for a discrete
    ride action, the conditional branch banks.

    Co-mapped ride states stay outermost: each co-map slice reads only its slice of
    the child carry, and the inner mesh solves inside that slice.
    """
    return _NBEGMTiledCore(
        savings_grid=savings_grid,
        continuation_plan=continuation_plan,
        statics=statics,
        regime_name=regime_name,
        cliff_candidates=cliff_candidates,
        schedule_spec=schedule_spec,
        envelope_build=envelope_build,
        publish_replay=publish_replay,
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class _NBEGMTiledCore:
    """The tile-local core of the ride-along solve.

    Walks the ride cells in blocks, reading each block's continuation rows and
    solving its envelope step before the next block starts; co-mapped ride states
    stay outermost. Returns the value array and the carry, and with
    `publish_replay` the consumption policy and, for a discrete ride action, the
    conditional branch banks.
    """

    savings_grid: Float1D
    """Exogenous post-decision savings grid the cells read the continuation on."""

    continuation_plan: ContinuationPlan
    """The period's continuation plan the cell reads bind into."""

    statics: _NBEGMRideAlongStatics
    """Build-time statics of the ride-along solve."""

    regime_name: RegimeName
    """Name of the regime being solved."""

    cliff_candidates: bool
    """Whether the cells read the save-to-cliff savings targets."""

    schedule_spec: _NBEGMScheduleSpec
    """The composed budget, its thresholds, and the discrete actions."""

    envelope_build: _NBEGMEnvelopeBuild
    """The per-cell envelope solve, bound to each period's inputs."""

    publish_replay: bool
    """Whether the policy and branch banks are returned alongside the carry."""

    def __call__(
        self,
        *,
        next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
        __lcm_cell_width__: int = 0,
        __lcm_interval_width__: int = 0,
        __lcm_branch_width__: int = 0,
        __lcm_stochastic_node_width__: int | None = None,
        _lcm_prepared_nested_floor: PreparedNestedFloor | None = None,
        **kwargs: EconFunctionArg,  # state grids + flat params (mixed dtypes)
    ) -> _NBEGMRideOutput | _NBEGMBoundedRideOutput | _NBEGMNestedRideOutput:
        """Solve every ride cell of one period against the filtered carries."""
        statics = replace(
            self.statics,
            cell_width=__lcm_cell_width__,
            interval_width=__lcm_interval_width__,
            branch_width=__lcm_branch_width__,
            stochastic_node_width=__lcm_stochastic_node_width__,
        )
        dtype = canonical_float_dtype()
        liquid = jnp.asarray(kwargs[statics.liquid_name], dtype=dtype)
        solve_one_cell = replace(self.envelope_build.bind_cell, statics=statics)(kwargs)
        stacks = _solve_nbegm_over_co_map(
            kwargs=kwargs,
            carry=next_regime_to_continuation,
            continuation_plan=self.continuation_plan,
            co_map_names=statics.co_map_state_names,
            solve_inner=functools.partial(
                _solve_nbegm_inner_mesh,
                kwargs=kwargs,
                prepared_source_floor=_lcm_prepared_nested_floor,
                solve_one_cell=solve_one_cell,
                savings_grid=self.savings_grid,
                continuation_plan=self.continuation_plan,
                statics=statics,
                regime_name=self.regime_name,
                cliff_candidates=self.cliff_candidates,
                schedule_spec=self.schedule_spec,
            ),
        )
        ride_shape = tuple(
            int(jnp.asarray(kwargs[name]).shape[0]) for name in statics.ride_names
        )
        numeric_stacks = stacks.output if isinstance(stacks, BoundedResult) else stacks
        conditional = None
        counts = None
        current_feasibility = None
        if isinstance(numeric_stacks, _NBEGMCellQueryPublication):
            if numeric_stacks.conditional_feasible is not None:
                masks = numeric_stacks.conditional_feasible
                masks = masks.reshape(*ride_shape, masks.shape[-2], liquid.size)
                masks = jnp.moveaxis(masks, -1, self.schedule_spec.liquid_axis_pos)
                current_feasibility = masks
            feasible = numeric_stacks.feasible
            counts = (
                sum_counts(counts=~feasible),
                sum_counts(counts=feasible & jnp.isinf(numeric_stacks.ordinary[0])),
            )
            if numeric_stacks.conditional is not None:
                model = self.envelope_build.bind_cell.query_model
                if model is None:
                    raise TypeError(
                        "Conditional NB publication requires independent "
                        "model authority."
                    )
                conditional = _assemble_nbegm_query_carry(
                    carry=numeric_stacks.conditional,
                    ride_shape=ride_shape,
                    working_state_names=statics.ride_names,
                    canonical_state_names=model.authority.state_names,
                    action_shape=tuple(
                        int(jnp.asarray(kwargs[name]).shape[0])
                        for name in model.authority.action_names
                    ),
                )
            numeric_stacks = numeric_stacks.ordinary
        outputs = _assemble_ride_carry(
            stacks=numeric_stacks,
            n_jumps=self.envelope_build.n_published_boundaries,
            n_action_branches=statics.n_action_branches,
            liquid=liquid,
            ride_shape=ride_shape,
            liquid_axis_pos=self.schedule_spec.liquid_axis_pos,
            dtype=dtype,
        )
        if conditional is not None:
            outputs = _nbegm_conditional_ride_outputs(
                outputs=outputs, conditional=conditional
            )
        if isinstance(stacks, BoundedResult):
            if counts is None:
                raise TypeError(
                    "NB bounded publication requires actual current eligibility counts."
                )
            selected = outputs if self.publish_replay else outputs[:2]
            if current_feasibility is not None:
                return BoundedResult(
                    output=(selected, counts, current_feasibility),
                    bound_status=stacks.bound_status,
                    query_status=stacks.query_status,
                    continuation_read_counts=stacks.continuation_read_counts,
                )
            return _nbegm_bounded_ride_outputs(
                outputs=selected,
                counts=counts,
                bound_status=stacks.bound_status,
                query_status=stacks.query_status,
                continuation_read_counts=stacks.continuation_read_counts,
            )
        if self.publish_replay:
            return outputs
        value_arr, carry = outputs[0], outputs[1]
        return value_arr, carry


def _nbegm_conditional_ride_outputs(
    *, outputs: _NBEGMRideReplayOutput, conditional: EGMCarry
) -> _NBEGMRideReplayOutput:
    """Replace only the carry while preserving the exact replay channel shape."""
    if len(outputs) == _NBEGM_RIDE_POLICY_ARITY:
        return outputs[0], conditional, outputs[2]
    return outputs[0], conditional, outputs[2], outputs[3], outputs[4]


def _nbegm_bounded_ride_outputs(
    *,
    outputs: _NBEGMRideOutput,
    counts: tuple[IntND, IntND],
    bound_status: tuple[BoundInputStatus, ...],
    query_status: tuple[QueryInputStatus, ...] | None,
    continuation_read_counts: ContinuationReadCounts | None = None,
) -> _NBEGMBoundedRideOutput:
    """Append the owning two-count summary to each finite ride output variant."""
    if len(outputs) == _NBEGM_RIDE_MAIN_ARITY:
        return BoundedResult(
            output=(outputs[0], outputs[1], counts),
            bound_status=bound_status,
            query_status=query_status,
            continuation_read_counts=continuation_read_counts,
        )
    if len(outputs) == _NBEGM_RIDE_POLICY_ARITY:
        return BoundedResult(
            output=(outputs[0], outputs[1], outputs[2], counts),
            bound_status=bound_status,
            query_status=query_status,
            continuation_read_counts=continuation_read_counts,
        )
    return BoundedResult(
        output=(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], counts),
        bound_status=bound_status,
        query_status=query_status,
        continuation_read_counts=continuation_read_counts,
    )


# Import-time beartype decoration precedes the concrete cell solver definition.
def _solve_nbegm_inner_mesh(
    *,
    carry: MappingProxyType[RegimeName, EGMCarry],
    comap_bindings: MappingProxyType[StateName, _RideCoordinate],
    kwargs: EconFunctionKwargs,
    solve_one_cell: "_NBEGMCellSolver",  # noqa: UP037
    savings_grid: Float1D,
    continuation_plan: ContinuationPlan,
    statics: _NBEGMRideAlongStatics,
    regime_name: RegimeName,
    cliff_candidates: bool,
    schedule_spec: _NBEGMScheduleSpec,
    prepared_source_floor: PreparedNestedFloor | None = None,
) -> _NBEGMCellOutput:
    """Solve the inner ride mesh of one co-map slice in cell blocks."""
    cell_continuation = _bind_nbegm_cell_continuation(
        kwargs=kwargs,
        carry=carry,
        comap_bindings=comap_bindings,
        savings_grid=savings_grid,
        continuation_plan=continuation_plan,
        statics=statics,
        regime_name=regime_name,
        cliff_candidates=cliff_candidates,
        schedule_spec=schedule_spec,
        current_floor=solve_one_cell.binder.current_floor,
        prepared_source_floor=prepared_source_floor,
    )
    solve_cell = functools.partial(
        _solve_nbegm_cell,
        cell_continuation=cell_continuation,
        solve_one_cell=solve_one_cell,
        comap_values=tuple(comap_bindings[name] for name in statics.co_map_state_names),
        cliff_candidates=cliff_candidates,
    )

    inner_cells = _nbegm_inner_ride_cells(kwargs=kwargs, statics=statics)
    if inner_cells is None:
        stacks = solve_cell(())
        if isinstance(stacks, BoundedResult):
            return replace(
                stacks,
                output=jax.tree.map(lambda leaf: leaf[jnp.newaxis], stacks.output),
            )
        if isinstance(stacks, _NBEGMCellQueryPublication):
            return jax.tree.map(lambda leaf: leaf[jnp.newaxis], stacks)
        return tuple(leaf[jnp.newaxis] for leaf in stacks)
    return map_over_leading_axis(
        func=solve_cell,
        xs=inner_cells,
        batch_size=statics.cell_width,
        bound_summary=cell_continuation.current_floor is not None
        or bool(continuation_plan.query_targets),
    )


# Prepared, streamed and direct reads retain distinct integrity owners.
# keyword-only-exempt: library-callback=_lcm.utils.dispatchers.map_over_leading_axis
def _solve_nbegm_cell(  # noqa: C901
    inner_values: tuple[_RideCoordinate, ...],
    *,
    cell_continuation: Callable[
        [tuple[_RideCoordinate, ...]],
        _NBEGMPreparedRead,
    ],
    solve_one_cell: "_NBEGMCellSolver",  # noqa: UP037
    comap_values: tuple[_RideCoordinate, ...],
    cliff_candidates: bool,
) -> _NBEGMCellOutput:
    """Read one inner ride cell's continuation and solve its envelope step."""
    continuation = cell_continuation(inner_values)
    if isinstance(continuation, _NBEGMPreparedCell):
        rows = continuation.continuation
        query_status = None
        read_counts = None
        if isinstance(rows, BoundedResult):
            if rows.bound_status:
                raise TypeError(
                    "A child NB read must expose integrity through its query slots."
                )
            query_status = rows.query_status
            read_counts = rows.continuation_read_counts
            rows = rows.output
        if isinstance(rows, _NBEGMIntervalContinuation):
            output = solve_one_cell(
                ride_values=(*comap_values, *inner_values),
                prepared_savings_grid=continuation.savings_grid,
                interval_continuation=rows,
            )
        else:
            output = solve_one_cell(
                ride_values=(*comap_values, *inner_values),
                prepared_savings_grid=continuation.savings_grid,
                **_nbegm_continuation_channels(
                    rows=rows, cliff_candidates=cliff_candidates
                ),
            )
        if isinstance(output, BoundedResult):
            if output.bound_status or query_status is not None:
                raise TypeError(
                    "An NB streamed cell must expose exactly its "
                    "declared child query slots."
                )
            query_status = output.query_status
            read_counts = merge_continuation_read_counts(
                left=read_counts, right=output.continuation_read_counts
            )
            output = output.output
        return BoundedResult(
            output=output,
            bound_status=(continuation.bound_status,),
            query_status=query_status,
            continuation_read_counts=read_counts,
        )
    if isinstance(continuation, BoundedResult):
        if continuation.bound_status:
            raise TypeError(
                "A child NB read must expose integrity through its query slots."
            )
        output = solve_one_cell(
            ride_values=(*comap_values, *inner_values),
            **_nbegm_continuation_channels(
                rows=continuation.output, cliff_candidates=cliff_candidates
            ),
        )
        if isinstance(output, BoundedResult):
            raise TypeError(
                "An already read NB continuation cannot publish a nested summary."
            )
        published: _NBEGMCellNumeric = output
        return BoundedResult(
            output=published,
            bound_status=continuation.bound_status,
            query_status=continuation.query_status,
            continuation_read_counts=continuation.continuation_read_counts,
        )
    if isinstance(continuation, _NBEGMIntervalContinuation):
        return solve_one_cell(
            ride_values=(*comap_values, *inner_values),
            interval_continuation=continuation,
        )
    return solve_one_cell(
        ride_values=(*comap_values, *inner_values),
        **_nbegm_continuation_channels(
            rows=continuation, cliff_candidates=cliff_candidates
        ),
    )


class _NBEGMContinuationChannels(TypedDict):
    """The exact numeric keyword roles of an already read continuation."""

    cont_value: FloatND
    cont_marginal: FloatND
    cliff_savings: NotRequired[FloatND]


def _nbegm_continuation_channels(
    *, rows: tuple[FloatND, ...], cliff_candidates: bool
) -> _NBEGMContinuationChannels:
    """Name one cell's continuation rows as the envelope solve's channels."""
    if cliff_candidates:
        cont_value, cont_marginal, cliff_savings = rows
        return {
            "cont_value": cont_value,
            "cont_marginal": cont_marginal,
            "cliff_savings": cliff_savings,
        }
    cont_value, cont_marginal = rows
    return {"cont_value": cont_value, "cont_marginal": cont_marginal}


def _nbegm_inner_ride_cells(
    *, kwargs: EconFunctionKwargs, statics: _NBEGMRideAlongStatics
) -> tuple[FloatND | IntND, ...] | None:
    """Flatten the mesh over the ride states that are not co-mapped.

    Returns `None` when every ride axis is co-mapped, so the caller solves a
    single inner cell per co-map slice.
    """
    inner_ride_names = statics.ride_names[len(statics.co_map_state_names) :]
    if not inner_ride_names:
        return None
    ride_grids = tuple(jnp.asarray(kwargs[name]) for name in inner_ride_names)
    mesh = jnp.meshgrid(*ride_grids, indexing="ij")
    return tuple(grid.ravel() for grid in mesh)


def _solve_nbegm_over_co_map(
    *,
    kwargs: EconFunctionKwargs,
    carry: MappingProxyType[RegimeName, EGMCarry],
    continuation_plan: ContinuationPlan,
    co_map_names: tuple[str, ...],
    solve_inner: Callable[..., _NBEGMCellOutput],
) -> _NBEGMCellOutput:
    """Run an inner-mesh solve once per co-map slice and merge the cell axes.

    A distributed, never-transitioning ride state is co-mapped: its axis is the
    leading ride axis, so the inner mesh solves inside an outer `vmap` that
    co-slices the child carry — each device reads only its slice, no all-gather.
    The leading co-map axes are merged into the flat inner-cell axis, keeping
    co-map states outermost — the meshgrid-`ij` order over every ride state.
    """
    return _solve_with_co_map(
        carry=carry,
        remaining=co_map_names,
        comap_bindings=MappingProxyType({}),
        kwargs=kwargs,
        continuation_plan=continuation_plan,
        solve_inner=solve_inner,
    )


def _solve_with_co_map(
    *,
    carry: MappingProxyType[RegimeName, EGMCarry],
    remaining: tuple[str, ...],
    comap_bindings: MappingProxyType[StateName, _RideCoordinate],
    kwargs: EconFunctionKwargs,
    continuation_plan: ContinuationPlan,
    solve_inner: Callable[..., _NBEGMCellOutput],
) -> _NBEGMCellOutput:
    """Peel one co-mapped state off `remaining` per `vmap` level, then solve."""
    if not remaining:
        return solve_inner(carry=carry, comap_bindings=comap_bindings)
    head, *tail = remaining
    head_grid = jnp.asarray(kwargs[head])
    # Only slice targets whose carry actually carries this co-mapped state as
    # a discrete axis; a target that does not (e.g. a kind-independent terminal
    # carry) is read whole for every slice.
    slice_targets = frozenset(
        target
        for target in continuation_plan.stateful_targets
        if head in continuation_plan.child_reads[target].discrete_state_names
    )
    in_axes = _carry_comap_in_axes(carry=carry, slice_targets=slice_targets)
    stacked = jax.vmap(
        functools.partial(
            _slice_solve,
            head=head,
            tail=tuple(tail),
            comap_bindings=comap_bindings,
            kwargs=kwargs,
            continuation_plan=continuation_plan,
            solve_inner=solve_inner,
        ),
        in_axes=(0, in_axes),
    )(head_grid, carry)
    if isinstance(stacked, BoundedResult):
        active = jnp.ones(head_grid.shape, dtype=jnp.bool_)
        return BoundedResult(
            output=jax.tree.map(
                lambda leaf: leaf.reshape(-1, *leaf.shape[2:]), stacked.output
            ),
            bound_status=reduce_bound_input_status(
                statuses=stacked.bound_status, active=active
            ),
            query_status=reduce_query_input_status(
                statuses=stacked.query_status, active=active
            ),
            continuation_read_counts=reduce_continuation_read_counts(
                counts=stacked.continuation_read_counts,
                active=active,
            ),
        )
    if isinstance(stacked, _NBEGMCellQueryPublication):
        return jax.tree.map(lambda leaf: leaf.reshape(-1, *leaf.shape[2:]), stacked)
    return tuple(leaf.reshape(-1, *leaf.shape[2:]) for leaf in stacked)


# keyword-only-exempt: library-callback=jax.vmap
def _slice_solve(
    head_value: _RideCoordinate,
    sliced_carry: MappingProxyType[RegimeName, EGMCarry],
    *,
    head: str,
    tail: tuple[str, ...],
    comap_bindings: MappingProxyType[StateName, _RideCoordinate],
    kwargs: EconFunctionKwargs,
    continuation_plan: ContinuationPlan,
    solve_inner: Callable[..., _NBEGMCellOutput],
) -> _NBEGMCellOutput:
    """Solve one slice of the co-mapped state `head` against its carry slice."""
    return _solve_with_co_map(
        carry=sliced_carry,
        remaining=tail,
        comap_bindings=MappingProxyType({**comap_bindings, head: head_value}),
        kwargs=kwargs,
        continuation_plan=continuation_plan,
        solve_inner=solve_inner,
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class _NBEGMCellContinuation:
    """One period's inputs bound into the per-cell continuation read.

    Called with one inner ride cell's coordinates (the ride states that are not
    co-mapped, in `ride_names` order), it returns the cell's continuation rows —
    expected value and expected marginal over the savings grid, with a leading
    branch axis when the regime carries a discrete action and an interval axis
    when the continuation reads the liquid state, plus the save-to-cliff savings
    targets when the one-sided read publishes jump topology. With a positive
    interval batch size it returns a branch-bindable reader instead; the interval
    scan requests and consumes one fixed-width row block at a time.
    """

    kwargs: EconFunctionKwargs
    """The period's state grids and flat params."""

    carry: MappingProxyType[RegimeName, EGMCarry]
    """The filtered child carries, sliced to this co-map slice."""

    comap_bindings: MappingProxyType[StateName, _RideCoordinate]
    """The co-mapped ride states' coordinates of this slice."""

    savings_grid: Float1D
    """Exogenous post-decision savings grid the rows are read on."""

    continuation_plan: ContinuationPlan
    """The period's continuation plan."""

    statics: _NBEGMRideAlongStatics
    """Build-time statics of the ride-along solve."""

    regime_name: RegimeName
    """Name of the regime being solved."""

    cliff_candidates: bool
    """Whether the rows carry the save-to-cliff savings targets."""

    schedule_spec: _NBEGMScheduleSpec
    """The composed budget, its thresholds, and the discrete actions."""

    current_floor: _NBEGMFloor | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Canonical source-cell floor; absent on the literal no-feature route."""

    prepared_source_floor: PreparedNestedFloor | None = None
    """Invocation-owned floor/status bank, separate from economic argument pools."""

    dtype: DTypeLike = field(init=False)
    """The canonical float dtype; derived at construction."""

    liquid: Float1D = field(init=False)
    """The liquid grid in the canonical dtype; derived at construction."""

    action_names: tuple[ActionName, ...] = field(init=False)
    """The declared discrete actions' names; derived at construction."""

    param_pool: EconFunctionKwargs = field(init=False)
    """Every non-state entry of `kwargs`; derived at construction."""

    def __post_init__(self) -> None:
        """Derive the per-period constants every cell read shares."""
        dtype = canonical_float_dtype()
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(
            self,
            "liquid",
            jnp.asarray(self.kwargs[self.statics.liquid_name], dtype=dtype),
        )
        object.__setattr__(
            self,
            "action_names",
            tuple(name for name, _ in self.schedule_spec.discrete_actions),
        )
        object.__setattr__(
            self,
            "param_pool",
            {
                key: v
                for key, v in self.kwargs.items()
                if key not in self.statics.state_names
            },
        )

    def __call__(self, ride_values: tuple[_RideCoordinate, ...]) -> _NBEGMPreparedRead:
        """Read one inner ride cell's continuation rows, or bind their reader."""
        statics = self.statics
        inner_ride_names = statics.ride_names[len(statics.co_map_state_names) :]
        cell = dict(zip(inner_ride_names, ride_values, strict=True))
        base_pool = {**self.param_pool, **self.comap_bindings, **cell}
        if self.current_floor is not None:
            _floor, nodes, status = self.current_floor.evaluate_floor(
                pool=base_pool,
                prepared_source_floor=self.prepared_source_floor,
            )
            prepared = replace(
                self,
                savings_grid=nodes,
                current_floor=None,
                prepared_source_floor=None,
            )
            return _NBEGMPreparedCell(
                continuation=_required_unprepared_nbegm_read(
                    read=prepared(ride_values)
                ),
                savings_grid=nodes,
                bound_status=status,
            )
        if statics.continuation_reads_liquid and statics.interval_width > 0:
            return _NBEGMIntervalContinuation(
                bind=functools.partial(
                    self._bind_interval_reader_for_branch, base_pool=base_pool
                ),
                query_targets=self.continuation_plan.query_targets,
                collect_continuation_read_counts=statics.collect_continuation_read_counts,
            )
        if not self.action_names:
            return self._cell_rows_for_pool(combo_pool=base_pool)

        # A discrete action that feeds the continuation reads a different
        # next-state per branch (the actions ride into `combo_pool` →
        # `next_state_func`), so the continuation is evaluated once per class of
        # branches agreeing on the actions the read consumes and gathered onto a
        # leading branch axis over the product of the declared grids. A
        # budget-only action leaves one class, so its branches share one read.
        # The class body compiles once and runs at a fixed vector width, so
        # per-class intermediates never all sit in flight whatever the partition.
        branch_bindings = self.schedule_spec.branch_bindings
        representatives = statics.continuation_representatives
        class_rows = map_over_leading_axis(
            func=functools.partial(self._rows_for_codes, base_pool=base_pool),
            xs=_stacked_branch_codes(
                branch_bindings=tuple(
                    branch_bindings[branch] for branch in representatives
                ),
                action_names=self.action_names,
            ),
            batch_size=statics.branch_width,
            bound_summary=bool(self.continuation_plan.query_targets)
            or statics.collect_continuation_read_counts,
        )
        if len(representatives) == len(branch_bindings):
            return class_rows
        class_of_branch = jnp.asarray(
            statics.continuation_class_of_branch, dtype=jnp.int32
        )
        if isinstance(class_rows, BoundedResult):
            return replace(
                class_rows,
                output=jax.tree.map(
                    functools.partial(_gather_leading_rows, indices=class_of_branch),
                    class_rows.output,
                ),
            )
        return jax.tree.map(
            functools.partial(_gather_leading_rows, indices=class_of_branch),
            class_rows,
        )

    # keyword-only-exempt: library-callback=_lcm.utils.dispatchers.map_over_leading_axis
    def _rows_for_codes(
        self, codes_row: IntND, *, base_pool: EconFunctionKwargs
    ) -> _NBEGMContinuationRows:
        """Read the rows of the branch class one representative's codes name."""
        binding = {
            name: codes_row[position] for position, name in enumerate(self.action_names)
        }
        return self._cell_rows_for_pool(combo_pool={**base_pool, **binding})

    def _bind_interval_reader_for_branch(
        self, *, action_binding: Mapping[str, IntND], base_pool: EconFunctionKwargs
    ) -> _NBEGMIntervalBlockReader:
        """Bind one branch's action codes into the cell's interval reader."""
        return self._bind_cell_interval_reader_for_pool(
            combo_pool={**base_pool, **action_binding}
        )

    def _bind_cell_interval_reader_for_pool(
        self, *, combo_pool: EconFunctionKwargs
    ) -> _NBEGMIntervalBlockReader:
        """Bind one branch and return read access to the intervals a scan asks for."""
        from _lcm.egm.nbegm_breakpoints import interval_midpoints  # noqa: PLC0415

        statics = self.statics
        cell = {name: combo_pool[name] for name in statics.ride_names}
        cell_action_binding = {
            name: combo_pool[name] for name in self.action_names if name in combo_pool
        }
        breakpoints, _ = _nbegm_cell_breakpoints(
            statics=statics,
            kwargs=self.kwargs,
            cell=cell,
            liquid_grid=self.liquid,
            dtype=self.dtype,
            action_binding=cell_action_binding,
        )
        midpoints = interval_midpoints(liquid_grid=self.liquid, breakpoints=breakpoints)
        cliff_targets = (
            self._cliff_targets_for(combo_pool=combo_pool, midpoints=midpoints)
            if self.cliff_candidates
            else None
        )
        return _NBEGMIntervalBlockReader(
            midpoints=midpoints,
            cliff_targets=cliff_targets,
            interval_rows=self._interval_rows_for(combo_pool=combo_pool),
        )

    def _cell_rows_for_pool(
        self, *, combo_pool: EconFunctionKwargs
    ) -> _NBEGMContinuationRows:
        """Read one cell's rows under one combo pool, interval by interval if needed."""
        from _lcm.egm.continuation import bind_continuation  # noqa: PLC0415
        from _lcm.egm.nbegm_breakpoints import interval_midpoints  # noqa: PLC0415

        statics = self.statics
        cell = {name: combo_pool[name] for name in statics.ride_names}

        if statics.continuation_reads_liquid:
            # The next-period state law carries a current-asset boundary, so
            # the continuation is constant only within each declared interval.
            # Bind the liquid (Euler) state to each interval's representative
            # node, building one continuation row per interval. `lax.map`
            # compiles the continuation DAG once and XLA iterates, rather than a
            # Python unroll that bakes one copy of the per-cell DAG into the
            # graph per interval. The interval partition follows the action when
            # it feeds the schedule variable — the branch rides in `combo_pool`,
            # so its per-branch breakpoints match the envelope's.
            cell_action_binding = {
                name: combo_pool[name]
                for name in self.action_names
                if name in combo_pool
            }
            breakpoints, _ = _nbegm_cell_breakpoints(
                statics=statics,
                kwargs=self.kwargs,
                cell=cell,
                liquid_grid=self.liquid,
                dtype=self.dtype,
                action_binding=cell_action_binding,
            )
            midpoints = interval_midpoints(
                liquid_grid=self.liquid, breakpoints=breakpoints
            )
            cliff_targets = (
                self._cliff_targets_for(combo_pool=combo_pool, midpoints=midpoints)
                if self.cliff_candidates
                else None
            )
            interval_inputs = (
                (midpoints,) if cliff_targets is None else (midpoints, cliff_targets)
            )
            rows = map_over_leading_axis(
                func=self._interval_rows_for(combo_pool=combo_pool),
                xs=interval_inputs,
                batch_size=statics.interval_width,
                bound_summary=bool(self.continuation_plan.query_targets)
                or statics.collect_continuation_read_counts,
                active=(
                    ~(
                        jnp.concatenate(
                            [jnp.asarray([-jnp.inf], dtype=self.dtype), breakpoints]
                        )
                        >= jnp.concatenate(
                            [breakpoints, jnp.asarray([jnp.inf], dtype=self.dtype)]
                        )
                    )
                    if self.continuation_plan.query_targets
                    else None
                ),
            )
            if cliff_targets is None:
                return rows
            if isinstance(rows, BoundedResult):
                published_rows: tuple[FloatND, ...] = (*rows.output, cliff_targets)
                return BoundedResult(
                    output=published_rows,
                    bound_status=rows.bound_status,
                    query_status=rows.query_status,
                    continuation_read_counts=rows.continuation_read_counts,
                )
            return (*rows, cliff_targets)

        continuation = bind_continuation(
            plan=self.continuation_plan,
            combo_pool=combo_pool,
            next_regime_to_continuation=self.carry,
            dtype=self.dtype,
            stochastic_node_width=statics.stochastic_node_width,
            co_map_state_names=statics.co_map_state_names,
            **(
                {"collect_continuation_read_counts": True}
                if statics.collect_continuation_read_counts
                else {}
            ),
        )
        cliff_targets = (
            self._cliff_targets_for(combo_pool=combo_pool, midpoints=None)
            if self.cliff_candidates
            else None
        )
        rows = jax.vmap(continuation)(
            self.savings_grid
            if cliff_targets is None
            else _query_with_targets(
                savings_grid=self.savings_grid, targets=cliff_targets
            )
        )
        rows = _fold_nbegm_read_rows(
            rows=rows, savings_grid=self.savings_grid, targets=cliff_targets
        )
        if cliff_targets is None:
            return rows
        if isinstance(rows, BoundedResult):
            published_rows: tuple[FloatND, ...] = (*rows.output, cliff_targets)
            return BoundedResult(
                output=published_rows,
                bound_status=rows.bound_status,
                query_status=rows.query_status,
                continuation_read_counts=rows.continuation_read_counts,
            )
        return (*rows, cliff_targets)

    def _cliff_targets_for(
        self,
        *,
        combo_pool: EconFunctionKwargs,
        midpoints: Float1D | None,
    ) -> FloatND:
        """The cell's save-to-cliff savings targets under one combo pool.

        Under the one-sided read, the cell also evaluates the blended continuation
        at each self-read cliff's one-sided savings targets; the extra columns ride
        at the end of the savings axis and the envelope solve adds them as point
        candidates.
        """
        return _cliff_savings_targets(
            continuation_plan=self.continuation_plan,
            regime_name=self.regime_name,
            statics=self.statics,
            child_carry=self.carry[self.regime_name],
            combo_pool=combo_pool,
            savings_grid=self.savings_grid,
            dtype=self.dtype,
            midpoints=midpoints,
        )

    def _interval_rows_for(
        self, *, combo_pool: EconFunctionKwargs
    ) -> Callable[[tuple[FloatND, ...]], _NBEGMReadResult]:
        """The per-interval row read with this cell's combo pool bound."""
        return functools.partial(
            _interval_rows,
            combo_pool=combo_pool,
            liquid_name=self.statics.liquid_name,
            continuation_plan=self.continuation_plan,
            carry=self.carry,
            dtype=self.dtype,
            stochastic_node_width=self.statics.stochastic_node_width,
            co_map_names=self.statics.co_map_state_names,
            savings_grid=self.savings_grid,
            collect_continuation_read_counts=self.statics.collect_continuation_read_counts,
        )


def _bind_nbegm_cell_continuation(
    *,
    kwargs: EconFunctionKwargs,
    carry: MappingProxyType[RegimeName, EGMCarry],
    comap_bindings: MappingProxyType[StateName, _RideCoordinate],
    savings_grid: Float1D,
    continuation_plan: ContinuationPlan,
    statics: _NBEGMRideAlongStatics,
    regime_name: RegimeName,
    cliff_candidates: bool,
    schedule_spec: _NBEGMScheduleSpec,
    current_floor: _NBEGMFloor | None = None,
    prepared_source_floor: PreparedNestedFloor | None = None,
) -> _NBEGMCellContinuation:
    """Bind one period's inputs into the per-cell continuation read.

    The returned callable takes one inner ride cell's coordinates (the ride states
    that are not co-mapped, in `ride_names` order) and returns the cell's
    continuation rows: expected value and expected marginal over the savings grid,
    each with a leading branch axis when the regime carries a discrete action and an
    interval axis when the continuation reads the liquid state, plus the
    save-to-cliff savings targets when the one-sided read publishes jump topology.
    With a positive interval batch size it instead returns a branch-bindable reader;
    the interval scan requests and consumes one fixed-width row block at a time.
    """
    return _NBEGMCellContinuation(
        kwargs=kwargs,
        carry=carry,
        comap_bindings=comap_bindings,
        savings_grid=savings_grid,
        continuation_plan=continuation_plan,
        statics=statics,
        regime_name=regime_name,
        cliff_candidates=cliff_candidates,
        schedule_spec=schedule_spec,
        current_floor=current_floor,
        prepared_source_floor=prepared_source_floor,
    )


# keyword-only-exempt: library-callback=jax.vmap
def _interval_rows(
    interval_inputs: tuple[FloatND, ...],
    *,
    combo_pool: EconFunctionKwargs,
    liquid_name: str,
    continuation_plan: ContinuationPlan,
    carry: MappingProxyType[RegimeName, EGMCarry],
    dtype: DTypeLike,
    stochastic_node_width: int | None,
    co_map_names: tuple[str, ...],
    savings_grid: Float1D,
    collect_continuation_read_counts: bool = False,
) -> _NBEGMReadResult:
    """One interval's continuation rows with the liquid state bound to its node."""
    from _lcm.egm.continuation import bind_continuation  # noqa: PLC0415

    midpoint, *interval_targets = interval_inputs
    interval_pool = {**combo_pool, liquid_name: midpoint}
    interval_continuation = bind_continuation(
        plan=continuation_plan,
        combo_pool=interval_pool,
        next_regime_to_continuation=carry,
        dtype=dtype,
        stochastic_node_width=stochastic_node_width,
        co_map_state_names=co_map_names,
        **(
            {"collect_continuation_read_counts": True}
            if collect_continuation_read_counts
            else {}
        ),
    )
    query = (
        savings_grid
        if not interval_targets
        else _query_with_targets(savings_grid=savings_grid, targets=interval_targets[0])
    )
    rows = jax.vmap(interval_continuation)(query)
    return _fold_nbegm_read_rows(
        rows=rows,
        savings_grid=savings_grid,
        targets=None if not interval_targets else interval_targets[0],
    )


def _query_with_targets(*, savings_grid: Float1D, targets: FloatND) -> Float1D:
    """The savings grid with the cliff targets appended, dead targets parked."""
    return jnp.concatenate(
        [
            savings_grid,
            jnp.where(jnp.isnan(targets), savings_grid[0], targets),
        ]
    )


# keyword-only-exempt: library-callback=jax.tree.map
def _gather_leading_rows(leaf: FloatND, *, indices: IntND) -> FloatND:
    """Gather one class row per branch along the leading axis."""
    return leaf[indices]


def _split_cliff_columns(
    *,
    cont_value: FloatND,
    cont_marginal: FloatND,
    n_nodes: int,
    has_cliff_columns: bool,
) -> tuple[FloatND, FloatND, FloatND | None]:
    """Split a cell's continuation rows into node columns and cliff columns.

    The continuation core rides the save-to-cliff targets' values at the end
    of the savings axis; the leading `n_nodes` columns are the savings-node
    rows the EGM step consumes, the rest feed the point-candidate family.
    """
    if not has_cliff_columns:
        return cont_value, cont_marginal, None
    return (
        cont_value[..., :n_nodes],
        cont_marginal[..., :n_nodes],
        cont_value[..., n_nodes:],
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class _NBEGMCellBinder:
    """Bind one period's states and params into the per-cell envelope solve.

    Calling it with the period's `kwargs` returns the bound cell solver, which
    takes `ride_values` (every ride state's coordinate, co-mapped states first),
    `cont_value`, `cont_marginal`, and optionally `cliff_savings` for one cell —
    or an interval continuation reader — and returns that cell's packaged
    result rows.
    """

    savings_grid: Float1D
    """Exogenous post-decision savings grid the cells solve on."""

    schedule_spec: _NBEGMScheduleSpec
    """The composed budget, utility, discount factor, and discrete actions."""

    statics: _NBEGMRideAlongStatics
    """Build-time statics of the ride-along solve."""

    is_epstein_zin: bool
    """Whether the step inverts the recursive Euler equation."""

    feasibility_constraints: tuple[_NBEGMFeasibilityConstraint, ...]
    """Declared feasibility boundaries resolved per cell."""

    action_lower: ScalarFloat
    """Lower bracket of the numerical Euler inversion."""

    action_upper: ScalarFloat
    """Initial upper bracket of the numerical Euler inversion."""

    utility_arg_names: frozenset[ReferenceName]
    """The utility DAG's parameter names; an action binds only if named here."""

    n_published_boundaries: int
    """Jump preimages plus feasibility boundaries each carry row publishes."""

    schedule_kinds: tuple[str, ...]
    """Every declared breakpoint's kind, in source order."""

    current_floor: _NBEGMFloor | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """One canonical per-cell evaluator, shared by continuation and branch solves."""

    query_model: NBEGMQueryModel | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Independent target authority requiring every conditional native tuple."""

    current_predicates: tuple[Callable[..., BoolND], ...] | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Canonical nonbudget current-state predicates for feature counts and max."""

    def __call__(self, kwargs: EconFunctionKwargs) -> _NBEGMCellSolver:
        """Bind one period's inputs into the per-cell solve."""
        dtype = canonical_float_dtype()
        return _NBEGMCellSolver(
            binder=self,
            kwargs=kwargs,
            dtype=dtype,
            liquid=jnp.asarray(kwargs[self.statics.liquid_name], dtype=dtype),
            coh_params={
                name: kwargs[name] for name in self.schedule_spec.coh_param_names
            },
            utility_params={
                name: kwargs[name] for name in self.statics.utility_param_names
            },
            discount_params={
                name: kwargs[name] for name in self.statics.discount_param_names
            },
            # Epstein-Zin: the aggregator curvature is `rho = 1/psi` where `psi` is
            # the Koopmans aggregator's `intertemporal_elasticity_of_substitution`.
            # The step reads the continuation pair as `(nu, dnu/ds)` and inverts
            # the recursive Euler equation; `None` keeps the additive
            # expected-utility step.
            inverse_eis=(
                1.0
                / cast(
                    "ScalarFloat",
                    kwargs[
                        "koopmans_aggregator__intertemporal_elasticity_of_substitution"
                    ],
                )
                if self.is_epstein_zin
                else None
            ),
        )


@dataclass(frozen=True, kw_only=True)
class _NBEGMEnvelopeBuild:
    """The envelope half of a ride-along solve, as a per-cell binder."""

    bind_cell: _NBEGMCellBinder
    """Bind one period's states and params into the per-cell envelope solve.

    The bound callable takes `ride_values` (every ride state's coordinate, co-mapped
    states first), `cont_value`, `cont_marginal`, and optionally `cliff_savings` for
    one cell, and returns that cell's packaged result rows."""

    n_published_boundaries: int
    """Jump preimages plus feasibility boundaries each carry row publishes."""


def _build_nbegm_envelope_core(
    *,
    savings_grid: Float1D,
    schedule_spec: _NBEGMScheduleSpec,
    statics: _NBEGMRideAlongStatics,
    is_epstein_zin: bool = False,
    feasibility_constraints: tuple[_NBEGMFeasibilityConstraint, ...] = (),
    query_model: NBEGMQueryModel | None = None,
    current_floor: _NBEGMFloor | None = None,
    current_predicates: tuple[Callable[..., BoolND], ...] | None = None,
    materializer: DatedGenerationMaterializer | None = None,
) -> _NBEGMEnvelopeBuild:
    """Build the EGM/envelope half of the ride-along solve.

    Per ride-along cell this re-derives the budget schedule, discount factor, and
    utility from the same (states, params), then solves the 1-D continuous-budget step
    against the cell's continuation rows. The interval regime runs the per-interval
    continuation step; the non-interval regime runs the multi-interval or unified jump
    step. The cells stack into the value array and carry with the ride-along axes
    leading the liquid axis, matching the canonical layout.

    The per-cell solve is published twice: as `bind_cell`, which the tile-local
    production core (`_build_nbegm_tiled_core`) runs per cell block right after that
    block's continuation read, and as the split oracle's `core`, which consumes the
    complete continuation stacks the split continuation core materializes.
    """
    from _lcm.egm.preferences import (  # noqa: PLC0415
        NEWTON_ACTION_FLOOR,
        newton_action_ceiling,
    )

    # This route composes period utility from the DAG, so the Euler equation has no
    # closed form to invert and its root is always bracketed numerically. The
    # clamped near-zero-marginal corner whose root exceeds the bracket lands far to
    # the right and is discarded by the upper envelope.
    action_upper = (
        newton_action_ceiling(savings_grid)
        if materializer is None
        else materializer.evaluate_validation(
            function=newton_action_ceiling, arguments={"scale_grid": savings_grid}
        )
    )
    action_lower = (
        jnp.asarray(NEWTON_ACTION_FLOOR, dtype=action_upper.dtype)
        if materializer is None
        else materializer.evaluate_validation(
            function=functools.partial(
                jnp.asarray, NEWTON_ACTION_FLOOR, dtype=action_upper.dtype
            ),
            arguments={},
        )
    )

    n_feasibility_boundaries = sum(
        len(constraint.program.surfaces) for constraint in feasibility_constraints
    )
    n_published_boundaries = statics.n_published_jumps + n_feasibility_boundaries
    return _NBEGMEnvelopeBuild(
        bind_cell=_NBEGMCellBinder(
            savings_grid=savings_grid,
            schedule_spec=schedule_spec,
            statics=statics,
            is_epstein_zin=is_epstein_zin,
            feasibility_constraints=feasibility_constraints,
            action_lower=action_lower,
            action_upper=action_upper,
            # The action binds into a branch's period utility only when the utility
            # DAG reads it (a leisure/effort-like term); otherwise the binding is
            # dropped so a utility that does not name the action is called with its
            # own arguments alone.
            utility_arg_names=frozenset(
                inspect.signature(schedule_spec.utility_dag).parameters
            ),
            n_published_boundaries=n_published_boundaries,
            schedule_kinds=tuple(source.kind for source in statics.sources),
            query_model=query_model,
            current_floor=current_floor,
            current_predicates=current_predicates,
        ),
        n_published_boundaries=n_published_boundaries,
    )


class _NBEGMCellGrid(NamedTuple):
    """Existing shared cell query axes and owner-aware feasibility partition."""

    query_grid: Float1D
    endog_row: Float1D
    unsort: IntND
    published_boundaries: Float1D
    feasibility_partition: ResolvedAxisPartition | None
    feasible_interval_mask: BoolND | None


def _nbegm_cell_publication_grid(
    *,
    binder: _NBEGMCellBinder,
    kwargs: EconFunctionKwargs,
    cell: Mapping[StateName, EconFunctionArg],
    liquid: Float1D,
    dtype: DTypeLike,
) -> _NBEGMCellGrid:
    """Prepare the same branch-independent physical and one-sided query nodes."""
    statics = binder.statics
    query_grid = liquid
    endog_row = liquid
    unsort = jnp.arange(liquid.shape[0], dtype=jnp.int32)
    published_boundaries = jnp.zeros((0,), dtype=dtype)
    feasibility_partition = None
    feasible_interval_mask = None
    if statics.n_published_jumps:
        breakpoints, jump_positions = _nbegm_cell_breakpoints(
            statics=statics,
            kwargs=kwargs,
            cell=cell,
            liquid_grid=liquid,
            dtype=dtype,
        )
        jumps = jnp.stack([breakpoints[p] for p in jump_positions])
        query_grid, endog_row, unsort = _augment_liquid_with_jump_sides(
            liquid_grid=liquid, jumps=jumps
        )
        published_boundaries = jumps
    elif binder.feasibility_constraints:
        cell_breakpoints, _jump_positions = _nbegm_cell_breakpoints(
            statics=statics,
            kwargs=kwargs,
            cell=cell,
            liquid_grid=liquid,
            dtype=dtype,
        )
        feasibility = _resolve_nbegm_feasibility(
            constraints=binder.feasibility_constraints,
            liquid=liquid,
            schedule_breakpoints=cell_breakpoints,
            schedule_kinds=binder.schedule_kinds,
            params=cast("Mapping[str, FloatND | IntND | BoolND]", kwargs),
        )
        query_grid, unsort = _augment_liquid_for_feasibility(
            liquid=liquid,
            feasibility=feasibility,
        )
        endog_row = query_grid
        published_boundaries = feasibility.carry_boundary_values
        feasibility_partition = feasibility.partition
        feasible_interval_mask = feasibility.feasible_interval_mask
    return _NBEGMCellGrid(
        query_grid,
        endog_row,
        unsort,
        published_boundaries,
        feasibility_partition,
        feasible_interval_mask,
    )


class _NBEGMRideSourcePoint(NamedTuple):
    """Original point channels, including mapped branch and query axes."""

    value: FloatND
    marginal: FloatND
    policy: FloatND
    savings: FloatND
    native_owner: IntND
    action_codes: IntND


@dataclass(frozen=True, eq=False, kw_only=True)
class _NBEGMCellSolver:
    """The per-cell envelope solve of one period, bound to its states and params."""

    binder: _NBEGMCellBinder
    """The period-invariant half of the solve."""

    kwargs: EconFunctionKwargs
    """The period's state grids and flat params."""

    dtype: DTypeLike
    """The canonical float dtype."""

    liquid: Float1D
    """The liquid grid in the canonical dtype."""

    coh_params: EconFunctionKwargs
    """The composed budget's params, read off `kwargs`."""

    utility_params: EconFunctionKwargs
    """The utility DAG's params, read off `kwargs`."""

    discount_params: EconFunctionKwargs
    """The per-cell discount factor's params, read off `kwargs`."""

    inverse_eis: FloatND | None
    """`1/psi` under Epstein-Zin, else `None` for the additive step."""

    def construct_point(
        self,
        *,
        ride_values: tuple[_RideCoordinate, ...],
        query: Float1D,
        cont_value: FloatND,
        cont_marginal: FloatND,
    ) -> _NBEGMRideSourcePoint:
        """Read the complete original bank at points without shrinking its support.

        The supplied continuation rows are the output of this period's original
        `_NBEGMCellContinuation`, including its branch-class leading axis.
        Branch-specific candidate construction and native selection stay in the
        original branch solver. This first point route has no side augmentation.
        """
        binder = self.binder
        statics = binder.statics
        if (
            statics.has_jump
            or statics.continuation_reads_liquid
            or binder.current_predicates is not None
            or binder.feasibility_constraints
            or binder.n_published_boundaries
            or binder.current_floor is not None
        ):
            raise UnsupportedOperationError(
                "Owned NB ride points require continuous literal-floor "
                "ordinary-child cells without extra feasibility boundaries."
            )
        cell = MappingProxyType(dict(zip(statics.ride_names, ride_values, strict=True)))
        branch_solver = _NBEGMBranchSolver(
            cell_solver=self,
            cell=cell,
            query_grid=self.liquid,
            feasibility_partition=None,
            feasible_interval_mask=None,
            interval_continuation=None,
        )
        names = tuple(name for name, _codes in binder.schedule_spec.discrete_actions)
        if names:
            codes = jnp.stack(
                jnp.meshgrid(
                    *(
                        jnp.asarray(self.kwargs[name], dtype=jnp.int32)
                        for name in names
                    ),
                    indexing="ij",
                ),
                axis=-1,
            ).reshape(-1, len(names))
            inputs = _branch_inputs(
                codes=codes,
                cont_value=cont_value,
                cont_marginal=cont_marginal,
                extra_cont_value=None,
                cliff_savings=None,
            )
            rows = map_over_leading_axis(
                func=functools.partial(
                    _solve_one_owned_branch,
                    branch_solver=branch_solver,
                    branch_action_names=names,
                    query=query,
                ),
                xs=inputs,
                batch_size=statics.branch_width,
            )
            # This is the original hard branch max, in its original mesh order.
            # The resulting row positions select every associated channel.
            modal = jnp.argmax(rows.value, axis=0)
            index = jnp.arange(rows.value.shape[1])
            return _NBEGMRideSourcePoint(
                value=rows.value[modal, index],
                marginal=rows.marginal[modal, index],
                policy=rows.policy[modal, index],
                savings=rows.savings[modal, index],
                native_owner=rows.native_owner[modal, index],
                action_codes=rows.action_codes[modal, index],
            )
        row = branch_solver.solve(
            action_binding={},
            branch_cont_value=cont_value,
            branch_cont_marginal=cont_marginal,
            branch_extra_cont_value=None,
            branch_cliff_savings=None,
            source_query=query,
            return_owner=True,
        )
        return _NBEGMRideSourcePoint(
            value=row.value,
            marginal=row.marginal,
            policy=row.policy,
            savings=row.query_data.savings,
            native_owner=row.owner,
            action_codes=jnp.zeros((query.shape[0], 0), dtype=jnp.int32),
        )

    def __call__(
        self,
        *,
        ride_values: tuple[_RideCoordinate, ...],
        cont_value: FloatND | None = None,
        cont_marginal: FloatND | None = None,
        cliff_savings: FloatND | None = None,
        interval_continuation: _NBEGMIntervalContinuation | None = None,
        prepared_savings_grid: Float1D | None = None,
    ) -> _NBEGMCellOutput:
        """Solve one ride cell against its continuation rows or reader."""
        if prepared_savings_grid is not None:
            prepared = replace(
                self, binder=replace(self.binder, savings_grid=prepared_savings_grid)
            )
            return prepared(
                ride_values=ride_values,
                cont_value=cont_value,
                cont_marginal=cont_marginal,
                cliff_savings=cliff_savings,
                interval_continuation=interval_continuation,
            )
        binder = self.binder
        statics = binder.statics
        schedule_spec = binder.schedule_spec
        kwargs = self.kwargs
        liquid = self.liquid
        dtype = self.dtype
        cont_value, cont_marginal, extra_cont_value = _nbegm_cell_continuation_columns(
            cont_value=cont_value,
            cont_marginal=cont_marginal,
            interval_continuation=interval_continuation,
            savings_grid=binder.savings_grid,
            cliff_savings=cliff_savings,
        )
        cell = MappingProxyType(dict(zip(statics.ride_names, ride_values, strict=True)))

        # With published jump breakpoints, the cell publishes each jump's preimage
        # and its exact one-sided value limits: the liquid query grid is augmented
        # with a point just inside each side of every jump, solved in the same call,
        # and split back out positionally. A published jump shares one query grid
        # across the branches, so the action cannot enter its schedule variable
        # (guarded), and the cell-level partition is branch-independent. The bridged
        # read skips the augmentation; each branch then partitions on its own
        # breakpoints (recomputed inside the branch solve) over the plain liquid
        # grid.
        (
            query_grid,
            endog_row,
            unsort,
            published_boundaries,
            feasibility_partition,
            feasible_interval_mask,
        ) = _nbegm_cell_publication_grid(
            binder=binder, kwargs=kwargs, cell=cell, liquid=liquid, dtype=dtype
        )

        branch_solver = _NBEGMBranchSolver(
            cell_solver=self,
            cell=cell,
            query_grid=query_grid,
            feasibility_partition=feasibility_partition,
            feasible_interval_mask=feasible_interval_mask,
            interval_continuation=interval_continuation,
        )

        # The discrete actions are enveloped over per cell: each branch is one
        # combination of their codes, solving the continuous subproblem with that
        # combination bound into cash-on-hand against its own continuation slice,
        # and the joint discrete choice is taken by the upper envelope. When an
        # action feeds a co-state, the continuation core adds a leading branch
        # axis over the same combinations (branch `pos` reads slice `pos`); when
        # they feed only the budget those slices are identical. Under a published
        # jump each branch's row spans the jump-augmented query grid, so the
        # envelope max takes the discrete choice over each branch's one-sided
        # cliff limits and the carry keeps the augmented row.
        #
        # Shared-parent-grid invariant: the pointwise max over branches is valid
        # only because every branch's result row is evaluated on the same parent
        # liquid query grid. Branch-specific inputs — continuation slices,
        # child-cliff candidates, per-branch breakpoint partitions — may change
        # branch values and candidate sets, never the parent abscissae. The one
        # violation (an action moving a *published* parent jump preimage per
        # branch) is refused at build (`_fail_if_unsupported_ride_discrete`).
        conditional_feasible = None
        branch_action_names = tuple(name for name, _ in schedule_spec.discrete_actions)
        if branch_action_names:
            # The shared EGM dispatcher batches the branch subproblem at
            # `branch_width`, or vectorizes the whole axis when it is 0.
            # The branch axis is never Python-unrolled. Optional
            # branch inputs enter the mapped pytree only when present.
            branch_inputs = _branch_inputs(
                codes=jnp.stack(
                    jnp.meshgrid(
                        *(
                            jnp.asarray(kwargs[name], dtype=jnp.int32)
                            for name in branch_action_names
                        ),
                        indexing="ij",
                    ),
                    axis=-1,
                ).reshape(-1, len(branch_action_names)),
                cont_value=cont_value,
                cont_marginal=cont_marginal,
                extra_cont_value=extra_cont_value,
                cliff_savings=cliff_savings,
            )
            branch_rows = map_over_leading_axis(
                func=functools.partial(
                    _solve_one_branch,
                    branch_solver=branch_solver,
                    branch_action_names=branch_action_names,
                ),
                xs=branch_inputs,
                batch_size=statics.branch_width,
                bound_summary=interval_continuation is not None
                and (
                    bool(interval_continuation.query_targets)
                    or statics.collect_continuation_read_counts
                ),
            )
            branch_rows, query_summary = _split_nbegm_step_summary(rows=branch_rows)
            value_stack, marginal_stack, policy_stack = branch_rows[:3]
            if binder.current_predicates is not None:
                feasible = jax.vmap(branch_solver.feasibility_for_codes)(
                    branch_inputs["codes"]
                )
                if (
                    binder.query_model is not None
                    and binder.query_model.retain_nested_feasibility
                ):
                    conditional_feasible = feasible
                value_stack = jnp.where(feasible, value_stack, -jnp.inf)
                marginal_stack = jnp.where(feasible, marginal_stack, 0.0)
                policy_stack = jnp.where(feasible, policy_stack, jnp.nan)
            modal = jnp.argmax(value_stack, axis=0)
            index = jnp.arange(value_stack.shape[1])
            value_row = value_stack[modal, index]
            marginal_row = marginal_stack[modal, index]
            policy_row = policy_stack[modal, index]
            branch_stacks = (value_stack, policy_stack)
            if binder.current_predicates is not None:
                feasible = jnp.any(feasible, axis=0)
        else:
            branch_rows = branch_solver.solve(
                action_binding={},
                branch_cont_value=cont_value,
                branch_cont_marginal=cont_marginal,
                branch_extra_cont_value=extra_cont_value,
                branch_cliff_savings=cliff_savings,
            )
            branch_rows, query_summary = _split_nbegm_step_summary(rows=branch_rows)
            value_row, marginal_row, policy_row = branch_rows[:3]
            if binder.current_predicates is not None:
                feasible = branch_solver.feasibility_for_codes(
                    jnp.zeros((0,), dtype=jnp.int32)
                )
                if (
                    binder.query_model is not None
                    and binder.query_model.retain_nested_feasibility
                ):
                    conditional_feasible = feasible[jnp.newaxis]
                value_row = jnp.where(feasible, value_row, -jnp.inf)
                marginal_row = jnp.where(feasible, marginal_row, 0.0)
                policy_row = jnp.where(feasible, policy_row, jnp.nan)
            branch_stacks = None

        return _finish_nbegm_cell_publication(
            binder=binder,
            branch_rows=branch_rows,
            value_row=value_row,
            marginal_row=marginal_row,
            policy_row=policy_row,
            branch_stacks=branch_stacks,
            unsort=unsort,
            n_liquid=liquid.shape[0],
            endog_row=endog_row,
            published_boundaries=published_boundaries,
            query_grid=query_grid,
            has_actions=bool(branch_action_names),
            feasible=feasible if binder.current_predicates is not None else None,
            conditional_feasible=conditional_feasible,
            query_summary=query_summary,
        )


def _split_nbegm_step_summary(
    *,
    rows: _NBEGMMappedStepChannels | BoundedResult[_NBEGMMappedStepChannels],
) -> tuple[_NBEGMMappedStepChannels, BoundedResult[_NBEGMMappedStepChannels] | None]:
    """Separate the actual numeric branch channels from their native summary."""
    if isinstance(rows, BoundedResult):
        return rows.output, rows
    return rows, None


def _finish_nbegm_cell_publication(
    *,
    binder: _NBEGMCellBinder,
    branch_rows: _NBEGMMappedStepChannels,
    value_row: FloatND,
    marginal_row: FloatND,
    policy_row: FloatND,
    branch_stacks: tuple[FloatND, FloatND] | None,
    unsort: IntND,
    n_liquid: int,
    endog_row: Float1D,
    published_boundaries: Float1D,
    query_grid: Float1D,
    has_actions: bool,
    feasible: BoolND | None,
    query_summary: BoundedResult[_NBEGMMappedStepChannels] | None,
    conditional_feasible: BoolND | None = None,
) -> _NBEGMCellOutput:
    """Publish ordinary cell channels beside their actual conditional records."""
    ordinary = _package_nbegm_cell_result(
        value_row=value_row,
        marginal_row=marginal_row,
        policy_row=policy_row,
        branch_stacks=branch_stacks,
        n_published_boundaries=binder.n_published_boundaries,
        unsort=unsort,
        n_liquid=n_liquid,
        endog_row=endog_row,
        published_boundaries=published_boundaries,
    )
    if binder.current_predicates is not None:
        if feasible is None:
            raise TypeError(
                "Current NB predicates require their actual feasibility row."
            )
        ordinary = _NBEGMCellQueryPublication(
            ordinary=ordinary,
            conditional=None
            if binder.query_model is None
            else _conditional_nbegm_cell_carry(
                rows=_required_nbegm_query_step(step=branch_rows),
                query_grid=query_grid,
                published_boundaries=published_boundaries,
                has_actions=has_actions,
            ),
            feasible=feasible[unsort][:n_liquid],
            conditional_feasible=None
            if conditional_feasible is None
            else conditional_feasible[..., unsort][..., :n_liquid],
        )
    if query_summary is not None:
        published: _NBEGMCellNumeric = ordinary
        return BoundedResult(
            output=published,
            bound_status=query_summary.bound_status,
            query_status=query_summary.query_status,
            continuation_read_counts=query_summary.continuation_read_counts,
        )
    return ordinary


@dataclass(frozen=True, eq=False, kw_only=True)
class _NBEGMBranchSolver:
    """One cell's continuous subproblem, solved per discrete branch.

    The jump augmentation is continuation-independent and computed once per
    cell by the cell solver; every branch reads it from here.
    """

    cell_solver: _NBEGMCellSolver
    """The cell's period-bound solve."""

    cell: MappingProxyType[StateName, _RideCoordinate]
    """Every ride state's coordinate of this cell."""

    query_grid: Float1D
    """The liquid query grid, augmented by published jump sides or feasibility."""

    feasibility_partition: ResolvedAxisPartition | None
    """The resolved feasibility partition, or `None` without constraints."""

    feasible_interval_mask: BoolND | None
    """Per interval, whether it is feasible, or `None` without constraints."""

    interval_continuation: _NBEGMIntervalContinuation | None
    """The branch-bindable interval reader, or `None` for one-shot rows."""

    def feasibility_for_codes(self, codes: IntND) -> BoolND:
        """Test actual budgets and predicates jointly at each physical liquid node."""
        solver = self.cell_solver
        binder = solver.binder
        predicates = binder.current_predicates
        if predicates is None:
            raise TypeError(
                "NB current eligibility requires canonical predicate declarations."
            )
        names = tuple(name for name, _ in binder.schedule_spec.discrete_actions)
        action = {name: codes[position] for position, name in enumerate(names)}
        pool = {**solver.kwargs, **self.cell, **action}
        resources = binder.schedule_spec.coh_of_liquid_dag
        resource_inputs = tuple(inspect.signature(resources).parameters)
        predicate_inputs = tuple(
            tuple(inspect.signature(predicate).parameters) for predicate in predicates
        )

        def one(liquid: ScalarFloat) -> BoolND:
            actual = {**pool, binder.statics.liquid_name: liquid}
            resource = resources(**{name: actual[name] for name in resource_inputs})
            eligible = resource > binder.savings_grid[0]
            for predicate, inputs in zip(predicates, predicate_inputs, strict=True):
                eligible &= predicate(**{name: actual[name] for name in inputs})
            return eligible

        return jax.vmap(one)(self.query_grid)

    @overload
    def solve(
        self,
        *,
        action_binding: Mapping[str, IntND],
        branch_cont_value: FloatND | None,
        branch_cont_marginal: FloatND | None,
        branch_extra_cont_value: FloatND | None,
        branch_cliff_savings: FloatND | None,
        source_query: Float1D | None = None,
        return_owner: Literal[False] = False,
    ) -> _NBEGMStepChannels | BoundedResult[_NBEGMStepChannels]: ...

    @overload
    def solve(
        self,
        *,
        action_binding: Mapping[str, IntND],
        branch_cont_value: FloatND | None,
        branch_cont_marginal: FloatND | None,
        branch_extra_cont_value: FloatND | None,
        branch_cliff_savings: FloatND | None,
        source_query: Float1D,
        return_owner: Literal[True],
    ) -> _OwnedNBEGMQueryStep: ...

    def solve(
        self,
        *,
        action_binding: Mapping[str, IntND],
        branch_cont_value: FloatND | None,
        branch_cont_marginal: FloatND | None,
        branch_extra_cont_value: FloatND | None,
        branch_cliff_savings: FloatND | None,
        source_query: Float1D | None = None,
        return_owner: bool = False,
    ) -> _NBEGMStepChannels | BoundedResult[_NBEGMStepChannels] | _OwnedNBEGMQueryStep:
        """Solve the cell's continuous subproblem for one discrete branch.

        `action_binding` binds the discrete action into cash-on-hand, and into the
        period utility and the discount factor where their declarations read it
        (empty when the regime carries no discrete action). `branch_cont_value` /
        `branch_cont_marginal` are this branch's continuation rows — a branch reads
        its own next-state continuation when the action feeds a co-state's law of
        motion, and identical rows when it feeds only the budget.
        """
        from _lcm.egm.nbegm_breakpoints import (  # noqa: PLC0415
            interval_midpoints,
            interval_segment_coefficients,
        )
        from _lcm.egm.nbegm_step import (  # noqa: PLC0415
            nbegm_per_interval_continuation_step_savings,
        )
        from _lcm.egm.preferences import preferences_from_utility  # noqa: PLC0415

        cell_solver = self.cell_solver
        binder = cell_solver.binder
        if return_owner and (
            source_query is None
            or binder.statics.has_jump
            or binder.statics.continuation_reads_liquid
            or binder.current_predicates is not None
            or binder.feasibility_constraints
            or binder.n_published_boundaries
            or binder.current_floor is not None
        ):
            raise UnsupportedOperationError(
                "Owned NB ride points require continuous literal-floor "
                "ordinary-child cells without extra feasibility boundaries."
            )
        statics = binder.statics
        schedule_spec = binder.schedule_spec
        kwargs = cell_solver.kwargs
        cell = self.cell
        query_grid = self.query_grid
        discount_factor_dag = schedule_spec.discount_factor_dag
        branch_discount_factor = (
            cast("ScalarFloat", kwargs["koopmans_aggregator__discount_factor"])
            if discount_factor_dag is None
            else discount_factor_dag(
                **{name: cell[name] for name in statics.discount_state_names},
                **cell_solver.discount_params,
                **{
                    name: action_binding[name] for name in statics.discount_action_names
                },
            )
        )

        coh_arg_names = frozenset(
            inspect.signature(schedule_spec.coh_of_liquid_dag).parameters
        )
        coh_of_liquid = _BoundScalarFunction(
            func=schedule_spec.coh_of_liquid_dag,
            argument_name=statics.liquid_name,
            bound={
                **{name: cell[name] for name in statics.coh_state_names},
                **cell_solver.coh_params,
                **{
                    name: value
                    for name, value in action_binding.items()
                    if name in coh_arg_names
                },
            },
        )
        utility_action_binding = {
            name: value
            for name, value in action_binding.items()
            if name in binder.utility_arg_names
        }
        utility_of_consumption = _BoundScalarFunction(
            func=schedule_spec.utility_dag,
            argument_name=statics.consumption_action_name,
            bound={
                **{name: cell[name] for name in statics.utility_state_names},
                **cell_solver.utility_params,
                **utility_action_binding,
            },
        )
        preferences = preferences_from_utility(
            utility_of_action=utility_of_consumption,
            action_lower=binder.action_lower,
            action_upper=binder.action_upper,
        )

        # Recompute the breakpoint partition with the action bound: when the
        # action enters the schedule variable, its asset preimage — and so the
        # interval partition and its midpoints — differ per branch. When the
        # action does not, the binding is dropped and this matches the shared
        # cell partition.
        branch_breakpoints, branch_jump_positions = _nbegm_cell_breakpoints(
            statics=statics,
            kwargs=kwargs,
            cell=cell,
            liquid_grid=cell_solver.liquid,
            dtype=cell_solver.dtype,
            action_binding=action_binding,
        )
        branch_midpoints = interval_midpoints(
            liquid_grid=cell_solver.liquid, breakpoints=branch_breakpoints
        )
        coh_slopes, coh_intercepts = interval_segment_coefficients(
            schedule=coh_of_liquid,
            interval_midpoints=branch_midpoints,
        )
        if statics.continuation_reads_liquid:
            # True cash-on-hand per liquid grid point keeps the step's corners
            # feasible where a partly-binding kink makes an interval's recovered
            # affine budget extrapolate below zero.
            coh_grid = jax.vmap(coh_of_liquid)(query_grid)
            branch_interval_reader = (
                None
                if self.interval_continuation is None
                else self.interval_continuation.bind(action_binding=action_binding)
            )
            return nbegm_per_interval_continuation_step_savings(
                cont_value=branch_cont_value,
                cont_marginal=branch_cont_marginal,
                liquid_grid=query_grid,
                savings_grid=binder.savings_grid,
                discount_factor=branch_discount_factor,
                preferences=preferences,
                coh_slopes=coh_slopes,
                coh_intercepts=coh_intercepts,
                breakpoints=branch_breakpoints,
                coh_grid=coh_grid,
                arithmetic=statics.envelope_arithmetic,
                extra_savings=branch_cliff_savings,
                extra_cont_value=branch_extra_cont_value,
                feasibility_partition=self.feasibility_partition,
                feasible_interval_mask=self.feasible_interval_mask,
                interval_block_reader=branch_interval_reader,
                interval_width=statics.interval_width,
                interval_indices=cast(
                    "Int1D | None", self.cell_solver.kwargs.get(_INTERVAL_COORDINATE)
                ),
                retain_query=binder.query_model is not None,
                query_status_slots=(
                    len(self.interval_continuation.query_targets)
                    if self.interval_continuation is not None
                    else 0
                ),
                collect_continuation_read_counts=statics.collect_continuation_read_counts,
            )
        if branch_cont_value is None or branch_cont_marginal is None:
            raise ValueError(
                "a smooth NB-EGM cell cannot use an interval continuation."
            )
        step = _solve_ride_along_cell_step(
            has_jump=statics.has_jump,
            jump_positions=branch_jump_positions,
            extra_savings=branch_cliff_savings,
            extra_cont_value=branch_extra_cont_value,
            cont_value=branch_cont_value,
            cont_marginal=branch_cont_marginal,
            liquid_grid=query_grid,
            savings_grid=binder.savings_grid,
            discount_factor=branch_discount_factor,
            preferences=preferences,
            coh_slopes=coh_slopes,
            coh_intercepts=coh_intercepts,
            breakpoints=branch_breakpoints,
            inverse_eis=cell_solver.inverse_eis,
            arithmetic=statics.envelope_arithmetic,
            feasibility_partition=self.feasibility_partition,
            feasible_interval_mask=self.feasible_interval_mask,
            retain_query=return_owner or binder.query_model is not None,
            source_query=source_query,
            return_owner=return_owner,
        )
        if return_owner:
            if not isinstance(step, _OwnedNBEGMQueryStep):
                raise TypeError("NB ride point lost its original native owner.")
            return step
        if isinstance(step, _OwnedNBEGMQueryStep):
            raise TypeError("Ordinary NB ride solve received an owned point record.")
        return step


# keyword-only-exempt: library-callback=_lcm.utils.dispatchers.map_over_leading_axis
def _solve_one_owned_branch(
    inputs: Mapping[str, FloatND | IntND],
    *,
    branch_solver: _NBEGMBranchSolver,
    branch_action_names: tuple[ActionName, ...],
    query: Float1D,
) -> _NBEGMRideSourcePoint:
    """Transport the original selected channels, keeping unused metadata local."""
    row = branch_solver.solve(
        action_binding={
            name: inputs["codes"][position]
            for position, name in enumerate(branch_action_names)
        },
        branch_cont_value=inputs["cont_value"],
        branch_cont_marginal=inputs["cont_marginal"],
        branch_extra_cont_value=None,
        branch_cliff_savings=None,
        source_query=query,
        return_owner=True,
    )
    return _NBEGMRideSourcePoint(
        value=row.value,
        marginal=row.marginal,
        policy=row.policy,
        savings=row.query_data.savings,
        native_owner=row.owner,
        action_codes=jnp.broadcast_to(
            inputs["codes"], (query.shape[0], len(branch_action_names))
        ),
    )


# keyword-only-exempt: library-callback=_lcm.utils.dispatchers.map_over_leading_axis
def _solve_one_branch(
    inputs: Mapping[str, FloatND | IntND],
    *,
    branch_solver: _NBEGMBranchSolver,
    branch_action_names: tuple[ActionName, ...],
) -> _NBEGMStepChannels | BoundedResult[_NBEGMStepChannels]:
    """Solve one branch of the mapped branch pytree."""
    binding = {
        name: inputs["codes"][position]
        for position, name in enumerate(branch_action_names)
    }
    step = branch_solver.solve(
        action_binding=binding,
        branch_cont_value=inputs.get("cont_value"),
        branch_cont_marginal=inputs.get("cont_marginal"),
        branch_extra_cont_value=inputs.get("extra_cont_value"),
        branch_cliff_savings=inputs.get("cliff_savings"),
    )
    if isinstance(step, BoundedResult):
        return step
    if branch_solver.cell_solver.binder.query_model is not None:
        return step
    return step[0], step[1], step[2]


def _package_nbegm_cell_result(
    *,
    value_row: FloatND,
    marginal_row: FloatND,
    policy_row: FloatND,
    branch_stacks: tuple[FloatND, FloatND] | None,
    n_published_boundaries: int,
    unsort: IntND,
    n_liquid: int,
    endog_row: FloatND,
    published_boundaries: FloatND,
) -> tuple[FloatND, ...]:
    """Package a cell's collapsed rows and optional conditional branch banks."""
    if n_published_boundaries == 0:
        if branch_stacks is None:
            return (value_row, marginal_row, policy_row)
        value_stack, policy_stack = branch_stacks
        return value_row, marginal_row, policy_row, value_stack, policy_stack

    # The carry keeps the augmented row, while published values use the original
    # liquid nodes recovered through the sort permutation.
    value_at_liquid = value_row[unsort][:n_liquid]
    policy_at_liquid = policy_row[unsort][:n_liquid]
    collapsed = (
        value_at_liquid,
        endog_row,
        value_row,
        marginal_row,
        policy_at_liquid,
        published_boundaries,
    )
    if branch_stacks is None:
        return collapsed
    value_stack, policy_stack = branch_stacks
    branch_value_at_liquid = value_stack[:, unsort][:, :n_liquid]
    branch_policy_at_liquid = policy_stack[:, unsort][:, :n_liquid]
    return (*collapsed, branch_value_at_liquid, branch_policy_at_liquid)


@dataclass(frozen=True)
class _NBEGMDiscreteSpec:
    """Build-time statics for a discrete-action regime with a smooth budget.

    The discrete action shifts cash-on-hand; the continuous consumption/savings
    subproblem is solved per discrete-action value by NBEGM and the discrete choice
    is taken by the upper envelope over the branch values.
    """

    coh_of_liquid_dag: Callable[..., FloatND]
    """Composed `coh` as a function of the liquid state, the discrete actions, and
    qualified params."""
    coh_param_names: tuple[QualifiedName, ...]
    """Qualified parameter names `coh` reads (excluding the liquid state and the
    discrete actions)."""
    coh_action_names: tuple[ActionName, ...]
    """Discrete action names read by the composed resource function."""
    liquid_state_name: StateName
    """Name of the liquid state the budget varies in."""
    discrete_actions: DiscreteActionCodes
    """Each discrete action enveloped over, paired with its grid codes."""
    param_checks: tuple[ParamCheck, ...] = ()
    """Parameter-dependent preconditions for every discrete budget branch."""

    @property
    def branch_bindings(self) -> tuple[MappingProxyType[ActionName, int], ...]:
        """One binding of every declared discrete action, per envelope branch."""
        return _branch_bindings(self.discrete_actions)


def _collect_nbegm_discrete_spec(
    *,
    context: SolverBuildContext,
    budget_target: FunctionName,
    post_decision_function: FunctionName,
    continuous_state: StateName,
    probe_failure: Literal["reject", "assume_declared"],
    probe_schedule: Literal["first_solve", "every_solve", "never"] = "never",
) -> _NBEGMDiscreteSpec:
    """Collect the discrete actions of a smooth regime and their grid codes.

    `continuous_state` names the Euler (liquid) axis explicitly; without it
    the single-state inference `state_names[0]` applies — wrong for a regime
    with ride-along states (discrete states order first), so a multi-state
    config must pass its declared axis.
    """
    import inspect  # noqa: PLC0415

    space = context.state_action_space
    discrete_actions = _discrete_actions_of(space=space)
    action_names = frozenset(name for name, _ in discrete_actions)
    liquid_state_name = _single_liquid_state_name(
        context=context, declared=continuous_state, path="discrete-envelope path"
    )
    for action_name in sorted(action_names):
        _fail_if_discrete_action_feeds_continuation(
            context=context,
            action_name=action_name,
            liquid_state_name=liquid_state_name,
            budget_target=budget_target,
            post_decision_function=post_decision_function,
        )
    coh_dag = concatenate_functions(dict(context.functions), targets=budget_target)
    coh_args = tuple(inspect.signature(coh_dag).parameters)
    coh_param_names = tuple(
        name
        for name in coh_args
        if name != liquid_state_name and name not in action_names
    )
    affinity_check = _budget_affinity_check(
        context=context,
        coh_dag=coh_dag,
        liquid_state_name=liquid_state_name,
        require_unit_slope=False,
        probe_failure=probe_failure,
        probe_schedule=probe_schedule,
    )
    return _NBEGMDiscreteSpec(
        coh_of_liquid_dag=coh_dag,
        coh_param_names=coh_param_names,
        coh_action_names=tuple(name for name in coh_args if name in action_names),
        liquid_state_name=liquid_state_name,
        discrete_actions=discrete_actions,
        param_checks=(affinity_check,),
    )


def _collect_nbegm_schedule_discrete_spec(
    *,
    context: SolverBuildContext,
    budget_target: FunctionName,
    continuous_state: StateName,
    post_decision_function: FunctionName,
    probe_failure: Literal["reject", "assume_declared"],
    probe_schedule: Literal["first_solve", "every_solve", "never"] = "never",
) -> _NBEGMScheduleDiscreteSpec:
    """Collect the discrete actions layered over a single-liquid cliff schedule."""
    import inspect  # noqa: PLC0415

    from _lcm.egm.nbegm import collect_nbegm_metadata  # noqa: PLC0415

    space = context.state_action_space
    discrete_actions = _discrete_actions_of(space=space)
    action_names = frozenset(name for name, _ in discrete_actions)

    liquid_state_name = _single_liquid_state_name(
        context=context, declared=continuous_state, path="schedule+discrete path"
    )

    for action_name in sorted(action_names):
        _fail_if_discrete_action_feeds_continuation(
            context=context,
            action_name=action_name,
            liquid_state_name=liquid_state_name,
            budget_target=budget_target,
            post_decision_function=post_decision_function,
        )
    user_functions = {
        name: func for name, func in context.functions.items() if callable(func)
    }
    registry = collect_nbegm_metadata(functions=user_functions)
    schedules = registry.piecewise_affine_schedules
    if any(schedule.variable != liquid_state_name for schedule in schedules):
        msg = (
            "NBEGM schedule+discrete path handles schedules on the liquid state "
            "only; a derived-variable schedule needs the ride-along path."
        )
        raise RegimeInitializationError(msg)
    _fail_if_single_liquid_schedules_unsupported(
        schedules=schedules,
        ride_along_state_names=(),
        regime_name=context.regime_name,
    )

    coh_dag = concatenate_functions(dict(context.functions), targets=budget_target)
    coh_args = tuple(inspect.signature(coh_dag).parameters)
    coh_param_names = tuple(
        name
        for name in coh_args
        if name != liquid_state_name and name not in action_names
    )
    first = schedules[0]
    threshold_param_names = tuple(
        f"{first.output}__{bp.threshold}" for bp in first.breakpoints
    )
    breakpoint_kinds = tuple(bp.kind for bp in first.breakpoints)
    breakpoint_sources = tuple(
        _NBEGMSource(
            variable=liquid_state_name,
            threshold_param_name=f"{first.output}__{bp.threshold}",
            kind=bp.kind,
            equality_owner=bp.equality_owner,
            derived_of_liquid_dag=None,
            derived_param_names=(),
            threshold_index_state=bp.indexed_by,
            threshold_static_index=bp.static_index,
            threshold_subkey=bp.threshold_subkey,
        )
        for bp in first.breakpoints
    )
    affinity_check = _budget_affinity_check(
        context=context,
        coh_dag=coh_dag,
        liquid_state_name=liquid_state_name,
        breakpoint_sources=breakpoint_sources,
        require_unit_slope=(
            bool(breakpoint_kinds) and all(kind == "jump" for kind in breakpoint_kinds)
        ),
        probe_failure=probe_failure,
        probe_schedule=probe_schedule,
    )
    return _NBEGMScheduleDiscreteSpec(
        coh_of_liquid_action_dag=coh_dag,
        coh_param_names=coh_param_names,
        coh_action_names=tuple(name for name in coh_args if name in action_names),
        liquid_state_name=liquid_state_name,
        discrete_actions=discrete_actions,
        threshold_param_names=threshold_param_names,
        breakpoint_kinds=breakpoint_kinds,
        param_checks=(affinity_check,),
    )


def _discrete_envelope_over_branches(
    *,
    value_stack: FloatND,
    marginal_stack: FloatND,
    taste_shock_scale: float,
) -> tuple[Float1D, Float1D]:
    """Take the discrete choice by the upper envelope over branch solves.

    `value_stack` and `marginal_stack` are `(n_branches, n_liquid)` — one solved
    branch per discrete-action value. Returns the enveloped value and marginal on
    the liquid grid:

    - Hard maximum (`taste_shock_scale == 0`): `max` over branches, with the
      winning branch's marginal by Danskin's theorem. At a value tie the envelope
      has a kink and the derivative is a subgradient set; the `argmax` convention
      selects the lowest-index tied branch's marginal — a well-defined subgradient,
      not the true (set-valued) derivative.
    - EV1 taste shocks (`taste_shock_scale > 0`): the scaled logsum value and the
      choice-probability-weighted branch marginal.
    """
    if taste_shock_scale == 0.0:
        modal = jnp.argmax(value_stack, axis=0)
        index = jnp.arange(value_stack.shape[1])
        return value_stack[modal, index], marginal_stack[modal, index]
    scaled = value_stack / taste_shock_scale
    probabilities = jax.nn.softmax(scaled, axis=0)
    value = taste_shock_scale * jax.scipy.special.logsumexp(scaled, axis=0)
    marginal = jnp.sum(probabilities * marginal_stack, axis=0)
    return value, marginal


def _build_nbegm_schedule_discrete_core(
    *,
    savings_grid: Float1D,
    functions: EconFunctionsMapping,
    consumption_action: ActionName,
    spec: _NBEGMScheduleDiscreteSpec,
    taste_shock_scale: float,
    envelope_arithmetic: ComparisonArithmetic = "ordinary",
) -> _NBEGMScheduleDiscreteCore:
    """Build the discrete-envelope core over a cliffed single-liquid budget.

    Per discrete-action value the core recovers the schedule's per-interval affine
    cash-on-hand and the liquid breakpoints and solves that branch with the
    kind-appropriate step (reading the continuation jump-aware, so the solve is
    exact through recurring jumps). The discrete choice is then taken by the upper
    envelope over the branch values — the hard maximum, or the EV1 logsum under a
    taste-shock scale.
    """
    from _lcm.egm.preferences import (  # noqa: PLC0415
        NEWTON_ACTION_FLOOR,
        get_preferences_builder,
        newton_action_ceiling,
    )

    return _NBEGMScheduleDiscreteCore(
        savings_grid=savings_grid,
        build_preferences=get_preferences_builder(
            functions=functions,
            action_name=consumption_action,
            action_lower=NEWTON_ACTION_FLOOR,
            action_upper=newton_action_ceiling(savings_grid),
        ),
        spec=spec,
        taste_shock_scale=taste_shock_scale,
        envelope_arithmetic=envelope_arithmetic,
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class _NBEGMScheduleDiscreteCore:
    """The discrete-envelope core over a cliffed single-liquid budget.

    Per discrete-action value it recovers the schedule's per-interval affine
    cash-on-hand and the liquid breakpoints and solves that branch with the
    kind-appropriate step; the upper envelope over the branch values then takes
    the discrete choice.
    """

    savings_grid: Float1D
    """Exogenous post-decision savings grid the step inverts on."""

    build_preferences: Callable[[EconFunctionKwargs], Preferences]
    """Bind the regime's flat params and a branch's codes into its preference maps."""

    spec: _NBEGMScheduleDiscreteSpec
    """The composed budget, its thresholds, and the discrete branches."""

    taste_shock_scale: float
    """EV1 taste-shock scale of the discrete choice; `0.0` is the hard maximum."""

    envelope_arithmetic: ComparisonArithmetic
    """Arithmetic the upper envelope merges the intervals with."""

    retain_query: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Keep independently declared conditional original-saving rows."""

    interval_recipe: NBEGMIntervalRecipe | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Canonical physical selection cuts for this target-period source recipe."""

    def __call__(
        self,
        *,
        liquid: Float1D,
        next_liquid_grid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        next_liquid: Float1D,
        marginal_return: Float1D,
        effective_savings_grid: Float1D | None = None,
        source_bound_status: BoundInputStatus | None = None,
        continuation_read: Callable[[FloatND, BoolND], tuple[FloatND, FloatND]]
        | None = None,
        **params: ValueND,
    ) -> _RawNBEGMOutput:
        """Solve one period of the regime, branch by branch, on the liquid grid."""
        from _lcm.egm.nbegm_breakpoints import (  # noqa: PLC0415
            interval_midpoints,
            interval_segment_coefficients,
        )

        spec = self.spec
        is_single_jump, is_multi_jump, is_mixed, jump_mask, flat_mask = (
            _schedule_kind_flags(spec.breakpoint_kinds)
        )
        order_sensitive = len(set(spec.breakpoint_kinds)) > 1
        coh_params = {name: params[name] for name in spec.coh_param_names}
        breakpoints = _sorted_thresholds(
            raw=jnp.stack([params[name] for name in spec.threshold_param_names]),
            order_sensitive=order_sensitive,
        )
        if self.interval_recipe is not None:
            cuts, _orientation = self.interval_recipe.evaluate_preimages(params)
            breakpoints = _sorted_thresholds(raw=cuts, order_sensitive=order_sensitive)
        midpoints = interval_midpoints(liquid_grid=liquid, breakpoints=breakpoints)

        values: list[Float1D] = []
        marginals: list[Float1D] = []
        query_rows: list[EGMQueryData] = []
        for binding in spec.branch_bindings:
            preferences = self.build_preferences({**params, **binding})
            coh_of_liquid = _BoundScalarFunction(
                func=spec.coh_of_liquid_action_dag,
                argument_name=spec.liquid_state_name,
                bound={
                    **{
                        name: jnp.asarray(binding[name])
                        for name in spec.coh_action_names
                    },
                    **coh_params,
                },
            )
            coh_slopes, coh_intercepts = interval_segment_coefficients(
                schedule=coh_of_liquid,
                interval_midpoints=midpoints,
            )
            step = _solve_cliffed_budget(
                next_value=next_value,
                next_marginal=next_marginal,
                liquid=liquid,
                next_liquid_grid=next_liquid_grid,
                savings_grid=self.savings_grid
                if effective_savings_grid is None
                else effective_savings_grid,
                discount_factor=params["koopmans_aggregator__discount_factor"],
                preferences=preferences,
                next_liquid=next_liquid,
                marginal_return=marginal_return,
                coh_slopes=coh_slopes,
                coh_intercepts=coh_intercepts,
                breakpoints=breakpoints,
                is_single_jump=is_single_jump,
                is_multi_jump=is_multi_jump,
                is_mixed=is_mixed,
                jump_mask=jump_mask,
                flat_mask=flat_mask,
                arithmetic=self.envelope_arithmetic,
                retain_query=self.retain_query,
                continuation_read=continuation_read,
            )
            branch_value, branch_marginal, _policy = step[:3]
            if self.retain_query:
                query_rows.append(_required_nbegm_query_data(step=step))
            values.append(branch_value)
            marginals.append(branch_marginal)

        value, marginal = _discrete_envelope_over_branches(
            value_stack=jnp.stack(values),
            marginal_stack=jnp.stack(marginals),
            taste_shock_scale=self.taste_shock_scale,
        )
        carry = EGMCarry(
            endog_grid=liquid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(self.taste_shock_scale, dtype=value.dtype),
        )
        if self.retain_query:
            data = jax.tree.map(lambda *leaves: jnp.stack(leaves), *query_rows)
            data = replace(data, floor_used=data.floor_used[0])
            carry = _scalar_conditional_nbegm_carry(
                value=jnp.stack(values),
                marginal=jnp.stack(marginals),
                data=data,
                grid=liquid,
                action_shape=tuple(
                    len(codes) for _name, codes in spec.discrete_actions
                ),
                breakpoints=None,
            )
        if source_bound_status is not None:
            return BoundedResult(
                output=(value, carry), bound_status=(source_bound_status,)
            )
        return value, carry


def _build_nbegm_discrete_core(
    *,
    savings_grid: Float1D,
    functions: EconFunctionsMapping,
    consumption_action: ActionName,
    discrete_spec: _NBEGMDiscreteSpec,
    taste_shock_scale: float,
    envelope_arithmetic: ComparisonArithmetic = "ordinary",
) -> _NBEGMDiscreteCore:
    """Build the jittable discrete-envelope core for one continuation target.

    Per discrete-action value the core recovers the smooth budget's affine cash-on-
    hand and solves the continuous subproblem with the multi-interval step, then
    takes the discrete choice by the upper envelope (`nbegm_discrete_envelope_step`).
    """
    from _lcm.egm.preferences import (  # noqa: PLC0415
        NEWTON_ACTION_FLOOR,
        get_preferences_builder,
        newton_action_ceiling,
    )

    return _NBEGMDiscreteCore(
        savings_grid=savings_grid,
        build_preferences=get_preferences_builder(
            functions=functions,
            action_name=consumption_action,
            action_lower=NEWTON_ACTION_FLOOR,
            action_upper=newton_action_ceiling(savings_grid),
        ),
        discrete_spec=discrete_spec,
        taste_shock_scale=taste_shock_scale,
        envelope_arithmetic=envelope_arithmetic,
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class _NBEGMDiscreteCore:
    """The jittable discrete-envelope core over a smooth single-liquid budget.

    Per discrete-action value it recovers the budget's affine cash-on-hand and
    solves the continuous subproblem with the multi-interval step; the upper
    envelope then takes the discrete choice.
    """

    savings_grid: Float1D
    """Exogenous post-decision savings grid the step inverts on."""

    build_preferences: Callable[[EconFunctionKwargs], Preferences]
    """Bind the regime's flat params and a branch's codes into its preference maps."""

    discrete_spec: _NBEGMDiscreteSpec
    """The composed budget and the discrete branches."""

    taste_shock_scale: float
    """EV1 taste-shock scale of the discrete choice; `0.0` is the hard maximum."""

    envelope_arithmetic: ComparisonArithmetic
    """Arithmetic the upper envelope merges the branches with."""

    retain_query: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Keep independently declared conditional original-saving rows."""

    def __call__(
        self,
        *,
        liquid: Float1D,
        next_liquid_grid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        next_liquid: Float1D,
        marginal_return: Float1D,
        effective_savings_grid: Float1D | None = None,
        source_bound_status: BoundInputStatus | None = None,
        continuation_read: Callable[[FloatND, BoolND], tuple[FloatND, FloatND]]
        | None = None,
        **params: ValueND,
    ) -> _RawNBEGMOutput:
        """Solve one period of the regime, branch by branch, on the liquid grid."""
        from _lcm.egm.nbegm_breakpoints import affine_coefficients  # noqa: PLC0415
        from _lcm.egm.nbegm_step import (  # noqa: PLC0415
            nbegm_discrete_envelope_step,
        )

        discrete_spec = self.discrete_spec
        coh_params = {name: params[name] for name in discrete_spec.coh_param_names}
        branch_preferences = tuple(
            self.build_preferences({**params, **binding})
            for binding in discrete_spec.branch_bindings
        )
        empty_breakpoints = jnp.zeros((0,), dtype=liquid.dtype)
        choices: list[dict[str, Float1D]] = []
        for binding in discrete_spec.branch_bindings:
            coh_of_liquid = _BoundScalarFunction(
                func=discrete_spec.coh_of_liquid_dag,
                argument_name=discrete_spec.liquid_state_name,
                bound={
                    **{
                        name: jnp.asarray(binding[name])
                        for name in discrete_spec.coh_action_names
                    },
                    **coh_params,
                },
            )
            slope, intercept = affine_coefficients(coh_of_liquid)
            choices.append(
                {
                    "coh_slopes": jnp.reshape(slope, (1,)),
                    "coh_intercepts": jnp.reshape(intercept, (1,)),
                    "breakpoints": empty_breakpoints,
                }
            )
        step = nbegm_discrete_envelope_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            next_liquid_grid=next_liquid_grid,
            savings_grid=self.savings_grid
            if effective_savings_grid is None
            else effective_savings_grid,
            discount_factor=params["koopmans_aggregator__discount_factor"],
            preferences=branch_preferences,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            choices=tuple(choices),
            taste_shock_scale=self.taste_shock_scale,
            arithmetic=self.envelope_arithmetic,
            retain_query=self.retain_query,
            continuation_read=continuation_read,
        )
        value, marginal, _policy, _choice = step[:4]
        carry = EGMCarry(
            endog_grid=liquid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(self.taste_shock_scale, dtype=value.dtype),
        )
        if self.retain_query:
            retained = _required_nbegm_discrete_step(step=step)
            carry = _scalar_conditional_nbegm_carry(
                value=retained.conditional_value,
                marginal=retained.conditional_marginal,
                data=retained.query_data,
                grid=liquid,
                action_shape=tuple(
                    len(codes) for _name, codes in discrete_spec.discrete_actions
                ),
                breakpoints=None,
            )
        if source_bound_status is not None:
            return BoundedResult(
                output=(value, carry), bound_status=(source_bound_status,)
            )
        return value, carry


def _build_nbegm_owned_single_axis_group(
    *,
    context: SolverBuildContext,
    solver: _BoundNBEGM,
    route: Literal["schedule_discrete", "schedule", "discrete", "case"],
    target: RegimeName,
    liquid_state_name: StateName,
    variable_names: frozenset[StateOrActionName],
    savings_grid: Float1D,
    feasibility_constraints: tuple[_NBEGMFeasibilityConstraint, ...],
    current_model: NBEGMQueryModel | None,
    plan: ContinuationPlan | None,
    has_child_slots: bool,
    period: int,
) -> tuple[OneRowCore, Callable[..., tuple[Float1D, Float1D]], tuple[ParamCheck, ...]]:
    """Attach source eligibility only to the actual group with query ownership."""
    core, law, checks = _build_nbegm_single_axis_group(
        context=context,
        solver=solver,
        route=route,
        target=target,
        liquid_state_name=liquid_state_name,
        variable_names=variable_names,
        savings_grid=savings_grid,
        feasibility_constraints=feasibility_constraints,
    )
    if current_model is not None or has_child_slots:
        if plan is None:
            raise TypeError(
                "An NB feature producer requires its canonical continuation plan."
            )
        core = replace(core, retain_query=current_model is not None)
        if current_model is not None and isinstance(
            core, (_NBEGMContinuousCore, _NBEGMScheduleDiscreteCore)
        ):
            core = replace(
                core,
                interval_recipe=current_model.intervals_by_period[period],
            )
        actions = _discrete_actions_of(space=context.state_action_space)
        core = _NBEGMOwnedSingleAxisCore(
            inner=core,
            continuation_plan=plan,
            liquid_name=liquid_state_name,
            resources=concatenate_functions(
                dict(context.functions), targets=solver.budget_target
            ),
            predicates=_nbegm_current_predicates(
                context=context, bound=solver.post_decision_bound
            ),
            action_names=tuple(name for name, _codes in actions),
            action_bindings=_branch_bindings(actions),
            owns_query_rows=current_model is not None,
        )
    return core, law, checks


def _build_nbegm_single_axis_group(
    *,
    context: SolverBuildContext,
    solver: _BoundNBEGM,
    route: Literal["schedule_discrete", "schedule", "discrete", "case"],
    target: RegimeName,
    liquid_state_name: StateName,
    variable_names: frozenset[StateOrActionName],
    savings_grid: Float1D,
    feasibility_constraints: tuple[_NBEGMFeasibilityConstraint, ...],
) -> tuple[
    _NBEGMSingleAxisCore,
    Callable[..., tuple[Float1D, Float1D]],
    tuple[ParamCheck, ...],
]:
    """Build one periodized single-axis core and its declared liquid law."""
    resolved_feasibility_constraints = tuple(
        replace(
            constraint,
            predicate=context.constraint_functions[constraint.program.constraint_name],
        )
        for constraint in feasibility_constraints
    )
    law = build_declared_liquid_law(
        transitions=context.transitions,
        functions=context.functions,
        post_decision_name=solver.post_decision_function,
        target=target,
        target_state=liquid_state_name,
        variable_names=variable_names,
    )
    if route == "schedule_discrete":
        spec = _collect_nbegm_schedule_discrete_spec(
            context=context,
            budget_target=solver.budget_target,
            continuous_state=solver.continuous_state,
            post_decision_function=solver.post_decision_function,
            probe_failure=solver.probe_failure,
            probe_schedule=solver.probe_schedule,
        )
        core = _build_nbegm_schedule_discrete_core(
            savings_grid=savings_grid,
            functions=context.functions,
            consumption_action=solver.continuous_action,
            spec=spec,
            taste_shock_scale=0.0,
            envelope_arithmetic=solver.envelope_arithmetic,
        )
        checks = spec.param_checks
    elif route == "schedule":
        spec = _collect_nbegm_schedule_spec(
            context=context,
            budget_target=solver.budget_target,
            continuous_state=solver.continuous_state,
            consumption_action_name=solver.continuous_action,
            probe_failure=solver.probe_failure,
            probe_schedule=solver.probe_schedule,
        )
        core = _build_nbegm_continuous_core(
            savings_grid=savings_grid,
            functions=context.functions,
            consumption_action=solver.continuous_action,
            schedule_spec=spec,
            feasibility_constraints=resolved_feasibility_constraints,
            envelope_arithmetic=solver.envelope_arithmetic,
        )
        checks = spec.param_checks
    elif route == "discrete":
        spec = _collect_nbegm_discrete_spec(
            context=context,
            budget_target=solver.budget_target,
            post_decision_function=solver.post_decision_function,
            continuous_state=solver.continuous_state,
            probe_failure=solver.probe_failure,
            probe_schedule=solver.probe_schedule,
        )
        core = _build_nbegm_discrete_core(
            savings_grid=savings_grid,
            functions=context.functions,
            consumption_action=solver.continuous_action,
            discrete_spec=spec,
            taste_shock_scale=0.0,
            envelope_arithmetic=solver.envelope_arithmetic,
        )
        checks = spec.param_checks
    else:
        spec = _collect_nbegm_case_spec(
            context=context,
            continuous_state=solver.continuous_state,
        )
        core = _build_nbegm_core(
            savings_grid=savings_grid,
            functions=context.functions,
            consumption_action=solver.continuous_action,
            case_spec=spec,
            envelope_arithmetic=solver.envelope_arithmetic,
        )
        checks = ()
    return core, law, checks


def _assemble_ride_carry(
    *,
    stacks: tuple[FloatND, ...],
    n_jumps: int,
    n_action_branches: int,
    liquid: Float1D,
    ride_shape: tuple[int, ...],
    liquid_axis_pos: int,
    dtype: DTypeLike,
) -> _NBEGMRideReplayOutput:
    """Reshape per-cell value, carry, policy, and optional branch stacks.

    - With jump breakpoints, the cell solve returns the value at the liquid
      nodes plus the augmented carry rows (duplicated jump abscissae with
      one-sided limits) and the jump locations.
    - Without jumps, it returns plain liquid-grid rows and the carry sits on
      the shared broadcast grid.
    - A discrete inner envelope additionally returns every conditional branch's
      value and inner action on the original state grid. Those banks are absent
      for the smooth path, preserving its output/storage contract.

    The published value array follows the productmap state order, so the
    liquid axis moves from the working layout's trailing position to its
    canonical index. The carry keeps the working layout (ride axes leading
    the row axis): it is read back only by `bind_continuation`, which
    produced it, so the round-trip stays self-consistent.
    """
    n_liquid = liquid.shape[0]
    has_discrete_branches = n_action_branches > 0
    carry_policy = not has_discrete_branches and n_jumps == 0
    if n_jumps:
        if has_discrete_branches:
            (
                value_stack,
                endog_stack,
                row_value_stack,
                row_marginal_stack,
                policy_stack,
                breakpoint_stack,
                branch_value_stack,
                branch_policy_stack,
            ) = stacks
        else:
            (
                value_stack,
                endog_stack,
                row_value_stack,
                row_marginal_stack,
                policy_stack,
                breakpoint_stack,
            ) = stacks
        n_row = n_liquid + 2 * n_jumps
        carry_rows = (
            endog_stack.reshape(*ride_shape, n_row).astype(dtype),
            row_value_stack.reshape(*ride_shape, n_row).astype(dtype),
            row_marginal_stack.reshape(*ride_shape, n_row).astype(dtype),
        )
        breakpoint_rows = breakpoint_stack.reshape(*ride_shape, n_jumps).astype(dtype)
        policy_rows = None
    elif carry_policy:
        # Continuous-only, jump-free rows: the cell solve returns the exact
        # consumption alongside value and marginal.
        value_stack, marginal_stack, policy_stack = stacks
        carry_rows = (
            jnp.broadcast_to(liquid, (*ride_shape, n_liquid)).astype(dtype),
            value_stack.reshape(*ride_shape, n_liquid).astype(dtype),
            marginal_stack.reshape(*ride_shape, n_liquid).astype(dtype),
        )
        breakpoint_rows = None
        policy_rows = policy_stack.reshape(*ride_shape, n_liquid).astype(dtype)
    else:
        if has_discrete_branches:
            (
                value_stack,
                marginal_stack,
                policy_stack,
                branch_value_stack,
                branch_policy_stack,
            ) = stacks
        else:
            value_stack, marginal_stack, policy_stack = stacks
        carry_rows = (
            jnp.broadcast_to(liquid, (*ride_shape, n_liquid)).astype(dtype),
            value_stack.reshape(*ride_shape, n_liquid).astype(dtype),
            marginal_stack.reshape(*ride_shape, n_liquid).astype(dtype),
        )
        breakpoint_rows = None
        policy_rows = None
    value_arr = jnp.moveaxis(
        value_stack.reshape(*ride_shape, n_liquid), -1, liquid_axis_pos
    )
    policy_arr = jnp.moveaxis(
        policy_stack.reshape(*ride_shape, n_liquid), -1, liquid_axis_pos
    )
    carry = EGMCarry(
        endog_grid=carry_rows[0],
        value=carry_rows[1],
        marginal_utility=carry_rows[2],
        taste_shock_scale=jnp.asarray(0.0, dtype=dtype),
        breakpoints=breakpoint_rows,
        policy=policy_rows,
    )
    if not has_discrete_branches:
        return value_arr, carry, policy_arr

    return (
        value_arr,
        carry,
        policy_arr,
        _assemble_branch_bank(
            stack=branch_value_stack,
            ride_shape=ride_shape,
            n_action_branches=n_action_branches,
            n_liquid=n_liquid,
            liquid_axis_pos=liquid_axis_pos,
        ),
        _assemble_branch_bank(
            stack=branch_policy_stack,
            ride_shape=ride_shape,
            n_action_branches=n_action_branches,
            n_liquid=n_liquid,
            liquid_axis_pos=liquid_axis_pos,
        ),
    )


def _assemble_branch_bank(
    *,
    stack: FloatND,
    ride_shape: tuple[int, ...],
    n_action_branches: int,
    n_liquid: int,
    liquid_axis_pos: int,
) -> FloatND:
    """Lay a per-cell branch stack out as a branch-leading canonical array."""
    working = stack.reshape(*ride_shape, n_action_branches, n_liquid)
    branch_leading = jnp.moveaxis(working, -2, 0)
    return jnp.moveaxis(branch_leading, -1, liquid_axis_pos + 1)


def _augment_liquid_with_jump_sides(
    *, liquid_grid: Float1D, jumps: Float1D
) -> tuple[Float1D, Float1D, IntND]:
    """Insert a query point one float step inside each side of every jump.

    Returns the sorted augmented query grid, the matching published
    abscissae — the same order with each side point relabeled to its exact
    jump location, so the row carries the jump as a duplicated abscissa —
    and the permutation mapping sorted positions back to concatenation
    order (liquid nodes first, then left-side points, then right-side
    points).
    """
    evaluation_points = jnp.concatenate(
        [
            liquid_grid,
            jnp.nextafter(jumps, -jnp.inf),
            jnp.nextafter(jumps, jnp.inf),
        ]
    )
    published_abscissae = jnp.concatenate([liquid_grid, jumps, jumps])
    sort_order = jnp.argsort(evaluation_points)
    return (
        evaluation_points[sort_order],
        published_abscissae[sort_order],
        # int32 permutation: the augmented grid has at most a few hundred
        # entries.
        jnp.argsort(sort_order).astype(jnp.int32),
    )


def _shard_ride_carry_template(
    *,
    template: EGMCarry,
    grids: Mapping[StateOrActionName, Grid],
    sharded_state_names: frozenset[StateName],
    ride_along_state_names: tuple[StateName, ...],
    submesh_device_ids: tuple[int, ...],
) -> EGMCarry:
    """Shard the ride-along carry template over its distributed ride axes."""
    return shard_carry_template(
        template=template,
        grids=grids,
        sharded_state_names=sharded_state_names,
        leading_axis_names=ride_along_state_names,
        devices=placed_devices_for_ids(submesh_device_ids=submesh_device_ids),
    )


def _build_ride_along_carry_template(
    *,
    liquid_grid: Float1D,
    ride_shape: tuple[int, ...],
    n_breakpoints: int,
    carry_policy: bool,
    materializer: DatedGenerationMaterializer | None = None,
) -> EGMCarry:
    """Build the all-finite case-piece carry template with ride-along axes leading.

    Each ride-along cell publishes one liquid-grid carry row; the template carries
    the ride-along (discrete/passive) axes ahead of the liquid axis, matching the
    canonical value-function layout the continuation reader interpolates.
    """
    # Same pytree as the runtime carry: a regime with jump breakpoints holds
    # each jump inside its rows as a duplicated abscissa (two extra row slots
    # per jump) and publishes the jump locations (kink breakpoints leave the
    # value continuous and add no row slots), so the lowering template shares
    # both fixed shapes. Repeating the top node keeps the template rows
    # weakly ascending and all-finite. `carry_policy` must match
    # `_assemble_ride_carry`'s predicate (continuous-only, jump-free): that path
    # returns a `policy` array leaf, so the template must carry the same leaf or
    # a standalone ride-along NBEGM continuation (rolled cross-period, lowered
    # against this template) would have a different pytree than the runtime
    # carry. The NNBEGM collapse strips its own runtime leaf, so its
    # outer continuation stays policy-free against its own template.
    if materializer is not None:
        return materializer.evaluate_validation(
            function=functools.partial(
                _build_ride_along_carry_template,
                ride_shape=ride_shape,
                n_breakpoints=n_breakpoints,
                carry_policy=carry_policy,
            ),
            arguments={"liquid_grid": liquid_grid},
        )
    row = jnp.concatenate(
        [liquid_grid, jnp.repeat(liquid_grid[-1:], 2 * n_breakpoints)]
    )
    block = jnp.broadcast_to(row, (*ride_shape, row.shape[0]))
    return EGMCarry(
        endog_grid=block,
        value=jnp.zeros_like(block),
        marginal_utility=jnp.zeros_like(block),
        taste_shock_scale=jnp.asarray(0.0, dtype=liquid_grid.dtype),
        breakpoints=(
            jnp.zeros((*ride_shape, n_breakpoints), dtype=liquid_grid.dtype)
            if n_breakpoints
            else None
        ),
        policy=jnp.zeros_like(block) if carry_policy else None,
    )


def _aggregates_nonlinearly(
    certainty_equivalent: CertaintyEquivalent | None,
) -> bool:
    """Whether the regime aggregates its continuation nonlinearly.

    Every non-terminal regime carries a certainty equivalent, so presence
    alone does not distinguish the Epstein-Zin kernels from the
    expected-utility ones: `LinearExpectation` is the expected-utility
    default and takes the ordinary route.
    """
    return certainty_equivalent is not None and not isinstance(
        certainty_equivalent, LinearExpectation
    )


@dataclass(frozen=True, kw_only=True)
class _NBEGMQueryIntervalEvaluators:
    """Canonical source-side comparisons beside the producer's selection preimages."""

    sources: tuple[_NBEGMSource, ...]
    """The independently processed source declarations, in canonical order."""
    liquid_name: StateName
    """Native liquid argument of the selected interval-held representation."""

    def preimages(self, pool: EconFunctionKwargs) -> tuple[Float1D, BoolND]:
        """Reuse exact producer preimage/equality arithmetic in declaration order."""
        dtype = canonical_float_dtype()
        preimages = []
        increasing = []
        for source in self.sources:
            preimages.append(
                _cell_breakpoint(
                    source=source,
                    liquid_name=self.liquid_name,
                    kwargs=pool,
                    cell=pool,
                    dtype=dtype,
                    action_binding=pool,
                )
            )
            if source.derived_of_liquid_dag is None:
                increasing.append(jnp.asarray(a=True))
            else:
                function = _BoundScalarFunction(
                    func=source.derived_of_liquid_dag,
                    argument_name=self.liquid_name,
                    bound={
                        name: pool[name]
                        for name in inspect.signature(
                            source.derived_of_liquid_dag
                        ).parameters
                        if name != self.liquid_name
                    },
                )
                increasing.append(
                    jax.grad(function)(jnp.asarray(0.0, dtype=dtype)) > 0.0
                )
        if not self.sources:
            return jnp.zeros((0,), dtype=dtype), jnp.zeros((0,), dtype=jnp.bool_)
        return jnp.stack(preimages), jnp.stack(increasing)

    def sides(self, pool: EconFunctionKwargs) -> BoolND:
        """Compare actual source variables with their declared equality owners."""
        sides = []
        for source in self.sources:
            threshold = _indexed_threshold_value(
                table=pool[source.threshold_param_name],
                subkey=source.threshold_subkey,
                index_state=source.threshold_index_state,
                static_index=source.threshold_static_index,
                cell=pool,
            )
            threshold = _numeric_nbegm_argument(value=threshold)
            if source.derived_of_liquid_dag is None:
                variable = _numeric_nbegm_argument(value=pool[self.liquid_name])
            else:
                function = source.derived_of_liquid_dag
                variable = _numeric_nbegm_argument(
                    value=function(
                        **{
                            name: pool[name]
                            for name in inspect.signature(function).parameters
                        }
                    )
                )
            sides.append(
                variable >= threshold
                if source.equality_owner == "above"
                else variable > threshold
            )
        return jnp.stack(sides) if self.sources else jnp.zeros((0,), dtype=jnp.bool_)


def build_nbegm_interval_recipe(
    *,
    declarations: Mapping[FunctionName, UserFunction],
    functions: EconFunctionsMapping,
    liquid_name: StateName,
    state_names: tuple[StateName, ...],
    action_names: tuple[ActionName, ...],
) -> NBEGMIntervalRecipe:
    """Resolve canonical schedule declarations without inspecting a numeric bank."""
    from _lcm.egm.nbegm import collect_nbegm_metadata  # noqa: PLC0415

    registry = collect_nbegm_metadata(functions=declarations)
    sources = []
    identities = []
    inputs = set()
    for schedule in registry.piecewise_affine_schedules:
        derived = (
            None
            if schedule.variable == liquid_name
            else concatenate_functions(dict(functions), targets=schedule.variable)
        )
        names = () if derived is None else tuple(inspect.signature(derived).parameters)
        params = tuple(
            name for name in names if name not in (*state_names, *action_names)
        )
        fixed_states = tuple(
            name for name in names if name in state_names and name != liquid_name
        )
        inputs.update(name for name in names if name != liquid_name)
        for schedule_breakpoint in schedule.breakpoints:
            threshold = f"{schedule.output}__{schedule_breakpoint.threshold}"
            sources.append(
                _NBEGMSource(
                    variable=schedule.variable,
                    threshold_param_name=threshold,
                    kind=schedule_breakpoint.kind,
                    equality_owner=schedule_breakpoint.equality_owner,
                    derived_of_liquid_dag=derived,
                    derived_param_names=params,
                    derived_state_names=fixed_states,
                    threshold_index_state=schedule_breakpoint.indexed_by,
                    threshold_static_index=schedule_breakpoint.static_index,
                    threshold_subkey=schedule_breakpoint.threshold_subkey,
                )
            )
            identities.append(
                (
                    schedule.output,
                    schedule.variable,
                    threshold,
                    schedule_breakpoint.kind,
                    schedule_breakpoint.equality_owner,
                    schedule_breakpoint.indexed_by,
                    schedule_breakpoint.static_index,
                    schedule_breakpoint.threshold_subkey,
                )
            )
            inputs.add(threshold)
            if schedule_breakpoint.indexed_by is not None:
                inputs.add(schedule_breakpoint.indexed_by)
    evaluator = _NBEGMQueryIntervalEvaluators(
        sources=tuple(sources), liquid_name=liquid_name
    )
    if registry.piece_sets:
        # The admitted case route has one state-independent liquid comparison.
        piece = registry.piece_sets[0]
        threshold, _when_below, owner = _normalize_case_boundary(
            boundary=registry.boundaries[piece.predicate_name],
            predicate_name=piece.predicate_name,
            liquid_state_name=liquid_name,
            reserved_names=frozenset((*state_names, *action_names)),
        )
        if isinstance(threshold, Const):
            read_threshold = _LiteralCaseThreshold(literal=threshold.value)
            threshold_identity = ("literal", threshold.value)
        elif threshold.name in functions:
            function = concatenate_functions(dict(functions), targets=threshold.name)
            argument_names = tuple(inspect.signature(function).parameters)
            read_threshold = _ComputedCaseThreshold(
                threshold_dag=function, argument_names=argument_names
            )
            inputs.update(argument_names)
            threshold_identity = ("target", threshold.name)
        else:
            qualified_name = f"{piece.predicate_name}__{threshold.name}"
            read_threshold = _ParameterCaseThreshold(qualified_name=qualified_name)
            inputs.add(qualified_name)
            threshold_identity = ("parameter", qualified_name)
        case = _NBEGMCaseIntervalEvaluators(
            threshold=read_threshold,
            liquid_name=liquid_name,
            equal_above=owner == "otherwise",
        )
        if sources or len(registry.piece_sets) != 1:
            raise RegimeInitializationError(
                "The admitted NB case route requires its single canonical boundary."
            )
        return NBEGMIntervalRecipe(
            source_identity=(
                ("case", piece.predicate_name, threshold_identity, owner),
            ),
            input_names=frozenset(inputs),
            equal_above=(owner == "otherwise",),
            evaluate_preimages=case.preimages,
            evaluate_sides=case.sides,
        )
    return NBEGMIntervalRecipe(
        source_identity=tuple(identities),
        input_names=frozenset(inputs),
        equal_above=tuple(source.equality_owner == "above" for source in sources),
        evaluate_preimages=evaluator.preimages,
        evaluate_sides=evaluator.sides,
    )


@dataclass(frozen=True, kw_only=True)
class _NBEGMCaseIntervalEvaluators:
    """The declared exact comparison and its representable liquid selection cut."""

    threshold: _LiteralCaseThreshold | _ParameterCaseThreshold | _ComputedCaseThreshold
    """Canonical state-independent case threshold resolver."""
    liquid_name: StateName
    """The actual liquid comparison argument."""
    equal_above: bool
    """Whether the declared above side owns equality."""

    def preimages(self, pool: EconFunctionKwargs) -> tuple[Float1D, BoolND]:
        """Use the case route's exact direct-variable equality convention."""
        threshold = jnp.asarray(self.threshold(pool), dtype=canonical_float_dtype())
        selection = threshold if self.equal_above else jnp.nextafter(threshold, jnp.inf)
        return selection[jnp.newaxis], jnp.ones((1,), dtype=jnp.bool_)

    def sides(self, pool: EconFunctionKwargs) -> BoolND:
        """Compare the actual liquid value to the canonical threshold."""
        threshold = self.threshold(pool)
        liquid = _numeric_nbegm_argument(value=pool[self.liquid_name])
        above = liquid >= threshold if self.equal_above else liquid > threshold
        return jnp.asarray(above)[jnp.newaxis]


def _conditional_nbegm_cell_carry(
    *,
    rows: tuple[FloatND, FloatND, FloatND, EGMQueryData],
    query_grid: Float1D,
    published_boundaries: Float1D,
    has_actions: bool,
) -> EGMCarry:
    """Retain each actual associated tuple before branch max or node extraction."""
    value, marginal, consumption, data = rows
    if has_actions:
        # The admitted floor is independent of discrete actions. Its one scalar
        # cell value is represented once, outside the conditional action axes.
        data = replace(data, floor_used=data.floor_used[0])
    return EGMCarry(
        endog_grid=jnp.broadcast_to(query_grid, value.shape),
        value=value,
        marginal_utility=marginal,
        policy=consumption,
        taste_shock_scale=jnp.asarray(0.0, dtype=value.dtype),
        breakpoints=(
            jnp.broadcast_to(
                published_boundaries, (*value.shape[:-1], published_boundaries.size)
            )
            if published_boundaries.size
            else None
        ),
        query_data=data,
    )


def _assemble_nbegm_query_carry(
    *,
    carry: EGMCarry,
    ride_shape: tuple[int, ...],
    working_state_names: tuple[StateName, ...],
    canonical_state_names: tuple[StateName, ...],
    action_shape: tuple[int, ...],
) -> EGMCarry:
    """Put full conditional rows in named canonical state/action order."""
    if len(working_state_names) != len(canonical_state_names) or frozenset(
        working_state_names
    ) != frozenset(canonical_state_names):
        raise TypeError("NB working and canonical fixed-state roles differ.")
    if carry.query_data is None:
        raise TypeError("Conditional NB publication lacks original query records.")
    state_permutation = tuple(
        working_state_names.index(name) for name in canonical_state_names
    )
    prefix_shape = (*ride_shape, *action_shape)
    source_prefix_rank = 1 + bool(action_shape)

    def reshape_conditional(leaf: FloatND | IntND | BoolND) -> FloatND | IntND | BoolND:
        tail = leaf.shape[source_prefix_rank:]
        shaped = leaf.reshape(*prefix_shape, *tail)
        permutation = (*state_permutation, *range(len(ride_shape), shaped.ndim))
        return jnp.transpose(shaped, axes=permutation)

    data = carry.query_data
    # The floor is stored once per fixed cell and therefore has no action axis.
    floor = data.floor_used.reshape(ride_shape)
    floor = jnp.transpose(floor, axes=state_permutation)
    if data.liquid_corner_data is not None:
        raise TypeError("NB publication requires its compact interval representation.")
    data = replace(
        data,
        floor_used=floor,
        savings=reshape_conditional(data.savings),
        node_kind=reshape_conditional(data.node_kind),
        link_valid=reshape_conditional(data.link_valid),
        owner_id=reshape_conditional(data.owner_id),
        side_id=reshape_conditional(data.side_id),
        corner_continuation=None
        if data.corner_continuation is None
        else reshape_conditional(data.corner_continuation),
        corner_kind=None
        if data.corner_kind is None
        else reshape_conditional(data.corner_kind),
        corner_interval_id=None
        if data.corner_interval_id is None
        else reshape_conditional(data.corner_interval_id),
        corner_side_id=None
        if data.corner_side_id is None
        else reshape_conditional(data.corner_side_id),
    )
    return EGMCarry(
        endog_grid=reshape_conditional(carry.endog_grid),
        value=None if carry.value is None else reshape_conditional(carry.value),
        marginal_utility=reshape_conditional(carry.marginal_utility),
        policy=None if carry.policy is None else reshape_conditional(carry.policy),
        breakpoints=None
        if carry.breakpoints is None
        else reshape_conditional(carry.breakpoints),
        taste_shock_scale=jnp.asarray(0.0, dtype=carry.endog_grid.dtype),
        query_data=data,
    )


def _fold_nbegm_read_rows(
    *,
    rows: tuple[FloatND, FloatND] | BoundedResult[tuple[FloatND, FloatND]],
    savings_grid: Float1D,
    targets: FloatND | None,
) -> tuple[FloatND, FloatND] | BoundedResult[tuple[FloatND, FloatND]]:
    """Fold actual source saving nodes and live cliff columns into child slots."""
    if not isinstance(rows, BoundedResult):
        return rows
    active = jnp.ones(savings_grid.shape, dtype=jnp.bool_)
    if targets is not None:
        active = jnp.concatenate([active, ~jnp.isnan(targets)])
    return replace(
        rows,
        bound_status=reduce_bound_input_status(
            statuses=rows.bound_status, active=active
        ),
        query_status=reduce_query_input_status(
            statuses=rows.query_status, active=active
        ),
        continuation_read_counts=reduce_continuation_read_counts(
            counts=rows.continuation_read_counts,
            active=active,
        ),
    )


def _build_nbegm_query_carry_template(
    *,
    context: SolverBuildContext,
    model: NBEGMQueryModel,
    retain_policy: bool = True,
) -> EGMCarry:
    """Build numeric conditional NB shapes solely from independent axis authority."""
    spec = model.authority
    state_shape = tuple(
        int(context.grids[name].to_jax().shape[0]) for name in spec.state_names
    )
    action_shape = tuple(
        int(context.grids[name].to_jax().shape[0]) for name in spec.action_names
    )
    data = build_template_egm_query_data(
        state_shape=state_shape,
        action_shape=action_shape,
        row_capacity=spec.row_capacity,
        interval_capacity=len(spec.interval_ids),
    )
    template = build_template_egm_carry(
        n_rows=spec.row_capacity,
        leading_shape=(*state_shape, *action_shape),
        query_data=data,
    )
    n_liquid = int(context.grids[spec.bound.liquid_state].to_jax().shape[0])
    n_boundaries = (spec.row_capacity - n_liquid) // 2
    template = replace(
        template,
        policy=jnp.zeros_like(template.endog_grid) if retain_policy else None,
        breakpoints=jnp.zeros(
            (*state_shape, *action_shape, n_boundaries), dtype=canonical_float_dtype()
        )
        if n_boundaries
        else None,
    )
    return shard_carry_template(
        template=template,
        grids=context.grids,
        leading_axis_names=spec.state_names,
        sharded_state_names=context.sharded_state_names,
        devices=placed_devices_for_ids(submesh_device_ids=context.submesh_device_ids),
    )


def _nbegm_bounded_output_roles(
    *,
    ordinary: tuple[OutputRoleTree, ...],
    own_bound_count: int,
    query_status_count: int,
) -> OutputRoleTree:
    """Describe scalar acceptance counts and each mandatory declaration slot."""
    if not own_bound_count and not query_status_count:
        return ordinary
    count = StateAxesLeading(state_names=(), shape=(), dtype=jnp.int32)
    scalar = StateAxesLeading(state_names=(), shape=(), dtype=jnp.bool_)
    inactive = jnp.asarray(a=False)
    numeric_ordinary = jax.tree.map(lambda _role: jnp.asarray(0.0), ordinary)
    template = BoundedResult(
        output=(*numeric_ordinary, (jnp.int32(0), jnp.int32(0))),
        bound_status=tuple(
            BoundInputStatus(inactive, inactive) for _ in range(own_bound_count)
        ),
        query_status=tuple(
            QueryInputStatus(inactive, inactive, inactive)
            for _ in range(query_status_count)
        )
        if query_status_count
        else None,
    )
    return jax.tree.unflatten(
        jax.tree.structure(template),
        jax.tree.leaves(
            (
                (*ordinary, (count, count)),
                jax.tree.map(lambda _leaf: scalar, template.bound_status),
                jax.tree.map(lambda _leaf: scalar, template.query_status),
            )
        ),
    )


def _required_unprepared_nbegm_read(*, read: _NBEGMPreparedRead) -> _NBEGMCellRead:
    """Keep one source floor preparation around the actual child continuation."""
    if isinstance(read, _NBEGMPreparedCell):
        raise TypeError("An NB continuation must prepare its source floor only once.")
    return read


def _numeric_nbegm_argument(*, value: EconFunctionArg) -> FloatND | float:
    """Require a numeric role after any declared grouped threshold selection."""
    if isinstance(value, jax.Array | float):
        return value
    raise TypeError("An NB numeric role cannot read a grouped parameter leaf.")


def _required_nbegm_plan(*, plan: ContinuationPlan | None) -> ContinuationPlan:
    """Require the already declared plan beside installed child query slots."""
    if plan is None:
        raise TypeError("Installed NB child slots require their continuation plan.")
    return plan


def _required_nbegm_child_query(
    *, plan: ContinuationPlan, target: RegimeName
) -> DCEGMQuery | NBEGMQuery:
    """Bind one actual declared target adapter without inventing authority."""
    query = plan.child_reads[target].query
    if query is None:
        raise TypeError("An NB query target requires its installed child adapter.")
    return query


def _with_nbegm_source_arguments(
    *,
    kernel: _EGMPeriodKernel,
    current_floor: _NBEGMFloor | None,
    owned_child_reads: bool,
) -> _EGMPeriodKernel:
    """Bind actual source floor preparation into the declared argument builders."""
    programs = {}
    for name, program in kernel.core_programs().items():
        builder = program.argument_builder
        if not isinstance(builder, _EGMArgumentBuilder):
            raise TypeError(
                "An NB source floor requires its declared law argument builder."
            )
        programs[name] = replace(
            program,
            argument_builder=_NBEGMArgumentBuilder(
                **{
                    entry.name: getattr(builder, entry.name)
                    for entry in fields(_EGMArgumentBuilder)
                },
                current_floor=current_floor,
                owned_child_reads=owned_child_reads,
            ),
        )
    return replace(kernel, _core_programs=MappingProxyType(programs))


def _nbegm_own_bound_specs(
    *, floor: _NBEGMFloor | None, bound: PostDecisionBoundSpec | None
) -> tuple[PostDecisionBoundSpec, ...]:
    """Declare the canonical source bound exactly when its floor is prepared."""
    if floor is None:
        return ()
    if bound is None:
        raise TypeError("An NB source floor requires its canonical bound declaration.")
    return (bound,)


def _nbegm_current_predicates(
    *,
    context: SolverBuildContext,
    bound: PostDecisionBoundSpec | None,
) -> tuple[Callable[..., BoolND], ...]:
    """Retain existing route predicates independently of their budget proof."""
    from _lcm.egm.budget import DCEGM_BUDGET_CONSTRAINT_NAME  # noqa: PLC0415

    proved = () if bound is None else bound.declaration_names
    evaluated = tuple(
        predicate
        for name, predicate in context.constraints.items()
        if name not in proved and name != DCEGM_BUDGET_CONSTRAINT_NAME
    )
    model = context.nbegm_source_query_model or (context.nbegm_query_models or {}).get(
        context.regime_name
    )
    if model is None or not model.evaluated_predicate_names:
        return evaluated
    compiled = _consume_nbegm_feasibility_constraints(
        context=context, solver_path=("nbegm",)
    )
    return (*evaluated, *(item.predicate for item in compiled))


def _required_nbegm_query_step(*, step: _NBEGMMappedStepChannels) -> _NBEGMQueryStep:
    """Require the actual associated producer variant on a retained-query route."""
    if not isinstance(step, _NBEGMQueryStep):
        raise TypeError("A retained NB step must publish its original query records.")
    return step


def _required_nbegm_query_data(
    *, step: _NBEGMStepChannels | _OwnedNBEGMQueryStep
) -> EGMQueryData:
    """Read associated records from the actual retained producer variant."""
    if not isinstance(step, (_NBEGMQueryStep, _OwnedNBEGMQueryStep)):
        raise TypeError("A retained NB step must publish original query records.")
    return step.query_data


def _required_nbegm_discrete_step(
    *, step: tuple[Float1D, Float1D, Float1D, IntND] | _NBEGMDiscreteQueryStep
) -> _NBEGMDiscreteQueryStep:
    """Require every actual conditional tuple from the discrete producer."""
    if not isinstance(step, _NBEGMDiscreteQueryStep):
        raise TypeError(
            "A retained discrete NB step must publish its conditional rows."
        )
    return step


def _scalar_conditional_nbegm_carry(
    *,
    value: FloatND,
    marginal: FloatND,
    data: EGMQueryData,
    grid: Float1D,
    action_shape: tuple[int, ...],
    breakpoints: Float1D | None,
) -> EGMCarry:
    """Shape original conditional scalar rows without maxing or slicing their bank."""
    carry = EGMCarry(
        endog_grid=jnp.broadcast_to(grid, value.shape),
        value=value,
        marginal_utility=marginal,
        taste_shock_scale=jnp.asarray(0.0, dtype=value.dtype),
        breakpoints=None
        if breakpoints is None
        else jnp.broadcast_to(breakpoints, (*value.shape[:-1], breakpoints.size)),
        query_data=data,
    )
    carry = jax.tree.map(lambda leaf: leaf[jnp.newaxis], carry)
    return _assemble_nbegm_query_carry(
        carry=carry,
        ride_shape=(),
        working_state_names=(),
        canonical_state_names=(),
        action_shape=action_shape,
    )


@dataclass(frozen=True, kw_only=True)
class _NBEGMScalarChildReads:
    """Share prepared child geometry and reduce each actual scalar read window."""

    continuation: Callable[[ScalarFloat], _NBEGMReadResult]
    """The once-bound reader on actual source saving nodes."""
    offnode: Callable[[ScalarFloat], _NBEGMReadResult]
    """The same prepared geometry with the tabulated source law off node."""
    plan: ContinuationPlan
    """Canonical child slot declarations, in their original order."""
    savings_grid: Float1D
    """The effective source nodes and safe inactive read operand."""
    records: list[BoundedResult[None]]
    """The existing owning call's scalar summaries, with no per-node history."""

    def read_rows(
        self,
        *,
        reader: Callable[[ScalarFloat], _NBEGMReadResult],
        savings: FloatND,
        active: BoolND,
    ) -> tuple[FloatND, FloatND]:
        """Read a bounded source window and append its reduced child summaries."""
        active = jnp.broadcast_to(active, savings.shape)
        safe = jnp.where(active, savings, self.savings_grid[0])
        rows = jax.vmap(reader)(safe.reshape(-1))
        if not isinstance(rows, BoundedResult):
            raise TypeError(
                "An installed NB child query omitted its mandatory records."
            )
        if rows.bound_status:
            raise TypeError("NB child integrity belongs to canonical query slots.")
        statuses = reduce_query_input_status(
            statuses=rows.query_status, active=active.reshape(-1)
        )
        if statuses is None or len(statuses) != len(self.plan.query_targets):
            raise TypeError("An NB child read changed its canonical declaration slots.")
        counts = reduce_continuation_read_counts(
            counts=rows.continuation_read_counts,
            active=active.reshape(-1),
        )
        self.records.append(
            BoundedResult(
                output=None,
                bound_status=(),
                query_status=statuses,
                continuation_read_counts=counts,
            )
        )

        value, marginal = rows.output
        return value.reshape(savings.shape), marginal.reshape(savings.shape)

    # keyword-only-exempt: library-callback=NBEGMContinuationRead
    def __call__(self, savings: FloatND, active: BoolND) -> tuple[FloatND, FloatND]:
        """Read the actual off-node saving against already prepared child rows."""
        return self.read_rows(reader=self.offnode, savings=savings, active=active)


def _prepare_nbegm_scalar_child_reads(
    *,
    plan: ContinuationPlan,
    params: EconFunctionKwargs,
    next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
    dtype: DTypeLike,
    effective_savings_grid: Float1D,
    next_liquid: Float1D,
    records: list[BoundedResult[None]],
    collect_continuation_read_counts: bool = False,
) -> _NBEGMScalarChildReads:
    """Bind original node and off-node laws to one prepared child geometry."""
    from _lcm.egm.continuation import (  # noqa: PLC0415
        _BoundContinuation,
        _ChildCarryReader,
        bind_continuation,
    )

    continuation = bind_continuation(
        plan=plan,
        combo_pool=params,
        next_regime_to_continuation=next_regime_to_continuation,
        dtype=dtype,
        **(
            {"collect_continuation_read_counts": True}
            if collect_continuation_read_counts
            else {}
        ),
    )
    offnode_plan = replace(
        plan,
        child_reads=MappingProxyType(
            {
                target: replace(
                    child,
                    euler_state_func=_NBEGMTabulatedEulerLaw(
                        savings_grid=effective_savings_grid,
                        next_liquid=next_liquid,
                        post_decision_name=plan.post_decision_name,
                        next_state_key=child.next_state_key,
                    ),
                )
                for target, child in plan.child_reads.items()
            }
        ),
    )
    if not isinstance(continuation, _BoundContinuation):
        raise TypeError(
            "NB scalar binding requires its canonical prepared continuation reader."
        )
    offnode_readers = {}
    for target, child_reader in continuation.child_readers.items():
        if not isinstance(child_reader, _ChildCarryReader):
            raise TypeError(
                "NB scalar binding requires its canonical prepared child reader."
            )
        offnode_readers[target] = replace(
            child_reader, read=offnode_plan.child_reads[target]
        )
    offnode_continuation = replace(
        continuation,
        plan=offnode_plan,
        child_readers=MappingProxyType(offnode_readers),
    )
    return _NBEGMScalarChildReads(
        continuation=continuation,
        offnode=offnode_continuation,
        plan=plan,
        savings_grid=effective_savings_grid,
        records=records,
    )


@dataclass(frozen=True, kw_only=True)
class _NBEGMActionEligibility:
    """Bind one actual action beside source resources and current predicates."""

    params: EconFunctionKwargs
    """The current source's already qualified fixed inputs."""
    binding: Mapping[ActionName, int]
    """One immutable product of the actual discrete action codes."""
    liquid_name: StateName
    """The source's declared physical liquid role."""
    resources: Callable[..., FloatND]
    """The actual composed source cash-on-hand map."""
    resource_names: tuple[str, ...]
    """External names read by the resource map."""
    predicates: tuple[Callable[..., BoolND], ...]
    """Current predicates evaluated jointly with each action's budget."""
    predicate_names: tuple[tuple[str, ...], ...]
    """External input names aligned with the current predicates."""
    lowest_savings: ScalarFloat
    """The first actual effective source saving node."""

    def __call__(self, point: ScalarFloat) -> BoolND:
        actual = {**self.params, **self.binding, self.liquid_name: point}
        resources = self.resources(
            **{name: actual[name] for name in self.resource_names}
        )
        result = resources > self.lowest_savings
        for predicate, names in zip(self.predicates, self.predicate_names, strict=True):
            result &= predicate(**{name: actual[name] for name in names})
        return result


@dataclass(frozen=True, kw_only=True)
class _NBEGMOwnedSingleAxisCore:
    """Publish actual source eligibility and fold every evaluated child read."""

    inner: _RawNBEGMCore
    """The existing case/schedule branch solver with optional original-A records."""
    continuation_plan: ContinuationPlan
    """Canonical next-period child readers and their exact declaration slots."""
    liquid_name: StateName
    """Physical current-state argument of the declared budget and predicates."""
    resources: Callable[..., FloatND]
    """Current resources evaluated independently of conditional carry values."""
    predicates: tuple[Callable[..., BoolND], ...]
    """Current fixed-state/action predicates admitted by this solver route."""
    action_names: tuple[ActionName, ...]
    """Canonical conditional action axes."""
    action_bindings: tuple[Mapping[ActionName, int], ...]
    """The exact product of existing discrete action codes, or one empty binding."""
    owns_query_rows: bool
    """Whether the current regime independently declares conditional query rows."""
    collect_continuation_read_counts: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Collect the genuine paired-read inventory in the checked specialization."""

    def __call__(
        self,
        *,
        liquid: Float1D,
        next_liquid_grid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        next_liquid: Float1D,
        marginal_return: Float1D,
        effective_savings_grid: Float1D,
        boundary_savings_targets: Float1D,
        boundary_next_liquid: Float1D,
        next_regime_to_continuation: MappingProxyType[
            RegimeName, EGMCarry
        ] = MappingProxyType({}),
        source_bound_status: BoundInputStatus | None = None,
        next_query_data: EGMQueryData | None = None,
        next_query_pool: EconFunctionKwargs | None = None,
        **params: ValueND,
    ) -> BoundedResult[tuple[FloatND, EGMCarry, tuple[IntND, IntND], BoolND]]:
        """Keep conditional banks raw and mask only the ordinary source publication."""
        # Child query arrays travel through the declared full-carry mapping. The
        # legacy flattened aliases have no independent evaluator authority.
        del next_query_data, next_query_pool
        records: list[BoundedResult[None]] = []
        continuation_read = None
        if self.continuation_plan.query_targets:
            prepared_reads = _prepare_nbegm_scalar_child_reads(
                plan=self.continuation_plan,
                params=params,
                next_regime_to_continuation=next_regime_to_continuation,
                dtype=liquid.dtype,
                effective_savings_grid=effective_savings_grid,
                next_liquid=next_liquid,
                records=records,
                collect_continuation_read_counts=self.collect_continuation_read_counts,
            )
            next_value, next_marginal = prepared_reads.read_rows(
                reader=prepared_reads.continuation,
                savings=effective_savings_grid,
                active=jnp.ones_like(effective_savings_grid, dtype=jnp.bool_),
            )
            next_liquid_grid = effective_savings_grid
            marginal_return = jnp.ones_like(effective_savings_grid)
            continuation_read = prepared_reads
        solved = self.inner(
            liquid=liquid,
            next_liquid_grid=next_liquid_grid,
            next_value=next_value,
            next_marginal=next_marginal,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            effective_savings_grid=effective_savings_grid,
            boundary_savings_targets=boundary_savings_targets,
            boundary_next_liquid=boundary_next_liquid,
            source_bound_status=source_bound_status,
            continuation_read=continuation_read,
            **params,
        )
        if isinstance(solved, BoundedResult):
            value, carry = solved.output
            own_status = solved.bound_status
            if solved.query_status:
                raise TypeError("A scalar NB step published duplicate child slots.")
        else:
            value, carry = solved
            own_status = ()
        eligibility = []
        resource_names = tuple(inspect.signature(self.resources).parameters)
        predicate_names = tuple(
            tuple(inspect.signature(predicate).parameters)
            for predicate in self.predicates
        )
        for binding in self.action_bindings:
            eligible = _NBEGMActionEligibility(
                params=params,
                binding=binding,
                liquid_name=self.liquid_name,
                resources=self.resources,
                resource_names=resource_names,
                predicates=self.predicates,
                predicate_names=predicate_names,
                lowest_savings=effective_savings_grid[0],
            )
            eligibility.append(jax.vmap(eligible)(liquid))
        feasible = jnp.stack(eligibility)
        state_feasible = jnp.any(feasible, axis=0)
        if self.owns_query_rows:
            carry = _with_scalar_nbegm_sampled_run(carry=carry, core=self.inner)
            raw_rows = carry.value.reshape((-1, carry.value.shape[-1]))
            grids = carry.endog_grid.reshape(raw_rows.shape)
            # Augmented feasibility rows stay in the carry. Locate original
            # physical nodes by the same exact stored abscissae, not by slicing.
            raw_at_nodes = jax.vmap(
                lambda x, y: y[jnp.searchsorted(x, liquid, side="left")]
            )(grids, raw_rows)
            source_value = jnp.max(jnp.where(feasible, raw_at_nodes, -jnp.inf), axis=0)
        else:
            source_value = jnp.where(state_feasible, value, -jnp.inf)
        child_status = None
        read_counts = None
        if self.continuation_plan.query_targets:
            child_status, read_counts = _fold_nbegm_scalar_read_records(
                records=records,
                n_slots=len(self.continuation_plan.query_targets),
                collect_counts=self.collect_continuation_read_counts,
            )
        return BoundedResult(
            output=(
                source_value,
                carry,
                (
                    sum_counts(counts=~state_feasible),
                    sum_counts(counts=state_feasible & jnp.isinf(source_value)),
                ),
                jnp.asarray(a=False),
            ),
            bound_status=own_status,
            query_status=child_status,
            continuation_read_counts=read_counts,
        )


def _record_nbegm_scalar_child_slots(
    *,
    plan: ContinuationPlan | None,
    period: int,
    query_slots_by_period: dict[
        int, tuple[tuple[RegimeName, int, PostDecisionBoundSpec], ...]
    ],
    reader_capabilities: dict[int, Mapping[RegimeName, ContinuationCapabilities]],
) -> tuple[tuple[RegimeName, int, PostDecisionBoundSpec], ...]:
    """Keep canonical child slots and installed read capabilities in plan order."""
    child_slots = (
        ()
        if plan is None
        else tuple(
            (
                child,
                period + 1,
                _required_nbegm_child_query(
                    plan=_required_nbegm_plan(plan=plan), target=child
                ).model.authority.bound,
            )
            for child in plan.query_targets
        )
    )
    if child_slots:
        query_slots_by_period[period] = child_slots
        reader_capabilities[period] = MappingProxyType(
            {
                child: _required_nbegm_child_query(
                    plan=_required_nbegm_plan(plan=plan), target=child
                ).capabilities
                for child in _required_nbegm_plan(plan=plan).query_targets
            }
        )
    return child_slots


def _nbegm_cell_continuation_columns(
    *,
    cont_value: FloatND | None,
    cont_marginal: FloatND | None,
    interval_continuation: _NBEGMIntervalContinuation | None,
    savings_grid: Float1D,
    cliff_savings: FloatND | None,
) -> tuple[FloatND | None, FloatND | None, FloatND | None]:
    """Separate actual one-shot savings and cliff columns without reading intervals."""
    if interval_continuation is None:
        if cont_value is None or cont_marginal is None:
            raise ValueError("one-shot NB-EGM cells require value and marginal rows.")
        cont_value, cont_marginal, extra_cont_value = _split_cliff_columns(
            cont_value=cont_value,
            cont_marginal=cont_marginal,
            n_nodes=savings_grid.shape[0],
            has_cliff_columns=cliff_savings is not None,
        )
    else:
        extra_cont_value = None
    return cont_value, cont_marginal, extra_cont_value


def _fold_nbegm_scalar_read_records(
    *, records: list[BoundedResult[None]], n_slots: int, collect_counts: bool
) -> tuple[tuple[QueryInputStatus, ...] | None, ContinuationReadCounts | None]:
    """Fold only executed scalar child reads, preserving their slots and counts."""
    child_status = None
    read_counts = None
    if not records:
        raise TypeError("An NB feature consumer evaluated no child declaration slots.")
    for record in records:
        if record.query_status is None or len(record.query_status) != n_slots:
            raise TypeError("An NB scalar read changed its declaration slots.")
        child_status = (
            record.query_status
            if child_status is None
            else merge_query_input_status(left=child_status, right=record.query_status)
        )
        if (record.continuation_read_counts is None) == collect_counts:
            raise TypeError("An NB scalar read changed its checked count presence.")
        read_counts = merge_continuation_read_counts(
            left=read_counts,
            right=record.continuation_read_counts,
        )
    return child_status, read_counts


def _validate_nbegm_integrity_slot_counts(
    *,
    outputs: BoundedResult[PytreeValue],
    own_bound_specs: tuple[PostDecisionBoundSpec, ...],
    query_input_authorities: tuple[tuple[RegimeName, int, PostDecisionBoundSpec], ...]
    | None,
) -> None:
    """Require the actual summary population to match canonical declaration slots."""
    if len(outputs.bound_status) != len(own_bound_specs) or len(
        outputs.query_status or ()
    ) != len(query_input_authorities or ()):
        raise TypeError("NB integrity records differ from canonical declaration slots.")


def _bounded_nbegm_kernel_output(
    *,
    value: FloatND,
    carry: EGMCarry,
    counts: FeasibilityCounts,
    summary: BoundedResult[PytreeValue],
    current_feasibility: BoolND | None,
    prepared_source_floor: PreparedNestedFloor | None,
    replay: Mapping[ArtifactKey, NBEGMGridPolicy] | None = None,
) -> FeasibilityKernelOutput:
    """Publish accepted values and carry beside the unchanged mandatory summaries."""
    if current_feasibility is not None:
        return _NBEGMNestedKernelOutput(
            value=value,
            continuations={EGM_CONTINUATION: carry},
            feasibility_counts=counts,
            bound_status=summary.bound_status,
            query_status=summary.query_status,
            continuation_read_counts=summary.continuation_read_counts,
            current_feasibility=current_feasibility,
            prepared_source_floor=prepared_source_floor,
            **({"replay": replay} if replay is not None else {}),
        )
    return FeasibilityKernelOutput(
        value=value,
        continuations={EGM_CONTINUATION: carry},
        feasibility_counts=counts,
        bound_status=summary.bound_status,
        query_status=summary.query_status,
        continuation_read_counts=summary.continuation_read_counts,
        **({"replay": replay} if replay is not None else {}),
    )


def _with_scalar_nbegm_sampled_run(
    *,
    carry: EGMCarry,
    core: _RawNBEGMCore,
) -> EGMCarry:
    """Publish the scalar continuous budget's connected sampled-row ownership.

    Scalar continuous kernels represent one interpolated physical row per action.
    Their affine budget intervals separately index the compact corner bank. Case
    jumps, interval-held continuation kernels and one-sided records retain their
    original ownership and links.
    """
    if isinstance(core, _NBEGMContinuousCore):
        if core.feasibility_constraints:
            return carry
        kinds = core.schedule_spec.breakpoint_kinds
    elif isinstance(core, _NBEGMScheduleDiscreteCore):
        kinds = core.spec.breakpoint_kinds
    else:
        return carry
    if any(kind != "continuous_kink" for kind in kinds):
        return carry
    data = carry.query_data_for_transport()
    if data is None:
        raise TypeError("A scalar conditional publication requires its original rows.")
    represented = (data.node_kind == QueryNodeKind.FINITE) | (
        data.node_kind == QueryNodeKind.ECONOMIC_MINUS_INF
    )
    sampled_labels = jnp.all(
        ~represented
        | ((data.side_id == 0) & (data.owner_id >= 0) & (data.owner_id <= len(kinds)))
    )
    sampled_owner = jnp.where(represented, 0, -1).astype(jnp.int32)
    sampled_links = data.link_valid | (
        represented[..., :-1]
        & represented[..., 1:]
        & (data.owner_id[..., :-1] != data.owner_id[..., 1:])
    )
    return replace(
        carry,
        query_data=replace(
            data,
            owner_id=jnp.where(sampled_labels, sampled_owner, data.owner_id),
            link_valid=jnp.where(sampled_labels, sampled_links, data.link_valid),
        ),
    )


def _with_nbegm_scalar_output_roles(
    *,
    kernel: _EGMPeriodKernel,
    liquid_grid: Float1D,
    carry_template: EGMCarry,
    own_bound_count: int,
    query_status_count: int,
) -> _EGMPeriodKernel:
    """Declare the exact conditional carry and real scalar acceptance outputs."""
    inactive = jnp.asarray(a=False)
    numeric = BoundedResult(
        output=(
            jnp.zeros_like(liquid_grid),
            carry_template,
            (jnp.int32(0), jnp.int32(0)),
            inactive,
        ),
        bound_status=tuple(
            BoundInputStatus(inactive, inactive) for _ in range(own_bound_count)
        ),
        query_status=tuple(
            QueryInputStatus(inactive, inactive, inactive)
            for _ in range(query_status_count)
        )
        if query_status_count
        else None,
    )
    roles = jax.tree.map_with_path(_scalar_egm_output_role, numeric)
    return replace(
        kernel,
        _core_programs=MappingProxyType(
            {
                name: replace(
                    program,
                    output_roles=roles,
                    checked_variant=None
                    if program.checked_variant is None
                    else replace(
                        program.checked_variant,
                        output_roles=replace(
                            roles,
                            continuation_read_counts=_unflatten_continuation_read_counts(
                                _aux=None,
                                children=tuple(
                                    StateAxesLeading(
                                        state_names=(), shape=(), dtype=jnp.int32
                                    )
                                    for _ in range(2)
                                ),
                            ),
                        ),
                    ),
                )
                for name, program in kernel.core_programs().items()
            }
        ),
    )


def _with_nbegm_scalar_carry_reads(
    *,
    kernel: PeriodKernel,
    context: SolverBuildContext,
    period: int,
) -> PeriodKernel:
    """Address all owned child leaves through their actual numeric carry mapping."""
    if not isinstance(kernel, _EGMPeriodKernel):
        return kernel
    builder = kernel.core_programs()["main"].argument_builder
    if not isinstance(builder, _NBEGMArgumentBuilder) or not builder.owned_child_reads:
        return kernel
    template = published_continuation_template(
        continuation_specs=context.continuation_specs,
        target=kernel.continuation_target,
    )
    if template is None:
        return kernel
    return replace(
        kernel,
        _core_programs=with_continuation_leaf_reads(
            programs=kernel.core_programs(),
            reads_by_core_key={
                "main": continuation_leaf_reads(
                    template=template,
                    artifact_key=EGM_CONTINUATION,
                    target=kernel.continuation_target,
                    source_regime=kernel.regime_name,
                    source_period=period,
                    core_key="main",
                )
            },
        ),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _NBEGMTabulatedEulerLaw:
    """The scalar core's declared-law tabulation at its existing off-node sites."""

    savings_grid: Float1D
    """Exactly the once-prepared source nodes used by the existing law reader."""
    next_liquid: Float1D
    """That canonical source law's existing readings at those nodes."""
    post_decision_name: FunctionName
    """The source candidate argument supplied by the continuation reader."""
    next_state_key: str
    """The canonical target Euler-state law's output key."""

    def __call__(self, **pool: ValueND) -> dict[str, FloatND]:
        """Preserve interpolation and endpoint clipping before the target query."""
        from _lcm.egm.nbegm_step import _law_at_savings  # noqa: PLC0415

        return {
            self.next_state_key: _law_at_savings(
                savings=pool[self.post_decision_name],
                savings_grid=self.savings_grid,
                next_liquid=self.next_liquid,
            )
        }


def _nbegm_child_koopmans(
    *, plan: ContinuationPlan, target: RegimeName
) -> EconFunction:
    """Read the installed child's processed period aggregator."""
    query = _required_nbegm_child_query(plan=plan, target=target)
    return query.model.koopmans_aggregator
