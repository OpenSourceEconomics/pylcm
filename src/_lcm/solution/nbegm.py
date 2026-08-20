"""The NB-EGM solver (non-convex-budget endogenous grid method) and its kernels.

`NBEGM` solves a 1-D consumption-savings problem whose budget may carry
declared breakpoints (kinks and cliffs): the budget is partitioned into
affine-in-liquid intervals, each case piece is solved by EGM, and the pieces
are merged by the MSS upper envelope. `build_period_kernels` returns one
`PeriodKernel` per period — a non-jitted adapter that wraps the solver's
shared jitted core (deduped by function identity, so periods sharing a core
reuse one compiled program) and assembles a `KernelResult` (value array,
continuation carry, published simulation policy) outside JIT.

The kernel-building imports are function-local so the public `lcm.solvers`
façade stays a thin re-export that pulls in no numerical engine modules.
"""

import ast
import functools
import inspect
import itertools
import math
import textwrap
import warnings
from collections.abc import Callable, Hashable, Iterator, Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from beartype import beartype
from dags import concatenate_functions

import lcm.typing as lcm_typing
from _lcm.beartype_conf import REGIME_CONF
from _lcm.certainty_equivalent import CertaintyEquivalent, LinearExpectation
from _lcm.constraints.bounds import without_proved_lower_bounds
from _lcm.constraints.processed import normalize_constraints
from _lcm.constraints.routes import ConstraintRoute
from _lcm.continuation import EGMContinuationLayout, EGMContinuationSpec
from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EGMCarry, shard_carry_template
from _lcm.egm.continuation_grids import (
    continuation_grid_signature,
    continuation_v_interpolation_info,
)
from _lcm.egm.declared_law import build_declared_liquid_law
from _lcm.egm.fixed_width_map import (
    FixedWidthMapGeometry,
    PyTree,
    map_partitioned,
)
from _lcm.egm.nbegm import NBEGMRegistry
from _lcm.egm.preferences import Preferences
from _lcm.egm.upper_envelope.query import ComparisonArithmetic
from _lcm.engine import StateActionSpace
from _lcm.grids import ContinuousGrid, DiscreteGrid
from _lcm.grids.base import Grid
from _lcm.params.mapping_leaf import MappingLeaf, UserMappingLeaf
from _lcm.solution.continuation_target import (
    _period_to_continuation_target,
    _union_fixed_params,
    _union_free_params,
    target_period_grid,
)
from _lcm.solution.contract import (
    ConstraintRouteContext,
    ContinuationPayload,
    KernelResult,
    OneMarginSolver,
    ParamCheck,
    PeriodKernel,
    SolutionKernels,
    SolverBuildContext,
    SolverModelContext,
    _BoundLiquidMargin,
)
from _lcm.solution.dcegm import _carry_subset
from _lcm.solution.egm import (
    _build_one_asset_carry_template,
    _EGMPeriodKernel,
)
from _lcm.typing import (
    EconFunctionsMapping,
    FlatParams,
    RegimeName,
    TransitionFunctionsMapping,
)
from lcm.ages import AgeGrid
from lcm.case_piece import EqualityOwner
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.fixed_forms import (
    FIXED_FORM_ATTRIBUTE,
    cash_on_hand_with_subsidy,
)
from lcm.phased import Phased
from lcm.typing import (
    ActionName,
    BoolND,
    Float1D,
    FloatND,
    FunctionName,
    IntND,
    ScalarFloat,
    StateName,
    StateOrActionName,
    UserFunction,
)

# Key under which ride-along periods share one compiled core: the continuation
# targets either side of a `"|"` separator, followed by those targets'
# age-specialized grid signatures at `period + 1`.
type _RideAlongGroupKey = tuple[RegimeName | Hashable, ...]

# Every discrete action the envelope branches over, paired with its grid codes.
# The branch axis is the product of these code sets, so the pairing has to keep
# each action's name next to its own codes.
type DiscreteActionCodes = tuple[tuple[ActionName, tuple[int, ...]], ...]


# A reference sweep on A100 hardware selected one 256-row ride microtile against
# a four-row branch microtile.  The fixed-width production map carries no GPU
# timing of its own, so these constants record that measured geometry without
# claiming a production speedup.  One ride microtile is also one ride window, so
# a request below the microtile is served by the admitted 256-row partition
# rather than by evaluating a wider block than it commits.  Branches keep the
# bounded 64-row window.
_RIDE_MAP_GEOMETRY = FixedWidthMapGeometry(
    microtile_width=256,
    profile_window=256,
)
_BRANCH_MAP_GEOMETRY = FixedWidthMapGeometry(
    microtile_width=4,
    profile_window=64,
)


def _map_ride_partitioned(
    *, func: Callable[[PyTree], PyTree], xs: PyTree, requested_block_size: int
) -> PyTree:
    """Route one ride/cell axis through the wide ride geometry."""
    return map_partitioned(
        func=func,
        xs=xs,
        requested_block_size=requested_block_size,
        geometry=_RIDE_MAP_GEOMETRY,
    )


def _map_branch_partitioned(
    *, func: Callable[[PyTree], PyTree], xs: PyTree, requested_block_size: int
) -> PyTree:
    """Route one case/discrete branch axis through the narrow branch geometry."""
    return map_partitioned(
        func=func,
        xs=xs,
        requested_block_size=requested_block_size,
        geometry=_BRANCH_MAP_GEOMETRY,
    )


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
    - Schedule with ride-along co-states: two independently-jitted cores per
      period (transition-aware continuation read, then the per-cell envelope
      solve in savings space), batched over the ride-along cells.
    - Discrete action over a smooth budget: one continuous subproblem per
      discrete-action value, merged by the discrete upper envelope.
    """

    savings_grid: ContinuousGrid
    """Exogenous post-decision savings grid `s = coh - consumption` (>= 0)."""
    jump_read: Literal["one_sided", "bridged"] = "one_sided"
    """How the parent's continuation read treats the child value's cliffs.

    The within-period case solve is jump-aware in both modes (masked cases,
    boundary-owner equality); the mode selects what the carry publishes for the
    parents that read it:

    - `"one_sided"` — each carry row holds every jump preimage as a duplicated
      abscissa carrying the exact one-sided value and marginal limits, so reads
      near a cliff are one-sided by construction. Publishing breakpoints gates
      the stochastic-dim fold off on jump-bearing reads, so this mode trades
      runtime for cliff fidelity.
    - `"bridged"` — plain liquid-grid rows with no breakpoints; the parent's
      interpolation may average across a cliff, like any finite-grid solver
      reading the same rows. The fold stays available, so this is the fast
      mode for solves whose consumer tolerates finite-grid cliff error (e.g.
      inner estimation loops, polished afterwards under `"one_sided"`).
    """
    stochastic_node_batch_size: int = 0
    """Block size for splaying the child stochastic-node expectation.

    The continuation read integrates the child's stochastic next-states (health,
    health-cost shocks, the wage residual) over their joint node mesh. `0` reads the
    whole mesh in one vectorized pass; a positive block size loops the mesh in chunks
    of that many nodes, trading compile/runtime for a smaller peak intermediate. Like
    `DCEGM.stochastic_node_batch_size`; raise it when the joint node mesh dominates
    the per-cell memory budget.
    """
    envelope_segment_block_size: int = 0
    """Block size for streaming the merged upper envelope over candidate segments.

    The per-interval envelope brackets every candidate segment against every liquid
    query point; `0` materialises that matrix in one pass, a positive block size
    streams it in blocks of that many segments (identical result, smaller peak
    intermediate). Raise it when the query grid is large enough that the per-cell
    bracket matrix dominates the per-cell memory budget.
    """
    envelope_arithmetic: ComparisonArithmetic = "certified"
    """Which arithmetic decides ownership in the merged upper envelope.

    Every case, corner, and node candidate is read at each liquid query point and
    the largest owns it. How that comparison is made is this knob:

    - `"certified"` compares in double-double precision and publishes NaN wherever
      no candidate is separated, so a reported winner is one the arithmetic could
      prove. Ordering survives the cancellation a nearly-tied crossing produces.
    - `"ordinary"` takes the largest read in the working format. It decides every
      bracketed query and costs a small fraction of the certified read, which is
      the dominant per-cell arithmetic in a case-piece solve. Adequate wherever
      candidate values are separated by much more than the format's resolution at
      their own magnitude — the usual case away from a crossing.
    """
    interval_batch_size: int = 0
    """Intervals committed per iteration of the per-interval continuation read.

    When a carry target's next-state law reads the current liquid state, the
    continuation core evaluates the continuation DAG once per declared liquid
    interval. The body runs at a fixed vector width, so peak intermediates
    follow that width and not this value. `0` commits the whole axis in one
    iteration, which is the fewest iterations and so the least work; a positive
    value commits that many intervals per iteration, rounded up to a multiple of
    the vector width. Every admitted value runs one executable and
    publishes bit-identical values, because the partition is a loop stride
    rather than part of the compilation key.
    """
    cell_block_size: int = 0
    """Ride cells committed per iteration of the ride-along solve.

    Both ride-along cores fan out per cell — the continuation core's transition/
    child-interpolation read and the envelope core's candidate solve. Each
    evaluates cells at a fixed vector width, so no cell's buffers wait on the
    whole mesh and peak intermediates follow that width rather than this value.
    `0` commits the whole mesh in one iteration, which is the fewest iterations
    and so the least work; a positive value commits that many cells per
    iteration, rounded up to a multiple of the vector width. Every admitted
    value runs one executable and publishes bit-identical values, because the
    partition is a loop stride rather than part of the compilation key.
    """
    branch_batch_size: int = 0
    """Discrete-action branches committed per iteration.

    Both ride-along cores evaluate one instance per discrete-action branch — the
    continuation core one continuation row per branch, the envelope core one
    continuous subproblem per branch. Each evaluates branches at a fixed vector
    width, so per-branch intermediates follow that width rather than this value.
    `0` commits the whole axis in one iteration, which is the fewest iterations
    and so the least work; a positive value commits that many branches per
    iteration, rounded up to a multiple of the vector width. Every admitted
    value runs one executable and publishes bit-identical values, because the
    partition is a loop stride rather than part of the compilation key.
    """
    probe_failure: Literal["reject", "assume_declared"] = "reject"
    """What to do when a derivative probe cannot evaluate the model.

    The affine-budget and interval-constancy probes differentiate the model's DAG
    functions on the first solve, reading the declared parameters' own values and
    synthesizing the states and actions they sweep. A DAG that cannot be
    differentiated that way leaves the precondition unverified:

    - `"reject"` — refuse to solve; the per-interval EGM preconditions must be
      machine-verified.
    - `"assume_declared"` — warn and solve; the model author asserts the budget's
      within-interval affinity and every liquid-reading law's interval-constancy,
      to be validated empirically (e.g. full-model brute-agreement gates).
    """

    def __post_init__(self) -> None:
        for name in (
            "stochastic_node_batch_size",
            "envelope_segment_block_size",
            "interval_batch_size",
            "cell_block_size",
            "branch_batch_size",
        ):
            size = getattr(self, name)
            if size < 0:
                msg = (
                    f"NBEGM.{name} must be non-negative, got {size}. Use 0 to run "
                    "the whole axis in one vectorized pass, or a positive value "
                    "to stream it in blocks of that many entries."
                )
                raise RegimeInitializationError(msg)

    def _with_liquid_margin(self, margin: _BoundLiquidMargin) -> _BoundNBEGM:
        """Bind regime-owned DAG names without exposing them on public `NBEGM`."""
        kwargs = {field.name: getattr(self, field.name) for field in fields(NBEGM)}
        return _BoundNBEGM(
            **kwargs,
            continuous_state=margin.state,
            continuous_action=margin.action,
            budget_target=margin.resources,
            post_decision_function=margin.post_decision_state,
        )

    @property
    def requires_continuation(self) -> bool:
        """The case-piece EGM step reads its continuation's marginal value."""
        return True

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
        """Declare the one route the case-piece kernels walk in each phase.

        Declared rather than left undeclared. The default would say the solver
        has not written its routes down, and nothing would be planned for it —
        which is the right answer for a solver nobody has described and the
        wrong one here, where the description is available and says that no
        name is readable anywhere along the pipeline.
        """
        from _lcm.egm.nbegm_routes import case_piece_routes  # noqa: PLC0415

        return case_piece_routes(
            context=context,
            savings_grid=self.savings_grid,
            post_decision_function=proved_post_decision_of(solver=self),
            solver_path=("nbegm",),
        )

    def validate_model(self, *, context: SolverModelContext) -> None:
        """Refuse a declared feasibility constraint the kernel cannot enforce.

        The case-piece kernel inverts the Euler equation at each node of the
        savings grid, so consumption is produced first and the liquid state
        falls out of the budget identity afterwards. A predicate over
        `(state, action)` is evaluable at no point in that step, and the
        candidates the kernel publishes are never masked by one. Declaring one
        and solving anyway answers a different problem than the one written
        down, with no diagnostic.

        A declared lower bound on the post-decision state is the exception: it
        states the number the savings grid already enforces, so it is proved
        against that grid and then carries no predicate for the kernel to
        evaluate. Which declarations qualify is asked of the same function that
        drops them from the engine's constraint set, so the exemption here and
        the drop there cannot come to disagree.

        Keep it that way: a local test for the bound's shape here would be a
        second spelling of a question that already has an answer, and the two
        would drift without a symptom. A bound exempted here but not dropped
        there reaches the engine's constraint set, which is built per discrete
        combo — no place a continuous post-decision state can be read.
        """
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
        unenforceable = without_proved_lower_bounds(
            # `Phased` is rejected in the constraints slot by the phase
            # grammar, so every value here is a bare declaration.
            constraints=normalize_constraints(
                constraints=cast(
                    "Mapping[FunctionName, UserFunction]", user_regime.constraints
                )
            ),
            proved_post_decision=proved_post_decision_of(solver=self),
        )
        if unenforceable:
            constraint_names = sorted(unenforceable)
            msg = (
                f"NBEGM regime '{context.regime_name}' declares constraints "
                f"{constraint_names}. The case-piece kernel evaluates no user "
                "constraint; encode the borrowing limit in the first node of "
                "`savings_grid` and the budget identity in the post-decision "
                "function, or use GridSearch."
            )
            raise ModelInitializationError(msg)

    def validate_build(self, *, context: SolverBuildContext) -> None:
        """Check case coverage and reject hidden branching in user pieces.

        Collecting the metadata enforces strict coverage (each split output has a
        `when` and an `otherwise` piece, every boundary declares a surface). Two
        complementary gates then run on the user pieces:

        - AST: rejects Python branching / hidden comparisons in a smooth piece and
          any non-comparison branching in the boundary predicate.
        - JAXPR: traces each smooth piece and rejects piecewise primitives
          (`select_n`, `lt`, …) hidden inside a called helper that the AST cannot
          see. A piece attested with `lcm.smooth_helper` is exempt.

        The boundary predicate is meant to compare, so only the AST gate runs on
        it; the JAXPR gate runs on the smooth pieces alone. Declared EV1 taste
        shocks are refused here too — the kernels solve the hard maximum.
        """
        bound = cast("_BoundNBEGM", self)
        fail_if_taste_shocks_declared(context=context)
        validate_case_piece_smoothness(
            context=context,
            liquid_state_name=resolve_liquid_state_name(
                context=context, declared=bound.continuous_state
            ),
            probe_failure=self.probe_failure,
        )

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one case-piece EGM period adapter per active period."""
        from _lcm.egm.nbegm import collect_nbegm_metadata  # noqa: PLC0415

        bound = cast("_BoundNBEGM", self)
        savings_grid = self.savings_grid.to_jax()

        functions = cast(
            "Mapping[FunctionName, Callable[..., object]]",
            context.user_regimes[context.regime_name].functions,
        )
        registry = collect_nbegm_metadata(functions=functions)
        has_discrete = bool(context.state_action_space.discrete_actions)
        has_ride_along = self._schedule_has_ride_along(context=context)
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
                context=context,
                budget_target=bound.budget_target,
                continuous_state=bound.continuous_state,
                post_decision_function=bound.post_decision_function,
            )
            if is_schedule_discrete
            else None
        )
        schedule_spec = (
            _collect_nbegm_schedule_spec(
                context=context,
                budget_target=bound.budget_target,
                continuous_state=bound.continuous_state,
                consumption_action_name=bound.continuous_action,
                probe_failure=self.probe_failure,
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
            )

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
            context=context,
            liquid_state_name=liquid_state_name,
            post_decision_name=post_decision_name,
        )
        liquid_grid = context.grids[liquid_state_name].to_jax()
        discrete_spec = (
            _collect_nbegm_discrete_spec(
                context=context,
                budget_target=bound.budget_target,
                post_decision_function=bound.post_decision_function,
                continuous_state=bound.continuous_state,
            )
            if is_discrete
            else None
        )
        case_spec = (
            _collect_nbegm_case_spec(
                context=context, continuous_state=bound.continuous_state
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

        period_to_target = _period_to_continuation_target(context=context)
        cores: dict[RegimeName, Callable] = {}
        laws: dict[RegimeName, Callable[..., tuple[Float1D, Float1D]]] = {}
        period_kernels: dict[int, PeriodKernel] = {}
        consumption_action = bound.continuous_action
        variable_names = (
            frozenset(context.state_action_space.states)
            | frozenset(context.state_action_space.continuous_actions)
            | frozenset(context.state_action_space.discrete_actions)
        )
        for period, target in period_to_target.items():
            if target not in cores:
                laws[target] = build_declared_liquid_law(
                    transitions=context.transitions,
                    functions=context.functions,
                    post_decision_name=post_decision_name,
                    target=target,
                    target_state=liquid_state_name,
                    variable_names=variable_names,
                )
                if schedule_discrete_spec is not None:
                    core = _build_nbegm_schedule_discrete_core(
                        savings_grid=savings_grid,
                        functions=context.functions,
                        consumption_action=consumption_action,
                        spec=schedule_discrete_spec,
                        taste_shock_scale=0.0,
                        envelope_arithmetic=self.envelope_arithmetic,
                    )
                elif schedule_spec is not None:
                    core = _build_nbegm_continuous_core(
                        savings_grid=savings_grid,
                        functions=context.functions,
                        consumption_action=consumption_action,
                        schedule_spec=schedule_spec,
                        envelope_arithmetic=self.envelope_arithmetic,
                    )
                elif discrete_spec is not None:
                    core = _build_nbegm_discrete_core(
                        savings_grid=savings_grid,
                        functions=context.functions,
                        consumption_action=consumption_action,
                        discrete_spec=discrete_spec,
                        taste_shock_scale=0.0,
                        envelope_arithmetic=self.envelope_arithmetic,
                    )
                else:
                    if case_spec is None:
                        msg = (
                            f"Regime {context.regime_name!r} declares neither case "
                            "pieces, a piecewise-affine schedule, nor a discrete "
                            "action, so NBEGM has no kernel to build for it. "
                            "Declare one of them, or use `GridSearch` for this "
                            "regime."
                        )
                        raise RegimeInitializationError(msg)
                    core = _build_nbegm_core(
                        savings_grid=savings_grid,
                        functions=context.functions,
                        consumption_action=consumption_action,
                        case_spec=case_spec,
                        envelope_arithmetic=self.envelope_arithmetic,
                    )
                cores[target] = jax.jit(core) if context.enable_jit else core
            period_kernels[period] = _EGMPeriodKernel(
                core=cores[target],
                declared_law=laws[target],
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
            )
        return SolutionKernels(
            period_kernels=MappingProxyType(period_kernels),
            continuation_spec=EGMContinuationSpec(
                template=_build_one_asset_carry_template(liquid_grid=liquid_grid),
                layout=self.egm_continuation_layout,
            ),
            param_checks=(
                schedule_spec.param_checks if schedule_spec is not None else ()
            ),
        )

    def _fail_if_unsupported_ride_discrete(
        self, *, context: SolverBuildContext, schedule_spec: _NBEGMScheduleSpec
    ) -> None:
        """Reject a ride-along discrete action the envelope path cannot handle.

        The ride-along discrete envelope solves the continuous subproblem per
        discrete branch — with the action bound into the budget, period utility,
        continuation (co-state laws, off-budget liquid law, regime transition),
        and the breakpoint partition — and takes the upper envelope. Outside that
        contract, model build refuses:

        - an action entering the discount factor (evaluated per cell, not per
          branch),
        - an action entering a *jumped* schedule variable under the one-sided
          cliff read (branches would not share the published parent query grid).

        Several discrete actions are supported: the branch axis is the product of
        their grids, and every branch binds a code for each. A jump breakpoint is
        likewise supported — each branch publishes its one-sided cliff limits and
        the envelope takes the max over branches.
        """
        import inspect  # noqa: PLC0415

        bound = cast("_BoundNBEGM", self)
        action_names = tuple(context.state_action_space.discrete_actions)
        # The discount factor is evaluated once per cell in the envelope core, not
        # once per branch, so an action-dependent discount factor would silently
        # use one branch's weight for all branches — refuse it.
        discount_factor_dag = schedule_spec.discount_factor_dag
        discount_args = (
            frozenset(inspect.signature(discount_factor_dag).parameters)
            if discount_factor_dag is not None
            else frozenset()
        )
        for action_name in action_names:
            if action_name in discount_args:
                msg = (
                    "NBEGM's schedule+ride-along discrete envelope evaluates the "
                    "discount factor per cell, not per discrete branch, so the "
                    f"action {action_name!r} must not enter the discount factor; "
                    f"regime {context.regime_name!r} reads it there."
                )
                raise RegimeInitializationError(msg)
            # The envelope binds each action into every branch's period utility, so
            # an action the utility reads is supported (a leisure/effort-like term).
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

    def _build_ride_along_kernels(
        self,
        *,
        context: SolverBuildContext,
        savings_grid: Float1D,
        schedule_spec: _NBEGMScheduleSpec,
    ) -> SolutionKernels:
        """Build the case-piece kernels for a regime carrying a ride-along co-state.

        The continuation is read through the transition-aware reader, so each
        period's plan depends on its reachable carry/scalar target split; cores
        are deduplicated by that split. The 1-D liquid solve runs once per
        ride-along cell, batched.
        """
        bound = cast("_BoundNBEGM", self)

        liquid_grid = context.grids[schedule_spec.liquid_state_name].to_jax()
        ride_shape = tuple(
            int(context.grids[name].to_jax().shape[0])
            for name in schedule_spec.ride_along_state_names
        )
        probe_arguments = _probe_arguments(context=context)
        param_checks: list[ParamCheck] = list(schedule_spec.param_checks)
        if _aggregates_nonlinearly(context.certainty_equivalent):
            param_checks.append(
                _deferred_probe(
                    _fail_if_flow_not_single_power,
                    regime_name=context.regime_name,
                    probe_arguments=probe_arguments,
                    utility_dag=schedule_spec.utility_dag,
                    consumption_action_name=bound.continuous_action,
                    probe_failure=self.probe_failure,
                )
            )
        transition_target_names = tuple(context.transitions)

        # The ride-along kernel takes the continuation as a probability-weighted
        # blend over the full reachable target set (`bind_continuation` sums the
        # per-target carries by `compute_regime_transition_probs`), so it admits a
        # stochastic multi-target lifecycle transition. Enumerate the regime's
        # active periods directly rather than resolving a single target per period.
        active_periods = sorted(context.regimes_to_active_periods[context.regime_name])
        continuation_cores: dict[_RideAlongGroupKey, Callable] = {}
        envelope_cores: dict[_RideAlongGroupKey, Callable] = {}
        statics_by_key: dict[_RideAlongGroupKey, _NBEGMRideAlongStatics] = {}
        cliff_candidates_by_key: dict[_RideAlongGroupKey, bool] = {}
        period_kernels: dict[int, PeriodKernel] = {}
        for period in active_periods:
            plan = _build_nbegm_continuation_plan(
                context=context,
                period=period,
                post_decision_name=bound.post_decision_function,
                stochastic_node_batch_size=self.stochastic_node_batch_size,
            )
            # One compiled core carries one set of continuation nodes, so periods
            # whose targets sit on different age-specialized grids must not share
            # it. The signature is empty for an age-invariant model, leaving the
            # grouping exactly as the target split alone would make it.
            key = (
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
            )
            if key not in continuation_cores:
                param_checks.append(
                    _deferred_probe(
                        _fail_if_liquid_reading_next_state_varies_within_interval,
                        regime_name=context.regime_name,
                        probe_arguments=probe_arguments,
                        continuation_plan=plan,
                        liquid_name=schedule_spec.liquid_state_name,
                        probe_failure=self.probe_failure,
                    )
                )
                statics = _nbegm_ride_along_statics(
                    savings_grid=savings_grid,
                    schedule_spec=schedule_spec,
                    continuation_plan=plan,
                    envelope_segment_block_size=self.envelope_segment_block_size,
                    envelope_arithmetic=self.envelope_arithmetic,
                    cell_block_size=self.cell_block_size,
                    interval_batch_size=self.interval_batch_size,
                    branch_batch_size=self.branch_batch_size,
                    publish_jump_topology=self.jump_read == "one_sided",
                    co_map_state_names=context.co_map_state_names,
                )
                # The Epstein-Zin kernels cover smooth and pure-kink budgets;
                # the unified jump-and-kink candidate step is additive. Reject
                # the combination here, at model build, rather than midway
                # through a traced solve.
                if (
                    _aggregates_nonlinearly(context.certainty_equivalent)
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
                    _aggregates_nonlinearly(context.certainty_equivalent)
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
                continuation_core = _build_nbegm_continuation_core(
                    savings_grid=savings_grid,
                    continuation_plan=plan,
                    statics=statics,
                    regime_name=context.regime_name,
                    cliff_candidates=cliff_candidates,
                    schedule_spec=schedule_spec,
                )
                envelope_core = _build_nbegm_envelope_core(
                    savings_grid=savings_grid,
                    schedule_spec=schedule_spec,
                    statics=statics,
                    is_epstein_zin=_aggregates_nonlinearly(
                        context.certainty_equivalent
                    ),
                )
                continuation_cores[key] = (
                    jax.jit(continuation_core)
                    if context.enable_jit
                    else continuation_core
                )
                envelope_cores[key] = (
                    jax.jit(envelope_core) if context.enable_jit else envelope_core
                )
                statics_by_key[key] = statics
                cliff_candidates_by_key[key] = cliff_candidates
            period_kernels[period] = _RideAlongNBEGMPeriodKernel(
                continuation_core=continuation_cores[key],
                envelope_core=envelope_cores[key],
                statics=statics_by_key[key],
                cliff_candidates=cliff_candidates_by_key[key],
                regime_name=context.regime_name,
                stateful_targets=frozenset(plan.stateful_targets),
                transition_target_names=transition_target_names,
            )
        return SolutionKernels(
            period_kernels=MappingProxyType(period_kernels),
            continuation_spec=EGMContinuationSpec(
                template=_shard_ride_carry_template(
                    template=_build_ride_along_carry_template(
                        liquid_grid=liquid_grid,
                        ride_shape=ride_shape,
                        n_breakpoints=(
                            next(iter(statics_by_key.values())).n_published_jumps
                            if statics_by_key
                            else 0
                        ),
                    ),
                    grids=context.grids,
                    ride_along_state_names=schedule_spec.ride_along_state_names,
                ),
                layout=self.egm_continuation_layout,
            ),
            param_checks=tuple(param_checks),
        )


@dataclass(frozen=True, kw_only=True)
class _BoundNBEGM(NBEGM):
    """Internal NB-EGM configuration with regime-resolved DAG role names."""

    continuous_state: StateName
    continuous_action: ActionName
    budget_target: FunctionName
    post_decision_function: FunctionName


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
class _RideAlongNBEGMPeriodKernel:
    """The case-piece EGM adapter for a regime carrying a ride-along co-state.

    The solve splits into two independently-jitted cores so neither XLA program
    carries the other's instruction graph:

    - `continuation`: reads `next_regime_to_continuation` and binds one continuation per
      ride-along cell through the transition-aware reader, returning the
      probability-weighted expected value and marginal over the savings grid.
    - `envelope`: re-derives each cell's budget and utility and solves the 1-D liquid
      step against the continuation core's stacks, returning the value array and the
      ride-along-axis-leading continuation carry a parent interpolates.

    Calling the adapter runs `continuation` then `envelope` unjitted and assembles the
    `KernelResult`; no JIT spans the two calls.
    """

    continuation_core: Callable
    """The jitted continuation half (`id`-deduped across periods)."""

    envelope_core: Callable
    """The jitted EGM/envelope half (`id`-deduped across periods)."""

    statics: _NBEGMRideAlongStatics
    """Build-time config — supplies the envelope core's placeholder stack shapes."""

    cliff_candidates: bool
    """Whether this period's cores exchange save-to-cliff candidate columns.

    True only when the carry publishes jump topology and the period's targets
    include the regime itself (the cliffs are the self-schedule's).
    """

    regime_name: RegimeName
    """Name of the regime whose flat params this adapter projects."""

    stateful_targets: frozenset[RegimeName]
    """Carry keys this period's core reads; the rolling carry is filtered to these."""

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    @property
    def core(self) -> Callable:
        """The continuation core, exposed for any single-core reader."""
        return self.continuation_core

    def cores(self) -> Mapping[str, Callable]:
        """Return the continuation and envelope cores under their own keys."""
        return MappingProxyType(
            {
                "continuation": self.continuation_core,
                "envelope": self.envelope_core,
            }
        )

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> _RideAlongNBEGMPeriodKernel:
        """Bind the regime's and its carry targets' fixed params into both cores."""
        bound = _union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=self.regime_name,
            transition_target_names=self.transition_target_names,
        )
        if not bound:
            return self
        return replace(
            self,
            continuation_core=functools.partial(self.continuation_core, **bound),
            envelope_core=functools.partial(self.envelope_core, **bound),
        )

    def build_lower_args(
        self,
        *,
        core_key: str = "continuation",
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> Mapping[str, object]:
        """Build the named core's lowering arguments.

        The continuation core takes the state grids, the filtered carries, and the
        regime's flat params. The envelope core takes the same state and param args
        minus the carries, plus correctly-shaped zero placeholders for the two
        continuation stacks (statically derivable from the ride-along grid sizes, the
        savings grid, and the interval count).
        """
        states = dict(state_action_space.states)
        params = self._kernel_params(flat_params=flat_params)
        if core_key == "envelope":
            return {
                **states,
                **self._stack_placeholders(states=states),
                **params,
                "period": jnp.int32(period),
                "age": ages.values[period],
            }
        return {
            **states,
            "next_regime_to_continuation": _carry_subset(
                next_regime_to_continuation=next_regime_to_continuation,
                stateful_targets=self.stateful_targets,
            ),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **params,
            "period": jnp.int32(period),
            "age": ages.values[period],
        }

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
    ) -> KernelResult:
        """Run the continuation then envelope core and assemble the `KernelResult`."""
        states = dict(state_action_space.states)
        params = self._kernel_params(flat_params=flat_params)
        continuation_stacks = compiled_cores["continuation"](
            **states,
            next_regime_to_continuation=_carry_subset(
                next_regime_to_continuation=next_regime_to_continuation,
                stateful_targets=self.stateful_targets,
            ),
            next_regime_to_V_arr=next_regime_to_V_arr,
            **params,
            period=jnp.int32(period),
            age=ages.values[period],
        )
        if self.cliff_candidates:
            cont_value_stack, cont_marginal_stack, cliff_stack = continuation_stacks
            cliff_kwargs = {"cliff_savings_stack": cliff_stack}
        else:
            cont_value_stack, cont_marginal_stack = continuation_stacks
            cliff_kwargs = {}
        V_arr, carry = compiled_cores["envelope"](
            **states,
            cont_value_stack=cont_value_stack,
            cont_marginal_stack=cont_marginal_stack,
            **cliff_kwargs,
            **params,
            period=jnp.int32(period),
            age=ages.values[period],
        )
        return KernelResult(V_arr=V_arr, continuation=carry)

    def _stack_placeholders(self, *, states: Mapping[str, object]) -> dict[str, object]:
        """Zero placeholders for the envelope core's continuation stacks.

        The interval regime reads one continuation row per declared interval, so the
        stacks carry an interval axis between the ride-cell and savings axes; the
        non-interval regime reads a single row over the savings grid. With co-mapped
        (distributed) ride states, the placeholders are committed to the ride-cell
        sharding the continuation core's runtime stacks arrive with — an uncommitted
        placeholder would leave the compiled-for input sharding to backend-specific
        propagation, which can compile the core for replicated stacks and reject
        every runtime call.
        """
        n_ride_cells = self.statics.n_ride_cells(states=states)
        n_extra = 2 * self.statics.n_published_jumps if self.cliff_candidates else 0
        n_savings = self.statics.n_savings + n_extra
        # A discrete action carries a leading branch axis on each cell's continuation
        # (branch `pos` reads slice `pos`); a branch-free regime keeps the plain shape.
        branch_axis: tuple[int, ...] = (
            ()
            if self.statics.n_action_branches == 0
            else (self.statics.n_action_branches,)
        )
        interval_axis: tuple[int, ...] = (
            (self.statics.n_intervals,)
            if self.statics.continuation_reads_liquid
            else ()
        )
        sharding = self._co_map_stack_sharding(states=states)
        zeros = jnp.zeros(
            (n_ride_cells, *branch_axis, *interval_axis, n_savings),
            dtype=canonical_float_dtype(),
            device=sharding,
        )
        placeholders: dict[str, object] = {
            "cont_value_stack": zeros,
            "cont_marginal_stack": zeros,
        }
        if self.cliff_candidates:
            placeholders["cliff_savings_stack"] = jnp.zeros(
                (n_ride_cells, *branch_axis, *interval_axis, n_extra),
                dtype=canonical_float_dtype(),
                device=sharding,
            )
        return placeholders

    def _co_map_stack_sharding(
        self, *, states: Mapping[str, object]
    ) -> jax.NamedSharding | None:
        """Sharding for the envelope core's continuation-stack placeholders.

        The co-mapped ride states are a leading prefix of the ride axes, so the
        runtime stacks arrive sharded along the flattened leading ride-cell axis
        (one block per device). Returns `None` when no ride state is distributed,
        keeping the placeholders on the default device.
        """
        co_map_state_names = self.statics.co_map_state_names
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

    def _kernel_params(self, *, flat_params: FlatParams) -> dict[str, object]:
        """Flat params fed into the cores: the regime's plus its targets'."""
        return _union_free_params(
            flat_params=flat_params,
            regime_name=self.regime_name,
            transition_target_names=self.transition_target_names,
        )


@dataclass(frozen=True, kw_only=True)
class _NBEGMCaseSpec:
    """Build-time statics describing one binary case split."""

    when_callable: Callable
    """The `when` piece — its contribution applies where the predicate holds."""
    otherwise_callable: Callable
    """The `otherwise` piece — its contribution applies where the predicate fails."""
    when_func: FunctionName
    """Qualified-name prefix of the `when` piece's params."""
    otherwise_func: FunctionName
    """Qualified-name prefix of the `otherwise` piece's params."""
    when_param_names: tuple[str, ...]
    """Parameter names of the `when` piece."""
    otherwise_param_names: tuple[str, ...]
    """Parameter names of the `otherwise` piece."""
    predicate_name: FunctionName
    """Qualified-name prefix of the boundary predicate's params."""
    threshold_name: str
    """Name of the predicate's threshold parameter."""
    equality_owner: EqualityOwner
    """Predicate side owning the exact-boundary point (`when` or `otherwise`)."""


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


def validate_case_piece_smoothness(
    *,
    context: SolverBuildContext,
    liquid_state_name: StateName,
    probe_failure: Literal["reject", "assume_declared"] = "reject",
) -> None:
    """Check case coverage and reject hidden branching in a regime's pieces.

    Shared by every solver that runs the case-piece EGM kernels on a liquid
    margin, so a nested solver applies the same gate to its inner margin as a
    bare `NBEGM` would.

    Args:
        context: The regime's solver build context.
        liquid_state_name: The state the case boundaries are declared on. Named
            explicitly rather than taken as the regime's first state, because a
            nested regime carries an outer margin the pieces never see.
        probe_failure: What to do with a piece the build-time scalar fills
            cannot trace — `"reject"` refuses the build, `"assume_declared"`
            warns and leaves the smoothness claim to the model author.

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

    functions = cast(
        "Mapping[FunctionName, Callable[..., object]]",
        context.user_regimes[context.regime_name].functions,
    )
    registry = collect_nbegm_metadata(functions=functions)
    space = context.state_action_space
    if registry.piece_sets and space.discrete_actions:
        msg = (
            f"Regime '{context.regime_name}' declares case pieces alongside the "
            f"discrete action(s) {sorted(space.discrete_actions)}. The "
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
        liquid_state_name=liquid_state_name,
        reserved_names=frozenset(space.state_names) | frozenset(space.action_names),
    )
    violations: list[str] = []
    for predicate_name in registry.boundaries:
        violations += find_ast_violations(functions[predicate_name], mode="boundary")
    for piece_set in registry.piece_sets:
        for piece_name in (piece_set.when_func, piece_set.otherwise_func):
            piece = functions[piece_name]
            if is_smooth_helper(piece):
                continue
            violations += find_ast_violations(piece, mode="smooth_user")
            n_params = len(inspect.signature(piece).parameters)
            abstract_args = tuple(jnp.asarray(1.0) for _ in range(n_params))
            violations += find_jaxpr_violations(
                piece,
                abstract_args=abstract_args,
                mode="smooth_user",
                probe_failure=probe_failure,
            )
    if violations:
        from lcm.exceptions import NBEGMCaseError  # noqa: PLC0415

        msg = "NBEGM smoothness gate failed:\n" + "\n".join(violations)
        raise NBEGMCaseError(msg)


def _validate_nbegm_boundary_scope(
    *,
    registry: NBEGMRegistry,
    functions: Mapping[FunctionName, Callable[..., object]],
    liquid_state_name: str,
    reserved_names: frozenset[str],
) -> None:
    """Reject case-piece declarations the case-piece kernels cannot solve.

    The case-piece route splits exactly one additive cash-on-hand `subsidy`
    across one jump boundary on the liquid state, owned by the `otherwise` side,
    with pieces that read only the flat params (not states or actions). Anything
    else (a `when`-owned boundary, a continuous kink or hard constraint, a
    boundary on another variable, a non-`subsidy` output, a state-dependent piece)
    is rejected here rather than silently solved under the wrong convention.
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
    for predicate_name, meta in registry.boundaries.items():
        for surface in meta.boundaries:
            if surface.equality_owner != "otherwise":
                msg = (
                    f"NBEGM case boundaries own equality on the "
                    f"`otherwise` side; "
                    f"{predicate_name!r} owns equality on the "
                    f"{surface.equality_owner!r} side."
                )
                raise NBEGMCaseError(msg)
            if surface.kind != "jump":
                msg = (
                    f"NBEGM case boundaries declare `kind='jump'`; "
                    f"{predicate_name!r} declares {surface.kind!r}."
                )
                raise NBEGMCaseError(msg)
            if surface.variable != liquid_state_name:
                msg = (
                    f"NBEGM case boundaries compare the liquid state "
                    f"{liquid_state_name!r}; {predicate_name!r} compares "
                    f"{surface.variable!r}."
                )
                raise NBEGMCaseError(msg)


# The cash-on-hand the case-piece kernels form themselves, rather than calling
# the regime's declared budget node.
_KERNEL_BUDGET_NODES = frozenset({cash_on_hand_with_subsidy.__name__})


def _declares_fixed_form(func: object, *, allowed: frozenset[str]) -> bool:
    """Report whether `func` is one of `allowed`, seen through the DAG's wrappers.

    The declaration reaches the solver wrapped, so the marker is followed along
    the `__wrapped__` chain rather than compared by identity.
    """
    seen: object | None = func
    while seen is not None:
        if getattr(seen, FIXED_FORM_ATTRIBUTE, None) in allowed:
            return True
        seen = getattr(seen, "__wrapped__", None)
    return False


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
    budget_target: str,
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
    budget_node = context.user_regimes[regime_name].functions.get(budget_target)
    if not routes_to_case_piece_core or budget_node is None:
        return
    if isinstance(budget_node, Phased):
        msg = (
            f"NBEGM's case-piece kernels form cash-on-hand as "
            f"`{liquid_state_name} + {split_output}` in both phases, so regime "
            f"{regime_name!r} cannot give {budget_target!r} a phase-dependent "
            "form. Declare a `lcm.piecewise_affine` schedule with a "
            "`post_decision_function` so the budget is composed from the DAG, or "
            "use `GridSearch` for this regime."
        )
        raise RegimeInitializationError(msg)
    if not _declares_fixed_form(budget_node, allowed=_KERNEL_BUDGET_NODES):
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


def _flat_params(func: Callable[..., object]) -> frozenset[str]:
    """Return the qualified flat parameters a function reads.

    States, actions, and other DAG nodes reach a function under their bare
    names; a flat parameter always arrives qualified by the function that owns
    it, so the qualifying separator is what tells the two apart.
    """
    return frozenset(name for name in _parameter_names(func) if "__" in name)


def _parameter_names(func: Callable[..., object]) -> frozenset[str]:
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

    functions = cast(
        "Mapping[FunctionName, Callable[..., object]]",
        context.user_regimes[context.regime_name].functions,
    )
    registry = collect_nbegm_metadata(functions=functions)
    if len(registry.piece_sets) != 1:
        msg = (
            "NBEGM case pieces split exactly one output; the regime declares "
            f"{len(registry.piece_sets)}."
        )
        raise RegimeInitializationError(msg)
    piece_set = registry.piece_sets[0]
    surfaces = registry.boundaries[piece_set.predicate_name].boundaries
    if len(surfaces) != 1:
        msg = (
            "NBEGM case boundaries declare exactly one surface; the predicate "
            f"{piece_set.predicate_name!r} declares {len(surfaces)}."
        )
        raise RegimeInitializationError(msg)
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
    when_callable = functions[piece_set.when_func]
    otherwise_callable = functions[piece_set.otherwise_func]
    return _NBEGMCaseSpec(
        when_callable=when_callable,
        otherwise_callable=otherwise_callable,
        when_func=piece_set.when_func,
        otherwise_func=piece_set.otherwise_func,
        when_param_names=tuple(inspect.signature(when_callable).parameters),
        otherwise_param_names=tuple(inspect.signature(otherwise_callable).parameters),
        predicate_name=piece_set.predicate_name,
        threshold_name=surfaces[0].threshold,
        equality_owner=surfaces[0].equality_owner,
    )


def _build_nbegm_core(
    *,
    savings_grid: Float1D,
    functions: EconFunctionsMapping,
    consumption_action: ActionName,
    case_spec: _NBEGMCaseSpec,
    envelope_arithmetic: ComparisonArithmetic = "certified",
) -> Callable:
    """Build the jittable case-piece EGM core closing over the case split.

    The core evaluates each piece's additive contribution and the boundary
    threshold from the regime's flat params, runs the two-case EGM merge, and
    returns the value array and the marginal-value carry on the liquid grid.
    """
    from _lcm.egm.nbegm_step import nbegm_one_asset_step  # noqa: PLC0415
    from _lcm.egm.preferences import (  # noqa: PLC0415
        NEWTON_ACTION_FLOOR,
        get_preferences_builder,
        newton_action_ceiling,
    )

    build_preferences = get_preferences_builder(
        functions=functions,
        action_name=consumption_action,
        action_lower=NEWTON_ACTION_FLOOR,
        action_upper=newton_action_ceiling(savings_grid),
    )

    def core(
        *,
        liquid: Float1D,
        next_liquid_grid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        next_liquid: Float1D,
        marginal_return: Float1D,
        **params: FloatND,
    ) -> tuple[Float1D, EGMCarry]:
        preferences = build_preferences(params)
        subsidy_when = case_spec.when_callable(
            **{
                p: params[f"{case_spec.when_func}__{p}"]
                for p in case_spec.when_param_names
            }
        )
        subsidy_otherwise = case_spec.otherwise_callable(
            **{
                p: params[f"{case_spec.otherwise_func}__{p}"]
                for p in case_spec.otherwise_param_names
            }
        )
        asset_limit = params[f"{case_spec.predicate_name}__{case_spec.threshold_name}"]
        value, marginal, _policy = nbegm_one_asset_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            next_liquid_grid=next_liquid_grid,
            savings_grid=savings_grid,
            discount_factor=params["koopmans_aggregator__discount_factor"],
            preferences=preferences,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            subsidy_when=subsidy_when,
            subsidy_otherwise=subsidy_otherwise,
            asset_limit=asset_limit,
            equality_owner=case_spec.equality_owner,
            arithmetic=envelope_arithmetic,
        )
        carry = EGMCarry(
            endog_grid=liquid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(0.0, dtype=value.dtype),
        )
        return value, carry

    return core


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
    threshold_param_name: str
    """Qualified parameter name of this breakpoint's threshold."""
    kind: str
    """Discontinuity kind: `continuous_kink`, `jump`, or `hard_constraint`."""
    derived_of_liquid_dag: Callable | None
    """Composed schedule variable as a function of the liquid state, or `None`
    when the schedule varies in the liquid state directly (no preimage needed)."""
    derived_param_names: tuple[str, ...]
    """Unqualified parameter names the schedule variable reads (non-state args)."""
    derived_state_names: tuple[str, ...] = ()
    """Ride-along state names the schedule variable reads, so the per-cell call
    passes only the cell entries the derived DAG accepts."""
    threshold_index_state: str | None = None
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

    coh_of_liquid_dag: Callable
    """Composed `coh` as a function of the liquid state and qualified params."""
    coh_param_names: tuple[str, ...]
    """Qualified parameter names `coh` reads (everything but the state axes)."""
    utility_dag: Callable
    """Composed period utility as a function of the consumption action, the
    ride-along states it reads, and qualified utility params. The ride-along core
    binds it per cell to invert the Euler equation and evaluate the period value."""
    consumption_action_name: ActionName
    """Name of the continuous consumption action the period utility reads."""
    liquid_state_name: str
    """Name of the liquid state the schedule and budget vary in."""
    ride_along_state_names: tuple[str, ...]
    """State axes other than the liquid axis (the budget varies per ride-along cell)."""
    liquid_axis_pos: int
    """Index of the liquid axis in the canonical productmap state order. The
    ride-along core solves in working layout (ride axes leading the liquid axis)
    and moves the liquid axis to this position so the published value array follows
    the productmap order — a no-op when every ride-along axis is a discrete state
    sorting ahead of the liquid axis, a genuine transpose for a continuous co-state
    declared after it."""
    threshold_param_names: tuple[str, ...]
    """Qualified parameter names of the schedule's thresholds."""
    breakpoint_kinds: tuple[str, ...]
    """Discontinuity kind per threshold, in the schedule's declared order."""
    sources: tuple[_NBEGMSource, ...] = ()
    """Every breakpoint across all declared schedules, merged on the liquid axis.
    The ride-along core maps each source to its own per-cell asset preimage."""
    discount_factor_dag: Callable | None = None
    """Composed `discount_factor` as a function of its ride-along state arguments and
    qualified params, or `None` when the regime uses pylcm's flat
    `koopmans_aggregator__discount_factor` parameter. When set, the ride-along
    core resolves the discount factor per cell."""
    discrete_actions: DiscreteActionCodes = ()
    """Each discrete action the budget shifts, paired with its grid codes, empty
    when the regime carries no discrete action. Excluded from `coh_param_names` —
    the envelope core binds them per branch."""
    param_checks: tuple[ParamCheck, ...] = ()
    """Preconditions on the composed budget, run on the first solve. Collected
    here because the budget is composed while the spec is, and checking its
    affinity in the liquid state needs the schedules' parameter values."""

    @property
    def branch_bindings(self) -> tuple[MappingProxyType[ActionName, int], ...]:
        """One binding of every declared discrete action, per envelope branch."""
        return _branch_bindings(self.discrete_actions)


def _fail_if_discrete_action_feeds_continuation(
    *,
    context: SolverBuildContext,
    action_name: str,
    liquid_state_name: str,
    budget_target: str,
    post_decision_function: str | None,
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

    def _reject(where: str) -> None:
        msg = (
            f"NBEGM's discrete envelope shares one continuation across the "
            f"branches of {action_name!r}, so the action may shift only the "
            f"current budget and utility; regime {context.regime_name!r} reads it "
            f"in {where}. Fix the action there, or use a solver that carries a "
            "branch-specific continuation."
        )
        raise RegimeInitializationError(msg)

    transition_probs = context.compute_regime_transition_probs
    if (
        not allow_continuation_feed
        and transition_probs is not None
        and action_name in inspect.signature(transition_probs).parameters
    ):
        _reject("the regime transition")

    regime = context.user_regimes[context.regime_name]
    funcs: dict[str, Callable[..., object]] = {
        name: func for name, func in regime.functions.items() if callable(func)
    }
    budget_nodes = {budget_target, post_decision_function}

    def _law_reads_action(law: Callable[..., object], *, cut_budget: bool) -> bool:
        # For the liquid law, drop the budget nodes so the action reaches the law
        # only through an off-budget path (an out-of-pocket cost on next assets).
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

    # The branch-indexed continuation reads each branch's own next-state
    # coordinate, so on the ride-along path every state-law feed is supported.
    if allow_continuation_feed:
        return

    for state_name, func in _state_laws(transitions=context.transitions):
        is_liquid = state_name == liquid_state_name
        if _law_reads_action(func, cut_budget=is_liquid):
            where = (
                f"the law of motion for {state_name!r} off the budget channel"
                if is_liquid
                else f"the law of motion for {state_name!r}"
            )
            _reject(where)


def _state_laws(
    *, transitions: TransitionFunctionsMapping
) -> Iterator[tuple[StateName, Callable[..., object]]]:
    """Yield each canonical state law with the state it evolves.

    Reads the canonical per-target laws rather than a regime's declared
    `state_transitions`: which targets a regime reaches is the graph's to say,
    and the canonical form has already broadcast bare laws over exactly those
    targets. The regime transition and the stochastic weight laws carry no
    state coordinate, so neither is yielded.
    """
    from lcm.transition import MarkovTransition  # noqa: PLC0415

    for target_laws in transitions.values():
        for law_name, candidate in target_laws.items():
            if law_name.startswith("weight_"):
                continue
            state_name = law_name.removeprefix("next_")
            if state_name == "regime":
                continue
            func = (
                candidate.func if isinstance(candidate, MarkovTransition) else candidate
            )
            if callable(func):
                yield state_name, func


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
    cont_value: FloatND,
    cont_marginal: FloatND,
    extra_cont_value: FloatND | None,
    cliff_savings: FloatND | None,
) -> dict[str, Any]:
    """Assemble the pytree `lax.map` streams over the branch axis.

    The optional continuations enter only where the regime supplies them, so a
    regime without child cliffs or an extra continuation maps a narrower tree
    rather than one padded with sentinels that every branch would then carry.
    """
    inputs: dict[str, Any] = {
        "codes": codes,
        "cont_value": cont_value,
        "cont_marginal": cont_marginal,
    }
    if extra_cont_value is not None:
        inputs["extra_cont_value"] = extra_cont_value
    if cliff_savings is not None:
        inputs["cliff_savings"] = cliff_savings
    return inputs


def _branch_bindings(
    discrete_actions: DiscreteActionCodes,
) -> tuple[MappingProxyType[ActionName, int], ...]:
    """Bind every combination of the declared discrete actions, in envelope order.

    The envelope's branch axis is the *product* of the declared grids: a regime
    declaring a binary and a five-valued action carries ten branches, each a
    complete assignment of a code to every action. A regime declaring none
    carries a single empty binding, so the branch axis is never degenerate and
    callers need no separate no-action case.

    The order is the contract between the two independently-jitted cores: the
    continuation core stacks one slice per branch and the envelope core reads
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


def _fail_if_budget_nonaffine_in_liquid(  # noqa: C901
    *,
    coh_dag: Callable[..., object],
    liquid_name: str,
    require_unit_slope: bool,
    regime_name: str,
    probe_arguments: _ProbeArguments,
    probe_failure: Literal["reject", "assume_declared"] = "reject",
) -> None:
    """Reject a budget that is not affine in the liquid state within an interval.

    NBEGM recovers each interval's budget from its slope and value at one interior
    point (`interval_segment_coefficients`), exact only when the composed budget is
    affine in the liquid state on the interval — a smooth nonlinear budget would be
    mis-tangented at every other point. A declared jump / kink is a *selection*
    between affine branches, so its second derivative in the liquid state stays zero
    at interior points; a genuine nonlinearity (a square, a product of the liquid
    state with itself, a reciprocal) shows a nonzero second derivative.

    With `require_unit_slope` — the liquid-direct, non-ride, all-jump path solved by
    the pure-jump step, which reads only per-interval intercepts — the budget must
    additionally have unit slope in the liquid state: a non-unit affine slope
    declared as jump-only would be solved as if `coh = liquid + intercept`.

    The probe evaluates the composed budget's first and second liquid-derivatives
    at a few interior points, with every declared parameter set to its own value —
    the real tax schedules and tables, so the real bracket structure is what gets
    differentiated — and every remaining argument filled by two constant sets so a
    parameter-dependent slope is caught; each integer-coded argument is additionally
    swept over its grid's actual codes one at a time. The probe is a finite
    diagnostic, not a certificate — a nonlinearity whose curvature vanishes at every
    probed point passes undetected. A budget the probe cannot differentiate is
    refused: the per-interval inversion's affinity precondition would otherwise go
    unverified.
    """
    import inspect  # noqa: PLC0415

    arg_names = tuple(inspect.signature(coh_dag).parameters)
    if liquid_name not in arg_names:
        return

    def _budget_of_liquid(
        liquid_value: FloatND,
        fill: float,
        *,
        array_floats: bool = False,
        array_rank: int = 1,
        leaf_rank: int = 1,
        int_overrides: Mapping[str, int] = MappingProxyType({}),
    ) -> FloatND:
        kwargs = {
            name: (
                liquid_value
                if name == liquid_name
                else jnp.asarray(int_overrides[name], dtype=jnp.int32)
                if name in int_overrides
                else probe_arguments.fill(
                    name,
                    fill,
                    array_floats=array_floats,
                    array_rank=array_rank,
                    leaf_rank=leaf_rank,
                )
            )
            for name in arg_names
        }
        return jnp.asarray(coh_dag(**kwargs)).reshape(())

    tol = 1e-6

    def _fail_unprobeable(probe_error: Exception) -> None:
        msg = (
            f"NBEGM could not verify that regime {regime_name!r}'s budget is affine "
            f"in the liquid state {liquid_name!r}: the probe failed to "
            "differentiate the budget on scalar inputs "
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

    def _max_abs_second() -> float | None:
        def _values(
            *, array_floats: bool, array_rank: int, leaf_rank: int
        ) -> list[float]:
            return [
                abs(
                    float(
                        jax.grad(
                            jax.grad(
                                lambda a, f=fill, o=overrides: _budget_of_liquid(
                                    a,
                                    f,
                                    array_floats=array_floats,
                                    array_rank=array_rank,
                                    leaf_rank=leaf_rank,
                                    int_overrides=o,
                                )
                            )
                        )(jnp.asarray(sample))
                    )
                )
                for fill in (1.0, 3.0)
                for overrides in _int_code_sweeps(
                    arg_names=arg_names, int_arg_values=probe_arguments.int_arg_values
                )
                for sample in (0.37, 1.63, 2.71)
            ]

        try:
            values = _evaluate_on_first_workable_fill(_values)
        except Exception as probe_error:  # noqa: BLE001
            _fail_unprobeable(probe_error)
            return None
        return max(values)

    def _liquid_slopes() -> tuple[float, ...] | None:
        def _slopes(
            *, array_floats: bool, array_rank: int, leaf_rank: int
        ) -> tuple[float, ...]:
            return tuple(
                float(
                    jax.grad(
                        lambda a, f=fill, o=overrides: _budget_of_liquid(
                            a,
                            f,
                            array_floats=array_floats,
                            array_rank=array_rank,
                            leaf_rank=leaf_rank,
                            int_overrides=o,
                        )
                    )(jnp.asarray(x))
                )
                for fill, x in ((1.0, 1.0), (3.0, 2.0))
                for overrides in _int_code_sweeps(
                    arg_names=arg_names, int_arg_values=probe_arguments.int_arg_values
                )
            )

        try:
            return _evaluate_on_first_workable_fill(_slopes)
        except Exception as probe_error:  # noqa: BLE001
            _fail_unprobeable(probe_error)
            return None

    worst_second = _max_abs_second()
    if worst_second is not None and worst_second > tol:
        msg = (
            f"NBEGM's budget must be affine in the liquid state {liquid_name!r} "
            f"within each interval, but regime {regime_name!r} has a nonzero second "
            "derivative there — a smooth nonlinear budget is not recovered by the "
            "per-interval affine segment. Declare the nonsmoothness as breakpoints or "
            "keep the budget affine per interval."
        )
        raise RegimeInitializationError(msg)

    slopes = _liquid_slopes() if require_unit_slope else None
    offending = (
        next((slope for slope in slopes if abs(slope - 1.0) > tol), None)
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
    utility_dag: Callable[..., object],
    consumption_action_name: str,
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
    each integer-coded argument swept over its actual grid codes. Rejected on the
    first solve:

    - a nonpositive flow or nonpositive marginal at any probed point (the
      recursion takes fractional powers of the flow; `q = -c` carries a
      constant positive elasticity, so signs are checked directly),
    - a varying elasticity (the closed-form inversion needs one global power),
    - a nonpositive elasticity.
    """
    import inspect  # noqa: PLC0415

    arg_names = tuple(inspect.signature(utility_dag).parameters)
    if consumption_action_name not in arg_names:
        return
    probe_consumptions = (0.5, 1.0, 2.0, 5.0)
    fill = 1.7

    def flow_of_consumption(
        consumption: ScalarFloat,
        int_overrides: Mapping[str, int],
        *,
        array_floats: bool,
        array_rank: int,
        leaf_rank: int,
    ) -> ScalarFloat:
        kwargs = {
            name: (
                consumption
                if name == consumption_action_name
                else jnp.asarray(int_overrides[name], dtype=jnp.int32)
                if name in int_overrides
                else probe_arguments.fill(
                    name,
                    fill,
                    array_floats=array_floats,
                    array_rank=array_rank,
                    leaf_rank=leaf_rank,
                )
            )
            for name in arg_names
        }
        return jnp.asarray(utility_dag(**kwargs)).reshape(())

    sweep_names = tuple(
        name for name in arg_names if name in probe_arguments.int_arg_names
    )

    def _readings(
        *, array_floats: bool, array_rank: int, leaf_rank: int
    ) -> _FlowProbeReadings:
        flows: list[float] = []
        marginals: list[float] = []
        elasticities: list[float] = []
        for overrides in _int_code_sweeps(
            arg_names=sweep_names, int_arg_values=probe_arguments.int_arg_values
        ):
            for probe_c in probe_consumptions:
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

    try:
        readings = _evaluate_on_first_workable_fill(_readings)
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
# type aliases plus the grouped-param leaf types — the vocabulary model functions
# annotate their parameters in, and so the whole of what a composed function's
# stringified annotations can name.
_PROBE_ANNOTATION_VOCABULARY: MappingProxyType[str, object] = MappingProxyType(
    {
        **{
            name: value
            for name, value in vars(lcm_typing).items()
            if not name.startswith("_")
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
    annotations, then completed with `with_params` on the first solve. Splitting
    it that way is what lets the probes read the model's real schedules and
    tables: the classification is a property of the source, the values are not
    known until the user supplies params.
    """

    int_arg_values: MappingProxyType[str, tuple[int, ...]] = MappingProxyType({})
    """Grid codes per integer-coded argument, swept one code at a time."""
    array_float_arg_names: frozenset[str] = frozenset()
    """Arguments whose consumers annotate them as float arrays."""
    array_arg_ranks: MappingProxyType[str, int] = MappingProxyType({})
    """Axis count per array argument, read off how its consumers subscript it."""
    annotated_int_arg_names: frozenset[str] = frozenset()
    """Arguments whose consumers annotate them with an integer dtype."""
    bool_arg_names: frozenset[str] = frozenset()
    """Arguments whose consumers annotate them with a boolean dtype."""
    mapping_leaf_arg_names: frozenset[str] = frozenset()
    """Arguments whose consumers annotate them as grouped params."""
    param_values: MappingProxyType[str, object] = MappingProxyType({})
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
        merged: dict[str, object] = {}
        for name, regime_params in flat_params.items():
            if name != regime_name:
                merged.update(regime_params)
        merged.update(flat_params.get(regime_name, MappingProxyType({})))
        return replace(self, param_values=MappingProxyType(merged))

    def fill(
        self,
        name: str,
        fill: float,
        *,
        array_floats: bool = False,
        array_rank: int = 1,
        leaf_rank: int = 1,
    ) -> object:
        """Build one argument at the given fill level and rung."""
        return _probe_fill(
            name,
            fill,
            self.int_arg_names,
            self.array_float_arg_names,
            self.array_arg_ranks,
            bool_arg_names=self.bool_arg_names,
            mapping_leaf_arg_names=self.mapping_leaf_arg_names,
            param_values=self.param_values,
            array_floats=array_floats,
            array_rank=array_rank,
            leaf_rank=leaf_rank,
        )


def _deferred_probe(
    probe: Callable[..., None],
    *,
    regime_name: RegimeName,
    probe_arguments: _ProbeArguments,
    **bound: object,
) -> ParamCheck:
    """Configure a probe at model build and run it on the first solve.

    The probe's target — the composed budget, the utility DAG, the continuation
    plan — is fixed by the model's structure and bound here. Its arguments are
    not: a budget reading tax schedules cannot be differentiated until those
    schedules have values, so the fills are completed from the params the engine
    supplies at solve.
    """

    def _check(*, flat_params: FlatParams) -> None:
        probe(
            regime_name=regime_name,
            probe_arguments=probe_arguments.with_params(
                flat_params=flat_params, regime_name=regime_name
            ),
            **bound,
        )

    return _check


def _probe_arguments(*, context: SolverBuildContext) -> _ProbeArguments:
    """Classify a regime's probe arguments from its grids and annotations."""
    annotation_sources = _probe_annotation_sources(context=context)
    return _ProbeArguments(
        int_arg_values=_int_probe_arg_values(context.grids),
        array_float_arg_names=_array_float_arg_names(functions=annotation_sources),
        array_arg_ranks=_indexed_arg_ranks(functions=annotation_sources),
        annotated_int_arg_names=_annotated_int_arg_names(functions=annotation_sources),
        bool_arg_names=_annotated_bool_arg_names(functions=annotation_sources),
        mapping_leaf_arg_names=_annotated_mapping_leaf_arg_names(
            functions=annotation_sources
        ),
    )


def _probe_fill(
    name: str,
    fill: float,
    int_arg_names: frozenset[str],
    array_float_arg_names: frozenset[str] = frozenset(),
    array_arg_ranks: Mapping[str, int] = MappingProxyType({}),
    *,
    bool_arg_names: frozenset[str] = frozenset(),
    mapping_leaf_arg_names: frozenset[str] = frozenset(),
    param_values: Mapping[str, object] = MappingProxyType({}),
    array_floats: bool = False,
    array_rank: int = 1,
    leaf_rank: int = 1,
) -> object:
    """Build a probe argument, preferring the model's own parameter value.

    A parameter the model declares answers with its own value: the probes run on
    the first solve, so the tax schedules, interpolation tables, and coefficients
    the budget reads are in hand. That is both simpler and stricter than any
    synthetic stand-in — the probe differentiates the real bracket structure
    rather than a fabricated one-bracket schedule.

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
        except Exception as rung_error:  # noqa: BLE001
            errors.append(rung_error)
    raise errors[0]


def _probe_annotation_sources(
    *,
    context: SolverBuildContext,
) -> MappingProxyType[str, Callable[..., object]]:
    """The functions whose signatures classify this regime's probe fills."""
    return _annotation_source_functions(
        functions=context.functions,
        transitions=context.transitions,
        user_regimes=context.user_regimes,
        compute_regime_transition_probs=context.compute_regime_transition_probs,
    )


def _indexed_arg_ranks(
    *,
    functions: Mapping[str, Callable[..., object]],
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


def _resolved_annotation(annotation: object) -> object | None:
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
        annotation = _PROBE_ANNOTATION_VOCABULARY.get(annotation)
        if annotation is None:
            return None
    return getattr(annotation, "__value__", annotation)


def _probe_arg_spellings(*, func_name: str, arg_name: str) -> tuple[str, str]:
    """Both names a classified parameter can reach the probe under.

    Processing qualifies a function's parameters with the function's own name,
    so a classification read off an unprocessed function has to answer to the
    qualified spelling too — the probe fills the composed DAG's leaves, which
    carry whichever of the two that function contributed.
    """
    return (arg_name, f"{func_name}__{arg_name}")


def _annotation_source_functions(
    *,
    functions: Mapping[str, Callable[..., object]],
    transitions: Mapping[str, Mapping[str, object]],
    user_regimes: Mapping[str, object] = MappingProxyType({}),
    compute_regime_transition_probs: Callable[..., object] | None = None,
) -> MappingProxyType[str, Callable[..., object]]:
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
    sources: dict[str, Callable[..., object]] = dict(functions)
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
    functions: Mapping[str, Callable[..., object]],
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
    scalar_args: set[str] = set()
    array_args: set[str] = set()
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
    functions: Mapping[str, Callable[..., object]],
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
    functions: Mapping[str, Callable[..., object]],
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
    functions: Mapping[str, Callable[..., object]],
) -> frozenset[str]:
    """Leaf parameter names declared as grouped param mappings, from annotations.

    A schedule the model author groups under one name — a tax table's brackets,
    rates, and intercepts — reaches the DAG as a single `MappingLeaf` argument
    rather than as separate arrays, so no array fill satisfies it.
    """
    leaf_args: set[str] = set()
    other_args: set[str] = set()
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
    functions: Mapping[str, Callable[..., object]],
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
    arg_names: tuple[str, ...],
    int_arg_values: Mapping[str, tuple[int, ...]],
) -> tuple[MappingProxyType[str, int], ...]:
    """One-at-a-time overrides sweeping each discrete argument's actual codes.

    The first assignment is empty (the plain synthetic fills); each further
    assignment pins one integer-coded argument to one of its grid codes while the
    other arguments keep their fills.
    """
    assignments: list[MappingProxyType[str, int]] = [MappingProxyType({})]
    for name in arg_names:
        codes = int_arg_values.get(name, ())
        assignments.extend(MappingProxyType({name: code}) for code in codes)
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


def _fail_if_liquid_reading_next_state_varies_within_interval(  # noqa: C901
    *,
    continuation_plan: Any,  # noqa: ANN401  # `ContinuationPlan`; import-cycle-safe
    liquid_name: str,
    regime_name: str,
    probe_arguments: _ProbeArguments,
    probe_failure: Literal["reject", "assume_declared"] = "reject",
) -> None:
    """Reject a carried-state law that varies smoothly in the liquid state.

    When a carried state's law of motion reads the current liquid (Euler) state,
    NBEGM binds the liquid state to each interval's node and reuses that
    continuation row across the interval — exact only when the law's liquid
    dependence is piecewise-constant (a level switched at a declared cliff, so its
    derivative in the liquid state is zero between breakpoints). A smooth (affine or
    curved) dependence makes the midpoint-bound row wrong for the interval's other
    liquid points, so it is rejected on the first solve.

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
    import inspect  # noqa: PLC0415

    tol = 1e-6

    def _fill_assignments(n_args: int) -> tuple[tuple[float, ...], ...]:
        constant_1 = tuple(1.0 for _ in range(n_args))
        constant_3 = tuple(3.0 for _ in range(n_args))
        ramp_up = tuple(1.0 + 2.0 * position for position in range(n_args))
        ramp_down = tuple(reversed(ramp_up))
        return (constant_1, constant_3, ramp_up, ramp_down)

    def _max_abs_first_derivative(
        func: Callable[..., object],
    ) -> float | None:
        # The composed law returns the child's whole carried-state vector, so probe
        # the Jacobian in the liquid argument and take the max over all outputs.
        arg_names = tuple(inspect.signature(func).parameters)
        if liquid_name not in arg_names:
            return None
        liquid_pos = arg_names.index(liquid_name)

        # Annotated `object` deliberately: these are the probe's own fills, and
        # whether each satisfies its contract is what the model's functions are
        # being called to decide. Naming a narrower union here only manufactures
        # a violation for every fill kind the union has not caught up with.
        def _positional(*args: object) -> object:
            return func(**dict(zip(arg_names, args, strict=True)))

        def _worst(*, array_floats: bool, array_rank: int, leaf_rank: int) -> float:
            worst = 0.0
            for fills in _fill_assignments(len(arg_names)):
                for overrides in _int_code_sweeps(
                    arg_names=arg_names, int_arg_values=probe_arguments.int_arg_values
                ):
                    for sample in (0.37, 1.63, 2.71):
                        args = [
                            probe_arguments.fill(
                                name,
                                fill,
                                array_floats=array_floats,
                                array_rank=array_rank,
                                leaf_rank=leaf_rank,
                            )
                            if name not in overrides
                            else jnp.asarray(overrides[name], dtype=jnp.int32)
                            for name, fill in zip(arg_names, fills, strict=True)
                        ]
                        args[liquid_pos] = jnp.asarray(sample)
                        jac = jax.jacfwd(_positional, argnums=liquid_pos)(*args)
                        leaves = jax.tree_util.tree_leaves(jac)
                        worst = max(
                            [
                                worst,
                                *(float(jnp.max(jnp.abs(leaf))) for leaf in leaves),
                            ]
                        )
            return worst

        try:
            worst = _evaluate_on_first_workable_fill(_worst)
        except Exception as probe_error:
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
        else:
            return worst

    for target in continuation_plan.stateful_targets:
        next_state_func = continuation_plan.child_reads[target].next_state_func
        worst = _max_abs_first_derivative(next_state_func)
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
    worst = _max_abs_first_derivative(continuation_plan.compute_regime_transition_probs)
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


def _collect_nbegm_schedule_spec(
    *,
    context: SolverBuildContext,
    budget_target: FunctionName,
    continuous_state: StateName,
    consumption_action_name: ActionName,
    probe_failure: Literal["reject", "assume_declared"] = "reject",
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

    user_functions = cast(
        "Mapping[FunctionName, Callable[..., object]]",
        context.user_regimes[context.regime_name].functions,
    )
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
    derived_dags: dict[str, tuple[Callable, tuple[str, ...], tuple[str, ...]]] = {}

    def _derived_dag(
        variable: str,
    ) -> tuple[Callable, tuple[str, ...], tuple[str, ...]]:
        if variable not in derived_dags:
            dag = concatenate_functions(dict(context.functions), targets=variable)
            dag_params = tuple(inspect.signature(dag).parameters)
            # A discrete action the schedule variable reads is bound per branch by the
            # envelope (it shifts the breakpoint partition), not read from kwargs, so it
            # is neither a state nor a param here.
            discrete_action_names = frozenset(
                context.state_action_space.discrete_actions
            )
            params = tuple(
                name
                for name in dag_params
                if name not in state_names and name not in discrete_action_names
            )
            states_read = tuple(
                name for name in dag_params if name in ride_along_state_names
            )
            derived_dags[variable] = (dag, params, states_read)
        return derived_dags[variable]

    sources: list[_NBEGMSource] = []
    for schedule in schedules:
        is_liquid_direct = schedule.variable == liquid_state_name
        dag, params, states_read = (
            (None, (), ()) if is_liquid_direct else _derived_dag(schedule.variable)
        )
        sources.extend(
            _NBEGMSource(
                variable=schedule.variable,
                threshold_param_name=f"{schedule.output}__{bracket.threshold}",
                kind=bracket.kind,
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
    affinity_check = _deferred_probe(
        _fail_if_budget_nonaffine_in_liquid,
        regime_name=context.regime_name,
        probe_arguments=_probe_arguments(context=context),
        coh_dag=coh_dag,
        liquid_name=liquid_state_name,
        # The pure-jump step (liquid-direct, non-ride, all-jump) reads only
        # intercepts and assumes unit slope; every other path recovers the slope.
        require_unit_slope=(
            not ride_along_state_names
            and not has_derived
            and bool(all_kinds)
            and all(kind == "jump" for kind in all_kinds)
        ),
        probe_failure=probe_failure,
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


def _sorted_thresholds(raw: Float1D, *, order_sensitive: bool) -> Float1D:
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
        return jnp.sort(raw)
    ascending = jnp.all(jnp.diff(raw) > 0.0)
    return jnp.where(ascending, jnp.sort(raw), jnp.nan)


def _fail_if_single_liquid_schedules_unsupported(
    *,
    schedules: tuple[Any, ...],
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
    arithmetic: ComparisonArithmetic = "certified",
) -> tuple[Float1D, Float1D, Float1D]:
    """Solve one period of a cliffed single-liquid budget, dispatching on kind.

    Reads the continuation jump-aware at every jump (no bridging), so the solve
    is exact through recurring jumps, not only at a terminal-adjacent period.
    The kind flags come from `_schedule_kind_flags`. The declared law reaches the
    kernels as its landing points on `savings_grid` and their derivative there.
    Returns this period's value, marginal value of liquid, and consumption policy
    on `liquid`.
    """
    from _lcm.egm.nbegm_step import (  # noqa: PLC0415
        nbegm_multi_interval_step,
        nbegm_one_asset_step,
        nbegm_recurring_jump_step,
        nbegm_unified_step,
    )

    if is_single_jump:
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
        )
    if is_multi_jump:
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
        )
    if is_mixed:
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
    )


def _build_nbegm_continuous_core(
    *,
    savings_grid: Float1D,
    functions: EconFunctionsMapping,
    consumption_action: ActionName,
    schedule_spec: _NBEGMScheduleSpec,
    envelope_arithmetic: ComparisonArithmetic = "certified",
) -> Callable:
    """Build the jittable continuous-schedule EGM core for one continuation target.

    The core reads the schedule's thresholds as liquid breakpoints, recovers the
    active affine cash-on-hand segment per interval by differentiating the composed
    `coh` at each interval's representative, and runs the kind-appropriate EGM step.
    """
    from _lcm.egm.nbegm_breakpoints import (  # noqa: PLC0415
        interval_midpoints,
        interval_segment_coefficients,
    )

    kinds = schedule_spec.breakpoint_kinds
    is_single_jump, is_multi_jump, is_mixed, jump_mask, flat_mask = (
        _schedule_kind_flags(kinds)
    )
    order_sensitive = len(set(kinds)) > 1

    from _lcm.egm.preferences import (  # noqa: PLC0415
        NEWTON_ACTION_FLOOR,
        get_preferences_builder,
        newton_action_ceiling,
    )

    build_preferences = get_preferences_builder(
        functions=functions,
        action_name=consumption_action,
        action_lower=NEWTON_ACTION_FLOOR,
        action_upper=newton_action_ceiling(savings_grid),
    )

    def core(
        *,
        liquid: Float1D,
        next_liquid_grid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        next_liquid: Float1D,
        marginal_return: Float1D,
        **params: FloatND,
    ) -> tuple[Float1D, EGMCarry]:
        preferences = build_preferences(params)
        coh_params = {name: params[name] for name in schedule_spec.coh_param_names}

        def coh_of_liquid(scalar_liquid: FloatND) -> FloatND:
            return schedule_spec.coh_of_liquid_dag(
                **{schedule_spec.liquid_state_name: scalar_liquid}, **coh_params
            )

        # Zero declared breakpoints ⇒ an empty partition: one interval covering
        # the whole liquid axis, solved as plain EGM.
        breakpoints = (
            _sorted_thresholds(
                jnp.stack(
                    [params[name] for name in schedule_spec.threshold_param_names]
                ),
                order_sensitive=order_sensitive,
            )
            if schedule_spec.threshold_param_names
            else jnp.zeros((0,), dtype=canonical_float_dtype())
        )
        midpoints = interval_midpoints(liquid_grid=liquid, breakpoints=breakpoints)
        coh_slopes, coh_intercepts = interval_segment_coefficients(
            schedule=coh_of_liquid, interval_midpoints=midpoints
        )
        value, marginal, _policy = _solve_cliffed_budget(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid=liquid,
            next_liquid_grid=next_liquid_grid,
            savings_grid=savings_grid,
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
            arithmetic=envelope_arithmetic,
        )
        carry = EGMCarry(
            endog_grid=liquid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(0.0, dtype=value.dtype),
        )
        return value, carry

    return core


def _build_nbegm_continuation_plan(
    *,
    context: SolverBuildContext,
    period: int,
    post_decision_name: FunctionName,
    stochastic_node_batch_size: int = 0,
) -> Any:  # noqa: ANN401  # `ContinuationPlan`; not annotated precisely (importing
    # module scope closes an import cycle (`continuation` → … → `lcm.solvers`).
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
    return build_continuation_plan(
        user_regimes=context.user_regimes,
        functions=context.functions,
        transitions=context.transitions,
        transition_laws=context.transition_laws,
        stateful_targets=stateful_targets,
        scalar_targets=scalar_targets,
        compute_regime_transition_probs=compute_regime_transition_probs,
        post_decision_name=post_decision_name,
        stochastic_node_batch_size=stochastic_node_batch_size,
        regime_to_v_interpolation_info=v_interpolation_info,
        risk_aversion_param_name=risk_aversion_param_name,
    )


def _solve_ride_along_cell_step(
    *,
    has_jump: bool,
    jump_positions: tuple[Any, ...],
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
) -> tuple[Float1D, Float1D, Float1D]:
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
    )


def _ride_along_jump_config(
    kinds: tuple[str, ...],
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
        jnp.asarray(jump_flags, dtype=bool),
        n_jumps,
        n_jumps > 0,
        static_jump_positions,
        dynamic_jumps,
    )


def _partition_jumps(
    preimages: Float1D,
    *,
    dynamic_jumps: bool,
    jump_flags: BoolND,
    n_jumps: int,
    static_jump_positions: tuple[int, ...],
) -> tuple[Float1D, tuple[Any, ...]]:
    """Sort a cell's breakpoint preimages and locate the jumps in the sorted order.

    With fixed jump positions the declared-order positions carry over; when the
    jumps reorder per cell the sorted-order jump indices are recovered from the
    permutation that sorts the preimages.
    """
    if dynamic_jumps:
        order = jnp.argsort(preimages)
        sorted_jumps = jnp.nonzero(jump_flags[order], size=n_jumps)[0]
        return preimages[order], tuple(sorted_jumps[k] for k in range(n_jumps))
    return jnp.sort(preimages), static_jump_positions


def _indexed_threshold_value(
    *,
    table: Any,  # noqa: ANN401  # scalar param, threshold table, or mapping leaf
    subkey: str | None,
    index_state: str | None,
    static_index: int | None,
    cell: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Read a breakpoint threshold from its param for one ride-along cell.

    The param is resolved to a value in this order:
    - `subkey` selects an entry inside a `MappingLeaf` (`leaf.data[subkey]`).
    - `index_state` reads the row at that ride-along state's code in this cell.
    - `static_index` selects a column (e.g. a bracket edge).

    A scalar threshold leaves every step disabled and passes through unchanged.
    """
    value = table
    if subkey is not None:
        value = value.data[subkey]
    if index_state is not None:
        value = value[cell[index_state]]
    if static_index is not None:
        value = value[static_index]
    return value


@dataclass(frozen=True)
class _NBEGMRideAlongStatics:
    """Build-time config the ride-along continuation and envelope cores share.

    Both cores rebuild each ride-along cell's breakpoint partition, budget schedule,
    discount factor, and utility identically off this config; the continuation core
    additionally reads the regime transition through `bind_continuation`. Every field
    is a Python-level static derived once from the schedule spec and continuation plan.
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
    state_names: tuple[str, ...]
    """Liquid plus ride-along state names — the kwargs that are state grids."""
    continuation_reads_liquid: bool
    """Whether the continuation reads the current liquid state — through a carry
    target's next-state law or the regime-transition probabilities — so the
    continuation is piecewise-constant across declared intervals and the per-interval
    path applies."""
    interval_batch_size: int
    """Batch size for the per-interval continuation read: `0` evaluates all
    intervals in one vectorized pass, a positive size runs sequential chunks of
    that many intervals."""
    branch_batch_size: int
    """Block size for the discrete-action branch axis in both cores: `0` runs the
    whole axis in one vectorized pass, a positive size scans it in blocks of that
    many branches."""
    consumption_action_name: ActionName
    """Name of the continuous consumption action the period utility reads."""
    utility_param_names: tuple[str, ...]
    """Qualified utility params (excluding the consumption action and states)."""
    utility_state_names: tuple[str, ...]
    """Ride-along states the period utility reads, bound per cell."""
    coh_state_names: tuple[str, ...]
    """Ride-along states the cash-on-hand schedule reads, bound per cell."""
    discount_param_names: tuple[str, ...]
    """Qualified params the discount-factor DAG reads, or empty for flat discount."""
    discount_state_names: tuple[str, ...]
    """Ride-along states the discount-factor DAG reads, or empty for flat discount."""
    n_intervals: int
    """Number of liquid intervals the breakpoints split each cell into (N + 1)."""
    n_savings: int
    """Length of the post-decision savings grid."""
    envelope_segment_block_size: int
    """Block size for streaming the merged upper envelope over candidate segments;
    `0` keeps the one-shot dense envelope (see `NBEGM.envelope_segment_block_size`)."""
    envelope_arithmetic: ComparisonArithmetic
    """Which arithmetic decides envelope ownership (see
    `NBEGM.envelope_arithmetic`)."""
    cell_block_size: int
    """Block size for streaming both ride-along cores over ride cells; `0` vmaps
    the whole flattened mesh at once (see `NBEGM.cell_block_size`)."""
    n_action_branches: int
    """Number of discrete-action branches the continuation carries a leading axis
    over; `0` when the regime carries no discrete action (no branch axis). A branch
    reads its own next-state continuation, so a co-state-feeding action gets a
    distinct row per branch and a budget-only action gets identical rows."""
    co_map_state_names: tuple[str, ...] = ()
    """Fixed, distributed ride-along states co-mapped with the child carry.

    A leading prefix of `ride_names`: each is distributed (sharded one block per
    device) and never transitions, so a ride cell's continuation depends only on
    its own slice of the next-period carry. The continuation core `vmap`s over
    these axes, co-slicing the carry, so each device interpolates only its slice
    and XLA inserts no all-gather. Empty when no ride state qualifies."""

    @property
    def n_published_jumps(self) -> int:
        """Number of jump preimages the carry publishes per row."""
        return self.n_jumps if self.publish_jump_topology else 0

    def n_ride_cells(self, *, states: Mapping[str, object]) -> int:
        """Number of flattened ride-along cells for the given state grids."""
        count = 1
        for name in self.ride_names:
            count *= int(jnp.asarray(states[name]).shape[0])
        return count


def _nbegm_ride_along_statics(
    *,
    savings_grid: Float1D,
    schedule_spec: _NBEGMScheduleSpec,
    continuation_plan: Any,  # noqa: ANN401  # `ContinuationPlan`; import-cycle-safe
    envelope_segment_block_size: int = 0,
    envelope_arithmetic: ComparisonArithmetic = "certified",
    cell_block_size: int = 0,
    interval_batch_size: int = 0,
    branch_batch_size: int = 0,
    publish_jump_topology: bool = True,
    co_map_state_names: tuple[str, ...] = (),
) -> _NBEGMRideAlongStatics:
    """Derive the static config the ride-along continuation and envelope cores share.

    Partitions the schedule's breakpoints, classifies the jump structure, and reads
    each component DAG's argument names (utility, cash-on-hand, discount factor) into
    the per-cell parameter and state splits both cores apply identically.
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
        _ride_along_jump_config(kinds)
    )

    liquid_name = schedule_spec.liquid_state_name
    ride_names = schedule_spec.ride_along_state_names
    state_names = (liquid_name, *ride_names)

    # The co-mapped ride states must be a leading prefix of the ride axes: the
    # continuation core's outer `vmap` peels them off the front of both the ride
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
    def _next_state_reads_liquid(target: str) -> bool:
        next_state_func = continuation_plan.child_reads[target].next_state_func
        return liquid_name in inspect.signature(next_state_func).parameters

    transition_probs_read_liquid = (
        liquid_name
        in inspect.signature(
            continuation_plan.compute_regime_transition_probs
        ).parameters
    )
    continuation_reads_liquid = transition_probs_read_liquid or any(
        _next_state_reads_liquid(target)
        for target in continuation_plan.stateful_targets
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
    # The discount factor is either pylcm's flat
    # `koopmans_aggregator__discount_factor` param or, when
    # the regime supplies a `discount_factor` DAG function (e.g. a per-preference-type
    # beta read off a ride-along state), resolved per cell from that function's
    # qualified params and ride-along state arguments.
    discount_factor_dag = schedule_spec.discount_factor_dag
    if discount_factor_dag is None:
        discount_param_names: tuple[str, ...] = ()
        discount_state_names: tuple[str, ...] = ()
    else:
        discount_arg_names = tuple(inspect.signature(discount_factor_dag).parameters)
        discount_param_names = tuple(
            name for name in discount_arg_names if name not in state_names
        )
        discount_state_names = tuple(
            name for name in ride_names if name in discount_arg_names
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
        discount_state_names=discount_state_names,
        n_intervals=len(sources) + 1,
        n_savings=int(savings_grid.shape[0]),
        envelope_segment_block_size=envelope_segment_block_size,
        envelope_arithmetic=envelope_arithmetic,
        cell_block_size=cell_block_size,
        interval_batch_size=interval_batch_size,
        branch_batch_size=branch_batch_size,
        n_action_branches=(
            0
            if not schedule_spec.discrete_actions
            else len(schedule_spec.branch_bindings)
        ),
        co_map_state_names=co_map_ride_names,
    )


def _nbegm_cell_breakpoints(
    *,
    statics: _NBEGMRideAlongStatics,
    kwargs: Mapping[str, Any],
    cell: dict[str, Any],
    liquid_grid: Float1D,
    dtype: Any,  # noqa: ANN401  # canonical float dtype
    action_binding: Mapping[str, Any] = MappingProxyType({}),
) -> tuple[Float1D, tuple[Any, ...]]:
    """Build one ride-along cell's sorted liquid breakpoints and jump positions.

    Each declared schedule's threshold maps to its asset value in its own variable
    (directly for a liquid-state schedule, via the per-cell affine preimage for a
    derived-variable schedule), and the sources merge into one sorted partition. A
    degenerate boundary — a derived variable with (near-)zero asset slope in this cell,
    so the threshold is never crossed — has a non-finite preimage; clamping to a margin
    just outside the grid collapses it to an empty edge interval instead of poisoning a
    live interval's affine segment.
    """
    import inspect  # noqa: PLC0415

    from _lcm.egm.nbegm_breakpoints import (  # noqa: PLC0415
        clamp_breakpoints_to_grid,
        linear_asset_preimage,
    )

    liquid_name = statics.liquid_name

    def cell_breakpoint(source: _NBEGMSource) -> FloatND:
        threshold_value = _indexed_threshold_value(
            table=kwargs[source.threshold_param_name],
            subkey=source.threshold_subkey,
            index_state=source.threshold_index_state,
            static_index=source.threshold_static_index,
            cell=cell,
        )
        threshold = jnp.asarray(threshold_value, dtype=dtype)
        if source.derived_of_liquid_dag is None:
            return threshold
        dag = source.derived_of_liquid_dag
        derived_params = {name: kwargs[name] for name in source.derived_param_names}
        cell_for_dag = {name: cell[name] for name in source.derived_state_names}
        dag_arg_names = frozenset(inspect.signature(dag).parameters)
        dag_action_binding = {
            name: value
            for name, value in action_binding.items()
            if name in dag_arg_names
        }

        def derived_of_liquid(scalar_liquid: FloatND) -> FloatND:
            return dag(
                **{liquid_name: scalar_liquid},
                **cell_for_dag,
                **derived_params,
                **dag_action_binding,
            )

        return linear_asset_preimage(derived_of_liquid, threshold=threshold)

    # Zero declared breakpoints ⇒ an empty partition (one interval per cell).
    preimages = (
        clamp_breakpoints_to_grid(
            breakpoints=jnp.stack(
                [cell_breakpoint(source) for source in statics.sources]
            ),
            liquid_grid=liquid_grid,
        )
        if statics.sources
        else jnp.zeros((0,), dtype=dtype)
    )
    return _partition_jumps(
        preimages,
        dynamic_jumps=statics.dynamic_jumps,
        jump_flags=statics.jump_flags_arr,
        n_jumps=statics.n_jumps,
        static_jump_positions=statics.static_jump_positions,
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
    dtype: Any,  # noqa: ANN401
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
    continuation_plan: Any,  # noqa: ANN401  # `ContinuationPlan`; import-cycle-safe
    regime_name: RegimeName,
    statics: _NBEGMRideAlongStatics,
    kwargs: dict[str, Any],
    cell: dict[str, Any],
    combo_pool: dict[str, Any],
    liquid_grid: Float1D,
    savings_grid: Float1D,
    dtype: Any,  # noqa: ANN401
    midpoints: Float1D | None = None,
) -> FloatND:
    """Map the self-read child's value cliffs to one-sided savings targets.

    A child value jump creates a legitimate one-sided optimum — save to just
    inside the cliff's owning side — that generically falls strictly between
    savings nodes. Per ride cell this recovers the cell's jump preimages in
    the child's liquid space, inverts the affine savings-form liquid law, and
    returns one target a few float margins inside each side of every jump
    (`2 * n_jumps` entries). Targets outside the savings grid's span, or under
    a non-increasing liquid law, are NaN — the envelope's point-candidate
    family treats NaN entries as dead.
    """
    read = continuation_plan.child_reads[regime_name]
    post_decision_name = continuation_plan.post_decision_name
    breakpoints, jump_positions = _nbegm_cell_breakpoints(
        statics=statics, kwargs=kwargs, cell=cell, liquid_grid=liquid_grid, dtype=dtype
    )
    jumps = jnp.stack([breakpoints[position] for position in jump_positions])

    def targets_for_pool(pool: dict[str, Any]) -> FloatND:
        def next_euler_state(savings_value: FloatND) -> FloatND:
            next_states = read.next_state_func(
                **pool, **{post_decision_name: savings_value}
            )
            return jnp.asarray(next_states[read.next_state_key], dtype=dtype)

        intercept = next_euler_state(jnp.asarray(0.0, dtype=dtype))
        slope = next_euler_state(jnp.asarray(1.0, dtype=dtype)) - intercept
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

    if midpoints is None:
        return targets_for_pool(combo_pool)
    # An interval-bound liquid law: the savings-to-liquid map (and so each
    # cliff's savings preimage) is specific to the interval whose node the
    # liquid state is bound to — one target row per interval.
    liquid_name = statics.liquid_name
    return jax.vmap(
        lambda midpoint: targets_for_pool({**combo_pool, liquid_name: midpoint})
    )(midpoints)


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
            lambda leaf, axis=axis: axis if jnp.ndim(leaf) > 0 else None,
            target_carry,
        )
    return MappingProxyType(result)


def _build_nbegm_continuation_core(  # noqa: C901, PLR0915
    *,
    savings_grid: Float1D,
    continuation_plan: Any,  # noqa: ANN401  # `ContinuationPlan`; import-cycle-safe
    statics: _NBEGMRideAlongStatics,
    regime_name: RegimeName,
    cliff_candidates: bool,
    schedule_spec: _NBEGMScheduleSpec,
) -> Callable:
    """Build the continuation half of the ride-along solve, jitted in isolation.

    Per ride-along cell the continuation is read through `bind_continuation` —
    integrating the next-period regime transition, stochastic shocks, the ride-along
    co-state transition, and the child value interpolation — and evaluated over the
    savings grid. The interval regime binds the liquid state to each interval's node
    and returns one continuation row per interval; the non-interval regime returns one
    row over the savings grid. The cells stack into `(n_ride_cells, [n_intervals,]
    n_savings)` expected-value and expected-marginal arrays the envelope core consumes.

    The heavy fan-out lives only here: this core builds no utility, cash-on-hand, or
    discount closure, so its compiled program never carries the EGM/envelope math.
    """
    from _lcm.egm.continuation import bind_continuation  # noqa: PLC0415
    from _lcm.egm.nbegm_breakpoints import interval_midpoints  # noqa: PLC0415

    liquid_name = statics.liquid_name
    ride_names = statics.ride_names
    state_names = statics.state_names
    action_names = tuple(name for name, _ in schedule_spec.discrete_actions)
    branch_bindings = schedule_spec.branch_bindings
    # A distributed, never-transitioning ride state is co-mapped: its axis is the
    # leading ride axis, so the mesh over the *remaining* ride states solves inside
    # an outer `vmap` that co-slices the child carry — each device reads only its
    # slice, no all-gather. Empty co-map leaves `inner_ride_names == ride_names`.
    co_map_names = statics.co_map_state_names
    inner_ride_names = ride_names[len(co_map_names) :]

    def continuation_core(  # noqa: C901
        *,
        next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],  # noqa: ARG001
        **kwargs: Any,  # noqa: ANN401  # state grids + flat params (mixed dtypes)
    ) -> tuple[FloatND, ...]:
        dtype = canonical_float_dtype()
        liquid = jnp.asarray(kwargs[liquid_name], dtype=dtype)
        param_pool = {key: v for key, v in kwargs.items() if key not in state_names}

        def _solve_inner_mesh(  # noqa: C901
            *,
            carry: MappingProxyType[RegimeName, EGMCarry],
            comap_bindings: dict[str, Any],
        ) -> tuple[FloatND, ...]:
            def cell_continuation(
                ride_values: tuple[Any, ...],
            ) -> tuple[FloatND, ...]:
                cell = dict(zip(inner_ride_names, ride_values, strict=True))

                def rows_for_pool(combo_pool: dict[str, Any]) -> tuple[FloatND, ...]:
                    return _cell_rows_for_pool(combo_pool)

                base_pool = {**param_pool, **comap_bindings, **cell}
                if not action_names:
                    return rows_for_pool(base_pool)

                # A discrete action that feeds the continuation reads a different
                # next-state per branch, so the continuation is evaluated per branch
                # (the actions ride into `combo_pool` → `next_state_func`). A leading
                # branch axis is added over the product of the declared grids; when
                # the actions do not feed the continuation the branch rows are
                # identical, matching the shared-continuation case. The branch body
                # compiles once and runs at a fixed vector width, so per-branch
                # intermediates never all sit in flight whatever the partition.
                def rows_for_codes(codes_row: IntND) -> tuple[FloatND, ...]:
                    binding = {
                        name: codes_row[position]
                        for position, name in enumerate(action_names)
                    }
                    return rows_for_pool({**base_pool, **binding})

                codes = _stacked_branch_codes(
                    branch_bindings=branch_bindings, action_names=action_names
                )
                return _map_branch_partitioned(
                    func=rows_for_codes,
                    xs=codes,
                    requested_block_size=statics.branch_batch_size,
                )

            def _cell_rows_for_pool(combo_pool: dict[str, Any]) -> tuple[FloatND, ...]:
                cell = {name: combo_pool[name] for name in ride_names}

                def cliff_targets_for(midpoints: Float1D | None) -> FloatND:
                    # Under the one-sided read, the cell also evaluates the blended
                    # continuation at each self-read cliff's one-sided savings
                    # targets; the extra columns ride at the end of the savings
                    # axis and the envelope core adds them as point candidates.
                    return _cliff_savings_targets(
                        continuation_plan=continuation_plan,
                        regime_name=regime_name,
                        statics=statics,
                        kwargs=kwargs,
                        cell=cell,
                        combo_pool=combo_pool,
                        liquid_grid=liquid,
                        savings_grid=savings_grid,
                        dtype=dtype,
                        midpoints=midpoints,
                    )

                def query_with(targets: FloatND) -> Float1D:
                    return jnp.concatenate(
                        [
                            savings_grid,
                            jnp.where(jnp.isnan(targets), savings_grid[0], targets),
                        ]
                    )

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
                        for name in action_names
                        if name in combo_pool
                    }
                    breakpoints, _ = _nbegm_cell_breakpoints(
                        statics=statics,
                        kwargs=kwargs,
                        cell=cell,
                        liquid_grid=liquid,
                        dtype=dtype,
                        action_binding=cell_action_binding,
                    )
                    midpoints = interval_midpoints(
                        liquid_grid=liquid, breakpoints=breakpoints
                    )
                    cliff_targets = (
                        cliff_targets_for(midpoints) if cliff_candidates else None
                    )

                    def interval_rows(
                        interval_inputs: tuple[FloatND, ...],
                        combo_pool: dict[str, Any] = combo_pool,
                    ) -> tuple[Float1D, Float1D]:
                        midpoint, *interval_targets = interval_inputs
                        interval_pool = {**combo_pool, liquid_name: midpoint}
                        interval_continuation = bind_continuation(
                            plan=continuation_plan,
                            combo_pool=interval_pool,
                            next_regime_to_continuation=carry,
                            dtype=dtype,
                            co_map_state_names=co_map_names,
                        )
                        query = (
                            savings_grid
                            if not interval_targets
                            else query_with(interval_targets[0])
                        )
                        return jax.vmap(interval_continuation)(query)

                    interval_inputs = (
                        (midpoints,)
                        if cliff_targets is None
                        else (midpoints, cliff_targets)
                    )
                    rows = _map_ride_partitioned(
                        func=interval_rows,
                        xs=interval_inputs,
                        requested_block_size=statics.interval_batch_size,
                    )
                    if cliff_targets is None:
                        return rows
                    return (*rows, cliff_targets)

                continuation = bind_continuation(
                    plan=continuation_plan,
                    combo_pool=combo_pool,
                    next_regime_to_continuation=carry,
                    dtype=dtype,
                    co_map_state_names=co_map_names,
                )
                cliff_targets = cliff_targets_for(None) if cliff_candidates else None
                rows = jax.vmap(continuation)(
                    savings_grid if cliff_targets is None else query_with(cliff_targets)
                )
                if cliff_targets is None:
                    return rows
                return (*rows, cliff_targets)

            if not inner_ride_names:
                # Every ride axis is co-mapped: a single inner cell per co-map slice.
                # Add a leading singleton so the co-map merge sees one inner cell.
                rows = cell_continuation(())
                return tuple(leaf[jnp.newaxis] for leaf in rows)

            ride_grids = tuple(jnp.asarray(kwargs[name]) for name in inner_ride_names)
            mesh = jnp.meshgrid(*ride_grids, indexing="ij")
            flat_cells = tuple(grid.ravel() for grid in mesh)
            return _map_ride_partitioned(
                func=cell_continuation,
                xs=flat_cells,
                requested_block_size=statics.cell_block_size,
            )

        def _solve_with_co_map(
            *,
            carry: MappingProxyType[RegimeName, EGMCarry],
            remaining: tuple[str, ...],
            comap_bindings: dict[str, Any],
        ) -> tuple[FloatND, ...]:
            if not remaining:
                return _solve_inner_mesh(carry=carry, comap_bindings=comap_bindings)
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

            def slice_solve(
                head_value: Any,  # noqa: ANN401
                sliced_carry: MappingProxyType[RegimeName, EGMCarry],
            ) -> tuple[FloatND, ...]:
                return _solve_with_co_map(
                    carry=sliced_carry,
                    remaining=tuple(tail),
                    comap_bindings={**comap_bindings, head: head_value},
                )

            stacked = jax.vmap(slice_solve, in_axes=(0, in_axes))(head_grid, carry)
            # Merge the new leading co-map axis into the flat inner-cell axis, keeping
            # co-map states outermost — the meshgrid-`ij` order the envelope expects.
            return tuple(leaf.reshape(-1, *leaf.shape[2:]) for leaf in stacked)

        return _solve_with_co_map(
            carry=next_regime_to_continuation,
            remaining=co_map_names,
            comap_bindings={},
        )

    return continuation_core


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


def _cell_solver(
    *,
    solve_one_cell: Callable,
    flat_cells: tuple[FloatND | IntND, ...],
    cont_value_stack: FloatND,
    cont_marginal_stack: FloatND,
    cliff_savings_stack: FloatND | None,
) -> tuple[Callable, tuple[FloatND | IntND, ...]]:
    """Bind the per-cell solve to one row of the ride mesh and its stacks.

    The trailing per-cell inputs are the continuation core's stacks — value and
    marginal rows, plus the save-to-cliff savings targets when the one-sided
    read publishes jump topology. The returned body takes one row of the mesh;
    the caller decides at which width rows are evaluated.
    """
    if cliff_savings_stack is None:
        return lambda row: solve_one_cell(row[:-2], row[-2], row[-1]), (
            *flat_cells,
            cont_value_stack,
            cont_marginal_stack,
        )
    return lambda row: solve_one_cell(row[:-3], row[-3], row[-2], row[-1]), (
        *flat_cells,
        cont_value_stack,
        cont_marginal_stack,
        cliff_savings_stack,
    )


def _build_nbegm_envelope_core(  # noqa: C901, PLR0915
    *,
    savings_grid: Float1D,
    schedule_spec: _NBEGMScheduleSpec,
    statics: _NBEGMRideAlongStatics,
    is_epstein_zin: bool = False,
) -> Callable:
    """Build the EGM/envelope half of the ride-along solve, jitted in isolation.

    Per ride-along cell this re-derives the budget schedule, discount factor, and
    utility from the same (states, params), then solves the 1-D continuous-budget step
    against the cell's continuation row supplied by the continuation core. The interval
    regime runs the per-interval continuation step; the non-interval regime runs the
    multi-interval or unified jump step. The cells stack into the value array and carry
    with the ride-along axes leading the liquid axis, matching the canonical layout.

    Re-deriving the breakpoints, cash-on-hand coefficients, and discount factor here is
    cheap closed-form work; this core calls no continuation reader, so the heavy
    transition fan-out never enters its compiled program.
    """
    from _lcm.egm.nbegm_breakpoints import (  # noqa: PLC0415
        interval_midpoints,
        interval_segment_coefficients,
    )
    from _lcm.egm.nbegm_step import (  # noqa: PLC0415
        nbegm_per_interval_continuation_step_savings,
    )
    from _lcm.egm.preferences import (  # noqa: PLC0415
        NEWTON_ACTION_FLOOR,
        newton_action_ceiling,
        preferences_from_utility,
    )

    liquid_name = statics.liquid_name
    ride_names = statics.ride_names
    discount_factor_dag = schedule_spec.discount_factor_dag
    # This route composes period utility from the DAG, so the Euler equation has no
    # closed form to invert and its root is always bracketed numerically. The
    # clamped near-zero-marginal corner whose root exceeds the bracket lands far to
    # the right and is discarded by the upper envelope.
    action_upper = newton_action_ceiling(savings_grid)
    action_lower = jnp.asarray(NEWTON_ACTION_FLOOR, dtype=action_upper.dtype)
    import inspect  # noqa: PLC0415

    # The action binds into a branch's period utility only when the utility DAG reads
    # it (a leisure/effort-like term); otherwise the binding is dropped so a utility
    # that does not name the action is called with its own arguments alone.
    utility_arg_names = frozenset(
        inspect.signature(schedule_spec.utility_dag).parameters
    )

    def envelope_core(
        *,
        cont_value_stack: FloatND,
        cont_marginal_stack: FloatND,
        cliff_savings_stack: FloatND | None = None,
        **kwargs: Any,  # noqa: ANN401  # state grids + flat params (mixed dtypes)
    ) -> tuple[FloatND, EGMCarry]:
        dtype = canonical_float_dtype()
        liquid = jnp.asarray(kwargs[liquid_name], dtype=dtype)
        coh_params = {name: kwargs[name] for name in schedule_spec.coh_param_names}
        utility_params = {name: kwargs[name] for name in statics.utility_param_names}
        discount_params = {name: kwargs[name] for name in statics.discount_param_names}
        # Epstein-Zin: the aggregator curvature is `rho = 1/psi` where `psi` is the
        # Koopmans aggregator's `intertemporal_elasticity_of_substitution`. The
        # step reads the continuation pair as `(nu, dnu/ds)` and inverts the
        # recursive Euler equation; `None` keeps the additive expected-utility step.
        inverse_eis = (
            1.0
            / kwargs["koopmans_aggregator__intertemporal_elasticity_of_substitution"]
            if is_epstein_zin
            else None
        )

        def solve_one_cell(
            ride_values: tuple[Any, ...],
            cont_value: FloatND,
            cont_marginal: FloatND,
            cliff_savings: FloatND | None = None,
        ) -> tuple[Float1D, ...]:
            cont_value, cont_marginal, extra_cont_value = _split_cliff_columns(
                cont_value=cont_value,
                cont_marginal=cont_marginal,
                n_nodes=savings_grid.shape[0],
                has_cliff_columns=cliff_savings is not None,
            )
            cell = dict(zip(ride_names, ride_values, strict=True))
            cell_discount_factor = (
                kwargs["koopmans_aggregator__discount_factor"]
                if discount_factor_dag is None
                else discount_factor_dag(
                    **{name: cell[name] for name in statics.discount_state_names},
                    **discount_params,
                )
            )

            # With published jump breakpoints, the cell publishes each jump's preimage
            # and its exact one-sided value limits: the liquid query grid is augmented
            # with a point just inside each side of every jump, solved in the same call,
            # and split back out positionally. A published jump shares one query grid
            # across the branches, so the action cannot enter its schedule variable
            # (guarded), and the cell-level partition is branch-independent. The bridged
            # read skips the augmentation; each branch then partitions on its own
            # breakpoints (recomputed inside `solve_branch`) over the plain liquid grid.
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
            else:
                query_grid = liquid

            def solve_branch(
                action_binding: Mapping[str, IntND],
                branch_cont_value: FloatND,
                branch_cont_marginal: FloatND,
                branch_extra_cont_value: FloatND | None,
                branch_cliff_savings: FloatND | None,
            ) -> tuple[Float1D, Float1D, Float1D]:
                """Solve the cell's continuous subproblem for one discrete branch.

                `action_binding` binds the discrete action into cash-on-hand (empty
                when the regime carries no discrete action). `branch_cont_value` /
                `branch_cont_marginal` are this branch's continuation rows — a branch
                reads its own next-state continuation when the action feeds a co-state's
                law of motion, and identical rows when it feeds only the budget. The
                breakpoint partition, utility, and jump augmentation are
                continuation-independent and computed once in the enclosing scope.
                """

                def coh_of_liquid(scalar_liquid: FloatND) -> FloatND:
                    return schedule_spec.coh_of_liquid_dag(
                        **{liquid_name: scalar_liquid},
                        **{name: cell[name] for name in statics.coh_state_names},
                        **coh_params,
                        **action_binding,
                    )

                utility_action_binding = {
                    name: value
                    for name, value in action_binding.items()
                    if name in utility_arg_names
                }

                def utility_of_consumption(consumption_value: FloatND) -> FloatND:
                    return schedule_spec.utility_dag(
                        **{statics.consumption_action_name: consumption_value},
                        **{name: cell[name] for name in statics.utility_state_names},
                        **utility_params,
                        **utility_action_binding,
                    )

                preferences = preferences_from_utility(
                    utility_of_action=utility_of_consumption,
                    action_lower=action_lower,
                    action_upper=action_upper,
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
                    liquid_grid=liquid,
                    dtype=dtype,
                    action_binding=action_binding,
                )
                branch_midpoints = interval_midpoints(
                    liquid_grid=liquid, breakpoints=branch_breakpoints
                )
                coh_slopes, coh_intercepts = interval_segment_coefficients(
                    schedule=coh_of_liquid, interval_midpoints=branch_midpoints
                )
                if statics.continuation_reads_liquid:
                    # True cash-on-hand per liquid grid point keeps the step's corners
                    # feasible where a partly-binding kink makes an interval's recovered
                    # affine budget extrapolate below zero.
                    coh_grid = jax.vmap(coh_of_liquid)(query_grid)
                    return nbegm_per_interval_continuation_step_savings(
                        cont_value=branch_cont_value,
                        cont_marginal=branch_cont_marginal,
                        liquid_grid=query_grid,
                        savings_grid=savings_grid,
                        discount_factor=cell_discount_factor,
                        preferences=preferences,
                        coh_slopes=coh_slopes,
                        coh_intercepts=coh_intercepts,
                        breakpoints=branch_breakpoints,
                        coh_grid=coh_grid,
                        envelope_segment_block_size=statics.envelope_segment_block_size,
                        arithmetic=statics.envelope_arithmetic,
                        extra_savings=branch_cliff_savings,
                        extra_cont_value=branch_extra_cont_value,
                    )
                return _solve_ride_along_cell_step(
                    has_jump=statics.has_jump,
                    jump_positions=branch_jump_positions,
                    extra_savings=branch_cliff_savings,
                    extra_cont_value=branch_extra_cont_value,
                    cont_value=branch_cont_value,
                    cont_marginal=branch_cont_marginal,
                    liquid_grid=query_grid,
                    savings_grid=savings_grid,
                    discount_factor=cell_discount_factor,
                    preferences=preferences,
                    coh_slopes=coh_slopes,
                    coh_intercepts=coh_intercepts,
                    breakpoints=branch_breakpoints,
                    inverse_eis=inverse_eis,
                    arithmetic=statics.envelope_arithmetic,
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
            branch_action_names = tuple(
                name for name, _ in schedule_spec.discrete_actions
            )
            if branch_action_names:
                # `lax.map` compiles the branch subproblem once and streams it in
                # `branch_batch_size` blocks (the whole axis in one vectorized pass
                # by default) — per-branch EGM intermediates never all sit in
                # flight, and the branch axis is never Python-unrolled. Optional
                # branch inputs enter the mapped pytree only when present.
                branch_inputs = _branch_inputs(
                    codes=_stacked_branch_codes(
                        branch_bindings=schedule_spec.branch_bindings,
                        action_names=branch_action_names,
                    ),
                    cont_value=cont_value,
                    cont_marginal=cont_marginal,
                    extra_cont_value=extra_cont_value,
                    cliff_savings=cliff_savings,
                )

                def solve_one_branch(
                    inputs: dict[str, Any],
                ) -> tuple[Float1D, Float1D]:
                    binding = {
                        name: inputs["codes"][position]
                        for position, name in enumerate(branch_action_names)
                    }
                    step = solve_branch(
                        binding,
                        inputs["cont_value"],
                        inputs["cont_marginal"],
                        inputs.get("extra_cont_value"),
                        inputs.get("cliff_savings"),
                    )
                    return step[0], step[1]

                value_stack, marginal_stack = _map_branch_partitioned(
                    func=solve_one_branch,
                    xs=branch_inputs,
                    requested_block_size=statics.branch_batch_size,
                )
                value_row, marginal_row = _discrete_envelope_over_branches(
                    value_stack=value_stack,
                    marginal_stack=marginal_stack,
                    taste_shock_scale=0.0,
                )
            else:
                value_row, marginal_row, _policy_row = solve_branch(
                    {}, cont_value, cont_marginal, extra_cont_value, cliff_savings
                )

            if statics.n_published_jumps == 0:
                return (value_row, marginal_row)
            # The carry keeps the whole augmented row — the jump rides inside
            # the endogenous grid as a duplicated abscissa carrying its exact
            # one-sided value and marginal limits. Only the published value
            # array needs the original liquid nodes, sliced back out through
            # the sort permutation.
            value_at_liquid = value_row[unsort][: liquid.shape[0]]
            return (value_at_liquid, endog_row, value_row, marginal_row, jumps)

        ride_grids = tuple(jnp.asarray(kwargs[name]) for name in ride_names)
        ride_shape = tuple(int(grid.shape[0]) for grid in ride_grids)
        mesh = jnp.meshgrid(*ride_grids, indexing="ij")
        flat_cells = tuple(grid.ravel() for grid in mesh)
        solve_cell, stream_inputs = _cell_solver(
            solve_one_cell=solve_one_cell,
            flat_cells=flat_cells,
            cont_value_stack=cont_value_stack,
            cont_marginal_stack=cont_marginal_stack,
            cliff_savings_stack=cliff_savings_stack,
        )
        stacks = _map_ride_partitioned(
            func=solve_cell,
            xs=stream_inputs,
            requested_block_size=statics.cell_block_size,
        )
        value_arr, carry = _assemble_ride_carry(
            stacks=stacks,
            n_jumps=statics.n_published_jumps,
            liquid=liquid,
            ride_shape=ride_shape,
            liquid_axis_pos=schedule_spec.liquid_axis_pos,
            dtype=dtype,
        )
        return value_arr, carry

    return envelope_core


@dataclass(frozen=True)
class _NBEGMDiscreteSpec:
    """Build-time statics for a discrete-action regime with a smooth budget.

    The discrete action shifts cash-on-hand; the continuous consumption/savings
    subproblem is solved per discrete-action value by NBEGM and the discrete choice
    is taken by the upper envelope over the branch values.
    """

    coh_of_liquid_dag: Callable
    """Composed `coh` as a function of the liquid state, the discrete actions, and
    qualified params."""
    coh_param_names: tuple[str, ...]
    """Qualified parameter names `coh` reads (excluding the liquid state and the
    discrete actions)."""
    liquid_state_name: str
    """Name of the liquid state the budget varies in."""
    discrete_actions: DiscreteActionCodes
    """Each discrete action enveloped over, paired with its grid codes."""

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
) -> _NBEGMDiscreteSpec:
    """Collect the discrete actions of a smooth regime and their grid codes."""
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
    return _NBEGMDiscreteSpec(
        coh_of_liquid_dag=coh_dag,
        coh_param_names=coh_param_names,
        liquid_state_name=liquid_state_name,
        discrete_actions=discrete_actions,
    )


@dataclass(frozen=True)
class _NBEGMScheduleDiscreteSpec:
    """Build-time statics for a discrete action over a cliffed single-liquid budget.

    Each discrete-action value shifts cash-on-hand and the budget also carries a
    declared schedule (kinks/jumps) on the liquid state. Per action value the
    continuous subproblem is solved by the multi-interval EGM step honouring the
    schedule, and the discrete choice is taken by the upper envelope over the
    branch values.
    """

    coh_of_liquid_action_dag: Callable
    """Composed budget node as a function of the liquid state, the discrete actions,
    and qualified params."""
    coh_param_names: tuple[str, ...]
    """Qualified parameter names the budget reads (excluding the liquid state and the
    discrete actions)."""
    liquid_state_name: str
    """Name of the liquid (Euler) state the budget varies in."""
    discrete_actions: DiscreteActionCodes
    """Each discrete action enveloped over, paired with its grid codes."""
    threshold_param_names: tuple[str, ...]
    """Qualified parameter names of the schedule's thresholds (liquid breakpoints)."""
    breakpoint_kinds: tuple[str, ...]
    """Discontinuity kind per threshold, in the schedule's declared order."""

    @property
    def branch_bindings(self) -> tuple[MappingProxyType[ActionName, int], ...]:
        """One binding of every declared discrete action, per envelope branch."""
        return _branch_bindings(self.discrete_actions)


def _collect_nbegm_schedule_discrete_spec(
    *,
    context: SolverBuildContext,
    budget_target: FunctionName,
    continuous_state: StateName,
    post_decision_function: FunctionName,
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
    return _NBEGMScheduleDiscreteSpec(
        coh_of_liquid_action_dag=coh_dag,
        coh_param_names=coh_param_names,
        liquid_state_name=liquid_state_name,
        discrete_actions=discrete_actions,
        threshold_param_names=threshold_param_names,
        breakpoint_kinds=breakpoint_kinds,
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
    envelope_arithmetic: ComparisonArithmetic = "certified",
) -> Callable:
    """Build the discrete-envelope core over a cliffed single-liquid budget.

    Per discrete-action value the core recovers the schedule's per-interval affine
    cash-on-hand and the liquid breakpoints and solves that branch with the
    kind-appropriate step (reading the continuation jump-aware, so the solve is
    exact through recurring jumps). The discrete choice is then taken by the upper
    envelope over the branch values — the hard maximum, or the EV1 logsum under a
    taste-shock scale.
    """
    from _lcm.egm.nbegm_breakpoints import (  # noqa: PLC0415
        interval_midpoints,
        interval_segment_coefficients,
    )

    is_single_jump, is_multi_jump, is_mixed, jump_mask, flat_mask = (
        _schedule_kind_flags(spec.breakpoint_kinds)
    )
    order_sensitive = len(set(spec.breakpoint_kinds)) > 1

    from _lcm.egm.preferences import (  # noqa: PLC0415
        NEWTON_ACTION_FLOOR,
        get_preferences_builder,
        newton_action_ceiling,
    )

    build_preferences = get_preferences_builder(
        functions=functions,
        action_name=consumption_action,
        action_lower=NEWTON_ACTION_FLOOR,
        action_upper=newton_action_ceiling(savings_grid),
    )

    def core(
        *,
        liquid: Float1D,
        next_liquid_grid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        next_liquid: Float1D,
        marginal_return: Float1D,
        **params: FloatND,
    ) -> tuple[Float1D, EGMCarry]:
        preferences = build_preferences(params)
        coh_params = {name: params[name] for name in spec.coh_param_names}
        breakpoints = _sorted_thresholds(
            jnp.stack([params[name] for name in spec.threshold_param_names]),
            order_sensitive=order_sensitive,
        )
        midpoints = interval_midpoints(liquid_grid=liquid, breakpoints=breakpoints)
        values: list[Float1D] = []
        marginals: list[Float1D] = []
        for binding in spec.branch_bindings:

            def coh_of_liquid(
                scalar_liquid: FloatND,
                binding: MappingProxyType[ActionName, int] = binding,
            ) -> FloatND:
                return spec.coh_of_liquid_action_dag(
                    **{spec.liquid_state_name: scalar_liquid},
                    **{name: jnp.asarray(code) for name, code in binding.items()},
                    **coh_params,
                )

            coh_slopes, coh_intercepts = interval_segment_coefficients(
                schedule=coh_of_liquid, interval_midpoints=midpoints
            )
            branch_value, branch_marginal, _policy = _solve_cliffed_budget(
                next_value=next_value,
                next_marginal=next_marginal,
                liquid=liquid,
                next_liquid_grid=next_liquid_grid,
                savings_grid=savings_grid,
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
                arithmetic=envelope_arithmetic,
            )
            values.append(branch_value)
            marginals.append(branch_marginal)

        value, marginal = _discrete_envelope_over_branches(
            value_stack=jnp.stack(values),
            marginal_stack=jnp.stack(marginals),
            taste_shock_scale=taste_shock_scale,
        )
        carry = EGMCarry(
            endog_grid=liquid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(taste_shock_scale, dtype=value.dtype),
        )
        return value, carry

    return core


def _build_nbegm_discrete_core(
    *,
    savings_grid: Float1D,
    functions: EconFunctionsMapping,
    consumption_action: ActionName,
    discrete_spec: _NBEGMDiscreteSpec,
    taste_shock_scale: float,
    envelope_arithmetic: ComparisonArithmetic = "certified",
) -> Callable:
    """Build the jittable discrete-envelope core for one continuation target.

    Per discrete-action value the core recovers the smooth budget's affine cash-on-
    hand and solves the continuous subproblem with the multi-interval step, then
    takes the discrete choice by the upper envelope (`nbegm_discrete_envelope_step`).
    """
    from _lcm.egm.nbegm_breakpoints import affine_coefficients  # noqa: PLC0415
    from _lcm.egm.nbegm_step import (  # noqa: PLC0415
        nbegm_discrete_envelope_step,
    )
    from _lcm.egm.preferences import (  # noqa: PLC0415
        NEWTON_ACTION_FLOOR,
        get_preferences_builder,
        newton_action_ceiling,
    )

    build_preferences = get_preferences_builder(
        functions=functions,
        action_name=consumption_action,
        action_lower=NEWTON_ACTION_FLOOR,
        action_upper=newton_action_ceiling(savings_grid),
    )

    def core(
        *,
        liquid: Float1D,
        next_liquid_grid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        next_liquid: Float1D,
        marginal_return: Float1D,
        **params: FloatND,
    ) -> tuple[Float1D, EGMCarry]:
        preferences = build_preferences(params)
        coh_params = {name: params[name] for name in discrete_spec.coh_param_names}
        empty_breakpoints = jnp.zeros((0,), dtype=liquid.dtype)
        choices: list[dict[str, Float1D]] = []
        for binding in discrete_spec.branch_bindings:

            def coh_of_liquid(
                scalar_liquid: FloatND,
                binding: MappingProxyType[ActionName, int] = binding,
            ) -> FloatND:
                return discrete_spec.coh_of_liquid_dag(
                    **{discrete_spec.liquid_state_name: scalar_liquid},
                    **{name: jnp.asarray(code) for name, code in binding.items()},
                    **coh_params,
                )

            slope, intercept = affine_coefficients(coh_of_liquid)
            choices.append(
                {
                    "coh_slopes": jnp.reshape(slope, (1,)),
                    "coh_intercepts": jnp.reshape(intercept, (1,)),
                    "breakpoints": empty_breakpoints,
                }
            )
        value, marginal, _policy, _choice = nbegm_discrete_envelope_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            next_liquid_grid=next_liquid_grid,
            savings_grid=savings_grid,
            discount_factor=params["koopmans_aggregator__discount_factor"],
            preferences=preferences,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            choices=tuple(choices),
            taste_shock_scale=taste_shock_scale,
            arithmetic=envelope_arithmetic,
        )
        carry = EGMCarry(
            endog_grid=liquid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(taste_shock_scale, dtype=value.dtype),
        )
        return value, carry

    return core


def _assemble_ride_carry(
    *,
    stacks: tuple[FloatND, ...],
    n_jumps: int,
    liquid: Float1D,
    ride_shape: tuple[int, ...],
    liquid_axis_pos: int,
    dtype: Any,  # noqa: ANN401  # jnp dtype object
) -> tuple[FloatND, EGMCarry]:
    """Reshape the per-cell solve stacks into the value array and the carry.

    - With jump breakpoints, the cell solve returns the value at the liquid
      nodes plus the augmented carry rows (duplicated jump abscissae with
      one-sided limits) and the jump locations.
    - Without jumps, it returns plain liquid-grid rows and the carry sits on
      the shared broadcast grid.

    The published value array follows the productmap state order, so the
    liquid axis moves from the working layout's trailing position to its
    canonical index. The carry keeps the working layout (ride axes leading
    the row axis): it is read back only by `bind_continuation`, which
    produced it, so the round-trip stays self-consistent.
    """
    n_liquid = liquid.shape[0]
    if n_jumps:
        (
            value_stack,
            endog_stack,
            row_value_stack,
            row_marginal_stack,
            breakpoint_stack,
        ) = stacks
        n_row = n_liquid + 2 * n_jumps
        carry_rows = (
            endog_stack.reshape(*ride_shape, n_row).astype(dtype),
            row_value_stack.reshape(*ride_shape, n_row).astype(dtype),
            row_marginal_stack.reshape(*ride_shape, n_row).astype(dtype),
        )
        breakpoint_rows = breakpoint_stack.reshape(*ride_shape, n_jumps).astype(dtype)
    else:
        value_stack, marginal_stack = stacks
        carry_rows = (
            jnp.broadcast_to(liquid, (*ride_shape, n_liquid)).astype(dtype),
            value_stack.reshape(*ride_shape, n_liquid).astype(dtype),
            marginal_stack.reshape(*ride_shape, n_liquid).astype(dtype),
        )
        breakpoint_rows = None
    value_arr = jnp.moveaxis(
        value_stack.reshape(*ride_shape, n_liquid), -1, liquid_axis_pos
    )
    carry = EGMCarry(
        endog_grid=carry_rows[0],
        value=carry_rows[1],
        marginal_utility=carry_rows[2],
        taste_shock_scale=jnp.asarray(0.0, dtype=dtype),
        breakpoints=breakpoint_rows,
    )
    return value_arr, carry


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
    ride_along_state_names: tuple[StateName, ...],
) -> EGMCarry:
    """Shard the ride-along carry template over its distributed ride axes."""
    return shard_carry_template(
        template=template,
        grids=grids,
        leading_axis_names=ride_along_state_names,
    )


def _build_ride_along_carry_template(
    *, liquid_grid: Float1D, ride_shape: tuple[int, ...], n_breakpoints: int
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
    # weakly ascending and all-finite.
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
