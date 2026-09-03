"""The N-NB-EGM solver: nested outer search around an NB-EGM inner solve.

`NNBEGM` runs the NEGM-style outer keeper/adjuster search over a durable
margin with an inner `NBEGM` consumption-saving solve, so declared liquid
kinks, jumps, and hard constraints keep their exact NB-EGM treatment inside
every outer candidate. The regime owns both margins' DAG role names; the
public solver contains numerical configuration only, and a private bound
companion carries the resolved names into the kernels.

The kernel-building imports are function-local so the public `lcm.solvers`
façade stays a thin re-export that pulls in no numerical engine modules.
"""

import functools
import inspect
import logging
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from dags import concatenate_functions

from _lcm.beartype_conf import REGIME_CONF
from _lcm.constraints.routes import ConstraintRoute
from _lcm.continuation import EGMContinuationLayout
from _lcm.egm.branch_aggregation import (
    DeterministicOuterMaximum,
    OuterBranchAggregator,
    UniformObservedFixedCost,
)
from _lcm.egm.carry import EGMCarry
from _lcm.egm.nested_published_policy import (
    NestedEGMSimPolicy,
    OuterPolicyBank,
    derive_inner_sim_policy,
)
from _lcm.egm.numeric_inverse import numeric_inverse_marginal_utility
from _lcm.egm.outer_candidates import (
    OuterCandidateResult,
    build_outer_candidate_bank,
)
from _lcm.egm.outer_carry import collapse_continuous_candidate_bank
from _lcm.egm.outer_inversion import (
    DeclaredOuterInverse,
    abstract_like,
    certify_declared_outer_inverse,
    invert_declared_outer_target,
)
from _lcm.egm.outer_refinement import refine_outer_mesh
from _lcm.egm.outer_replay_capability import (
    OuterReplayCapability,
    fail_if_continuous_outer_replay_is_unsupported,
    resolve_outer_replay_capability,
)
from _lcm.egm.outer_search import AdaptiveOuterMesh, FiniteOuterGrid, OuterSearch
from _lcm.egm.published_policy import (
    EGMSimPolicy,
    NBEGMGridPolicy,
    NNBEGMSimPolicy,
)
from _lcm.engine import ParamCheck, StateActionSpace
from _lcm.execution.core_program import (
    CoreArgumentBuilder,
    CoreBuildContext,
    CoreProgram,
    ProgramScope,
    core_program_graph,
    select_programs,
)
from _lcm.grids import ContinuousGrid, DiscreteGrid, Grid
from _lcm.solution.contract import (
    ConstraintRouteContext,
    ContinuationPayload,
    GeneratedReplayAuthority,
    KernelResult,
    PeriodKernel,
    SimulationPolicy,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    SolverModelContext,
    TwoMarginSolver,
    _BoundLiquidMargin,
    _BoundOuterContinuousMargin,
)
from _lcm.solution.nbegm import (
    NBEGM,
    _BoundNBEGM,
    _RideAlongNBEGMPeriodKernel,
    _validate_nbegm_case_piece_declarations,
    proved_post_decision_of,
)
from _lcm.solution.negm import (
    _fail_if_outer_grid_is_stochastic,
    _stack_carry_template,
    _with_no_adjustment_outer_function,
    _with_outer_post_decision,
    _without_outer_post_decision,
)
from _lcm.solution.periodization import (
    resolve_solver_build_context,
    restrict_solver_build_context_to_period_group,
    solver_period_group_key,
)
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from _lcm.typing import (
    EconFunctionArg,
    EconFunctionsMapping,
    FlatParams,
    RegimeName,
)
from lcm.ages import AgeGrid
from lcm.exceptions import (
    ModelInitializationError,
    RegimeInitializationError,
    UnrepresentableOuterCandidateError,
)
from lcm.typing import (
    ActionName,
    BoolND,
    Float1D,
    FloatND,
    FunctionName,
    IntND,
    StateName,
)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class NNBEGM(TwoMarginSolver):
    """N-NB-EGM — an outer durable grid search over inner 1-D NB-EGM solves.

    The regime carries two continuous margins. The outer post-decision margin
    (a durable/illiquid stock) is selected by a finite search: a *keeper* holds
    it unchanged for free, and an *adjuster* sweep binds it to each exogenous
    outer-grid node. Conditional on the outer node, the remaining problem is a
    one-dimensional consumption-saving solve on the liquid state, handled by
    the inner `NBEGM` config — so declared liquid kinks, jumps, and hard
    constraints keep their exact NB-EGM treatment inside every outer candidate.

    The outer axis is collapsed by `V = max(V_keeper, max_j W_j)`; the solution
    is exact relative to the finite outer candidate set (grid plus keeper). The
    published continuation is the pointwise upper envelope of the candidates'
    carry rows on the shared liquid state grid — a finite-grid (bridged) outer
    envelope, so the inner config must not publish jump-topology rows.

    No outer Euler condition is assumed: adjustment frictions and caps make a
    second Euler inversion unreliable, which is the reason to nest rather than
    to solve two coupled first-order conditions (that case belongs to
    the two-continuous-state solver published with its own paper).
    """

    inner: NBEGM
    """Numerical configuration of the inner 1-D NB-EGM solve."""

    outer_search: OuterSearch
    """How the outer margin's candidates are generated and refined.

    `FiniteOuterGrid` reproduces the historical finite-candidate behavior;
    `AdaptiveOuterMesh` is the canonical continuous-outer approximation. The
    strategy carries its own numerics, including any batch size."""

    def __post_init__(self) -> None:
        _fail_if_inner_is_not_nbegm(self.inner)
        search = self.outer_search
        match search:
            case FiniteOuterGrid():
                _fail_if_outer_grid_is_stochastic(search.grid)
            case AdaptiveOuterMesh():
                _fail_if_outer_grid_is_stochastic(search.initial_grid)
            case _:
                pass

    def _fail_if_aggregation_is_unsupported(
        self, aggregator: OuterBranchAggregator
    ) -> None:
        """State which declared aggregations this configuration can execute.

        The declaration itself belongs to the regime's outer margin; the solver
        answers only whether its kernels can aggregate it here, under the
        configured outer search.
        """
        _fail_if_aggregator_unsupported(aggregator)
        if isinstance(aggregator, UniformObservedFixedCost) and not isinstance(
            self.outer_search, AdaptiveOuterMesh
        ):
            msg = (
                "UniformObservedFixedCost aggregates the keeper/adjuster "
                "branches through the continuous collapse, so a regime "
                "declaring `adjustment_cost=UniformObservedFixedCost(...)` "
                "needs `outer_search=AdaptiveOuterMesh(...)` on its NNBEGM."
            )
            raise RegimeInitializationError(msg)

    def _with_margins(
        self,
        *,
        liquid: _BoundLiquidMargin,
        outer: _BoundOuterContinuousMargin,
    ) -> _BoundNNBEGM:
        """Bind both regime-owned margins into a private runtime config."""
        aggregator = outer.adjustment_cost or DeterministicOuterMaximum()
        self._fail_if_aggregation_is_unsupported(aggregator)
        kwargs = {
            field.name: getattr(self, field.name)
            for field in fields(NNBEGM)
            if field.name != "inner"
        }
        inner = self.inner._with_liquid_margin(liquid)  # noqa: SLF001
        return _BoundNNBEGM(
            **kwargs,
            inner=inner,
            outer_action=outer.action,
            outer_state=outer.state,
            outer_post_decision=outer.post_decision_state,
            outer_no_adjustment_candidate=outer.no_adjustment,
            branch_aggregator=aggregator,
        )

    @property
    def requires_continuation(self) -> bool:
        """NNBEGM runs an inner NB-EGM solve that inverts the Euler equation."""
        return True

    @property
    def supports_nonlinear_certainty_equivalent(self) -> bool:
        """The inner NB-EGM solve inverts the recursive Euler equation."""
        return self.inner.supports_nonlinear_certainty_equivalent

    @property
    def egm_continuation_layout(self) -> EGMContinuationLayout:
        """Describe the carry published by the configured outer search."""
        if isinstance(self.outer_search, FiniteOuterGrid):
            return replace(
                self.inner.egm_continuation_layout,
                n_stacked_candidates=(
                    int(self.outer_search.grid.to_jax().shape[0]) + 1
                ),
            )
        return self.inner.egm_continuation_layout

    @property
    def publishes_simulation_policy(self) -> bool:
        """Request collection of the self-describing nested replay payload.

        Canonical processing also installs `NNBEGMPolicyRead` as the consuming-route
        marker used by labelled-result preflight. This solver-side declaration keeps
        policy collection explicit for legacy automatic solve/simulate paths; the
        payload itself still names the actions, state, and search settings its reader
        needs.
        """
        return True

    def build_constraint_routes(
        self, *, context: ConstraintRouteContext
    ) -> tuple[ConstraintRoute, ...]:
        """Declare the two routes the nested solve walks in the solve phase.

        The outer margin is selected by a finite search whose two branches reach
        the inner solve through *different* function pools: the adjuster's has
        the outer post-decision function removed and its name promoted to a
        bound parameter, the keeper's has that function replaced by the
        no-adjustment law. A site carries the pool it is entered with, so the
        two branches are two routes rather than one described twice.

        Both inherit the inner kernel's declaration of what can happen along
        them, since the inner case-piece solve is the only place a liquid
        constraint could be met and it evaluates none.
        """
        from _lcm.egm.nbegm_constraint_boundaries import (  # noqa: PLC0415
            build_nbegm_feasibility_boundary_compiler,
        )
        from _lcm.egm.nbegm_routes import case_piece_routes  # noqa: PLC0415

        if context.phase == "simulate":
            return case_piece_routes(
                context=context,
                post_decision_function=proved_post_decision_of(solver=self.inner),
                solver_path=("nnbegm",),
            )
        bound = cast("_BoundNNBEGM", self)
        boundary_compilers = (
            build_nbegm_feasibility_boundary_compiler(
                liquid_state=bound.inner.continuous_state
            ),
        )
        return tuple(
            route
            for branch, pool in (
                (
                    "adjuster",
                    _without_outer_post_decision(
                        functions=context.functions,
                        outer_post_decision=bound.outer_post_decision,
                    ),
                ),
                (
                    "keeper",
                    _with_no_adjustment_outer_function(
                        functions=context.functions,
                        durable_state=bound.outer_state,
                        outer_post_decision=bound.outer_post_decision,
                        no_adjustment_func=(
                            context.functions[bound.outer_no_adjustment_candidate]
                            if bound.outer_no_adjustment_candidate is not None
                            else None
                        ),
                    ),
                ),
            )
            for route in case_piece_routes(
                context=context,
                post_decision_function=proved_post_decision_of(solver=bound.inner),
                solver_path=("nnbegm", branch),
                function_pool=pool,
                boundary_compilers=boundary_compilers,
            )
        )

    def validate_model(self, *, context: SolverModelContext) -> None:
        """Validate the nested contract, kernel grids, and borrowing limit."""
        from _lcm.egm.nnbegm_validation import (  # noqa: PLC0415
            validate_nnbegm_regime,
        )
        from _lcm.egm.validation import (  # noqa: PLC0415
            fail_if_declared_lower_bound_disagrees_with_the_grid,
            fail_if_kernel_grids_withhold_their_points,
        )

        user_regime = context.user_regimes[context.regime_name]
        _fail_if_nnbegm_phase_variation(
            regime_name=context.regime_name,
            variations=context.phase_variation_paths,
        )
        validate_nnbegm_regime(
            regime_name=context.regime_name,
            user_regime=user_regime,
        )
        bound = cast("_BoundNNBEGM", self)
        outer_state = bound.outer_state
        liquid = bound.inner.continuous_state
        kernel_grids: dict[str, Grid] = {
            "inner savings grid": bound.inner.savings_grid,
            f"grid of the outer state '{outer_state}'": cast(
                "Grid", user_regime.states[outer_state]
            ),
            f"grid of the liquid state '{liquid}'": cast(
                "Grid", user_regime.states[liquid]
            ),
        }
        match bound.outer_search:
            case FiniteOuterGrid():
                kernel_grids["outer grid"] = bound.outer_search.grid
            case AdaptiveOuterMesh():
                kernel_grids["outer grid"] = bound.outer_search.initial_grid
        fail_if_kernel_grids_withhold_their_points(
            grids=kernel_grids,
            regime_name=context.regime_name,
            solver_name="NNBEGM",
        )
        _fail_if_the_outer_search_leaves_the_outer_state_domain(
            regime_name=context.regime_name,
            outer_state=outer_state,
            outer_state_grid=cast("Grid", user_regime.states[outer_state]),
            outer_search=bound.outer_search,
        )
        fail_if_declared_lower_bound_disagrees_with_the_grid(
            regime_name=context.regime_name,
            user_regime=user_regime,
            solver=bound.inner,
            solver_name="NNBEGM",
        )
        _validate_nbegm_case_piece_declarations(context=context, solver=bound.inner)

    def validate_build(self, *, context: SolverBuildContext) -> None:
        """Apply the inner solver's build-time gates to the liquid margin.

        The inner NB-EGM kernels run unchanged inside every outer candidate, so
        a piece that hides branching breaks the inner Euler inversion here
        exactly as it would under a bare `NBEGM`, and declared taste shocks are
        ignored by the inner envelopes here exactly as they would be there. The
        smoothness gate is pointed at the inner spec's Euler state rather than
        the regime's first state, because the regime also carries the outer
        margin the pieces never see.
        """
        bound = cast("_BoundNNBEGM", self)
        bound.inner.validate_build(context=context)

    def build_period_kernels(  # noqa: PLR0915
        self, *, context: SolverBuildContext
    ) -> SolutionKernels:
        r"""Build one nested period adapter per period, wrapping inner kernels.

        Mirrors the NEGM keeper/adjuster split with an NB-EGM inner:

        - the *adjuster* strips the outer post-decision transition and admits
          the outer value as a flat param bound per outer-grid node;
        - the *keeper* injects $s_t^\textit{post-dec} = keep(\textit{durable}_t)$
          into the econ functions, so the durable becomes a genuine passive
          ride-along state.
        """
        bound = cast("_BoundNNBEGM", self)
        # The adjuster's outer post-decision arrives per outer-grid node as a
        # bound param, so the function declaring the chosen stock leaves the
        # inner DAG — leaving it in would let the inner scope check walk through
        # it to the outer action, which is exactly what binding the node
        # removes.
        #
        # The durable's own law of motion stays exactly as the regime declares
        # it. It reads the post-decision, which is that bound leaf here, so it
        # is decision-independent without being replaced — and a declared
        # `next_<durable>` $= (1 - \delta)\, s_t^\textit{post-dec}$ is therefore
        # the stock the continuation is read at, not the raw node the outer
        # search picked.
        grouped_periods: dict[Hashable, list[int]] = {}
        for period in context.regimes_to_active_periods[context.regime_name]:
            targets = (
                ()
                if period == context.solution_reachability.n_periods - 1
                else context.solution_reachability.targets(
                    period=period, source=context.regime_name
                )
            )
            key = solver_period_group_key(
                context=context,
                period=period,
                continuation_targets=targets,
                solver_path=("nnbegm",),
            )
            grouped_periods.setdefault(key, []).append(period)

        adjuster_by_period: dict[int, _RideAlongNBEGMPeriodKernel] = {}
        keeper_by_period: dict[int, _RideAlongNBEGMPeriodKernel] = {}
        resolved_by_period: dict[int, SolverBuildContext] = {}
        outer_target_function_by_period: dict[int, Callable] = {}
        grouped_param_checks = []
        keeper_continuation_spec = None
        for periods in grouped_periods.values():
            # The complete key above has already established that every period in
            # this group may share one concrete inner build. Resolve that pool once,
            # then narrow only the source regime's active-period tuple before handing
            # it to NBEGM. Otherwise the inner builder pairs this representative pool
            # with foreign periods and retains checks for combinations the model never
            # evaluates. The target regimes' lifecycle metadata remains intact.
            group_periods = tuple(periods)
            representative_period = group_periods[0]
            resolved = resolve_solver_build_context(
                context=context, period=representative_period
            )
            group_context = restrict_solver_build_context_to_period_group(
                context=resolved,
                periods=group_periods,
            )
            adjuster_context = replace(
                group_context,
                functions=_without_outer_post_decision(
                    functions=group_context.functions,
                    outer_post_decision=bound.outer_post_decision,
                ),
                flat_param_names=context.flat_param_names | {bound.outer_post_decision},
                constraint_plan=(
                    None
                    if context.constraint_plan is None
                    else context.constraint_plan.for_solver_path(
                        solver_path=("nnbegm", "adjuster")
                    )
                ),
            )
            adjuster_group = bound.inner.build_period_kernels(context=adjuster_context)
            no_adjustment_func = (
                group_context.functions[bound.outer_no_adjustment_candidate]
                if bound.outer_no_adjustment_candidate is not None
                else None
            )
            # The keeper computes the post-decision from the durable leaf instead
            # of taking it as a bound param, so the declared law again stands.
            keeper_context = replace(
                group_context,
                functions=_with_no_adjustment_outer_function(
                    functions=group_context.functions,
                    durable_state=bound.outer_state,
                    outer_post_decision=bound.outer_post_decision,
                    no_adjustment_func=no_adjustment_func,
                ),
                constraint_plan=(
                    None
                    if context.constraint_plan is None
                    else context.constraint_plan.for_solver_path(
                        solver_path=("nnbegm", "keeper")
                    )
                ),
            )
            keeper_group = bound.inner.build_period_kernels(context=keeper_context)
            replay_targets = [bound.outer_post_decision]
            if bound.outer_no_adjustment_candidate is not None:
                replay_targets.append(bound.outer_no_adjustment_candidate)
            outer_target_function = concatenate_functions(
                functions=group_context.functions,
                targets=replay_targets,
                return_type="dict",
                set_annotations=True,
            )
            for period in group_periods:
                resolved_by_period[period] = group_context
                adjuster_by_period[period] = _ride_along_inner_kernel(
                    kernel=adjuster_group.period_kernels[period], role="adjuster"
                )
                keeper_by_period[period] = _ride_along_inner_kernel(
                    kernel=keeper_group.period_kernels[period], role="keeper"
                )
                outer_target_function_by_period[period] = outer_target_function
            grouped_param_checks.extend(adjuster_group.param_checks)
            grouped_param_checks.extend(keeper_group.param_checks)
            if keeper_continuation_spec is None:
                keeper_continuation_spec = keeper_group.continuation_spec
        inner_template = (
            None
            if keeper_continuation_spec is None
            else keeper_continuation_spec.template
        )
        _fail_if_inner_carry_rows_not_grid_aligned(inner=bound.inner)
        if not (
            context.constraint_plan and context.constraint_plan.compiled_boundaries
        ):
            _fail_if_nnbegm_carry_publishes_topology_rows(template=inner_template)
        search = self.outer_search
        build_kernel: Callable[..., _NNBEGMPeriodKernel]
        match search:
            case FiniteOuterGrid():
                build_kernel = functools.partial(
                    _FiniteNNBEGMPeriodKernel, outer_search=search
                )
                outer_grid_values = search.grid.to_jax()
                outer_batch_size = search.batch_size
                template = _stack_carry_template(
                    template=inner_template,
                    n_candidates=int(outer_grid_values.shape[0]) + 1,
                )
            case AdaptiveOuterMesh():
                build_kernel = functools.partial(
                    _AdaptiveNNBEGMPeriodKernel, outer_search=search
                )
                outer_grid_values = search.initial_grid.to_jax()
                outer_batch_size = search.batch_size
                # The continuous collapse republishes a policy-free carry. Its
                # nested simulation payload reads the raw keeper/adjuster rows
                # instead, so the standalone inner policy leaf must not leak into
                # the cross-period continuation template.
                template = (
                    replace(inner_template, policy=None)
                    if isinstance(inner_template, EGMCarry)
                    else inner_template
                )
            case _:
                msg = (
                    f"NNBEGM outer search strategy {type(search).__name__} "
                    "is not wired into the period kernels; use "
                    "FiniteOuterGrid or AdaptiveOuterMesh."
                )
                raise RegimeInitializationError(msg)
        # The inner Euler-state slots the outer kernel needs come from the bound
        # inner config: the regime's liquid margin already resolved them, so no
        # normalization over inner solver types is left to do.
        spec = bound.inner
        inner_action = _nnbegm_inner_action(
            context=context, outer_action=bound.outer_action
        )
        # Carry-row axis names, in the carry contract's order: discrete states
        # first (V state order), then passive continuous states (every
        # continuous state except the inner Euler axis). Used to derive the
        # published inner policies for the nested simulation reader.
        row_discrete_state_names = tuple(
            name
            for name in context.state_action_space.state_names
            if isinstance(context.grids[name], DiscreteGrid)
        )
        row_passive_state_names = tuple(
            name
            for name in context.state_action_space.state_names
            if isinstance(context.grids[name], ContinuousGrid)
            and name != spec.continuous_state
        )
        # A domain endpoint is a node value, so it is the solved period's own.
        # With an age-specialized outer grid the representative age's endpoints
        # are the wrong ones everywhere else: a stock only the later ages hold
        # would be judged out of domain and dropped, and a stock past a
        # narrower age's edge would be admitted with no value function to read
        # it on.
        representative_outer_values = context.grids[bound.outer_state].to_jax()

        def _outer_state_domain_at(period: int) -> tuple[float, float]:
            per_period = context.period_to_state_nodes
            nodes = (
                representative_outer_values
                if per_period is None
                else per_period.get(period, {}).get(
                    bound.outer_state, representative_outer_values
                )
            )
            return float(nodes[0]), float(nodes[-1])

        branch_aggregation_by_period = {
            period: _resolve_branch_fixed_cost(
                aggregator=bound.branch_aggregator,
                context=resolved_by_period[period],
            )
            for period in adjuster_by_period
        }
        period_kernels = MappingProxyType(
            {
                period: build_kernel(
                    keeper_kernel=keeper_by_period[period],
                    adjuster_kernel=adjuster_kernel,
                    regime_name=context.regime_name,
                    outer_grid_values=outer_grid_values,
                    outer_state_name=bound.outer_state,
                    outer_state_domain=_outer_state_domain_at(period),
                    outer_post_decision=bound.outer_post_decision,
                    outer_target_function=outer_target_function_by_period[period],
                    outer_batch_size=outer_batch_size,
                    outer_action=bound.outer_action,
                    inner_action=inner_action,
                    resources_target=spec.budget_target,
                    savings_lower_bound=float(spec.savings_grid.to_jax()[0]),
                    liquid_grid_values=context.grids[spec.continuous_state].to_jax(),
                    liquid_state_name=spec.continuous_state,
                    outer_no_adjustment_name=bound.outer_no_adjustment_candidate,
                    inverse_marginal=_nested_inverse_marginal(
                        context=resolved_by_period[period],
                        rows_on_state_grid=self.egm_continuation_layout.rows_share_state_grid,
                        inner_action=inner_action,
                        savings_top=float(spec.savings_grid.to_jax()[-1]),
                    ),
                    row_discrete_state_names=row_discrete_state_names,
                    row_passive_state_names=row_passive_state_names,
                    inner_discrete_action_names=tuple(
                        context.state_action_space.discrete_actions
                    ),
                    replay_functions=resolved_by_period[period].functions,
                    replay_bindable_names=(
                        context.flat_param_names
                        | {"period", "age"}
                        | set(context.state_action_space.state_names)
                    ),
                    replay_state_names=frozenset(
                        context.state_action_space.state_names
                    ),
                    branch_fixed_cost=branch_aggregation_by_period[period][0],
                    branch_scale_function=branch_aggregation_by_period[period][1],
                )
                for period, adjuster_kernel in adjuster_by_period.items()
            }
        )
        # The bridged outer envelope folds candidates pointwise on shared inner
        # abscissae. Plain rows use the liquid grid; compiled feasibility augments
        # keeper and adjuster identically, and the forwarded spec retains that
        # one-sided geometry for the parent read.
        # The fixed cost's scale is a per-period scalar read off the params,
        # so its supported range can only be checked once params exist.
        scale_check = _branch_scale_check(
            regime_name=context.regime_name,
            ages=context.ages,
            branch_aggregation_by_period=branch_aggregation_by_period,
        )
        return SolutionKernels(
            period_kernels=period_kernels,
            continuation_spec=(
                None
                if keeper_continuation_spec is None
                else replace(keeper_continuation_spec, template=template)
            ),
            # Both inner margins are solved by the inner solver, so both sets of
            # parameter-dependent preconditions still apply to this regime.
            param_checks=(
                tuple(grouped_param_checks)
                if scale_check is None
                else (*grouped_param_checks, scale_check)
            ),
        )


@dataclass(frozen=True, kw_only=True)
class _BoundNNBEGM(NNBEGM):
    """Internal N-NB-EGM config with both regime margins resolved."""

    inner: _BoundNBEGM
    outer_action: ActionName
    outer_state: StateName
    outer_post_decision: FunctionName
    outer_no_adjustment_candidate: FunctionName | None
    branch_aggregator: OuterBranchAggregator
    """The fold resolved from the regime's declared `adjustment_cost`, with the
    deterministic maximum standing in where no cost was declared."""


def _fail_if_nnbegm_phase_variation(
    *,
    regime_name: RegimeName,
    variations: tuple[str, ...],
) -> None:
    """Reject phase variation before NNBEGM period kernels are constructed.

    NNBEGM publishes only the keeper-plus-outer-grid candidates solved during
    backward induction. A genuinely phase-varying declaration would require a
    separate simulate-phase policy over that same candidate set; generic
    action-grid maximization is not an equivalent fallback.
    """
    if not variations:
        return
    raise ModelInitializationError(
        f"NNBEGM replay capability for regime {regime_name!r} does not support "
        "phase variation. The solve policy ranks keeper plus NNBEGM.outer_grid "
        "candidates, so simulation cannot silently fall back to generic "
        "action-grid maximization when declarations differ between solve and "
        f"simulate. Unsupported slots: {list(variations)}. Any carried-only "
        "state is phase-varying by construction. Use identical declaration "
        "objects in both phases, remove carried-only state, or use GridSearch "
        "until phase-specific NNBEGM replay is implemented."
    )


def _conditional_nnbegm_banks(
    *,
    policy: NBEGMGridPolicy,
    collapsed_value: FloatND,
    state_names: tuple[StateName, ...],
    discrete_action_names: tuple[ActionName, ...],
    branch_codes: IntND | None,
) -> tuple[FloatND, FloatND]:
    """Validate and return one outer candidate's conditional inner banks."""
    if policy.state_names != state_names:
        raise ValueError("NNBEGM candidate policies disagree on state-axis order.")
    if policy.discrete_action_names != discrete_action_names:
        raise ValueError("NNBEGM candidate policies disagree on discrete-action order.")
    if discrete_action_names:
        if (
            policy.branch_inner_action is None
            or policy.branch_value is None
            or policy.branch_discrete_actions is None
            or branch_codes is None
        ):
            raise ValueError("NNBEGM discrete replay requires every inner branch bank.")
        codes_match = bool(
            jax.device_get(
                jnp.array_equal(policy.branch_discrete_actions, branch_codes)
            )
        )
        if not codes_match:
            raise ValueError(
                "NNBEGM candidate policies disagree on discrete branch codes."
            )
        if (
            policy.branch_inner_action.shape != policy.branch_value.shape
            or policy.branch_inner_action.shape[0]
            != policy.branch_discrete_actions.shape[0]
        ):
            raise ValueError(
                "NNBEGM inner branch action/value/code banks are misaligned."
            )
        return policy.branch_inner_action, policy.branch_value
    if any(
        field is not None
        for field in (
            policy.branch_inner_action,
            policy.branch_value,
            policy.branch_discrete_actions,
        )
    ):
        raise ValueError(
            "NNBEGM smooth replay received unexpected discrete branch banks."
        )
    return policy.action[None, ...], collapsed_value[None, ...]


@dataclass(frozen=True, kw_only=True)
class _OuterNodeBinding:
    """The outer post-decision value one republished program is lowered with."""

    regime_name: RegimeName
    """Name of the regime whose flat params receive the bound node."""

    outer_post_decision: FunctionName
    """Name of the outer post-decision function the node is bound as."""

    value: FloatND
    """The outer node value, a scalar."""


@dataclass(frozen=True, kw_only=True)
class _NestedArgumentBuilder:
    """Build one republished inner program's arguments from the nested context.

    The keeper program takes the period's context unchanged. The adjuster program
    binds the outer post-decision at the first outer node before delegating, so
    the lowered program has exactly the shape every per-node call of the outer
    search traces.
    """

    inner: CoreArgumentBuilder
    """The inner program's own argument builder."""

    outer_node: _OuterNodeBinding | None
    """The node bound into the flat params, or `None` for the keeper."""

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
        """Return the inner program's arguments for the (possibly bound) context."""
        if self.outer_node is None:
            return self.inner(context)
        bound_flat_params = _with_outer_post_decision(
            flat_params=cast("FlatParams", context.flat_params),
            regime_name=self.outer_node.regime_name,
            outer_post_decision=self.outer_node.outer_post_decision,
            value=self.outer_node.value,
        )
        return self.inner(replace(context, flat_params=bound_flat_params))


@dataclass(frozen=True, kw_only=True)
class _NNBEGMPeriodKernel:
    """The NNBEGM period kernel — a keeper plus an adjuster outer search.

    Holds two inner ride-along NB-EGM period kernels and the exogenous outer
    grid. The kernel's native program graph republishes each inner kernel's
    replay program under a `keeper:`/`adjuster:` prefix, with the inner output
    roles and requirements and a builder that hands the inner program its own
    context; the prefix is stripped when the inner kernel is called. Both
    republished programs are dispatched under every retention, because the
    nested collapse reads the inner policy banks whether or not a replay
    artifact is kept.

    Calling it runs the keeper once, settles the period's replay capability,
    and hands both to the outer search of the concrete subclass:

    - `_FiniteNNBEGMPeriodKernel` runs the adjuster once per outer-grid node,
      collapses the outer axis by `V = max(V_keeper, max_j W_j)`, and retains
      the complete finite candidate identities for exact replay.
    - `_AdaptiveNNBEGMPeriodKernel` refines the shared outer mesh adaptively and
      collapses continuously.

    Either publishes the pointwise (bridged) upper envelope of the candidates'
    carry rows on the shared liquid state grid. The kernel is non-jitted: it
    dispatches the shared compiled inner programs, matching `_NEGMPeriodKernel`.
    """

    keeper_kernel: _RideAlongNBEGMPeriodKernel
    """The keeper inner kernel — a passive per-durable-state NB-EGM."""

    adjuster_kernel: _RideAlongNBEGMPeriodKernel
    """The adjuster inner kernel whose compiled program is swept."""

    regime_name: RegimeName
    """Name of the regime whose flat params the outer node binds into."""

    outer_grid_values: FloatND
    r"""Exogenous grid over the outer post-decision margin $s_t^\textit{post-dec}$."""

    outer_state_name: StateName
    """Name of the state-specific keeper source."""

    outer_state_domain: tuple[float, float]
    """Endpoints of the outer state's declared grid. A recovered stock outside
    them has no value function, so the candidate that reaches one is dropped."""

    outer_post_decision: FunctionName
    """Name of the outer post-decision function bound per outer-grid node."""

    outer_target_function: Callable
    """Resolved solve-phase DAG used to recover the outer action bank."""

    outer_batch_size: int
    """Outer-grid nodes solved per chunk before folding into the running
    maximum; `0` solves every node at once."""

    outer_action: ActionName
    """The regime's outer continuous action (published for the nested
    simulation reader)."""

    inner_action: ActionName
    """The regime's inner continuous action (the consumption the published
    inner policies map resources to)."""

    resources_target: FunctionName
    """The inner budget node the published policy rows are read at."""

    savings_lower_bound: float
    """Lower bound of the inner savings grid (the intrinsic budget check of
    the simulation policy read)."""

    liquid_grid_values: Float1D
    """The inner Euler (liquid) state grid — the shared abscissae the inner
    NB-EGM's published carry rows are re-read on
    (`carry_rows_share_state_grid`)."""

    liquid_state_name: StateName
    """Name of the inner Euler (liquid) state (published for the nested
    simulation reader's row query)."""

    outer_no_adjustment_name: FunctionName | None
    """The keeper's no-adjustment candidate function name, or `None` when
    keeping holds the current durable unchanged (published for the nested
    simulation reader's keeper-action recovery)."""

    inverse_marginal: Callable[..., FloatND] | None
    """The regime's closed-form inverse marginal utility with
    `marginal_continuation` as its only free parameter, or `None` when
    unavailable — then no nested simulation payload is derived and simulate
    keeps the grid-argmax path."""

    row_discrete_state_names: tuple[StateName, ...]
    """Names of the carry rows' leading discrete-state axes, in axis order."""

    row_passive_state_names: tuple[StateName, ...]
    """Names of the carry rows' passive continuous-state axes (every
    continuous state except the inner Euler state), after the discrete
    states."""

    inner_discrete_action_names: tuple[ActionName, ...]
    """The regime's discrete action names. When non-empty the inner solve makes
    a discrete choice whose winning branch is collapsed out of the published
    carry rows (`derive_inner_sim_policy` cannot recover which branch won
    off-grid), so the nested payload is NOT published and simulation keeps the
    grid-argmax path. Empty for a continuous-only regime, where publication
    proceeds."""

    replay_functions: EconFunctionsMapping
    """The regime's processed functions, read for the signatures a simulation
    replay must be able to bind. NNBEGM refuses any phase variation before the
    kernels are built, so these are the declarations simulation reads too."""

    replay_bindable_names: frozenset[str]
    """Everything a replay can supply at a realized state: the regime's states,
    its flat parameter names (from the params template, so the verdict does not
    depend on one call's params), and `period`/`age`."""

    replay_state_names: frozenset[StateName]
    """The states a replay reads at each subject."""

    branch_fixed_cost: UniformObservedFixedCost | None
    """The uniform observed fixed-cost aggregator, or `None` for the
    deterministic keeper/adjuster maximum."""

    branch_scale_function: Callable[..., FloatND] | None
    """The fixed cost's scale function, arguments restricted to
    `period`/`age`/flat params (resolved per period at call time)."""

    _core_programs: Mapping[str, CoreProgram] = field(
        init=False, repr=False, compare=False
    )
    """The republished graph, `keeper:replay` then `adjuster:replay`; derived
    from the inner kernels at construction."""

    def __post_init__(self) -> None:
        """Republish the inner replay programs under the role prefixes.

        The keeper and adjuster are distinct traced programs built from
        different contexts, so each (role, program) pair keeps its own
        compilation key. The nested solve reads the inner policy banks under
        every retention, so of each retention-scoped inner graph exactly the
        programs a replay-retaining solve dispatches are republished, with
        scope `ANY`.
        """
        programs: dict[str, CoreProgram] = {}
        for role, kernel, outer_node in (
            ("keeper", self.keeper_kernel, None),
            (
                "adjuster",
                self.adjuster_kernel,
                _OuterNodeBinding(
                    regime_name=self.regime_name,
                    outer_post_decision=self.outer_post_decision,
                    value=self.outer_grid_values[0],
                ),
            ),
        ):
            inner_programs = select_programs(
                graph=core_program_graph(kernel=kernel), retain_replay=True
            )
            for name, program in inner_programs.items():
                programs[f"{role}:{name}"] = replace(
                    program,
                    name=f"{role}:{name}",
                    argument_builder=_NestedArgumentBuilder(
                        inner=program.argument_builder, outer_node=outer_node
                    ),
                    scope=ProgramScope.ANY,
                )
        object.__setattr__(self, "_core_programs", MappingProxyType(programs))

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Return the native graph used by eager, JIT, AOT, and replay paths."""
        return self._core_programs

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> _NNBEGMPeriodKernel:
        """Bind the regime's fixed params into both inner kernels."""
        return replace(
            self,
            keeper_kernel=self.keeper_kernel.with_fixed_params(
                fixed_flat_params=fixed_flat_params
            ),
            adjuster_kernel=self.adjuster_kernel.with_fixed_params(
                fixed_flat_params=fixed_flat_params
            ),
        )

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
    ) -> KernelResult:
        """Solve the keeper, settle the replay capability, run the outer search.

        The finite search folds completed chunks immediately, so
        `outer_batch_size` bounds retained candidate data while publishing the
        complete finite candidate identities for exact replay. The adaptive
        search keeps its exact-node bank because interpolation and policy
        publication consume every refined node.
        """
        keeper_result = self._solve_keeper(
            compiled_cores=compiled_cores,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
            logger=logger,
        )
        # One certificate per period, resolved before either outer search runs.
        # It reads the declared map's structure, which is what both searches
        # depend on and neither owns; certifying inside one of them leaves the
        # other free to publish a replay policy for a map nothing can invert.
        replay_capability = derive_nnbegm_replay_capability(
            period_kernel=self,
            state_action_space=state_action_space,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        return self._solve_outer(
            replay_capability=replay_capability,
            keeper_result=keeper_result,
            compiled_cores=compiled_cores,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
            logger=logger,
        )

    def _solve_outer(
        self,
        *,
        replay_capability: OuterReplayCapability,
        keeper_result: KernelResult,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
        logger: logging.Logger,
    ) -> KernelResult:
        """Run the outer search against the solved keeper; concrete per subclass."""
        raise NotImplementedError

    def _solve_keeper(
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
    ) -> KernelResult:
        """Run the keeper inner solve — the state-dependent no-adjustment branch."""
        return self.keeper_kernel(
            compiled_cores=_subcores(compiled_cores=compiled_cores, role="keeper"),
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
            logger=logger,
        )

    def _solve_adjuster_node(
        self,
        *,
        node: FloatND,
        adjuster_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
        logger: logging.Logger,
    ) -> OuterCandidateResult:
        """Run one adjuster node's exact conditional inner solve."""
        result = self.adjuster_kernel(
            compiled_cores=adjuster_cores,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=_with_outer_post_decision(
                flat_params=flat_params,
                regime_name=self.regime_name,
                outer_post_decision=self.outer_post_decision,
                value=node,
            ),
            period=period,
            ages=ages,
            logger=logger,
        )
        return OuterCandidateResult(
            outer_node=node,
            V_arr=result.V_arr,
            carry=cast("EGMCarry", result.continuation),
            # An inner 1-D kernel publishes a flat policy or nothing; the
            # isinstance narrows the widened payload union for the bank.
            sim_policy=(
                result.simulation_policy
                if isinstance(result.simulation_policy, EGMSimPolicy)
                else None
            ),
        )

    def _certify_outer_inverse(
        self,
        *,
        state_action_space: StateActionSpace,
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> DeclaredOuterInverse:
        """Return the declared outer map's certified inverse for this period.

        The certificate is structural: it reads argument names and shapes and
        never values, so scalar stand-ins for the action arguments certify the
        same map the search later binds to full candidate arrays.

        Raises:
            RegimeInitializationError: If the outer action does not enter the
                declared map affinely with an exactly invertible coefficient.
        """
        params = dict(flat_params[self.regime_name])
        accepted = inspect.signature(self.outer_target_function).parameters
        scalar = jnp.zeros(())
        pool: dict[str, object] = {
            **params,
            **{
                name: jnp.asarray(values)
                for name, values in state_action_space.states.items()
            },
            **{name: jnp.int32(0) for name in self.inner_discrete_action_names},
            self.inner_action: scalar,
            self.outer_action: scalar,
            "period": jnp.int32(period),
            "age": ages.values[period],
        }
        bound = {name: value for name, value in pool.items() if name in accepted}
        return certify_declared_outer_inverse(
            func=self.outer_target_function,
            arg_names=tuple(bound),
            abstract_args=abstract_like(tuple(bound.values())),
            outer_action_name=self.outer_action,
            outer_post_decision_name=self.outer_post_decision,
            outer_state_domain=self.outer_state_domain,
            regime_name=self.regime_name,
        )

    def _resolve_replay_capability(
        self, *, inverse: DeclaredOuterInverse
    ) -> OuterReplayCapability:
        """Settle what a replay of this period may assume, before either search runs.

        Reads the declared structure only — names, signatures, row axes — so the
        answer is the same for both outer searches and for every state a
        simulation later arrives at. NNBEGM refuses phase variation before the
        kernels are built, so the declarations read here are the ones simulation
        reads; and the bindable names come from the params template rather than
        a call's params, so a later `simulate` call cannot widen or narrow them.

        Args:
            inverse: The period's certified inverse of the declared outer map.

        Returns:
            The `OuterReplayCapability` the published policy carries.
        """
        return resolve_outer_replay_capability(
            inverse=inverse,
            functions=self.replay_functions,
            bindable_names=self.replay_bindable_names,
            outer_post_decision_name=self.outer_post_decision,
            outer_action_name=self.outer_action,
            outer_no_adjustment_name=self.outer_no_adjustment_name,
            outer_state_name=self.outer_state_name,
            state_names=self.replay_state_names,
            row_passive_state_names=self.row_passive_state_names,
            # The published rows carry a discrete-action axis exactly when the
            # inner solve makes a discrete choice.
            row_discrete_action_names=self.inner_discrete_action_names,
        )

    def _candidate_outer_targets(
        self,
        *,
        outer_inverse: DeclaredOuterInverse,
        candidate_inner_action: FloatND,
        candidate_discrete_actions: IntND | None,
        discrete_action_names: tuple[ActionName, ...],
        n_discrete_branches: int,
        state_action_space: StateActionSpace,
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
        state_names: tuple[StateName, ...],
        logger: logging.Logger,
    ) -> FloatND:
        """Retain the candidate target identities the solve can hand simulation.

        Simulation recovers the outer action from the retained target at each
        realized state, through the same certified inverse used here. A target
        this inversion cannot reach -- because the recovered action lands off
        the outer state's declared domain, or misses a declared endpoint -- is
        dropped by writing `nan`, so simulation never inherits a candidate the
        solve could not reconstruct.
        """
        state_shape = candidate_inner_action.shape[1:]
        n_state_axes = len(state_names)
        state_inputs = {
            name: jnp.asarray(state_action_space.states[name]).reshape(
                (1,) * axis
                + (jnp.asarray(state_action_space.states[name]).shape[0],)
                + (1,) * (n_state_axes - axis - 1)
            )
            for axis, name in enumerate(state_names)
        }
        if discrete_action_names:
            if candidate_discrete_actions is None:
                raise ValueError("NNBEGM discrete replay is missing candidate codes.")
            discrete_inputs = {
                name: candidate_discrete_actions[:, position].reshape(
                    (candidate_inner_action.shape[0],) + (1,) * n_state_axes
                )
                for position, name in enumerate(discrete_action_names)
            }
        else:
            discrete_inputs = {}
        params = dict(flat_params[self.regime_name])
        accepted = inspect.signature(self.outer_target_function).parameters

        def bind(outer_action: FloatND) -> dict[str, EconFunctionArg]:
            pool = {
                **params,
                **state_inputs,
                **discrete_inputs,
                self.inner_action: candidate_inner_action,
                self.outer_action: outer_action,
                "period": jnp.int32(period),
                "age": ages.values[period],
            }
            return {name: value for name, value in pool.items() if name in accepted}

        def evaluate(outer_action: FloatND) -> Mapping[str, FloatND]:
            return self.outer_target_function(**bind(outer_action))

        # The map evaluated at a zero outer action: the offset the certified
        # inverse subtracts. The certificate itself is resolved once per period
        # before either outer search runs, and arrives as `outer_inverse`.
        bound_at_zero = bind(jnp.zeros_like(candidate_inner_action))
        at_zero_results = self.outer_target_function(**bound_at_zero)
        at_zero = jnp.broadcast_to(
            jnp.asarray(at_zero_results[self.outer_post_decision]),
            candidate_inner_action.shape,
        )
        inverse = outer_inverse

        if self.outer_no_adjustment_name is None:
            keeper_base = jnp.broadcast_to(
                state_inputs[self.outer_state_name], state_shape
            )
            keeper_targets = jnp.broadcast_to(
                keeper_base, (n_discrete_branches, *state_shape)
            )
        else:
            keeper_targets = jnp.broadcast_to(
                jnp.asarray(at_zero_results[self.outer_no_adjustment_name]),
                candidate_inner_action.shape,
            )[:n_discrete_branches]
        adjuster_nodes = jnp.repeat(self.outer_grid_values, repeats=n_discrete_branches)
        adjuster_shape = (adjuster_nodes.shape[0],) + (1,) * len(state_shape)
        adjuster_targets = jnp.broadcast_to(
            adjuster_nodes.reshape(adjuster_shape),
            (adjuster_nodes.shape[0], *state_shape),
        )
        candidate_targets = jnp.concatenate((keeper_targets, adjuster_targets), axis=0)
        if candidate_targets.shape != candidate_inner_action.shape:
            raise ValueError(
                "NNBEGM outer/discrete candidate target bank is misaligned."
            )

        def forward(outer_action: FloatND) -> FloatND:
            return jnp.broadcast_to(
                jnp.asarray(evaluate(outer_action)[self.outer_post_decision]),
                candidate_inner_action.shape,
            )

        inversion = invert_declared_outer_target(
            inverse=inverse,
            target=candidate_targets,
            at_zero=at_zero,
            forward=forward,
        )
        live = jnp.isfinite(candidate_inner_action)
        represented = live & inversion.admissible
        # Only a candidate whose target is a DECLARED node can indict the
        # declaration: the adjuster bank always, and the keeper only when
        # keeping holds the outer state at its own grid value. A custom
        # no-adjustment target is an arbitrary DAG value at a solve cell, so it
        # is the same category of thing simulation meets at a realized state --
        # it is dropped, not treated as a contradiction between law and grids.
        keeper_targets_are_nodes = self.outer_no_adjustment_name is None
        target_is_declared_node = jnp.concatenate(
            (
                jnp.full(keeper_targets.shape, keeper_targets_are_nodes, dtype=bool),
                jnp.ones(adjuster_targets.shape, dtype=bool),
            ),
            axis=0,
        )
        _fail_if_the_solve_grid_cannot_reconstruct_a_candidate(
            logger=logger,
            dropped=live & ~inversion.admissible & target_is_declared_node,
            n_live=live & target_is_declared_node,
            regime_name=self.regime_name,
            period=period,
        )
        return jnp.where(represented, candidate_targets, jnp.nan)


@dataclass(frozen=True, kw_only=True)
class _FiniteNNBEGMPeriodKernel(_NNBEGMPeriodKernel):
    """The nested kernel of a finite outer grid: exact candidate collapse."""

    outer_search: FiniteOuterGrid
    """The finite outer-grid search whose candidate set is collapsed exactly."""

    def _solve_outer(
        self,
        *,
        replay_capability: OuterReplayCapability,
        keeper_result: KernelResult,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
        logger: logging.Logger,
    ) -> KernelResult:
        """Fold finite candidates and retain their complete replay identities."""
        V_arr = keeper_result.V_arr
        keeper_carry = cast("EGMCarry", keeper_result.continuation)
        keeper_policy = cast("NBEGMGridPolicy", keeper_result.simulation_policy)
        discrete_action_names = keeper_policy.discrete_action_names
        branch_codes = keeper_policy.branch_discrete_actions
        keeper_inner, keeper_values = _conditional_nnbegm_banks(
            policy=keeper_policy,
            collapsed_value=keeper_result.V_arr,
            state_names=keeper_policy.state_names,
            discrete_action_names=discrete_action_names,
            branch_codes=branch_codes,
        )
        candidate_inner_by_outer = [keeper_inner]
        candidate_value_by_outer = [keeper_values]
        adjuster_carries: list[EGMCarry] = []
        adjuster_cores = _subcores(compiled_cores=compiled_cores, role="adjuster")
        nodes = list(self.outer_grid_values)
        chunk_size = self.outer_batch_size or len(nodes)
        for chunk_start in range(0, len(nodes), chunk_size):
            chunk_results = [
                self.adjuster_kernel(
                    compiled_cores=adjuster_cores,
                    state_action_space=state_action_space,
                    next_regime_to_V_arr=next_regime_to_V_arr,
                    next_regime_to_continuation=next_regime_to_continuation,
                    flat_params=_with_outer_post_decision(
                        flat_params=flat_params,
                        regime_name=self.regime_name,
                        outer_post_decision=self.outer_post_decision,
                        value=node,
                    ),
                    period=period,
                    ages=ages,
                    logger=logger,
                )
                for node in nodes[chunk_start : chunk_start + chunk_size]
            ]
            for adjuster_result in chunk_results:
                V_arr = jnp.fmax(V_arr, adjuster_result.V_arr)
                adjuster_carries.append(cast("EGMCarry", adjuster_result.continuation))
                adjuster_policy = cast(
                    "NBEGMGridPolicy", adjuster_result.simulation_policy
                )
                adjuster_inner, adjuster_values = _conditional_nnbegm_banks(
                    policy=adjuster_policy,
                    collapsed_value=adjuster_result.V_arr,
                    state_names=keeper_policy.state_names,
                    discrete_action_names=discrete_action_names,
                    branch_codes=branch_codes,
                )
                candidate_inner_by_outer.append(adjuster_inner)
                candidate_value_by_outer.append(adjuster_values)
            V_arr, _ = jax.block_until_ready((V_arr, adjuster_carries[chunk_start:]))

        from _lcm.egm.outer_envelope import stack_candidate_carries  # noqa: PLC0415

        carry = stack_candidate_carries(
            candidates=(keeper_carry, *adjuster_carries),
            nan_is_infeasible=True,
        )
        n_outer_candidates = len(candidate_inner_by_outer)
        n_discrete_branches = int(candidate_inner_by_outer[0].shape[0])
        state_shape = candidate_inner_by_outer[0].shape[1:]
        candidate_inner_action = jnp.stack(candidate_inner_by_outer).reshape(
            n_outer_candidates * n_discrete_branches, *state_shape
        )
        candidate_value = jnp.stack(candidate_value_by_outer).reshape(
            n_outer_candidates * n_discrete_branches, *state_shape
        )
        if discrete_action_names:
            candidate_discrete_actions = jnp.tile(
                cast("IntND", branch_codes), (n_outer_candidates, 1)
            )
        else:
            candidate_discrete_actions = None
        candidate_outer_target = self._candidate_outer_targets(
            outer_inverse=replay_capability.inverse,
            candidate_inner_action=candidate_inner_action,
            candidate_discrete_actions=candidate_discrete_actions,
            discrete_action_names=discrete_action_names,
            n_discrete_branches=n_discrete_branches,
            state_action_space=state_action_space,
            flat_params=flat_params,
            period=period,
            ages=ages,
            state_names=keeper_policy.state_names,
            logger=logger,
        )
        return KernelResult(
            V_arr=V_arr,
            continuation=carry,
            simulation_policy=NNBEGMSimPolicy(
                candidate_inner_action=candidate_inner_action,
                candidate_outer_target=candidate_outer_target,
                candidate_value=candidate_value,
                outer_grid_values=self.outer_grid_values,
                candidate_discrete_actions=candidate_discrete_actions,
                discrete_action_names=discrete_action_names,
                state_names=keeper_policy.state_names,
                inner_action_name=self.inner_action,
                outer_action_name=self.outer_action,
                n_keeper_candidates=n_discrete_branches,
                replay_capability=replay_capability,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class _AdaptiveNNBEGMPeriodKernel(_NNBEGMPeriodKernel):
    """The nested kernel of an adaptive outer mesh: continuous collapse."""

    outer_search: AdaptiveOuterMesh
    """The adaptive mesh search whose refinement and collapse settings apply."""

    def _solve_outer(
        self,
        *,
        replay_capability: OuterReplayCapability,
        keeper_result: KernelResult,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
        logger: logging.Logger,
    ) -> KernelResult:
        """Adaptively refine the shared outer mesh, then collapse continuously.

        The mesh driver's exact-solve callback runs the adjuster's inner
        solve per requested node (chunked by the strategy's `batch_size`)
        and caches every `OuterCandidateResult` by node value, so the final
        bank reuses the refinement solves instead of re-solving. The keeper
        stays a separate exact branch throughout; its `sim_policy` rides
        through unchanged until the continuous simulation reader lands.

        The search itself never inverts the declared outer map; the payload it
        publishes is replayed by a reader that does, so the settled replay
        capability gates publication rather than the search.
        """
        config = self.outer_search
        adjuster_cores = _subcores(compiled_cores=compiled_cores, role="adjuster")
        cache: dict[float, OuterCandidateResult] = {}

        def solve_nodes(nodes_arr: Float1D) -> FloatND:
            requested = [float(node) for node in np.asarray(nodes_arr)]
            pending = [node for node in requested if node not in cache]
            chunk_size = config.batch_size or max(len(pending), 1)
            for chunk_start in range(0, len(pending), chunk_size):
                chunk = pending[chunk_start : chunk_start + chunk_size]
                chunk_results = [
                    self._solve_adjuster_node(
                        node=jnp.asarray(node),
                        adjuster_cores=adjuster_cores,
                        state_action_space=state_action_space,
                        next_regime_to_V_arr=next_regime_to_V_arr,
                        next_regime_to_continuation=next_regime_to_continuation,
                        flat_params=flat_params,
                        period=period,
                        ages=ages,
                        logger=logger,
                    )
                    for node in chunk
                ]
                jax.block_until_ready(
                    [(result.V_arr, result.carry) for result in chunk_results]
                )
                cache.update(zip(chunk, chunk_results, strict=True))
            return jnp.stack([cache[node].V_arr for node in requested])

        mesh = refine_outer_mesh(
            initial_nodes=self.outer_grid_values,
            solve_at=solve_nodes,
            config=config,
            fail_closed=config.fail_closed,
        )
        mesh_nodes_host = np.asarray(mesh.nodes)
        bank = build_outer_candidate_bank(
            outer_nodes=mesh.nodes,
            results=[cache[float(node)] for node in mesh_nodes_host],
        )
        if self.branch_fixed_cost is None:
            fixed_cost_scale = None
            fixed_cost_support = None
        else:
            fixed_cost_scale = _resolve_branch_scale(
                scale_function=self.branch_scale_function,
                regime_params=flat_params[self.regime_name],
                period=period,
                ages=ages,
            )
            fixed_cost_support = (
                self.branch_fixed_cost.lower,
                self.branch_fixed_cost.upper,
            )
        collapse = collapse_continuous_candidate_bank(
            keeper_v_arr=keeper_result.V_arr,
            keeper_carry=cast("EGMCarry", keeper_result.continuation),
            bank=bank,
            config=config,
            fixed_cost_scale=fixed_cost_scale,
            fixed_cost_support=fixed_cost_support,
        )
        # Derive both branches' inner simulation policies. An NB-EGM inner
        # publishes no `EGMSimPolicy` of its own; on the smooth v1 scope its
        # unrefined carry rows determine the policy exactly (`consumption =
        # resources - savings` node by node), so derive both sides from the
        # carries and fail closed (no nested payload, grid simulation
        # unchanged) whenever the rows are not derivation-safe.
        keeper_policy = (
            keeper_result.simulation_policy
            if isinstance(keeper_result.simulation_policy, EGMSimPolicy)
            else derive_inner_sim_policy(
                carry=cast("EGMCarry", keeper_result.continuation),
                state_grid_values=self.liquid_grid_values,
                row_discrete_state_names=self.row_discrete_state_names,
                row_passive_state_names=self.row_passive_state_names,
            )
        )
        adjuster_policies = (
            bank.sim_policy
            if bank.sim_policy is not None
            else derive_inner_sim_policy(
                carry=bank.carry,
                state_grid_values=self.liquid_grid_values,
                row_discrete_state_names=self.row_discrete_state_names,
                row_passive_state_names=self.row_passive_state_names,
                extra_leading_axes=1,
            )
        )
        # Publish the nested payload only when both inner policies are
        # derivation-safe AND the branch is a deterministic hard maximum AND the
        # inner solve makes no discrete choice: the continuous reader replays
        # keeper vs adjuster off-grid from exactly these conditional ingredients.
        # Under a fixed-cost aggregation the realized branch depends on the drawn
        # cost, and an inner DISCRETE action's winning branch is collapsed out of
        # the published carry rows — the reader cannot replay
        # either, so simulation falls back to the grid argmax, which is precisely
        # what `policy_fallback_mask` reports (so the mask is set from this same
        # condition rather than hard-coded).
        nested_published = (
            keeper_policy is not None
            and adjuster_policies is not None
            and self.branch_fixed_cost is None
            and not self.inner_discrete_action_names
        )
        diagnostics = SolverDiagnostics(
            max_outer_interpolation_error=jnp.asarray(mesh.max_validation_error),
            max_outer_bracket_width=jnp.max(collapse.value_search.bracket_width),
            outer_nodes_used=jnp.asarray(bank.n_candidates, dtype=jnp.int32),
            outer_at_lower_bound=collapse.value_search.at_lower_bound,
            outer_at_upper_bound=collapse.value_search.at_upper_bound,
            keeper_adjuster_margin=collapse.keeper_adjuster_margin,
            best_second_best_margin=collapse.best_second_best_margin,
            policy_fallback_mask=jnp.asarray(not nested_published),
            unresolved_mask=jnp.asarray(mesh.unresolved),
            n_outer_all_invalid_cells=jnp.asarray(
                mesh.n_cells_all_invalid, dtype=jnp.int32
            ),
            adjustment_probability=collapse.adjustment_probability,
        )
        sim_policy: SimulationPolicy | None = None
        if nested_published:
            # Only a regime that actually publishes reaches the refusal. A
            # regime whose inner solve makes a discrete choice never gets here:
            # `nested_published` is already false for it, so the capability's
            # discrete-action verdict has nothing to stop.
            fail_if_continuous_outer_replay_is_unsupported(
                capability=replay_capability,
                regime_name=self.regime_name,
                outer_action_name=self.outer_action,
            )
            sim_policy = NestedEGMSimPolicy(
                keeper=keeper_policy,
                adjuster=OuterPolicyBank(
                    outer_nodes=mesh.nodes,
                    policies=adjuster_policies,
                ),
                outer_action_name=self.outer_action,
                outer_state_name=self.outer_state_name,
                outer_post_decision_name=self.outer_post_decision,
                inner_action_name=self.inner_action,
                liquid_state_name=self.liquid_state_name,
                outer_no_adjustment_name=self.outer_no_adjustment_name,
                resources_target_name=self.resources_target,
                savings_lower_bound=self.savings_lower_bound,
                golden_iterations=config.golden_iterations,
                replay_capability=replay_capability,
                value_atol=config.value_atol,
                value_rtol=config.value_rtol,
            )
        return KernelResult(
            V_arr=collapse.V_arr,
            continuation=collapse.carry,
            simulation_policy=sim_policy,
            generated_replay_authority=(
                GeneratedReplayAuthority(
                    adaptive_outer_nodes=tuple(float(node) for node in mesh_nodes_host)
                )
                if nested_published
                else None
            ),
            diagnostics=diagnostics,
        )


def derive_nnbegm_replay_capability(
    *,
    period_kernel: PeriodKernel,
    state_action_space: StateActionSpace,
    flat_params: FlatParams,
    period: int,
    ages: AgeGrid,
) -> OuterReplayCapability:
    """Derive one period's replay capability from canonical model inputs.

    The solve calls this before publishing a policy, and labelled-result preflight
    calls it again without consulting that policy. Keeping the structural
    certificate behind one model-owned seam prevents a returned capability from
    authenticating its own inverse coefficient or route diagnostics.
    """
    if not isinstance(period_kernel, _NNBEGMPeriodKernel):
        msg = (
            "NNBEGM replay authority requires the canonical NNBEGM period kernel, "
            f"got {type(period_kernel).__name__}."
        )
        raise TypeError(msg)
    inverse = period_kernel._certify_outer_inverse(  # noqa: SLF001
        state_action_space=state_action_space,
        flat_params=flat_params,
        period=period,
        ages=ages,
    )
    return period_kernel._resolve_replay_capability(  # noqa: SLF001
        inverse=inverse
    )


def _fail_if_the_solve_grid_cannot_reconstruct_a_candidate(
    *,
    logger: logging.Logger,
    dropped: BoolND,
    n_live: BoolND,
    regime_name: RegimeName,
    period: int,
) -> None:
    """Stop any solve that cannot reconstruct a candidate it retained.

    Applies only to candidates whose target is a declared node -- the adjuster
    search grid, and the keeper when keeping holds the outer state at its own
    grid value. For those, an inverse reaching outside the outer state's domain
    means the declaration and the grids disagree, which is a defect in the model
    rather than one realized state landing awkwardly, so the solve is the loud
    phase. A custom no-adjustment target is a computed DAG value and is excluded
    by the caller; it drops like a realized state instead.

    This refuses at **every** log level, `"off"` included. The log level governs
    diagnostics, and this is not one: the failure is known before anything is
    published, and dropping it quietly leaves a policy bank whose contents depend
    on the diagnostic setting -- the same model would publish different policies
    at `"off"` and at `"debug"`. Reading the count back costs a host transfer per
    period, which is the price of not letting a published policy depend on how
    loudly the run was asked to talk.

    A failure first met at a realized off-grid subject is a different case and
    keeps its drop-and-announce behaviour: there the state landed awkwardly,
    which the model author cannot be expected to have precluded.
    """
    n_dropped = int(jnp.sum(dropped))
    if n_dropped == 0:
        return
    total = int(jnp.sum(n_live))
    msg = (
        f"Regime {regime_name!r} at period {period}: the solve retained "
        f"{total} outer candidates at declared nodes but could not reconstruct "
        f"{n_dropped} of them. The outer action recovered from those targets "
        "reaches a stock outside the outer state's declared domain, where "
        "there is no value function. The solve stops rather than publish a "
        "candidate bank missing them. Widen the outer state's grid so every "
        "declared node is reachable, or narrow the outer search to nodes it "
        "can reach."
    )
    # Logged as well as raised: the exception may be caught by a caller running
    # a sweep, and the log is then the only surviving record of which regime and
    # period failed.
    logger.error(msg)
    raise UnrepresentableOuterCandidateError(msg)


def _fail_if_the_outer_search_leaves_the_outer_state_domain(
    *,
    regime_name: RegimeName,
    outer_state: StateName,
    outer_state_grid: Grid,
    outer_search: OuterSearch,
) -> None:
    """Refuse an outer search that names a stock the outer state cannot hold.

    The outer search's nodes are post-decision targets for the outer state, so
    a node outside that state's own grid asks the solve to retain a value the
    state does not represent. Nothing downstream rejects it: the value function
    read extrapolates linearly past the edge, so the excursion surfaces a period
    later as an out-of-support state rather than at the declaration that caused
    it.

    Both grids are declared, so this compares them directly and probes no
    floating-point value.
    """
    match outer_search:
        case FiniteOuterGrid():
            nodes = outer_search.grid.to_jax()
            label = "outer grid"
        case AdaptiveOuterMesh():
            nodes = outer_search.initial_grid.to_jax()
            label = "initial outer mesh"
        case _:
            return

    domain = outer_state_grid.to_jax()
    low, high = domain[0], domain[-1]
    below = jnp.min(nodes) < low
    above = jnp.max(nodes) > high
    if not bool(below | above):
        return

    side = "below" if bool(below) else "above"
    offending = float(jnp.min(nodes)) if bool(below) else float(jnp.max(nodes))
    msg = (
        f"Regime {regime_name!r}: the NNBEGM {label} reaches outside the "
        f"declared domain of the outer state {outer_state!r}. Its node "
        f"{offending} lies {side} that state's grid, which spans "
        f"[{float(low)}, {float(high)}]. Every outer node is a post-decision "
        f"target the outer state must be able to hold, so narrow the outer "
        f"search to that domain or widen the grid of {outer_state!r}."
    )
    raise ModelInitializationError(msg)


def _subcores(
    *, compiled_cores: Mapping[str, Callable], role: str
) -> Mapping[str, Callable]:
    """Select one role's inner cores, stripping the `role:` prefix."""
    token = f"{role}:"
    return MappingProxyType(
        {
            key.removeprefix(token): core
            for key, core in compiled_cores.items()
            if key.startswith(token)
        }
    )


def _nnbegm_inner_action(
    *, context: SolverBuildContext, outer_action: ActionName
) -> ActionName:
    """The regime's single inner continuous action (not the outer one).

    The v1 nested scope carries exactly one inner continuous action; its
    name identifies which recorded action the published inner policy
    replaces in simulation.
    """
    names = [
        name
        for name in context.state_action_space.continuous_actions
        if name != outer_action
    ]
    if len(names) != 1:
        msg = (
            "NNBEGM supports exactly one inner continuous action besides "
            f"the outer action '{outer_action}', found {sorted(names)}."
        )
        raise RegimeInitializationError(msg)
    return names[0]


def _nested_inverse_marginal(
    *,
    context: SolverBuildContext,
    rows_on_state_grid: bool,
    inner_action: ActionName,
    savings_top: float,
) -> Callable[..., FloatND] | None:
    """The regime's inverse marginal utility, if payload-derivation-safe.

    The nested simulation payload derives the inner consumption rows from the
    carry's marginal via the envelope theorem, which requires (a) the inner
    carry rows to live on the shared liquid state grid and (b) an inverse of
    `u'` free of state/param bindings (a state-dependent utility would need
    per-row bindings the kernel-level derivation does not perform). Mirrors
    the inner solve's own choice: the model's closed-form
    `inverse_marginal_utility` when its only parameter is
    `marginal_continuation`, else the iEGM numeric inversion of the utility's
    action-derivative under the same bracket convention as the solve
    (`step_core`), provided utility is a function of the inner action alone.
    Anything else returns `None`: the solve is unaffected and simulation
    keeps the grid-argmax path.
    """
    import inspect  # noqa: PLC0415

    if not rows_on_state_grid:
        return None
    closed_form = context.functions.get("inverse_marginal_utility")
    if closed_form is not None and tuple(inspect.signature(closed_form).parameters) == (
        "marginal_continuation",
    ):
        return closed_form
    utility = context.functions.get("utility")
    if utility is None or tuple(inspect.signature(utility).parameters) != (
        inner_action,
    ):
        return None
    marginal_utility = jax.grad(lambda c: utility(**{inner_action: c}))
    action_upper = jnp.asarray(savings_top * 1000.0 + 1000.0)
    action_lower = jnp.asarray(1e-8, dtype=action_upper.dtype)

    def inverse_marginal(marginal_continuation: FloatND) -> FloatND:
        flat = jnp.ravel(jnp.asarray(marginal_continuation))
        roots = jax.vmap(
            lambda m: numeric_inverse_marginal_utility(
                marginal_continuation=m,
                marginal_utility=marginal_utility,
                c_lower=action_lower,
                c_upper=action_upper,
            )
        )(flat)
        return roots.reshape(jnp.shape(marginal_continuation))

    return inverse_marginal


def _resolve_branch_fixed_cost(
    *,
    aggregator: OuterBranchAggregator,
    context: SolverBuildContext,
) -> tuple[UniformObservedFixedCost | None, Callable[..., FloatND] | None]:
    """Validate and resolve a fixed-cost branch aggregator at build time.

    Returns `(None, None)` for the deterministic maximum. For
    `UniformObservedFixedCost`, checks the analytic-integration contract:

    - the shock must *not* be a solve state (the closed form replaces its
      grid; a leftover state would integrate the cost twice);
    - the scale function must exist and read only `period`, `age`, and flat
      params — the collapse applies one scalar scale per period, so a state-
      dependent scale is out of the supported scope.

    An aggregator outside the supported set is rejected rather than run as
    the deterministic maximum, which would publish a value function for an
    aggregation the caller did not ask for.
    """
    import inspect  # noqa: PLC0415

    _fail_if_aggregator_unsupported(aggregator)
    if not isinstance(aggregator, UniformObservedFixedCost):
        return None, None
    if aggregator.shock_name in context.state_action_space.states:
        msg = (
            f"UniformObservedFixedCost integrates the shock "
            f"'{aggregator.shock_name}' analytically; remove its solve-state "
            f"grid from regime '{context.regime_name}' (keeping it would "
            "integrate the cost twice)."
        )
        raise RegimeInitializationError(msg)
    scale_function = context.functions.get(aggregator.scale_function)
    if scale_function is None:
        msg = (
            f"UniformObservedFixedCost.scale_function "
            f"'{aggregator.scale_function}' is not a function of regime "
            f"'{context.regime_name}'."
        )
        raise RegimeInitializationError(msg)
    unresolvable = [
        name
        for name in inspect.signature(scale_function).parameters
        if name not in ("period", "age") and name not in context.flat_param_names
    ]
    if unresolvable:
        msg = (
            f"UniformObservedFixedCost.scale_function "
            f"'{aggregator.scale_function}' reads {sorted(unresolvable)}; the "
            "per-period scalar scale may only read `period`, `age`, and flat "
            "params (a state-dependent scale is outside the supported scope)."
        )
        raise RegimeInitializationError(msg)
    return aggregator, scale_function


def _fail_if_aggregator_unsupported(aggregator: OuterBranchAggregator) -> None:
    """Reject a branch aggregator whose fold the kernels do not implement.

    `NNBEGM` executes exactly two folds — the deterministic hard maximum and
    the analytically integrated uniform observed fixed cost. Any other
    concrete `OuterBranchAggregator` names an aggregation with no kernel
    behind it, so it is refused here instead of silently taking the
    deterministic branch.
    """
    if isinstance(aggregator, DeterministicOuterMaximum | UniformObservedFixedCost):
        return
    msg = (
        f"NNBEGM does not implement the branch aggregation "
        f"{type(aggregator).__name__}; use DeterministicOuterMaximum() or "
        "UniformObservedFixedCost(...)."
    )
    raise RegimeInitializationError(msg)


def _branch_scale_check(
    *,
    regime_name: RegimeName,
    ages: AgeGrid,
    branch_aggregation_by_period: Mapping[
        int, tuple[UniformObservedFixedCost | None, Callable[..., FloatND] | None]
    ],
) -> ParamCheck | None:
    """Build the preflight over the fixed cost's per-period scale, if any.

    The ages are closed over here rather than taken through the check's own
    call, which stays `(*, flat_params)` — the signature every solver author
    writes a `ParamCheck` against.

    Returns `None` when no period aggregates a fixed cost, so a deterministic
    regime carries no check at all.
    """
    periods = tuple(
        period
        for period, (fixed_cost, _) in branch_aggregation_by_period.items()
        if fixed_cost is not None
    )
    if not periods:
        return None

    def _check(*, flat_params: FlatParams) -> None:
        _fail_if_branch_scale_outside_support(
            regime_name=regime_name,
            periods=periods,
            branch_aggregation_by_period=branch_aggregation_by_period,
            regime_params=flat_params[regime_name],
            ages=ages,
        )

    return _check


def _fail_if_branch_scale_outside_support(
    *,
    regime_name: RegimeName,
    periods: tuple[int, ...],
    branch_aggregation_by_period: Mapping[
        int, tuple[UniformObservedFixedCost | None, Callable[..., FloatND] | None]
    ],
    regime_params: Mapping[str, object],
    ages: AgeGrid,
) -> None:
    """Reject a fixed-cost scale outside the closed form's support.

    The analytic fold is defined for finite `B >= 0`. Negative values are
    adjustment subsidies rather than costs; NaN and infinity poison or
    degenerate the cutoff calculation. Every period carrying a fixed cost is
    checked because an age-varying schedule can leave the range only once.
    """
    for period in periods:
        _, scale_function = branch_aggregation_by_period[period]
        scale = _resolve_branch_scale(
            scale_function=scale_function,
            regime_params=regime_params,
            period=period,
            ages=ages,
        )
        values = np.asarray(scale, dtype=float).reshape(-1)
        supported = np.isfinite(values) & (values >= 0.0)
        if np.all(supported):
            continue
        msg = (
            f"UniformObservedFixedCost in regime '{regime_name}' needs a "
            f"finite scale `B >= 0`; the scale function evaluates to "
            f"{values[~supported][0]} at period {period}. The closed form "
            "reads every nonpositive scale as `B = 0` (the deterministic "
            "maximum), so a negative or nonfinite draw would publish an "
            "aggregation that was never requested."
        )
        raise RegimeInitializationError(msg)


def _resolve_branch_scale(
    *,
    scale_function: Callable[..., FloatND] | None,
    regime_params: Mapping[str, object],
    period: int,
    ages: AgeGrid,
) -> FloatND:
    """Evaluate the fixed cost's per-period scalar scale at kernel-call time."""
    import inspect  # noqa: PLC0415

    if scale_function is None:  # pragma: no cover - guarded at build time
        msg = "branch_fixed_cost set without a resolved scale function"
        raise RegimeInitializationError(msg)
    kwargs: dict[str, object] = {}
    for name in inspect.signature(scale_function).parameters:
        if name == "period":
            kwargs[name] = jnp.asarray(period)
        elif name == "age":
            kwargs[name] = jnp.asarray(ages.values[period])
        else:
            kwargs[name] = regime_params[name]
    return jnp.asarray(scale_function(**kwargs))


def _ride_along_inner_kernel(
    *, kernel: PeriodKernel, role: str
) -> _RideAlongNBEGMPeriodKernel:
    """Return the inner kernel as the ride-along NB-EGM kernel it must be.

    The nested composite calls its inner kernels directly and republishes their
    program graphs, so it requires the concrete ride-along kernel rather than
    the abstract period-kernel protocol. The outer state is a ride-along state
    of every inner solve, so any other kernel is a wiring defect.
    """
    if not isinstance(kernel, _RideAlongNBEGMPeriodKernel):
        msg = (
            f"NNBEGM's {role} inner kernel must be a ride-along NB-EGM period "
            f"kernel; got {type(kernel).__name__}."
        )
        raise TypeError(msg)
    return kernel


def _fail_if_inner_is_not_nbegm(inner: object) -> None:
    """Enforce the public NNBEGM composition despite inert type stubs.

    The planned DCEGM-or-NBEGM inner unification belongs to the follow-on NEGM
    fold.  The current public NNBEGM solver is the NBEGM-specific composition,
    so accepting another object here would defer a structural error until
    private margin binding or kernel construction.
    """
    if not isinstance(inner, NBEGM):
        cls = type(inner)
        raise RegimeInitializationError(
            "NNBEGM.inner must be an NBEGM numerical configuration, got "
            f"{cls.__module__}.{cls.__qualname__}."
        )


def _fail_if_inner_carry_rows_not_grid_aligned(*, inner: Solver) -> None:
    """Refuse an inner solver whose carry rows do not sit on the state grid.

    The bridged outer envelope replaces `value` and `marginal_utility` per
    candidate and rides the keeper's `endog_grid` through unchanged, which is
    only correct when every candidate publishes rows at the same abscissae.
    """
    if not inner.egm_continuation_layout.rows_share_state_grid:
        msg = (
            f"NNBEGM's inner solver {type(inner).__name__} publishes carry rows "
            "off the shared state grid, but the bridged outer envelope folds "
            "candidates pointwise and reuses the keeper's `endog_grid`, so the "
            "folded rows would pair one candidate's values with another's "
            "abscissae. Use an inner solver whose "
            "`egm_continuation_layout.rows_share_state_grid` is True."
        )
        raise RegimeInitializationError(msg)


def _fail_if_nnbegm_carry_publishes_topology_rows(
    *, template: ContinuationPayload | None
) -> None:
    if isinstance(template, EGMCarry) and template.breakpoints is not None:
        msg = (
            "NNBEGM publishes a bridged (pointwise, finite-grid) outer "
            "envelope, which cannot represent the inner config's jump-topology "
            "rows. Use `jump_read='bridged'` on the inner NBEGM or remove the "
            "declared jump breakpoints."
        )
        raise RegimeInitializationError(msg)
