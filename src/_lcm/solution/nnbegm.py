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

import inspect
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
from beartype import beartype
from dags import concatenate_functions

from _lcm.beartype_conf import REGIME_CONF
from _lcm.constraints.routes import ConstraintRoute
from _lcm.continuation import EGMContinuationLayout, EGMContinuationSpec
from _lcm.egm.carry import EGMCarry
from _lcm.egm.published_policy import NBEGMGridPolicy, NNBEGMSimPolicy
from _lcm.engine import StateActionSpace
from _lcm.grids import ContinuousGrid, Grid
from _lcm.regime_building.phases import phase_variation_paths
from _lcm.solution.contract import (
    ConstraintRouteContext,
    ContinuationPayload,
    KernelResult,
    PeriodKernel,
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
    _validate_nbegm_case_piece_declarations,
    proved_post_decision_of,
)
from _lcm.solution.negm import (
    _fail_if_outer_batch_size_negative,
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
from _lcm.typing import FlatParams, RegimeName
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.typing import ActionName, FloatND, FunctionName, IntND, StateName

if TYPE_CHECKING:
    from lcm.regime import Regime as UserRegime
else:
    # Importing the public regime class here closes a cycle through the
    # `lcm.solvers` facade, which re-exports `NNBEGM` from this module. ty sees
    # the precise type above; the runtime annotation stays deliberately broad.
    UserRegime = object


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

    outer_grid: ContinuousGrid
    """Exogenous candidate grid for the outer post-decision margin."""

    outer_batch_size: int = 0
    """Outer-grid nodes solved per chunk before folding into the running
    maximum; `0` solves every node at once. A memory knob only —
    value-invariant."""

    def __post_init__(self) -> None:
        _fail_if_inner_is_not_nbegm(self.inner)
        _fail_if_outer_grid_is_stochastic(self.outer_grid)
        _fail_if_outer_batch_size_negative(self.outer_batch_size, solver_name="NNBEGM")

    def _with_margins(
        self,
        *,
        liquid: _BoundLiquidMargin,
        outer: _BoundOuterContinuousMargin,
    ) -> _BoundNNBEGM:
        """Bind both regime-owned margins into a private runtime config."""
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
        """The carry keeps the keeper plus every finite outer-grid candidate."""
        return replace(
            self.inner.egm_continuation_layout,
            n_stacked_candidates=int(self.outer_grid.to_jax().shape[0]) + 1,
        )

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
            user_regime=user_regime,
        )
        validate_nnbegm_regime(
            regime_name=context.regime_name,
            user_regime=user_regime,
        )
        bound = cast("_BoundNNBEGM", self)
        outer_state = bound.outer_state
        liquid = bound.inner.continuous_state
        fail_if_kernel_grids_withhold_their_points(
            grids={
                "outer grid": bound.outer_grid,
                "inner savings grid": bound.inner.savings_grid,
                f"grid of the outer state '{outer_state}'": cast(
                    "Grid", user_regime.states[outer_state]
                ),
                f"grid of the liquid state '{liquid}'": cast(
                    "Grid", user_regime.states[liquid]
                ),
            },
            regime_name=context.regime_name,
            solver_name="NNBEGM",
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

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
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

        adjuster_by_period: dict[int, PeriodKernel] = {}
        keeper_by_period: dict[int, PeriodKernel] = {}
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
                adjuster_by_period[period] = adjuster_group.period_kernels[period]
                keeper_by_period[period] = keeper_group.period_kernels[period]
                outer_target_function_by_period[period] = outer_target_function
            grouped_param_checks.extend(adjuster_group.param_checks)
            grouped_param_checks.extend(keeper_group.param_checks)
            if keeper_continuation_spec is None:
                keeper_continuation_spec = keeper_group.continuation_spec
        template = (
            None
            if keeper_continuation_spec is None
            else keeper_continuation_spec.template
        )
        if not (
            context.constraint_plan and context.constraint_plan.compiled_boundaries
        ):
            _fail_if_nnbegm_carry_publishes_topology_rows(template=template)
        outer_grid_values = self.outer_grid.to_jax()
        period_kernels = MappingProxyType(
            {
                period: _NNBEGMPeriodKernel(
                    keeper_kernel=keeper_by_period[period],
                    adjuster_kernel=adjuster_kernel,
                    regime_name=context.regime_name,
                    outer_grid_values=outer_grid_values,
                    inner_action_name=bound.inner.continuous_action,
                    outer_action_name=bound.outer_action,
                    outer_state_name=bound.outer_state,
                    outer_post_decision=bound.outer_post_decision,
                    outer_no_adjustment_target=bound.outer_no_adjustment_candidate,
                    outer_target_function=outer_target_function_by_period[period],
                    outer_batch_size=self.outer_batch_size,
                )
                for period, adjuster_kernel in adjuster_by_period.items()
            }
        )
        n_candidates = int(outer_grid_values.shape[0]) + 1
        stacked_template = _stack_carry_template(
            template=template, n_candidates=n_candidates
        )
        return SolutionKernels(
            period_kernels=period_kernels,
            continuation_spec=(
                None
                if stacked_template is None
                else EGMContinuationSpec(
                    template=stacked_template, layout=self.egm_continuation_layout
                )
            ),
            # Both inner margins are solved by the inner solver, so both sets of
            # parameter-dependent preconditions still apply to this regime.
            param_checks=tuple(grouped_param_checks),
        )


@dataclass(frozen=True, kw_only=True)
class _BoundNNBEGM(NNBEGM):
    """Internal N-NB-EGM config with both regime margins resolved."""

    inner: _BoundNBEGM
    outer_action: ActionName
    outer_state: StateName
    outer_post_decision: FunctionName
    outer_no_adjustment_candidate: FunctionName | None


def _fail_if_nnbegm_phase_variation(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
) -> None:
    """Reject phase variation before NNBEGM period kernels are constructed.

    NNBEGM publishes only the keeper-plus-outer-grid candidates solved during
    backward induction. A genuinely phase-varying declaration would require a
    separate simulate-phase policy over that same candidate set; generic
    action-grid maximization is not an equivalent fallback.
    """
    variations = phase_variation_paths(user_regime=user_regime)
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
class _NNBEGMPeriodKernel:
    """The NNBEGM period adapter — a keeper plus an adjuster outer sweep.

    Holds two inner NB-EGM period adapters and the exogenous outer grid. Each
    inner adapter can expose several independently-traced cores (the ride-along
    NB-EGM kernel splits into a continuation and an envelope core), so the
    nested adapter republishes every inner core under a `keeper:`/`adjuster:`
    prefix and strips the prefix when delegating.

    Calling it runs the keeper once and the adjuster once per outer-grid node,
    collapses the outer axis by `V = max(V_keeper, max_j W_j)`, and publishes
    the pointwise (bridged) upper envelope of the candidates' carry rows on the
    shared liquid state grid. The adapter is non-jitted: it dispatches the
    shared jitted inner cores, matching `_NEGMPeriodKernel`.
    """

    keeper_kernel: PeriodKernel
    """The keeper inner adapter — a passive per-durable-state NB-EGM."""

    adjuster_kernel: PeriodKernel
    """The adjuster inner adapter whose shared jitted cores are swept."""

    regime_name: RegimeName
    """Name of the regime whose flat params the outer node binds into."""

    outer_grid_values: FloatND
    r"""Exogenous grid over the outer post-decision margin $s_t^\textit{post-dec}$."""

    inner_action_name: ActionName
    """Name of the conditional inner action retained per candidate."""

    outer_action_name: ActionName
    """Name of the outer action reconstructed from the candidate target."""

    outer_state_name: StateName
    """Name of the state-specific keeper source."""

    outer_post_decision: FunctionName
    """Name of the outer post-decision function bound per outer-grid node."""

    outer_no_adjustment_target: FunctionName | None
    """Custom keeper target, or ``None`` for identity in the outer state."""

    outer_target_function: Callable
    """Resolved solve-phase DAG used to recover the outer action bank."""

    outer_batch_size: int
    """Outer-grid nodes solved per chunk before folding into the running
    maximum; `0` solves every node at once."""

    @property
    def core(self) -> Callable:
        """The adjuster's primary core, exposed for any single-core reader."""
        return self.adjuster_kernel.core

    def cores(self) -> Mapping[str, Callable]:
        """Return every inner core under a `keeper:`/`adjuster:` prefix.

        The keeper and adjuster are distinct traced programs built from
        different contexts, and each inner adapter may expose several cores of
        its own; prefixing keeps every (role, inner-core) pair under its own
        AOT compilation key.
        """
        return MappingProxyType(
            {
                **{
                    f"keeper:{name}": core
                    for name, core in self.keeper_kernel.cores().items()
                },
                **{
                    f"adjuster:{name}": core
                    for name, core in self.adjuster_kernel.cores().items()
                },
            }
        )

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

    def build_lower_args(
        self,
        *,
        core_key: str,
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> Mapping[str, object]:
        """Delegate the named inner core's lowering arguments.

        The prefix selects the role; the remainder is the inner adapter's own
        core key. The adjuster binds `outer_post_decision` at the first
        outer-grid node so its lowered program matches the shape every per-node
        call traces; the keeper lowers with no outer binding.
        """
        role, inner_key = core_key.split(":", maxsplit=1)
        if role == "keeper":
            return self.keeper_kernel.build_lower_args(
                core_key=inner_key,
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                next_regime_to_continuation=next_regime_to_continuation,
                flat_params=flat_params,
                period=period,
                ages=ages,
            )
        return self.adjuster_kernel.build_lower_args(
            core_key=inner_key,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=_with_outer_post_decision(
                flat_params=flat_params,
                regime_name=self.regime_name,
                outer_post_decision=self.outer_post_decision,
                value=self.outer_grid_values[0],
            ),
            period=period,
            ages=ages,
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
    ) -> KernelResult:
        """Run keeper and adjuster sweep, collapse by `max`, and retain identities.

        The value/carry fold remains the ordinary outer hard maximum. The replay
        payload separately keeps the complete outer-times-discrete candidate
        product in solve order: keeper before declared outer nodes, and each
        outer candidate crossed with the inner envelope's branch-product order.
        """
        keeper_result = self.keeper_kernel(
            compiled_cores=_subcores(compiled_cores=compiled_cores, role="keeper"),
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
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
                )
                for node in nodes[chunk_start : chunk_start + chunk_size]
            ]
            for adjuster_result in chunk_results:
                # `fmax`, not `maximum`: the inner NB-EGM NaN-dead masks cells
                # an outer node makes infeasible, and one infeasible candidate
                # must not poison a cell another candidate solves. A cell stays
                # NaN only when every candidate is infeasible there.
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
        candidate_outer_action = self._candidate_outer_actions(
            candidate_inner_action=candidate_inner_action,
            candidate_discrete_actions=candidate_discrete_actions,
            discrete_action_names=discrete_action_names,
            n_discrete_branches=n_discrete_branches,
            state_action_space=state_action_space,
            flat_params=flat_params,
            period=period,
            ages=ages,
            state_names=keeper_policy.state_names,
        )
        return KernelResult(
            V_arr=V_arr,
            continuation=carry,
            simulation_policy=NNBEGMSimPolicy(
                candidate_inner_action=candidate_inner_action,
                candidate_outer_action=candidate_outer_action,
                candidate_value=candidate_value,
                candidate_discrete_actions=candidate_discrete_actions,
                discrete_action_names=discrete_action_names,
                state_names=keeper_policy.state_names,
                inner_action_name=self.inner_action_name,
                outer_action_name=self.outer_action_name,
            ),
        )

    def _candidate_outer_actions(
        self,
        *,
        candidate_inner_action: FloatND,
        candidate_discrete_actions: IntND | None,
        discrete_action_names: tuple[ActionName, ...],
        n_discrete_branches: int,
        state_action_space: StateActionSpace,
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
        state_names: tuple[StateName, ...],
    ) -> FloatND:
        """Recover each complete candidate's outer action on the solve state grid."""
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

        def evaluate(outer_action: FloatND) -> Mapping[str, FloatND]:
            pool = {
                **params,
                **state_inputs,
                **discrete_inputs,
                self.inner_action_name: candidate_inner_action,
                self.outer_action_name: outer_action,
                "period": jnp.int32(period),
                "age": ages.values[period],
            }
            return self.outer_target_function(
                **{name: value for name, value in pool.items() if name in accepted}
            )

        zeros = jnp.zeros_like(candidate_inner_action)
        at_zero_results = evaluate(zeros)
        at_zero = jnp.broadcast_to(
            jnp.asarray(at_zero_results[self.outer_post_decision]),
            candidate_inner_action.shape,
        )
        at_one = jnp.broadcast_to(
            jnp.asarray(
                evaluate(jnp.ones_like(candidate_inner_action))[
                    self.outer_post_decision
                ]
            ),
            candidate_inner_action.shape,
        )
        slope = at_one - at_zero

        if self.outer_no_adjustment_target is None:
            keeper_base = jnp.broadcast_to(
                state_inputs[self.outer_state_name], state_shape
            )
            keeper_targets = jnp.broadcast_to(
                keeper_base, (n_discrete_branches, *state_shape)
            )
        else:
            keeper_targets = jnp.broadcast_to(
                jnp.asarray(at_zero_results[self.outer_no_adjustment_target]),
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

        candidate_outer_action = (candidate_targets - at_zero) / slope
        reconstructed = jnp.broadcast_to(
            jnp.asarray(evaluate(candidate_outer_action)[self.outer_post_decision]),
            candidate_inner_action.shape,
        )
        eps = jnp.finfo(candidate_inner_action.dtype).eps
        tolerance = 128 * eps * jnp.maximum(1.0, jnp.abs(candidate_targets))
        represented = (
            jnp.isfinite(candidate_inner_action)
            & jnp.isfinite(candidate_outer_action)
            & jnp.isfinite(slope)
            & (slope != 0)
            & (jnp.abs(reconstructed - candidate_targets) <= tolerance)
        )
        inversion_failed = jnp.isfinite(candidate_inner_action) & ~represented
        if bool(jax.device_get(jnp.any(inversion_failed))):
            raise RegimeInitializationError(
                "NNBEGM requires the outer post-decision target to depend "
                "affinely on the outer action with a finite, nonzero slope "
                "conditional on its other inputs. The declared target could "
                "not be inverted and reconstructed for every represented "
                "candidate. Use an affine mapping such as "
                "`new = old + 2 * action`, or select a solver that supports "
                "an explicit inverse for the declared mapping."
            )
        return jnp.where(represented, candidate_outer_action, jnp.nan)


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
