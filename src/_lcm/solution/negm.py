"""The NEGM solver (nested endogenous grid method, Druedahl 2021).

`NEGM` nests a `DCEGM` inner solve: a per-durable-node passive keeper alongside
one adjuster solve per exogenous outer post-decision node, whose conditional
carries are lifted into common cash-on-hand (via the declared outer cost) and
stacked; the parent read collapses the candidate axis by the exact query-side
maximum. `_NEGMPeriodKernel` publishes a native two-program graph: `keeper` is
the inner keeper's own program under a new name, and `outer_sweep` is one
deliberately dense program that maps the inner adjuster over the outer grid,
takes the exact maximum with the keeper value, and stacks every candidate carry.
The graph declares the dependency between them: `keeper` publishes its value and
carry as typed internal outputs, `outer_sweep` names them as internal inputs, so
the engine lowers the sweep against the keeper's own abstract output and the
kernel hands the keeper's outputs over under those names.
"""

import functools
import logging
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from beartype import beartype
from dags import concatenate_functions, get_annotations, with_signature
from dags.annotations import ensure_annotations_are_strings

from _lcm.beartype_conf import REGIME_CONF
from _lcm.constraints.bounds import proves_the_savings_grids_lower_bound
from _lcm.constraints.routes import (
    ConstraintRoute,
    ConstraintRouteKey,
    ConstraintSite,
)
from _lcm.continuation import EGMContinuationLayout, EGMContinuationSpec
from _lcm.egm.carry import EGMCarry, egm_carry_role_tree
from _lcm.egm.outer_envelope import build_stacked_outer_carry
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import (
    CoreArgumentBuilder,
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    InternalInputRef,
    InternalOutputSpec,
    core_program_graph,
)
from _lcm.execution.output_layout import VALUE, StateAxesLeading
from _lcm.grids import ContinuousGrid
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.solution.continuation_target import union_fixed_params
from _lcm.solution.contract import (
    ConstraintRouteContext,
    ContinuationPayload,
    ParamCheck,
    PeriodKernel,
    SolutionKernels,
    SolverBuildContext,
    SolverModelContext,
    TwoMarginSolver,
    _BoundLiquidMargin,
    _BoundOuterContinuousMargin,
    bind_roles,
    simulation_route,
)
from _lcm.solution.dcegm import DCEGM, _BoundDCEGM, _combination_inputs
from _lcm.typing import (
    EconFunction,
    EconFunctionsMapping,
    FlatParams,
    RegimeName,
)
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidParamsError, RegimeInitializationError
from lcm.solver_api import EGM_CONTINUATION, ArtifactKey, KernelOutput
from lcm.typing import (
    ActionName,
    Float1D,
    FloatND,
    FunctionName,
    ScalarFloat,
    StateName,
    StateOrActionName,
)

_NEGM_SWEEP_DENSE_REASON = (
    "deliberately_dense:negm_outer_candidates_retained_not_reduced"
)
# The keeper's outputs and the sweep's own inputs enter the compiled sweep next
# to the inner adjuster's arguments, under engine-only names. A public regime,
# function, state, or action name cannot contain the ``__`` qualified-name
# separator, and a generated qualified parameter name always has a non-empty
# component before its separator, so a key that starts with the separator can
# never be produced by a supported model.
_TRANSPORT_PREFIX = "__lcm_negm_"
_KEEPER_VALUE = f"{_TRANSPORT_PREFIX}keeper_value"
_KEEPER_CARRY = f"{_TRANSPORT_PREFIX}keeper_carry"
_OUTER_NODES = f"{_TRANSPORT_PREFIX}outer_nodes"
_COH_SHIFTS = f"{_TRANSPORT_PREFIX}coh_shifts"


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class NEGM(TwoMarginSolver):
    r"""Nested-EGM solver: an outer grid search over a durable/illiquid margin.

    NEGM solves a model with one continuous margin the Euler equation cleanly
    inverts on (liquid consumption-savings) plus a second continuous margin
    that does not admit a clean inverse-Euler (a durable/illiquid stock with
    adjustment frictions) by nesting:

    - an *inner* standard 1-D DC-EGM solve of the consumption-savings problem,
      conditional on the outer margin being fixed (this is exactly the existing
      `DCEGM` kernel, with the outer margin entering inner resources and utility
      as a constant and indexing the child durable state);
    - an *outer* deterministic `max` over a grid of the outer post-decision
      margin plus mandatory kink candidates (the no-adjustment point
      $s_t^\textit{post-dec} = s_t$,
      the floor corner).

    The outer step is a search, not a second inverse-Euler: the outer value is
    generically non-concave (adjustment-cost kink, floor corners), so a second
    EGM inversion would be invalid there.

    The `NEGM(inner=DCEGM(...), …)` composition makes "NEGM nests the 1-D
    DC-EGM" literal: it reuses every inner field and its upper-envelope backend,
    reuses `DCEGM.__post_init__` validation wholesale, and keeps the
    outer-margin contract in one place. The model-contract check
    The model-stage validation hook rejects, at `Model` construction, any model NEGM
    does not fit (no outer margin, a coupled-2-Euler pension shape, a
    taste-shock-ordering violation), naming the offending feature and the
    correct alternative solver.
    """

    inner: DCEGM
    """The inner 1-D DC-EGM config.

    Carries the liquid Euler state, the consumption action, the resources and
    post-decision functions, the savings grid, and the upper-envelope backend.
    Its `__post_init__` guards run on construction, so an invalid inner config
    is rejected before NEGM's own guards.
    """

    outer_grid: ContinuousGrid
    r"""Exogenous grid over the outer post-decision margin $s_t^\textit{post-dec}$."""

    outer_batch_size: int = 0
    """Number of outer-grid nodes solved per block of the compiled outer sweep.

    The sweep maps the inner adjuster over the exogenous outer grid in blocks
    of this many nodes; a block is the vector width the adjuster is compiled
    for, so the knob bounds the *solve-side* block transients only. It does not
    bound the period's peak, whose remaining candidate-scaled contributions
    blocking cannot remove:

    - the candidate *carries* are all retained — the published stacked
      continuation holds every outer candidate (`(A+1) * n_pad` grid slots per
      leading cell), inherent to the exact query-side outer maximum,
    - while the stack is built, the unstacked candidate carries and the
      stacked output coexist transiently,
    - the parent's continuation read prepares a search key of the full stacked
      shape and evaluates every candidate per query.

    A positive value solves that many nodes per block; `0` (the default)
    solves every node in one block — fastest, but its solve-side peak grows
    with the outer-grid size. It is a memory-vs-parallelism knob only: the
    solved value function and every carry leaf agree across batch sizes to
    the working format's spacing.
    """

    def __post_init__(self) -> None:
        _fail_if_outer_grid_is_stochastic(self.outer_grid)
        _fail_if_outer_batch_size_negative(outer_batch_size=self.outer_batch_size)

    def _with_margins(
        self,
        *,
        liquid: _BoundLiquidMargin,
        outer: _BoundOuterContinuousMargin,
    ) -> _BoundNEGM:
        """Bind both regime-owned margins into a private runtime config."""
        # The inner solver is bound first and then overrides the copy taken from
        # this solver, so the nest carries the *bound* inner rather than the
        # public one it was declared with.
        return cast(
            "_BoundNEGM",
            bind_roles(
                solver=self,
                role_type=_BoundNEGM,
                inner=self.inner._with_liquid_margin(liquid),  # noqa: SLF001
                outer_action=outer.action,
                outer_state=outer.state,
                outer_post_decision=outer.post_decision_state,
                outer_no_adjustment_candidate=outer.no_adjustment,
                outer_cost=liquid.cost,
                outer_cost_base=liquid.before_cost,
            ),
        )

    @property
    def required_continuation_keys(self) -> frozenset[ArtifactKey]:
        """NEGM nests a DC-EGM solve that inverts the Euler equation."""
        return frozenset({EGM_CONTINUATION})

    @property
    def egm_continuation_layout(self) -> EGMContinuationLayout:
        """The carry stacks the keeper plus one row per outer-grid node."""
        return EGMContinuationLayout(
            n_stacked_candidates=int(self.outer_grid.to_jax().shape[0]) + 1
        )

    def validate_model(self, *, context: SolverModelContext) -> None:
        """Validate the user-level nested-EGM contract for this regime."""
        from _lcm.egm.negm_validation import validate_negm_regime  # noqa: PLC0415

        validate_negm_regime(
            regime_name=context.regime_name,
            user_regime=context.user_regimes[context.regime_name],
        )

    def validate_build(self, *, context: SolverBuildContext) -> None:
        """Apply the inner solver's build-time gates to this regime.

        The inner kernels run unchanged inside every outer candidate, so a
        build-time capability the inner solver requires is required here for
        exactly the same reason. The inner gates read the inner solver's own
        configuration rather than any state name, so they carry over without
        being re-pointed at the nest's axes.
        """
        self.inner.validate_build(context=context)

    def build_constraint_routes(
        self, *, context: ConstraintRouteContext
    ) -> tuple[ConstraintRoute, ...]:
        """Declare the adjuster, keeper, and simulation routes NEGM walks.

        The solve has two branches whose inner DC-EGM kernels receive different
        function pools:

        - the adjuster removes the outer post-decision function and binds its
          value to one outer-grid node;
        - the keeper replaces that function by the no-adjustment map of the
          durable state.

        A constraint is therefore rebound against each branch's own pool. The
        routes carry the same period groups the keeper build uses, because an
        age-specialized no-adjustment map changes which concrete function is in
        scope. Simulation sees a complete subject-level candidate and uses the
        shared unrestricted route.
        """
        bound = cast("_BoundNEGM", self)
        proofs = (
            proves_the_savings_grids_lower_bound(
                post_decision=bound.inner.post_decision_function
            ),
        )
        if context.phase == "simulate":
            return (simulation_route(context=context, solver_path=("negm",)),)

        from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
            resolve_periodized_nodes,
        )

        routes: list[ConstraintRoute] = []
        for period_group in _periodized_function_groups(
            periods=context.active_periods, functions=context.functions
        ):
            group_functions = cast(
                "EconFunctionsMapping",
                resolve_periodized_nodes(
                    mapping=context.functions, period=period_group[0]
                ),
            )
            adjuster_context = replace(
                context,
                functions=_without_outer_post_decision(
                    functions=group_functions,
                    outer_post_decision=bound.outer_post_decision,
                ),
                flat_param_names=context.flat_param_names | {bound.outer_post_decision},
            )
            no_adjustment_func = (
                group_functions[bound.outer_no_adjustment_candidate]
                if bound.outer_no_adjustment_candidate is not None
                else None
            )
            keeper_context = replace(
                context,
                functions=_with_no_adjustment_outer_function(
                    functions=group_functions,
                    durable_state=bound.outer_state,
                    outer_post_decision=bound.outer_post_decision,
                    no_adjustment_func=no_adjustment_func,
                ),
            )
            routes.extend(
                (
                    ConstraintRoute(
                        key=ConstraintRouteKey(
                            phase="solve",
                            period_group=period_group,
                            solver_path=("negm", "adjuster"),
                        ),
                        sites=(
                            ConstraintSite(
                                stage="outer_candidate",
                                function_pool=adjuster_context.functions,
                                available_names=_combination_inputs(
                                    context=adjuster_context,
                                    euler_state=bound.inner.continuous_state,
                                ),
                                structural_proofs=proofs,
                            ),
                        ),
                    ),
                    ConstraintRoute(
                        key=ConstraintRouteKey(
                            phase="solve",
                            period_group=period_group,
                            solver_path=("negm", "keeper"),
                        ),
                        sites=(
                            ConstraintSite(
                                stage="keeper_candidate",
                                function_pool=keeper_context.functions,
                                available_names=_combination_inputs(
                                    context=keeper_context,
                                    euler_state=bound.inner.continuous_state,
                                ),
                                structural_proofs=proofs,
                            ),
                        ),
                    ),
                )
            )
        return tuple(routes)

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one NEGM period adapter per period, wrapping the inner kernels.

        The model-stage and build-stage validation hooks guarantee the outer
        margin is present and distinct from the inner margin, that it is not
        Euler-coupled to the inner state, and that any taste-shocked discrete
        choice is the outermost aggregation. The inner DC-EGM period kernels are
        built once (with the outer margin bound so it enters the inner resources
        and utility as a constant and indexes the child durable state); each is
        wrapped in an outer adapter that sweeps the outer grid plus the mandatory
        per-node candidates and collapses the outer axis by `max`.
        """
        bound_self = cast("_BoundNEGM", self)
        # The adjuster is the inner DC-EGM with the outer post-decision supplied
        # per outer-grid node rather than recomputed from the outer action:
        # `_with_outer_post_decision` binds it into the regime's flat params at
        # runtime, so the inner kernel reads it as a bound param — admit it as a
        # flat param at build time too, so the inner scope check accepts the
        # inner resources / utility reading it (the service-flow
        # `utility(serviced(<outer post-decision>))` pattern) and dropping it
        # from the econ-function DAG leaves it a leaf rather than an expression
        # in the outer action.
        #
        # The durable's own law of motion stays exactly as the regime declares
        # it. It reads the post-decision, which is that bound leaf here, so it
        # is decision-independent without being replaced — and a declared
        # `next_<durable> = (1 - delta) * s_t^post-dec` is therefore the stock the
        # continuation is read at, not the raw node the outer search picked.
        adjuster_context = replace(
            context,
            functions=_without_outer_post_decision(
                functions=context.functions,
                outer_post_decision=bound_self.outer_post_decision,
            ),
            flat_param_names=context.flat_param_names
            | {bound_self.outer_post_decision},
        )
        adjuster_kernels = bound_self.inner.build_period_kernels(
            context=adjuster_context
        )
        # The keeper is a normal passive DC-EGM: the outer post-decision is held
        # at its no-adjustment level (`s_t^post-dec = keep(<durable>_t)`), so the
        # durable becomes a genuine decision-independent passive state and
        # `credited(<durable>, keep(<durable>)) = 0` makes keeping free. The
        # keeper map is injected into the econ functions, so the inner resources
        # DAG computes the post-decision from the durable leaf rather than
        # demanding it as a bound param, as the adjuster does. The declared law
        # of motion again stays as written and reads that value, so what the
        # keeper carries is `next_<durable>(keep(<durable>))`. With no
        # `outer_no_adjustment_candidate`, `keep` is the identity (hold the
        # stock); a stock that lands off the durable grid — because `keep` or
        # the law shrinks it — is blended over the grid by the inner passive
        # read. Function-local so the public `lcm.solvers` façade stays a thin
        # re-export that pulls in no engine modules.
        from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
            resolve_periodized_nodes,
        )

        outer_grid_values = bound_self.outer_grid.to_jax()
        durable_state = bound_self.outer_state
        own_v_info = context.regime_to_v_interpolation_info[context.regime_name]
        discrete_state_names = tuple(
            name
            for name in own_v_info.state_names
            if name in own_v_info.discrete_states
        )
        passive_state_names = tuple(
            name
            for name in own_v_info.continuous_states
            if name != bound_self.inner.continuous_state
        )
        durable_axis_in_carry = len(discrete_state_names) + passive_state_names.index(
            durable_state
        )
        # Both outer helpers are read here, one layer above the inner builder that
        # resolves periodized nodes — so they must be resolved first. An
        # `AgeSpecializedFunction` is a different function at each age, and the
        # no-adjustment candidate is *closed over* by the injected keeper function
        # rather than left as a pool node, so nothing downstream can resolve it.
        # Periods group by the user's declared signature over the whole function
        # pool: an age-invariant regime yields exactly one group and one keeper
        # build, which is today's behaviour.
        all_periods = tuple(sorted(adjuster_kernels.period_kernels))
        keeper_kernels_by_period: dict[int, PeriodKernel] = {}
        coh_shift_by_period: dict[int, Callable[..., FloatND]] = {}
        keeper_continuation_template: EGMCarry | None = None
        # A regime whose inner builder produced no period kernels still owes the
        # caller a continuation template, so fall back to one group holding no
        # periods: it builds the template and contributes no per-period entries.
        groups = _periodized_function_groups(
            periods=all_periods, functions=context.functions
        ) or ((),)
        active = context.regimes_to_active_periods.get(context.regime_name, ())
        template_period = (
            all_periods[0] if all_periods else (active[0] if active else 0)
        )
        keeper_param_checks: tuple[ParamCheck, ...] = ()
        for group_periods in groups:
            representative_period = (
                group_periods[0] if group_periods else (template_period)
            )
            group_functions = cast(
                "EconFunctionsMapping",
                resolve_periodized_nodes(
                    mapping=context.functions, period=representative_period
                ),
            )
            no_adjustment_func = (
                group_functions[bound_self.outer_no_adjustment_candidate]
                if bound_self.outer_no_adjustment_candidate is not None
                else None
            )
            keeper_context = replace(
                context,
                functions=_with_no_adjustment_outer_function(
                    functions=group_functions,
                    durable_state=durable_state,
                    outer_post_decision=bound_self.outer_post_decision,
                    no_adjustment_func=no_adjustment_func,
                ),
            )
            group_keeper_kernels = bound_self.inner.build_period_kernels(
                context=keeper_context
            )
            keeper_param_checks += group_keeper_kernels.param_checks
            keeper_continuation_template = (
                None
                if group_keeper_kernels.continuation_spec is None
                else cast("EGMCarry", group_keeper_kernels.continuation_spec.template)
            )
            group_coh_shift_func = _build_coh_shift_function(
                functions=group_functions,
                durable_state_name=durable_state,
                outer_post_decision=bound_self.outer_post_decision,
                no_adjustment_func=no_adjustment_func,
                outer_cost_name=bound_self.outer_cost,
            )
            for period in group_periods:
                keeper_kernels_by_period[period] = group_keeper_kernels.period_kernels[
                    period
                ]
                coh_shift_by_period[period] = group_coh_shift_func
        assert keeper_continuation_template is not None  # noqa: S101
        # The durable nodes enter a numerical lift (the credited-cost shift of cash
        # on hand), so they must be the solved period's own. With an age-specialized
        # durable grid the representative age's nodes are the wrong ones everywhere
        # else, so read the period's own when the schedule offers them.
        representative_durable_values = context.grids[durable_state].to_jax()

        def _durable_values_at(period: int) -> Float1D:
            per_period = context.period_to_state_nodes
            if per_period is None:
                return representative_durable_values
            return per_period.get(period, {}).get(
                durable_state, representative_durable_values
            )

        carry_row_state_names = discrete_state_names + passive_state_names
        period_kernels = MappingProxyType(
            {
                period: _NEGMPeriodKernel(
                    keeper_kernel=keeper_kernels_by_period[period],
                    adjuster_kernel=adjuster_kernel,
                    regime_name=context.regime_name,
                    transition_target_names=tuple(context.transitions),
                    outer_grid_values=outer_grid_values,
                    outer_post_decision=bound_self.outer_post_decision,
                    coh_shift_func=coh_shift_by_period[period],
                    durable_grid_values=_durable_values_at(period),
                    durable_axis_in_carry=durable_axis_in_carry,
                    carry_row_state_names=carry_row_state_names,
                    outer_batch_size=bound_self.outer_batch_size,
                )
                for period, adjuster_kernel in adjuster_kernels.period_kernels.items()
            }
        )
        stacked_template = _stack_carry_template(
            template=keeper_continuation_template,
            n_candidates=outer_grid_values.shape[0] + 1,
        )
        # `_stack_carry_template` passes `None` through, and the keeper template
        # was established as present above, so the stacked template is too.
        assert stacked_template is not None  # noqa: S101
        return SolutionKernels(
            period_kernels=period_kernels,
            continuation_spec=EGMContinuationSpec(
                template=stacked_template,
                layout=self.egm_continuation_layout,
            ),
            # Both inner margins are solved by the inner solver, so both sets of
            # parameter-dependent preconditions still apply to this regime.
            param_checks=(*adjuster_kernels.param_checks, *keeper_param_checks),
        )


def _periodized_function_groups(
    *, periods: tuple[int, ...], functions: EconFunctionsMapping
) -> tuple[tuple[int, ...], ...]:
    """Partition periods by the declared signature of the function pool.

    Periods land in the same group exactly when the user's `signature(age)` agrees
    for every `AgeSpecializedFunction` in the pool, so they resolve to the same
    concrete functions and may share one keeper build. A pool with no age
    specialization yields a single group holding every period.

    Args:
        periods: The periods to partition, in ascending order.
        functions: The regime's processed functions, possibly holding periodized
            nodes.

    Returns:
        Tuple of period groups, each a tuple of periods sharing one signature.

    """
    from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
        periodized_tree_signature,
    )

    groups: dict[Hashable, list[int]] = {}
    for period in periods:
        groups.setdefault(
            periodized_tree_signature(tree=functions, period=period), []
        ).append(period)
    return tuple(tuple(group) for group in groups.values())


@dataclass(frozen=True, kw_only=True)
class _BoundNEGM(NEGM):
    """Internal NEGM configuration with both regime margins resolved."""

    inner: _BoundDCEGM
    """The liquid-margin solver run inside every outer candidate."""

    outer_action: ActionName
    """Name of the continuous action setting the outer margin."""

    outer_state: StateName
    """Name of the continuous state the outer margin searches over."""

    outer_post_decision: FunctionName
    """Name of the function giving the outer post-decision state."""

    outer_no_adjustment_candidate: FunctionName | None
    """Name of the no-adjustment map, or `None` for the identity map."""

    outer_cost: FunctionName | None
    """Cost function of a composed `NetOfAdjustmentCost`, else `None`."""

    outer_cost_base: FunctionName | None
    """Gross-resources function of that composition, else `None`."""


@dataclass(frozen=True, kw_only=True)
class _NEGMPeriodKernel:
    """The NEGM period kernel — a keeper program and a compiled outer sweep.

    Holds two inner DC-EGM period kernels and the exogenous outer grid. The
    outer durable choice splits into two programs of the kernel's native graph:

    - `keeper` — the inner keeper's own program, a per-durable-state passive
      DC-EGM (`next_illiquid = illiquid`, identity) that keeps the durable
      stock unchanged for free (`credited(s, s) = 0`), run once over the full
      durable grid; and
    - `outer_sweep` — one deliberately dense program that maps the inner
      adjuster (the DC-EGM with the outer transition stripped) over the
      exogenous outer grid with `outer_post_decision` bound to each node,
      collapses the value by `V = max(V_keeper, max_j W_j)` and stacks the
      keeper carry with every node carry, lifted into common cash on hand, on
      the candidate axis.

    `keeper` publishes its value and carry as typed internal outputs and
    `outer_sweep` names them as internal inputs, so the engine lowers the sweep
    against the keeper's own abstract output and calling the kernel runs
    `keeper`, then dispatches `outer_sweep` with the keeper's outputs under
    those argument names. The published simulation policy is absent: a
    keeper-only proposal cannot represent the joint durable/liquid decision, so
    simulation retains its canonical full-grid action pair.
    """

    keeper_kernel: PeriodKernel
    """The keeper inner kernel — a passive per-durable-state DC-EGM."""

    adjuster_kernel: PeriodKernel
    """The adjuster inner kernel whose program the sweep maps over the nodes.

    Held with no fixed params bound: the sweep binds them itself, so periods
    sharing one adjuster program keep sharing one compiled sweep.
    """

    regime_name: RegimeName
    """Name of the regime whose flat params the outer node binds into."""

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose fixed params the inner
    adjuster reads under their namespace."""

    outer_grid_values: Float1D
    r"""Exogenous grid over the outer post-decision margin $s_t^\textit{post-dec}$."""

    outer_post_decision: FunctionName
    """Name of the outer post-decision function bound per outer-grid node."""

    coh_shift_func: Callable[..., FloatND]
    """Per-(durable, outer-node) cash-on-hand shift of each adjuster candidate.

    Maps the durable grid, the outer grid, and the regime's flat params to the
    shift matrix `credited(z, z'_j)` that lifts each adjuster's endogenous grid
    into the keeper's cash-on-hand axis.
    """

    durable_grid_values: Float1D
    """The durable state's grid used for the credited-cost lift."""

    durable_axis_in_carry: int
    """Position of the durable state among the carry's leading state axes."""

    carry_row_state_names: tuple[StateName, ...]
    """The discrete then passive state names leading every carry row."""

    outer_batch_size: int
    """Outer-grid nodes solved per block of the sweep; `0` is one block."""

    fixed_sweep_kwargs: Mapping[str, object] = MappingProxyType({})
    """The regime's and its targets' fixed params, bound into the sweep."""

    _core_programs: Mapping[str, CoreProgram] = field(
        init=False, repr=False, compare=False
    )
    """The native graph, `keeper` then `outer_sweep`; derived at construction."""

    def __post_init__(self) -> None:
        """Derive the graph from the inner kernels and the sweep's bindings."""
        keeper = core_program_graph(kernel=self.keeper_kernel)["main"]
        adjuster = core_program_graph(kernel=self.adjuster_kernel)["main"]
        sweep_function = functools.partial(
            _outer_sweep_program,
            inner_core=adjuster.function,
            outer_post_decision=self.outer_post_decision,
            durable_axis=self.durable_axis_in_carry,
            outer_batch_size=self.outer_batch_size,
            **self.fixed_sweep_kwargs,
        )
        row = StateAxesLeading(state_names=self.carry_row_state_names)
        programs = {
            "keeper": replace(
                keeper,
                name="keeper",
                internal_outputs=(
                    InternalOutputSpec(label="value", path=(0,)),
                    InternalOutputSpec(label="carry", path=(1,)),
                ),
            ),
            "outer_sweep": CoreProgram(
                name="outer_sweep",
                function=sweep_function,
                argument_builder=_NEGMSweepArgumentBuilder(
                    adjuster_builder=adjuster.argument_builder,
                    regime_name=self.regime_name,
                    outer_post_decision=self.outer_post_decision,
                    outer_grid_values=self.outer_grid_values,
                    durable_grid_values=self.durable_grid_values,
                    coh_shift_func=self.coh_shift_func,
                ),
                requirements=CoreExecutionRequirements(
                    internal_inputs=MappingProxyType(
                        {
                            _KEEPER_VALUE: InternalInputRef(
                                producer="keeper", label="value"
                            ),
                            _KEEPER_CARRY: InternalInputRef(
                                producer="keeper", label="carry"
                            ),
                        }
                    )
                ),
                output_roles=(
                    VALUE,
                    egm_carry_role_tree(
                        row=row,
                        scalar=StateAxesLeading(state_names=(), shape=()),
                        breakpoints=None,
                        policy=None,
                    ),
                ),
                disposition=CoreExecutionDisposition.DENSE,
                disposition_reason=_NEGM_SWEEP_DENSE_REASON,
                donation_candidates=(),
            ),
        }
        object.__setattr__(self, "_core_programs", MappingProxyType(programs))

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Return the native graph used by eager, JIT, AOT, and replay paths."""
        return self._core_programs

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> _NEGMPeriodKernel:
        """Bind the regime's fixed params into the keeper, the sweep, and the shift.

        The keeper binds them into its own program. The sweep binds the union
        of the regime's and its targets' fixed params as its own keyword
        arguments, which it forwards to the inner adjuster per node; binding
        them on the sweep rather than on the adjuster's program keeps one
        compiled sweep per adjuster program across periods. The cash-on-hand
        shift evaluates the regime's inner resources, which may read a fixed
        param, so the same values are bound into `coh_shift_func` as well.
        """
        regime_fixed = dict(
            fixed_flat_params.get(self.regime_name, MappingProxyType({}))
        )
        coh_shift_func = self.coh_shift_func
        if regime_fixed:
            coh_shift_func = functools.partial(coh_shift_func, **regime_fixed)
        return replace(
            self,
            keeper_kernel=self.keeper_kernel.with_fixed_params(
                fixed_flat_params=fixed_flat_params
            ),
            coh_shift_func=coh_shift_func,
            fixed_sweep_kwargs=MappingProxyType(
                {
                    **self.fixed_sweep_kwargs,
                    **union_fixed_params(
                        fixed_flat_params=fixed_flat_params,
                        regime_name=self.regime_name,
                        transition_target_names=self.transition_target_names,
                    ),
                }
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
    ) -> KernelOutput:
        r"""Run the keeper, then the sweep fed by the keeper's declared outputs.

        The keeper runs the passive DC-EGM once, yielding the value of leaving
        the durable stock unchanged at every durable state; its value and carry
        travel to the sweep under the argument names the sweep declares as its
        internal inputs. The sweep solves the adjuster at every exogenous
        outer-grid node $s_{t,j}^\textit{post-dec}$,
        collapses the outer axis into the value array by
        `V = max(V_keeper, max_j W_j)`, and retains every candidate carry in
        the published continuation so the parent read can take the exact
        `max_j V_j(q)` at its own query.
        """
        keeper_output = self.keeper_kernel(
            compiled_cores={"main": compiled_cores["keeper"]},
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
            logger=logger,
        )
        keeper_value = jnp.asarray(keeper_output.value)
        keeper_carry = cast("EGMCarry", keeper_output.continuations[EGM_CONTINUATION])
        arguments = dict(
            self._core_programs["outer_sweep"].argument_builder(
                CoreBuildContext(
                    state_action_space=state_action_space,
                    next_regime_to_V_arr=next_regime_to_V_arr,
                    next_regime_to_continuation=next_regime_to_continuation,
                    flat_params=flat_params,
                    period=period,
                    ages=ages,
                )
            )
        )
        V_arr, carry = compiled_cores["outer_sweep"](
            **arguments, **{_KEEPER_VALUE: keeper_value, _KEEPER_CARRY: keeper_carry}
        )
        return KernelOutput(value=V_arr, continuations={EGM_CONTINUATION: carry})


@dataclass(frozen=True, kw_only=True)
class _NEGMSweepArgumentBuilder:
    """Build the outer sweep's arguments for lowering and execution.

    Delegates to the inner adjuster's builder with the outer post-decision
    bound at the first outer node, so the per-node arguments have exactly the
    shape the sweep traces, and adds the sweep's own inputs: the outer nodes and
    the credited-cost shifts. The keeper's value and carry are not built here —
    they are internal inputs: the engine lowers the sweep against the keeper's
    abstract output, and the kernel hands the keeper's real outputs over at
    dispatch.
    """

    adjuster_builder: CoreArgumentBuilder
    """The inner adjuster program's own argument builder."""

    regime_name: RegimeName
    """Name of the regime whose flat params receive the bound node."""

    outer_post_decision: FunctionName
    """Name of the outer post-decision function the node is bound as."""

    outer_grid_values: Float1D
    """The exogenous outer grid the sweep maps over."""

    durable_grid_values: Float1D
    """The durable state's grid used for the credited-cost lift."""

    coh_shift_func: Callable[..., FloatND]
    """The per-(durable, outer-node) cash-on-hand shift of each adjuster."""

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
        """Return the exact kwargs shared by lowering and the runtime call."""
        flat_params = cast("FlatParams", context.flat_params)
        arguments = dict(
            self.adjuster_builder(
                replace(
                    context,
                    flat_params=_with_outer_post_decision(
                        flat_params=flat_params,
                        regime_name=self.regime_name,
                        outer_post_decision=self.outer_post_decision,
                        value=self.outer_grid_values[0],
                    ),
                )
            )
        )
        own = {
            _OUTER_NODES: self.outer_grid_values,
            _COH_SHIFTS: self.coh_shift_func(
                durable_values=self.durable_grid_values,
                outer_values=self.outer_grid_values,
                **flat_params[self.regime_name],
            ),
        }
        _fail_if_sweep_inputs_collide_with_the_adjusters(
            arguments=arguments, own=own, regime_name=self.regime_name
        )
        return MappingProxyType({**arguments, **own})


def _outer_sweep_program(
    *,
    inner_core: Callable[..., tuple[FloatND, EGMCarry]],
    outer_post_decision: FunctionName,
    durable_axis: int,
    outer_batch_size: int,
    **arguments: object,
) -> tuple[FloatND, EGMCarry]:
    """Solve the adjuster at every outer node and stack it with the keeper.

    The inner adjuster program runs once per exogenous node with the outer
    post-decision bound into its arguments, in blocks of `outer_batch_size`
    nodes (`0`: one block). The value is the exact maximum of the keeper value
    and every node value; the continuation is the keeper carry followed by
    every node carry lifted into common cash on hand on the candidate axis.

    The keeper's value and carry, the outer nodes, and the cash-on-hand shifts
    arrive among `arguments` under the engine-only transport keys; everything
    else is the inner adjuster's own argument tree.
    """
    adjuster_arguments = dict(arguments)
    keeper_value = cast("FloatND", adjuster_arguments.pop(_KEEPER_VALUE))
    keeper_carry = cast("EGMCarry", adjuster_arguments.pop(_KEEPER_CARRY))
    outer_nodes = cast("Float1D", adjuster_arguments.pop(_OUTER_NODES))
    coh_shifts = cast("FloatND", adjuster_arguments.pop(_COH_SHIFTS))
    n_nodes = outer_nodes.shape[0]

    def solve_node(node: ScalarFloat) -> tuple[FloatND, EGMCarry]:
        value, carry = inner_core(**{**adjuster_arguments, outer_post_decision: node})
        return value, carry

    node_values, node_carries = jax.lax.map(
        solve_node, outer_nodes, batch_size=outer_batch_size or n_nodes
    )
    V_arr = jnp.maximum(keeper_value, jnp.max(node_values, axis=0))
    carry = build_stacked_outer_carry(
        keeper_carry=keeper_carry,
        adjuster_carries=tuple(
            jax.tree.map(lambda leaf, index=index: leaf[index], node_carries)
            for index in range(n_nodes)
        ),
        coh_shifts=coh_shifts,
        durable_axis=durable_axis,
    )
    return V_arr, carry


def _fail_if_sweep_inputs_collide_with_the_adjusters(
    *,
    arguments: Mapping[str, object],
    own: Mapping[str, object],
    regime_name: RegimeName,
) -> None:
    collisions = sorted(set(arguments) & set(own))
    if collisions:
        msg = (
            f"Regime '{regime_name}' produced arguments {collisions} that collide "
            "with the NEGM outer sweep's engine-only transport keys. No supported "
            "model can produce these names; this is an internal namespace error."
        )
        raise RegimeInitializationError(msg)


def _stack_carry_template(
    *, template: EGMCarry | None, n_candidates: int
) -> EGMCarry | None:
    """Stack a keeper carry template into the published candidate-axis shape.

    The NEGM continuation carry retains every outer candidate (the keeper plus
    one per outer-grid node) on a candidate axis inserted just before the grid
    axis. The parent period's kernel is AOT-compiled against this template, so
    the template must carry that axis: the keeper template is broadcast across
    the `n_candidates` slots, keeping every row finite and ascending so a parent
    evaluated against the template stays finite.
    """
    if template is None:
        return None

    def stack(arr: FloatND) -> FloatND:
        return jnp.broadcast_to(
            arr[..., None, :], (*arr.shape[:-1], n_candidates, arr.shape[-1])
        )

    return EGMCarry(
        endog_grid=stack(template.endog_grid),
        value=stack(template.value),
        marginal_utility=stack(template.marginal_utility),
        taste_shock_scale=template.taste_shock_scale,
    )


def _build_coh_shift_function(
    *,
    functions: EconFunctionsMapping,
    durable_state_name: StateName,
    outer_post_decision: FunctionName,
    no_adjustment_func: EconFunction | None,
    outer_cost_name: FunctionName | None,
) -> Callable[..., FloatND]:
    """Build the per-(durable, outer-node) cash-on-hand shift of each adjuster.

    Adjuster `j`'s inner endogenous grid lives in resources space `R_j = coh -
    cost(z, z'_j)`; mapping it into the keeper's cash-on-hand axis adds back
    the credited cost relative to the free keep:

    `shift(z, z'_j) = cost(z, z'_j) - cost(z, keep(z))`,

    evaluated directly on the regime's declared outer-cost DAG
    (`liquid.resources.cost`), whose inputs are only the durable state, the outer
    post-decision, and params. Nothing about the shift is inferred from the
    wider resources function — with a declared cost the resources are composed
    at model build as `<resources>_before_outer_cost - <outer_cost>`, so their
    affine use of the cost (coefficient exactly `-1`) holds by construction
    and the credited difference is exactly the resources translation. `keep`
    is the keeper's no-adjustment map (`no_adjustment_func`; the identity when
    the regime declares none) — the level whose credited cost is zero, e.g.
    the depreciated stock `z (1 - delta)`. The axis change has derivative 1,
    so each candidate's value and resource-marginal transfer into coh space
    unchanged. With `outer_cost_name=None` — validated at model build to mean
    the resources never read the outer post-decision — the shift is
    identically zero.

    The returned callable takes the durable grid (`durable_values`), the outer
    grid (`outer_values`), and the regime's flat params, and returns the shift
    matrix of shape `(n_durable, n_outer)`.
    """
    if outer_cost_name is None:

        def zero_shifts(
            *, durable_values: FloatND, outer_values: FloatND, **params: object
        ) -> FloatND:
            del params
            return jnp.zeros(
                (durable_values.shape[0], outer_values.shape[0]),
                dtype=durable_values.dtype,
            )

        return zero_shifts

    # The outer post-decision is a leaf: the lift binds it per outer-grid node,
    # so the cost must ask for it directly rather than have it recomputed from
    # the outer action.
    cost_func = concatenate_functions(
        functions={
            name: func
            for name, func in functions.items()
            if name != outer_post_decision
        },
        targets=outer_cost_name,
        enforce_signature=False,
        set_annotations=True,
    )
    cost_arg_names = set(get_annotations(cost_func)) - {"return"}

    def keeper_level(durable: FloatND) -> FloatND:
        # The keeper core realises the outer post-decision at its own
        # no-adjustment level `keep(durable)` — the level whose credited cost
        # is zero. With an identity keeper this is `durable` itself.
        return durable if no_adjustment_func is None else no_adjustment_func(durable)

    def coh_shifts(
        *, durable_values: FloatND, outer_values: FloatND, **params: object
    ) -> FloatND:
        # Defense in depth behind the model-build ancestor check: a cost DAG
        # demanding any binding other than the durable, the outer
        # post-decision, and params cannot be evaluated per (durable, outer)
        # cell.
        cost_extra_arg_names = (
            cost_arg_names - {durable_state_name, outer_post_decision} - set(params)
        )
        if cost_extra_arg_names:
            msg = (
                f"The declared NEGM outer cost '{outer_cost_name}' reads "
                f"{sorted(cost_extra_arg_names)}. It may read only the durable "
                f"state '{durable_state_name}', the outer post-decision "
                f"'{outer_post_decision}', and params — the credited-cost lift "
                "is a constant per (durable, outer-node) cell, so no other "
                "state or action can vary inside it."
            )
            raise InvalidParamsError(msg)

        def cost_at(*, durable: FloatND, outer: FloatND) -> FloatND:
            bindings = {durable_state_name: durable, outer_post_decision: outer}
            return cost_func(
                **{
                    name: value
                    for name, value in bindings.items()
                    if name in cost_arg_names
                },
                **params,
            )

        return jax.vmap(
            lambda durable: jax.vmap(
                lambda outer: (
                    cost_at(durable=durable, outer=outer)
                    - cost_at(durable=durable, outer=keeper_level(durable))
                )
            )(outer_values)
        )(durable_values)

    return coh_shifts


def _without_outer_post_decision(
    *,
    functions: EconFunctionsMapping,
    outer_post_decision: FunctionName,
) -> EconFunctionsMapping:
    """Drop the outer post-decision from the adjuster's econ-function DAG.

    The adjuster is solved with the post-decision bound to an outer-grid node,
    so its value arrives as a param rather than being recomputed from the outer
    action. Leaving the declaring function in the pool would let the inner DAG
    walk through it and conclude that utility and resources depend on the outer
    action, which is exactly what binding the node removes.
    """
    return MappingProxyType(
        {name: func for name, func in functions.items() if name != outer_post_decision}
    )


def _with_no_adjustment_outer_function(
    *,
    functions: EconFunctionsMapping,
    durable_state: StateName,
    outer_post_decision: FunctionName,
    no_adjustment_func: EconFunction | None,
) -> EconFunctionsMapping:
    """Add the keeper's outer post-decision to the econ-function DAG.

    The inner resources function reads the outer post-decision by name. The
    adjuster binds it as a per-node param; the keeper instead holds it at its
    no-adjustment level, so the resources DAG computes it as `keep(...)`. The
    injected function declares every argument the map reads — the durable leaf
    state and any further states, params, or DAG nodes (e.g. a permanent-income
    growth factor) — so concatenation wires each combo/DAG value into resources.
    With no `no_adjustment_func`, `keep` is the identity.

    The injected function replaces whatever the regime declares under the
    post-decision's name, so the keeper's level is what every consumer reads.
    """
    # The outer post-decision keeps its consumer annotation off the existing
    # functions so the DAG's annotation-consistency check stays satisfied.
    outer_annotation = _annotation_of_arg(
        functions=functions, arg_name=outer_post_decision
    )
    if no_adjustment_func is None:
        arg_names: tuple[str, ...] = (durable_state,)
        args_spec = {
            durable_state: _annotation_of_arg(
                functions=functions, arg_name=durable_state
            )
        }
    else:
        annotations = ensure_annotations_are_strings(
            get_annotations(no_adjustment_func)
        )
        arg_names = tuple(name for name in annotations if name != "return")
        args_spec = {name: annotations[name] for name in arg_names}
        args_spec[durable_state] = "ContinuousState"

    @with_signature(args=args_spec, return_annotation=outer_annotation)
    def keep_outer_post_decision(**kwargs: FloatND) -> FloatND:
        if no_adjustment_func is None:
            return kwargs[durable_state]
        return no_adjustment_func(**{name: kwargs[name] for name in arg_names})

    keep_outer_post_decision.__name__ = outer_post_decision
    return MappingProxyType(
        {
            **dict(functions),
            outer_post_decision: cast("EconFunction", keep_outer_post_decision),
        }
    )


def _annotation_of_arg(
    *, functions: EconFunctionsMapping, arg_name: StateOrActionName
) -> str:
    """Return the annotation the regime's functions use for one argument.

    The DAG's annotation-consistency check requires every consumer of a leaf to
    agree on its annotation, so the injected keeper function copies it from the
    first regime function that declares the argument. Falls back to `"FloatND"`
    when no function annotates it.
    """
    for func in functions.values():
        annotations = ensure_annotations_are_strings(get_annotations(func))
        annotation = annotations.get(arg_name, "no_annotation_found")
        if annotation != "no_annotation_found":
            return annotation
    return "FloatND"


def _with_outer_post_decision(
    *,
    flat_params: FlatParams,
    regime_name: RegimeName,
    outer_post_decision: FunctionName,
    value: ScalarFloat,
) -> FlatParams:
    """Bind the outer post-decision value into the regime's flat params.

    The inner DC-EGM core threads its per-combo pool from `flat_params`, so
    binding the value there makes the inner resources and the child-state index
    read the fixed outer node as a constant. Only the post-decision is bound:
    the durable's declared law of motion reads it and produces the next-period
    stock itself, so a law that is not the identity is honoured rather than
    replaced by the node the outer search picked.
    """
    regime_params = {
        **dict(flat_params[regime_name]),
        outer_post_decision: value,
    }
    return MappingProxyType(
        {
            name: (
                MappingProxyType(regime_params) if name == regime_name else regime_pool
            )
            for name, regime_pool in flat_params.items()
        }
    )


def _fail_if_outer_batch_size_negative(
    *, outer_batch_size: int, solver_name: str = "NEGM"
) -> None:
    """Reject a negative outer batch size, naming the solver that declared it."""
    if outer_batch_size < 0:
        msg = (
            f"{solver_name}.outer_batch_size must be non-negative, got "
            f"{outer_batch_size}. Use 0 to solve every outer-grid node in one "
            "block, or a positive value to sweep the outer grid in blocks of "
            "that many nodes."
        )
        raise RegimeInitializationError(msg)


def _fail_if_outer_grid_is_stochastic(outer_grid: ContinuousGrid) -> None:
    if isinstance(outer_grid, _ContinuousStochasticProcess):
        msg = (
            "NEGM.outer_grid must be a deterministic continuous grid, not a "
            f"stochastic process ({type(outer_grid).__name__}). The outer grid "
            "is the exogenous search grid over the durable post-decision margin; "
            "it carries no transition. A stochastic durable margin belongs in a "
            "process state, not the NEGM outer search."
        )
        raise RegimeInitializationError(msg)
