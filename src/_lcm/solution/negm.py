"""The NEGM solver (nested endogenous grid method, Druedahl 2021).

`NEGM` nests a `DCEGM` inner solve: a per-durable-node passive keeper alongside
one adjuster solve per exogenous outer post-decision node, whose conditional
carries are lifted into common cash-on-hand (via the declared outer cost) and
stacked; the parent read collapses the candidate axis by the exact query-side
maximum. `_NEGMPeriodKernel` publishes a native two-program graph: `keeper` is
the inner keeper's own program under a new name, and `outer_sweep` is one
program that maps the inner adjuster over the outer grid in blocks of the
planner's `outer_candidate` width, takes the exact maximum with the keeper
value, and stacks every candidate carry.
The graph declares the dependency between them: `keeper` publishes its value and
carry as typed internal outputs, `outer_sweep` names them as internal inputs, so
the engine lowers the sweep against the keeper's own abstract output and the
kernel hands the keeper's outputs over under those names.
"""

import functools
import logging
import operator
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import ClassVar, cast

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
from _lcm.egm.carry import EGMCarry, egm_carry_role_tree, stack_egm_carries
from _lcm.egm.dcegm_query_model import DCEGMQueryModel
from _lcm.egm.nested_floor import (
    NESTED_FLOOR_INPUT,
    PreparedNestedFloor,
    prepare_nested_floor,
)
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
    ReducedAxis,
    _CheckedCoreVariant,
    core_program_graph,
)
from _lcm.execution.output_layout import VALUE, OutputRoleTree, StateAxesLeading
from _lcm.execution.reductions import HARD_MAX_WITH_CARRY_REDUCTION
from _lcm.grids import ContinuousGrid
from _lcm.params.edges import regime_kernel_params
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.solution.continuation_reads import rekeyed_value_reads
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
from _lcm.solution.dcegm import (
    DCEGM,
    _BoundDCEGM,
    _combination_inputs,
    _consume_bounded_dcegm_output,
    _counted_dcegm_output_roles,
    _DCEGMPeriodKernel,
    dcegm_kernel_with_declared_reads,
)
from _lcm.solution.feasibility import (
    BoundedResult,
    ContinuationCountedKernelOutput,
    FeasibilityCounts,
    _unflatten_bound_status,
    count_current_feasibility,
    merge_bound_input_status,
    merge_continuation_read_counts,
    merge_query_input_status,
    reduce_bound_input_status,
    reduce_continuation_read_counts,
    reduce_query_input_status,
)
from _lcm.time import TimeAxis
from _lcm.typing import (
    EconFunction,
    EconFunctionsMapping,
    FlatParams,
    FlatRegimeParams,
    ParamsLeaf,
    PytreeValue,
    QualifiedName,
    RegimeName,
    ShapeDtypePytree,
)
from lcm._solver_api.capabilities import SolverExecutionCapabilities
from lcm.exceptions import InvalidParamsError, RegimeInitializationError
from lcm.solver_api import (
    EGM_CONTINUATION,
    EGM_ENDOGENOUS_COORDINATE,
    ArtifactKey,
    ContinuationCapabilities,
    KernelOutput,
)
from lcm.typing import (
    ActionName,
    BoolND,
    Float1D,
    FloatND,
    FunctionName,
    ReferenceName,
    ScalarFloat,
    StateName,
    StateOrActionName,
)

# Planner name of the outer post-decision candidate axis the sweep folds.
OUTER_CANDIDATE_AXIS = "outer_candidate"

_OUTER_CANDIDATE_WIDTH_KEYWORD = "_lcm_outer_candidate_width"
# The keeper's outputs and the sweep's own inputs enter the compiled sweep next
# to the inner adjuster's arguments, under engine-only names. A public regime,
# function, state, or action name cannot contain the ``__`` qualified-name
# separator, and a generated qualified parameter name always has a non-empty
# component before its separator, so a key that starts with the separator can
# never be produced by a supported model.
_TRANSPORT_PREFIX = "__lcm_negm_"
_KEEPER_VALUE = f"{_TRANSPORT_PREFIX}keeper_value"
_KEEPER_CARRY = f"{_TRANSPORT_PREFIX}keeper_carry"
_KEEPER_RESULT = f"{_TRANSPORT_PREFIX}keeper_result"
_OUTER_NODES = f"{_TRANSPORT_PREFIX}outer_nodes"
_COH_SHIFTS = f"{_TRANSPORT_PREFIX}coh_shifts"
_DURABLE_NODES = f"{_TRANSPORT_PREFIX}durable_nodes"
_COH_PARAMS = f"{_TRANSPORT_PREFIX}coh_params"

_NESTED_INNER_FIELD_COUNT = 4
_NESTED_KEEPER_FIELD_COUNT = 5

type _NestedInnerOutput = BoundedResult[
    tuple[FloatND, EGMCarry, FeasibilityCounts | None, BoolND]
]
type _NestedKeeperOutput = BoundedResult[
    tuple[FloatND, EGMCarry, FeasibilityCounts | None, BoolND, PreparedNestedFloor]
]
type _NestedSweepOutput = BoundedResult[
    tuple[FloatND, EGMCarry, FeasibilityCounts | None]
]


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

    @property
    def capabilities(self) -> SolverExecutionCapabilities:
        """Describe this configured solver without building numerical kernels."""
        return SolverExecutionCapabilities(
            required_declaration=(
                "NestedConsumptionSavingsRegime with liquid and outer margins"
            ),
            problem_shape="DCEGM inner solve conditional on a finite outer grid",
            prerequisites=(
                "Inner DCEGM contract plus outer state/action, post-decision, "
                "no-adjustment and cost roles; no EV1 taste shocks"
            ),
            main_tradeoff=(
                "Exact relative to the outer candidate set; retained candidates can "
                "dominate memory"
            ),
            reduced_axes=(*self.inner.capabilities.reduced_axes, "outer_candidate"),
            tiled_axes=self.inner.capabilities.tiled_axes,
            host_axes=(),
            host_driven_programs=(),
            supports_ev1_taste_shocks=False,
            supports_nonlinear_certainty_equivalent=False,
        )

    inner: DCEGM
    """The inner 1-D DC-EGM config.

    Carries the liquid Euler state, the consumption action, the resources and
    post-decision functions, the savings grid, and the upper-envelope backend.
    Its `__post_init__` guards run on construction, so an invalid inner config
    is rejected before NEGM's own guards.
    """

    outer_grid: ContinuousGrid
    r"""Exogenous grid over the outer post-decision margin $s_t^\textit{post-dec}$."""

    def __post_init__(self) -> None:
        _fail_if_outer_grid_is_stochastic(self.outer_grid)

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
                inner=replace(
                    self.inner._with_liquid_margin(liquid),  # noqa: SLF001
                    _negm_current_publication=True,
                ),
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
    def required_continuation_capabilities(self) -> ContinuationCapabilities:
        """The EGM step reads the target's value and its marginal in resources."""
        return ContinuationCapabilities(
            value=True, marginal_states=frozenset({EGM_ENDOGENOUS_COORDINATE})
        )

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
            law=context.laws[context.regime_name],
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
                post_decision=bound.inner.post_decision_function,
                bound_spec=bound.inner.post_decision_bound,
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

    def declare_continuation_reads(
        self, *, kernels: SolutionKernels, context: SolverBuildContext
    ) -> SolutionKernels:
        """Declare the carry rows the composite reads through its keeper.

        The two published cores read their targets' carries through the inner
        DC-EGM keeper, so the keeper declares them and the period adapter
        re-derives `keeper` and the sweep built around it — each under its own
        core name — from the keeper the declaration produced.
        """
        return replace(
            kernels,
            period_kernels=MappingProxyType(
                {
                    period: _with_declared_keeper_reads(
                        kernel=kernel, context=context, period=period
                    )
                    for period, kernel in kernels.period_kernels.items()
                }
            ),
        )

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
        query_model = (context.dcegm_query_models or {}).get(context.regime_name)
        if query_model is not None and query_model.authority.finite_outer is None:
            raise TypeError("Finite NEGM requires its canonical outer query authority.")
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
        if query_model is not None:
            adjuster_context = replace(
                adjuster_context,
                dcegm_source_query_model=_derive_negm_inner_query_model(
                    model=query_model,
                    functions=adjuster_context.functions,
                    periods=query_model.authority.target_periods,
                ),
            )
        inner_with_child_transport = replace(
            bound_self.inner, _retain_child_feasibility=True
        )
        adjuster_kernels = inner_with_child_transport.build_period_kernels(
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
            periodized_tree_signature,
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
        keeper_group_keys_by_period: dict[int, Hashable] = {}
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
            if query_model is not None:
                keeper_context = replace(
                    keeper_context,
                    dcegm_source_query_model=_derive_negm_inner_query_model(
                        model=query_model,
                        functions=keeper_context.functions,
                        periods=group_periods or (template_period,),
                    ),
                )
            group_keeper_kernels = inner_with_child_transport.build_period_kernels(
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
                keeper_group_keys_by_period[period] = (
                    group_keeper_kernels.period_group_keys.get(period)
                )
                coh_shift_by_period[period] = group_coh_shift_func
        assert keeper_continuation_template is not None  # noqa: S101
        representative_durable_values = context.grids[durable_state].to_jax()
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
                    durable_grid_values=_durable_values_at(
                        period=period,
                        period_to_state_nodes=context.period_to_state_nodes,
                        durable_state=durable_state,
                        representative_values=representative_durable_values,
                    ),
                    durable_axis_in_carry=durable_axis_in_carry,
                    carry_row_state_names=carry_row_state_names,
                    floor_preparation=(
                        None
                        if query_model is None
                        or query_model.authority.bound.kind != "dag"
                        else _build_negm_floor_preparation(
                            model=query_model,
                            period=period,
                            fixed_state_names=carry_row_state_names,
                        )
                    ),
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
            continuation_reader_capabilities=(
                adjuster_kernels.continuation_reader_capabilities
            ),
            query_input_authorities=(adjuster_kernels.query_input_authorities),
            # The outer helper pool splits the keeper build, and the inner
            # solver may split either branch further, so a period is grouped by
            # its own declared helper signature together with both inner keys.
            period_group_keys=MappingProxyType(
                {
                    period: (
                        "negm",
                        periodized_tree_signature(
                            tree=context.functions, period=period
                        ),
                        adjuster_kernels.period_group_keys.get(period),
                        keeper_group_keys_by_period.get(period),
                    )
                    for period in period_kernels
                }
            ),
            continuation_spec=EGMContinuationSpec(
                template=stacked_template,
                layout=(
                    self.egm_continuation_layout
                    if query_model is None
                    else replace(
                        self.egm_continuation_layout, query_spec=query_model.authority
                    )
                ),
            ),
            # Both inner margins are solved by the inner solver, so both sets of
            # parameter-dependent preconditions still apply to this regime.
            param_checks=(*adjuster_kernels.param_checks, *keeper_param_checks),
        )


def _derive_negm_inner_query_model(
    *,
    model: DCEGMQueryModel,
    functions: EconFunctionsMapping,
    periods: tuple[int, ...],
) -> DCEGMQueryModel:
    """Derive an inner producer view from the canonical finite outer declaration."""
    from _lcm.continuation import build_query_model_authority  # noqa: PLC0415
    from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
        resolve_periodized_nodes,
    )

    if model.authority.finite_outer is None:
        raise TypeError("An inner query view needs canonical finite outer authority.")
    pool = MappingProxyType(
        {
            **{
                name: function
                for name, function in model.functions.items()
                if name != model.authority.finite_outer.outer_post_decision
            },
            **functions,
        }
    )
    by_period = MappingProxyType(
        {
            period: cast(
                "EconFunctionsMapping",
                resolve_periodized_nodes(mapping=pool, period=period),
            )
            for period in periods
        }
    )
    parent = model.authority
    authority = build_query_model_authority(
        bound=parent.bound,
        state_names=parent.state_names,
        action_names=parent.action_names,
        coordinate_kind="resources",
        marginal_convention=parent.marginal_convention,
        corner_coordinate_kind=parent.corner_coordinate_kind,
        row_capacity=parent.row_capacity,
        interval_ids=parent.interval_ids,
        interval_side_ids=parent.interval_side_ids,
        evaluator_targets=tuple(
            (purpose, target)
            for purpose, target in parent.evaluator_targets
            if not purpose.startswith("outer_")
        ),
        functions=by_period[periods[0]],
        functions_by_period=by_period,
        target_periods=periods,
        row_topology=parent.row_topology,
    )
    return replace(model, authority=authority, functions=pool)


@dataclass(frozen=True, kw_only=True)
class _NestedFloorPreparation:
    """Canonical fixed-current computation and the exact transient role tree."""

    function: Callable[..., PreparedNestedFloor]
    """One b(C) evaluation, independent of O and D."""

    output_roles: BoundedResult[OutputRoleTree]
    """Named C axes for the shared floors, shifted nodes and input flags."""


def _build_negm_floor_preparation(
    *,
    model: DCEGMQueryModel,
    period: int,
    fixed_state_names: tuple[StateName, ...],
) -> _NestedFloorPreparation:
    """Declare one canonical b(C) producer before either nested cost evaluation."""
    from _lcm.egm.preferences import concatenate_regime_function  # noqa: PLC0415
    from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
        resolve_periodized_nodes,
    )

    target = model.authority.bound.lower_target
    if target is None or fixed_state_names != model.authority.state_names:
        raise TypeError(
            "Nested floor preparation requires its canonical fixed-state roles."
        )
    functions = cast(
        "EconFunctionsMapping",
        resolve_periodized_nodes(mapping=model.functions, period=period),
    )
    floating = StateAxesLeading(
        state_names=fixed_state_names, dtype=model.offsets.dtype
    )
    boolean = StateAxesLeading(state_names=fixed_state_names, dtype=jnp.bool_)
    return _NestedFloorPreparation(
        function=functools.partial(
            prepare_nested_floor,
            lower=concatenate_regime_function(functions=functions, target=target),
            offsets=model.offsets,
            state_names=fixed_state_names,
        ),
        output_roles=BoundedResult(
            output=floating,
            bound_status=(_unflatten_bound_status(None, (boolean, boolean)),),
        ),
    )


def _durable_values_at(
    *,
    period: int,
    period_to_state_nodes: (
        MappingProxyType[int, MappingProxyType[StateName, Float1D]] | None
    ),
    durable_state: StateName,
    representative_values: Float1D,
) -> Float1D:
    """Return the durable state's node values in one period.

    The durable nodes enter a numerical lift (the credited-cost shift of cash on
    hand), so they must be the solved period's own. With an age-specialized
    durable grid the representative age's nodes are the wrong ones everywhere
    else, so read the period's own when the schedule offers them.

    Args:
        period: The period whose nodes are wanted.
        period_to_state_nodes: Immutable mapping of period to that period's
            age-specialized state nodes, or `None` when the regime has no
            age-specialized state.
        durable_state: Name of the outer margin's state.
        representative_values: The representative age's durable nodes, returned
            whenever the schedule offers none for `period`.

    Returns:
        The durable state's node values in `period`.

    """
    if period_to_state_nodes is None:
        return representative_values
    return period_to_state_nodes.get(period, MappingProxyType({})).get(
        durable_state, representative_values
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


def _with_declared_keeper_reads(
    *, kernel: PeriodKernel, context: SolverBuildContext, period: int
) -> PeriodKernel:
    """Return one NEGM period's adapter rebuilt around a declaring keeper."""
    if not isinstance(kernel, _NEGMPeriodKernel):
        return kernel
    return replace(
        kernel,
        keeper_kernel=dcegm_kernel_with_declared_reads(
            kernel=kernel.keeper_kernel, context=context, period=period
        ),
    )


@dataclass(frozen=True, kw_only=True)
class _NEGMPeriodKernel:
    """The NEGM period kernel — a keeper program and a compiled outer sweep.

    Holds two inner DC-EGM period kernels and the exogenous outer grid. The
    outer durable choice splits into two programs of the kernel's native graph:

    - `keeper` — the inner keeper's own program, a per-durable-state passive
      DC-EGM (`next_illiquid = illiquid`, identity) that keeps the durable
      stock unchanged for free (`credited(s, s) = 0`), run once over the full
      durable grid; and
    - `outer_sweep` — one program that maps the inner adjuster (the DC-EGM
      with the outer transition stripped) over the exogenous outer grid with
      `outer_post_decision` bound to each node, in blocks of the planner's
      `outer_candidate` width, collapses the value by
      `V = max(V_keeper, max_j W_j)` and stacks the keeper carry with every
      node carry, lifted into common cash on hand, on the candidate axis.

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

    fixed_sweep_kwargs: Mapping[QualifiedName, ParamsLeaf] = MappingProxyType({})
    """The regime's and its targets' fixed params, bound into the sweep."""

    floor_preparation: _NestedFloorPreparation | None = None
    """Canonical current-floor preparation preceding both nested cost paths."""

    _core_programs: MappingProxyType[str, CoreProgram] = field(
        init=False, repr=False, compare=False
    )
    """The native graph, `keeper` then `outer_sweep`; derived at construction."""

    def __post_init__(self) -> None:
        """Derive the graph from the inner kernels and the sweep's bindings."""
        keeper = core_program_graph(kernel=self.keeper_kernel)["main"]
        adjuster = core_program_graph(kernel=self.adjuster_kernel)["main"]
        retaining = (
            isinstance(self.keeper_kernel, _DCEGMPeriodKernel)
            and self.keeper_kernel.retains_current_feasibility
        )
        if retaining:
            object.__setattr__(
                self,
                "_core_programs",
                _nested_feasibility_programs(
                    kernel=self, keeper=keeper, adjuster=adjuster
                ),
            )
            return
        sweep_function = functools.partial(
            _outer_sweep_program,
            inner_core=adjuster.function,
            outer_post_decision=self.outer_post_decision,
            durable_axis=self.durable_axis_in_carry,
            **self.fixed_sweep_kwargs,
        )
        row = StateAxesLeading(state_names=self.carry_row_state_names)
        # Both cores read the carries the inner keeper's program declared: the
        # sweep's arguments are the adjuster's plus its own, so the rolling
        # continuation channel those reads address is present in both. Only the
        # locator's core changes.
        inherited = keeper.requirements.value_reads
        programs = {
            "keeper": replace(
                keeper,
                name="keeper",
                # The outer composition owns checked event folds and locators.
                checked_variant=None,
                requirements=replace(
                    keeper.requirements,
                    value_reads=rekeyed_value_reads(reads=inherited, core_key="keeper"),
                ),
                internal_outputs=(
                    InternalOutputSpec(label="value", path=(0,)),
                    InternalOutputSpec(label="carry", path=(1,)),
                ),
            ),
            "outer_sweep": CoreProgram(
                name="outer_sweep",
                function=sweep_function,
                compiler_options=adjuster.compiler_options,
                argument_builder=_NEGMSweepArgumentBuilder(
                    adjuster_builder=adjuster.argument_builder,
                    regime_name=self.regime_name,
                    outer_post_decision=self.outer_post_decision,
                    outer_grid_values=self.outer_grid_values,
                    durable_grid_values=self.durable_grid_values,
                    coh_shift_func=self.coh_shift_func,
                ),
                requirements=CoreExecutionRequirements(
                    reduced_axes=(
                        *adjuster.requirements.reduced_axes,
                        ReducedAxis(
                            name=OUTER_CANDIDATE_AXIS,
                            coordinate_names=(_OUTER_NODES,),
                            coordinate_extents=(self.outer_grid_values.shape[0],),
                            canonical_order="c",
                            reduction=HARD_MAX_WITH_CARRY_REDUCTION,
                            width_keyword=_OUTER_CANDIDATE_WIDTH_KEYWORD,
                        ),
                    ),
                    tiled_axes=adjuster.requirements.tiled_axes,
                    internal_inputs=MappingProxyType(
                        {
                            _KEEPER_VALUE: InternalInputRef(
                                producer="keeper", label="value"
                            ),
                            _KEEPER_CARRY: InternalInputRef(
                                producer="keeper", label="carry"
                            ),
                        }
                    ),
                    value_reads=rekeyed_value_reads(
                        reads=inherited, core_key="outer_sweep"
                    ),
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
                disposition=CoreExecutionDisposition.PLANNED,
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
            regime_kernel_params(fixed_flat_params, regime_name=self.regime_name)
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
            floor_preparation=(
                None
                if self.floor_preparation is None
                else replace(
                    self.floor_preparation,
                    function=functools.partial(
                        self.floor_preparation.function,
                        **union_fixed_params(
                            fixed_flat_params=fixed_flat_params,
                            regime_name=self.regime_name,
                            transition_target_names=self.transition_target_names,
                        ),
                    ),
                )
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
        if (
            isinstance(self.keeper_kernel, _DCEGMPeriodKernel)
            and self.keeper_kernel.retains_current_feasibility
        ):
            return _run_nested_feasibility_kernel(
                kernel=self,
                compiled_cores=compiled_cores,
                context=CoreBuildContext(
                    state_action_space=state_action_space,
                    next_regime_to_V_arr=next_regime_to_V_arr,
                    next_regime_to_continuation=next_regime_to_continuation,
                    flat_params=flat_params,
                    period=period,
                    ages=ages,
                ),
            )
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
        V_arr, carry = cast(
            "tuple[FloatND, EGMCarry]",
            compiled_cores["outer_sweep"](
                **arguments,
                **{_KEEPER_VALUE: keeper_value, _KEEPER_CARRY: keeper_carry},
            ),
        )
        return KernelOutput(value=V_arr, continuations={EGM_CONTINUATION: carry})


def _nested_feasibility_programs(
    *, kernel: _NEGMPeriodKernel, keeper: CoreProgram, adjuster: CoreProgram
) -> MappingProxyType[str, CoreProgram]:
    """Connect actual action eligibility and integrity through both graph variants."""
    keeper_kernel = kernel.keeper_kernel
    adjuster_kernel = kernel.adjuster_kernel
    if (
        not isinstance(keeper_kernel, _DCEGMPeriodKernel)
        or not isinstance(adjuster_kernel, _DCEGMPeriodKernel)
        or not adjuster_kernel.retains_current_feasibility
        or keeper_kernel.own_bound_specs != adjuster_kernel.own_bound_specs
        or keeper_kernel.query_input_authorities
        != adjuster_kernel.query_input_authorities
        or keeper.checked_variant is None
        or adjuster.checked_variant is None
        or type(keeper.output_roles) is not BoundedResult
        or type(keeper.checked_variant.output_roles) is not BoundedResult
    ):
        raise TypeError("Nested DC producers must preserve common integrity owners.")
    keeper_roles = _nested_tuple_roles(roles=keeper.output_roles)
    checked_keeper_roles = _nested_tuple_roles(
        roles=keeper.checked_variant.output_roles
    )
    result_output = (InternalOutputSpec(label="result", path=()),)
    keeper_program = replace(
        keeper,
        name="keeper",
        requirements=replace(
            keeper.requirements,
            value_reads=rekeyed_value_reads(
                reads=keeper.requirements.value_reads, core_key="keeper"
            ),
        ),
        internal_outputs=result_output,
        checked_variant=replace(keeper.checked_variant, internal_outputs=result_output),
    )
    if kernel.floor_preparation is not None:
        preparation = kernel.floor_preparation
        keeper_program = replace(
            keeper_program,
            function=functools.partial(
                _keeper_with_prepared_floor,
                inner=keeper.function,
                prepare_floor=preparation.function,
            ),
            output_roles=replace(
                keeper_roles,
                output=(*keeper_roles.output, preparation.output_roles),
            ),
            checked_variant=replace(
                keeper.checked_variant,
                function=functools.partial(
                    _keeper_with_prepared_floor,
                    inner=keeper.checked_variant.function,
                    prepare_floor=preparation.function,
                ),
                output_roles=replace(
                    checked_keeper_roles,
                    output=(
                        *checked_keeper_roles.output,
                        preparation.output_roles,
                    ),
                ),
                internal_outputs=result_output,
            ),
        )
    output_roles = replace(
        keeper_roles,
        output=(
            VALUE,
            _stacked_nested_carry_roles(
                roles=keeper_roles.output[1],
                n_candidates=kernel.outer_grid_values.shape[0] + 1,
            ),
            keeper_roles.output[2],
        ),
    )
    requirements = CoreExecutionRequirements(
        reduced_axes=(
            *adjuster.requirements.reduced_axes,
            ReducedAxis(
                name=OUTER_CANDIDATE_AXIS,
                coordinate_names=(_OUTER_NODES,),
                coordinate_extents=(kernel.outer_grid_values.shape[0],),
                canonical_order="c",
                reduction=HARD_MAX_WITH_CARRY_REDUCTION,
                width_keyword=_OUTER_CANDIDATE_WIDTH_KEYWORD,
            ),
        ),
        tiled_axes=adjuster.requirements.tiled_axes,
        internal_inputs=MappingProxyType(
            {
                _KEEPER_RESULT: InternalInputRef(producer="keeper", label="result"),
            }
        ),
        value_reads=rekeyed_value_reads(
            reads=keeper.requirements.value_reads, core_key="outer_sweep"
        ),
    )
    common = {
        "outer_post_decision": kernel.outer_post_decision,
        "durable_axis": kernel.durable_axis_in_carry,
        "coh_shift_func": (
            kernel.coh_shift_func if kernel.floor_preparation is not None else None
        ),
        **kernel.fixed_sweep_kwargs,
    }
    return MappingProxyType(
        {
            "keeper": keeper_program,
            "outer_sweep": CoreProgram(
                name="outer_sweep",
                function=functools.partial(
                    _nested_outer_sweep_program,
                    inner_core=adjuster.function,
                    **common,
                ),
                compiler_options=adjuster.compiler_options,
                argument_builder=_NEGMSweepArgumentBuilder(
                    adjuster_builder=adjuster.argument_builder,
                    regime_name=kernel.regime_name,
                    outer_post_decision=kernel.outer_post_decision,
                    outer_grid_values=kernel.outer_grid_values,
                    durable_grid_values=kernel.durable_grid_values,
                    coh_shift_func=kernel.coh_shift_func,
                    defer_costs=kernel.floor_preparation is not None,
                ),
                requirements=requirements,
                output_roles=output_roles,
                disposition=CoreExecutionDisposition.PLANNED,
                donation_candidates=(),
                checked_variant=_CheckedCoreVariant(
                    function=functools.partial(
                        _nested_outer_sweep_program,
                        inner_core=adjuster.checked_variant.function,
                        **common,
                    ),
                    output_roles=_counted_dcegm_output_roles(roles=output_roles),
                ),
            ),
        }
    )


def _nested_tuple_roles(
    *, roles: OutputRoleTree
) -> BoundedResult[tuple[OutputRoleTree, ...]]:
    """Require the bounded tuple roles already declared by the nested DC producer."""
    if type(roles) is not BoundedResult or type(roles.output) is not tuple:
        raise TypeError("A nested DC program must declare bounded tuple roles.")
    return cast("BoundedResult[tuple[OutputRoleTree, ...]]", roles)


def _stacked_nested_carry_roles(
    *, roles: OutputRoleTree, n_candidates: int
) -> OutputRoleTree:
    """Apply the actual carry stack's axis insertion to exact query-buffer shapes."""
    return jax.tree.map_with_path(
        functools.partial(_stacked_nested_carry_role, n_candidates=n_candidates),
        roles,
    )


# keyword-only-exempt: library-callback=jax.tree.map_with_path
def _stacked_nested_carry_role(
    path: tuple[jax.tree_util.GetAttrKey, ...],
    role: StateAxesLeading,
    *,
    n_candidates: int,
) -> StateAxesLeading:
    """Keep fixed-cell floors shared and stack candidate-dependent leaves."""
    names = tuple(key.name for key in path)
    if (
        role.shape is None
        or names in (("taste_shock_scale",), ("query_data", "floor_used"))
        or "adaptive_outer_data" in names
    ):
        return role
    axis = len(role.shape) - (2 if "liquid_corner_data" in names else 1)
    return replace(role, shape=(*role.shape[:axis], n_candidates, *role.shape[axis:]))


def _keeper_with_prepared_floor(
    *,
    inner: Callable[..., PytreeValue],
    prepare_floor: Callable[..., PreparedNestedFloor],
    **arguments: PytreeValue,
) -> _NestedKeeperOutput:
    """Prepare b(C) before keeper costs and transport it to all adjuster cores."""
    prepared = prepare_floor(**arguments)
    result = inner(**arguments, **{NESTED_FLOOR_INPUT: prepared})
    if (
        type(result) is not BoundedResult
        or type(result.output) is not tuple
        or len(result.output) != _NESTED_INNER_FIELD_COUNT
    ):
        raise TypeError("A prepared keeper requires the genuine inner output channels.")
    native = cast("_NestedInnerOutput", result)
    return BoundedResult(
        output=(*native.output, prepared),
        bound_status=native.bound_status,
        query_status=native.query_status,
        continuation_read_counts=native.continuation_read_counts,
    )


def _run_nested_feasibility_kernel(
    *,
    kernel: _NEGMPeriodKernel,
    compiled_cores: Mapping[str, Callable[..., PytreeValue]],
    context: CoreBuildContext,
) -> KernelOutput:
    """Dispatch the raw keeper record before the sole composite host publication."""
    keeper_program = kernel.core_programs()["keeper"]
    keeper = compiled_cores["keeper"](**keeper_program.argument_builder(context))
    sweep_program = kernel.core_programs()["outer_sweep"]
    evaluated = compiled_cores["outer_sweep"](
        **sweep_program.argument_builder(context),
        **{_KEEPER_RESULT: keeper},
    )
    if type(evaluated) is not BoundedResult:
        raise TypeError("A nested sweep must return its declared integrity record.")
    inner = cast("_DCEGMPeriodKernel", kernel.keeper_kernel)
    if inner.query_input_authorities is not None:
        return _consume_bounded_dcegm_output(
            evaluated=evaluated,
            variant="main",
            own_bound_specs=inner.own_bound_specs,
            query_slots=inner.query_input_authorities,
        )
    value, carry, summary = evaluated.output
    if (
        evaluated.bound_status
        or evaluated.query_status is not None
        or summary is not None
    ):
        raise TypeError("Nested integrity records require model-owned output slots.")
    if evaluated.continuation_read_counts is not None:
        return ContinuationCountedKernelOutput(
            value=value,
            continuations={EGM_CONTINUATION: carry},
            continuation_read_counts=evaluated.continuation_read_counts,
        )
    return KernelOutput(value=value, continuations={EGM_CONTINUATION: carry})


def _nested_outer_sweep_program(
    *,
    inner_core: Callable[..., _NestedInnerOutput],
    outer_post_decision: FunctionName,
    durable_axis: int,
    _lcm_outer_candidate_width: int,
    coh_shift_func: Callable[..., FloatND] | None = None,
    **arguments: PytreeValue,
) -> _NestedSweepOutput:
    """Fold every actual outer result while retaining action identity until counting."""
    adjuster_arguments = dict(arguments)
    keeper = cast("_NestedInnerOutput", adjuster_arguments.pop(_KEEPER_RESULT))
    outer_nodes = cast("Float1D", adjuster_arguments.pop(_OUTER_NODES))
    if coh_shift_func is None:
        coh_shifts = cast("FloatND", adjuster_arguments.pop(_COH_SHIFTS))
    else:
        if len(keeper.output) != _NESTED_KEEPER_FIELD_COUNT:
            raise TypeError("A prepared nested keeper must transport its shared floor.")
        prepared = keeper.output[-1]
        keeper = replace(keeper, output=keeper.output[:-1])
        adjuster_arguments[NESTED_FLOOR_INPUT] = prepared
        coh_shifts = coh_shift_func(
            durable_values=adjuster_arguments.pop(_DURABLE_NODES),
            outer_values=outer_nodes,
            **cast("FlatRegimeParams", adjuster_arguments.pop(_COH_PARAMS)),
        )
    n_nodes = outer_nodes.shape[0]
    results = jax.lax.map(
        _NestedNodeSolver(
            inner_core=inner_core,
            outer_post_decision=outer_post_decision,
            adjuster_arguments=MappingProxyType(adjuster_arguments),
        ),
        outer_nodes,
        batch_size=min(_lcm_outer_candidate_width, n_nodes),
    )
    keeper_value, keeper_carry, _keeper_counts, keeper_feasible = keeper.output
    node_values, node_carries, _node_counts, node_feasible = results.output
    value = jnp.maximum(keeper_value, jnp.max(node_values, axis=0))
    carry = build_stacked_outer_carry(
        keeper_carry=keeper_carry,
        adjuster_carries=tuple(
            jax.tree.map(operator.itemgetter(index), node_carries)
            for index in range(n_nodes)
        ),
        coh_shifts=coh_shifts,
        durable_axis=durable_axis,
    )
    active_outer = jnp.ones(outer_nodes.shape, dtype=jnp.bool_)
    node_bound_status = reduce_bound_input_status(
        statuses=results.bound_status, active=active_outer
    )
    if len(keeper.bound_status) != len(node_bound_status):
        raise TypeError("Nested outer results must retain canonical bound slots.")
    return BoundedResult(
        output=(
            value,
            carry,
            (
                _count_nested_current_feasibility(
                    keeper=keeper_feasible, adjusters=node_feasible, value=value
                )
                if keeper.bound_status or keeper.query_status is not None
                else None
            ),
        ),
        bound_status=tuple(
            merge_bound_input_status(left=left, right=right)
            for left, right in zip(keeper.bound_status, node_bound_status, strict=True)
        ),
        query_status=merge_query_input_status(
            left=keeper.query_status,
            right=reduce_query_input_status(
                statuses=results.query_status, active=active_outer
            ),
        ),
        continuation_read_counts=merge_continuation_read_counts(
            left=keeper.continuation_read_counts,
            right=reduce_continuation_read_counts(
                counts=results.continuation_read_counts, active=active_outer
            ),
        ),
    )


def _count_nested_current_feasibility(
    *, keeper: BoolND, adjusters: BoolND, value: FloatND
) -> FeasibilityCounts:
    """Count physical states after joint eligibility over genuine outer/action pairs."""
    if (
        keeper.shape[:-1] != value.shape
        or adjusters.shape[1:] != keeper.shape
        or keeper.shape[-1] < 1
    ):
        raise TypeError("Nested eligibility must retain state, outer and action axes.")
    joint = keeper | jnp.any(adjusters, axis=0)
    return count_current_feasibility(
        budget_feasible=joint,
        predicate_feasible=jnp.ones((*joint.shape, 0), dtype=jnp.bool_),
        value=value,
        active_states=jnp.ones(value.shape, dtype=jnp.bool_),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _NestedNodeSolver:
    """Return a genuine inner result with its independent integrity/count channels."""

    inner_core: Callable[..., _NestedInnerOutput]
    """The actual adjuster producer, including its retained action mask."""

    outer_post_decision: FunctionName
    """Name under which the outer node enters the actual inner DAG."""

    adjuster_arguments: Mapping[ReferenceName, PytreeValue]
    """The inner producer's declared live arguments."""

    def __call__(self, node: ScalarFloat) -> _NestedInnerOutput:
        """Evaluate one actual outer node without selecting diagnostic winners."""
        return self.inner_core(
            **{**self.adjuster_arguments, self.outer_post_decision: node}
        )


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

    defer_costs: bool = False
    """Evaluate costs inside the sweep after the shared floor and keeper dependency."""

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
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
        if self.defer_costs:
            own = {
                _OUTER_NODES: self.outer_grid_values,
                _DURABLE_NODES: self.durable_grid_values,
                _COH_PARAMS: regime_kernel_params(
                    flat_params, regime_name=self.regime_name
                ),
            }
            _fail_if_sweep_inputs_collide_with_the_adjusters(
                arguments=arguments, own=own, regime_name=self.regime_name
            )
            return MappingProxyType({**arguments, **own})
        own = {
            _OUTER_NODES: self.outer_grid_values,
            _COH_SHIFTS: self.coh_shift_func(
                durable_values=self.durable_grid_values,
                outer_values=self.outer_grid_values,
                **regime_kernel_params(flat_params, regime_name=self.regime_name),
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
    _lcm_outer_candidate_width: int,
    **arguments: PytreeValue,
) -> tuple[FloatND, EGMCarry]:
    """Solve the adjuster at every outer node and stack it with the keeper.

    The inner adjuster program runs once per exogenous node with the outer
    post-decision bound into its arguments, in blocks of the planner's
    `outer_candidate` width. The value is the exact maximum of the keeper value
    and every node value; the continuation is the keeper carry followed by
    every node carry lifted into common cash on hand on the candidate axis.
    Every candidate carry is retained, so the block width reschedules the sweep
    without changing which operations run on which operands.

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

    node_values, node_carries = jax.lax.map(
        _NodeSolver(
            inner_core=inner_core,
            outer_post_decision=outer_post_decision,
            adjuster_arguments=MappingProxyType(adjuster_arguments),
        ),
        outer_nodes,
        batch_size=min(_lcm_outer_candidate_width, n_nodes),
    )
    V_arr = jnp.maximum(keeper_value, jnp.max(node_values, axis=0))
    carry = build_stacked_outer_carry(
        keeper_carry=keeper_carry,
        adjuster_carries=tuple(
            jax.tree.map(operator.itemgetter(index), node_carries)
            for index in range(n_nodes)
        ),
        coh_shifts=coh_shifts,
        durable_axis=durable_axis,
    )
    return V_arr, carry


@dataclass(frozen=True, kw_only=True, eq=False)
class _NodeSolver:
    """Solve the inner adjuster at one outer node.

    Every input is an explicit field, so the adjuster core stays reachable only
    through this instance, which the sweep drops when its trace ends. The
    argument tree holds traced arrays, whose truth value is undefined, so
    instances compare by identity (`eq=False`).
    """

    inner_core: Callable[..., tuple[FloatND, EGMCarry]]
    """The adjuster program, called with the outer post-decision bound in."""

    outer_post_decision: FunctionName
    """Argument name under which the outer node is bound."""

    adjuster_arguments: Mapping[ReferenceName, PytreeValue]
    """Immutable mapping of the adjuster's own argument tree."""

    def __call__(self, node: ScalarFloat) -> tuple[FloatND, EGMCarry]:
        """Return the adjuster's value and carry at `node`."""
        return self.inner_core(
            **{**self.adjuster_arguments, self.outer_post_decision: node}
        )


def _fail_if_sweep_inputs_collide_with_the_adjusters(
    *,
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
    own: Mapping[ReferenceName, PytreeValue],
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
    if template.query_data_for_transport() is None:
        leaves = template.leaves()
        return EGMCarry(
            endog_grid=_broadcast_over_candidates(
                arr=leaves[("endog_grid",)], n_candidates=n_candidates
            ),
            value=_broadcast_over_candidates(
                arr=leaves[("value",)], n_candidates=n_candidates
            ),
            marginal_utility=_broadcast_over_candidates(
                arr=leaves[("marginal_utility",)], n_candidates=n_candidates
            ),
            taste_shock_scale=leaves[("taste_shock_scale",)],
        )
    return stack_egm_carries(candidates=(template,) * n_candidates)


def _broadcast_over_candidates(*, arr: FloatND, n_candidates: int) -> FloatND:
    """Broadcast one carry leaf across the candidate axis before the grid axis."""
    return jnp.broadcast_to(
        arr[..., None, :], (*arr.shape[:-1], n_candidates, arr.shape[-1])
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
        return _zero_coh_shifts
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
    return _CreditedCoHShifts(
        cost_func=cost_func,
        cost_arg_names=frozenset(get_annotations(cost_func)) - {"return"},
        durable_state_name=durable_state_name,
        outer_post_decision=outer_post_decision,
        outer_cost_name=outer_cost_name,
        no_adjustment_func=no_adjustment_func,
    )


def _zero_coh_shifts(
    *, durable_values: FloatND, outer_values: FloatND, **params: ParamsLeaf
) -> FloatND:
    """Return the identically zero cash-on-hand shift matrix.

    A regime whose resources never read the outer post-decision credits no cost
    back, so every (durable, outer-node) cell shifts by zero.
    """
    del params
    return jnp.zeros(
        (durable_values.shape[0], outer_values.shape[0]),
        dtype=durable_values.dtype,
    )


@dataclass(frozen=True, kw_only=True)
class _CreditedCoHShifts:
    """The per-(durable, outer-node) cash-on-hand shift of a declared outer cost.

    Every value the shift reads is an explicit field, so the concatenated cost
    DAG stays reachable only through this instance, which the model build drops
    with the period kernel that holds it.
    """

    cost_func: Callable[..., FloatND]
    """The concatenated DAG targeting the regime's declared outer-cost node."""

    cost_arg_names: frozenset[ReferenceName]
    """Immutable set of the argument names that DAG declares."""

    durable_state_name: StateName
    """Name of the durable state the cost may read."""

    outer_post_decision: FunctionName
    """Name of the outer post-decision the cost may read."""

    outer_cost_name: FunctionName
    """Name of the declared outer-cost node, named in the diagnostic."""

    no_adjustment_func: EconFunction | None
    """The keeper's no-adjustment map, or `None` for the identity."""

    def __call__(
        self, *, durable_values: FloatND, outer_values: FloatND, **params: ParamsLeaf
    ) -> FloatND:
        """Return the shift matrix of shape `(n_durable, n_outer)`."""
        _fail_if_the_outer_cost_reads_beyond_one_cell(
            cost_arg_names=self.cost_arg_names,
            durable_state_name=self.durable_state_name,
            outer_post_decision=self.outer_post_decision,
            outer_cost_name=self.outer_cost_name,
            param_names=frozenset(params),
        )
        cell = _CreditedShiftAtCell(
            cost=_OuterCostAtCell(
                cost_func=self.cost_func,
                cost_arg_names=self.cost_arg_names,
                durable_state_name=self.durable_state_name,
                outer_post_decision=self.outer_post_decision,
                params=MappingProxyType(dict(params)),
            ),
            no_adjustment_func=self.no_adjustment_func,
        )
        over_outer_nodes = jax.vmap(cell, in_axes=(None, 0))
        return jax.vmap(over_outer_nodes, in_axes=(0, None))(
            durable_values, outer_values
        )


def _fail_if_the_outer_cost_reads_beyond_one_cell(
    *,
    cost_arg_names: frozenset[ReferenceName],
    durable_state_name: StateName,
    outer_post_decision: FunctionName,
    outer_cost_name: FunctionName,
    param_names: frozenset[QualifiedName],
) -> None:
    """Reject a cost DAG that needs a binding the per-cell lift cannot supply.

    Defense in depth behind the model-build ancestor check: a cost DAG demanding
    any binding other than the durable, the outer post-decision, and params
    cannot be evaluated per (durable, outer) cell.
    """
    cost_extra_arg_names = (
        cost_arg_names - {durable_state_name, outer_post_decision} - param_names
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


@dataclass(frozen=True, kw_only=True, eq=False)
class _OuterCostAtCell:
    """Evaluate the regime's declared outer cost at one (durable, outer) cell.

    The bound params may hold arrays, whose truth value is undefined, so
    instances compare by identity (`eq=False`).
    """

    cost_func: Callable[..., FloatND]
    """The concatenated DAG targeting the declared outer-cost node."""

    cost_arg_names: frozenset[ReferenceName]
    """Immutable set of the argument names that DAG declares."""

    durable_state_name: StateName
    """Name the cost DAG reads the durable state under."""

    outer_post_decision: FunctionName
    """Name the cost DAG reads the outer post-decision under."""

    params: Mapping[QualifiedName, ParamsLeaf]
    """Immutable mapping of the regime's flat params bound into the cost."""

    def __call__(self, *, durable: FloatND, outer: FloatND) -> FloatND:
        """Return the declared cost at one durable level and one outer node."""
        bindings = {self.durable_state_name: durable, self.outer_post_decision: outer}
        return self.cost_func(
            **{
                name: value
                for name, value in bindings.items()
                if name in self.cost_arg_names
            },
            **self.params,
        )


@dataclass(frozen=True, kw_only=True)
class _CreditedShiftAtCell:
    """The credited-cost shift of cash on hand at one (durable, outer) cell."""

    cost: _OuterCostAtCell
    """The declared outer cost, evaluated per cell."""

    no_adjustment_func: EconFunction | None
    """The keeper's no-adjustment map, or `None` for the identity."""

    # keyword-only-exempt: library-callback=jax.vmap
    def __call__(self, durable: FloatND, outer: FloatND) -> FloatND:
        """Return `cost(durable, outer) - cost(durable, keep(durable))`.

        The keeper core realises the outer post-decision at its own
        no-adjustment level `keep(durable)` — the level whose credited cost is
        zero. With an identity keeper this is `durable` itself.
        """
        keeper_level = (
            durable
            if self.no_adjustment_func is None
            else self.no_adjustment_func(durable)
        )
        return self.cost(durable=durable, outer=outer) - self.cost(
            durable=durable, outer=keeper_level
        )


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
        arg_names: tuple[ReferenceName, ...] = (durable_state,)
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

    keep_outer_post_decision = with_signature(
        _KeeperOuterPostDecision(
            durable_state=durable_state,
            arg_names=arg_names,
            no_adjustment_func=no_adjustment_func,
        ),
        args=args_spec,
        return_annotation=outer_annotation,
    )
    keep_outer_post_decision.__name__ = outer_post_decision  # ty: ignore[unresolved-attribute]
    return MappingProxyType(
        {
            **dict(functions),
            outer_post_decision: cast("EconFunction", keep_outer_post_decision),
        }
    )


@dataclass(frozen=True, kw_only=True)
class _KeeperOuterPostDecision:
    """Realize the outer post-decision at the keeper's no-adjustment level.

    Every value the map reads is an explicit field, so the regime's
    no-adjustment candidate stays reachable only through this instance, which
    the model build drops with the keeper function pool that holds it. The
    caller wraps it in `dags.with_signature` to advertise the arguments the
    inner resources DAG binds by name.
    """

    __name__: ClassVar[str] = "keep_outer_post_decision"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    durable_state: StateName
    """Name of the durable leaf state the identity keeper returns."""

    arg_names: tuple[ReferenceName, ...]
    """Tuple of the argument names the no-adjustment map declares."""

    no_adjustment_func: EconFunction | None
    """The regime's no-adjustment map, or `None` for the identity."""

    def __call__(self, **kwargs: FloatND) -> FloatND:
        """Return the keeper's outer post-decision level."""
        if self.no_adjustment_func is None:
            return kwargs[self.durable_state]
        return self.no_adjustment_func(
            **{name: kwargs[name] for name in self.arg_names}
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

    Only the source's own branch receives the node. Its edge parameters stay in
    `flat_params["edges"]`, which the inner argument builder joins with the own
    branch when it binds the kernel.
    """
    regime_params = {
        **dict(cast("FlatRegimeParams", flat_params[regime_name])),
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
