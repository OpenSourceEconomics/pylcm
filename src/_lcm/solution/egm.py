"""The one-asset EGM solver.

`EGM` runs the single-asset endogenous grid method for a regime with
one continuous (Euler) state and no discrete kinks — the specialization whose
step needs no upper envelope. The kernel-building imports are function-local
so the public `lcm.solvers` façade stays a thin re-export that pulls in no
numerical engine modules.
"""

import functools
import inspect
import logging
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.constraints.bounds import proves_the_savings_grids_lower_bound
from _lcm.constraints.routes import (
    ConstraintRoute,
    ConstraintRouteKey,
    ConstraintSite,
)
from _lcm.continuation import EGMContinuationSpec
from _lcm.egm.carry import EGMCarry, egm_carry_role_tree, egm_query_role_tree
from _lcm.egm.interp import interp_on_regular_grid
from _lcm.egm.one_asset_egm_step import RetiredEGMResult
from _lcm.egm.preferences import Preferences
from _lcm.egm.query_data import (
    EGMQueryData,
    QueryNodeKind,
    build_template_egm_query_data,
)
from _lcm.egm.scalar_query import (
    PreparedScalarEGMQuery,
    ScalarEGMQuery,
    build_scalar_egm_query,
)
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
)
from _lcm.execution.output_layout import VALUE, OutputRoleLeaf, StateAxesLeading
from _lcm.grids import ContinuousGrid
from _lcm.post_decision_bound import PostDecisionBoundSpec, grid_origin_bound_spec
from _lcm.probability import is_negative, regime_mass_is_a_distribution
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
    PeriodKernel,
    SolutionKernels,
    SolverBuildContext,
    SolverModelContext,
    _BoundLiquidMargin,
    bind_roles,
    simulation_route,
)
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    FeasibilityCounts,
    FeasibilityKernelOutput,
    count_current_feasibility,
    validate_runtime_bound_status,
)
from _lcm.solution.periodization import (
    resolve_solver_build_context,
    solver_period_group_key,
)
from _lcm.time import TimeAxis
from _lcm.typing import (
    EconFunction,
    EconFunctionArg,
    EconFunctionKwargs,
    EconFunctionsMapping,
    FlatParams,
    ParamsLeaf,
    PytreeValue,
    QualifiedName,
    RegimeName,
    RegimeTransitionFunction,
    ShapeDtypePytree,
)
from _lcm.zero_safe import zero_safe_weighted_term
from lcm._solver_api.capabilities import SolverExecutionCapabilities
from lcm.exceptions import (
    InvalidValueFunctionError,
    ModelInitializationError,
    RegimeInitializationError,
    UnsupportedOperationError,
)
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
    IntND,
    ReferenceName,
    ScalarBool,
    ScalarFloat,
    StateName,
    ValueND,
)

# A jit-able one-row core: the value array and the marginal-value carry on the
# liquid grid.
type BoundedOneRowOutput = BoundedResult[
    tuple[FloatND, EGMCarry, FeasibilityCounts, ScalarBool]
]
type OneRowOutput = tuple[FloatND, EGMCarry] | BoundedOneRowOutput
type OneRowCore = Callable[..., OneRowOutput]


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class EGM(OneMarginSolver):
    """Endogenous-grid solver for a 1-D consumption--saving regime.

    A regime with exactly one continuous state (the liquid wealth), one
    continuous consumption action, and no discrete choice is a plain
    consumption--saving problem. The single continuous state needs no upper
    envelope: inverting the consumption Euler equation on the post-decision
    savings grid and mapping the resulting endogenous wealth back onto the
    regular grid solves the period exactly. The step carries the marginal
    value of liquid backward (the envelope theorem makes it exact, unlike a
    finite difference of a coarse value array), so each period both reads its
    continuation's marginal and publishes its own.
    """

    @property
    def capabilities(self) -> SolverExecutionCapabilities:
        """Describe this configured solver without building numerical kernels."""
        return SolverExecutionCapabilities(
            required_declaration="ConsumptionSavingsRegime with one LiquidMargin",
            problem_shape="Smooth concave one-state/one-action cash-on-hand problem",
            prerequisites=(
                "One continuous state and action; no discrete/process axes; identity "
                "resources; additive continuation; provable post-decision lower bound"
            ),
            main_tradeoff="Narrow structural contract with no upper envelope",
            reduced_axes=(),
            tiled_axes=(),
            host_axes=(),
            host_driven_programs=(),
            supports_ev1_taste_shocks=False,
            supports_nonlinear_certainty_equivalent=False,
        )

    savings_grid: ContinuousGrid
    """Exogenous post-decision savings grid; its lower bound is the borrowing limit.

    Zero for a household that cannot borrow, minus the limit for one that can.
    The corner is read off this bound rather than assumed to be zero savings.
    """

    def _with_liquid_margin(self, margin: _BoundLiquidMargin) -> _BoundEGM:
        """Bind regime-owned DAG names without exposing them on public `EGM`."""
        return cast(
            "_BoundEGM",
            bind_roles(
                solver=self,
                role_type=_BoundEGM,
                continuous_state=margin.state,
                continuous_action=margin.action,
                resources=margin.resources,
                post_decision_function=margin.post_decision_state,
                post_decision_bound=grid_origin_bound_spec(
                    post_decision=margin.post_decision_state,
                    liquid_state=margin.state,
                    action=margin.action,
                    resources_target=margin.resources,
                    grid=self.savings_grid,
                ),
            ),
        )

    @property
    def required_continuation_keys(self) -> frozenset[ArtifactKey]:
        """The 1-D EGM step reads its continuation's marginal value of liquid."""
        return frozenset({EGM_CONTINUATION})

    @property
    def required_continuation_capabilities(self) -> ContinuationCapabilities:
        """The EGM step reads the target's value and its marginal in resources."""
        return ContinuationCapabilities(
            value=True, marginal_states=frozenset({EGM_ENDOGENOUS_COORDINATE})
        )

    def validate_model(  # noqa: C901, PLR0912, PLR0915
        self, *, context: SolverModelContext
    ) -> None:
        """Check the preconditions the envelope-free EGM kernel assumes.

        Plain EGM is specialized to cash-on-hand as its continuous state: the
        kernel constructs its endogenous grid as `state = action + savings`, so
        `resources(state) == state` and `post_decision(state, action) ==
        state - action` are **assumptions of the method**, not properties it
        verifies and not capabilities it offers. A model whose resources map
        genuinely transforms the state belongs on DCEGM or GridSearch, or
        should redefine its state as cash-on-hand.

        What happens here is a spot check for ordinary specification mistakes,
        and it is a diagnostic rather than a proof. Resources is evaluated at
        every represented state node; the post-decision identity is sampled.
        Neither establishes an identity *between* nodes, and the kernel can
        evaluate off-grid, so a callable that agrees wherever it is checked and
        differs elsewhere will be admitted and then silently not applied. That
        is a limit of finite evaluation, not something a denser probe closes —
        so this must not be read as a guarantee that every incompatible
        callable is rejected.
        """
        bound = cast("_BoundEGM", self)
        from dags import concatenate_functions  # noqa: PLC0415

        from _lcm.egm.validation import (  # noqa: PLC0415
            _call_with_varied,
            _continuous_non_process_names,
            _dag_ancestors,
            _grid_sample,
            _isclose,
            _solve_grids,
            fail_if_custom_koopmans_aggregator,
            fail_if_declared_lower_bound_disagrees_with_the_grid,
        )

        regime_name = context.regime_name
        user_regime = context.user_regimes[regime_name]
        fail_if_custom_koopmans_aggregator(
            regime_name=regime_name,
            user_regime=user_regime,
            solver_name="EGM",
        )
        fail_if_declared_lower_bound_disagrees_with_the_grid(
            regime_name=regime_name,
            user_regime=user_regime,
            solver=bound,
            solver_name="EGM",
        )
        if context.laws[regime_name].terminal:
            msg = (
                f"Regime '{regime_name}' is terminal but configured with EGM. "
                "Terminal regimes have no optimization problem; remove the "
                "solver setting."
            )
            raise ModelInitializationError(msg)

        state_grids = _solve_grids(slot=user_regime.states)
        action_grids = _solve_grids(slot=user_regime.actions)
        continuous_states = _continuous_non_process_names(grids=state_grids)
        continuous_actions = _continuous_non_process_names(grids=action_grids)
        if len(continuous_states) != 1:
            msg = (
                f"EGM regime '{regime_name}' must have exactly one continuous "
                f"state, but has {len(continuous_states)}: "
                f"{list(continuous_states)}."
            )
            raise ModelInitializationError(msg)
        if len(continuous_actions) != 1:
            msg = (
                f"EGM regime '{regime_name}' must have exactly one continuous "
                f"action, but has {len(continuous_actions)}: "
                f"{list(continuous_actions)}."
            )
            raise ModelInitializationError(msg)
        discrete_states = sorted(set(state_grids) - set(continuous_states))
        if discrete_states:
            msg = (
                f"EGM regime '{regime_name}' has discrete or process states "
                f"{discrete_states}. The envelope-free one-row kernel publishes "
                "a one-dimensional continuation and cannot carry those axes; "
                "use DCEGM or GridSearch."
            )
            raise ModelInitializationError(msg)
        discrete_actions = sorted(set(action_grids) - set(continuous_actions))
        if discrete_actions:
            msg = (
                f"EGM regime '{regime_name}' has discrete actions "
                f"{discrete_actions}. A discrete choice makes the candidate "
                "correspondence multi-valued; use DCEGM."
            )
            raise ModelInitializationError(msg)

        liquid_state = continuous_states[0]
        consumption_action = continuous_actions[0]
        if bound.post_decision_function not in user_regime.functions:
            msg = (
                f"EGM regime '{regime_name}' is missing the declared "
                f"post-decision function '{bound.post_decision_function}'."
            )
            raise ModelInitializationError(msg)
        functions = context.solve_functions
        post_func = functions[bound.post_decision_function]
        post_ancestors = _dag_ancestors(functions=functions, target_func=post_func)
        missing_roles = sorted({liquid_state, consumption_action} - set(post_ancestors))
        if missing_roles:
            msg = (
                f"The post-decision function '{bound.post_decision_function}' "
                f"of EGM regime '{regime_name}' must depend on the state "
                f"'{liquid_state}' and action '{consumption_action}'; its DAG "
                f"does not reach {missing_roles}."
            )
            raise ModelInitializationError(msg)

        utility_ancestors = _dag_ancestors(
            functions=functions, target_func=functions["utility"]
        )
        if liquid_state in utility_ancestors:
            msg = (
                f"The utility DAG of EGM regime '{regime_name}' depends on "
                f"the continuous state '{liquid_state}'. The envelope-free "
                "kernel evaluates utility as a function of the continuous "
                f"action '{consumption_action}' only."
            )
            raise ModelInitializationError(msg)

        composed_post = concatenate_functions(
            functions, targets=bound.post_decision_function
        )
        post_arguments = set(inspect.signature(composed_post).parameters)
        expected_arguments = {liquid_state, consumption_action}
        if post_arguments != expected_arguments:
            msg = (
                f"The post-decision DAG '{bound.post_decision_function}' of EGM "
                f"regime '{regime_name}' must be exactly a function of "
                f"'{liquid_state}' and '{consumption_action}', but its leaf "
                f"arguments are {sorted(post_arguments)}."
            )
            raise ModelInitializationError(msg)
        state_sample = _grid_sample(grid=state_grids[liquid_state])
        action_sample = _grid_sample(grid=action_grids[consumption_action])
        x64_enabled = bool(jax.config.read("jax_enable_x64"))
        atol = 1e-8 if x64_enabled else 1e-4
        rtol = 1e-6 if x64_enabled else 1e-3
        composed_resources = concatenate_functions(functions, targets=bound.resources)
        resources_arguments = set(inspect.signature(composed_resources).parameters)
        if resources_arguments != {liquid_state}:
            msg = (
                f"The resources DAG '{bound.resources}' of EGM regime "
                f"'{regime_name}' must be exactly a function of "
                f"'{liquid_state}', but its leaf arguments are "
                f"{sorted(resources_arguments)}. The envelope-free kernel reads "
                f"resources off the liquid state itself, so anything the "
                f"declared resources add or subtract is silently ignored; use "
                f"DCEGM or GridSearch."
            )
            raise ModelInitializationError(msg)
        # Every represented state node, not a sample: resources takes exactly
        # the liquid state, so one vectorized call costs no more than a few
        # scalar ones and catches a mistake at any tabulated point. It remains
        # a diagnostic -- the kernel can evaluate off-grid, and no finite set of
        # evaluations establishes an identity between the nodes.
        all_state_nodes = state_grids[liquid_state].to_jax()
        resources_at_nodes = _call_with_varied(
            func=composed_resources, fixed={}, varied={liquid_state: all_state_nodes}
        )
        disagreeing = ~jnp.isclose(
            jnp.asarray(resources_at_nodes), all_state_nodes, rtol=rtol, atol=atol
        )
        if bool(jnp.any(disagreeing)):
            first = int(jnp.argmax(disagreeing))
            msg = (
                f"The resources function '{bound.resources}' of EGM regime "
                f"'{regime_name}' must equal the liquid state "
                f"'{liquid_state}'. At {liquid_state}="
                f"{float(all_state_nodes[first])} it returns "
                f"{float(jnp.asarray(resources_at_nodes)[first])}. The "
                f"envelope-free kernel inverts the Euler equation against the "
                f"state directly, so a resources function that transforms it is "
                f"not applied; use DCEGM or GridSearch."
            )
            raise ModelInitializationError(msg)
        for state_value in state_sample:
            for action_value in action_sample:
                actual = _call_with_varied(
                    func=composed_post,
                    fixed={},
                    varied={
                        liquid_state: state_value,
                        consumption_action: action_value,
                    },
                )
                expected = state_value - action_value
                if not _isclose(actual=actual, expected=expected, rtol=rtol, atol=atol):
                    msg = (
                        f"Consumption recovery fails in EGM regime "
                        f"'{regime_name}': '{bound.post_decision_function}' must "
                        f"equal `{liquid_state} - {consumption_action}`. At "
                        f"{liquid_state}={float(state_value)}, "
                        f"{consumption_action}={float(action_value)}, it returns "
                        f"{float(actual)} rather than {float(expected)}."
                    )
                    raise ModelInitializationError(msg)

    def build_constraint_routes(
        self, *, context: ConstraintRouteContext
    ) -> tuple[ConstraintRoute, ...]:
        """Declare the one route plain EGM walks in each phase.

        The envelope-free kernel calls no user predicate anywhere. It builds
        its endogenous grid by inverting the Euler equation on the savings
        grid, and the only requirement it honours is the borrowing limit that
        grid's lowest node already is. Its solve route is therefore a single
        site that evaluates nothing — a place a constraint can be discharged,
        never one where it can be called — and anything that site cannot
        discharge is refused rather than quietly dropped.

        Simulation is a different pipeline with a different answer. There the
        subject's own realized action is in hand, so the feasibility check runs
        over a whole candidate, including the budget constraint the phase
        synthesizes for an endogenous-grid regime.
        """
        bound = cast("_BoundEGM", self)
        proofs = (
            proves_the_savings_grids_lower_bound(
                post_decision=bound.post_decision_function,
                bound_spec=bound.post_decision_bound,
            ),
        )
        if context.phase == "simulate":
            return (simulation_route(context=context, solver_path=("egm",)),)
        return (
            ConstraintRoute(
                key=ConstraintRouteKey(
                    phase="solve", period_group=None, solver_path=("egm",)
                ),
                sites=(
                    ConstraintSite(
                        stage="savings_stage",
                        function_pool=context.functions,
                        available_names=frozenset(),
                        structural_proofs=proofs,
                    ),
                ),
            ),
        )

    def validate_build(self, *, context: SolverBuildContext) -> None:
        """Check the regime and its targets are 1-D consumption--saving problems.

        The solver's liquid role is filled positionally, which is only
        unambiguous with a single continuous state. The same is asked of each
        target, and for the same reason: with one continuous state on each side
        the correspondence is determined, whatever the two regimes call it.
        Every message reports the regimes' own state names, never the solver's
        internal role vocabulary.

        The regime must also carry no discrete action and keep the default
        Koopmans aggregator: a discrete choice folds the candidate value
        correspondence, which is the case the envelope-free step does not
        cover, and the Euler inversion the step runs is the one that
        aggregator implies.
        """
        from _lcm.egm.validation import (  # noqa: PLC0415
            fail_if_custom_koopmans_aggregator,
        )

        fail_if_custom_koopmans_aggregator(
            regime_name=context.regime_name,
            user_regime=context.user_regimes[context.regime_name],
            solver_name="EGM",
        )
        discrete_actions = tuple(context.state_action_space.discrete_actions)
        if discrete_actions:
            msg = (
                f"EGM regime '{context.regime_name}' has discrete actions "
                f"{sorted(discrete_actions)}. A discrete choice makes the "
                f"candidate value correspondence multi-valued, which the "
                f"envelope-free one-asset step does not solve. Use DCEGM, "
                f"which refines the candidates to their upper envelope."
            )
            raise ModelInitializationError(msg)
        continuous = tuple(
            context.regime_to_v_interpolation_info[
                context.regime_name
            ].continuous_states
        )
        if len(continuous) != 1:
            msg = (
                f"EGM regime '{context.regime_name}' must have exactly one "
                f"continuous state, but has {len(continuous)}: {list(continuous)}. "
                f"Use a solver that handles more than one Euler state, or move the "
                f"extra states to discrete grids."
            )
            raise ModelInitializationError(msg)
        for target in set(period_to_continuation_target(context=context).values()):
            target_user_regime = context.user_regimes[target]
            if context.laws[target].terminal and target_user_regime.actions:
                msg = (
                    f"EGM regime '{context.regime_name}' continues into terminal "
                    f"regime '{target}', which has actions "
                    f"{list(target_user_regime.actions)}. A terminal EGM carry "
                    "is utility on the state grid and cannot represent a final-"
                    "period optimization; remove the actions or use GridSearch "
                    "for the parent."
                )
                raise ModelInitializationError(msg)
            target_states = tuple(
                context.regime_to_v_interpolation_info[target].continuous_states
            )
            if len(target_states) != 1:
                msg = (
                    f"EGM regime '{context.regime_name}' continues into target "
                    f"regime '{target}', whose continuous states are "
                    f"{sorted(target_states)}. The Euler inversion reads a single "
                    f"continuation state, so the target must declare exactly one — "
                    f"its name need not match this regime's '{continuous[0]}'."
                )
                raise ModelInitializationError(msg)
            target_discrete = tuple(
                context.regime_to_v_interpolation_info[target].discrete_states
            )
            if target_discrete:
                msg = (
                    f"EGM regime '{context.regime_name}' continues into target "
                    f"regime '{target}', whose discrete/process states are "
                    f"{sorted(target_discrete)}. The one-row continuation "
                    "kernel cannot carry those axes."
                )
                raise ModelInitializationError(msg)

    def declare_continuation_reads(
        self, *, kernels: SolutionKernels, context: SolverBuildContext
    ) -> SolutionKernels:
        """Declare the three carry rows each period's arguments carry."""
        return declare_egm_carry_reads(kernels=kernels, context=context)

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one 1-D EGM period adapter per active period.

        Each period's adapter knows the single deterministic continuation
        target (the transition target whose value array and marginal-utility carry
        feed the Euler inversion).
        """
        bound = cast("_BoundEGM", self)

        savings_grid = self.savings_grid.to_jax()
        query_models = context.scalar_query_models or {}
        current_model = query_models.get(context.regime_name)
        liquid_state = next(
            iter(
                context.regime_to_v_interpolation_info[
                    context.regime_name
                ].continuous_states
            )
        )
        liquid_grid = context.grids[liquid_state].to_jax()
        # The regime's single continuous action fills the consumption role, the
        # same positional reading as the liquid state above. It is the argument
        # the regime's felicity, its marginal, and its inverse are functions of.
        consumption_action = next(iter(context.state_action_space.continuous_actions))

        from _lcm.egm.declared_law import build_declared_liquid_law  # noqa: PLC0415

        variable_names = (
            frozenset(context.state_action_space.states)
            | frozenset(context.state_action_space.continuous_actions)
            | frozenset(context.state_action_space.discrete_actions)
        )
        period_to_target = period_to_continuation_target(context=context)
        cores: dict[Hashable, OneRowCore] = {}
        source_queries: dict[Hashable, ScalarEGMQuery | None] = {}
        child_capabilities: dict[Hashable, ContinuationCapabilities] = {}
        reader_capabilities: dict[
            int, Mapping[RegimeName, ContinuationCapabilities]
        ] = {}
        laws: dict[Hashable, Callable[..., tuple[Float1D, Float1D]]] = {}
        period_kernels: dict[int, PeriodKernel] = {}
        period_group_keys: dict[int, Hashable] = {}
        # The target's own name for its single continuous state. It is read off
        # that regime: the value grid it is tabulated on and the namespace its
        # transition params live under are both facts about the target, so
        # neither is inherited from this regime's spelling.
        target_state_names = {
            target: next(
                iter(context.regime_to_v_interpolation_info[target].continuous_states)
            )
            for target in period_to_target.values()
        }
        for period, target in period_to_target.items():
            target_state = target_state_names[target]
            group_key = solver_period_group_key(
                context=context,
                period=period,
                continuation_targets=(target,),
                solver_path=("egm",),
            )
            next_model = query_models.get(target)
            from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
                periodized_tree_signature,
                resolve_periodized_node,
                resolve_periodized_nodes,
            )

            if next_model is not None:
                group_key = (
                    group_key,
                    periodized_tree_signature(
                        tree=next_model.functions, period=period + 1
                    ),
                )
            if group_key not in cores:
                resolved = resolve_solver_build_context(context=context, period=period)
                laws[group_key] = build_declared_liquid_law(
                    transitions=resolved.transitions,
                    functions=resolved.functions,
                    post_decision_name=bound.post_decision_function,
                    target=target,
                    target_state=target_state,
                    variable_names=variable_names,
                )
                core = _build_egm_core(
                    declared_law=laws[group_key],
                    savings_grid=savings_grid
                    if current_model is None
                    else current_model.offsets,
                    functions=resolved.functions,
                    koopmans_aggregator=cast(
                        "EconFunction", resolved.koopmans_aggregator
                    ),
                    consumption_action=consumption_action,
                    current_query=None
                    if current_model is None
                    else build_scalar_egm_query(
                        model=current_model,
                        functions=resolved.functions,
                        koopmans_aggregator=cast(
                            "EconFunction", resolved.koopmans_aggregator
                        ),
                    ),
                    next_query=None
                    if next_model is None
                    else build_scalar_egm_query(
                        model=next_model,
                        functions=cast(
                            "EconFunctionsMapping",
                            resolve_periodized_nodes(
                                mapping=next_model.functions, period=period + 1
                            ),
                        ),
                        koopmans_aggregator=cast(
                            "EconFunction",
                            resolve_periodized_node(
                                node=next_model.koopmans_aggregator, period=period + 1
                            ),
                        ),
                    ),
                )
                cores[group_key] = guard_regime_mass(
                    core=core, enable_jit=context.enable_jit
                )
                source_queries[group_key] = core.current_query
                if core.next_query is not None:
                    child_capabilities[group_key] = core.next_query.capabilities
            period_group_keys[period] = group_key
            if group_key in child_capabilities:
                reader_capabilities[period] = MappingProxyType(
                    {target: child_capabilities[group_key]}
                )
            period_kernels[period] = _build_egm_period_kernel(
                core=cores[group_key],
                declared_law=laws[group_key],
                compute_regime_transition_probs=context.compute_regime_transition_probs,
                savings_grid=savings_grid,
                regime_name=context.regime_name,
                continuation_target=target,
                liquid_state=liquid_state,
                transition_target_names=tuple(context.transitions),
                next_liquid_grid=target_period_grid(
                    context=context,
                    period=period,
                    target=target,
                    target_state_name=target_state,
                ),
                current_query=source_queries[group_key],
                query_template=None
                if current_model is None
                else build_template_egm_query_data(
                    state_shape=(),
                    action_shape=(),
                    row_capacity=liquid_grid.size,
                    interval_capacity=1,
                ),
                own_bound_specs=()
                if current_model is None
                else (current_model.authority.bound,),
                inherited_bound_specs=()
                if next_model is None
                else (next_model.authority.bound,),
                source_period=period,
            )
        return SolutionKernels(
            continuation_reader_capabilities=reader_capabilities or None,
            period_kernels=MappingProxyType(period_kernels),
            period_group_keys=MappingProxyType(period_group_keys),
            continuation_spec=EGMContinuationSpec(
                template=replace(
                    _build_one_asset_carry_template(liquid_grid=liquid_grid),
                    query_data=None
                    if current_model is None
                    else build_template_egm_query_data(
                        state_shape=(),
                        action_shape=(),
                        row_capacity=liquid_grid.size,
                        interval_capacity=1,
                    ),
                ),
                layout=self.egm_continuation_layout
                if current_model is None
                else replace(
                    self.egm_continuation_layout,
                    rows_share_state_grid=True,
                    query_spec=current_model.authority,
                ),
            ),
        )


@dataclass(frozen=True, kw_only=True)
class _BoundEGM(EGM):
    """Internal EGM configuration with regime-resolved DAG role names.

    Public `EGM` contains numerical configuration only.  A
    `ConsumptionSavingsRegime` binds its liquid margin into this private
    subclass before model processing, allowing the numerical implementation to
    keep using explicit names without re-exposing them on the public solver.
    """

    continuous_state: StateName
    """Name of the liquid state whose resources the action is drawn from."""

    continuous_action: ActionName
    """Name of the liquid action the endogenous grid is expressed in."""

    resources: FunctionName
    """Name of the function giving resources available for that action."""

    post_decision_function: FunctionName
    """Name of the function giving the savings the exogenous grid spans."""
    post_decision_bound: PostDecisionBoundSpec | None = None
    """Literal grid floor installed as a construction capability."""


# The carry rows `_EGMArgumentBuilder.__call__` flattens into named program
# arguments, keyed by the leaf path each row occupies in the target's payload.
_EGM_ARGUMENT_BY_LEAF: MappingProxyType[tuple[str, ...], str] = MappingProxyType(
    {
        ("endog_grid",): "next_liquid_grid",
        ("value",): "next_value",
        ("marginal_utility",): "next_marginal",
        ("query_data",): "next_query_data",
    }
)


def declare_egm_carry_reads(
    *, kernels: SolutionKernels, context: SolverBuildContext
) -> SolutionKernels:
    """Attach each one-row EGM period's declared carry reads to its program.

    The one-row argument builder flattens its target's carry into three named
    arguments, so each period declares one read per named row, addressed by the
    argument holding it. Shared by every solver that builds this adapter. A
    period whose adapter is a different graph, and a target publishing no
    readable payload, declare nothing.
    """
    return replace(
        kernels,
        period_kernels=MappingProxyType(
            {
                period: _with_declared_carry_reads(
                    kernel=kernel, context=context, period=period
                )
                for period, kernel in kernels.period_kernels.items()
            }
        ),
    )


def _with_declared_carry_reads(
    *, kernel: PeriodKernel, context: SolverBuildContext, period: int
) -> PeriodKernel:
    """Return one period's adapter with its three carry-row reads declared."""
    if not isinstance(kernel, _EGMPeriodKernel):
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
                    argument_by_leaf=_EGM_ARGUMENT_BY_LEAF,
                )
            },
        ),
    )


def _build_egm_period_kernel(
    *,
    core: OneRowCore,
    declared_law: Callable[..., tuple[Float1D, Float1D]],
    compute_regime_transition_probs: RegimeTransitionFunction | None,
    savings_grid: Float1D,
    regime_name: RegimeName,
    continuation_target: RegimeName,
    liquid_state: StateName,
    transition_target_names: tuple[RegimeName, ...],
    next_liquid_grid: Float1D,
    publishes_breakpoints: bool = False,
    current_query: ScalarEGMQuery | None = None,
    query_template: EGMQueryData | None = None,
    own_bound_specs: tuple[PostDecisionBoundSpec, ...] = (),
    inherited_bound_specs: tuple[PostDecisionBoundSpec, ...] = (),
    source_period: int = 0,
) -> _EGMPeriodKernel:
    """Declare one period's EGM program graph around a shared core.

    The core is shared by every period of one solver group, so those periods'
    programs share one compiled function; the argument builder and the kernel
    are the period's own. The single program `main` publishes the value array
    and the one-row carry, both replicated across any device mesh: the regime's
    only state is the liquid axis the rows sit on. `publishes_breakpoints` says
    whether the core's carry carries a breakpoints row (a single-liquid NB-EGM
    core with feasibility boundaries does; the plain EGM core does not).

    The program's continuation-leaf reads are attached later, by
    `declare_egm_carry_reads`: which rows the target publishes is known only
    once every regime is built.
    """
    argument_builder = _EGMArgumentBuilder(
        regime_name=regime_name,
        continuation_target=continuation_target,
        liquid_state=liquid_state,
        transition_target_names=transition_target_names,
        declared_law=declared_law,
        savings_grid=savings_grid,
        compute_regime_transition_probs=compute_regime_transition_probs,
        current_query=current_query,
        has_child_query=bool(inherited_bound_specs),
        own_bound_specs=own_bound_specs,
    )
    scalar_role = StateAxesLeading(state_names=(), shape=())
    ordinary_roles = (
        VALUE,
        egm_carry_role_tree(
            row=StateAxesLeading(state_names=()),
            scalar=scalar_role,
            breakpoints=StateAxesLeading(state_names=())
            if publishes_breakpoints
            else None,
            policy=None,
            query_data=None
            if query_template is None
            else egm_query_role_tree(query_data=query_template, state_names=()),
        ),
    )
    output_roles = ordinary_roles
    if own_bound_specs or inherited_bound_specs:
        numeric_template = BoundedResult(
            output=(
                jnp.zeros_like(savings_grid),
                replace(
                    _build_one_asset_carry_template(liquid_grid=savings_grid),
                    query_data=query_template,
                ),
                (jnp.int32(0), jnp.int32(0)),
                jnp.asarray(a=False),
            ),
            bound_status=tuple(
                BoundInputStatus(
                    nonfinite_floor=jnp.asarray(a=False),
                    invalid_shifted_grid=jnp.asarray(a=False),
                )
                for _ in (*own_bound_specs, *inherited_bound_specs)
            ),
        )
        output_roles = jax.tree.map_with_path(_scalar_egm_output_role, numeric_template)
    program = CoreProgram(
        name="main",
        function=core,
        argument_builder=argument_builder,
        requirements=CoreExecutionRequirements(),
        output_roles=output_roles,
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )
    return _EGMPeriodKernel(
        _core_programs=MappingProxyType({"main": program}),
        regime_name=regime_name,
        continuation_target=continuation_target,
        liquid_state=liquid_state,
        transition_target_names=transition_target_names,
        next_liquid_grid=next_liquid_grid,
        own_bound_specs=own_bound_specs,
        inherited_bound_specs=inherited_bound_specs,
        source_period=source_period,
    )


@dataclass(frozen=True, kw_only=True)
class _EGMArgumentBuilder:
    """Build the EGM program's arguments for lowering and execution.

    The program takes the liquid grid, the continuation target's carry rows, the
    two readings of the regime's declared law of motion on the savings grid, the
    savings candidates bracketing the target's breakpoints, and the free params
    of the regime and of its transition targets. No target value array is read:
    the continuation is interpolated from the carry alone.
    """

    regime_name: RegimeName
    """Name of the regime whose flat params the program projects."""

    continuation_target: RegimeName
    """The regime active next period; its carry continues this one."""

    liquid_state: StateName
    """The regime's own name for the state filling the program's liquid role.

    The core takes its state grid under the private keyword `liquid`; this is
    the name the modeller gave it, used to look the grid up in the state-action
    space and to qualify the liquid law's parameters.
    """

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    declared_law: Callable[..., tuple[Float1D, Float1D]]
    """The regime's own law toward the target, as a function of savings."""

    savings_grid: Float1D
    """The post-decision grid the law is tabulated on."""

    compute_regime_transition_probs: RegimeTransitionFunction | None
    """The regime's transition probabilities, possibly periodized by age."""

    bound_params: Mapping[QualifiedName, ParamsLeaf] = MappingProxyType({})
    """Fixed params bound into the core, kept so the law can read them too."""
    current_query: ScalarEGMQuery | None = None
    """Installed source floor evaluator, absent on literal and legacy adapters."""
    has_child_query: bool = False
    """Whether the plain source core has a trusted feature-child adapter."""
    own_bound_specs: tuple[PostDecisionBoundSpec, ...] = ()
    """Canonical source declarations used before an aborting law-order error."""

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        """Return the exact kwargs shared by lowering and the runtime call."""
        state_action_space = cast("StateActionSpace", context.state_action_space)
        flat_params = cast("FlatParams", context.flat_params)
        next_carry = context.next_regime_to_continuation[self.continuation_target]
        next_carry = cast("EGMCarry", next_carry)
        leaves = next_carry.leaves()
        query_data = next_carry.query_data_for_transport()
        (
            effective_savings_grid,
            next_liquid,
            marginal_return,
            boundary_savings_targets,
            boundary_next_liquid,
            source_bound_status,
        ) = self._law_readings(
            flat_params=flat_params,
            next_breakpoints=leaves.get(("breakpoints",)),
            period=context.period,
            age=cast("TimeAxis", context.ages).values[context.period],
        )
        pool = {
            **self.bound_params,
            **union_free_params(
                flat_params=flat_params,
                regime_name=self.regime_name,
                transition_target_names=self.transition_target_names,
            ),
        }
        child_args = {}
        if self.has_child_query:
            prefix = f"{self.continuation_target}__"
            child_args = {
                "next_query_pool": MappingProxyType(
                    {
                        **{
                            name.removeprefix(prefix): value
                            for name, value in pool.items()
                            if name.startswith(prefix)
                        },
                        "period": jnp.int32(context.period + 1),
                        "age": cast("TimeAxis", context.ages).values[
                            context.period + 1
                        ],
                    }
                )
            }
        return MappingProxyType(
            {
                "liquid": state_action_space.states[self.liquid_state],
                "next_liquid_grid": leaves[("endog_grid",)],
                "next_liquid": next_liquid,
                "marginal_return": marginal_return,
                "effective_savings_grid": effective_savings_grid,
                "boundary_savings_targets": boundary_savings_targets,
                "boundary_next_liquid": boundary_next_liquid,
                "next_value": leaves[("value",)],
                "next_marginal": leaves[("marginal_utility",)],
                **(
                    {"source_bound_status": source_bound_status}
                    if source_bound_status is not None
                    else {}
                ),
                **child_args,
                **({"next_query_data": query_data} if query_data is not None else {}),
                "retains_regime_mass": self._retains_regime_mass(
                    liquid=state_action_space.states[self.liquid_state],
                    flat_params=flat_params,
                    period=context.period,
                    ages=cast("TimeAxis", context.ages),
                ),
                **union_free_params(
                    flat_params=flat_params,
                    regime_name=self.regime_name,
                    transition_target_names=self.transition_target_names,
                ),
                "period": jnp.int32(context.period),
                "age": cast("TimeAxis", context.ages).values[context.period],
            }
        )

    def _retains_regime_mass(
        self, *, liquid: Float1D, flat_params: FlatParams, period: int, ages: TimeAxis
    ) -> BoolND:
        """Whether the continuation target carries the regime's whole mass.

        The core reads the target's carry at weight one, so every other target
        the regime sends probability to is unrepresented. The target's own
        probability must be a distribution on its own, under the same tolerance
        and sign test the grid-search continuation applies, at every liquid node.
        """
        from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
            resolve_periodized_node,
        )

        pool = {
            **self.bound_params,
            **union_free_params(
                flat_params=flat_params,
                regime_name=self.regime_name,
                transition_target_names=self.transition_target_names,
            ),
            "period": jnp.int32(period),
            "age": ages.values[period],
        }
        compute_probs = cast(
            "RegimeTransitionFunction",
            resolve_periodized_node(
                node=self.compute_regime_transition_probs, period=period
            ),
        )

        def target_probability(liquid_node: FloatND) -> FloatND:
            # The solve-phase transition is a scalar DAG. Map the whole call,
            # including coarse-ID/one-hot conversion, before stacking nodes;
            # a row call would confuse the state axis with the regime axis.
            node_pool = cast(
                "dict[str, EconFunctionArg]",
                {**pool, self.liquid_state: liquid_node},
            )
            return jnp.asarray(compute_probs(**node_pool)[self.continuation_target])

        prob = jax.vmap(target_probability)(liquid)
        return jnp.all(
            regime_mass_is_a_distribution(
                probability_mass=prob, has_negative_probability=is_negative(prob)
            )
        )

    def _law_readings(
        self,
        *,
        flat_params: FlatParams,
        next_breakpoints: FloatND | None,
        period: int,
        age: ParamsLeaf,
    ) -> tuple[Float1D, Float1D, Float1D, Float1D, Float1D, BoundInputStatus | None]:
        """Read the declared law on the savings grid and check it can be inverted.

        Evaluated here rather than inside the compiled core: the readings depend
        on the params alone, and their ordering is a Python-level decision that
        no traced value can make. The check therefore has to see concrete
        numbers, which is exactly what this side of the boundary has.
        """
        law_params = {
            **self.bound_params,
            **union_free_params(
                flat_params=flat_params,
                regime_name=self.regime_name,
                transition_target_names=self.transition_target_names,
            ),
            "period": jnp.int32(period),
            "age": age,
        }
        from _lcm.egm.declared_law import (  # noqa: PLC0415
            fail_if_declared_law_is_not_increasing,
        )

        saving_nodes = self.savings_grid
        status = None
        if self.current_query is not None:
            _floor, saving_nodes, status = self.current_query.evaluate_floor(
                pool=law_params
            )
        next_liquid, marginal_return = self.declared_law(
            savings_grid=saving_nodes,
            **law_params,
        )
        try:
            fail_if_declared_law_is_not_increasing(
                next_liquid=next_liquid,
                regime_name=self.regime_name,
                target=self.continuation_target,
            )
        except RegimeInitializationError:
            if status is not None:
                validate_runtime_bound_status(
                    statuses=(status,),
                    bound_specs=self.own_bound_specs,
                    regime_name=self.regime_name,
                    period=period,
                )
            raise
        if next_breakpoints is None:
            empty = jnp.zeros((0,), dtype=self.savings_grid.dtype)
            return (
                saving_nodes,
                next_liquid,
                marginal_return,
                empty,
                empty,
                status,
            )

        boundary_targets = _one_sided_boundary_savings_targets(
            savings_grid=self.savings_grid,
            next_liquid=next_liquid,
            breakpoints=jnp.ravel(next_breakpoints),
            declared_law=self.declared_law,
            law_params=law_params,
        )
        safe_boundary_targets = jnp.where(
            jnp.isfinite(boundary_targets),
            boundary_targets,
            self.savings_grid[0],
        )
        effective_savings_grid = jnp.sort(
            a=jnp.concatenate((self.savings_grid, safe_boundary_targets))
        )
        next_liquid, marginal_return = self.declared_law(
            savings_grid=effective_savings_grid,
            **law_params,
        )
        boundary_next_liquid, _ = self.declared_law(
            savings_grid=boundary_targets,
            **law_params,
        )
        return (
            effective_savings_grid,
            next_liquid,
            marginal_return,
            boundary_targets,
            boundary_next_liquid,
            status,
        )


@dataclass(frozen=True, kw_only=True)
class _EGMPeriodKernel:
    """The 1-D EGM period kernel: one native program around the shared core.

    `main` runs `egm_one_asset_step` for the period and publishes the value
    array and the marginal-value carry a parent EGM regime interpolates. The
    one-row kernel reads its continuation from the target's carry alone and
    declares no target value access; its regime carries no stochastic state, so
    it declares no streamed axis either and the plan owns nothing to widen.
    Calling the kernel builds the program's arguments through its declared
    builder and returns a public `KernelOutput`.
    """

    _core_programs: Mapping[str, CoreProgram]
    """The immutable one-node program graph, `main`."""

    regime_name: RegimeName
    """Name of the regime whose flat params this kernel projects."""

    continuation_target: RegimeName
    """The regime active next period; its carry continues this one."""

    liquid_state: StateName
    """The regime's own name for the state filling the program's liquid role."""

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    next_liquid_grid: Float1D
    """The continuation target's liquid nodes in the *next* period.

    The abscissae of the continuation value and marginal this kernel reads. Equal
    to this period's own grid unless the liquid state is an `AgeSpecializedGrid`.
    """
    own_bound_specs: tuple[PostDecisionBoundSpec, ...] = ()
    """Source-owned records forwarded to ordinary backward acceptance."""
    inherited_bound_specs: tuple[PostDecisionBoundSpec, ...] = ()
    """Canonical child declaration slots, independent of source bound presence."""
    source_period: int = 0
    """Source computational period supplying the child's exact next-period slot."""

    def __post_init__(self) -> None:
        """Snapshot the graph and require exactly the `main` program."""
        programs = MappingProxyType(dict(self._core_programs))
        if tuple(programs) != ("main",):
            msg = (
                "An EGM kernel publishes exactly the program 'main'; got "
                f"{tuple(programs)}."
            )
            raise ValueError(msg)
        object.__setattr__(self, "_core_programs", programs)

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Return the native graph used by eager, JIT, AOT, and replay paths."""
        return self._core_programs

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> _EGMPeriodKernel:
        """Bind the regime's and its targets' fixed params into the program.

        The core receives them as bound keywords; the builder keeps them so the
        declared law reads the same params the core does.
        """
        bound = union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=self.regime_name,
            transition_target_names=self.transition_target_names,
        )
        if not bound:
            return self
        program = self._core_programs["main"]
        argument_builder = cast("_EGMArgumentBuilder", program.argument_builder)
        return replace(
            self,
            _core_programs=MappingProxyType(
                {
                    "main": replace(
                        program,
                        function=functools.partial(program.function, **bound),
                        argument_builder=replace(
                            argument_builder,
                            bound_params=MappingProxyType(
                                {**argument_builder.bound_params, **bound}
                            ),
                        ),
                    )
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
    ) -> KernelOutput:
        """Run the compiled `main` program and publish its typed continuation."""
        if (
            self.own_bound_specs or self.inherited_bound_specs
        ) and period != self.source_period:
            raise TypeError(
                "Scalar query period differs from its model-owned authority."
            )
        program = self._core_programs["main"]
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
        evaluated = compiled_cores["main"](**arguments)
        if self.own_bound_specs or self.inherited_bound_specs:
            if type(evaluated) is not BoundedResult:
                raise TypeError(
                    "A scalar query consumer is missing mandatory integrity records."
                )
            own_count = len(self.own_bound_specs)
            if len(evaluated.bound_status) != own_count + len(
                self.inherited_bound_specs
            ):
                raise TypeError(
                    "Scalar query records differ from model-owned declaration slots."
                )
            validate_runtime_bound_status(
                statuses=evaluated.bound_status[own_count:],
                bound_specs=self.inherited_bound_specs,
                regime_name=self.continuation_target,
                period=period + 1,
            )
            V_arr, carry, counts, coverage = evaluated.output
            if bool(coverage):
                validate_runtime_bound_status(
                    statuses=evaluated.bound_status[:own_count],
                    bound_specs=self.own_bound_specs,
                    regime_name=self.regime_name,
                    period=period,
                )
                raise InvalidValueFunctionError(
                    f"query_coverage_missing in regime '{self.continuation_target}', "
                    f"period {period + 1}."
                )
            return FeasibilityKernelOutput(
                value=V_arr,
                continuations={EGM_CONTINUATION: carry},
                feasibility_counts=counts,
                bound_status=evaluated.bound_status[:own_count],
            )
        V_arr, carry = cast("tuple[FloatND, EGMCarry]", evaluated)
        return KernelOutput(
            value=V_arr,
            continuations={EGM_CONTINUATION: carry},
        )


def _one_sided_boundary_savings_targets(
    *,
    savings_grid: Float1D,
    next_liquid: Float1D,
    breakpoints: Float1D,
    declared_law: Callable[..., tuple[Float1D, Float1D]],
    law_params: EconFunctionKwargs,
) -> Float1D:
    """Map child boundaries to exact and adjacent savings candidates."""
    finite = jnp.isfinite(breakpoints)
    safe_breakpoints = jnp.where(finite, breakpoints, next_liquid[0])
    roots = jnp.interp(safe_breakpoints, next_liquid, savings_grid)
    for _ in range(8):
        landed, slope = declared_law(savings_grid=roots, **law_params)
        safe_slope = jnp.where(slope > 0.0, slope, 1.0)
        roots = jnp.clip(
            roots - (landed - safe_breakpoints) / safe_slope,
            savings_grid[0],
            savings_grid[-1],
        )

    landed, slope = declared_law(savings_grid=roots, **law_params)
    local_intercept = landed - slope * roots
    rounding = jnp.finfo(savings_grid.dtype).eps * (
        jnp.abs(slope * roots) + jnp.abs(local_intercept)
    )
    margin = 4.0 * rounding / jnp.where(slope > 0.0, slope, 1.0)
    candidates = jnp.stack((roots - margin, roots, roots + margin), axis=-1)
    in_law_range = (
        finite
        & (breakpoints >= next_liquid[0])
        & (breakpoints <= next_liquid[-1])
        & (slope > 0.0)
    )
    valid = (
        in_law_range[:, None]
        & (candidates >= savings_grid[0])
        & (candidates <= savings_grid[-1])
    )
    return jnp.where(valid, candidates, jnp.nan).reshape(-1)


# keyword-only-exempt: library-callback=jax.tree.map_with_path
def _scalar_egm_output_role(
    path: tuple[jax.tree_util.GetAttrKey | jax.tree_util.SequenceKey, ...],
    leaf: FloatND | BoolND | IntND,
) -> OutputRoleLeaf:
    """Reconstruct feature output roles through the registered numeric pytree."""
    if path[0] == jax.tree_util.GetAttrKey("output") and path[
        1
    ] == jax.tree_util.SequenceKey(0):
        return VALUE
    if (
        path[0] == jax.tree_util.GetAttrKey("output")
        and path[1] == jax.tree_util.SequenceKey(1)
        and path[2]
        in (
            jax.tree_util.GetAttrKey("endog_grid"),
            jax.tree_util.GetAttrKey("value"),
            jax.tree_util.GetAttrKey("marginal_utility"),
        )
    ):
        return StateAxesLeading(state_names=())
    return StateAxesLeading(state_names=(), shape=leaf.shape, dtype=leaf.dtype)


def _build_egm_core(
    *,
    savings_grid: Float1D,
    functions: EconFunctionsMapping,
    koopmans_aggregator: EconFunction,
    consumption_action: ActionName,
    declared_law: Callable[..., tuple[Float1D, Float1D]],
    current_query: ScalarEGMQuery | None = None,
    next_query: ScalarEGMQuery | None = None,
) -> _EGMCore:
    """Build the jit-able 1-D EGM core for the savings grid.

    The core reads the state grid under the private role keyword `liquid`, the
    continuation value and marginal, the two readings of the declared law of
    motion (where each savings level lands, and how that landing point moves
    with savings), and the regime's scalar params. It runs `egm_one_asset_step`
    and returns the value array and the marginal-value carry on the liquid grid.

    The period adapter checks the law's Euler-node readings outside the compiled
    region. The core also uses the same composed law to value reconstructed interior
    actions at their actual associated savings, with source time and parameters.

    Preferences and the discount factor come from the regime itself: the
    felicity trio is bound out of `functions` at each call, and beta is read
    off the aggregator's own signature.
    """
    from _lcm.egm.one_asset_egm_step import egm_one_asset_step  # noqa: PLC0415
    from _lcm.egm.preferences import (  # noqa: PLC0415
        NEWTON_ACTION_FLOOR,
        get_discount_factor_reader,
        get_preferences_builder,
        newton_action_ceiling,
    )

    build_preferences = get_preferences_builder(
        functions=functions,
        action_name=consumption_action,
        action_lower=NEWTON_ACTION_FLOOR,
        action_upper=newton_action_ceiling(savings_grid),
    )
    read_discount_factor = get_discount_factor_reader(
        functions=functions, koopmans_aggregator=koopmans_aggregator
    )
    return _EGMCore(
        egm_one_asset_step=egm_one_asset_step,
        declared_law=declared_law,
        build_preferences=build_preferences,
        read_discount_factor=read_discount_factor,
        current_query=current_query,
        next_query=next_query,
    )


@dataclass(frozen=True, eq=False)
class _EGMCore:
    """The jit-able 1-D EGM core bound to one regime's preferences and discounting."""

    egm_one_asset_step: Callable[..., RetiredEGMResult]
    """The one-asset EGM step the core runs."""
    declared_law: Callable[..., tuple[Float1D, Float1D]]
    """Actual source law used to value reconstructed associated savings."""
    build_preferences: Callable[[EconFunctionKwargs], Preferences]
    """Build the felicity trio from the regime's scalar params."""
    read_discount_factor: Callable[[EconFunctionKwargs], ScalarFloat]
    """Read the discount factor off the regime's scalar params."""
    current_query: ScalarEGMQuery | None = None
    """Source feature activation, sharing the installed canonical bound."""
    next_query: ScalarEGMQuery | None = None
    """Trusted scalar target adapter bound before leaf-read declarations."""

    def __call__(
        self,
        *,
        liquid: Float1D,
        next_liquid_grid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        next_liquid: Float1D,
        marginal_return: Float1D,
        boundary_savings_targets: Float1D,  # noqa: ARG002
        boundary_next_liquid: Float1D,  # noqa: ARG002
        effective_savings_grid: Float1D,
        next_query_data: EGMQueryData | None = None,
        source_bound_status: BoundInputStatus | None = None,
        next_query_pool: EconFunctionKwargs | None = None,
        **params: ValueND,
    ) -> OneRowOutput:
        """Run one EGM step and return the value array and the marginal-value carry."""
        continuation_args = {}
        prepared_child = None
        statuses = () if source_bound_status is None else (source_bound_status,)
        coverage = jnp.asarray(a=False)
        if self.next_query is not None:
            if next_query_data is None or next_query_pool is None:
                raise TypeError(
                    "A model-bound scalar query lacks its declared numeric inputs."
                )
            prepared_child = self.next_query.prepare(pool=next_query_pool)
            result = self.next_query.read_prepared(
                query=next_liquid,
                grid=next_liquid_grid,
                value=next_value,
                marginal=next_marginal,
                data=next_query_data,
                prepared=prepared_child,
            )
            continuation_args = {
                "continuation_value": result.value,
                "continuation_marginal": result.marginal,
            }
            statuses = (*statuses, result.bound_status)
            coverage = jnp.any(result.coverage_missing)
        elif next_query_data is not None:
            raise UnsupportedOperationError(
                "Feature rows require the model-bound query adapter."
            )
        preferences = self.build_preferences(params)
        beta = self.read_discount_factor(params)
        step = self.egm_one_asset_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            next_liquid_grid=next_liquid_grid,
            savings_grid=effective_savings_grid,
            discount_factor=beta,
            preferences=preferences,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            **continuation_args,
            feature_floor=self.current_query is not None or self.next_query is not None,
        )
        if step.coverage_missing is not None:
            coverage |= jnp.any(step.coverage_missing)
        value, marginal, valuation_coverage = self._value_reconstructed_action(
            step=step,
            effective_savings_grid=effective_savings_grid,
            next_liquid_grid=next_liquid_grid,
            next_value=next_value,
            next_marginal=next_marginal,
            next_query_data=next_query_data,
            prepared_child=prepared_child,
            preferences=preferences,
            beta=beta,
            params=params,
        )
        coverage |= valuation_coverage
        query_data = None
        if self.current_query is not None:
            if step.savings is None or step.corner_continuation is None:
                raise TypeError("A scalar floor producer lacks candidate provenance.")
            legal = liquid > effective_savings_grid[0]
            finite = (
                legal
                & jnp.isfinite(value)
                & jnp.isfinite(marginal)
                & jnp.isfinite(step.savings)
            )
            economic = legal & jnp.isneginf(value) & jnp.isfinite(step.savings)
            kinds = jnp.where(
                finite,
                QueryNodeKind.FINITE,
                jnp.where(
                    economic, QueryNodeKind.ECONOMIC_MINUS_INF, QueryNodeKind.PAD
                ),
            ).astype(jnp.int32)
            query_data = EGMQueryData(
                savings=step.savings,
                node_kind=kinds,
                link_valid=(kinds[:-1] != QueryNodeKind.PAD)
                & (kinds[1:] != QueryNodeKind.PAD),
                owner_id=jnp.zeros(liquid.shape, dtype=jnp.int32),
                side_id=jnp.zeros(liquid.shape, dtype=jnp.int32),
                floor_used=effective_savings_grid[0],
                corner_continuation=jnp.reshape(step.corner_continuation, (1,)),
                corner_kind=jnp.reshape(
                    jnp.where(
                        jnp.isfinite(step.corner_continuation),
                        QueryNodeKind.FINITE,
                        jnp.where(
                            jnp.isneginf(step.corner_continuation),
                            QueryNodeKind.ECONOMIC_MINUS_INF,
                            QueryNodeKind.PAD,
                        ),
                    ).astype(jnp.int32),
                    (1,),
                ),
                corner_interval_id=jnp.zeros((1,), dtype=jnp.int32),
                corner_side_id=jnp.zeros((1,), dtype=jnp.int32),
            )
        carry = EGMCarry(
            endog_grid=liquid,
            value=value,
            marginal_utility=marginal,
            taste_shock_scale=jnp.asarray(0.0, dtype=value.dtype),
            query_data=query_data,
        )
        if self.current_query is not None or self.next_query is not None:
            counts = count_current_feasibility(
                budget_feasible=(liquid > effective_savings_grid[0])[:, None],
                predicate_feasible=jnp.ones((liquid.size, 1, 0), dtype=jnp.bool_),
                value=value,
                active_states=jnp.ones(liquid.shape, dtype=jnp.bool_),
            )
            return BoundedResult(
                output=(value, carry, counts, coverage), bound_status=statuses
            )
        return value, carry

    def _value_reconstructed_action(
        self,
        *,
        step: RetiredEGMResult,
        effective_savings_grid: Float1D,
        next_liquid_grid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        next_query_data: EGMQueryData | None,
        prepared_child: PreparedScalarEGMQuery | None,
        preferences: Preferences,
        beta: ScalarFloat,
        params: EconFunctionKwargs,
    ) -> tuple[Float1D, Float1D, BoolND]:
        """Value strict interior actions against the installed child value reader."""
        if step.savings is None or step.rescore_interior is None:
            raise TypeError("A plain EGM producer lacks associated action provenance.")
        rescore = step.rescore_interior
        coverage = jnp.asarray(a=False)
        valuation_landing, _ = self.declared_law(
            savings_grid=jnp.where(rescore, step.savings, effective_savings_grid[0]),
            **params,
        )
        if self.next_query is not None:
            if next_query_data is None or prepared_child is None:
                raise TypeError("A scalar valuation lacks its prepared target inputs.")
            valuation = self.next_query.read_prepared(
                query=valuation_landing,
                grid=next_liquid_grid,
                value=next_value,
                marginal=next_marginal,
                data=next_query_data,
                prepared=prepared_child,
            )
            continuation_value = valuation.value
            coverage |= jnp.any(rescore & valuation.coverage_missing)
        else:
            continuation_value = interp_on_regular_grid(
                x_query=valuation_landing, xp=next_liquid_grid, fp=next_value
            )
        safe_consumption = jnp.where(rescore, step.consumption, 1.0)
        action_value = preferences.utility(safe_consumption) + zero_safe_weighted_term(
            weight=beta,
            value=jnp.where(rescore, continuation_value, 0.0),
            subnormal_is_accounted_for=False,
        )
        value = jnp.where(rescore, action_value, step.value)
        # Finite actions retain their economic envelope approximation. An impossible
        # continuation has the neutral marginal used by economic minus-infinity rows.
        marginal = jnp.where(rescore & jnp.isneginf(action_value), 0.0, step.marginal)
        return value, marginal, coverage


def guard_regime_mass(*, core: OneRowCore, enable_jit: bool) -> OneRowCore:
    """Wrap a one-row core in the lost-mass guard, compiled when JIT is enabled."""
    guarded = _RegimeMassGuardedCore(core=core)
    return jax.jit(guarded) if enable_jit else guarded


@dataclass(frozen=True, eq=False)
class _RegimeMassGuardedCore:
    """A one-row core that publishes NaN when its target loses regime mass.

    The one-row core reads its single target's carry at weight one. Where the
    regime sends mass to a target inactive next period, that weight is wrong and
    the core's finite output ignores the missing mass. The guard publishes NaN in
    the value and in both carry channels instead, as the grid-search and blended
    NB-EGM continuations do. The arithmetic states it, so no log level skips it.
    """

    core: OneRowCore
    """The one-row core whose output the guard poisons."""

    def __call__(
        self, *, retains_regime_mass: BoolND, **kwargs: PytreeValue
    ) -> OneRowOutput:
        """Run the core and poison its value and carry channels on lost mass."""
        evaluated = self.core(**kwargs)
        if type(evaluated) is BoundedResult:
            bounded = cast(
                "BoundedOneRowOutput",
                evaluated,
            )
            value, carry = bounded.output[:2]
        else:
            value, carry = cast("tuple[FloatND, EGMCarry]", evaluated)
        leaves = carry.leaves()
        guarded = (
            jnp.where(retains_regime_mass, value, jnp.nan),
            replace(
                carry,
                value=jnp.where(retains_regime_mass, leaves[("value",)], jnp.nan),
                marginal_utility=jnp.where(
                    retains_regime_mass, leaves[("marginal_utility",)], jnp.nan
                ),
            ),
        )
        return (
            replace(bounded, output=(*guarded, *bounded.output[2:]))
            if type(evaluated) is BoundedResult
            else guarded
        )


def _build_one_asset_carry_template(*, liquid_grid: Float1D) -> EGMCarry:
    """Build the all-finite 1-D EGM carry template on the liquid grid."""
    return EGMCarry(
        endog_grid=liquid_grid,
        value=jnp.zeros_like(liquid_grid),
        marginal_utility=jnp.zeros_like(liquid_grid),
        taste_shock_scale=jnp.asarray(0.0, dtype=liquid_grid.dtype),
    )
