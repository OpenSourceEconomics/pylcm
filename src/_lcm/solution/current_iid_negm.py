"""Bind finite current-U NEGM to its genuine original nested numerical owner."""

import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from dags import concatenate_functions

from _lcm.certainty_equivalent import aggregates_nonlinearly
from _lcm.continuation import ContinuationPayload, ProjectedContinuationSpec
from _lcm.current_iid import CurrentIIDPlan
from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_case_u_replay import (
    CASE_U_FUTURE_REPLAY,
    CASE_U_REPLAY_IDENTITY,
    CaseUReplayFutureDate,
    CaseUReplayPeriod,
    CaseUReplaySpec,
)
from _lcm.egm.current_iid_negm import NEGMCaseUPoint
from _lcm.egm.current_iid_negm_origin import DatedNEGMCaseUSource
from _lcm.egm.current_iid_negm_program import NEGMCaseUArguments, NEGMCaseUProgram
from _lcm.egm.current_iid_nnbegm import NNBEGMCaseUPoint
from _lcm.egm.outer_inversion import certify_declared_outer_inverse
from _lcm.egm.preferences import concatenate_regime_function
from _lcm.egm.projected_continuation import PROJECTED_EGM_CONTINUATION
from _lcm.egm.step import _EGMStep
from _lcm.egm.upper_envelope import get_upper_envelope
from _lcm.egm.validation import _dag_ancestors, savings_stage_reads_euler_state
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    ProgramScope,
)
from _lcm.execution.output_layout import StateAxesLeading
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.regime_building.age_normalization import resolve_periodized_nodes
from _lcm.solution.continuation_reads import continuation_leaf_reads
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
from _lcm.solution.contract import ParamCheck, SolutionKernels, SolverBuildContext
from _lcm.solution.current_iid_case_r_build import (
    _initial_resources_coefficient,
    _ResourcesStructureCheck,
)
from _lcm.solution.current_iid_case_u_build import (
    materialize_case_u_schema,
    trace_case_u_shape,
)
from _lcm.solution.current_iid_case_u_native import (
    CaseUExactArguments,
    CaseUExactOutput,
    CaseURideExactFunction,
    _case_u_replay_program,
)
from _lcm.solution.current_iid_dated import DatedBalancedKernels
from _lcm.solution.dcegm import ExactEnvelope, _DCEGMPeriodKernel
from _lcm.solution.feasibility import BoundedResult, FeasibilityKernelOutput
from _lcm.solution.model_authority import _authority_from_template
from _lcm.solution.negm import (
    _BoundNEGM,
    _NEGMPeriodKernel,
    _with_no_adjustment_outer_function,
)
from _lcm.solution.nnbegm import _BoundNNBEGM
from _lcm.time import TimeAxis
from _lcm.typing import EconFunctionsMapping, FlatParams, PytreeValue
from lcm.exceptions import UnsupportedOperationError
from lcm.solver_api import ArtifactChannel, DeclaredReplay, PersistencePolicy
from lcm.typing import FloatND, RegimeName, StateName


@dataclass(frozen=True, kw_only=True, eq=False)
class PreparedNEGMCaseUPoint:
    """The genuine original source pieces and their original host checks."""

    point: NEGMCaseUPoint | NNBEGMCaseUPoint
    """Actual complete keeper and finite-outer native row selection."""
    param_checks: tuple[ParamCheck, ...]
    """Original inner checks followed by matching resource structure checks."""
    state_names: tuple[StateName, ...]
    """Actual source state order, independent of passive-leading carry order."""


def prepare_original_negm_u_point(
    *,
    context: SolverBuildContext,
    solver: _BoundNEGM,
    call: CoreBuildContext,
    fixed_flat_params: FlatParams,
    period: int,
) -> PreparedNEGMCaseUPoint:
    """Prepare original NEGM math while the enclosing source owns current draws.

    The original outer source authenticates each canonically derived keeper and
    adjuster context before their numerical preparation. Only the nested assembly
    entry suppresses recursive inner publication; all original physical inputs,
    law, current plan, CE, continuation and source declarations remain intact.
    """
    plan = context.current_iid_plans.get(period)
    if plan is None or call.period != period or call.ages is not context.ages:
        raise TypeError("Finite NEGM U requires its original dated source call.")
    _admit_finite_negm_u(context=context, solver=solver, plan=plan)
    space = call.state_action_space
    if space is None:
        raise TypeError("Finite NEGM U requires actual call-owned source support.")
    envelope = solver.inner.envelope
    if not isinstance(envelope, ExactEnvelope):
        raise UnsupportedOperationError("Finite NEGM U requires native exact rows.")
    # The original NEGM builder still constructs both original DCEGM numerical
    # pieces, keeper DAG, continuation plans, cost binding and outer-node order.
    source = DatedNEGMCaseUSource(
        context=context,
        solver=solver,
        period=period,
        date=context.projected_continuation_declarations[context.regime_name].date_at(
            period=period
        ),
        plan=plan,
    )
    solver.validate_build(context=context)
    built = solver.build_period_kernels(context=context, case_u_source=source)
    owner = built.period_kernels[period]
    if not isinstance(owner, _NEGMPeriodKernel):
        raise TypeError("Finite NEGM U lost its genuine original nested owner.")
    keeper_kernel, adjuster_kernel = owner.keeper_kernel, owner.adjuster_kernel
    if (
        not isinstance(keeper_kernel, _DCEGMPeriodKernel)
        or not isinstance(adjuster_kernel, _DCEGMPeriodKernel)
        or not isinstance(keeper_kernel.demand_step, _EGMStep)
        or not isinstance(adjuster_kernel.demand_step, _EGMStep)
    ):
        raise TypeError("Finite NEGM U lost original inner Euler constructors.")
    keeper = keeper_kernel.demand_step
    adjuster = adjuster_kernel.demand_step
    targets = tuple(context.transitions)
    source_grid = space.states[solver.inner.continuous_state]
    outer_grid = space.states[solver.outer_state]
    actual_pool: dict[str, PytreeValue] = {
        **union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=context.regime_name,
            transition_target_names=targets,
        ),
        **union_free_params(
            flat_params=MappingProxyType(call.flat_params),
            regime_name=context.regime_name,
            transition_target_names=targets,
        ),
        **{name: values[0] for name, values in space.states.items()},
        solver.outer_post_decision: owner.outer_grid_values[0],
        solver.outer_action: outer_grid[0],
        solver.inner.continuous_action: source_grid[0],
        "period": jnp.int32(period),
        "age": context.ages.values[period],
    }
    keeper_coefficient, keeper_function, keeper_dtypes = _initial_resources_coefficient(
        numerical=keeper, plan=plan, actual_pool=actual_pool
    )
    adjuster_coefficient, adjuster_function, adjuster_dtypes = (
        _initial_resources_coefficient(
            numerical=adjuster, plan=plan, actual_pool=actual_pool
        )
    )
    if keeper_coefficient != adjuster_coefficient:
        raise UnsupportedOperationError(
            "Finite NEGM U requires a shared positive affine liquid-resource "
            "Jacobian for its original common-state germ ordering."
        )
    functions = cast(
        "EconFunctionsMapping",
        resolve_periodized_nodes(mapping=context.functions, period=period),
    )
    keeper_functions = _with_no_adjustment_outer_function(
        functions=functions,
        durable_state=solver.outer_state,
        outer_post_decision=solver.outer_post_decision,
        no_adjustment_func=(
            None
            if solver.outer_no_adjustment_candidate is None
            else functions[solver.outer_no_adjustment_candidate]
        ),
    )
    outer_function = concatenate_functions(
        functions=functions,
        targets=[solver.outer_post_decision],
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )
    names = tuple(inspect.signature(outer_function).parameters)
    inverse = certify_declared_outer_inverse(
        func=outer_function,
        arg_names=names,
        abstract_args=tuple(
            jax.ShapeDtypeStruct(value.shape, value.dtype)
            if isinstance(value := actual_pool.get(name), jax.Array)
            else jax.ShapeDtypeStruct((), keeper_dtypes.get(name, source_grid.dtype))
            for name in names
        ),
        outer_action_name=solver.outer_action,
        outer_post_decision_name=solver.outer_post_decision,
        outer_state_domain=(float(outer_grid[0]), float(outer_grid[-1])),
        regime_name=context.regime_name,
    )
    point = NEGMCaseUPoint(
        context=context,
        plan=plan,
        period=period,
        owner=owner,
        keeper=keeper.pieces,
        adjuster=adjuster.pieces,
        refine=get_upper_envelope(
            solver=solver.inner,
            n_refined=keeper.pieces.n_pad,
            publish_provenance=True,
        ),
        source_grid=source_grid,
        savings_nodes=keeper.pieces.savings_nodes,
        borrowing_limit=keeper.pieces.borrowing_limit,
        outer_nodes=owner.outer_grid_values,
        coefficient=keeper_coefficient,
        outer_state=solver.outer_state,
        outer_action=solver.outer_action,
        keeper_target=concatenate_regime_function(
            functions=keeper_functions, target=solver.outer_post_decision
        ),
        outer_function=outer_function,
        outer_inverse=inverse,
        owner_capacity=envelope.max_runs,
    )
    dtype = source_grid.dtype
    return PreparedNEGMCaseUPoint(
        point=point,
        param_checks=(
            *built.param_checks,
            _ResourcesStructureCheck(
                function=keeper_function,
                state=solver.inner.continuous_state,
                physical_dtypes=keeper_dtypes,
                regime=context.regime_name,
                targets=targets,
                coefficient=keeper_coefficient,
                dtype=dtype,
            ),
            _ResourcesStructureCheck(
                function=adjuster_function,
                state=solver.inner.continuous_state,
                physical_dtypes=adjuster_dtypes,
                regime=context.regime_name,
                targets=targets,
                coefficient=adjuster_coefficient,
                dtype=dtype,
            ),
        ),
        state_names=tuple(space.states),
    )


def _admit_finite_negm_u(
    *, context: SolverBuildContext, solver: _BoundNEGM, plan: CurrentIIDPlan
) -> None:
    """Name the unimplemented intersections before building original branches."""
    user = context.user_regimes[context.regime_name]
    law = context.laws[context.regime_name]
    if (
        set(context.state_action_space.states)
        != {solver.inner.continuous_state, solver.outer_state}
        or context.state_action_space.discrete_actions
        or context.has_taste_shocks
        or context.sharded_state_names
        or context.co_map_state_names
        or context.action_partitions != 1
        or context.phase_variation_paths
        or aggregates_nonlinearly(context.certainty_equivalent)
        or context.arrival_queries
        or law.arrivals
        or law.gated_edges
        or user.exits
        or user.same_period_refs
        or user.value_constraints
        or solver.inner.accuracy == "certified"
        or (
            solver.inner.post_decision_bound is not None
            and solver.inner.post_decision_bound.kind != "literal"
        )
        or any(binding.state_inputs for _name, binding, _nodes in plan.bindings)
    ):
        raise UnsupportedOperationError(
            "Finite original NEGM U requires unsharded liquid/durable states, "
            "a literal floor, ordinary linear future reads and a current law "
            "independent of the pre-state. Adaptive/CE/arrival/query/moving-floor "
            "intersections are not installed."
        )
    if savings_stage_reads_euler_state(user_regime=user, law=law, solver=solver.inner):
        raise UnsupportedOperationError(
            "Finite NEGM U pre-liquid-dependent future branches are not installed."
        )
    # The original nested builder's inner functions preserve the actual physical
    # frame. Direct pre-liquid utility or discount channels need additional
    # envelope directions; do not silently replace them by the budget Jacobian.
    opaque = {
        name: function
        for name, function in context.solve_functions.items()
        if name != solver.inner.resources
    }
    roots = [context.solve_functions["utility"]]
    if "inverse_marginal_utility" in opaque:
        roots.append(opaque["inverse_marginal_utility"])
    if "discount_factor" in opaque:
        roots.append(opaque["discount_factor"])
    dependencies = tuple(
        _dag_ancestors(functions=opaque, target_func=function) for function in roots
    )
    if any(
        solver.inner.continuous_state in inputs or solver.inner.resources in inputs
        for inputs in dependencies
    ):
        raise UnsupportedOperationError(
            "Finite NEGM U direct-liquid/resource utility, inverse or discount "
            "directions are not installed."
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class NativeNEGMCaseUPeriodKernel:
    """Actual nested point, integrated publication and original future graph."""

    context: SolverBuildContext
    """Installed full original dated source context."""
    point: NEGMCaseUPoint | NNBEGMCaseUPoint
    """Genuine complete keeper/outer numerical constructor."""
    period: int
    """Original source date."""
    spec: ProjectedContinuationSpec
    """Installed exact projected continuation declaration."""
    program: CoreProgram
    """Original integrated values publication."""
    replay_program: CoreProgram
    """Same producer with the original future bundle returned."""
    template: BoundedResult[CaseUExactOutput]
    """Actual source schema whose projected buffers the spec borrows."""
    future: CaseUReplayFutureDate
    """Exact original future declaration used by physical reconstruction."""
    param_checks: tuple[ParamCheck, ...]
    """Identical original host validators, separate from the numerical callback."""

    def core_programs(self) -> Mapping[str, CoreProgram]:
        return MappingProxyType({"main": self.program, "replay": self.replay_program})

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> NativeNEGMCaseUPeriodKernel:
        arguments = self.program.argument_builder
        if not isinstance(arguments, CaseUExactArguments) or not isinstance(
            arguments.point, NEGMCaseUArguments
        ):
            raise TypeError("Finite NEGM U lost its original parameter projection.")
        arguments = replace(
            arguments,
            point=replace(arguments.point, fixed_flat_params=fixed_flat_params),
        )
        return replace(
            self,
            program=replace(self.program, argument_builder=arguments),
            replay_program=replace(self.replay_program, argument_builder=arguments),
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
    ) -> FeasibilityKernelOutput:
        del logger
        if period != self.period:
            raise TypeError("Finite NEGM U cannot use another original date.")
        key = "replay" if "replay" in compiled_cores else "main"
        call = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        output = compiled_cores[key](**self.core_programs()[key].argument_builder(call))
        if not isinstance(output, BoundedResult) or not isinstance(
            output.output, CaseUExactOutput
        ):
            raise TypeError("Finite NEGM U lost its actual native output channels.")
        result = output.output
        replay = {}
        if key == "replay":
            if result.future is None or result.future.child_names != tuple(
                self.future.specs
            ):
                raise TypeError(
                    "Finite NEGM U replay lost its actually produced original future."
                )
            replay[CASE_U_FUTURE_REPLAY] = result.future
        elif result.future is not None:
            raise TypeError("NEGM values-only program cannot publish replay inputs.")
        return FeasibilityKernelOutput(
            value=result.value,
            continuations={PROJECTED_EGM_CONTINUATION: result.projected},
            replay=replay,
            feasibility_counts=(
                jnp.sum(result.empty, dtype=jnp.int32),
                jnp.sum(jnp.isinf(result.value) & ~result.empty, dtype=jnp.int32),
            ),
            bound_status=output.bound_status,
            query_status=output.query_status,
            continuation_read_counts=output.continuation_read_counts,
            current_iid_invalid=result.current_iid_invalid,
        )


def prepare_finite_negm_u_kernels(  # noqa: C901, PLR0915 - genuine family, date, child and publication ownership
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    solvers: Mapping[RegimeName, _BoundNEGM | _BoundNNBEGM],
    declarations: Mapping[RegimeName, ProjectedContinuationSpec],
    ordinary_kernels: Mapping[RegimeName, SolutionKernels],
    call_contexts: Mapping[tuple[RegimeName, int], CoreBuildContext],
    fixed_flat_params: FlatParams,
    materializer: DatedGenerationMaterializer | None,
) -> DatedBalancedKernels:
    """Install the genuine finite original nested source and its physical replay."""
    if materializer is not None:
        raise UnsupportedOperationError(
            "Cold finite NEGM U source/terminal admission remains required."
        )
    if set(declarations) != set(solvers):
        raise UnsupportedOperationError(
            "Finite NEGM U mixed-family dated source preparation is not installed."
        )
    kernels = dict(ordinary_kernels)
    actual_contexts = dict(contexts)
    for name, solver in solvers.items():
        context = contexts[name]
        periods = context.regimes_to_active_periods[name]
        if len(periods) != 1:
            raise UnsupportedOperationError("Finite NEGM U requires one source date.")
        (period,) = periods
        targets = context.solution_reachability.targets(period=period, source=name)
        if (
            len(targets) != 1
            or targets[0] not in ordinary_kernels
            or not context.laws[targets[0]].terminal
        ):
            raise UnsupportedOperationError(
                "Finite NEGM U requires one actual ordinary terminal future."
            )
        (target,) = targets
        child_spec = ordinary_kernels[target].continuation_spec
        if child_spec is None:
            raise TypeError("Finite NEGM U lost its original child declaration.")
        child = child_spec.template_at(period=period + 1)
        if not isinstance(child, EGMCarry) or child.breakpoints is not None:
            raise UnsupportedOperationError(
                "Finite NEGM U requires its original ordinary non-kink child carry."
            )
        call = replace(
            call_contexts[name, period],
            next_regime_to_continuation=MappingProxyType({target: child}),
        )
        if isinstance(solver, _BoundNNBEGM):
            from _lcm.solution.current_iid_nnbegm import (  # noqa: PLC0415
                prepare_original_nnbegm_u_point,
            )

            prepared = prepare_original_nnbegm_u_point(
                context=context,
                solver=solver,
                call=call,
                fixed_flat_params=fixed_flat_params,
                period=period,
            )
        else:
            prepared = prepare_original_negm_u_point(
                context=context,
                solver=solver,
                call=call,
                fixed_flat_params=fixed_flat_params,
                period=period,
            )
        point = prepared.point
        if point.target != target:
            raise TypeError("Finite NEGM U lost its original nested future target.")
        future = CaseUReplayFutureDate(
            source=name, period=period, specs={target: child_spec}
        )
        arguments = CaseUExactArguments(
            point=NEGMCaseUArguments(
                pieces=point,
                regime_name=name,
                ages=context.ages,
                fixed_flat_params=fixed_flat_params,
            )
        )
        function = CaseURideExactFunction(
            point=NEGMCaseUProgram(pieces=point, plan=point.plan, period=period),
            capacity=declarations[name].date_at(period=period).authority.capacity,
            passive_names=point.passive_names,
            state_names=prepared.state_names,
        )
        template = materialize_case_u_schema(
            schema=trace_case_u_shape(function=function, arguments=arguments(call)),
            materializer=None,
        )
        if not isinstance(template, BoundedResult) or not isinstance(
            template.output, CaseUExactOutput
        ):
            raise TypeError("Finite NEGM U lost its genuine source output schema.")
        date = replace(
            declarations[name].date_at(period=period),
            template=template.output.projected,
        )
        spec = replace(
            declarations[name],
            template=date.template,
            templates_by_period={period: date.template},
            dates_by_period={period: date},
        )
        context = replace(
            context, projected_continuation_declarations={**declarations, name: spec}
        )
        point = replace(point, context=context)
        arguments = replace(arguments, point=replace(arguments.point, pieces=point))
        function = replace(function, point=replace(function.point, pieces=point))
        reads = continuation_leaf_reads(
            template=child,
            artifact_key=child.artifact_key,
            target=target,
            source_regime=name,
            source_period=period,
            core_key="main",
        )

        # keyword-only-exempt: library-callback=jax.tree_util.tree_map_with_path
        def role(
            path: tuple[jax.tree_util.KeyEntry, ...],
            leaf: jax.Array | jax.ShapeDtypeStruct,
            *,
            state_names: tuple[StateName, ...] = prepared.state_names,
        ) -> StateAxesLeading:
            names = tuple(getattr(key, "name", None) for key in path)
            return StateAxesLeading(
                state_names=state_names
                if names in (("output", "value"), ("output", "empty"))
                else (),
                shape=leaf.shape,
                dtype=leaf.dtype,
            )

        program = CoreProgram(
            name="main",
            function=function,
            argument_builder=arguments,
            output_roles=jax.tree_util.tree_map_with_path(role, template),
            requirements=CoreExecutionRequirements(value_reads=reads),
            disposition=CoreExecutionDisposition.DENSE,
            disposition_reason=(
                "Original finite NEGM keeper/outer native inner rows, complete "
                "physical-frame selection, then current-node expectation."
            ),
            scope=ProgramScope.VALUES_ONLY,
        )
        kernel = NativeNEGMCaseUPeriodKernel(
            context=context,
            point=point,
            period=period,
            spec=spec,
            program=program,
            replay_program=_case_u_replay_program(original=program, future=future),
            template=template,
            future=future,
            param_checks=prepared.param_checks,
        )
        replay = CaseUReplaySpec(
            regime=name,
            periods={
                period: CaseUReplayPeriod(
                    future=future,
                    date=date,
                    pieces=None,
                    negm_point=point,
                    fixed_flat_params=fixed_flat_params,
                    transition_target_names=tuple(context.transitions),
                    savings_point_width=None,
                    stochastic_node_width=None,
                    envelope_cell_width=1,
                )
            },
        )
        if template.query_status is None:
            raise TypeError("Finite NEGM U lost its original mandatory source status.")
        future_template = future.template()
        kernels[name] = SolutionKernels(
            period_kernels={period: kernel},
            period_group_keys={period: ("negm_u", period)},
            continuation_spec=spec,
            query_input_authorities={
                period: tuple(
                    (name, period, date.authority) for _ in template.query_status
                )
            },
            replay_route=DeclaredReplay.UNSUPPORTED,
            case_u_replay_spec=replay,
            param_checks=kernel.param_checks,
            artifact_authorities={
                CASE_U_FUTURE_REPLAY: _authority_from_template(
                    key=CASE_U_FUTURE_REPLAY,
                    channel=ArtifactChannel.REPLAY,
                    persistence=PersistencePolicy.MODEL_VERIFIABLE,
                    payload_runtime_type=type(future_template),
                    template=future_template,
                    consumer_route=CASE_U_REPLAY_IDENTITY,
                    applicable=True,
                    required=True,
                )
            },
        )
        actual_contexts[name] = context
    return DatedBalancedKernels(
        kernels=MappingProxyType(kernels),
        contexts=MappingProxyType(actual_contexts),
        steps=MappingProxyType({}),
        solvers=MappingProxyType({}),
    )
