"""Private original scalar NB current-U Model publication and reconstruction."""

import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.certainty_equivalent import PowerMean, aggregates_nonlinearly
from _lcm.continuation import (
    ContinuationPayload,
    ProjectedContinuationSpec,
)
from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_case_u_replay import (
    CASE_U_FUTURE_REPLAY,
    CASE_U_REPLAY_IDENTITY,
    CaseUReplayFutureDate,
    CaseUReplayPeriod,
    CaseUReplaySpec,
)
from _lcm.egm.current_iid_demands import ParentResourcesDemand
from _lcm.egm.current_iid_nbegm import (
    NBCaseUArguments,
    NBCaseUPoint,
    NBCaseUProgram,
    NBCaseURidePoint,
)
from _lcm.egm.current_iid_resources_proof import prove_resources_only
from _lcm.egm.nbegm import collect_nbegm_metadata
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
)
from _lcm.egm.validation_materialization import _validation_grid_first
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
from _lcm.solution.continuation_reads import continuation_leaf_reads
from _lcm.solution.continuation_target import union_fixed_params
from _lcm.solution.contract import ParamCheck, SolutionKernels, SolverBuildContext
from _lcm.solution.current_iid_case_u_build import (
    guard_case_u_shape,
    materialize_case_u_schema,
    trace_case_u_shape,
)
from _lcm.solution.current_iid_case_u_native import (
    CaseUExactArguments,
    CaseUExactFunction,
    CaseUExactOutput,
    CaseURideExactFunction,
    _case_u_replay_program,
)
from _lcm.solution.current_iid_dated import DatedBalancedKernels
from _lcm.solution.feasibility import BoundedResult, FeasibilityKernelOutput
from _lcm.solution.model_authority import _authority_from_template
from _lcm.solution.nbegm import (
    _BoundNBEGM,
    _build_nbegm_continuation_plan,
    _build_nbegm_single_axis_group,
    _NBEGMContinuousCore,
    _NBEGMTiledCore,
    _RideAlongNBEGMPeriodKernel,
    _solve_certified_continuous_budget,
)
from _lcm.time import TimeAxis
from _lcm.typing import FlatParams, PytreeValue
from lcm.exceptions import UnsupportedOperationError
from lcm.koopmans_aggregation import CESAggregator
from lcm.solver_api import ArtifactChannel, DeclaredReplay, PersistencePolicy
from lcm.typing import (
    FloatND,
    ReferenceName,
    RegimeName,
    ScalarFloat,
    ScalarInt,
    StateName,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class NativeNBCaseUPeriodKernel:
    """Actual NB original point, integrated publication and original future graph."""

    context: SolverBuildContext
    point: NBCaseUPoint | NBCaseURidePoint
    period: int
    spec: ProjectedContinuationSpec
    program: CoreProgram
    replay_program: CoreProgram
    template: BoundedResult[CaseUExactOutput]
    future: CaseUReplayFutureDate
    param_checks: tuple[ParamCheck, ...]
    """Identical original host validators, separate from the numerical callback."""

    def __post_init__(self) -> None:
        if isinstance(self.point, NBCaseURidePoint) and (
            self.point.certainty_equivalent is not None
        ):
            function = self.program.function
            if (
                type(self.point) is not NBCaseURidePoint
                or type(function) is not CaseURideExactFunction
                or type(function.point) is not NBCaseUProgram
                or function.point.pieces is not self.point
                or self.point.certainty_equivalent
                is not self.context.certainty_equivalent
                or self.point.koopmans_aggregator
                is not self.context.koopmans_aggregator
                or type(
                    self.context.user_regimes[
                        self.context.regime_name
                    ].get_koopmans_aggregator(phase="solve")
                )
                is not CESAggregator
            ):
                raise TypeError(
                    "NB recursive kernel changed its Model-owned declarations."
                )

    def core_programs(self) -> Mapping[str, CoreProgram]:
        return MappingProxyType({"main": self.program, "replay": self.replay_program})

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> NativeNBCaseUPeriodKernel:
        arguments = self.program.argument_builder
        if not isinstance(arguments, CaseUExactArguments) or not isinstance(
            arguments.point, NBCaseUArguments
        ):
            raise TypeError("NB U lost its original parameter projection.")
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
            raise TypeError("NB U cannot use another original date.")
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
            raise TypeError("NB U lost its actual native output channels.")
        result = output.output
        replay = {}
        if key == "replay":
            if result.future is None or result.future.child_names != tuple(
                self.future.specs
            ):
                raise TypeError(
                    "NB U replay lost its actually produced original future."
                )
            replay[CASE_U_FUTURE_REPLAY] = result.future
        elif result.future is not None:
            raise TypeError("NB values-only program cannot publish replay inputs.")
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


def prepare_scalar_nbegm_u_kernels(  # noqa: C901, PLR0912, PLR0915 - bounded original NB preparation and installation
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    solvers: Mapping[RegimeName, _BoundNBEGM],
    declarations: Mapping[RegimeName, ProjectedContinuationSpec],
    ordinary_kernels: Mapping[RegimeName, SolutionKernels],
    call_contexts: Mapping[tuple[RegimeName, int], CoreBuildContext],
    fixed_flat_params: FlatParams,
    materializer: DatedGenerationMaterializer | None,
    parent_demands: Mapping[
        tuple[RegimeName, int], tuple[ParentResourcesDemand, ...]
    ] = MappingProxyType({}),
) -> DatedBalancedKernels:
    """Build only original terminal-to-scalar-NB sources before Model installation."""
    if set(declarations) != set(solvers):
        raise UnsupportedOperationError(
            "Scalar NB U requires original NB source declarations only."
        )
    kernels = dict(ordinary_kernels)
    actual_contexts = dict(contexts)
    for name, solver in solvers.items():
        context = contexts[name]
        if materializer is not None and solver.envelope_arithmetic != "certified":
            raise UnsupportedOperationError(
                "Cold ordinary NB U requires admitted original "
                "backend-probe preparation."
            )
        periods = context.regimes_to_active_periods[name]
        if len(periods) != 1:
            raise UnsupportedOperationError(
                "Scalar NB U requires one original source date."
            )
        period = periods[0]
        targets = context.solution_reachability.targets(period=period, source=name)
        ride = tuple(context.state_action_space.states) != (solver.continuous_state,)
        recursive = aggregates_nonlinearly(context.certainty_equivalent)
        if recursive and (
            not ride
            or type(context.certainty_equivalent) is not PowerMean
            or type(context.user_regimes[name].get_koopmans_aggregator(phase="solve"))
            is not CESAggregator
            or context.koopmans_aggregator is None
        ):
            raise UnsupportedOperationError(
                "Recursive NB U requires its original ride PowerMean/CES declarations."
            )
        if (
            not targets
            or (not recursive and len(targets) != 1)
            or any(
                target not in ordinary_kernels or not context.laws[target].terminal
                for target in targets
            )
        ):
            raise UnsupportedOperationError(
                "Scalar NB U requires one original ordinary terminal child."
            )
        target = targets[0]
        plan = context.current_iid_plans.get(period)
        if plan is None:
            raise UnsupportedOperationError(
                "The NB U prototype requires its actual current shock plan."
            )
        if any(binding.state_inputs for _group, binding, _nodes in plan.bindings):
            raise UnsupportedOperationError(
                "NB U state-dependent current law derivatives are not installed."
            )
        if ride and materializer is not None:
            raise UnsupportedOperationError(
                "Cold NB ride producer admission remains required."
            )
        resources_proof = prove_resources_only(
            context=context, solver=solver, plan=plan
        )
        case_r = not ride and (
            not resources_proof.bypasses
            and solver.continuous_state not in resources_proof.context_inputs
        )
        if case_r:
            reasons = list(resources_proof.unsupported)
            if solver.budget_target in resources_proof.root_dependencies["utility"]:
                reasons.append(
                    "resource-dependent local utility candidate representation"
                )
            if reasons:
                raise UnsupportedOperationError(
                    "Original NB Case R does not yet represent "
                    + "; ".join(reasons)
                    + "."
                )
            if materializer is not None:
                raise UnsupportedOperationError(
                    "Cold original NB Case R bank admission is not installed."
                )
        if (
            (not ride and bool(context.state_action_space.discrete_actions))
            or context.has_taste_shocks
            or context.sharded_state_names
            or context.co_map_state_names
            or context.action_partitions != 1
            or context.phase_variation_paths
            or context.arrival_queries
            or context.laws[name].gated_edges
            or context.laws[name].arrivals
            or context.user_regimes[name].exits
            or context.user_regimes[name].same_period_refs
            or context.user_regimes[name].value_constraints
            or (
                solver.post_decision_bound is not None
                and solver.post_decision_bound.kind != "literal"
            )
        ):
            raise UnsupportedOperationError(
                "Scalar NB U requires its literal-floor unsharded "
                "additive schedule source."
            )
        registry = collect_nbegm_metadata(functions=context.solve_functions)
        if registry.piece_sets or any(
            source.variable != solver.continuous_state
            for source in registry.piecewise_affine_schedules
        ):
            raise UnsupportedOperationError(
                "Scalar NB U requires direct-liquid schedule declarations."
            )
        if any(
            source.kind != "continuous_kink"
            for schedule in registry.piecewise_affine_schedules
            for source in schedule.breakpoints
        ):
            raise UnsupportedOperationError(
                "NB U source-point ownership requires a continuous kink schedule."
            )
        proved = (
            ()
            if solver.post_decision_bound is None
            else solver.post_decision_bound.declaration_names
        )
        if any(
            constraint not in proved
            for constraint in context.user_regimes[name].constraints
        ):
            raise UnsupportedOperationError(
                "NB U nonbudget source feasibility rows require "
                "their original point mapping."
            )
        solver.validate_build(context=context)
        nodes = (
            solver.savings_grid.to_jax()
            if materializer is None
            else materializer.materialize_grid(spec=solver.savings_grid)
        )
        source_call = call_contexts[name, period]
        if source_call.state_action_space is None:
            raise TypeError("NB U lost its original call-owned source support.")
        if ride:
            point, checks = _prepare_nbegm_ride_point(
                context=context,
                solver=solver,
                source_call=source_call,
                period=period,
                recursive=recursive,
            )
        else:
            core, law, checks = _build_nbegm_single_axis_group(
                context=context,
                solver=solver,
                route="schedule",
                target=target,
                liquid_state_name=solver.continuous_state,
                variable_names=frozenset(context.state_action_space.state_names)
                | frozenset(context.state_action_space.continuous_actions),
                savings_grid=nodes,
                feasibility_constraints=(),
                materializer=materializer,
            )
            if type(core) is not _NBEGMContinuousCore:
                raise TypeError("NB U lost its original scalar schedule producer.")
            if case_r:
                from _lcm.solution.current_iid_nbegm_r import (  # noqa: PLC0415 - exact original R bank factory
                    prepare_scalar_nbegm_r_kernel,
                )

                child_spec = ordinary_kernels[target].continuation_spec
                if child_spec is None:
                    raise TypeError("NB R child lost its original continuation.")
                child = child_spec.template_at(period=period + 1)
                if (
                    not isinstance(child, EGMCarry)
                    or child.endog_grid.ndim != 1
                    or child.breakpoints is not None
                ):
                    raise UnsupportedOperationError(
                        "Scalar NB R requires an ordinary scalar terminal carry."
                    )
                actual_context, kernel = prepare_scalar_nbegm_r_kernel(
                    context=context,
                    solver=solver,
                    original_core=core,
                    declared_law=law,
                    continuation_plan=_build_nbegm_continuation_plan(
                        context=context,
                        period=period,
                        post_decision_name=solver.post_decision_function,
                    ),
                    checks=checks,
                    nodes=nodes,
                    period=period,
                    target=target,
                    plan=plan,
                    declaration=declarations[name],
                    child_spec=child_spec,
                    source_call=replace(
                        call_contexts[name, period],
                        next_regime_to_continuation=MappingProxyType({target: child}),
                    ),
                    fixed_flat_params=fixed_flat_params,
                )
                actual_contexts[name] = actual_context
                kernels[name] = kernel
                continue
            physical_names = {
                shock
                for _name, binding, _nodes in plan.bindings
                for shock in binding.group.shocks
            }
            if not physical_names.intersection(
                inspect.signature(core.schedule_spec.utility_dag).parameters
            ):
                raise UnsupportedOperationError(
                    "The original NB problem is not resources-only and its admitted "
                    "U class requires a current utility shock."
                )
            if (
                solver.continuous_state
                in inspect.signature(core.schedule_spec.utility_dag).parameters
            ):
                raise UnsupportedOperationError(
                    "Direct-liquid NB U utility envelope derivatives are not installed."
                )
            if core.schedule_spec.discount_factor_dag is not None:
                raise UnsupportedOperationError(
                    "NB U dynamic discount envelope derivatives are not installed."
                )
            source_grid = source_call.state_action_space.states[solver.continuous_state]
            if source_grid.dtype != canonical_float_dtype():
                raise TypeError("NB U requires its actual canonical source precision.")
            point = NBCaseUPoint(
                core=replace(
                    core,
                    retain_query=True,
                    retain_policy=True,
                    solve_budget=(
                        _solve_certified_continuous_budget
                        if materializer is not None
                        else core.solve_budget
                    ),
                ),
                declared_law=law,
                continuation_plan=_build_nbegm_continuation_plan(
                    context=context,
                    period=period,
                    post_decision_name=solver.post_decision_function,
                ),
                euler_state_name=solver.continuous_state,
                action_name=solver.continuous_action,
                target=target,
                savings_nodes=nodes,
                borrowing_limit=(
                    nodes[0]
                    if materializer is None
                    else materializer.evaluate_validation(
                        function=_validation_grid_first, arguments={"points": nodes}
                    )
                ),
                source_grid=source_grid,
            )
        child_specs = {}
        children = {}
        for target in targets:
            child_spec = ordinary_kernels[target].continuation_spec
            if child_spec is None:
                raise TypeError("NB U child did not publish its original continuation.")
            child = child_spec.template_at(period=period + 1)
            if (
                not isinstance(child, EGMCarry)
                or (not ride and child.endog_grid.ndim != 1)
                or child.breakpoints is not None
            ):
                raise UnsupportedOperationError(
                    "Scalar NB U requires an ordinary scalar terminal carry "
                    "without breakpoints."
                )
            child_specs[target] = child_spec
            children[target] = child
        future = CaseUReplayFutureDate(source=name, period=period, specs=child_specs)
        call = replace(
            call_contexts[name, period],
            next_regime_to_continuation=MappingProxyType(children),
        )
        demands = parent_demands.get((name, period), ())
        arguments = CaseUExactArguments(
            demands=demands,
            fixed_pools=tuple(
                union_fixed_params(
                    fixed_flat_params=fixed_flat_params,
                    regime_name=demand.source,
                    transition_target_names=demand.transition_targets,
                )
                for demand in demands
            ),
            point=NBCaseUArguments(
                pieces=point,
                regime_name=name,
                plan=plan,
                period=period,
                ages=context.ages,
                transition_target_names=tuple(context.transitions),
                fixed_flat_params=fixed_flat_params,
            ),
        )
        integrated = NBCaseUProgram(pieces=point, plan=plan, period=period)
        capacity = declarations[name].date_at(period=period).authority.capacity
        function = (
            CaseURideExactFunction(
                point=integrated,
                capacity=capacity,
                passive_names=point.passive_names,
                state_names=tuple(context.state_action_space.states),
                demands=demands,
            )
            if isinstance(point, NBCaseURidePoint)
            else CaseUExactFunction(
                point=integrated, capacity=capacity, demands=demands
            )
        )
        if materializer is not None:
            captures = guard_case_u_shape(
                function=function,
                context=call,
                fixed_flat_params=fixed_flat_params,
                materializer=materializer,
            )
            arguments = replace(arguments, trace_capture_roots=captures)
        actual_arguments = (
            arguments(call)
            if materializer is None
            else arguments.prepare(context=call, materializer=materializer)
        )
        if materializer is not None:
            clock_values: dict[ReferenceName, ScalarFloat | ScalarInt] = {}
            for clock_name in ("period", "age"):
                value = actual_arguments[clock_name]
                if not isinstance(value, jax.Array) or value.ndim != 0:
                    raise TypeError("Cold NB lost its admitted original scalar clock.")
                clock_values[clock_name] = value
            arguments = replace(
                arguments,
                point=replace(
                    arguments.point, clock_values=MappingProxyType(clock_values)
                ),
            )
        template = materialize_case_u_schema(
            schema=trace_case_u_shape(function=function, arguments=actual_arguments),
            materializer=materializer,
        )
        if not isinstance(template, BoundedResult) or not isinstance(
            template.output, CaseUExactOutput
        ):
            raise TypeError("NB U shape preparation lost its original native schema.")
        if demands:
            date = declarations[name].date_at(period=period)
            original = date.template
            actual = template.output.projected
            if jax.tree.structure(original) != jax.tree.structure(actual) or tuple(
                (leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(original)
            ) != tuple((leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(actual)):
                raise TypeError("ND parent closure changed its frozen producer schema.")
            template = replace(
                template, output=replace(template.output, projected=original)
            )
            spec = declarations[name]
        else:
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
        reads = tuple(
            read
            for target, child in children.items()
            for read in continuation_leaf_reads(
                template=child,
                artifact_key=child.artifact_key,
                target=target,
                source_regime=name,
                source_period=period,
                core_key="main",
            )
        )

        state_names = tuple(context.state_action_space.states)

        # keyword-only-exempt: library-callback=jax.tree_util.tree_map_with_path
        def role(
            path: tuple[jax.tree_util.KeyEntry, ...],
            leaf: jax.Array | jax.ShapeDtypeStruct,
            *,
            state_names: tuple[StateName, ...] = state_names,
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
                "Original NB passive-cell/discrete-branch/current-node/asset solve "
                "with complete transient branch-specific economic CE banks."
                if ride
                else "Original scalar NB liquid-grid/current-node/asset solve with "
                "a complete transient economic CE bank."
            ),
            scope=ProgramScope.VALUES_ONLY,
        )
        kernel = NativeNBCaseUPeriodKernel(
            context=context,
            point=point,
            period=period,
            spec=spec,
            program=program,
            replay_program=_case_u_replay_program(original=program, future=future),
            template=template,
            future=future,
            param_checks=checks,
        )
        replay = CaseUReplaySpec(
            regime=name,
            periods={
                period: CaseUReplayPeriod(
                    future=future,
                    date=date,
                    pieces=None,
                    nbegm_point=point,
                    fixed_flat_params=fixed_flat_params,
                    transition_target_names=tuple(context.transitions),
                    savings_point_width=None,
                    stochastic_node_width=None,
                    envelope_cell_width=1,
                )
            },
        )
        if template.query_status is None:
            raise TypeError("NB U template lost its original mandatory source status.")
        kernels[name] = SolutionKernels(
            period_kernels={period: kernel},
            period_group_keys={period: ("nb_case_u", period)},
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
                    payload_runtime_type=type(future.template()),
                    template=future.template(),
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


def _prepare_nbegm_ride_point(
    *,
    context: SolverBuildContext,
    solver: _BoundNBEGM,
    source_call: CoreBuildContext,
    period: int,
    recursive: bool = False,
) -> tuple[NBCaseURidePoint, tuple[ParamCheck, ...]]:
    """Reuse the actual ordinary ride builder and its branch-class continuation."""
    if source_call.state_action_space is None:
        raise TypeError("NB ride U lost its actual resolved source support.")
    original = solver.build_period_kernels(context=replace(context, enable_jit=False))
    kernel = original.period_kernels[period]
    if type(kernel) is not _RideAlongNBEGMPeriodKernel:
        raise UnsupportedOperationError(
            "NB passive U requires its genuine ride kernel."
        )
    core = kernel.core_programs()["main"].function
    if type(core) is not _NBEGMTiledCore:
        raise TypeError("NB ride U lost its actual numerical cell constructor.")
    plan = context.current_iid_plans.get(period)
    if plan is None:
        raise TypeError("NB ride U lost its actual current-IID declaration.")
    physical_names = {
        shock
        for _name, binding, _nodes in plan.bindings
        for shock in binding.group.shocks
    }
    if not physical_names.intersection(
        inspect.signature(core.schedule_spec.utility_dag).parameters
    ):
        raise UnsupportedOperationError(
            "The first NB passive U class requires a current utility shock; "
            "NB passive Case R remains required."
        )
    binder = core.envelope_build.bind_cell
    declared_ce = context.certainty_equivalent
    if recursive and type(declared_ce) is not PowerMean:
        raise TypeError("NB recursive point requires its actual PowerMean declaration.")
    if (
        core.statics.continuation_reads_liquid
        or core.statics.has_jump
        or core.cliff_candidates
        or core.statics.co_map_state_names
        or core.continuation_plan.query_targets
        or core.continuation_plan.scalar_targets
        or binder.is_epstein_zin != recursive
        or binder.current_floor is not None
        or binder.current_predicates is not None
        or binder.feasibility_constraints
        or binder.n_published_boundaries
        or core.schedule_spec.discount_factor_dag is not None
        or solver.continuous_state
        in inspect.signature(core.schedule_spec.utility_dag).parameters
    ):
        raise UnsupportedOperationError(
            "NB ride U requires continuous additive literal-floor ordinary-child "
            "cells without direct-liquid utility or interval-dependent future reads."
        )
    return NBCaseURidePoint(
        core=core,
        euler_state_name=solver.continuous_state,
        action_name=solver.continuous_action,
        savings_nodes=core.savings_grid,
        borrowing_limit=core.savings_grid[0],
        source_grid=source_call.state_action_space.states[solver.continuous_state],
        passive_names=tuple(
            name
            for name in source_call.state_action_space.states
            if name != solver.continuous_state
        ),
        certainty_equivalent=(
            declared_ce if isinstance(declared_ce, PowerMean) and recursive else None
        ),
        koopmans_aggregator=(context.koopmans_aggregator if recursive else None),
    ), original.param_checks
