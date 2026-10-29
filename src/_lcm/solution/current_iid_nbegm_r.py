"""Private scalar original NB resource-only Model publication and retained replay."""

import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.continuation import (
    ContinuationPayload,
    ContinuationSpec,
    ProjectedContinuationSpec,
)
from _lcm.current_iid import CurrentIIDPlan
from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import ContinuationPlan
from _lcm.egm.current_iid_nbegm_r import NBCaseRProgram
from _lcm.egm.current_iid_pullback import certify_resources_map
from _lcm.egm.current_iid_replay import (
    CASE_R_REPLAY_IDENTITY,
    CASE_R_REPLAY_PROGRAM,
    CaseRReplayPeriod,
    CaseRReplaySpec,
)
from _lcm.egm.current_iid_resources_artifact import (
    resources_row_leaf_specs,
    resources_row_payload,
    resources_row_reads,
)
from _lcm.egm.nbegm_resources import NBEGMResourcesBank
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    ProgramScope,
)
from _lcm.execution.output_layout import StateAxesLeading
from _lcm.solution.continuation_reads import continuation_leaf_reads
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
from _lcm.solution.contract import ParamCheck, SolutionKernels, SolverBuildContext
from _lcm.solution.current_iid_case_r import (
    CASE_R_RESOURCES_REPLAY,
    _zeros_for_template,
)
from _lcm.solution.current_iid_case_r_build import (
    _PositionalResources,
    _resources_parameter_argument,
    _ResourcesStructureCheck,
)
from _lcm.solution.feasibility import BoundedResult, FeasibilityKernelOutput
from _lcm.solution.model_authority import _authority_from_template
from _lcm.solution.nbegm import _BoundNBEGM, _NBEGMContinuousCore
from _lcm.time import TimeAxis
from _lcm.typing import FlatParams, PytreeValue
from lcm.exceptions import UnsupportedOperationError
from lcm.shocks import Discrete
from lcm.solver_api import ArtifactChannel, DeclaredReplay, PersistencePolicy
from lcm.typing import Float1D, FloatND, ReferenceName, RegimeName, StateName


@dataclass(frozen=True, kw_only=True, eq=False)
class NBCaseRArguments:
    """Project the genuine source support, original child, law and parameter pools."""

    source: RegimeName
    """Actual source owning this program."""
    target: RegimeName
    """Original ordinary terminal child."""
    state: StateName
    """Original pre-shock state name."""
    period: int
    """Actual source computational date."""
    ages: TimeAxis
    """Canonical source clock; a different wrapper cannot be substituted."""
    savings_nodes: Float1D
    """Original numeric savings support, including its literal floor."""
    plan: CurrentIIDPlan
    """Actual source group-law and joint-node declarations."""
    fixed_flat_params: FlatParams
    """Original fixed parameter namespaces."""
    transition_target_names: tuple[RegimeName, ...]
    """Complete target namespace roster for the original callbacks."""

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue]:
        if context.period != self.period or context.ages is not self.ages:
            raise TypeError("NB R lost its original source clock.")
        if context.state_action_space is None:
            raise TypeError("NB R lost its actual scalar source support.")
        return MappingProxyType(
            {
                "liquid_grid": context.state_action_space.states[self.state],
                "savings_nodes": self.savings_nodes,
                "next_regime_to_continuation": MappingProxyType(
                    {self.target: context.next_regime_to_continuation[self.target]}
                ),
                "group_params": {
                    name: {
                        **self.fixed_flat_params.get(name, {}),
                        **context.flat_params.get(name, {}),
                    }
                    for name, _binding, _nodes in self.plan.bindings
                },
                "node_operands": self.plan.node_operands,
                "indices": self.plan.indices,
                "pool": {
                    **union_fixed_params(
                        fixed_flat_params=self.fixed_flat_params,
                        regime_name=self.source,
                        transition_target_names=self.transition_target_names,
                    ),
                    **union_free_params(
                        flat_params=MappingProxyType(context.flat_params),
                        regime_name=self.source,
                        transition_target_names=self.transition_target_names,
                    ),
                    "period": jnp.int32(self.period),
                    "age": self.ages.values[self.period],
                },
            }
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class NativeNBCaseRPeriodKernel:
    """Publish the actual NB integrated row and its original retained candidate bank."""

    context: SolverBuildContext
    """Canonical source context sharing the final projected declaration."""
    period: int
    """Actual source date."""
    programs: Mapping[str, CoreProgram]
    """Original economic graph and optional same-bank replay output variant."""
    spec: ProjectedContinuationSpec
    """Exact installed declaration owning this producer's pre-shock channels."""
    projected_template: ProjectedEGMCarry
    """Original projected output, shared with the installed date without a copy."""

    def core_programs(self) -> Mapping[str, CoreProgram]:
        return self.programs

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> NativeNBCaseRPeriodKernel:
        programs = {}
        for name, program in self.programs.items():
            arguments = program.argument_builder
            if not isinstance(arguments, NBCaseRArguments):
                raise TypeError("NB R lost its actual original argument projection.")
            programs[name] = replace(
                program,
                argument_builder=replace(
                    arguments, fixed_flat_params=fixed_flat_params
                ),
            )
        return replace(self, programs=MappingProxyType(programs))

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
            raise TypeError("NB R cannot use another source date.")
        key = "replay" if "replay" in compiled_cores else "main"
        call = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        result = compiled_cores[key](**self.programs[key].argument_builder(call))
        if not isinstance(result, BoundedResult) or not isinstance(
            result.output, tuple
        ):
            raise TypeError("NB R lost its original output tuple.")
        value, projected, bank, counts, invalid, unsupported_economic = result.output
        if bool(unsupported_economic):
            raise UnsupportedOperationError(
                "Original NB Case R cannot represent feasible economic "
                "minus-infinity candidates; finite candidate values are required."
            )
        if not isinstance(projected, ProjectedEGMCarry):
            raise TypeError("NB R lost its pre-shock economic projection.")
        replay = {}
        if key == "replay":
            if type(bank) is not NBEGMResourcesBank:
                raise TypeError("NB R replay lost its original retained bank.")
            replay[CASE_R_RESOURCES_REPLAY] = resources_row_payload(row=bank)
        elif bank is not None:
            raise TypeError("NB R values-only execution cannot publish a replay bank.")
        return FeasibilityKernelOutput(
            value=value,
            continuations={PROJECTED_EGM_CONTINUATION: projected},
            replay=replay,
            feasibility_counts=counts,
            current_iid_invalid=invalid,
            bound_status=result.bound_status,
            query_status=result.query_status,
            continuation_read_counts=result.continuation_read_counts,
        )


def prepare_scalar_nbegm_r_kernel(  # noqa: C901 - exact original construction, declaration, and row custody stay together
    *,
    context: SolverBuildContext,
    solver: _BoundNBEGM,
    original_core: _NBEGMContinuousCore,
    declared_law: Callable[..., tuple[Float1D, Float1D]],
    continuation_plan: ContinuationPlan,
    checks: tuple[ParamCheck, ...],
    nodes: Float1D,
    period: int,
    target: RegimeName,
    plan: CurrentIIDPlan,
    declaration: ProjectedContinuationSpec,
    child_spec: ContinuationSpec,
    source_call: CoreBuildContext,
    fixed_flat_params: FlatParams,
) -> tuple[SolverBuildContext, SolutionKernels]:
    """Bind the authentic original scalar producer to the exact resource-bank route."""
    if (
        original_core.schedule_spec.breakpoint_kinds
        or original_core.schedule_spec.threshold_param_names
        or original_core.interval_recipe is not None
        or original_core.feasibility_constraints
        or original_core.schedule_spec.discount_factor_dag is not None
    ):
        raise UnsupportedOperationError(
            "Original NB Case R requires global affine resource topology."
        )
    if continuation_plan.query_targets:
        raise UnsupportedOperationError(
            "Original NB Case R requires ordinary typed future carries."
        )
    arguments = NBCaseRArguments(
        source=context.regime_name,
        target=target,
        state=solver.continuous_state,
        period=period,
        ages=context.ages,
        savings_nodes=nodes,
        plan=plan,
        fixed_flat_params=fixed_flat_params,
        transition_target_names=tuple(context.transitions),
    )
    actual_arguments = arguments(source_call)
    pool = actual_arguments["pool"]
    if not isinstance(pool, Mapping):
        raise TypeError("NB R lost its real original parameter pool.")
    resource_function = _PositionalResources(
        function=original_core.schedule_spec.coh_of_liquid_dag,
        names=tuple(
            inspect.signature(original_core.schedule_spec.coh_of_liquid_dag).parameters
        ),
    )
    dtype = jnp.dtype(canonical_float_dtype())
    physical_dtypes = MappingProxyType(
        {
            name: jnp.dtype(jnp.int32) if isinstance(law, Discrete) else dtype
            for _group, binding, _nodes in plan.bindings
            for name, law in binding.group.shocks.items()
        }
    )
    resource_arguments = []
    for name in resource_function.names:
        if name == solver.continuous_state or name in physical_dtypes:
            resource_arguments.append(
                jax.ShapeDtypeStruct((), physical_dtypes.get(name, dtype))
            )
        elif name in pool:
            resource_arguments.append(
                _resources_parameter_argument(name=name, value=pool[name])
            )
        else:
            raise UnsupportedOperationError(
                f"Original NB resource operand {name!r} has no actual schema."
            )
    certificate = certify_resources_map(
        func=resource_function,
        state_name=solver.continuous_state,
        abstract_args=resource_arguments,
        arg_names=resource_function.names,
    )
    coefficient = certificate.coefficient
    if certificate.violation is not None or coefficient is None:
        raise UnsupportedOperationError(
            "Original NB Case R needs its genuine positive affine resources proof: "
            f"{certificate.violation}."
        )
    structure_check = _ResourcesStructureCheck(
        function=resource_function,
        state=solver.continuous_state,
        physical_dtypes=physical_dtypes,
        regime=context.regime_name,
        targets=tuple(context.transitions),
        coefficient=coefficient,
        dtype=dtype,
    )
    function = NBCaseRProgram(
        core=original_core,
        declared_law=declared_law,
        continuation_plan=continuation_plan,
        plan=plan,
        euler_state_name=solver.continuous_state,
        action_name=solver.continuous_action,
        target=target,
        coefficient=coefficient,
        period=period,
        capacity=declaration.date_at(period=period).authority.capacity,
        retain_resources=True,
    )
    abstract = jax.eval_shape(function, **actual_arguments)
    template = _zeros_for_template(schema=abstract)
    if not isinstance(template, BoundedResult):
        raise TypeError("NB R lost its original output schema.")
    bank = template.output[2]
    projected = template.output[1]
    if type(bank) is not NBEGMResourcesBank or not isinstance(
        projected, ProjectedEGMCarry
    ):
        raise TypeError("NB R lost its actual bank or projected source schema.")
    name = context.regime_name
    date = replace(declaration.date_at(period=period), template=projected)
    spec = replace(
        declaration,
        template=projected,
        templates_by_period={period: projected},
        dates_by_period={period: date},
    )
    context = replace(
        context,
        projected_continuation_declarations={
            **context.projected_continuation_declarations,
            name: spec,
        },
    )
    child = child_spec.template_at(period=period + 1)
    if not isinstance(child, EGMCarry):
        raise TypeError("NB R requires its actual ordinary terminal child schema.")
    requirements = CoreExecutionRequirements(
        value_reads=continuation_leaf_reads(
            template=child,
            artifact_key=child.artifact_key,
            target=target,
            source_regime=name,
            source_period=period,
            core_key="main",
        )
    )
    programs = {}
    for retain in (False, True):
        key = "replay" if retain else "main"
        actual_function = replace(function, retain_resources=retain)
        actual_template = (
            template
            if retain
            else replace(
                template, output=(*template.output[:2], None, *template.output[3:])
            )
        )

        # keyword-only-exempt: library-callback=jax.tree_util.tree_map_with_path
        def role(
            path: tuple[jax.tree_util.KeyEntry, ...],
            leaf: jax.Array | jax.ShapeDtypeStruct,
        ) -> StateAxesLeading:
            return StateAxesLeading(
                state_names=(solver.continuous_state,)
                if tuple(
                    getattr(key, "name", getattr(key, "idx", None)) for key in path
                )
                == ("output", 0)
                else (),
                shape=leaf.shape,
                dtype=leaf.dtype,
            )

        programs[key] = CoreProgram(
            name=key,
            function=actual_function,
            argument_builder=arguments,
            output_roles=jax.tree_util.tree_map_with_path(role, actual_template),
            requirements=replace(
                requirements,
                value_reads=tuple(
                    replace(read, source=replace(read.source, core_key=key))
                    for read in requirements.value_reads
                ),
            ),
            disposition=CoreExecutionDisposition.DENSE,
            disposition_reason=(
                "Original scalar NB resource bank constructed once, with transient "
                "joint-node reads and pre-shock projection."
            ),
            scope=ProgramScope.REPLAY if retain else ProgramScope.VALUES_ONLY,
            retained_artifact_keys=(CASE_R_RESOURCES_REPLAY,) if retain else (),
            retained_artifact_payload_types={CASE_R_RESOURCES_REPLAY: tuple}
            if retain
            else {},
            replaces_program="main" if retain else None,
        )
    replay_period = CaseRReplayPeriod(
        period=period,
        liquid_state=solver.continuous_state,
        action_name=solver.continuous_action,
        resources=original_core.schedule_spec.coh_of_liquid_dag,
        utility=original_core.schedule_spec.utility_dag,
        coefficient=coefficient,
        owner_capacity=1,
        row_reads=resources_row_reads(
            row=bank,
            key=CASE_R_RESOURCES_REPLAY,
            period=period,
            regime=name,
            core=CASE_R_REPLAY_PROGRAM,
        ),
        row_tree=jax.tree.structure(bank),
        row_leaf_specs=resources_row_leaf_specs(row=bank),
    )
    kernel = NativeNBCaseRPeriodKernel(
        context=context,
        period=period,
        programs=MappingProxyType(programs),
        spec=spec,
        projected_template=projected,
    )
    return context, SolutionKernels(
        period_kernels={period: kernel},
        period_group_keys={period: ("nb_case_r", period)},
        continuation_spec=spec,
        query_input_authorities={period: ((name, period, date.authority),)},
        replay_route=DeclaredReplay.UNSUPPORTED,
        case_r_replay_spec=CaseRReplaySpec(
            regime=name,
            artifact_key=CASE_R_RESOURCES_REPLAY,
            periods={period: replay_period},
        ),
        param_checks=(*checks, structure_check),
        artifact_authorities={
            CASE_R_RESOURCES_REPLAY: _authority_from_template(
                key=CASE_R_RESOURCES_REPLAY,
                channel=ArtifactChannel.REPLAY,
                persistence=PersistencePolicy.MODEL_VERIFIABLE,
                payload_runtime_type=tuple,
                template=resources_row_payload(row=bank),
                consumer_route=CASE_R_REPLAY_IDENTITY,
                applicable=True,
                required=True,
            )
        },
    )
