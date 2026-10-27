"""Scope-checked assembly of the first native owned-resources projection graph."""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp

from _lcm.continuation import ProjectedProducerDate
from _lcm.current_iid import CurrentIIDPlan
from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.current_iid_native import NativeCaseRCore, NativeCaseROutput
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
from _lcm.egm.current_iid_resources_proof import prove_resources_only
from _lcm.egm.step import _EGMStep
from _lcm.egm.upper_envelope import get_upper_envelope
from _lcm.execution.core_program import CoreBuildContext, CoreExecutionRequirements
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
from _lcm.solution.contract import SolutionKernels, SolverBuildContext
from _lcm.solution.current_iid_case_r import (
    CASE_R_NODE_SELECTION,
    CASE_R_RESOURCES_REPLAY,
    CaseRArguments,
    NativeCaseRPeriodKernel,
    case_r_core_programs,
    case_r_native_inputs,
    install_balanced_case_r_date,
    native_case_r_date_template,
)
from _lcm.solution.dcegm import (
    EGMStepBuild,
    ExactEnvelope,
    _BoundDCEGM,
    _dcegm_query_slots_and_capabilities,
    _DCEGMArgumentBuilder,
    _DCEGMQuerySlots,
    _stochastic_node_axis,
)
from _lcm.solution.model_authority import _authority_from_template
from _lcm.typing import FlatParams, PytreeValue, ShapeDtypePytree
from lcm.exceptions import UnsupportedOperationError
from lcm.shocks import Discrete
from lcm.solver_api import (
    ArtifactChannel,
    ContinuationCapabilities,
    DeclaredReplay,
    PersistencePolicy,
)
from lcm.typing import FloatND, ReferenceName, RegimeName, ValueND


@dataclass(frozen=True, kw_only=True, eq=False)
class _PositionalResources:
    """Exact canonical callback and signature used by the structure certificate."""

    function: Callable[..., FloatND]
    names: tuple[ReferenceName, ...]

    def __call__(self, *values: ValueND) -> FloatND:
        return self.function(**dict(zip(self.names, values, strict=True)))


@dataclass(frozen=True, kw_only=True)
class _ResourcesStructureCheck:
    """Recheck the certified map on actual runtime parameter operand schemas."""

    function: _PositionalResources
    state: ReferenceName
    physical_dtypes: Mapping[ReferenceName, jnp.dtype]
    regime: RegimeName
    targets: tuple[RegimeName, ...]
    coefficient: Fraction
    dtype: jnp.dtype

    def __call__(self, *, flat_params: FlatParams) -> None:
        pool = union_free_params(
            flat_params=flat_params,
            regime_name=self.regime,
            transition_target_names=self.targets,
        )
        self.check_pool(pool=pool)

    def check_pool(
        self, *, pool: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]
    ) -> None:
        """Use the same certificate on a caller's actual consumed operand schemas."""
        arguments = []
        for name in self.function.names:
            dtype = self.physical_dtypes.get(name, self.dtype)
            if name == "period":
                argument = jax.ShapeDtypeStruct((), jnp.int32)
            elif name in pool:
                argument = _resources_parameter_argument(name=name, value=pool[name])
            else:
                argument = jax.ShapeDtypeStruct((), dtype)
            arguments.append(argument)
        proof = certify_resources_map(
            func=self.function,
            state_name=self.state,
            abstract_args=arguments,
            arg_names=self.function.names,
        )
        if proof.violation is not None or proof.coefficient != self.coefficient:
            raise UnsupportedOperationError(
                "Current-IID resources have no matching monotone affine structure "
                f"on these actual parameter operands: {proof.violation}."
            )


def _resources_parameter_argument(
    *, name: ReferenceName, value: PytreeValue | ShapeDtypePytree
) -> jax.ShapeDtypeStruct:
    """Require the numerical parameter schema the resources proof can trace."""
    if not isinstance(value, jax.Array):
        raise UnsupportedOperationError(
            f"Current-IID resources parameter '{name}' requires an array operand; "
            "grouped parameter leaves are not represented by this proof."
        )
    return jax.ShapeDtypeStruct(value.shape, value.dtype)


def _initial_resources_coefficient(
    *,
    numerical: _EGMStep,
    plan: CurrentIIDPlan,
    actual_pool: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree] | None = None,
) -> tuple[Fraction, _PositionalResources, Mapping[ReferenceName, jnp.dtype]]:
    function = _PositionalResources(
        function=numerical.pieces.own_resources_func,
        names=tuple(inspect.signature(numerical.pieces.own_resources_func).parameters),
    )
    dtype = jnp.dtype(canonical_float_dtype())
    physical_dtypes = {
        name: jnp.dtype(jnp.int32) if isinstance(law, Discrete) else dtype
        for _, binding, _ in plan.bindings
        for name, law in binding.group.shocks.items()
    }
    if actual_pool is None:
        arguments = tuple(
            jax.ShapeDtypeStruct(
                (),
                jnp.int32 if name == "period" else physical_dtypes.get(name, dtype),
            )
            for name in function.names
        )
    else:
        arguments_list = []
        for name in function.names:
            if name in actual_pool:
                argument = _resources_parameter_argument(
                    name=name, value=actual_pool[name]
                )
            elif name == numerical.pieces.euler_state_name or name in physical_dtypes:
                argument = jax.ShapeDtypeStruct((), physical_dtypes.get(name, dtype))
            else:
                raise UnsupportedOperationError(
                    f"Actual normalized resource operand {name!r} has no schema."
                )
            arguments_list.append(argument)
        arguments = tuple(arguments_list)
    proof = certify_resources_map(
        func=function,
        state_name=numerical.pieces.euler_state_name,
        abstract_args=arguments,
        arg_names=function.names,
    )
    if proof.violation is not None or proof.coefficient is None:
        raise UnsupportedOperationError(
            "The first native current-IID producer needs a structurally proved "
            f"positive monotone affine resources map: {proof.violation}."
        )
    return proof.coefficient, function, MappingProxyType(physical_dtypes)


def _admit_first_case_r(
    *,
    context: SolverBuildContext,
    solver: _BoundDCEGM,
    plan: CurrentIIDPlan,
    numerical: _EGMStep,
) -> None:
    """Retain complete dependency admission and explicit first-slice boundaries."""
    proof = prove_resources_only(context=context, solver=solver, plan=plan)
    reasons = list(proof.unsupported)
    if proof.bypasses:
        reasons.append(f"current innovation outside resources: {dict(proof.bypasses)}")
    if numerical.pieces.combo_names or context.has_taste_shocks:
        reasons.append("choice/passive-state aggregation for a projected row")
    if solver.continuous_state in proof.context_inputs:
        reasons.append("pre-state-dependent candidate context outside resources")
    if solver.resources in proof.root_dependencies["utility"]:
        reasons.append("resources-dependent local utility candidate representation")
    if numerical.pieces.lower_func is not None:
        reasons.append("callable borrowing-bound projected candidate representation")
    if not isinstance(solver.envelope, ExactEnvelope):
        reasons.append("owned candidate publication for the selected envelope family")
    if context.sharded_state_names or context.co_map_state_names:
        reasons.append("projected row sharding/co-map installation")
    if reasons:
        raise UnsupportedOperationError(
            "Native current-IID Case-R does not yet represent "
            + "; ".join(reasons)
            + "."
        )
    # Unbound native graphs still refuse publication. The model-wide read pass
    # may install only a closed population of exact scalar Euler demands.


@dataclass(frozen=True, kw_only=True)
class PreparedNativeCaseRDate:
    """The genuine admitted numerical core before final native programs."""

    date: ProjectedProducerDate
    core: NativeCaseRCore
    arguments: CaseRArguments
    template: NativeCaseROutput
    child_slots: _DCEGMQuerySlots
    capabilities: Mapping[RegimeName, ContinuationCapabilities] | None
    structure_check: _ResourcesStructureCheck


def prepare_native_case_r_date(
    *,
    context: SolverBuildContext,
    solver: _BoundDCEGM,
    build: EGMStepBuild,
    period: int,
    date: ProjectedProducerDate,
    call_context: CoreBuildContext | None = None,
    fixed_flat_params: FlatParams | None = None,
    materializer: DatedGenerationMaterializer | None = None,
) -> PreparedNativeCaseRDate:
    """Prepare the original core and actual template without a CoreProgram."""
    if date.producer != context.regime_name or date.authority.period != period:
        raise TypeError("Native preparation cannot borrow another producer date.")
    step = build.steps[period]
    plan = context.current_iid_plans.get(period)
    if plan is None:
        raise UnsupportedOperationError(
            "Mixed ordinary/current-IID producer dates are not installed."
        )
    numerical = cast("_EGMStep", step)
    _admit_first_case_r(context=context, solver=solver, plan=plan, numerical=numerical)
    envelope = solver.envelope
    if not isinstance(envelope, ExactEnvelope):
        raise UnsupportedOperationError(
            "Native Case-R requires the admitted owned ExactEnvelope."
        )
    actual_pool = None
    if call_context is not None:
        if fixed_flat_params is None:
            raise TypeError(
                "Dated resource preparation requires normalized fixed operands."
            )
        if call_context.period != period or call_context.ages is not context.ages:
            raise TypeError("Dated resource preparation lost its actual source clock.")
        base = _DCEGMArgumentBuilder(
            regime_name=context.regime_name,
            stateful_targets=frozenset(),
            transition_target_names=tuple(context.transitions),
        )
        fixed = union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=context.regime_name,
            transition_target_names=tuple(context.transitions),
        )
        flat = {
            **fixed,
            **(
                base(call_context)
                if materializer is None
                else base.prepare(context=call_context, materializer=materializer)
            ),
        }
        actual_pool = {
            name: value
            for name, value in flat.items()
            if name
            not in {numerical.pieces.euler_state_name, "next_regime_to_continuation"}
        }
    coefficient, function, dtypes = _initial_resources_coefficient(
        numerical=numerical, plan=plan, actual_pool=actual_pool
    )
    child_slots, caps = _dcegm_query_slots_and_capabilities(
        core=step,
        regime_name=context.regime_name,
        period=period,
        own_bound_specs=(),
    )
    child_slots = () if child_slots is None else child_slots
    core = NativeCaseRCore(
        pieces=replace(numerical.pieces, collect_continuation_read_counts=True),
        plan=plan,
        coefficient=coefficient,
        refine=get_upper_envelope(
            solver=solver,
            n_refined=numerical.pieces.n_pad,
            publish_provenance=True,
        ),
        owner_capacity=envelope.max_runs,
        projection_capacity=solver.current_iid_projection_capacity,
    )
    template = native_case_r_date_template(
        date=date,
        liquid_shape=context.state_action_space.states[solver.continuous_state].shape,
        resources_capacity=numerical.pieces.n_pad,
        child_slots=len(child_slots),
        current_nodes=plan.indices.shape[0],
        materializer=materializer,
    )
    arguments = CaseRArguments(
        base=_DCEGMArgumentBuilder(
            regime_name=context.regime_name,
            stateful_targets=frozenset(
                (
                    *core.pieces.continuation_plan.stateful_targets,
                    *core.pieces.continuation_plan.scalar_targets,
                )
            ),
            transition_target_names=tuple(context.transitions),
        ),
        plan=plan,
    )
    return PreparedNativeCaseRDate(
        date=date,
        core=core,
        arguments=arguments,
        template=template,
        child_slots=child_slots,
        capabilities=caps,
        structure_check=_ResourcesStructureCheck(
            function=function,
            state=solver.continuous_state,
            physical_dtypes=dtypes,
            regime=context.regime_name,
            targets=tuple(context.transitions),
            coefficient=coefficient,
            dtype=jnp.dtype(canonical_float_dtype()),
        ),
    )


def prepare_balanced_case_r_date(
    *,
    prepared: PreparedNativeCaseRDate,
    call_context: CoreBuildContext,
    fixed_flat_params: FlatParams,
    materializer: DatedGenerationMaterializer | None = None,
) -> PreparedNativeCaseRDate:
    """Install a real guarded date from actual normalized call-owned operands."""
    if call_context.period != prepared.date.authority.period:
        raise TypeError("Balanced call-owned preparation has another source date.")
    if call_context.ages is None:
        raise TypeError("Balanced preparation requires its actual canonical clock.")
    fixed = union_fixed_params(
        fixed_flat_params=fixed_flat_params,
        regime_name=prepared.date.producer,
        transition_target_names=prepared.arguments.base.transition_target_names,
    )
    arguments = {
        **fixed,
        **(
            prepared.arguments(call_context)
            if materializer is None
            else prepared.arguments.prepare(
                context=call_context, materializer=materializer
            )
        ),
    }
    native = case_r_native_inputs(
        core=prepared.core, arguments=arguments, materializer=materializer
    )
    missing = set(prepared.structure_check.function.names) - (
        set(native["pool"])
        | set(prepared.structure_check.physical_dtypes)
        | {prepared.structure_check.state}
    )
    if missing:
        raise UnsupportedOperationError(
            f"Actual normalized resource operands are missing: {sorted(missing)!r}."
        )
    prepared.structure_check.check_pool(pool=native["pool"])
    core, date, template = install_balanced_case_r_date(
        core=prepared.core,
        date=prepared.date,
        template=prepared.template,
        native_arguments=native,
        materializer=materializer,
    )
    return replace(prepared, core=core, date=date, template=template)


def _retain_original_node_selection(
    *,
    prepared_dates: Mapping[int, PreparedNativeCaseRDate] | None,
) -> bool:
    """Require consistent optional-bank coordinates across a regime's dates."""
    if prepared_dates is None:
        return True
    modes = {prepared.core.balanced is None for prepared in prepared_dates.values()}
    if len(modes) > 1:
        raise UnsupportedOperationError(
            "Date-varying original node-selection applicability is not installed."
        )
    return False not in modes


def build_native_case_r_kernels(
    *,
    context: SolverBuildContext,
    solver: _BoundDCEGM,
    build: EGMStepBuild,
    prepared_dates: Mapping[int, PreparedNativeCaseRDate] | None = None,
    materializer: DatedGenerationMaterializer | None = None,
) -> SolutionKernels:
    """Use the one actual producer declaration in both build and read-declare phases.

    This private assembly does not open the public current-IID admission guard.
    Public replay installation and a complete model-bound retention route are
    still required before this source prototype may be admitted for execution.
    """
    build.require_ordinary_publication()
    if materializer is not None and prepared_dates is None:
        raise UnsupportedOperationError(
            "Cold native kernel assembly requires its already-admitted dated banks."
        )
    spec = context.projected_continuation_declarations.get(context.regime_name)
    if spec is None:
        raise TypeError("Native Case-R lacks its actual shared producer declaration.")
    retain_node_selection = _retain_original_node_selection(
        prepared_dates=prepared_dates
    )
    programs = {}
    authorities = {}
    readers = {}
    checks = []
    replay_template = None
    selection_template = None
    replay_periods = {}
    for period, step in build.steps.items():
        preparation = (
            prepare_native_case_r_date(
                context=context,
                solver=solver,
                build=build,
                period=period,
                date=spec.date_at(period=period),
            )
            if prepared_dates is None
            else prepared_dates[period]
        )
        if preparation.date is not spec.date_at(period=period):
            raise TypeError("Native program lost its original dated declaration.")
        core = preparation.core
        template = preparation.template
        arguments = preparation.arguments
        child_slots = preparation.child_slots
        envelope = solver.envelope
        if not isinstance(envelope, ExactEnvelope):
            raise TypeError("Prepared Case-R lost its admitted owned envelope.")
        numerical = cast("_EGMStep", step)
        coefficient = preparation.structure_check.coefficient
        authorities[period] = (
            *child_slots,
            (context.regime_name, period, preparation.date.authority),
        )
        if preparation.capabilities is not None:
            readers[period] = preparation.capabilities
        replay_periods[period] = CaseRReplayPeriod(
            period=period,
            liquid_state=numerical.pieces.euler_state_name,
            action_name=numerical.pieces.action_name,
            resources=numerical.pieces.own_resources_func,
            utility=numerical.pieces.utility_func,
            coefficient=coefficient,
            owner_capacity=envelope.max_runs,
            row_reads=resources_row_reads(
                row=template.resources_row,
                key=CASE_R_RESOURCES_REPLAY,
                regime=context.regime_name,
                period=period,
                core=CASE_R_REPLAY_PROGRAM,
            ),
            row_tree=jax.tree.structure(template.resources_row),
            row_leaf_specs=resources_row_leaf_specs(row=template.resources_row),
        )
        if replay_template is not None and (
            jax.tree.structure(replay_template)
            != jax.tree.structure(template.resources_row)
            or tuple((x.shape, x.dtype) for x in jax.tree.leaves(replay_template))
            != tuple(
                (x.shape, x.dtype) for x in jax.tree.leaves(template.resources_row)
            )
        ):
            raise UnsupportedOperationError(
                "Date-varying retained resources-row schemas are not installed."
            )
        replay_template = template.resources_row
        if selection_template is not None and tuple(
            (leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(selection_template)
        ) != tuple(
            (leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(template.selection)
        ):
            raise UnsupportedOperationError(
                "Date-varying current-node selection schemas are not installed."
            )
        selection_template = template.selection if retain_node_selection else None
        graph = case_r_core_programs(
            materializer=materializer,
            core=core,
            arguments=arguments,
            template=template,
            requirements=CoreExecutionRequirements(
                reduced_axes=_stochastic_node_axis(
                    node_axes=build.stochastic_node_axes_by_period[period],
                    state_action_space=context.state_action_space,
                ),
            ),
            child_slots=len(child_slots),
        )
        programs[period] = NativeCaseRPeriodKernel(
            programs=graph,
            spec=spec,
            period=period,
            query_input_authorities=authorities[period],
            regime_name=context.regime_name,
            transition_target_names=tuple(context.transitions),
        )
        checks.append(preparation.structure_check)
    if replay_template is None:
        raise UnsupportedOperationError("Native Case-R has no actual producer date.")
    return SolutionKernels(
        case_r_replay_spec=CaseRReplaySpec(
            regime=context.regime_name,
            artifact_key=CASE_R_RESOURCES_REPLAY,
            periods=replay_periods,
        ),
        period_kernels=programs,
        period_group_keys={
            period: ("native_case_r", period, build.step_group_keys[period])
            for period in programs
        },
        continuation_spec=spec,
        continuation_reader_capabilities=readers or None,
        query_input_authorities=authorities,
        param_checks=tuple(checks),
        replay_route=DeclaredReplay.UNSUPPORTED,
        artifact_authorities={
            **(
                {
                    CASE_R_NODE_SELECTION: _authority_from_template(
                        key=CASE_R_NODE_SELECTION,
                        channel=ArtifactChannel.REPLAY,
                        persistence=PersistencePolicy.NOT_PERSISTED,
                        payload_runtime_type=tuple,
                        template=selection_template,
                        applicable=True,
                        required=False,
                        array_copier=None
                        if materializer is None
                        else materializer.copy_solution_leaf,
                    ),
                }
                if retain_node_selection
                else {}
            ),
            CASE_R_RESOURCES_REPLAY: _authority_from_template(
                key=CASE_R_RESOURCES_REPLAY,
                channel=ArtifactChannel.REPLAY,
                # The balanced dated producer owns the original row schema and
                # its CaseRReplaySpec. Exact-partition prototypes stay private.
                persistence=(
                    PersistencePolicy.NOT_PERSISTED
                    if retain_node_selection
                    else PersistencePolicy.MODEL_VERIFIABLE
                ),
                payload_runtime_type=tuple,
                template=resources_row_payload(row=replay_template),
                consumer_route=CASE_R_REPLAY_IDENTITY,
                applicable=True,
                required=True,
                array_copier=None
                if materializer is None
                else materializer.copy_solution_leaf,
            ),
        },
    )
