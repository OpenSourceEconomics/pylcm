"""Private native graph assembly for the first owned-row Case-R producer.

The ordinary public non-GridSearch admission guard remains closed. This graph
is installed only by a future reviewed scope factory that supplies the actual
producer spec, structural certificate, templates and replay authority together.
"""

import functools
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp

from _lcm.continuation import (
    ContinuationPayload,
    ProjectedContinuationSpec,
    ProjectedProducerDate,
)
from _lcm.current_iid import (
    CURRENT_IID_ARGUMENT,
    CURRENT_IID_NODES_ARGUMENT,
    CURRENT_IID_ROWS_ARGUMENT,
    AdmittedGroupParams,
    CurrentIIDPlan,
    NodeOperands,
)
from _lcm.egm.current_iid_demands import ParentResourcesDemand
from _lcm.egm.current_iid_native import (
    NativeCaseRCore,
    NativeCaseRInputs,
    NativeCaseROutput,
)
from _lcm.egm.current_iid_resources_artifact import resources_row_payload
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.egm.step_core import OwnedResourcesRow
from _lcm.egm.validation_materialization import _validation_grid_first
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    ProgramScope,
    bind_core_program_fixed_params,
)
from _lcm.execution.output_layout import VALUE, OutputRoleTree, StateAxesLeading
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.dcegm import _DCEGMArgumentBuilder, _DCEGMQuerySlots
from _lcm.solution.feasibility import (
    BoundedResult,
    FeasibilityKernelOutput,
    QueryInputStatus,
    _unflatten_query_input_status,
)
from _lcm.solution.preparation_clock import prepare_case_u_clock
from _lcm.time import TimeAxis
from _lcm.typing import (
    EconFunctionArg,
    EconFunctionKwargs,
    FlatParams,
    PytreeValue,
    ShapeDtypePytree,
)
from lcm.solver_api import ArtifactKey
from lcm.typing import (
    FloatND,
    IntND,
    ReferenceName,
    RegimeName,
    ScalarFloat,
    ScalarInt,
)

CASE_R_RESOURCES_REPLAY = ArtifactKey(
    type_id="pylcm.egm.owned_resources_replay", schema_version=1
)
CASE_R_NODE_SELECTION = ArtifactKey(
    type_id="pylcm.egm.current_node_selection", schema_version=1
)


@dataclass(frozen=True, kw_only=True)
class CaseRArguments:
    """The actual ordinary DC operands plus explicit current-law operand banks."""

    base: _DCEGMArgumentBuilder
    """Original child-carry/parameter/time projection, shared by solve and lower."""
    plan: CurrentIIDPlan
    """Exact dated joint-node plan; fixed banks remain visible native inputs."""

    parent_demands: tuple[ParentResourcesDemand, ...] = ()
    """Authenticated consumers whose dynamic law operands this core prepares."""

    fixed_demand_pools: tuple[EconFunctionKwargs, ...] = ()
    """Fixed parent operands projected into native arguments, never hidden in code."""

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        arguments = dict(self.base(context))

        def clock(
            demand: ParentResourcesDemand,
        ) -> Mapping[ReferenceName, ScalarFloat | ScalarInt]:
            ages = cast("TimeAxis", context.ages)
            return {
                "period": jnp.int32(demand.period),
                "age": ages.values[demand.period],
            }

        return self._arguments(context=context, arguments=arguments, clock=clock)

    def prepare(
        self, *, context: CoreBuildContext, materializer: DatedGenerationMaterializer
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        """Admit source and parent clocks with original parameter/node assembly."""
        arguments = dict(self.base.prepare(context=context, materializer=materializer))
        if context.ages is None:
            raise TypeError("Parent demand arguments require the actual clock.")
        ages = context.ages
        return self._arguments(
            context=context,
            arguments=arguments,
            clock=lambda demand: prepare_case_u_clock(
                period=demand.period, ages=ages, materializer=materializer
            ),
        )

    def _arguments(
        self,
        *,
        context: CoreBuildContext,
        arguments: dict[ReferenceName, PytreeValue | ShapeDtypePytree],
        clock: Callable[
            [ParentResourcesDemand],
            Mapping[ReferenceName, ScalarFloat | ScalarInt],
        ],
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        """Share the unchanged current-node and incoming-demand operand mapping."""
        arguments[CURRENT_IID_ARGUMENT] = {
            name: dict(context.flat_params.get(name, {}))
            for name, _, _ in self.plan.bindings
        }
        arguments[CURRENT_IID_NODES_ARGUMENT] = self.plan.node_operands
        arguments[CURRENT_IID_ROWS_ARGUMENT] = self.plan.indices
        if self.parent_demands:
            if context.ages is None:
                raise TypeError("Parent demand arguments require the actual clock.")
            fixed = self.fixed_demand_pools or tuple({} for _ in self.parent_demands)
            arguments["_lcm_parent_demand_pools"] = tuple(
                {
                    **fixed_pool,
                    **union_free_params(
                        flat_params=MappingProxyType(context.flat_params),
                        regime_name=demand.source,
                        transition_target_names=demand.transition_targets,
                    ),
                    **clock(demand),
                }
                for demand, fixed_pool in zip(self.parent_demands, fixed, strict=True)
            )
            arguments["_lcm_parent_demand_savings"] = tuple(
                demand.pieces.savings_nodes for demand in self.parent_demands
            )
        return MappingProxyType(arguments)


@dataclass(frozen=True, kw_only=True, eq=False)
class CaseRProgramFunction:
    """Output-specialized callable with ordinary flat parameter binding semantics."""

    core: NativeCaseRCore
    """Actual canonical candidate producer and status-aware node expectation."""
    retain_resources: bool
    """Whether this graph variant publishes the optional original replay row."""

    def __call__(
        self,
        *,
        next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
        _lcm_current_iid: AdmittedGroupParams,
        _lcm_current_iid_nodes: NodeOperands,
        _lcm_current_iid_rows: IntND,
        _lcm_parent_demand_pools: tuple[EconFunctionKwargs, ...] = (),
        _lcm_parent_demand_savings: tuple[FloatND, ...] = (),
        _lcm_savings_point_width: int | None = None,
        _lcm_stochastic_node_width: int | None = None,
        _lcm_envelope_cell_width: int = 1,
        **kwargs: EconFunctionArg,
    ) -> BoundedResult[tuple[PytreeValue, ...]]:
        """Keep every current/future/law input in the actual native argument tree."""
        arguments = {
            **kwargs,
            "next_regime_to_continuation": next_regime_to_continuation,
            "_lcm_current_iid": _lcm_current_iid,
            "_lcm_current_iid_nodes": _lcm_current_iid_nodes,
            "_lcm_current_iid_rows": _lcm_current_iid_rows,
            "_lcm_parent_demand_pools": _lcm_parent_demand_pools,
            "_lcm_parent_demand_savings": _lcm_parent_demand_savings,
            "_lcm_savings_point_width": _lcm_savings_point_width,
            "_lcm_stochastic_node_width": _lcm_stochastic_node_width,
            "_lcm_envelope_cell_width": _lcm_envelope_cell_width,
        }
        result = self.core(**case_r_native_inputs(core=self.core, arguments=arguments))
        row = result.resources_row
        inactive = jnp.zeros((), dtype=jnp.bool_)
        return BoundedResult(
            output=(
                result.value,
                result.projected,
                resources_row_payload(row=row) if self.retain_resources else None,
                (result.empty_count, result.other_inf_count),
                result.current_iid_invalid,
                result.selection
                if self.retain_resources and self.core.balanced is None
                else None,
            ),
            bound_status=(),
            query_status=(
                *(row.child_status or ()),
                QueryInputStatus(inactive, inactive, ~result.valid),
            ),
            continuation_read_counts=row.continuation_read_counts,
        )


def case_r_native_inputs(
    *,
    core: NativeCaseRCore,
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
    materializer: DatedGenerationMaterializer | None = None,
) -> NativeCaseRInputs:
    """The original runtime projection, shared by actual shape preparation.

    The canonical argument builder owns this keyed tree. Casts only describe
    its existing typed slots; no numeric value or nested parameter is coerced.
    """
    from _lcm.solution.dcegm import _carry_subset  # noqa: PLC0415

    special = {
        "next_regime_to_continuation",
        "_lcm_current_iid",
        "_lcm_current_iid_nodes",
        "_lcm_current_iid_rows",
        "_lcm_parent_demand_pools",
        "_lcm_parent_demand_savings",
        "_lcm_savings_point_width",
        "_lcm_stochastic_node_width",
        "_lcm_envelope_cell_width",
    }
    state = core.pieces.euler_state_name
    liquid = cast("FloatND", arguments[state])
    pool = {
        name: cast("EconFunctionArg", value)
        for name, value in arguments.items()
        if name not in special and name != state
    }
    continuation = cast(
        "MappingProxyType[RegimeName, ContinuationPayload]",
        arguments["next_regime_to_continuation"],
    )
    return NativeCaseRInputs(
        liquid_grid=liquid,
        pool=pool,
        next_regime_to_continuation=_carry_subset(
            next_regime_to_continuation=continuation,
            stateful_targets=frozenset(continuation),
        ),
        law_arguments={
            **pool,
            state: (
                liquid[0]
                if materializer is None
                else materializer.evaluate_validation(
                    function=_validation_grid_first, arguments={"points": liquid}
                )
            ),
        },
        group_params=cast("AdmittedGroupParams", arguments["_lcm_current_iid"]),
        node_operands=cast("NodeOperands", arguments["_lcm_current_iid_nodes"]),
        indices=cast("IntND", arguments["_lcm_current_iid_rows"]),
        savings_point_width=cast(
            "int | None", arguments.get("_lcm_savings_point_width")
        ),
        stochastic_node_width=cast(
            "int | None", arguments.get("_lcm_stochastic_node_width")
        ),
        envelope_cell_width=cast("int", arguments.get("_lcm_envelope_cell_width", 1)),
        demand_pools=cast(
            "tuple[EconFunctionKwargs, ...]",
            arguments.get("_lcm_parent_demand_pools", ()),
        ),
        demand_savings=cast(
            "tuple[FloatND, ...]", arguments.get("_lcm_parent_demand_savings", ())
        ),
    )


def case_r_output_roles(
    *,
    template: NativeCaseROutput,
    retain_resources: bool,
    child_slots: int,
    retain_node_selection: bool = True,
    materializer: DatedGenerationMaterializer | None = None,
) -> OutputRoleTree:
    """Declare every actual output shape/dtype without adding a current-node axis."""

    def replicated(leaf: jax.Array | jax.ShapeDtypeStruct) -> StateAxesLeading:
        return StateAxesLeading(state_names=(), dtype=leaf.dtype, shape=leaf.shape)

    inactive = (
        jnp.zeros((), dtype=jnp.bool_)
        if materializer is None
        else jax.ShapeDtypeStruct((), jnp.bool_, weak_type=False)
    )
    payload = (
        template.value,
        template.projected,
        resources_row_payload(row=template.resources_row) if retain_resources else None,
        (template.empty_count, template.other_inf_count),
        template.current_iid_invalid,
        template.selection if retain_resources and retain_node_selection else None,
    )
    roles = jax.tree.map(replicated, payload)
    return BoundedResult(
        output=(VALUE, *roles[1:]),
        bound_status=(),
        query_status=tuple(
            jax.tree.map(
                replicated,
                QueryInputStatus(inactive, inactive, inactive)
                if not isinstance(inactive, jax.ShapeDtypeStruct)
                else _unflatten_query_input_status(
                    None, (inactive, inactive, inactive)
                ),
            )
            for _ in range(child_slots + 1)
        ),
        continuation_read_counts=jax.tree.map(
            replicated, template.resources_row.continuation_read_counts
        ),
    )


def case_r_core_programs(
    *,
    core: NativeCaseRCore,
    arguments: CaseRArguments,
    template: NativeCaseROutput,
    requirements: CoreExecutionRequirements,
    child_slots: int,
    materializer: DatedGenerationMaterializer | None = None,
) -> Mapping[str, CoreProgram]:
    """Assemble native variants with original reads and explicit dense execution.

    The first current-node projection is a fixed-capacity device computation.
    It makes no claim of streamed current-node or adaptive-host execution.
    The caller supplies actual child-leaf reads and the existing future/node
    widths under which the canonical candidate body was built.
    """
    programs = {}
    for retain in (False, True):
        # Replay consumes the original row. Balanced projection has no retained
        # coordinates for the optional exact-partition node-selection bank.
        name = "replay" if retain else "main"
        programs[name] = CoreProgram(
            name=name,
            function=CaseRProgramFunction(core=core, retain_resources=retain),
            argument_builder=arguments,
            requirements=requirements,
            output_roles=case_r_output_roles(
                template=template,
                retain_resources=retain,
                retain_node_selection=core.balanced is None,
                child_slots=child_slots,
                materializer=materializer,
            ),
            disposition=CoreExecutionDisposition.DENSE,
            disposition_reason=(
                "One owned resources row and a bounded device projection; "
                "current quadrature rows are temporary dense operands."
            ),
            donation_candidates=(),
            scope=ProgramScope.REPLAY if retain else ProgramScope.VALUES_ONLY,
            retained_artifact_keys=(
                (CASE_R_RESOURCES_REPLAY, CASE_R_NODE_SELECTION)
                if core.balanced is None
                else (CASE_R_RESOURCES_REPLAY,)
            )
            if retain
            else (),
            retained_artifact_payload_types=(
                {
                    CASE_R_RESOURCES_REPLAY: tuple,
                    **({CASE_R_NODE_SELECTION: tuple} if core.balanced is None else {}),
                }
                if retain
                else {}
            ),
            replaces_program="main" if retain else None,
        )
    return MappingProxyType(programs)


@dataclass(frozen=True, kw_only=True)
class NativeCaseRPeriodKernel:
    """Status-aware host envelope around the declared native producer graph."""

    programs: Mapping[str, CoreProgram]
    """Exact native output variants and their real operand/read declarations."""
    spec: ProjectedContinuationSpec
    """Actual date-owned projection schema and physical liquid coordinate."""
    period: int
    """Canonical producer date, held separately from dynamic numerical age."""
    query_input_authorities: _DCEGMQuerySlots
    """Actual child slots followed by this producer's projection integrity slot."""
    regime_name: RegimeName
    """Owner of parameter and artifact publication."""
    transition_target_names: tuple[RegimeName, ...]
    """Original future target names whose parameters the canonical producer reads."""

    def __post_init__(self) -> None:
        authority = self.spec.authority_at(period=self.period)
        own = self.query_input_authorities[-1]
        if own[2] is not authority or own != (
            self.regime_name,
            self.period,
            authority,
        ):
            raise TypeError("Case-R own status lacks its actual producer authority.")
        object.__setattr__(self, "programs", MappingProxyType(dict(self.programs)))

    def core_programs(self) -> Mapping[str, CoreProgram]:
        return self.programs

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> NativeCaseRPeriodKernel:
        fixed = union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=self.regime_name,
            transition_target_names=self.transition_target_names,
        )
        programs = {}
        for name, program in self.programs.items():
            arguments = program.argument_builder
            if not isinstance(arguments, CaseRArguments):
                raise TypeError("Case-R fixed binding lost its original arguments.")
            pools = tuple(
                union_fixed_params(
                    fixed_flat_params=fixed_flat_params,
                    regime_name=demand.source,
                    transition_target_names=demand.transition_targets,
                )
                for demand in arguments.parent_demands
            )
            programs[name] = bind_core_program_fixed_params(
                program=replace(
                    program,
                    argument_builder=replace(arguments, fixed_demand_pools=pools),
                ),
                params=fixed,
            )
        return replace(
            self,
            programs=programs,
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
        """Publish original buffers and mandatory integrity for engine consumption."""
        del logger
        if period != self.period:
            raise TypeError("Case-R kernel cannot borrow another producer date.")
        name = "replay" if "replay" in compiled_cores else "main"
        arguments = self.programs[name].argument_builder(
            CoreBuildContext(
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                next_regime_to_continuation=next_regime_to_continuation,
                flat_params=flat_params,
                period=period,
                ages=ages,
            )
        )
        evaluated = compiled_cores[name](**arguments)
        if not isinstance(evaluated, BoundedResult):
            raise TypeError("Case-R native output lacks mandatory integrity.")
        value, projected, resources, counts, current_iid_invalid, selection = (
            evaluated.output
        )
        return FeasibilityKernelOutput(
            value=value,
            continuations={PROJECTED_EGM_CONTINUATION: projected},
            replay={}
            if resources is None
            else {
                CASE_R_RESOURCES_REPLAY: resources,
                **({CASE_R_NODE_SELECTION: selection} if selection is not None else {}),
            },
            query_status=evaluated.query_status,
            continuation_read_counts=evaluated.continuation_read_counts,
            feasibility_counts=counts,
            current_iid_invalid=current_iid_invalid,
        )


def native_case_r_template(
    *,
    spec: ProjectedContinuationSpec,
    period: int,
    liquid_shape: tuple[int, ...],
    resources_capacity: int,
    child_slots: int,
    current_nodes: int,
) -> NativeCaseROutput:
    """Declare finite exact-shape native buffers without running user economics."""

    return native_case_r_date_template(
        date=spec.date_at(period=period),
        liquid_shape=liquid_shape,
        resources_capacity=resources_capacity,
        child_slots=child_slots,
        current_nodes=current_nodes,
    )


def native_case_r_date_template(
    *,
    date: ProjectedProducerDate,
    liquid_shape: tuple[int, ...],
    resources_capacity: int,
    child_slots: int,
    current_nodes: int,
    materializer: DatedGenerationMaterializer | None = None,
) -> NativeCaseROutput:
    """Use a genuine prepared date without requiring its multi-date aggregate."""
    projected = date.template
    if not isinstance(projected, ProjectedEGMCarry):
        raise TypeError("Case-R producer template is not its projected declaration.")
    factory = functools.partial(
        _native_case_r_array_template,
        liquid_shape=liquid_shape,
        resources_capacity=resources_capacity,
        child_slots=child_slots,
        current_nodes=current_nodes,
    )
    if materializer is None:
        return factory(projected=projected)
    result = materializer.produce(function=factory, arguments={"projected": projected})
    # Producer/date identity is a Python authority, not a reconstructed output
    # PyTree wrapper. Its exact original leaves remain explicit input operands.
    return result._replace(projected=projected)


def _native_case_r_array_template(
    *,
    projected: ProjectedEGMCarry,
    liquid_shape: tuple[int, ...],
    resources_capacity: int,
    child_slots: int,
    current_nodes: int,
) -> NativeCaseROutput:
    """Original finite template arithmetic, with all numeric operands explicit."""
    from _lcm.egm.upper_envelope.segment_envelope import OwnedEnvelope  # noqa: PLC0415
    from _lcm.solution.feasibility import ContinuationReadCounts  # noqa: PLC0415

    dtype = projected.grid.dtype
    grid = jnp.arange(resources_capacity, dtype=dtype)
    floating = jnp.zeros((resources_capacity,), dtype=dtype)
    indices = jnp.zeros((resources_capacity,), dtype=jnp.int32)
    inactive = jnp.zeros((), dtype=jnp.bool_)
    scalar = jnp.zeros((), dtype=dtype)
    envelope = OwnedEnvelope(
        grid=grid,
        policy=jnp.ones_like(grid),
        value=floating,
        n_kept=jnp.asarray(resources_capacity, dtype=jnp.int32),
        savings=floating,
        owner_id=indices,
        side_id=indices,
        link_valid=jnp.ones((resources_capacity - 1,), dtype=jnp.bool_),
        source_left_index=indices,
        source_right_index=indices,
    )
    row = OwnedResourcesRow(
        envelope=envelope,
        marginal=jnp.ones_like(grid),
        node_kind=jnp.zeros((resources_capacity,), dtype=jnp.int32),
        borrowing_limit=scalar,
        upper_savings=scalar,
        expected_floor_value=scalar,
        discount_factor=scalar,
        discounted_floor_value=scalar,
        resources_domain=jnp.zeros((2,), dtype=dtype),
        construction_invalid=inactive,
        floor_status=None,
        child_status=tuple(
            QueryInputStatus(inactive, inactive, inactive) for _ in range(child_slots)
        )
        if child_slots
        else None,
        continuation_read_counts=ContinuationReadCounts(
            jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.int32)
        ),
    )
    from _lcm.egm.current_iid_resource_query import OwnedResourcesRead  # noqa: PLC0415

    selection_shape = (projected.grid.size, current_nodes)
    selected_scalar = OwnedResourcesRead(
        value=scalar,
        marginal=scalar,
        policy=scalar,
        savings=scalar,
        native_owner=jnp.zeros((), dtype=jnp.int32),
        side_id=jnp.zeros((), dtype=jnp.int32),
        corner=inactive,
        feasible=inactive,
        valid=inactive,
    )
    return NativeCaseROutput(
        value=jnp.zeros(liquid_shape, dtype=dtype),
        projected=projected,
        resources_row=row,
        valid=~inactive,
        current_iid_invalid=inactive,
        empty_count=jnp.zeros((), dtype=jnp.int32),
        other_inf_count=jnp.zeros((), dtype=jnp.int32),
        selection=(
            jax.tree.map(
                lambda leaf: jnp.zeros(selection_shape, dtype=leaf.dtype),
                selected_scalar,
            ),
            jnp.zeros((current_nodes,), dtype=jnp.bool_),
        ),
    )


def case_r_kernel_with_declared_reads(
    *, kernel: NativeCaseRPeriodKernel, context: SolverBuildContext, period: int
) -> NativeCaseRPeriodKernel:
    """Attach actual producer-date leaf addresses without rebuilding economics."""
    from _lcm.solution.continuation_reads import (  # noqa: PLC0415
        continuation_leaf_reads,
        published_continuation_template,
    )

    base = kernel.programs["main"].argument_builder
    if not isinstance(base, CaseRArguments):
        raise TypeError("Case-R program has no actual declared argument builder.")
    programs = {}
    for name, program in kernel.programs.items():
        reads = []
        for target in sorted(base.base.stateful_targets):
            template = published_continuation_template(
                continuation_specs=context.continuation_specs,
                target=target,
                period=period + 1,
            )
            if template is None:
                raise TypeError("Case-R child has no actual producer-date template.")
            reads.extend(
                continuation_leaf_reads(
                    template=template,
                    artifact_key=context.continuation_specs[target].artifact_key,
                    target=target,
                    source_regime=context.regime_name,
                    source_period=period,
                    core_key=name,
                )
            )
        programs[name] = replace(
            program,
            requirements=replace(program.requirements, value_reads=tuple(reads)),
        )
    return replace(kernel, programs=programs)


def install_balanced_case_r(
    *,
    core: NativeCaseRCore,
    spec: ProjectedContinuationSpec,
    period: int,
    template: NativeCaseROutput,
    native_arguments: NativeCaseRInputs,
) -> tuple[NativeCaseRCore, ProjectedContinuationSpec, NativeCaseROutput]:
    """Derive one private balanced schema from the actual native producer.

    The caller supplies the materialized argument tree, not invented guard
    values. Shape tracing derives every guard/catalog/row buffer from the same
    core that will produce it. The immutable returned declaration must be fed
    into the genuine parent builder before materialization. This explicit
    installer does not replace the model's shared-declaration identity gate;
    automatic model-wide balanced declarations remain a separate requirement.
    """
    balanced, date, actual_template = install_balanced_case_r_date(
        core=core,
        date=spec.date_at(period=period),
        template=template,
        native_arguments=native_arguments,
    )
    declaration = replace(
        spec,
        template=date.template,
        templates_by_period={**spec.templates_by_period, period: date.template},
        dates_by_period={**spec.dates_by_period, period: date},
        guard_programs={**spec.guard_programs, period: date.guard_program},
    )
    return balanced, declaration, actual_template


def trace_native_case_r(
    *, core: NativeCaseRCore, native_arguments: NativeCaseRInputs
) -> NativeCaseROutput:
    """Trace original numeric operands with its Python planner widths bound static."""
    trace = functools.partial(
        core,
        savings_point_width=native_arguments["savings_point_width"],
        stochastic_node_width=native_arguments["stochastic_node_width"],
        envelope_cell_width=native_arguments["envelope_cell_width"],
    )
    output = jax.eval_shape(
        trace,
        **{
            name: value
            for name, value in native_arguments.items()
            if name
            not in {
                "savings_point_width",
                "stochastic_node_width",
                "envelope_cell_width",
            }
        },
    )
    if not isinstance(output, NativeCaseROutput):
        raise TypeError("Native Case-R tracing lost its original output schema.")
    return output


def install_balanced_case_r_date(
    *,
    core: NativeCaseRCore,
    date: ProjectedProducerDate,
    template: NativeCaseROutput,
    native_arguments: NativeCaseRInputs,
    materializer: DatedGenerationMaterializer | None = None,
) -> tuple[NativeCaseRCore, ProjectedProducerDate, NativeCaseROutput]:
    """Authenticate and trace the actual core before an aggregate spec exists."""
    from _lcm.egm.current_iid_balanced import BalancedResourcesProgram  # noqa: PLC0415

    if date.authority.capacity != core.projection_capacity:
        raise TypeError("Balanced core capacity differs from its actual declaration.")
    if template.projected is not date.template:
        raise TypeError("Balanced installation requires the actual native template.")
    pool = native_arguments.get("pool")
    if not isinstance(pool, Mapping) or "period" not in pool:
        raise TypeError("Balanced native arguments require their actual dated pool.")
    period = pool["period"]
    if (
        not isinstance(period, jax.Array)
        or period.shape != ()
        or period.dtype != jnp.int32
    ):
        raise TypeError("Balanced preparation requires its canonical scalar period.")
    if int(period) != date.authority.period:
        raise TypeError("Balanced preparation cannot redate its actual producer.")
    program = BalancedResourcesProgram.authenticate(
        pieces=core.pieces,
        owner_capacity=core.owner_capacity,
        abstract_pool=pool,
    )
    balanced = replace(core, balanced=program)
    output = trace_native_case_r(core=balanced, native_arguments=native_arguments)
    # Numeric placeholders establish shape/dtype only and grant no coverage.
    actual_template = (
        _zeros_for_template(schema=output)
        if materializer is None
        else materializer.produce(
            function=functools.partial(_zeros_for_template, schema=output),
            arguments={},
        )
    )
    if not isinstance(actual_template, NativeCaseROutput):
        raise TypeError("Balanced shape trace lost its original native output schema.")
    if (actual_template.projected.grid.shape, actual_template.projected.grid.dtype) != (
        date.template.grid.shape,
        date.template.grid.dtype,
    ):
        raise TypeError("Balanced schema changed its original declared liquid support.")
    # Retain declared coordinates; zero-valued validity/status still grants no coverage.
    actual_template = actual_template._replace(
        projected=replace(
            actual_template.projected,
            grid=date.template.grid,
            domain_lower=date.template.domain_lower,
            domain_upper=date.template.domain_upper,
        )
    )
    binding = ProjectedProducerDate(
        authority=date.authority,
        template=actual_template.projected,
        guard_program=program,
    )
    return balanced, binding, actual_template


def _zeros_for_template(*, schema: ShapeDtypePytree | NativeCaseROutput) -> PytreeValue:
    """Materialize the actual core's traced output schema without running it."""
    return jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype), schema)
