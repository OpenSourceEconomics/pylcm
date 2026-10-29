"""Private installed U publication with exact records and guarded BALANCED links.

Original point evaluations supply every value and economic marginal. The proved
saturated-span class may refine stored PWA links; other U classes remain exact.
Original conditional candidate reconstruction supplies forward policies.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields, replace
from dataclasses import field as dataclass_field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from _lcm.certainty_equivalent import aggregates_nonlinearly
from _lcm.continuation import (
    ContinuationPayload,
    ContinuationSpec,
    ProjectedContinuationSpec,
    ProjectedProducerDate,
)
from _lcm.current_iid import AdmittedGroupParams, NodeOperands
from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_balanced import _ordered_points
from _lcm.egm.current_iid_case_u_balanced import (
    BalancedCaseUCarry,
    CaseUBalancedOperands,
    SaturatedCaseUProgram,
    prepare_saturated_case_u,
)
from _lcm.egm.current_iid_case_u_replay import (
    CASE_U_FUTURE_REPLAY,
    CASE_U_REPLAY_IDENTITY,
    CaseUReplayChildren,
    CaseUReplayFutureDate,
    CaseUReplayPeriod,
    CaseUReplaySpec,
)
from _lcm.egm.current_iid_demands import (
    ParentResourcesDemand,
    prepare_parent_resources_demands,
)
from _lcm.egm.current_iid_evaluation import CurrentResourcesEvaluation
from _lcm.egm.current_iid_nbegm import (
    NBCaseUArguments,
    NBCaseUProgram,
    NBCaseURidePoint,
)
from _lcm.egm.current_iid_negm import NEGMCaseUPoint
from _lcm.egm.current_iid_negm_program import NEGMCaseUArguments, NEGMCaseUProgram
from _lcm.egm.current_iid_nnbegm import NNBEGMCaseUPoint
from _lcm.egm.current_iid_resources_proof import prove_resources_only
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.egm.query_data import QueryNodeKind
from _lcm.egm.step import _EGMStep
from _lcm.egm.upper_envelope import get_bracket_finder
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import CoreBuildContext, CoreProgram, ProgramScope
from _lcm.execution.output_layout import StateAxesLeading
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
from _lcm.solution.contract import SolutionKernels, SolverBuildContext
from _lcm.solution.current_iid_case_u_build import (
    CaseUArguments,
    CaseUProgramFunction,
    guard_case_u_shape,
    materialize_case_u_schema,
    prepare_case_u_clock,
    prepare_case_u_point_program,
    trace_case_u_shape,
)
from _lcm.solution.dcegm import EGMStepBuild, _BoundDCEGM
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    FeasibilityKernelOutput,
    QueryInputStatus,
    reduce_bound_input_status,
    reduce_continuation_read_counts,
    reduce_query_input_status,
)
from _lcm.solution.fingerprint import _unwrap_exact_beartype_wrapper
from _lcm.solution.generation import CanonicalModelGeneration
from _lcm.solution.model_authority import _authority_from_template
from _lcm.time import TimeAxis
from _lcm.typing import (
    EconFunctionArg,
    EconFunctionKwargs,
    FlatParams,
    PytreeChild,
    PytreeValue,
    ShapeDtypePytree,
)
from lcm._solver_api.authority import _ArrayCopier
from lcm.exceptions import UnsupportedOperationError
from lcm.solver_api import ArtifactChannel, DeclaredReplay, PersistencePolicy
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    IntND,
    ReferenceName,
    RegimeName,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
    StateName,
)


def is_case_u_source(
    *, context: SolverBuildContext, solver: _BoundDCEGM, period: int
) -> bool:
    """Classify original dependency roots, not exceptions or sampled values."""
    plan = context.current_iid_plans.get(period)
    if plan is None:
        raise TypeError("Current-IID classification requires its actual dated plan.")
    proof = prove_resources_only(context=context, solver=solver, plan=plan)
    return bool(
        proof.bypasses
        or solver.continuous_state in proof.context_inputs
        or aggregates_nonlinearly(context.certainty_equivalent)
    )


def _admit_case_u(
    *, context: SolverBuildContext, build: EGMStepBuild, period: int
) -> None:
    """Keep genuine first-installed scalar capabilities explicit."""
    step = build.steps[period]
    if not isinstance(step, _EGMStep):
        raise TypeError("Case-U installation requires its original Euler step.")
    pieces = step.pieces
    if (
        pieces.continuation_plan.certainty_equivalent
        is not context.certainty_equivalent
    ):
        raise TypeError("Installed Case-U lost its original CE declaration identity.")
    user = context.user_regimes[context.regime_name]
    law = context.laws[context.regime_name]
    if (
        tuple(context.state_action_space.states) != (pieces.euler_state_name,)
        or context.state_action_space.discrete_actions
        or pieces.combo_names
        or pieces.lower_func is not None
        or pieces.query_spec is not None
        or pieces.continuation_plan.query_targets
        or pieces._prepared_nested_floor is not None  # noqa: SLF001
        or pieces._prepared_nested_axes  # noqa: SLF001
        or context.sharded_state_names
        or context.co_map_state_names
        or context.action_partitions != 1
        or context.has_taste_shocks
        or context.phase_variation_paths
        or user.exits
        or user.same_period_refs
        or user.value_constraints
        or law.arrivals
        or law.gated_edges
        or context.arrival_queries
    ):
        raise UnsupportedOperationError(
            "Installed exact Case-U still requires "
            "passive/choice/query/projected-child/"
            "moving-floor/nested/phase/gated/arrival/sharded capability."
        )


@dataclass(frozen=True, kw_only=True)
class CaseUExactOutput:
    """U-specific integrated source channels; no per-current-node policy artifact."""

    value: FloatND
    empty: BoolND
    projected: ProjectedEGMCarry
    source: CurrentResourcesEvaluation
    current_iid_invalid: ScalarBool
    future: CaseUReplayChildren | None = None


type _ExactChild = (
    PytreeChild | ProjectedEGMCarry | CurrentResourcesEvaluation | CaseUReplayChildren
)


def _flatten_exact_output(
    output: CaseUExactOutput,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, _ExactChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(field.name), getattr(output, field.name))
        for field in fields(output)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_exact_output(
    metadata: None, children: Iterable[_ExactChild]
) -> CaseUExactOutput:
    del metadata
    output = object.__new__(CaseUExactOutput)
    for field, value in zip(fields(CaseUExactOutput), children, strict=True):
        object.__setattr__(output, field.name, value)
    return output


jax.tree_util.register_pytree_with_keys(
    CaseUExactOutput, _flatten_exact_output, _unflatten_exact_output
)


@dataclass(frozen=True, kw_only=True, eq=False)
class CaseUExactArguments:
    """Actual point inputs plus every original incoming parent's numeric law inputs."""

    point: CaseUArguments | NBCaseUArguments | NEGMCaseUArguments
    demands: tuple[ParentResourcesDemand, ...] = ()
    fixed_pools: tuple[EconFunctionKwargs, ...] = ()
    trace_capture_roots: tuple[jax.Array, ...] = ()
    """Original exact-publisher callback buffers retained with its program."""

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        arguments = dict(self.point(context))
        if context.ages is None:
            raise TypeError("Case-U parent demand projection lost its actual clock.")
        ages = context.ages
        return self._arguments(
            context=context,
            arguments=arguments,
            clock=lambda demand: {
                "period": jnp.int32(demand.period),
                "age": ages.values[demand.period],
            },
        )

    def prepare(
        self, *, context: CoreBuildContext, materializer: DatedGenerationMaterializer
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        """Admit point and each original incoming parent's separate clock."""
        arguments = dict(self.point.prepare(context=context, materializer=materializer))
        if context.ages is None:
            raise TypeError("Case-U parent demand projection lost its actual clock.")
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
        """Share the original parent namespace/savings assembly unchanged."""
        arguments["_lcm_parent_demand_pools"] = tuple(
            {
                **fixed,
                **union_free_params(
                    flat_params=MappingProxyType(context.flat_params),
                    regime_name=demand.source,
                    transition_target_names=demand.transition_targets,
                ),
                **clock(demand),
            }
            for demand, fixed in zip(self.demands, self.fixed_pools, strict=True)
        )
        arguments["_lcm_parent_demand_savings"] = tuple(
            demand.pieces.savings_nodes for demand in self.demands
        )
        return MappingProxyType(arguments)


@runtime_checkable
class _CaseUProjectionPublisher(Protocol):
    """Publish the actual source's stored guarded continuation."""

    def __call__(
        self, *, initial: ProjectedEGMCarry, guard: CaseUBalancedOperands
    ) -> BalancedCaseUCarry: ...


@dataclass(frozen=True, kw_only=True, eq=False)
class CaseUExactFunction:
    """Original enclosing computation at all mandatory declared/parent coordinates."""

    point: CaseUProgramFunction | NBCaseUProgram | NEGMCaseUProgram
    capacity: int
    demands: tuple[ParentResourcesDemand, ...] = ()
    replay_source: RegimeName | None = None
    replay_child_names: tuple[RegimeName, ...] = ()
    balanced: SaturatedCaseUProgram | None = dataclass_field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    _publish_balanced: _CaseUProjectionPublisher | None = dataclass_field(
        default=None, init=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Original bound publisher selected from the retained source authority."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_publish_balanced",
            None if self.balanced is None else self.balanced.publish,
        )

    def __call__(
        self,
        *,
        _lcm_parent_demand_pools: tuple[EconFunctionKwargs, ...],
        _lcm_parent_demand_savings: tuple[Float1D, ...],
        next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
        _lcm_current_iid: AdmittedGroupParams,
        _lcm_current_iid_nodes: NodeOperands,
        _lcm_current_iid_rows: IntND,
        _lcm_case_u_savings_nodes: FloatND,
        _lcm_case_u_borrowing_limit: ScalarFloat,
        _lcm_case_u_floor_status: BoundInputStatus | None,
        _lcm_case_u_child_node_values: tuple[tuple[FloatND | IntND, ...], ...],
        **arguments: EconFunctionArg,
    ) -> BoundedResult[CaseUExactOutput]:
        state = self.point.pieces.euler_state_name
        declared = arguments[state]
        if not isinstance(declared, jax.Array) or declared.ndim != 1:
            raise TypeError("Case-U exact publisher lost its actual liquid support.")
        queries, laws_valid = prepare_parent_resources_demands(
            declarations=self.demands,
            pools=_lcm_parent_demand_pools,
            savings=_lcm_parent_demand_savings,
            dtype=declared.dtype,
        )
        raw = jnp.concatenate((declared, queries))
        lower, upper = declared[0], declared[-1]
        inputs_valid = (
            laws_valid
            & jnp.all(jnp.isfinite(raw))
            & jnp.all(declared[1:] > declared[:-1])
            & jnp.all((queries >= lower) & (queries <= upper))
        )
        points, length, capacity_valid = _ordered_points(
            raw=raw, active=jnp.ones(raw.shape, dtype=jnp.bool_), capacity=self.capacity
        )
        active = jnp.arange(self.capacity) < length
        grid = jnp.where(active, points, lower)
        # Capacity failure is mandatory invalidity; no truncated prefix is certified.
        survives = jnp.all(
            jnp.any((raw[:, None] == grid[None, :]) & active[None, :], axis=1)
        )
        result, law_valid = self.point.evaluate_with_law(
            next_regime_to_continuation=next_regime_to_continuation,
            _lcm_current_iid=_lcm_current_iid,
            _lcm_current_iid_nodes=_lcm_current_iid_nodes,
            _lcm_current_iid_rows=_lcm_current_iid_rows,
            _lcm_case_u_savings_nodes=_lcm_case_u_savings_nodes,
            _lcm_case_u_borrowing_limit=_lcm_case_u_borrowing_limit,
            _lcm_case_u_floor_status=_lcm_case_u_floor_status,
            _lcm_case_u_child_node_values=_lcm_case_u_child_node_values,
            **{**arguments, state: grid},
        )
        source = result.output
        known = jnp.isfinite(source.value) | (
            jnp.isneginf(source.value) & (source.economic_marginal == 0)
        )
        valid = (
            inputs_valid
            & capacity_valid
            & survives
            & jnp.all(~active | (source.valid & known))
        )
        statuses = reduce_query_input_status(
            statuses=result.query_status, active=active
        )
        if statuses is None:
            raise TypeError(
                "Case-U exact publication lost its original source statuses."
            )
        own = statuses[-1]
        statuses = (
            *statuses[:-1],
            QueryInputStatus(
                own.nonfinite_floor,
                own.invalid_shifted_grid,
                own.coverage_missing | ~valid,
            ),
        )
        indices = jnp.searchsorted(points, declared)
        indices = jnp.clip(indices, 0, self.capacity - 1)
        projected = ProjectedEGMCarry(
            grid=grid,
            value=jnp.where(active, source.value, 0),
            economic_marginal=jnp.where(active, source.economic_marginal, 0),
            # This is an exact-address index, never a candidate-topology certificate.
            owner_id=jnp.where(active, jnp.arange(self.capacity), -1).astype(jnp.int32),
            side_id=jnp.zeros((self.capacity,), dtype=jnp.int32),
            node_kind=jnp.where(
                active & known,
                jnp.where(
                    jnp.isneginf(source.value),
                    QueryNodeKind.ECONOMIC_MINUS_INF,
                    QueryNodeKind.FINITE,
                ),
                QueryNodeKind.PAD,
            ).astype(jnp.int32),
            link_valid=jnp.zeros((self.capacity - 1,), dtype=jnp.bool_),
            valid_length=length,
            domain_lower=lower,
            domain_upper=upper,
            valid=valid,
        )
        if self._publish_balanced is not None:
            projected = self._publish_balanced(
                initial=projected,
                guard=CaseUBalancedOperands.create(
                    pool=MappingProxyType(
                        {
                            name: value
                            for name, value in arguments.items()
                            if name != state
                        }
                    ),
                    children=next_regime_to_continuation,
                    group_params=_lcm_current_iid,
                    node_operands=_lcm_current_iid_nodes,
                    indices=_lcm_current_iid_rows,
                    savings_nodes=_lcm_case_u_savings_nodes,
                    borrowing_limit=_lcm_case_u_borrowing_limit,
                    floor_status=_lcm_case_u_floor_status,
                    child_nodes=_lcm_case_u_child_node_values,
                    lower=lower,
                    upper=upper,
                ),
            )
            own = statuses[-1]
            statuses = (
                *statuses[:-1],
                QueryInputStatus(
                    own.nonfinite_floor,
                    own.invalid_shifted_grid,
                    own.coverage_missing | ~projected.valid,
                ),
            )
        return BoundedResult(
            output=CaseUExactOutput(
                value=jnp.where(valid, source.value[indices], jnp.nan),
                empty=source.empty[indices],
                projected=projected,
                source=source,
                current_iid_invalid=~jnp.all(~active | law_valid),
                future=_case_u_future_output(
                    source=self.replay_source,
                    period=self.point.period,
                    names=self.replay_child_names,
                    children=next_regime_to_continuation,
                ),
            ),
            bound_status=reduce_bound_input_status(
                statuses=result.bound_status, active=active
            ),
            query_status=statuses,
            continuation_read_counts=reduce_continuation_read_counts(
                counts=result.continuation_read_counts, active=active
            ),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class CaseURideExactFunction:
    """Original exact U publisher mapped over authentic passive-state support."""

    point: NBCaseUProgram | NEGMCaseUProgram
    capacity: int
    passive_names: tuple[StateName, ...]
    state_names: tuple[StateName, ...]
    demands: tuple[ParentResourcesDemand, ...] = ()
    replay_source: RegimeName | None = None
    replay_child_names: tuple[RegimeName, ...] = ()

    def __call__(
        self,
        *,
        _lcm_parent_demand_pools: tuple[EconFunctionKwargs, ...],
        _lcm_parent_demand_savings: tuple[Float1D, ...],
        next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
        _lcm_current_iid: AdmittedGroupParams,
        _lcm_current_iid_nodes: NodeOperands,
        _lcm_current_iid_rows: IntND,
        _lcm_case_u_savings_nodes: FloatND,
        _lcm_case_u_borrowing_limit: ScalarFloat,
        _lcm_case_u_floor_status: BoundInputStatus | None,
        _lcm_case_u_child_node_values: tuple[tuple[FloatND | IntND, ...], ...],
        **arguments: EconFunctionArg,
    ) -> BoundedResult[CaseUExactOutput]:
        if not (
            (
                type(self.point) is NBCaseUProgram
                and type(self.point.pieces) is NBCaseURidePoint
            )
            or (
                type(self.point) is NEGMCaseUProgram
                and type(self.point.pieces) in (NEGMCaseUPoint, NNBEGMCaseUPoint)
            )
        ):
            raise TypeError(
                "Passive U publication lost its genuine source constructor."
            )
        supports = {}
        for name in self.passive_names:
            operand = arguments[name]
            if not isinstance(operand, jax.Array) or operand.ndim != 1:
                raise TypeError("Passive NB publication lost its actual state support.")
            supports[name] = operand
        shape = tuple(supports[name].shape[0] for name in self.passive_names)
        mesh = jnp.meshgrid(
            *(supports[name] for name in self.passive_names), indexing="ij"
        )
        cells = {
            name: values.reshape(-1)
            for name, values in zip(self.passive_names, mesh, strict=True)
        }
        scalar = CaseUExactFunction(
            point=self.point, capacity=self.capacity, demands=self.demands
        )

        def one(
            cell: Mapping[StateName, ScalarFloat | ScalarInt],
        ) -> BoundedResult[CaseUExactOutput]:
            return scalar(
                _lcm_parent_demand_pools=_lcm_parent_demand_pools,
                _lcm_parent_demand_savings=_lcm_parent_demand_savings,
                next_regime_to_continuation=next_regime_to_continuation,
                _lcm_current_iid=_lcm_current_iid,
                _lcm_current_iid_nodes=_lcm_current_iid_nodes,
                _lcm_current_iid_rows=_lcm_current_iid_rows,
                _lcm_case_u_savings_nodes=_lcm_case_u_savings_nodes,
                _lcm_case_u_borrowing_limit=_lcm_case_u_borrowing_limit,
                _lcm_case_u_floor_status=_lcm_case_u_floor_status,
                _lcm_case_u_child_node_values=_lcm_case_u_child_node_values,
                **{**arguments, **cell},
            )

        mapped = jax.tree.map(
            lambda leaf: leaf.reshape((*shape, *leaf.shape[1:])),
            jax.vmap(one)(cells),
        )
        output = mapped.output
        active = jnp.ones(shape, dtype=jnp.bool_)
        # Projected rows keep passive-leading layout; V keeps the real state order.
        state_axis = self.state_names.index(self.point.pieces.euler_state_name)
        return BoundedResult(
            output=CaseUExactOutput(
                value=jnp.moveaxis(output.value, -1, state_axis),
                empty=jnp.moveaxis(output.empty, -1, state_axis),
                projected=output.projected,
                source=output.source,
                current_iid_invalid=jnp.any(output.current_iid_invalid),
                future=_case_u_future_output(
                    source=self.replay_source,
                    period=self.point.period,
                    names=self.replay_child_names,
                    children=next_regime_to_continuation,
                ),
            ),
            bound_status=reduce_bound_input_status(
                statuses=mapped.bound_status, active=active
            ),
            query_status=reduce_query_input_status(
                statuses=mapped.query_status, active=active
            ),
            continuation_read_counts=reduce_continuation_read_counts(
                counts=mapped.continuation_read_counts, active=active
            ),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class PreparedCaseUDate:
    """Actual pre-install objects later authenticated by their installed generation."""

    context: SolverBuildContext
    build: EGMStepBuild
    period: int
    date: ProjectedProducerDate
    program: CoreProgram
    template: BoundedResult[CaseUExactOutput]


def _require_case_u_date_lineage(
    *,
    context: SolverBuildContext,
    solver: _BoundDCEGM,
    build: EGMStepBuild,
    period: int,
    date: ProjectedProducerDate,
    previous: PreparedCaseUDate | None,
) -> None:
    """Authenticate initial declaration or the actual producer being closed."""
    step = build.steps[period]
    if not isinstance(step, _EGMStep):
        raise TypeError("Case-U preparation lost its original Euler step.")
    source = build.native_case_u_sources.get(period)
    if source is None:
        raise TypeError(
            "Case-U preparation lacks its original dated numerical authority."
        )
    if previous is None:
        if source.date is not date:
            raise TypeError("Case-U initial preparation lost its original source date.")
    else:
        previous_function = previous.program.function
        if not isinstance(previous_function, CaseUExactFunction):
            raise TypeError("Case-U closure lost its actual prepared producer lineage.")
        expected_guard = source.date.guard_program
        if previous_function.balanced is not None:
            retained = previous_function.balanced
            if (
                type(retained) is not SaturatedCaseUProgram
                or source.date.guard_program is not None
                or retained.point is not previous_function.point
                or retained.point.pieces is not step.pieces
                or retained.point.plan is not source.plan
                or retained.point.period != period
            ):
                raise TypeError("Case-U closure lost its original saturated producer.")
            expected_guard = retained
        if (
            previous.date is not date
            or previous.period != period
            or previous.build.native_case_u_sources.get(period) is not source
            or previous.build.steps.get(period) is not build.steps.get(period)
            or previous.template.output.projected is not date.template
            or date.authority is not source.date.authority
            or date.guard_program is not expected_guard
            or not isinstance(previous.program.function, CaseUExactFunction)
            or not isinstance(previous.program.function.point, CaseUProgramFunction)
            or not isinstance(previous.program.argument_builder, CaseUExactArguments)
            or not isinstance(previous.program.argument_builder.point, CaseUArguments)
            or previous.program.function.point.plan is not source.plan
            or previous.program.argument_builder.point.plan is not source.plan
            or previous.program.function.point.pieces
            is not previous.program.argument_builder.point.pieces
            or previous.program.function.point.pieces.utility_func
            is not step.pieces.utility_func
            or previous.program.function.point.pieces.continuation_plan
            is not step.pieces.continuation_plan
        ):
            raise TypeError("Case-U closure lost its actual prepared producer lineage.")
        source.require(context=previous.context, solver=solver, period=period)
    source.require(context=context, solver=solver, period=period)


def _case_u_future_output(
    *,
    source: RegimeName | None,
    period: int,
    names: tuple[RegimeName, ...],
    children: Mapping[RegimeName, ContinuationPayload],
) -> CaseUReplayChildren | None:
    """Return only the declared replay variant's actual numerical child inputs."""
    if source is None:
        return None
    values: list[EGMCarry] = []
    for name in names:
        child = children[name]
        if not isinstance(child, EGMCarry):
            raise TypeError("Case-U replay lost its actual ordinary child.")
        values.append(child)
    return CaseUReplayChildren(
        source=source,
        period=period,
        child_names=names,
        child_values=tuple(values),
    )


def _case_u_replay_program(
    *, original: CoreProgram, future: CaseUReplayFutureDate
) -> CoreProgram:
    """Declare the actual returned future bundle and its complete original reads."""
    function = original.function
    roles = original.output_roles
    if (
        not isinstance(function, (CaseUExactFunction, CaseURideExactFunction))
        or not isinstance(roles, BoundedResult)
        or not isinstance(roles.output, CaseUExactOutput)
    ):
        raise TypeError("Case-U replay lost its original exact producer schema.")
    future_roles = jax.tree.map(
        lambda leaf: StateAxesLeading(
            state_names=(), shape=leaf.shape, dtype=leaf.dtype
        ),
        future.template(),
    )
    replay_output = _unflatten_exact_output(
        None,
        (
            future_roles
            if field.name == "future"
            else getattr(roles.output, field.name)
            for field in fields(CaseUExactOutput)
        ),
    )
    replay_roles = jax.tree.map(
        lambda value: replay_output if isinstance(value, CaseUExactOutput) else value,
        roles,
        is_leaf=lambda value: isinstance(value, CaseUExactOutput),
    )
    return replace(
        original,
        name="replay",
        function=replace(
            function,
            replay_source=future.source,
            replay_child_names=tuple(future.specs),
        ),
        output_roles=replay_roles,
        requirements=replace(
            original.requirements,
            value_reads=tuple(
                replace(read, source=replace(read.source, core_key="replay"))
                for read in original.requirements.value_reads
            ),
        ),
        scope=ProgramScope.REPLAY,
        retained_artifact_keys=(CASE_U_FUTURE_REPLAY,),
        retained_artifact_payload_types={CASE_U_FUTURE_REPLAY: CaseUReplayChildren},
        replaces_program="main",
    )


def prepare_case_u_date(  # noqa: C901 - authenticate canonical closure before publisher preparation
    *,
    context: SolverBuildContext,
    solver: _BoundDCEGM,
    build: EGMStepBuild,
    period: int,
    date: ProjectedProducerDate,
    call: CoreBuildContext,
    fixed_flat_params: FlatParams,
    demands: tuple[ParentResourcesDemand, ...] = (),
    previous: PreparedCaseUDate | None = None,
    materializer: DatedGenerationMaterializer | None = None,
) -> PreparedCaseUDate:
    """Prepare the original core, admitted publisher and actual abstract schema."""
    _require_case_u_date_lineage(
        context=context,
        solver=solver,
        build=build,
        period=period,
        date=date,
        previous=previous,
    )
    _admit_case_u(context=context, build=build, period=period)
    if (
        date.authority.producer != context.regime_name
        or date.authority.period != period
    ):
        raise TypeError("Case-U publication borrowed another original source/date.")
    point, _ = prepare_case_u_point_program(
        source=context,
        build=build,
        period=period,
        call_context=call,
        fixed_flat_params=fixed_flat_params,
        materializer=materializer,
    )
    if not isinstance(point.argument_builder, CaseUArguments) or not isinstance(
        point.function, CaseUProgramFunction
    ):
        raise TypeError("Case-U exact publisher lost its original point core.")
    pools = tuple(
        union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=demand.source,
            transition_target_names=demand.transition_targets,
        )
        for demand in demands
    )
    arguments = CaseUExactArguments(
        point=point.argument_builder, demands=demands, fixed_pools=pools
    )
    function = CaseUExactFunction(
        point=point.function, capacity=date.authority.capacity, demands=demands
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
    balanced = prepare_saturated_case_u(
        context=context,
        solver=solver,
        build=build,
        period=period,
        point=point.function,
        arguments=actual_arguments,
    )
    if previous is not None and isinstance(
        previous.program.function, CaseUExactFunction
    ):
        retained = previous.program.function.balanced
        if retained is not None:
            if (
                balanced is None
                or point.function.pieces is not retained.point.pieces
                or point.function.plan is not retained.point.plan
                or point.function.period != retained.point.period
                or (
                    point.function.savings_point_width,
                    point.function.stochastic_node_width,
                    point.function.envelope_cell_width,
                )
                != (
                    retained.point.savings_point_width,
                    retained.point.stochastic_node_width,
                    retained.point.envelope_cell_width,
                )
            ):
                raise UnsupportedOperationError(
                    "Parent closure changed the original saturated-U eligibility."
                )
            balanced = retained
    if balanced is not None:
        function = replace(function, balanced=balanced)
        if materializer is not None:
            guard_case_u_shape(
                function=function,
                context=call,
                fixed_flat_params=fixed_flat_params,
                materializer=materializer,
            )
    template = trace_case_u_shape(function=function, arguments=actual_arguments)
    template = materialize_case_u_schema(schema=template, materializer=materializer)
    if not isinstance(template, BoundedResult) or not isinstance(
        template.output, CaseUExactOutput
    ):
        raise TypeError("Case-U shape preparation lost its actual U output schema.")
    if template.bound_status:
        raise UnsupportedOperationError(
            "Installed Case-U source-bound slot ownership is still required."
        )

    # No projected child queries are admitted in this first installed class:
    # every remaining status originates in this original U source computation.
    # keyword-only-exempt: library-callback=jax.tree_util.tree_map_with_path
    def role(
        path: tuple[jax.tree_util.KeyEntry, ...], leaf: jax.Array | jax.ShapeDtypeStruct
    ) -> StateAxesLeading:
        names = tuple(getattr(key, "name", None) for key in path)
        is_state_row = names in (("output", "value"), ("output", "empty"))
        return StateAxesLeading(
            state_names=(solver.continuous_state,) if is_state_row else (),
            shape=leaf.shape,
            dtype=leaf.dtype,
        )

    roles = jax.tree_util.tree_map_with_path(role, template)
    program = replace(
        point,
        name="main",
        function=function,
        argument_builder=arguments,
        requirements=replace(
            point.requirements,
            value_reads=tuple(
                replace(read, source=replace(read.source, core_key="main"))
                for read in point.requirements.value_reads
            ),
        ),
        output_roles=roles,
    )
    actual_date = replace(
        date, template=template.output.projected, guard_program=balanced
    )
    return PreparedCaseUDate(
        context=context,
        build=build,
        period=period,
        date=actual_date,
        program=program,
        template=template,
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class NativeCaseUPeriodKernel:
    """Real private U publisher with dated scalar reconstruction when admitted."""

    prepared: PreparedCaseUDate
    spec: ProjectedContinuationSpec
    replay_future: CaseUReplayFutureDate | None = None
    replay_program: CoreProgram | None = None

    def core_programs(self) -> Mapping[str, CoreProgram]:
        if self.replay_program is None:
            return MappingProxyType({"main": self.prepared.program})
        return MappingProxyType(
            {
                "main": self.prepared.program,
                "replay": self.replay_program,
            }
        )

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> NativeCaseUPeriodKernel:
        arguments = self.prepared.program.argument_builder
        if not isinstance(arguments, CaseUExactArguments):
            raise TypeError("Installed Case-U lost its original argument projection.")
        arguments = replace(
            arguments,
            point=replace(arguments.point, fixed_flat_params=fixed_flat_params),
            fixed_pools=tuple(
                union_fixed_params(
                    fixed_flat_params=fixed_flat_params,
                    regime_name=d.source,
                    transition_target_names=d.transition_targets,
                )
                for d in arguments.demands
            ),
        )
        return replace(
            self,
            prepared=replace(
                self.prepared,
                program=replace(self.prepared.program, argument_builder=arguments),
            ),
            replay_program=None
            if self.replay_program is None
            else replace(self.replay_program, argument_builder=arguments),
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
        if period != self.prepared.period:
            raise TypeError("Installed Case-U cannot borrow another date.")
        context = CoreBuildContext(
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        name = "replay" if "replay" in compiled_cores else "main"
        program = self.core_programs()[name]
        evaluated = compiled_cores[name](**program.argument_builder(context))
        if not isinstance(evaluated, BoundedResult) or not isinstance(
            evaluated.output, CaseUExactOutput
        ):
            raise TypeError("Installed Case-U lost its actual native source channels.")
        output = evaluated.output
        replay = {}
        if name == "replay":
            future = output.future
            if (
                self.replay_future is None
                or not isinstance(future, CaseUReplayChildren)
                or future.source != self.replay_future.source
                or future.period != self.replay_future.period
                or future.child_names != tuple(self.replay_future.specs)
            ):
                raise TypeError("Case-U replay lost its actual produced future bundle.")
            replay[CASE_U_FUTURE_REPLAY] = future
        elif output.future is not None:
            raise TypeError("Values-only Case-U cannot publish a future replay bundle.")
        return FeasibilityKernelOutput(
            value=output.value,
            continuations={PROJECTED_EGM_CONTINUATION: output.projected},
            replay=replay,
            feasibility_counts=(
                jnp.sum(output.empty, dtype=jnp.int32),
                jnp.sum(jnp.isinf(output.value) & ~output.empty, dtype=jnp.int32),
            ),
            bound_status=evaluated.bound_status,
            query_status=evaluated.query_status,
            continuation_read_counts=evaluated.continuation_read_counts,
            current_iid_invalid=output.current_iid_invalid,
        )


def build_case_u_kernels(
    *,
    context: SolverBuildContext,
    spec: ProjectedContinuationSpec,
    prepared: Mapping[int, PreparedCaseUDate],
    build: EGMStepBuild,
    future_specs: Mapping[RegimeName, ContinuationSpec],
    array_copier: _ArrayCopier | None = None,
) -> SolutionKernels:
    """Install final exact date/schema identity into the genuine Model graph."""
    if set(build.native_case_u_sources) != set(prepared) or set(build.steps) != set(
        prepared
    ):
        raise TypeError(
            "Case-U final publication has incomplete numerical source dates."
        )
    for period, source in build.native_case_u_sources.items():
        source.require(context=context, solver=source.solver, period=period)
    periods = {}
    authorities = {}
    replay_template = None
    replay_periods = {}
    for period, original in prepared.items():
        date = spec.date_at(period=period)
        if original.date is not date:
            raise TypeError("Case-U final aggregate substituted its original date.")
        actual = replace(original, context=context, build=build)
        step = build.steps[period]
        if not isinstance(step, _EGMStep):
            raise TypeError("Case-U replay lost its actual original Euler step.")
        targets = (
            *step.pieces.continuation_plan.stateful_targets,
            *step.pieces.continuation_plan.scalar_targets,
        )
        replay_future = None
        # The existing authority map has one schema per regime/key. Date-varying
        # bundles remain unsupported until their real per-date authority exists.
        if (
            len(prepared) == 1
            and targets
            and all(
                target in future_specs
                and isinstance(
                    future_specs[target].template_at(period=period + 1), EGMCarry
                )
                for target in targets
            )
        ):
            replay_future = CaseUReplayFutureDate(
                source=context.regime_name,
                period=period,
                specs={target: future_specs[target] for target in targets},
            )
            replay_template = replay_future.template()
            function = actual.program.function
            arguments = actual.program.argument_builder
            if (
                not isinstance(function, CaseUExactFunction)
                or not isinstance(arguments, CaseUExactArguments)
                or not isinstance(function.point, CaseUProgramFunction)
                or not isinstance(arguments.point, CaseUArguments)
            ):
                raise TypeError("Case-U reconstruction lost its real point factory.")
            replay_periods[period] = CaseUReplayPeriod(
                future=replay_future,
                date=date,
                pieces=function.point.pieces,
                replay_refine_to_bracket=get_bracket_finder(
                    solver=build.native_case_u_sources[period].solver,
                    n_refined=function.point.pieces.n_pad,
                ),
                fixed_flat_params=arguments.point.fixed_flat_params,
                transition_target_names=arguments.point.base.transition_target_names,
                savings_point_width=function.point.savings_point_width,
                stochastic_node_width=function.point.stochastic_node_width,
                envelope_cell_width=function.point.envelope_cell_width,
            )
        periods[period] = NativeCaseUPeriodKernel(
            prepared=actual,
            spec=spec,
            replay_future=replay_future,
            replay_program=None
            if replay_future is None
            else _case_u_replay_program(original=actual.program, future=replay_future),
        )
        if actual.template.query_status is None:
            raise TypeError("Case-U installed source status roster is missing.")
        authorities[period] = tuple(
            (context.regime_name, period, date.authority)
            for _ in actual.template.query_status
        )
    replay = _case_u_replay_spec(
        regime_name=context.regime_name, periods=replay_periods, build=build
    )
    return SolutionKernels(
        period_kernels=periods,
        period_group_keys={
            p: ("case_u_exact", p, build.step_group_keys[p]) for p in periods
        },
        continuation_spec=spec,
        query_input_authorities=authorities,
        replay_route=DeclaredReplay.UNSUPPORTED,
        case_u_replay_spec=replay,
        artifact_authorities={}
        if replay_template is None
        else {
            CASE_U_FUTURE_REPLAY: _authority_from_template(
                key=CASE_U_FUTURE_REPLAY,
                channel=ArtifactChannel.REPLAY,
                persistence=PersistencePolicy.MODEL_VERIFIABLE,
                payload_runtime_type=type(replay_template),
                template=replay_template,
                consumer_route=CASE_U_REPLAY_IDENTITY,
                applicable=True,
                required=replay is not None and replay.reconstructs_candidates,
                array_copier=array_copier,
            )
        },
    )


def _case_u_replay_spec(
    *,
    regime_name: RegimeName,
    periods: Mapping[int, CaseUReplayPeriod],
    build: EGMStepBuild,
) -> CaseUReplaySpec | None:
    """Bind one immutable replay/profile choice to the actual dated sources."""
    if not periods:
        return None
    sources = tuple(build.native_case_u_sources.values())
    solver = sources[0].solver
    if any(
        source.solver.replay != solver.replay
        or source.solver.accuracy != solver.accuracy
        for source in sources
    ):
        raise UnsupportedOperationError(
            "Date-varying EGM replay/profile declarations are not installed."
        )
    return CaseUReplaySpec(
        regime=regime_name,
        periods=periods,
        replay=solver.replay,
        accuracy=solver.accuracy,
    )


def acquire_installed_case_u_date(
    *, generation: CanonicalModelGeneration, regime_name: RegimeName, period: int
) -> PreparedCaseUDate:
    """Authenticate the same genuine preparation after final Model installation."""
    owner = generation.solver_preparation(regime_name=regime_name, period=period)
    owner.validate_owner()
    kernel = owner.installed_kernel
    if not isinstance(kernel, NativeCaseUPeriodKernel):
        raise UnsupportedOperationError(
            "This genuine Model date has no installed Case-U publisher."
        )
    prepared = kernel.prepared
    if (
        prepared.context is not owner.context
        or prepared.build is not owner.build
        or prepared.build.steps[period] is not owner.step
        or prepared.context.current_iid_plans[period] is not owner.plan
        or kernel.spec is not owner.declarations[regime_name]
        or prepared.date is not kernel.spec.date_at(period=period)
        or kernel.core_programs()["main"] is not prepared.program
    ):
        raise TypeError("Installed Case-U substituted its actual dated preparation.")
    return prepared


_ORIGINAL_CASE_U_PUBLISHER_METHODS = tuple(
    (owner, name, method, method.__code__, raw, raw.__code__)
    for owner, names in (
        (CaseUExactFunction, ("__post_init__", "__call__")),
        (CaseURideExactFunction, ("__call__",)),
    )
    for name in names
    for method in (getattr(owner, name),)
    for raw in (_unwrap_exact_beartype_wrapper(method) or method,)
)
