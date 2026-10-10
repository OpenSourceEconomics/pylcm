"""The default grid-search solver.

`GridSearch` runs the max-Q-over-a grid search. Its `build_period_kernels`
returns one `PeriodKernel` per period. Eligible ordinary hard-max kernels declare their
canonical action product for blockwise execution, and the engine binds the block width
before lowering. Collective and EV1 kernels deliberately retain their canonical dense
reduction order. Each streamed period program also names its exact value-input artifacts
and argument paths so the engine can resolve their transfers.
Fixed distributed states co-map ordinary continuation leaves with the streamed state
cell. Singleton folded-state routes stream actions before
the unchanged quadrature reduction; the fold axis itself remains materialized. Co-map
routes with separate same-period or edge-reference value channels retain the dense
kernel. The adapter assembles the resulting `KernelOutput` outside JIT.

The max-Q kernel-building imports are function-local so
the public `lcm.solvers` façade stays a thin re-export that pulls in no
numerical engine modules.
"""

import functools
import inspect
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.constraints.routes import (
    ConstraintRoute,
    ConstraintRouteKey,
    ConstraintSite,
)
from _lcm.continuation import EGMContinuationLayout
from _lcm.engine import (
    StateActionSpace,
    _build_regime_sharding,
    _RegimeSharding,
    placed_devices_for_ids,
)
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    InvariantBinding,
    ReducedAxis,
    TiledOutputAxis,
    ValueRead,
)
from _lcm.execution.invariant_blocks import block_state_action_space
from _lcm.execution.output_layout import (
    DISSOLUTION_FLAG,
    VALUE,
)
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.params.edges import regime_kernel_params
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION
from _lcm.solution.continuation_reads import rekeyed_value_reads
from _lcm.solution.contract import (
    ConstraintRouteContext,
    ContinuationPayload,
    PeriodKernel,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    simulation_route,
)
from _lcm.solution.dcegm import CELL_AXIS
from _lcm.time import TimeAxis
from _lcm.transition_plans import SupportOrigin
from _lcm.typing import (
    FlatParams,
    FlatRegimeParams,
    MaxQOverAFunction,
    QAndFArg,
    QAndFFunction,
    ReferenceName,
    RegimeName,
    StateName,
)
from lcm._solver_api.capabilities import SolverExecutionCapabilities
from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import DISSOLUTION_FLAG as DISSOLUTION_FLAG_ARTIFACT
from lcm.solver_api import KernelOutput
from lcm.typing import (
    FloatND,
)

# Planner name of the flattened Cartesian action axis grid search reduces.
ACTION_PRODUCT_AXIS = "action_product"

_ACTION_WIDTH_KEYWORD = "_lcm_action_block_width"
_CELL_WIDTH_KEYWORD = "_lcm_cell_width"
_CORE_RUNTIME_ARG_NAMES = frozenset(
    {
        "next_regime_to_V_arr",
        "same_period_regime_to_V_arr",
        "same_period_regime_to_params",
        "edge_reference_regime_to_V_arr",
        "edge_reference_regime_to_params",
        "period",
        "age",
    }
)


class _ActionStreamingDisposition(StrEnum):
    """Why one GridSearch solve route streams actions or keeps the dense core."""

    STREAMED = "streamed"
    DENSE_EV1_NONCANONICAL = "deliberately_dense:ev1_canonical_reduction_order"
    DENSE_COLLECTIVE_RESOURCES = "deliberately_dense:collective_resource_regression"
    DENSE_TRIVIAL_ACTION_PRODUCT = "deliberately_dense:trivial_action_product"
    DENSE_CO_MAP_REFERENCE_CHANNEL = (
        "deliberately_dense:co_map_with_separate_reference_channel"
    )
    UNSUPPORTED_COLLECTIVE_EV1 = "unsupported:collective_ev1"
    UNSUPPORTED_EV1_FOLD = "unsupported:ev1_fold"
    UNSUPPORTED_COLLECTIVE_FOLD = "unsupported:collective_fold"
    UNSUPPORTED_EV1_WITHOUT_DISCRETE_ACTION = "unsupported:ev1_without_discrete_action"

    @property
    def category(self) -> str:
        """Return the stable streamed/deliberately-dense/unsupported category."""
        return self.value.partition(":")[0]


def _select_action_width_keyword(*, context: SolverBuildContext) -> str:
    """Choose a deterministic planner keyword outside the model namespace."""
    return _select_width_keyword(context=context, prefix=_ACTION_WIDTH_KEYWORD)


def _select_cell_width_keyword(*, context: SolverBuildContext) -> str:
    """Choose the state-cell width keyword outside every model input namespace."""
    return _select_width_keyword(context=context, prefix=_CELL_WIDTH_KEYWORD)


def _select_width_keyword(*, context: SolverBuildContext, prefix: str) -> str:
    """Select an unoccupied deterministic suffix for a planner-owned width."""
    occupied = set(_CORE_RUNTIME_ARG_NAMES)
    occupied.update(context.flat_param_names)
    occupied.update(context.state_action_space.action_names)
    occupied.update(context.state_action_space.state_names)
    for Q_and_F in context.Q_and_F_functions.values():
        occupied.update(inspect.signature(Q_and_F).parameters)
    if context.pareto_weights is not None:
        occupied.update(context.pareto_weights.param_names)

    candidate = prefix
    suffix = 0
    while candidate in occupied:
        suffix += 1
        candidate = f"{prefix}_{suffix}"
    return candidate


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class GridSearch(Solver):
    """Grid-search solver over the full state-action product (the default)."""

    @property
    def capabilities(self) -> SolverExecutionCapabilities:
        """Describe this configured solver without building numerical kernels."""
        return SolverExecutionCapabilities(
            required_declaration="Regime or a specialized regime",
            problem_shape="General discrete-continuous action product",
            prerequisites=(
                "Ordinary callable constraints; EV1 taste shocks; transition-local "
                "joint lotteries"
            ),
            main_tradeoff=(
                "Broad representation; eligible singleton hard-max routes stream "
                "actions, while EV1 and collective reductions use dense actions"
            ),
            reduced_axes=("action_product",),
            tiled_axes=("cell",),
            host_axes=(),
            host_driven_programs=(),
            supports_ev1_taste_shocks=True,
            supports_nonlinear_certainty_equivalent=True,
        )

    @property
    def transition_local_lottery_origins(self) -> frozenset[SupportOrigin]:
        """Grid search enumerates every transition-local lottery inside Q."""
        return frozenset({SupportOrigin.DECLARED, SupportOrigin.SOURCE_PROCESS})

    @property
    def egm_continuation_layout(self) -> EGMContinuationLayout:
        """A brute child publishes one action-maxed row on its state grid."""
        return EGMContinuationLayout(
            retains_discrete_action_rows=False,
            rows_share_state_grid=True,
        )

    def build_constraint_routes(
        self, *, context: ConstraintRouteContext
    ) -> tuple[ConstraintRoute, ...]:
        """Declare the one route grid search walks: whole candidates, nothing hidden.

        The search enumerates the entire state-action product, so every name a
        constraint could read is bound where it evaluates. There is no inner
        stage to fall through to and nothing its construction enforces on a
        constraint's behalf, which is why the route is one unrestricted site
        carrying neither a proof nor a compiler.

        One route, not one per period: the search does not resolve its pool
        differently at any age, so a per-period key would put an entry per
        period in the plan where there is a single fact.
        """
        if context.phase == "simulate":
            return (simulation_route(context=context, solver_path=("grid_search",)),)
        return (
            ConstraintRoute(
                key=ConstraintRouteKey(
                    phase="solve",
                    period_group=None,
                    solver_path=("grid_search",),
                ),
                sites=(
                    ConstraintSite(
                        stage="state_action",
                        function_pool=context.functions,
                        available_names=None,
                    ),
                ),
            ),
        )

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one max-Q-over-a period adapter per period.

        Periods sharing the same Q_and_F object reuse the same selected program
        function so the execution layer can deduplicate their lowerings. An eligible
        route constructs only the streamed function; its dense evaluator remains an
        independently constructed test oracle rather than a second production path.
        """
        from _lcm.regime_building.max_Q_over_a import (  # noqa: PLC0415
            get_action_partitioned_max_Q_over_a,
            get_max_Q_over_a,
            get_streaming_max_Q_over_a,
        )
        from _lcm.regime_building.processing import (  # noqa: PLC0415
            get_conditioned_fold_weights_by_code,
        )

        program_functions: dict[int, MaxQOverAFunction] = {}
        broadcast_extents: dict[int, int] = {}
        result: dict[int, PeriodKernel] = {}
        # Fold weights are the folded process's own marginal distribution, a
        # plain constant computed once here at kernel-build time and never
        # inside the traced core (`_validate_fold_declarations` rejects a
        # runtime-parameterized process). Two shapes, by declaration:
        # - an unconditioned process contributes one row. Its
        #   `compute_transition_probs` returns an `(n_points, n_points)` matrix
        #   whose every row is that marginal — the "IID" part — so row 0 is it.
        # - a `StateConditioned` `sigma` contributes one row per category of
        #   the conditioning state, ordered by that categorical's integer code,
        #   which the fold reduction gathers along the conditioning axis.
        fold_weights: dict[StateName, FloatND] = {}
        fold_conditioning: dict[StateName, StateName] = {}
        for name in context.fold_state_names:
            process = cast("_ContinuousStochasticProcess", context.grids[name])
            if process.state_conditioned is None:
                fold_weights[name] = process.get_transition_probs()[0]
            else:
                fold_weights[name] = get_conditioned_fold_weights_by_code(
                    name=name, grid=process, grids=context.grids
                )
                fold_conditioning[name] = process.state_conditioned.on
        action_streaming = _classify_action_streaming(context=context)
        stream_actions = action_streaming is _ActionStreamingDisposition.STREAMED
        action_partition_mesh = _action_partition_mesh(
            context=context, action_streaming=action_streaming
        )
        action_width_keyword = _select_action_width_keyword(context=context)
        action_names = context.state_action_space.action_names
        action_extents = context.state_action_space.actions_grid_shapes
        untiled_state_names = tuple(
            name
            for name in context.state_action_space.state_names
            if name in context.sharded_state_names
            and name not in context.co_map_state_names
        )
        inner_state_names = tuple(
            name
            for name in context.state_action_space.state_names
            if name not in context.co_map_state_names
            and name not in untiled_state_names
        )
        # A blocked state enters each block program at its one bound code.
        state_extents = {
            name: 1
            if name in context.invariant_bindings
            else context.state_action_space.states[name].shape[0]
            for name in context.state_action_space.state_names
        }
        cell_extent = math.prod(state_extents[name] for name in inner_state_names)
        cell_width_keyword = (
            _select_cell_width_keyword(context=context) if cell_extent > 1 else None
        )
        for period, Q_and_F in context.Q_and_F_functions.items():
            q_id = id(Q_and_F)
            if q_id not in program_functions:
                broadcast_state_names = (
                    _continuation_unread_state_names(
                        Q_and_F=Q_and_F, inner_state_names=inner_state_names
                    )
                    if cell_width_keyword is not None
                    else ()
                )
                broadcast_extents[q_id] = math.prod(
                    state_extents[name] for name in broadcast_state_names
                )
                common_kwargs = {
                    "Q_and_F": Q_and_F,
                    "batch_sizes": dict.fromkeys(
                        context.state_action_space.state_names, 0
                    ),
                    "action_names": action_names,
                    "state_names": context.state_action_space.state_names,
                    "cell_width_keyword": cell_width_keyword,
                    "untiled_state_names": untiled_state_names,
                    "broadcast_state_names": broadcast_state_names,
                    "n_discrete_action_axes": len(
                        context.state_action_space.discrete_actions
                    ),
                    "has_taste_shocks": context.has_taste_shocks,
                    "co_map_state_names": context.co_map_state_names,
                    "co_map_v_arr_in_axes": context.co_map_v_arr_in_axes,
                    "stakeholders": context.stakeholders,
                    "pareto_weights": context.pareto_weights,
                    "fold_state_names": context.fold_state_names,
                    "fold_weights": MappingProxyType(fold_weights),
                    "fold_conditioning": MappingProxyType(fold_conditioning),
                }
                if action_partition_mesh is not None:
                    program_functions[q_id] = get_action_partitioned_max_Q_over_a(
                        Q_and_F=Q_and_F,
                        batch_sizes=common_kwargs["batch_sizes"],
                        action_names=action_names,
                        state_names=context.state_action_space.state_names,
                        n_partitions=context.action_partitions,
                        mesh=action_partition_mesh,
                        action_width_keyword=action_width_keyword,
                        cell_width_keyword=cell_width_keyword,
                        untiled_state_names=untiled_state_names,
                        broadcast_state_names=broadcast_state_names,
                    )
                else:
                    program_functions[q_id] = (
                        get_streaming_max_Q_over_a(
                            **common_kwargs,
                            action_width_keyword=action_width_keyword,
                        )
                        if stream_actions
                        else get_max_Q_over_a(**common_kwargs)
                    )
            target_regimes = (
                ()
                if period == context.solution_reachability.n_periods - 1
                else context.solution_reachability.targets(
                    period=period,
                    source=context.regime_name,
                )
            )
            edge_reference_regimes = _edge_reference_regimes_for_targets(
                context=context,
                target_regimes=target_regimes,
            )
            argument_builder = _GridSearchArgumentBuilder(
                regime_name=context.regime_name,
                same_period_ref_regimes=context.same_period_ref_regimes,
                edge_reference_regimes=edge_reference_regimes,
                edge_target_regimes=context.edge_target_regimes,
            )
            requirements = CoreExecutionRequirements(
                reduced_axes=(
                    (
                        ReducedAxis(
                            name=ACTION_PRODUCT_AXIS,
                            coordinate_names=action_names,
                            coordinate_extents=action_extents,
                            canonical_order="c",
                            reduction=HARD_MAX_REDUCTION,
                            width_keyword=action_width_keyword,
                        ),
                    )
                    if stream_actions
                    else ()
                ),
                tiled_axes=(
                    (
                        TiledOutputAxis(
                            name=CELL_AXIS,
                            state_names=inner_state_names,
                            extent=cell_extent,
                            width_keyword=cell_width_keyword,
                            # A width counts whole product points. The
                            # continuation is broadcast only at multiples of
                            # the broadcast extent, so the planner prefers them;
                            # every other width runs the plain layout.
                            preferred_alignment=broadcast_extents[q_id],
                            halve_on_materialised_gather=True,
                        ),
                    )
                    if cell_width_keyword is not None
                    else ()
                ),
                value_reads=_value_reads(
                    regime_name=context.regime_name,
                    period=period,
                    target_regimes=target_regimes,
                    same_period_ref_regimes=context.same_period_ref_regimes,
                    edge_reference_regimes=edge_reference_regimes,
                    edge_target_regimes=context.edge_target_regimes,
                ),
            )
            # A bound continuation needs its selected-view transfer even when
            # all local state and action products have extent one. "Dense"
            # arithmetic must not bypass the engine's value-read planning.
            requires_plan = bool(requirements.axes) or bool(context.invariant_bindings)
            program = CoreProgram(
                name="main",
                function=program_functions[q_id],
                argument_builder=argument_builder,
                requirements=requirements,
                output_roles=(
                    (VALUE, DISSOLUTION_FLAG)
                    if context.stakeholders is not None
                    else VALUE
                ),
                disposition=(
                    CoreExecutionDisposition.PLANNED
                    if requires_plan
                    else CoreExecutionDisposition.DENSE
                ),
                disposition_reason=(None if requires_plan else action_streaming.value),
                donation_candidates=(),
            )
            result[period] = _GridSearchPeriodKernel(
                _core_programs=MappingProxyType({"main": program})
                if not context.invariant_bindings
                else _bound_programs(program=program, context=context)
            )
        return SolutionKernels(period_kernels=MappingProxyType(result))


def _bound_programs(
    *, program: CoreProgram, context: SolverBuildContext
) -> MappingProxyType[str, CoreProgram]:
    """Declare one copy of `program` per code of the regime's blocked state.

    Each copy is named after its code, bound to it, and owns the reads `program`
    declared. All copies share `program`'s function, so they form one family
    that compiles once.
    """
    (state_name,) = context.invariant_bindings
    codes = context.state_action_space.states[state_name].tolist()
    programs = {}
    for start, code in enumerate(codes):
        name = f"{program.name}[{state_name}={int(code)}]"
        programs[name] = replace(
            program,
            name=name,
            requirements=replace(
                program.requirements,
                value_reads=rekeyed_value_reads(
                    reads=program.requirements.value_reads, core_key=name
                ),
            ),
            invariant_binding=InvariantBinding(
                state_name=state_name,
                start=start,
                code=int(code),
                family=program.name,
            ),
        )
    return MappingProxyType(programs)


def _action_partition_mesh(
    *,
    context: SolverBuildContext,
    action_streaming: _ActionStreamingDisposition,
) -> jax.sharding.Mesh | None:
    """Return the regime's mesh when its action product is shared, else `None`.

    Model construction admits a partition request only on the ordinary
    streamed singleton route without folded processes or co-mapped states;
    reaching here with anything else is an internal planning error.
    """
    if context.action_partitions == 1:
        return None
    unserved = [
        reason
        for failed, reason in (
            (
                action_streaming is not _ActionStreamingDisposition.STREAMED,
                f"its actions are not streamed ({action_streaming.value})",
            ),
            (bool(context.fold_state_names), "it folds a process"),
            (bool(context.co_map_state_names), "it co-maps a sharded state"),
        )
        if failed
    ]
    if unserved:
        msg = (
            f"Regime {context.regime_name!r} cannot share its actions over "
            f"{context.action_partitions} devices: " + "; ".join(unserved) + ". "
            "Remove the regime from ExecutionConfig.action_partitions."
        )
        raise ExecutionPlanningError(msg)
    plan = _build_regime_sharding(
        grids=context.grids,
        sharded_state_names=context.sharded_state_names,
        devices=placed_devices_for_ids(submesh_device_ids=context.submesh_device_ids),
        action_partitions=context.action_partitions,
    )
    return cast("_RegimeSharding", plan).mesh


def _continuation_unread_state_names(
    *, Q_and_F: QAndFFunction, inner_state_names: tuple[StateName, ...]
) -> tuple[StateName, ...]:
    """Cell states the continuation of `Q_and_F` never reads, in cell order.

    Such a state enters `Q` through utility, feasibility and the aggregator
    only, so the continuation is constant along its axis. Mapping it outside the
    flat cell lets the continuation be computed once per remaining cell and
    broadcast along it. The result is empty when:

    - the kernel does not state what its continuation reads (terminal and
      collective kernels, or a dependency with variadic arguments);
    - the continuation reads every cell state, or none of them, so no state
      would remain to tile.
    """
    reads = getattr(Q_and_F, "continuation_reads", None)
    if reads is None:
        return ()
    unread = tuple(name for name in inner_state_names if name not in reads)
    return unread if 0 < len(unread) < len(inner_state_names) else ()


def _classify_action_streaming(
    *, context: SolverBuildContext
) -> _ActionStreamingDisposition:
    """Classify one solve route without conflating dense and unsupported cases."""
    action_extents = context.state_action_space.actions_grid_shapes
    if context.has_taste_shocks and context.stakeholders is not None:
        disposition = _ActionStreamingDisposition.UNSUPPORTED_COLLECTIVE_EV1
    elif context.has_taste_shocks and context.fold_state_names:
        disposition = _ActionStreamingDisposition.UNSUPPORTED_EV1_FOLD
    elif context.stakeholders is not None and context.fold_state_names:
        disposition = _ActionStreamingDisposition.UNSUPPORTED_COLLECTIVE_FOLD
    elif context.has_taste_shocks and not context.state_action_space.discrete_actions:
        disposition = (
            _ActionStreamingDisposition.UNSUPPORTED_EV1_WITHOUT_DISCRETE_ACTION
        )
    elif not context.state_action_space.action_names or math.prod(action_extents) <= 1:
        disposition = _ActionStreamingDisposition.DENSE_TRIVIAL_ACTION_PRODUCT
    elif context.co_map_state_names and (
        context.same_period_ref_regimes or context.edge_reference_regimes
    ):
        disposition = _ActionStreamingDisposition.DENSE_CO_MAP_REFERENCE_CHANNEL
    elif context.has_taste_shocks:
        disposition = _ActionStreamingDisposition.DENSE_EV1_NONCANONICAL
    elif context.stakeholders is not None:
        disposition = _ActionStreamingDisposition.DENSE_COLLECTIVE_RESOURCES
    else:
        disposition = _ActionStreamingDisposition.STREAMED
    return disposition


def _supports_action_streaming(*, context: SolverBuildContext) -> bool:
    """Return whether the classified route has a streamed solve program."""
    return (
        _classify_action_streaming(context=context)
        is _ActionStreamingDisposition.STREAMED
    )


def _edge_reference_regimes_for_targets(
    *,
    context: SolverBuildContext,
    target_regimes: tuple[RegimeName, ...],
) -> tuple[RegimeName, ...]:
    """Return only edge references read by targets reachable this period."""
    law = context.laws[context.regime_name]
    references: list[RegimeName] = []
    for target in target_regimes:
        edge = law.gated_edges.get(target)
        if edge is not None:
            references.extend(edge.reference_regimes(phases=("solve",)))
    return tuple(dict.fromkeys(references))


def _value_reads(
    *,
    regime_name: RegimeName,
    period: int,
    target_regimes: tuple[RegimeName, ...],
    same_period_ref_regimes: tuple[RegimeName, ...],
    edge_reference_regimes: tuple[RegimeName, ...],
    edge_target_regimes: tuple[RegimeName, ...],
) -> tuple[ValueRead, ...]:
    """Declare every stored value leaf read by one GridSearch program."""
    reads: list[ValueRead] = []
    for target_regime in target_regimes:
        target = (
            ValueArtifactAddress(
                kind=ValueArtifactKind.GATED_CONTINUATION,
                period=period + 1,
                regime=regime_name,
                target_regime=target_regime,
            )
            if target_regime in edge_target_regimes
            else ValueArtifactAddress(
                kind=ValueArtifactKind.REGIME_VALUE,
                period=period + 1,
                regime=target_regime,
            )
        )
        reads.append(
            _value_read(
                regime_name=regime_name,
                period=period,
                target=target,
                channel=ValueInputChannel.NEXT_REGIME_VALUE,
                path=(target_regime,),
            )
        )
    reads.extend(
        _value_read(
            regime_name=regime_name,
            period=period,
            target=ValueArtifactAddress(
                kind=ValueArtifactKind.REGIME_VALUE,
                period=period,
                regime=reference_regime,
            ),
            channel=ValueInputChannel.SAME_PERIOD_VALUE,
            path=(reference_regime,),
        )
        for reference_regime in same_period_ref_regimes
    )
    reads.extend(
        _value_read(
            regime_name=regime_name,
            period=period,
            target=ValueArtifactAddress(
                kind=ValueArtifactKind.REGIME_VALUE,
                period=period + 1,
                regime=reference_regime,
            ),
            channel=ValueInputChannel.EDGE_REFERENCE_VALUE,
            path=(reference_regime,),
        )
        for reference_regime in edge_reference_regimes
    )
    return tuple(reads)


def _value_read(
    *,
    regime_name: RegimeName,
    period: int,
    target: ValueArtifactAddress,
    channel: ValueInputChannel,
    path: tuple[str | int, ...],
) -> ValueRead:
    """Pair one logical target artifact with its exact program-argument leaf."""
    return ValueRead(
        target=target,
        source=ValueConsumerAddress(
            source_period=period,
            source_regime=regime_name,
            core_key="main",
            channel=channel,
            path=path,
        ),
    )


@dataclass(frozen=True, kw_only=True)
class _GridSearchArgumentBuilder:
    """Build the one GridSearch program's arguments for lowering and execution."""

    regime_name: RegimeName
    same_period_ref_regimes: tuple[RegimeName, ...] = ()
    edge_reference_regimes: tuple[RegimeName, ...] = ()
    edge_target_regimes: tuple[RegimeName, ...] = ()

    def __call__(self, context: CoreBuildContext) -> Mapping[ReferenceName, QAndFArg]:
        """Return the exact kwargs shared by lowering and the runtime call."""
        state_action_space = cast("StateActionSpace", context.state_action_space)
        next_regime_to_V_arr = cast(
            "Mapping[RegimeName, FloatND]", context.next_regime_to_V_arr
        )
        flat_params = cast("FlatParams", context.flat_params)
        ages = cast("TimeAxis", context.ages)
        raw_next_regime_to_V_arr = next_regime_to_V_arr
        next_regime_to_V_arr = self._with_edge_substitution(
            next_regime_to_V_arr=next_regime_to_V_arr,
            edge_regime_to_V_arr=cast(
                "Mapping[RegimeName, FloatND] | None",
                context.edge_regime_to_V_arr,
            ),
        )
        arguments: dict[ReferenceName, QAndFArg] = {
            **dict(state_action_space.states),
            **dict(state_action_space.actions),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **dict(regime_kernel_params(flat_params, regime_name=self.regime_name)),
            "period": jnp.int32(context.period),
            "age": ages.values[context.period],
        }
        if self.same_period_ref_regimes:
            reference_values = (
                MappingProxyType(
                    {
                        name: raw_next_regime_to_V_arr[name]
                        for name in self.same_period_ref_regimes
                    }
                )
                if context.same_period_regime_to_V_arr is None
                else cast(
                    "Mapping[RegimeName, FloatND]",
                    context.same_period_regime_to_V_arr,
                )
            )
            arguments["same_period_regime_to_V_arr"] = reference_values
            arguments["same_period_regime_to_params"] = self._same_period_params(
                flat_params=flat_params
            )
        arguments.update(
            self._edge_reference_args(
                next_regime_to_V_arr=raw_next_regime_to_V_arr,
                flat_params=flat_params,
            )
        )
        return MappingProxyType(arguments)

    def _with_edge_substitution(
        self,
        *,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        edge_regime_to_V_arr: Mapping[RegimeName, FloatND] | None,
    ) -> Mapping[RegimeName, FloatND]:
        """Replace gated targets' raw values with their continuation objects."""
        if not self.edge_target_regimes:
            return next_regime_to_V_arr
        if edge_regime_to_V_arr is None:
            msg = (
                f"Regime '{self.regime_name}' declares gated edges into "
                f"{self.edge_target_regimes} but the solve loop passed no edge "
                "continuation arrays."
            )
            raise RuntimeError(msg)
        return MappingProxyType(
            {
                name: (
                    edge_regime_to_V_arr[name]
                    if name in self.edge_target_regimes
                    else arr
                )
                for name, arr in next_regime_to_V_arr.items()
            }
        )

    def _edge_reference_args(
        self,
        *,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        flat_params: FlatParams,
    ) -> dict[
        ReferenceName,
        MappingProxyType[RegimeName, FloatND]
        | MappingProxyType[RegimeName, FlatRegimeParams],
    ]:
        """Build the edge-reference value and parameter channels."""
        if not self.edge_reference_regimes:
            return {}
        return {
            "edge_reference_regime_to_V_arr": MappingProxyType(
                {
                    name: next_regime_to_V_arr[name]
                    for name in self.edge_reference_regimes
                }
            ),
            "edge_reference_regime_to_params": MappingProxyType(
                {
                    name: regime_kernel_params(flat_params, regime_name=name)
                    for name in self.edge_reference_regimes
                }
            ),
        }

    def _same_period_params(
        self, *, flat_params: FlatParams
    ) -> MappingProxyType[RegimeName, FlatRegimeParams]:
        """Return each same-period reference regime's own flat parameters."""
        return MappingProxyType(
            {
                name: regime_kernel_params(flat_params, regime_name=name)
                for name in self.same_period_ref_regimes
            }
        )


@dataclass(frozen=True, kw_only=True)
class _GridSearchPeriodKernel:
    """One period adapter whose native program graph is its sole core authority."""

    _core_programs: Mapping[str, CoreProgram]
    """The immutable GridSearch program graph: `main`, or one program per code."""

    def __post_init__(self) -> None:
        """Snapshot and require the one mathematical GridSearch core family."""
        programs = MappingProxyType(dict(self._core_programs))
        blocked = bool(programs) and all(
            program.invariant_binding is not None
            and program.invariant_binding.family == "main"
            for program in programs.values()
        )
        if tuple(programs) != ("main",) and not blocked:
            msg = (
                "GridSearch requires exactly one core program named 'main', or "
                "one bound program per code of an invariant state."
            )
            raise ValueError(msg)
        object.__setattr__(self, "_core_programs", programs)

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Return the sole native declaration used by eager, JIT, and AOT paths."""
        return self._core_programs

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> _GridSearchPeriodKernel:
        """Bind the regime's fixed params into the core.

        The core threads its `**kwargs` into the per-combo pool, so binding the
        regime's own fixed params restores the values removed from the live
        `flat_params`; the captured functions read only the keys they need.
        """
        program = next(iter(self._core_programs.values()))
        argument_builder = cast("_GridSearchArgumentBuilder", program.argument_builder)
        regime_fixed = dict(
            regime_kernel_params(
                fixed_flat_params, regime_name=argument_builder.regime_name
            )
        )
        if not regime_fixed:
            return self
        bound_program = replace(
            program,
            function=functools.partial(program.function, **regime_fixed),
        )
        # Programs of one family share one function, so they share one binding.
        return replace(
            self,
            _core_programs=MappingProxyType(
                {
                    name: replace(member, function=bound_program.function)
                    for name, member in self._core_programs.items()
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
        ages: TimeAxis,
        logger: logging.Logger,  # noqa: ARG002
        same_period_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
        edge_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
    ) -> KernelOutput:
        """Evaluate the grid search and assemble the `KernelOutput`.

        `same_period_regime_to_V_arr` is passed by the solve loop only for a
        regime declaring `same_period_refs`; `edge_regime_to_V_arr` only for
        a regime with gated edges (substituted into
        `next_regime_to_V_arr` before the core call). Every other kernel keeps
        the uniform `PeriodKernel` call signature.

        A regime solved one invariant code at a time is handed exactly one of
        its bound programs per call and evaluates it on that code's block of
        the state grid.
        """
        (core_key,) = (name for name in compiled_cores if name in self._core_programs)
        program = self._core_programs[core_key]
        if program.invariant_binding is not None:
            state_action_space = block_state_action_space(
                space=state_action_space, binding=program.invariant_binding
            )
        argument_builder = cast("_GridSearchArgumentBuilder", program.argument_builder)
        if (
            argument_builder.same_period_ref_regimes
            and same_period_regime_to_V_arr is None
        ):
            msg = (
                f"Regime '{argument_builder.regime_name}' declares same_period_refs "
                f"on {argument_builder.same_period_ref_regimes} but the solve loop "
                "passed no same-period V arrays."
            )
            raise RuntimeError(msg)
        arguments = program.argument_builder(
            CoreBuildContext(
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                next_regime_to_continuation=next_regime_to_continuation,
                flat_params=flat_params,
                period=period,
                ages=ages,
                edge_regime_to_V_arr=edge_regime_to_V_arr,
                same_period_regime_to_V_arr=same_period_regime_to_V_arr,
            )
        )
        out = compiled_cores[core_key](**arguments)
        if program.output_roles == (VALUE, DISSOLUTION_FLAG):
            V_arr, dissolution = out
            return KernelOutput(
                value=V_arr,
                solve_time_artifacts={DISSOLUTION_FLAG_ARTIFACT: dissolution},
            )
        return KernelOutput(value=out)
