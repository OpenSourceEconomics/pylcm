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

import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.constraints.routes import (
    ConstraintRoute,
    ConstraintRouteKey,
    ConstraintSite,
)
from _lcm.continuation import EGMContinuationLayout
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    StreamableProductAxis,
    _TargetValueAccess,
)
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
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION
from _lcm.solution.contract import (
    ConstraintRouteContext,
    ContinuationPayload,
    PeriodKernel,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    simulation_route,
)
from _lcm.typing import (
    FlatParams,
    MaxQOverAFunction,
    RegimeName,
    StateName,
)
from lcm.ages import AgeGrid
from lcm.solver_api import DISSOLUTION_FLAG as DISSOLUTION_FLAG_ARTIFACT
from lcm.solver_api import KernelOutput
from lcm.typing import (
    FloatND,
)

_ACTION_AXIS_NAME = "action"
_ACTION_WIDTH_KEYWORD = "_lcm_action_block_width"
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
    occupied = set(_CORE_RUNTIME_ARG_NAMES)
    occupied.update(context.flat_param_names)
    occupied.update(context.state_action_space.action_names)
    occupied.update(context.state_action_space.state_names)
    for Q_and_F in context.Q_and_F_functions.values():
        occupied.update(inspect.signature(Q_and_F).parameters)
    if context.pareto_weights is not None:
        occupied.update(context.pareto_weights.param_names)

    candidate = _ACTION_WIDTH_KEYWORD
    suffix = 0
    while candidate in occupied:
        suffix += 1
        candidate = f"{_ACTION_WIDTH_KEYWORD}_{suffix}"
    return candidate


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class GridSearch(Solver):
    """Grid-search solver over the full state-action product (the default)."""

    @property
    def supports_transition_local_lotteries(self) -> bool:
        """Grid search enumerates transition-local lotteries inside Q."""
        return True

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
            get_max_Q_over_a,
            get_streaming_max_Q_over_a,
        )
        from _lcm.regime_building.processing import (  # noqa: PLC0415
            get_conditioned_fold_weights_by_code,
        )

        program_functions: dict[int, MaxQOverAFunction] = {}
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
        action_width_keyword = _select_action_width_keyword(context=context)
        action_names = context.state_action_space.action_names
        action_extents = context.state_action_space.actions_grid_shapes
        for period, Q_and_F in context.Q_and_F_functions.items():
            q_id = id(Q_and_F)
            if q_id not in program_functions:
                common_kwargs = {
                    "Q_and_F": Q_and_F,
                    "batch_sizes": {
                        name: grid.batch_size
                        for name, grid in context.grids.items()
                        if name in context.state_action_space.state_names
                    },
                    "action_names": action_names,
                    "state_names": context.state_action_space.state_names,
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
                streamable_axes=(
                    (
                        StreamableProductAxis(
                            name=_ACTION_AXIS_NAME,
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
                target_value_accesses=_target_value_accesses(
                    regime_name=context.regime_name,
                    period=period,
                    target_regimes=target_regimes,
                    same_period_ref_regimes=context.same_period_ref_regimes,
                    edge_reference_regimes=edge_reference_regimes,
                    edge_target_regimes=context.edge_target_regimes,
                ),
            )
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
                    if stream_actions
                    else CoreExecutionDisposition.DENSE
                ),
                disposition_reason=(None if stream_actions else action_streaming.value),
                donation_candidates=(),
            )
            result[period] = _GridSearchPeriodKernel(
                _core_programs=MappingProxyType({"main": program})
            )
        return SolutionKernels(period_kernels=MappingProxyType(result))


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
    source = context.user_regimes[context.regime_name]
    references: list[RegimeName] = []
    for target in target_regimes:
        edge = source.gated_edges.get(target)
        if edge is None:
            continue
        references.extend(ref.regime for ref in edge.gate_refs.values())
        references.extend(route.solve_fallback.regime for route in edge.legs.values())
    return tuple(dict.fromkeys(references))


def _target_value_accesses(
    *,
    regime_name: RegimeName,
    period: int,
    target_regimes: tuple[RegimeName, ...],
    same_period_ref_regimes: tuple[RegimeName, ...],
    edge_reference_regimes: tuple[RegimeName, ...],
    edge_target_regimes: tuple[RegimeName, ...],
) -> tuple[_TargetValueAccess, ...]:
    """Declare every stored value leaf read by one GridSearch program."""
    accesses: list[_TargetValueAccess] = []
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
        accesses.append(
            _target_value_access(
                regime_name=regime_name,
                period=period,
                target=target,
                channel=ValueInputChannel.NEXT_REGIME_VALUE,
                path=(target_regime,),
            )
        )
    accesses.extend(
        _target_value_access(
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
    accesses.extend(
        _target_value_access(
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
    return tuple(accesses)


def _target_value_access(
    *,
    regime_name: RegimeName,
    period: int,
    target: ValueArtifactAddress,
    channel: ValueInputChannel,
    path: tuple[str | int, ...],
) -> _TargetValueAccess:
    """Pair one logical target artifact with its exact program-argument leaf."""
    return _TargetValueAccess(
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

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
        """Return the exact kwargs shared by lowering and the runtime call."""
        state_action_space = cast("StateActionSpace", context.state_action_space)
        next_regime_to_V_arr = cast(
            "Mapping[RegimeName, FloatND]", context.next_regime_to_V_arr
        )
        flat_params = cast("FlatParams", context.flat_params)
        ages = cast("AgeGrid", context.ages)
        raw_next_regime_to_V_arr = next_regime_to_V_arr
        next_regime_to_V_arr = self._with_edge_substitution(
            next_regime_to_V_arr=next_regime_to_V_arr,
            edge_regime_to_V_arr=cast(
                "Mapping[RegimeName, FloatND] | None",
                context.edge_regime_to_V_arr,
            ),
        )
        arguments: dict[str, object] = {
            **dict(state_action_space.states),
            **dict(state_action_space.actions),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **dict(flat_params[self.regime_name]),
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
                else context.same_period_regime_to_V_arr
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
    ) -> dict[str, object]:
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
                {name: flat_params[name] for name in self.edge_reference_regimes}
            ),
        }

    def _same_period_params(
        self, *, flat_params: FlatParams
    ) -> MappingProxyType[RegimeName, Mapping[str, object]]:
        """Return each same-period reference regime's own flat parameters."""
        return MappingProxyType(
            {name: flat_params[name] for name in self.same_period_ref_regimes}
        )


@dataclass(frozen=True, kw_only=True)
class _GridSearchPeriodKernel:
    """One period adapter whose native program graph is its sole core authority."""

    _core_programs: Mapping[str, CoreProgram]
    """The immutable one-node GridSearch program graph."""

    def __post_init__(self) -> None:
        """Snapshot and require the one mathematical GridSearch core."""
        programs = MappingProxyType(dict(self._core_programs))
        if tuple(programs) != ("main",):
            msg = "GridSearch requires exactly one core program named 'main'."
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
        program = self._core_programs["main"]
        argument_builder = cast("_GridSearchArgumentBuilder", program.argument_builder)
        regime_fixed = dict(
            fixed_flat_params.get(
                argument_builder.regime_name,
                MappingProxyType({}),
            )
        )
        if not regime_fixed:
            return self
        bound_program = replace(
            program,
            function=functools.partial(program.function, **regime_fixed),
        )
        return replace(
            self,
            _core_programs=MappingProxyType({"main": bound_program}),
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
        logger: logging.Logger,  # noqa: ARG002
        same_period_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
        edge_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
    ) -> KernelOutput:
        """Evaluate the grid search and assemble the `KernelOutput`.

        `same_period_regime_to_V_arr` is passed by the solve loop only for a
        regime declaring `same_period_refs`; `edge_regime_to_V_arr` only for
        a regime declaring `gated_edges` (substituted into
        `next_regime_to_V_arr` before the core call). Every other kernel keeps
        the uniform `PeriodKernel` call signature.
        """
        program = self._core_programs["main"]
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
        out = compiled_cores["main"](**arguments)
        if program.output_roles == (VALUE, DISSOLUTION_FLAG):
            V_arr, dissolution = out
            return KernelOutput(
                value=V_arr,
                solve_time_artifacts={DISSOLUTION_FLAG_ARTIFACT: dissolution},
            )
        return KernelOutput(value=out)
