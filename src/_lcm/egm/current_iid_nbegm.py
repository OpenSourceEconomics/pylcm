"""Original scalar NB candidate construction for realised current shocks."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from _lcm.continuation import ContinuationPayload
from _lcm.current_iid import AdmittedGroupParams, CurrentIIDPlan, NodeOperands
from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import ContinuationPlan
from _lcm.egm.current_iid_evaluation import CurrentResourcesEvaluation
from _lcm.egm.current_iid_resource_query import OwnedResourcesRead
from _lcm.execution.core_program import CoreBuildContext
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.probability import is_represented_zero
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    QueryInputStatus,
    reduce_bound_input_status,
    reduce_query_input_status,
)
from _lcm.solution.nbegm import (
    _NBEGMCellContinuation,
    _NBEGMContinuousCore,
    _NBEGMSourcePoint,
    _NBEGMTiledCore,
)
from _lcm.solution.preparation_clock import prepare_case_u_clock
from _lcm.time import TimeAxis
from _lcm.typing import EconFunctionArg, EconFunctionKwargs, FlatParams, PytreeValue
from _lcm.zero_safe import factored_weighted_term
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import (
    ActionName,
    BoolND,
    Float1D,
    FloatND,
    IntND,
    ReferenceName,
    RegimeName,
    ScalarFloat,
    ScalarInt,
    StateName,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class NBCaseUPoint:
    """The actual original scalar NB producer, law and future declaration."""

    core: _NBEGMContinuousCore
    declared_law: Callable[..., tuple[Float1D, Float1D]]
    continuation_plan: ContinuationPlan
    euler_state_name: StateName
    action_name: ActionName
    target: RegimeName
    savings_nodes: Float1D
    borrowing_limit: ScalarFloat
    source_grid: Float1D

    def __post_init__(self) -> None:
        if type(self.core) is not _NBEGMContinuousCore or not (
            self.core.retain_query and self.core.retain_policy
        ):
            raise TypeError(
                "NB reconstruction requires the original retained producer."
            )
        if self.continuation_plan.query_targets:
            raise UnsupportedOperationError("Scalar NB U requires ordinary child rows.")

    def node(
        self,
        *,
        liquid: ScalarFloat,
        physical: EconFunctionKwargs,
        pool: EconFunctionKwargs,
        children: Mapping[RegimeName, EGMCarry],
    ) -> BoundedResult[OwnedResourcesRead]:
        """Invoke the same NB Euler/interval step and retain its associated tuple."""
        if not isinstance(self.core, _NBEGMContinuousCore):
            raise TypeError("NB U lost its original source constructor.")
        actual = {**pool, **physical}
        child = children[self.target]
        next_liquid, returns = self.declared_law(
            savings_grid=self.savings_nodes, **actual
        )
        invoke = cast(
            "Callable[..., _NBEGMSourcePoint | BoundedResult[_NBEGMSourcePoint]]",
            self.core.construct_point,
        )
        raw = invoke(
            liquid=self.source_grid,
            source_query=liquid[None],
            next_liquid_grid=child.endog_grid,
            next_value=child.value,
            next_marginal=child.marginal_utility,
            next_liquid=next_liquid,
            marginal_return=returns,
            effective_savings_grid=self.savings_nodes,
            boundary_savings_targets=jnp.zeros((0,), dtype=liquid.dtype),
            boundary_next_liquid=jnp.zeros((0,), dtype=liquid.dtype),
            source_bound_status=None,
            continuation_read=None,
            **actual,
        )
        if isinstance(raw, BoundedResult):
            original = raw.output
            bound = raw.bound_status
            statuses = raw.query_status or ()
        else:
            original = raw
            bound, statuses = (), ()
        if not isinstance(original, _NBEGMSourcePoint):
            raise TypeError("NB U lost its actual original point record.")
        value, carry, native_owner = original
        if native_owner is None:
            raise TypeError("NB U lost its original selected candidate identity.")
        data = carry.query_data_for_transport()
        if data is None or carry.policy is None:
            raise TypeError(
                "NB point lost original saving, policy or interval records."
            )
        probability = self.continuation_plan.compute_regime_transition_probs(
            **{**actual, self.euler_state_name: liquid}
        )[self.target]
        valid = (
            jnp.all(jnp.isfinite(next_liquid))
            & jnp.all(next_liquid[1:] > next_liquid[:-1])
            & jnp.all(jnp.isfinite(returns) & (returns > 0))
            & jnp.all(jnp.isfinite(child.value) | jnp.isneginf(child.value))
            & jnp.all(jnp.isfinite(child.marginal_utility))
            & jnp.isfinite(probability)
            & (probability == 1)
        )
        feasible = jnp.isfinite(value[0])
        valid &= feasible | jnp.isneginf(value[0])
        valid &= ~feasible | (
            jnp.isfinite(carry.policy[0])
            & jnp.isfinite(data.savings[0])
            & jnp.isfinite(carry.marginal_utility[0])
        )
        no = jnp.asarray(a=False)
        return BoundedResult(
            output=OwnedResourcesRead(
                value=value[0],
                marginal=jnp.where(feasible, carry.marginal_utility[0], 0.0),
                policy=carry.policy[0],
                savings=data.savings[0],
                native_owner=native_owner[0],
                # The first class publishes exact points, never side limits.
                side_id=jnp.int32(0),
                corner=data.savings[0] == self.savings_nodes[0],
                feasible=feasible,
                valid=valid,
            ),
            bound_status=bound,
            query_status=(*statuses, QueryInputStatus(no, no, ~valid)),
        )

    def with_savings(self, *, nodes: Float1D) -> NBCaseUPoint:
        """Keep original numeric saving operands explicit in forward dispatch."""
        return replace(self, savings_nodes=nodes, borrowing_limit=nodes[0])


class NBRideSelectedRead(NamedTuple):
    """Original native candidate channels and their selected discrete codes."""

    original: OwnedResourcesRead
    action_codes: IntND


@dataclass(frozen=True, kw_only=True, eq=False)
class NBCaseURidePoint:
    """A genuine ride-cell constructor, including branch-specific future reads."""

    core: _NBEGMTiledCore
    euler_state_name: StateName
    action_name: ActionName
    target: RegimeName
    savings_nodes: Float1D
    borrowing_limit: ScalarFloat
    source_grid: Float1D
    passive_names: tuple[StateName, ...]

    @property
    def continuation_plan(self) -> ContinuationPlan:
        return self.core.continuation_plan

    @property
    def discrete_action_names(self) -> tuple[ActionName, ...]:
        return tuple(name for name, _codes in self.core.schedule_spec.discrete_actions)

    def with_savings(self, *, nodes: Float1D) -> NBCaseURidePoint:
        return replace(self, savings_nodes=nodes, borrowing_limit=nodes[0])

    def node(
        self,
        *,
        liquid: ScalarFloat,
        physical: EconFunctionKwargs,
        pool: EconFunctionKwargs,
        children: Mapping[RegimeName, EGMCarry],
    ) -> BoundedResult[OwnedResourcesRead]:
        selected = self.select(
            liquid=liquid, physical=physical, pool=pool, children=children
        )
        return BoundedResult(
            output=selected.output.original,
            bound_status=selected.bound_status,
            query_status=selected.query_status,
            continuation_read_counts=selected.continuation_read_counts,
        )

    def select(
        self,
        *,
        liquid: ScalarFloat,
        physical: EconFunctionKwargs,
        pool: EconFunctionKwargs,
        children: Mapping[RegimeName, EGMCarry],
    ) -> BoundedResult[NBRideSelectedRead]:
        """Read the actual cell's distinct continuation classes before hard choice."""
        core = self.core
        if (
            type(core) is not _NBEGMTiledCore
            or core.statics.co_map_state_names
            or core.statics.continuation_reads_liquid
            or core.statics.has_jump
            or core.cliff_candidates
            or core.continuation_plan.query_targets
        ):
            raise UnsupportedOperationError(
                "NB current-U ride points require unsharded continuous "
                "ordinary-child cells without interval-dependent future reads."
            )
        kwargs = {
            **pool,
            **physical,
            self.euler_state_name: self.source_grid,
            **{
                name: jnp.asarray(codes, dtype=jnp.int32)
                for name, codes in core.schedule_spec.discrete_actions
            },
        }
        coordinates = []
        for name in core.statics.ride_names:
            coordinate = kwargs[name]
            if not isinstance(coordinate, jax.Array) or coordinate.ndim != 0:
                raise TypeError(
                    "NB ride point requires actual scalar cell coordinates."
                )
            coordinates.append(coordinate)
        ride_values = tuple(coordinates)
        solver = replace(
            core.envelope_build.bind_cell, savings_grid=self.savings_nodes
        )(kwargs)
        continuation = _NBEGMCellContinuation(
            kwargs=kwargs,
            carry=MappingProxyType(dict(children)),
            comap_bindings=MappingProxyType({}),
            savings_grid=self.savings_nodes,
            continuation_plan=core.continuation_plan,
            statics=core.statics,
            regime_name=core.regime_name,
            cliff_candidates=False,
            schedule_spec=core.schedule_spec,
        )(ride_values)
        bound = ()
        statuses = ()
        counts = None
        if isinstance(continuation, BoundedResult):
            bound = continuation.bound_status
            statuses = continuation.query_status or ()
            counts = continuation.continuation_read_counts
            continuation = continuation.output
        if not isinstance(continuation, tuple) or len(continuation) != len(
            ("value", "marginal")
        ):
            raise TypeError("NB ride point lost its original continuation pair.")
        value, marginal = continuation
        if not isinstance(value, jax.Array) or not isinstance(marginal, jax.Array):
            raise TypeError("NB ride point requires its original numerical rows.")
        selected = solver.construct_point(
            ride_values=ride_values,
            query=liquid[None],
            cont_value=value,
            cont_marginal=marginal,
        )
        feasible = jnp.isfinite(selected.value[0])
        valid = (
            (feasible | jnp.isneginf(selected.value[0]))
            & jnp.all(jnp.isfinite(value) | jnp.isneginf(value))
            & jnp.all(jnp.isfinite(marginal))
            & (
                ~feasible
                | (
                    jnp.isfinite(selected.policy[0])
                    & jnp.isfinite(selected.savings[0])
                    & jnp.isfinite(selected.marginal[0])
                    & (selected.native_owner[0] >= 0)
                )
            )
        )
        no = jnp.asarray(a=False)
        return BoundedResult(
            output=NBRideSelectedRead(
                original=OwnedResourcesRead(
                    value=selected.value[0],
                    marginal=jnp.where(feasible, selected.marginal[0], 0.0),
                    policy=selected.policy[0],
                    savings=selected.savings[0],
                    native_owner=selected.native_owner[0],
                    side_id=jnp.int32(0),
                    corner=selected.savings[0] == self.savings_nodes[0],
                    feasible=feasible,
                    valid=valid,
                ),
                action_codes=selected.action_codes[0],
            ),
            bound_status=bound,
            query_status=(*statuses, QueryInputStatus(no, no, ~valid)),
            continuation_read_counts=counts,
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class NBCaseUArguments:
    """Original NB parameter namespaces, dated support and numerical child inputs."""

    pieces: NBCaseUPoint | NBCaseURidePoint
    regime_name: RegimeName
    plan: CurrentIIDPlan
    period: int
    ages: TimeAxis
    transition_target_names: tuple[RegimeName, ...]
    fixed_flat_params: FlatParams
    clock_values: Mapping[ReferenceName, ScalarFloat | ScalarInt] | None = None
    """Original admitted date scalars, borrowed by later runtime argument builders."""

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue]:
        return self._arguments(
            context=context,
            clock=lambda: (
                {
                    "period": jnp.int32(self.period),
                    "age": self.ages.values[self.period],
                }
                if self.clock_values is None
                else self.clock_values
            ),
        )

    def _arguments(
        self,
        *,
        context: CoreBuildContext,
        clock: Callable[[], Mapping[ReferenceName, PytreeValue]],
    ) -> Mapping[ReferenceName, PytreeValue]:
        if context.period != self.period or context.ages is not self.ages:
            raise TypeError("NB U arguments lost their original clock.")
        if context.state_action_space is None:
            raise TypeError("NB U arguments require original resolved support.")
        fixed = self.fixed_flat_params
        return MappingProxyType(
            {
                self.pieces.euler_state_name: context.state_action_space.states[
                    self.pieces.euler_state_name
                ],
                **(
                    {
                        name: context.state_action_space.states[name]
                        for name in self.pieces.passive_names
                    }
                    if isinstance(self.pieces, NBCaseURidePoint)
                    else {}
                ),
                "next_regime_to_continuation": MappingProxyType(
                    {
                        self.pieces.target: context.next_regime_to_continuation[
                            self.pieces.target
                        ]
                    }
                ),
                "_lcm_current_iid": {
                    name: {**fixed.get(name, {}), **context.flat_params.get(name, {})}
                    for name, _binding, _nodes in self.plan.bindings
                },
                "_lcm_current_iid_nodes": self.plan.node_operands,
                "_lcm_current_iid_rows": self.plan.indices,
                "_lcm_case_u_savings_nodes": self.pieces.savings_nodes,
                "_lcm_case_u_borrowing_limit": self.pieces.borrowing_limit,
                "_lcm_case_u_floor_status": None,
                "_lcm_case_u_child_node_values": (),
                "_lcm_nbegm_source_grid": context.state_action_space.states[
                    self.pieces.euler_state_name
                ],
                **union_fixed_params(
                    fixed_flat_params=fixed,
                    regime_name=self.regime_name,
                    transition_target_names=self.transition_target_names,
                ),
                **union_free_params(
                    flat_params=MappingProxyType(context.flat_params),
                    regime_name=self.regime_name,
                    transition_target_names=self.transition_target_names,
                ),
                **clock(),
            }
        )

    def prepare(
        self, *, context: CoreBuildContext, materializer: DatedGenerationMaterializer
    ) -> Mapping[ReferenceName, PytreeValue]:
        return self._arguments(
            context=context,
            clock=lambda: (
                prepare_case_u_clock(
                    period=self.period, ages=self.ages, materializer=materializer
                )
                if self.clock_values is None
                else self.clock_values
            ),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class NBCaseUProgram:
    """Integrate original NB point results, never endogenous candidate rows."""

    pieces: NBCaseUPoint | NBCaseURidePoint
    plan: CurrentIIDPlan
    period: int

    def evaluate_with_law(
        self,
        *,
        next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
        _lcm_current_iid: AdmittedGroupParams,
        _lcm_current_iid_nodes: NodeOperands,
        _lcm_current_iid_rows: IntND,
        _lcm_case_u_savings_nodes: Float1D,
        _lcm_case_u_borrowing_limit: ScalarFloat,
        _lcm_case_u_floor_status: BoundInputStatus | None,
        _lcm_case_u_child_node_values: tuple[tuple[FloatND | IntND, ...], ...],
        _lcm_nbegm_source_grid: EconFunctionArg,
        **arguments: EconFunctionArg,
    ) -> tuple[BoundedResult[CurrentResourcesEvaluation], BoolND]:
        del _lcm_case_u_borrowing_limit, _lcm_case_u_child_node_values
        if _lcm_case_u_floor_status is not None:
            raise TypeError("Scalar NB U only admits the original literal floor.")
        if (
            not isinstance(_lcm_nbegm_source_grid, jax.Array)
            or _lcm_nbegm_source_grid.ndim != 1
        ):
            raise TypeError("NB U requires its original full interval support.")
        point = replace(
            self.pieces.with_savings(nodes=_lcm_case_u_savings_nodes),
            source_grid=_lcm_nbegm_source_grid,
        )
        grid = arguments[point.euler_state_name]
        if not isinstance(grid, jax.Array) or grid.ndim != 1:
            raise TypeError("NB U requires its actual scalar source support.")
        child = next_regime_to_continuation[point.target]
        if not isinstance(child, EGMCarry):
            raise TypeError("NB U requires its original ordinary child carry.")
        children = MappingProxyType({point.target: child})
        pool = {
            name: value
            for name, value in arguments.items()
            if name != point.euler_state_name
        }

        def integrate(
            liquid: ScalarFloat,
        ) -> tuple[BoundedResult[CurrentResourcesEvaluation], BoolND]:
            physical, factors, law_valid = self.plan.realise_nodes(
                arguments={**pool, point.euler_state_name: liquid},
                group_params=_lcm_current_iid,
                node_operands=_lcm_current_iid_nodes,
                indices=_lcm_current_iid_rows,
            )
            selected = jax.vmap(
                lambda frame: point.node(
                    liquid=liquid,
                    physical=frame,
                    pool=pool,
                    children=children,
                )
            )(physical)
            row = selected.output
            terms = factored_weighted_term(factors=factors, values=row.value)
            marginal = factored_weighted_term(factors=factors, values=row.marginal)
            live = jnp.all(~is_represented_zero(factors), axis=0)
            empty = jnp.any(live & ~row.feasible)
            value = jnp.sum(terms)
            valid = (
                law_valid
                & (jnp.asarray(arguments["period"]) == self.period)
                & jnp.all(~live | row.valid)
                & (jnp.isfinite(value) | (empty & jnp.isneginf(value)))
                & jnp.all(jnp.isfinite(marginal))
            )
            statuses = reduce_query_input_status(
                statuses=selected.query_status, active=live
            )
            if statuses is None:
                raise TypeError("NB U integration lost original source status.")
            own = statuses[-1]
            result = BoundedResult(
                output=CurrentResourcesEvaluation(
                    value,
                    jnp.where(empty, 0.0, jnp.sum(marginal)),
                    empty,
                    valid,
                ),
                bound_status=reduce_bound_input_status(
                    statuses=selected.bound_status, active=live
                ),
                query_status=(
                    *statuses[:-1],
                    QueryInputStatus(
                        own.nonfinite_floor,
                        own.invalid_shifted_grid,
                        own.coverage_missing | ~valid,
                    ),
                ),
            )
            return result, law_valid

        return jax.vmap(integrate)(grid)
