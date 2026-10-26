"""Integrate complete original finite NEGM choices after current information."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.continuation import ContinuationPayload
from _lcm.current_iid import AdmittedGroupParams, CurrentIIDPlan, NodeOperands
from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_evaluation import CurrentResourcesEvaluation
from _lcm.egm.current_iid_negm import NEGMCaseUPoint
from _lcm.execution.core_program import CoreBuildContext
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.probability import is_represented_zero
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    QueryInputStatus,
    reduce_bound_input_status,
    reduce_continuation_read_counts,
    reduce_query_input_status,
)
from _lcm.time import TimeAxis
from _lcm.typing import EconFunctionArg, FlatParams, PytreeValue
from _lcm.zero_safe import factored_weighted_term
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    IntND,
    ReferenceName,
    RegimeName,
    ScalarFloat,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class NEGMCaseUArguments:
    """Borrow actual source grids, original future and admitted current law data."""

    pieces: NEGMCaseUPoint
    """The genuine original nested source, including its unchanged future plans."""
    regime_name: RegimeName
    """Original source parameter namespace."""
    ages: TimeAxis
    """Original model clock object."""
    fixed_flat_params: FlatParams
    """Actual fixed parameters; numerical arrays remain explicit program inputs."""

    def prepare(
        self, *, context: CoreBuildContext, materializer: DatedGenerationMaterializer
    ) -> Mapping[ReferenceName, PytreeValue]:
        """Refuse the unimplemented cold producer before any array preparation."""
        del context, materializer
        raise UnsupportedOperationError(
            "Cold finite NEGM U admission remains required."
        )

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue]:
        point = self.pieces
        if context.period != point.period or context.ages is not self.ages:
            raise TypeError("Finite NEGM U arguments lost their original clock.")
        space = context.state_action_space
        if space is None or self.ages is not point.context.ages:
            raise TypeError("Finite NEGM U requires its actual call-owned source.")
        fixed = self.fixed_flat_params
        targets = point.owner.transition_target_names
        return MappingProxyType(
            {
                **space.states,
                "next_regime_to_continuation": MappingProxyType(
                    {point.target: context.next_regime_to_continuation[point.target]}
                ),
                "_lcm_current_iid": {
                    name: {**fixed.get(name, {}), **context.flat_params.get(name, {})}
                    for name, _binding, _nodes in point.plan.bindings
                },
                "_lcm_current_iid_nodes": point.plan.node_operands,
                "_lcm_current_iid_rows": point.plan.indices,
                "_lcm_case_u_savings_nodes": point.savings_nodes,
                "_lcm_case_u_borrowing_limit": point.borrowing_limit,
                "_lcm_case_u_floor_status": None,
                "_lcm_case_u_child_node_values": (),
                "_lcm_negm_source_grid": space.states[point.euler_state_name],
                "_lcm_negm_outer_nodes": point.outer_nodes,
                **union_fixed_params(
                    fixed_flat_params=fixed,
                    regime_name=self.regime_name,
                    transition_target_names=targets,
                ),
                **union_free_params(
                    flat_params=MappingProxyType(context.flat_params),
                    regime_name=self.regime_name,
                    transition_target_names=targets,
                ),
                "period": jnp.int32(point.period),
                "age": self.ages.values[point.period],
            }
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class NEGMCaseUProgram:
    """Current-node expectation of genuine complete nested original solutions."""

    pieces: NEGMCaseUPoint
    """Exact genuine original keeper, adjuster and outer-choice constructor."""
    plan: CurrentIIDPlan
    """The same complete original law as the dated source point."""
    period: int
    """The same original source date."""

    def __post_init__(self) -> None:
        if self.plan is not self.pieces.plan or self.period != self.pieces.period:
            raise TypeError("Finite NEGM U lost its original dated law owner.")

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
        _lcm_negm_source_grid: EconFunctionArg,
        _lcm_negm_outer_nodes: EconFunctionArg,
        **arguments: EconFunctionArg,
    ) -> tuple[BoundedResult[CurrentResourcesEvaluation], BoolND]:
        del _lcm_case_u_child_node_values
        if _lcm_case_u_floor_status is not None:
            raise TypeError("Finite NEGM U only admits its actual literal floor.")
        if (
            not isinstance(_lcm_negm_source_grid, jax.Array)
            or _lcm_negm_source_grid.ndim != 1
            or not isinstance(_lcm_negm_outer_nodes, jax.Array)
            or _lcm_negm_outer_nodes.ndim != 1
        ):
            raise TypeError("Finite NEGM U lost original source or outer support.")
        point = self.pieces.with_operands(
            source_grid=_lcm_negm_source_grid,
            savings_nodes=_lcm_case_u_savings_nodes,
            borrowing_limit=_lcm_case_u_borrowing_limit,
            outer_nodes=_lcm_negm_outer_nodes,
        )
        grid = arguments[point.euler_state_name]
        if not isinstance(grid, jax.Array) or grid.ndim != 1:
            raise TypeError("Finite NEGM U requires its actual source coordinates.")
        child = next_regime_to_continuation[point.target]
        if not isinstance(child, EGMCarry):
            raise TypeError("Finite NEGM U lost its genuine original future carry.")
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
            # Every original keeper, inner Euler row and declared outer node
            # sees this frame before any maximizing or expectation takes place.
            selected = jax.vmap(
                lambda frame: point.select(
                    liquid=liquid, physical=frame, pool=pool, children=children
                )
            )(physical)
            row, _outer_action, _outer_index = selected.output
            terms = factored_weighted_term(factors=factors, values=row.value)
            marginal = factored_weighted_term(factors=factors, values=row.marginal)
            live = jnp.all(~is_represented_zero(factors), axis=0)
            empty = jnp.any(live & ~row.feasible)
            value = jnp.sum(terms)
            valid = (
                law_valid
                & (jnp.asarray(arguments["period"]) == self.period)
                & jnp.all(~live | row.valid)
                & (jnp.isfinite(value) | jnp.isneginf(value))
                & jnp.all(jnp.isfinite(marginal))
            )
            statuses = reduce_query_input_status(
                statuses=selected.query_status, active=live
            )
            if statuses is None:
                raise TypeError("Finite NEGM U lost its original source statuses.")
            own = statuses[-1]
            return BoundedResult(
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
                continuation_read_counts=reduce_continuation_read_counts(
                    counts=selected.continuation_read_counts, active=live
                ),
            ), law_valid

        return jax.vmap(integrate)(grid)
