"""Private conditional DC-EGM points and explicit current-law economic tangents.

Candidate rows are invocation-local. This slice installs neither projection nor
native replay and does not open the public current-IID solver admission guard.
"""

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp

from _lcm.certainty_equivalent import aggregates_nonlinearly
from _lcm.current_iid import AdmittedGroupParams, CurrentIIDPlan, NodeOperands
from _lcm.egm.asset_row import (
    _ConstructedAssetNode,
    _continuation_of_euler_state,
    _publish_node_selection,
    _SolveOneNode,
)
from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_ce import EconomicCEContinuation
from _lcm.egm.current_iid_economic import read_case_u_continuation
from _lcm.egm.current_iid_evaluation import CurrentResourcesEvaluation
from _lcm.egm.current_iid_resource_query import OwnedResourcesRead
from _lcm.egm.preferences import BoundUtilityOfAction
from _lcm.egm.projected_continuation import ProjectedEGMCarry
from _lcm.egm.step_core import ResourcesOfState, _EgmKernelPieces
from _lcm.egm.upper_envelope import OwnedQueryBracket
from _lcm.egm.upper_envelope.segment_envelope import OwnedEnvelope
from _lcm.probability import is_represented_zero
from _lcm.solution.feasibility import (
    BoundedResult,
    ContinuationReadCounts,
    QueryInputStatus,
    merge_continuation_read_counts,
    reduce_bound_input_status,
    reduce_continuation_read_counts,
    reduce_query_input_status,
)
from _lcm.typing import EconFunctionKwargs, QAndFArg
from _lcm.zero_safe import factored_weighted_term
from lcm.exceptions import UnsupportedOperationError
from lcm.shocks import Discrete, ShockName
from lcm.typing import (
    FloatND,
    IntND,
    ReferenceName,
    RegimeName,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)


@dataclass(frozen=True, kw_only=True)
class CaseUCurrentNodes:
    """Keep actual fixed-base law values and their numeric state tangents."""

    physical: Mapping[ShockName, FloatND | IntND]
    """Original joint physical rows, including canonical categorical codes."""
    physical_tangent: Mapping[ShockName, FloatND]
    """Continuous physical derivatives at fixed named quadrature bases."""
    factors: FloatND
    """Original factor-by-node probability bank, without premature products."""
    factor_tangent: FloatND
    """Derivative of each original probability factor before liveness masking."""
    law_valid: ScalarBool
    """Unconditional original law/support integrity, separate from economics."""


def realise_case_u_nodes(
    *,
    plan: CurrentIIDPlan,
    liquid: ScalarFloat,
    liquid_state: ReferenceName,
    arguments: Mapping[ReferenceName, QAndFArg],
    group_params: AdmittedGroupParams,
    node_operands: NodeOperands,
    indices: IntND,
) -> CaseUCurrentNodes:
    """Differentiate physical transforms and probabilities, never category draws."""
    floating = tuple(
        name
        for _, binding, _ in plan.bindings
        for name, law in binding.group.shocks.items()
        if not isinstance(law, Discrete)
    )

    def numeric(
        state: ScalarFloat,
    ) -> tuple[
        tuple[Mapping[ShockName, FloatND], FloatND],
        tuple[Mapping[ShockName, FloatND | IntND], ScalarBool],
    ]:
        physical, factors, valid = plan.realise_nodes(
            arguments={**arguments, liquid_state: state},
            group_params=group_params,
            node_operands=node_operands,
            indices=indices,
        )
        return (
            {name: physical[name] for name in floating},
            factors,
        ), (physical, valid)

    (_, factors), (physical_tangent, factor_tangent), (physical, law_valid) = jax.jvp(
        numeric, (liquid,), (jnp.ones_like(liquid),), has_aux=True
    )
    return CaseUCurrentNodes(
        physical=physical,
        physical_tangent=physical_tangent,
        factors=factors,
        factor_tangent=factor_tangent,
        law_valid=law_valid,
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class CaseUPointSolver:
    """Reuse the actual scalar asset-node candidate construction for one frame."""

    pieces: _EgmKernelPieces
    """Original canonical utility, resources, floor and future expectation."""
    pool: EconFunctionKwargs
    """Dynamic parameters and resolved source clock, without current physicals."""
    children: MappingProxyType[RegimeName, EGMCarry | ProjectedEGMCarry]
    """Original shock-free future artifacts consumed by this point construction."""
    savings_point_width: int | None = None
    """Existing savings-stage dispatch width, not a new numerical grid."""
    stochastic_node_width: int | None = None
    """Existing future-lottery dispatch width."""
    envelope_cell_width: int = 1
    """Existing exact-envelope workspace width."""

    def __post_init__(self) -> None:
        if (
            self.pieces.combo_names
            or self.pieces.query_spec is not None
            or self.pieces.lower_func is not None
        ):
            raise UnsupportedOperationError(
                "The private Case-U point has not installed combo aggregation, "
                "source queries or moving floors."
            )
        if (
            self.pieces.continuation_plan.risk_aversion_param_name is not None
            and not aggregates_nonlinearly(
                self.pieces.continuation_plan.certainty_equivalent
            )
        ):
            raise UnsupportedOperationError(
                "Case-U nonlinear CE requires its actual declared CE identity."
            )
        if not self.pieces.check_query_construction:
            raise TypeError("Case-U points require original owned construction status.")

    def prepare_node(
        self,
        *,
        liquid: ScalarFloat,
        physical: Mapping[ShockName, ScalarFloat | ScalarInt],
        physical_tangent: Mapping[ShockName, ScalarFloat],
        owned_refine: Callable[..., OwnedEnvelope] | None = None,
    ) -> tuple[
        _ConstructedAssetNode,
        ScalarFloat,
        ScalarFloat,
        EconomicCEContinuation | None,
        EconFunctionKwargs,
        EconFunctionKwargs,
        Mapping[ReferenceName, ScalarFloat],
    ]:
        """Expose the original construction for conditional topology."""
        pieces = self.pieces
        pool = {**self.pool, **physical}
        context = {**pool, pieces.euler_state_name: liquid}
        direction = {pieces.euler_state_name: jnp.ones_like(liquid), **physical_tangent}
        names = tuple(direction)

        def read_discount(values: tuple[ScalarFloat, ...]) -> ScalarFloat:
            (discount,) = tuple(
                pieces.build_W_kwargs(
                    {**context, **dict(zip(names, values, strict=True))}
                ).values()
            )
            return cast("ScalarFloat", discount)

        discount, discount_direction = jax.jvp(
            read_discount,
            (tuple(context[name] for name in names),),
            (tuple(direction[name] for name in names),),
        )
        economic_ce = (
            EconomicCEContinuation(
                power_mean_owner=self,
                plan=pieces.continuation_plan,
                pool=context,
                direction=direction,
                children=self.children,
                dtype=liquid.dtype,
                stochastic_node_width=self.stochastic_node_width,
                collect_read_counts=pieces.collect_continuation_read_counts,
            )
            if aggregates_nonlinearly(pieces.continuation_plan.certainty_equivalent)
            else None
        )
        resources = ResourcesOfState(
            resources_func=pieces.own_resources_func,
            euler_state_name=pieces.euler_state_name,
            bound=pool,
        )
        construct = _SolveOneNode(
            pieces=pieces,
            combo_pool=pool,
            discount_factor=discount,
            next_regime_to_continuation=self.children,
            dtype=liquid.dtype,
            stochastic_node_width=self.stochastic_node_width,
            envelope_cell_width=self.envelope_cell_width,
            resolved_process_grids=MappingProxyType({}),
            savings_point_width=self.savings_point_width,
            own_resources_of_state=resources,
            economic_continuation=economic_ce.savings
            if economic_ce is not None
            else None,
            continuation_of_euler_state=functools.partial(
                _continuation_of_euler_state,
                pieces=pieces,
                combo_pool=pool,
                next_regime_to_continuation=self.children,
                dtype=liquid.dtype,
                stochastic_node_width=self.stochastic_node_width,
                resolved_process_grids=MappingProxyType({}),
            ),
        ).construct(node_value=liquid, owned_refine=owned_refine)
        return (
            construct,
            discount,
            discount_direction,
            economic_ce,
            pool,
            context,
            direction,
        )

    def node(
        self,
        *,
        liquid: ScalarFloat,
        physical: Mapping[ShockName, ScalarFloat | ScalarInt],
        physical_tangent: Mapping[ShockName, ScalarFloat],
    ) -> BoundedResult[OwnedResourcesRead]:
        """Select original candidates and compute the separate economic envelope."""
        pieces = self.pieces
        (
            construct,
            discount,
            discount_direction,
            economic_ce,
            pool,
            context,
            direction,
        ) = self.prepare_node(
            liquid=liquid, physical=physical, physical_tangent=physical_tangent
        )
        bracket = construct.bracket
        if not isinstance(bracket, OwnedQueryBracket):
            raise TypeError("Case-U selection requires the original owned bracket.")
        value, policy, corner = _publish_node_selection(
            bracket=bracket.bracket,
            n_pad=pieces.n_pad,
            resources_at_node=construct.resources_at_node,
            borrowing_limit=construct.pieces.borrowing_limit,
            utility_of_action=construct.utility_of_action,
            discounted_expected_value_at_limit=discount * construct.expected_values[0],
        )
        # Saving is the original selected saving (the closed-form floor at a
        # corner), held fixed only in the economic envelope derivative, never
        # through candidate construction. Rebuilding it as m-c would lose the
        # exact savings node a rounded endogenous point m=a+c came from.
        saving = jnp.where(corner, construct.pieces.borrowing_limit, bracket.savings)
        floating = {name: physical[name] for name in physical_tangent}

        # keyword-only-exempt: library-callback=jax.jvp
        def current_utility(
            state: ScalarFloat, draw: Mapping[ShockName, ScalarFloat]
        ) -> ScalarFloat:
            context = {**self.pool, **physical, **draw, pieces.euler_state_name: state}
            cash = ResourcesOfState(
                resources_func=pieces.own_resources_func,
                euler_state_name=pieces.euler_state_name,
                bound={
                    name: value
                    for name, value in context.items()
                    if name != pieces.euler_state_name
                },
            )(state)
            utility = BoundUtilityOfAction(
                utility_func=pieces.utility_func,
                action_name=pieces.action_name,
                bound=context,
            )
            return utility(cash - saving)

        _, marginal = jax.jvp(
            current_utility,
            (liquid, floating),
            (jnp.ones_like(liquid), physical_tangent),
        )
        future = (
            economic_ce.source(saving)
            if economic_ce is not None
            else read_case_u_continuation(
                plan=pieces.continuation_plan,
                combo_pool=context,
                direction=direction,
                children=self.children,
                savings=saving,
                dtype=liquid.dtype,
                collect_read_counts=pieces.collect_continuation_read_counts,
                stochastic_node_width=self.stochastic_node_width,
            )
        )
        future_value, future_direction = future.output
        marginal += factored_weighted_term(
            factors=jnp.stack((discount,)), values=future_direction
        ) + factored_weighted_term(
            factors=jnp.stack((discount_direction,)), values=future_value
        )
        feasible = jnp.isfinite(value) | jnp.isneginf(value)
        feasible &= policy > 0.0
        if pieces.feasibility_func is not None:
            feasible &= pieces.feasibility_func(
                **{**pool, pieces.euler_state_name: liquid}
            )
        valid = (
            ~bracket.construction_invalid
            & jnp.isfinite(saving)
            & jnp.isfinite(policy)
            & jnp.isfinite(marginal)
            & (saving >= construct.pieces.borrowing_limit)
            & (saving <= construct.pieces.savings_nodes[-1])
            & (corner | bracket.read_supported)
            & (jnp.isfinite(value) | jnp.isneginf(value))
            & jnp.isfinite(discount)
            & jnp.isfinite(discount_direction)
        )
        for child_status in future.query_status or ():
            valid &= ~jnp.any(jnp.stack(child_status))
        if economic_ce is not None:
            for ce_status in construct.child_status or ():
                valid &= ~jnp.any(jnp.stack(ce_status))
        status = construct.child_status
        inactive = jnp.zeros((), dtype=jnp.bool_)
        return BoundedResult(
            output=OwnedResourcesRead(
                value=jnp.where(feasible, value, -jnp.inf),
                marginal=jnp.where(feasible & jnp.isfinite(value), marginal, 0.0),
                policy=policy,
                savings=saving,
                native_owner=jnp.where(corner, -1, bracket.owner_id),
                side_id=jnp.where(corner, 0, bracket.side_id),
                corner=corner,
                feasible=feasible,
                valid=valid,
            ),
            bound_status=()
            if construct.pieces.floor_status is None
            else (construct.pieces.floor_status,),
            query_status=(
                *(status or ()),
                *(future.query_status or ()),
                QueryInputStatus(inactive, inactive, ~valid),
            ),
            continuation_read_counts=merge_continuation_read_counts(
                left=construct.read_counts, right=future.continuation_read_counts
            ),
        )

    def integrate(
        self, *, liquid: ScalarFloat, nodes: CaseUCurrentNodes
    ) -> BoundedResult[CurrentResourcesEvaluation]:
        """Fold explicit value and full weight/context marginal channels."""
        factors, tangents = nodes.factors, nodes.factor_tangent
        live = jnp.all(~is_represented_zero(factors), axis=0)
        derivative_needed = jnp.zeros_like(live)
        for column in range(factors.shape[0]):
            others = jnp.concatenate((factors[:column], factors[column + 1 :]), axis=0)
            derivative_needed |= ~is_represented_zero(tangents[column]) & jnp.all(
                ~is_represented_zero(others), axis=0
            )
        needed = live | derivative_needed
        zero = jnp.zeros_like(liquid)
        no = jnp.zeros((), dtype=jnp.bool_)
        neutral = BoundedResult(
            output=OwnedResourcesRead(
                value=zero,
                marginal=zero,
                policy=zero,
                savings=zero,
                native_owner=jnp.int32(-1),
                side_id=jnp.int32(0),
                corner=no,
                feasible=no,
                valid=~no,
            ),
            bound_status=()
            if self.pieces.floor_status is None
            else (self.pieces.floor_status,),
            query_status=tuple(
                QueryInputStatus(no, no, no)
                for _ in range(
                    2 * len(self.pieces.continuation_plan.query_targets)
                    + 2
                    + int(
                        aggregates_nonlinearly(
                            self.pieces.continuation_plan.certainty_equivalent
                        )
                    )
                )
            ),
            continuation_read_counts=ContinuationReadCounts(jnp.int32(0), jnp.int32(0))
            if self.pieces.collect_continuation_read_counts
            else None,
        )

        def select(
            inputs: tuple[
                Mapping[ShockName, ScalarFloat | ScalarInt],
                Mapping[ShockName, ScalarFloat],
                ScalarBool,
            ],
        ) -> BoundedResult[OwnedResourcesRead]:
            physical, tangent, active = inputs
            return jax.lax.cond(
                active,
                lambda: self.node(
                    liquid=liquid, physical=physical, physical_tangent=tangent
                ),
                lambda: neutral,
            )

        selected = jax.lax.map(select, (nodes.physical, nodes.physical_tangent, needed))
        row = selected.output
        value_terms = factored_weighted_term(factors=factors, values=row.value)
        marginal_terms = factored_weighted_term(factors=factors, values=row.marginal)
        for column in range(factors.shape[0]):
            differentiated = jnp.concatenate(
                (
                    factors[:column],
                    tangents[column : column + 1],
                    factors[column + 1 :],
                ),
                axis=0,
            )
            marginal_terms += factored_weighted_term(
                factors=differentiated, values=row.value
            )
        value = jnp.sum(value_terms)
        raw_marginal = jnp.sum(marginal_terms)
        economic = jnp.any(live & jnp.isneginf(row.value))
        empty = jnp.any(live & ~row.feasible)
        valid_terms = jnp.all(
            ~live
            | (jnp.isfinite(row.value) & jnp.isfinite(value_terms))
            | (jnp.isneginf(row.value) & jnp.isneginf(value_terms))
        )
        physical_valid = jnp.ones((), dtype=jnp.bool_)
        for tangent in nodes.physical_tangent.values():
            physical_valid &= jnp.all(jnp.isfinite(tangent))
        valid = (
            nodes.law_valid
            & jnp.all(~needed | row.valid)
            & jnp.all(jnp.isfinite(tangents))
            & physical_valid
            & valid_terms
            & (jnp.isfinite(value) | (economic & jnp.isneginf(value)))
            & jnp.all(jnp.isfinite(marginal_terms))
            & jnp.isfinite(raw_marginal)
            & jnp.all(~derivative_needed | jnp.isfinite(row.value))
        )
        statuses = reduce_query_input_status(
            statuses=selected.query_status, active=needed
        )
        if statuses is None:
            raise TypeError("Case-U integration requires its original source slot.")
        own = statuses[-1]
        return BoundedResult(
            output=CurrentResourcesEvaluation(
                value, jnp.where(empty | economic, 0.0, raw_marginal), empty, valid
            ),
            bound_status=reduce_bound_input_status(
                statuses=selected.bound_status, active=needed
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
                counts=selected.continuation_read_counts, active=needed
            ),
        )
