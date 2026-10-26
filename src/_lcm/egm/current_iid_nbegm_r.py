"""Solve-once original NB resource banks and their current-node economic reads."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import jax
import jax.numpy as jnp
from jax.interpreters.ad import JVPTracer

from _lcm.continuation import ContinuationPayload
from _lcm.current_iid import AdmittedGroupParams, CurrentIIDPlan, NodeOperands
from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import ContinuationPlan
from _lcm.egm.current_iid_balanced import _ordered_points
from _lcm.egm.current_iid_ce_nodes import needed_economic_nodes
from _lcm.egm.current_iid_evaluation import CurrentResourcesEvaluation
from _lcm.egm.current_iid_native import BoundResourcesMap
from _lcm.egm.current_iid_resource_query import OwnedResourcesRead
from _lcm.egm.nbegm_resources import NBEGMResourcesBank
from _lcm.egm.nbegm_resources_read import read_nbegm_resources_bank
from _lcm.egm.projected_continuation import ProjectedEGMCarry
from _lcm.egm.query_data import QueryNodeKind
from _lcm.probability import is_represented_zero
from _lcm.solution.feasibility import BoundedResult, QueryInputStatus
from _lcm.solution.nbegm import _NBEGMContinuousCore
from _lcm.typing import EconFunctionKwargs, HostArray, PytreeValue
from _lcm.zero_safe import factored_weighted_term
from lcm.exceptions import UnsupportedOperationError
from lcm.shocks import ShockName
from lcm.typing import (
    ActionName,
    BoolND,
    Float1D,
    FloatND,
    IntND,
    RegimeName,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
    StateName,
)

type NBCaseROperands = tuple[
    Float1D,
    Float1D,
    Mapping[RegimeName, ContinuationPayload],
    AdmittedGroupParams,
    NodeOperands,
    IntND,
    EconFunctionKwargs,
]


@dataclass(frozen=True, kw_only=True, eq=False)
class NBCaseRProgram:
    """Construct the genuine original NB bank once outside every read map."""

    core: _NBEGMContinuousCore
    """Exact original scalar producer and its envelope comparison arithmetic."""
    declared_law: Callable[..., tuple[Float1D, Float1D]]
    """Original savings-to-child landing and marginal-return producer."""
    continuation_plan: ContinuationPlan
    """Original ordinary-future and regime-probability declaration."""
    plan: CurrentIIDPlan
    """Actual independent-group joint rows and current-law bindings."""
    euler_state_name: StateName
    """Pre-shock scalar state whose economic marginal is published."""
    action_name: ActionName
    """Original consumption action, independent of its action-grid support."""
    target: RegimeName
    """Original ordinary terminal child."""
    coefficient: Fraction
    """Certified exact positive resource Jacobian with respect to pre-state."""
    period: int
    """Original source date."""
    capacity: int
    """Declared bounded exact-address projection allocation."""
    retain_resources: bool
    """Whether this variant exports the original bank for physical replay."""

    def __call__(
        self,
        *,
        liquid_grid: Float1D,
        savings_nodes: Float1D,
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        group_params: AdmittedGroupParams,
        node_operands: NodeOperands,
        indices: IntND,
        pool: EconFunctionKwargs,
    ) -> BoundedResult[tuple[PytreeValue, ...]]:
        """Read the shared original bank at every actual resource coordinate."""
        return _call_nbegm_r(
            self,
            (
                liquid_grid,
                savings_nodes,
                next_regime_to_continuation,
                group_params,
                node_operands,
                indices,
                pool,
            ),
        )

    def evaluate(  # noqa: C901 - original bank, primal statuses, and null-direction arithmetic share the same source graph
        self,
        *,
        operands: NBCaseROperands,
        required: BoolND | None = None,
        weight_direction: FloatND | None = None,
    ) -> BoundedResult[tuple[PytreeValue, ...]]:
        """Evaluate original operands under the required primal or tangent events."""
        (
            liquid_grid,
            savings_nodes,
            next_regime_to_continuation,
            group_params,
            node_operands,
            indices,
            pool,
        ) = operands
        child = next_regime_to_continuation[self.target]
        if not isinstance(child, EGMCarry):
            raise TypeError("NB R requires its original ordinary future carry.")
        physical, factors, law_valid = self.plan.realise_nodes(
            arguments=pool,
            group_params=group_params,
            node_operands=node_operands,
            indices=indices,
        )
        live = jnp.all(~is_represented_zero(factors), axis=0)
        read = live if required is None else live | required
        resources = BoundResourcesMap(
            function=self.core.schedule_spec.coh_of_liquid_dag,
            liquid_state=self.euler_state_name,
            pool=pool,
            names=(self.euler_state_name, *self.core.schedule_spec.coh_param_names),
        )

        def support_at_state(liquid: ScalarFloat) -> FloatND:
            def support_at_draw(
                frame: tuple[Mapping[ShockName, ScalarFloat | ScalarInt], ScalarBool],
            ) -> ScalarFloat:
                draw, active = frame
                return jax.lax.cond(
                    active,
                    lambda: resources(liquid, draw),
                    lambda: jnp.asarray(jnp.inf, dtype=liquid.dtype),
                )

            return jax.lax.map(support_at_draw, (physical, read))

        raw_support = jax.lax.map(support_at_state, liquid_grid)
        lower = jnp.min(raw_support)
        support = jnp.sort(jnp.where(read[None, :], raw_support, lower).reshape(-1))
        next_liquid, marginal_return = self.declared_law(
            savings_grid=savings_nodes, **pool
        )
        construct = cast(
            "Callable[..., NBEGMResourcesBank]", self.core.construct_resources_bank
        )
        bank = construct(
            resources_support=support,
            next_liquid_grid=child.endog_grid,
            next_value=child.value,
            next_marginal=child.marginal_utility,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            effective_savings_grid=savings_nodes,
            **pool,
        )
        probability = self.continuation_plan.compute_regime_transition_probs(**pool)[
            self.target
        ]
        resource_slope = jnp.asarray(float(self.coefficient), dtype=liquid_grid.dtype)
        input_valid = (
            law_valid
            & jnp.isfinite(probability)
            & (probability == 1)
            & (jnp.asarray(pool["period"]) == self.period)
            & jnp.all(jnp.isfinite(liquid_grid))
            & jnp.all(liquid_grid[1:] > liquid_grid[:-1])
            & jnp.isfinite(resource_slope)
            & (resource_slope > 0)
        )
        bank = bank._replace(
            construction_invalid=bank.construction_invalid | ~input_valid
        )

        def integrate(liquid: ScalarFloat) -> CurrentResourcesEvaluation:
            def select(
                draw: Mapping[ShockName, ScalarFloat | ScalarInt],
            ) -> OwnedResourcesRead:
                return read_nbegm_resources_bank(
                    bank=bank,
                    resources=resources(liquid, draw),
                    arithmetic=self.core.envelope_arithmetic,
                )

            # Resource evaluation is conditional on exact event liveness.
            neutral = OwnedResourcesRead(
                value=jnp.zeros_like(liquid),
                marginal=jnp.zeros_like(liquid),
                policy=jnp.zeros_like(liquid),
                savings=jnp.zeros_like(liquid),
                native_owner=jnp.int32(-1),
                side_id=jnp.int32(0),
                corner=jnp.asarray(a=False),
                feasible=jnp.asarray(a=False),
                valid=jnp.asarray(a=True),
            )

            def active_select(
                frame: tuple[Mapping[ShockName, ScalarFloat | ScalarInt], ScalarBool],
            ) -> OwnedResourcesRead:
                draw, active = frame
                return jax.lax.cond(active, lambda: select(draw), lambda: neutral)

            selected = jax.lax.map(active_select, (physical, read))
            values = (
                selected.value
                if weight_direction is None
                else jnp.where(live, 0.0, selected.value)
            )
            economic = selected.marginal * resource_slope
            if weight_direction is not None:
                economic = jnp.where(live, 0.0, economic)

            def weighted_value(weights: FloatND) -> ScalarFloat:
                return jnp.sum(factored_weighted_term(factors=weights, values=values))

            def weighted_marginal(weights: FloatND) -> ScalarFloat:
                return jnp.sum(
                    factored_weighted_term(
                        factors=weights,
                        values=economic,
                    )
                )

            if weight_direction is None:
                value = weighted_value(factors)
                marginal = weighted_marginal(factors)
            else:
                value = jax.jvp(weighted_value, (factors,), (weight_direction,))[1]
                marginal = jax.jvp(weighted_marginal, (factors,), (weight_direction,))[
                    1
                ]
            empty = jnp.any(live & ~selected.feasible)
            valid = (
                input_valid
                & ~bank.construction_invalid
                & jnp.all(~read | selected.valid)
                & jnp.all(
                    ~(read & ~live)
                    | (
                        selected.feasible
                        & jnp.isfinite(selected.value)
                        & jnp.isfinite(selected.marginal)
                    )
                )
                & (jnp.isfinite(value) | (empty & jnp.isneginf(value)))
                & jnp.isfinite(marginal)
            )
            return CurrentResourcesEvaluation(value, marginal, empty, valid)

        points, length, capacity_valid = _ordered_points(
            raw=liquid_grid,
            active=jnp.ones(liquid_grid.shape, dtype=jnp.bool_),
            capacity=self.capacity,
        )
        active = jnp.arange(self.capacity) < length
        grid = jnp.where(active, points, liquid_grid[0])
        source = jax.lax.map(integrate, grid)
        valid = input_valid & capacity_valid & jnp.all(~active | source.valid)
        projected = ProjectedEGMCarry(
            grid=grid,
            value=jnp.where(active, source.value, 0),
            economic_marginal=jnp.where(active, source.economic_marginal, 0),
            owner_id=jnp.where(active, jnp.arange(self.capacity), -1).astype(jnp.int32),
            side_id=jnp.zeros((self.capacity,), dtype=jnp.int32),
            node_kind=jnp.where(
                active,
                jnp.where(
                    jnp.isneginf(source.value),
                    QueryNodeKind.ECONOMIC_MINUS_INF,
                    QueryNodeKind.FINITE,
                ),
                QueryNodeKind.PAD,
            ).astype(jnp.int32),
            link_valid=jnp.zeros((self.capacity - 1,), dtype=jnp.bool_),
            valid_length=length,
            domain_lower=liquid_grid[0],
            domain_upper=liquid_grid[-1],
            valid=valid,
        )
        positions = jnp.clip(
            jnp.searchsorted(points, liquid_grid), 0, self.capacity - 1
        )
        no = jnp.asarray(a=False)
        return BoundedResult(
            output=(
                jnp.where(valid, source.value[positions], jnp.nan),
                projected,
                bank if self.retain_resources else None,
                (
                    jnp.sum(source.empty[positions], dtype=jnp.int32),
                    jnp.sum(
                        jnp.isinf(source.value[positions]) & ~source.empty[positions],
                        dtype=jnp.int32,
                    ),
                ),
                ~law_valid,
                bank.unsupported_economic_minus_inf,
            ),
            bound_status=(),
            query_status=(QueryInputStatus(no, no, ~valid),),
        )


# keyword-only-exempt: library-callback=jax.custom_jvp
def _call_nbegm_r_primal(
    program: NBCaseRProgram, operands: NBCaseROperands
) -> BoundedResult[tuple[PytreeValue, ...]]:
    return program.evaluate(operands=operands)


_call_nbegm_r = jax.custom_jvp(_call_nbegm_r_primal, nondiff_argnums=(0,))


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
@_call_nbegm_r.defjvp
def _call_nbegm_r_jvp(
    program: NBCaseRProgram,
    primals: tuple[NBCaseROperands],
    tangents: tuple[PytreeValue],
) -> tuple[
    BoundedResult[tuple[PytreeValue, ...]],
    BoundedResult[tuple[PytreeValue, ...]],
]:
    """Retain original first-order directions through finite null counterfactuals.

    shortcut: direction-dependent support qualifies first-order forward AD only;
    mixed higher-order requests need a derivative-order-aware support contract.
    Original certified native reads retain their reverse-mode refusal.
    """
    if any(
        isinstance(leaf, JVPTracer) for leaf in jax.tree.leaves((primals, tangents))
    ):
        raise UnsupportedOperationError(
            "Original NB Case R supports first-order forward sensitivities only."
        )
    operands = primals[0]

    def factors(actual: NBCaseROperands) -> FloatND:
        return program.plan.realise_nodes(
            arguments=actual[6],
            group_params=actual[3],
            node_operands=actual[4],
            indices=actual[5],
        )[1]

    original_factors, factor_direction = jax.jvp(factors, primals, tangents)
    required = needed_economic_nodes(
        factors=original_factors, tangents=factor_direction
    )

    def original(actual: NBCaseROperands) -> BoundedResult[tuple[PytreeValue, ...]]:
        return program.evaluate(operands=actual)

    primal, direction = jax.jvp(original, primals, tangents)
    live = jnp.all(~is_represented_zero(original_factors), axis=0)
    correction = program.evaluate(
        operands=operands,
        required=required,
        weight_direction=jnp.where(~live[None, :], factor_direction, 0.0),
    )
    status = correction.query_status
    if status is None:
        raise TypeError("NB R null-event direction lost mandatory source status.")
    valid = ~status[0].coverage_missing
    projected = correction.output[1]
    if not isinstance(projected, ProjectedEGMCarry):
        raise TypeError("NB R null-event direction lost its original projection.")

    # Only null weight terms are missing. Original bank and live-read tangents
    # must not inherit geometry from the transient counterfactual support.
    # keyword-only-exempt: library-callback=jax.tree_util.tree_map_with_path
    def add_null_direction(
        path: tuple[jax.tree_util.KeyEntry, ...], leaf: jax.Array | HostArray
    ) -> jax.Array | HostArray:
        address = tuple(getattr(key, "name", getattr(key, "idx", None)) for key in path)
        if address == ("output", 0):
            extra = cast("FloatND", correction.output[0])
        elif address == ("output", 1, "value"):
            extra = projected.value
        elif address == ("output", 1, "economic_marginal"):
            extra = projected.economic_marginal
        else:
            return leaf
        return jnp.where(valid, leaf + extra, jnp.nan)

    return primal, jax.tree.map_with_path(add_null_direction, direction)
