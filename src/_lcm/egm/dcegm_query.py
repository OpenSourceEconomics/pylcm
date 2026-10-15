"""Model-bound reads of DC conditional row candidates."""

import functools
import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from _lcm.egm.carry import EGM_ENDOGENOUS_COORDINATE, OwnedRowRead, read_owned_row
from _lcm.egm.dcegm_query_model import DCEGMQueryModel
from _lcm.egm.interp import locate_on_grid
from _lcm.egm.preferences import (
    NEWTON_ACTION_FLOOR,
    Preferences,
    concatenate_regime_function,
    get_discount_factor_reader,
    get_preferences_builder,
    newton_action_ceiling,
)
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind, select_query_data
from _lcm.grids.discrete import read_at_codes
from _lcm.probability import exact_product, is_negative, is_represented_zero
from _lcm.solution.feasibility import BoundInputStatus, shift_savings_grid
from _lcm.typing import EconFunction, EconFunctionKwargs, EconFunctionsMapping
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.solver_api import ContinuationCapabilities
from lcm.typing import (
    ActionName,
    BoolND,
    Float1D,
    FloatND,
    IntND,
    ReferenceName,
    ScalarFloat,
    ScalarInt,
)


class DCEGMQueryResult(NamedTuple):
    """One associated conditional candidate and its mandatory actual-query status."""

    value: FloatND
    marginal: FloatND
    savings: FloatND
    coverage_missing: BoolND
    bound_status: BoundInputStatus


def build_dcegm_query(
    *,
    model: DCEGMQueryModel,
    functions: EconFunctionsMapping,
    koopmans_aggregator: EconFunction,
) -> DCEGMQuery:
    """Bind conditional reads to the independently compiled target model."""
    if model.authority.row_topology is None:
        raise TypeError("A DC query requires independently compiled row topology.")
    bound = model.authority.bound
    if bound.lower_target is None:
        raise TypeError("A DC feature query requires the installed floor target.")
    if len(model.authority.interval_ids) != 1:
        raise TypeError("A smooth DC query requires one model-owned corner interval.")
    predicates = tuple(
        concatenate_regime_function(functions=functions, target=target)
        for purpose, target in model.authority.evaluator_targets
        if purpose.startswith("predicate:") and target not in bound.declaration_names
    )
    return DCEGMQuery(
        model=model,
        lower=concatenate_regime_function(
            functions=functions, target=bound.lower_target
        ),
        resources=concatenate_regime_function(
            functions=functions, target=bound.resources_target
        ),
        predicates=predicates,
        build_preferences=get_preferences_builder(
            functions=functions,
            action_name=bound.action,
            action_lower=NEWTON_ACTION_FLOOR,
            action_upper=newton_action_ceiling(model.offsets),
        ),
        read_discount_factor=get_discount_factor_reader(
            functions=functions, koopmans_aggregator=koopmans_aggregator
        ),
    )


@dataclass(frozen=True, kw_only=True)
class DCEGMQuery:
    """Select complete owned tuples before passive interpolation and choice ranking."""

    model: DCEGMQueryModel
    lower: Callable[..., FloatND]
    resources: Callable[..., FloatND]
    predicates: tuple[Callable[..., BoolND], ...]
    build_preferences: Callable[[EconFunctionKwargs], Preferences]
    read_discount_factor: Callable[[EconFunctionKwargs], ScalarFloat]

    @property
    def input_names(self) -> frozenset[ReferenceName]:
        """External actual-target inputs, excluding internally chosen quantities."""
        bound = self.model.authority.bound
        internal = {bound.action, bound.post_decision, "utility", "CE"}
        return frozenset(
            name
            for _, names in self.model.authority.evaluator_inputs
            for name in names
            if name not in internal
        )

    @property
    def capabilities(self) -> ContinuationCapabilities:
        """Offer value and native marginal reads from the installed adapter."""
        return ContinuationCapabilities(
            value=True, marginal_states=frozenset({EGM_ENDOGENOUS_COORDINATE})
        )

    def read_choices(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        passive_values: tuple[ScalarFloat, ...],
        passive_grids: tuple[Float1D, ...],
        action_names: tuple[ActionName, ...],
        action_values: tuple[FloatND | IntND, ...],
        pool: EconFunctionKwargs,
        euler_gradient: ScalarFloat,
        paired_marginal_read: bool = False,
    ) -> DCEGMQueryResult:
        """Read each conditional action using one actual fixed-cell floor evaluation."""
        if action_names != self.model.authority.action_names:
            raise TypeError("DC query actions differ from canonical target authority.")
        floor_state = self.evaluate_floor(pool=pool)
        action_shape = tuple(values.size for values in action_values)

        def read_one(index: ScalarInt) -> DCEGMQueryResult:
            action_index = (
                jnp.unravel_index(index, action_shape) if action_shape else ()
            )
            positions = (slice(None),) * len(passive_grids) + action_index
            actual = {
                **pool,
                **{
                    name: values[position]
                    for name, values, position in zip(
                        action_names, action_values, action_index, strict=True
                    )
                },
            }
            state_name = self.model.authority.bound.liquid_state

            def resources_at(state: ScalarFloat) -> ScalarFloat:
                return _evaluate(
                    function=self.resources, pool={**actual, state_name: state}
                )

            resource_slope = jax.grad(resources_at)(actual[state_name])
            return self._read_at_floor(
                grid=read_at_codes(array=grid, positions=positions),
                value=read_at_codes(array=value, positions=positions),
                marginal=read_at_codes(array=marginal, positions=positions),
                data=select_query_data(
                    data=data, positions=positions, floor_positions=()
                ),
                row_queries=read_at_codes(array=row_queries, positions=positions),
                row_gradients=read_at_codes(array=row_gradients, positions=positions),
                passive_values=passive_values,
                passive_grids=passive_grids,
                pool=actual,
                resource_gradient=resource_slope * euler_gradient,
                floor_state=floor_state,
                paired_marginal_read=paired_marginal_read,
            )

        return jax.vmap(read_one)(jnp.arange(math.prod(action_shape), dtype=jnp.int32))

    def __call__(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        passive_values: tuple[ScalarFloat, ...],
        passive_grids: tuple[Float1D, ...],
        pool: EconFunctionKwargs,
        resource_gradient: ScalarFloat,
        paired_marginal_read: bool = False,
    ) -> DCEGMQueryResult:
        """Read the canonical owned row tuple using one passive stencil."""
        return self.read_prepared(
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            row_queries=row_queries,
            row_gradients=row_gradients,
            passive_values=passive_values,
            passive_grids=passive_grids,
            pool=pool,
            resource_gradient=resource_gradient,
            floor_state=self.evaluate_floor(pool=pool),
            paired_marginal_read=paired_marginal_read,
        )

    def read_prepared(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        passive_values: tuple[ScalarFloat, ...],
        passive_grids: tuple[Float1D, ...],
        pool: EconFunctionKwargs,
        resource_gradient: ScalarFloat,
        floor_state: tuple[ScalarFloat, Float1D, BoundInputStatus],
        paired_marginal_read: bool = False,
    ) -> DCEGMQueryResult:
        """Read a conditional tuple with the actual cell's prepared target floor."""
        return self._read_at_floor(
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            row_queries=row_queries,
            row_gradients=row_gradients,
            passive_values=passive_values,
            passive_grids=passive_grids,
            pool=pool,
            resource_gradient=resource_gradient,
            floor_state=floor_state,
            paired_marginal_read=paired_marginal_read,
        )

    def _read_at_floor(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        passive_values: tuple[ScalarFloat, ...],
        passive_grids: tuple[Float1D, ...],
        pool: EconFunctionKwargs,
        resource_gradient: ScalarFloat,
        floor_state: tuple[ScalarFloat, Float1D, BoundInputStatus],
        paired_marginal_read: bool = False,
    ) -> DCEGMQueryResult:
        """Read one conditional alternative with its cell's already evaluated floor."""
        floor, nodes, bound_status = floor_state
        resources = _evaluate(function=self.resources, pool=pool)
        eligible = jnp.asarray(a=True)
        for predicate in self.predicates:
            eligible &= _evaluate(function=predicate, pool=pool)
        domain = resources > floor
        preferences = self.build_preferences(pool)
        beta = self.read_discount_factor(pool)
        weights, active, negative = _passive_weights(
            values=passive_values, grids=passive_grids
        )
        weights, active, negative = (
            tuple(weight.reshape(-1) for weight in weights),
            active.reshape(-1),
            negative.reshape(-1),
        )
        row = self._read_rows(
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            row_queries=row_queries,
            row_gradients=row_gradients,
            weights=weights,
            active=active,
            negative=negative,
            paired_marginal_read=paired_marginal_read,
        )
        row_legal = (
            row.supported
            & ~row.invalid_numeric
            & (row.savings >= floor)
            & (row.savings <= nodes[-1])
            & (resources - row.savings > 0.0)
        )
        if self.model.authority.corner_coordinate_kind == "liquid_row":
            corner = self._read_liquid_corner(
                grid=grid,
                data=data,
                row_queries=row_queries,
                weights=weights,
                active=active,
                negative=negative,
                paired_marginal_read=paired_marginal_read,
            )
        else:
            corner = _read_compact_corner(
                data=data,
                weights=weights,
                active=active,
                negative=negative,
                interval_id=self.model.authority.interval_ids[0],
                side_id=self.model.authority.interval_side_ids[0],
            )
        safe_consumption = jnp.where(domain & eligible, resources - floor, 1.0)
        corner_value = preferences.utility(safe_consumption) + zero_safe_weighted_term(
            weight=beta,
            value=corner.value,
            subnormal_is_accounted_for=True,
        )
        corner_marginal = (
            preferences.marginal_utility(safe_consumption)
            + beta * corner.direct_marginal
        ) * resource_gradient
        invalid_corner = corner.invalid_numeric | (
            eligible
            & domain
            & corner.supported
            & (
                (~jnp.isfinite(corner_value) & ~jnp.isneginf(corner_value))
                | ~jnp.isfinite(corner_marginal)
            )
        )
        finite_corner = (
            corner.supported
            & ~invalid_corner
            & jnp.isfinite(corner_value)
            & jnp.isfinite(corner_marginal)
        )
        take_row = row_legal & (~finite_corner | (row.value >= corner_value))
        represented = row_legal | finite_corner
        return DCEGMQueryResult(
            value=jnp.where(
                eligible & domain & represented,
                jnp.where(take_row, row.value, corner_value),
                -jnp.inf,
            ),
            marginal=jnp.where(
                eligible & domain & represented,
                jnp.where(take_row, row.marginal, corner_marginal),
                0.0,
            ),
            savings=jnp.where(take_row, row.savings, floor),
            coverage_missing=row.invalid_numeric
            | invalid_corner
            | (eligible & domain & ~represented),
            bound_status=bound_status,
        )

    def evaluate_floor(
        self, *, pool: EconFunctionKwargs
    ) -> tuple[ScalarFloat, Float1D, BoundInputStatus]:
        """Evaluate actual bound arithmetic before any economic eligibility mask."""
        floor = _evaluate(function=self.lower, pool=pool)
        nodes, status = shift_savings_grid(floor=floor, offsets=self.model.offsets)
        invalid = status.nonfinite_floor | status.invalid_shifted_grid
        return (
            jnp.where(invalid, 0.0, floor),
            jnp.where(invalid, self.model.offsets, nodes),
            status,
        )

    def _read_rows(
        self,
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        row_gradients: FloatND,
        weights: tuple[Float1D, ...],
        active: BoolND,
        negative: BoolND,
        paired_marginal_read: bool,
    ) -> _BlendedRow:
        """Keep the owned tuple and every active stencil status associated."""
        topology = self.model.authority.row_topology
        if topology is None:
            raise TypeError("A DC query requires independently compiled row topology.")
        capacity = grid.shape[-1]
        n_rows = row_queries.size
        rows = jax.tree.map(
            lambda leaf: leaf.reshape(n_rows, *leaf.shape[row_queries.ndim :]), data
        )
        read = _read_owned_derivative if paired_marginal_read else read_owned_row
        result = jax.vmap(
            functools.partial(
                read,
                owner_capacity=topology.owner_capacity,
                side_is_row_side=self.model.authority.corner_coordinate_kind
                != "liquid_row",
            )
        )(
            query=row_queries.reshape(n_rows),
            grid=grid.reshape(n_rows, capacity),
            value=value.reshape(n_rows, capacity),
            marginal=marginal.reshape(n_rows, capacity),
            data=rows,
        )
        economic = jnp.isneginf(result.value)
        negative_economic = jnp.any(active & economic & negative)
        positive_economic = jnp.any(active & economic & ~negative)
        finite_values = jnp.where(economic, 0.0, result.value)
        read_value = _weighted_sum(weights=weights, values=finite_values)
        read_marginal = _weighted_sum(
            weights=weights,
            values=result.marginal,
            cofactors=(row_gradients.reshape(-1),),
        )
        read_savings = _weighted_sum(weights=weights, values=result.savings)
        supported = jnp.all(~active | result.supported) & ~negative_economic
        invalid_numeric = jnp.any(active & result.invalid_numeric) | (
            supported
            & (
                (~jnp.isfinite(read_value) & ~positive_economic)
                | ~jnp.isfinite(read_marginal)
                | ~jnp.isfinite(read_savings)
            )
        )
        return _BlendedRow(
            value=jnp.where(positive_economic, -jnp.inf, read_value),
            marginal=jnp.where(positive_economic, 0.0, read_marginal),
            savings=read_savings,
            supported=supported,
            invalid_numeric=invalid_numeric,
        )

    def _read_liquid_corner(
        self,
        *,
        grid: FloatND,
        data: EGMQueryData,
        row_queries: FloatND,
        weights: tuple[Float1D, ...],
        active: BoolND,
        negative: BoolND,
        paired_marginal_read: bool,
    ) -> _CornerRead:
        """Read floor continuation and direct native marginal on real row geometry."""
        corner = data.liquid_corner_data
        if corner is None:
            raise TypeError("A liquid-row DC query requires its declared corner bank.")
        interval = self.model.authority.interval_ids[0]
        side = self.model.authority.interval_side_ids[0]
        kind = corner.node_kind[..., 0]
        canonical = (
            (corner.owner_id[..., 0] == interval)
            & (corner.interval_id[..., 0] == interval)
            & (corner.side_id[..., 0] == side)
        )
        # Unknown live labels remain fatal in the shared selected-node validator.
        kind = jnp.where(canonical | (kind == QueryNodeKind.PAD), kind, -1)
        corner_data = replace(
            data,
            savings=jnp.broadcast_to(data.floor_used[..., None], grid.shape),
            node_kind=kind,
            owner_id=corner.owner_id[..., 0],
            side_id=corner.side_id[..., 0],
            link_valid=corner.link_valid[..., 0],
        )
        result = self._read_rows(
            grid=grid,
            value=corner.continuation[..., 0],
            marginal=corner.direct_marginal[..., 0],
            data=corner_data,
            row_queries=row_queries,
            row_gradients=jnp.ones_like(row_queries),
            weights=weights,
            active=active,
            negative=negative,
            paired_marginal_read=paired_marginal_read,
        )
        return _CornerRead(
            value=result.value,
            direct_marginal=result.marginal,
            supported=result.supported,
            invalid_numeric=result.invalid_numeric,
        )


class _BlendedRow(NamedTuple):
    """A complete represented tuple after one signed passive stencil."""

    value: FloatND
    marginal: FloatND
    savings: FloatND
    supported: BoolND
    invalid_numeric: BoolND


class _CornerRead(NamedTuple):
    """Independent floor continuation support on the same passive stencil."""

    value: FloatND
    direct_marginal: FloatND
    supported: BoolND
    invalid_numeric: BoolND


def _read_compact_corner(
    *,
    data: EGMQueryData,
    weights: tuple[Float1D, ...],
    active: BoolND,
    negative: BoolND,
    interval_id: int,
    side_id: int,
) -> _CornerRead:
    """Read known corner support without granting missing interiors economic status."""
    if (
        data.corner_continuation is None
        or data.corner_kind is None
        or data.corner_interval_id is None
        or data.corner_side_id is None
    ):
        raise TypeError("A compact DC query requires declared corner records.")
    value = data.corner_continuation[..., 0].reshape(-1)
    kind = data.corner_kind[..., 0].reshape(-1)
    owned = (data.corner_interval_id[..., 0].reshape(-1) == interval_id) & (
        data.corner_side_id[..., 0].reshape(-1) == side_id
    )
    finite = kind == QueryNodeKind.FINITE
    economic = kind == QueryNodeKind.ECONOMIC_MINUS_INF
    known = finite | economic
    invalid_node = (kind != QueryNodeKind.PAD) & (
        ~known
        | ~owned
        | (finite & ~jnp.isfinite(value))
        | (economic & ~jnp.isneginf(value))
    )
    negative_economic = jnp.any(active & economic & negative)
    positive_economic = jnp.any(active & economic & ~negative)
    supported = jnp.all(~active | (known & owned)) & ~negative_economic
    blended = _weighted_sum(
        weights=weights, values=jnp.where(finite & owned, value, 0.0)
    )
    return _CornerRead(
        value=jnp.where(positive_economic, -jnp.inf, blended),
        direct_marginal=jnp.asarray(0.0),
        supported=supported,
        invalid_numeric=jnp.any(active & invalid_node)
        | (supported & ~positive_economic & ~jnp.isfinite(blended)),
    )


def _evaluate[Value: FloatND | BoolND](
    *, function: Callable[..., Value], pool: EconFunctionKwargs
) -> Value:
    """Bind only canonical evaluator leaves from the actual target pool."""
    return function(
        **{name: pool[name] for name in inspect.signature(function).parameters}
    )


def _passive_weights(
    *,
    values: tuple[ScalarFloat, ...],
    grids: tuple[Float1D, ...],
) -> tuple[tuple[FloatND, ...], BoolND, BoolND]:
    """Keep each signed axis factor and exact-zero stencil activity."""
    stencil_shape = tuple(grid.size for grid in grids)
    weights: list[FloatND] = []
    active = jnp.ones(stencil_shape, dtype=jnp.bool_)
    negative = jnp.zeros(stencil_shape, dtype=jnp.bool_)
    for axis, (query, grid) in enumerate(zip(values, grids, strict=True)):
        lower, upper, weight_upper = locate_on_grid(x_query=query, grid=grid)
        indices = jnp.arange(grid.size)
        axis_weights = (indices == lower) * (1.0 - weight_upper) + (
            indices == upper
        ) * weight_upper
        shape = [1] * len(grids)
        shape[axis] = grid.size
        weights.append(jnp.broadcast_to(axis_weights.reshape(shape), stencil_shape))
        active = active & ~is_represented_zero(axis_weights).reshape(shape)
        negative = negative ^ is_negative(axis_weights).reshape(shape)
    return tuple(weights), active, negative


def _read_owned_derivative(
    *,
    query: ScalarFloat,
    grid: Float1D,
    value: Float1D,
    marginal: Float1D,
    data: EGMQueryData,
    owner_capacity: int,
    side_is_row_side: bool = True,
) -> OwnedRowRead:
    """Differentiate the selected value piece and retain its associated tuple."""
    result, tangent = jax.jvp(
        functools.partial(
            _owned_at_query,
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            owner_capacity=owner_capacity,
            side_is_row_side=side_is_row_side,
        ),
        (query,),
        (jnp.ones_like(query),),
    )
    return result._replace(
        marginal=jnp.where(jnp.isneginf(result.value), 0.0, tangent.value)
    )


# keyword-only-exempt: library-callback=jax.jvp
def _owned_at_query(
    query: ScalarFloat,
    *,
    grid: Float1D,
    value: Float1D,
    marginal: Float1D,
    data: EGMQueryData,
    owner_capacity: int,
    side_is_row_side: bool = True,
) -> OwnedRowRead:
    """Expose the scalar query position for the owning-piece JVP transformation."""
    return read_owned_row(
        query=query,
        grid=grid,
        value=value,
        marginal=marginal,
        data=data,
        owner_capacity=owner_capacity,
        side_is_row_side=side_is_row_side,
    )


def _weighted_sum(
    *,
    weights: tuple[Float1D, ...],
    values: Float1D,
    cofactors: tuple[Float1D, ...] = (),
) -> ScalarFloat:
    """Accumulate associated terms before rounding their full factor product."""
    active = jnp.ones_like(values, dtype=jnp.bool_)
    for weight in weights:
        active &= ~is_represented_zero(weight)
    numeric = tuple(
        jnp.where(~active & ~jnp.isfinite(leaf), 0.0, leaf)
        for leaf in (values, *cofactors)
    )
    return jnp.sum(_factor_product(jnp.stack((*weights, *numeric))))


def _factor_product_bits(factors: FloatND) -> FloatND:
    """Keep finite term factors separate until their product is representable."""
    if factors.shape[0] == 0:
        return jnp.ones(factors.shape[1:], dtype=factors.dtype)
    if factors.shape[0] == 1:
        return factors[0]
    product = exact_product(factors)
    finite_zero = jnp.all(jnp.isfinite(factors), axis=0) & jnp.any(
        is_represented_zero(factors), axis=0
    )
    return jnp.where(finite_zero, jnp.zeros_like(product), product)


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _factor_product_jvp(
    primals: tuple[FloatND], tangents: tuple[FloatND]
) -> tuple[FloatND, FloatND]:
    """Use each other-factor product as the corresponding multilinear slope."""
    (factors,), (factor_tangents,) = primals, tangents
    if factors.shape[0] == 1:
        return factors[0], factor_tangents[0]
    result = _factor_product(factors)
    tangent = jnp.zeros_like(result)
    for index in range(factors.shape[0]):
        others = jnp.concatenate((factors[:index], factors[index + 1 :]))
        tangent += _factor_product(others) * factor_tangents[index]
    return result, tangent


_factor_product = jax.custom_jvp(_factor_product_bits)
_factor_product.defjvp(_factor_product_jvp)
