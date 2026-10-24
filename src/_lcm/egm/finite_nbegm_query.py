"""Read native NB finite outer tuples before blending their passive coordinates."""

import functools
import math

import jax
import jax.numpy as jnp

from _lcm.continuation import EGMRowTopologySpec, FiniteOuterQuerySpec
from _lcm.egm.dcegm_query import FiniteNEGMQueryResult, _evaluate, _passive_weights
from _lcm.egm.ez_kernel import ez_marginal_of_resource, ez_period_value
from _lcm.egm.finite_negm_query import (
    _collapse_finite_selection,
    _FiniteCornerRead,
    _prepare_finite_passive_blend,
    _read_finite_native_row,
    _select_finite_outer_profiles,
    _validate_finite_outer_rows,
)
from _lcm.egm.nbegm_query import NBEGMQuery, _read_nb_compact_corner_rows
from _lcm.egm.preferences import concatenate_regime_function
from _lcm.egm.query_data import EGMQueryData, select_query_data
from _lcm.grids.discrete import read_at_codes
from _lcm.solution.feasibility import ContinuationReadCounts
from _lcm.typing import EconFunctionKwargs
from _lcm.utils.counts import sum_counts
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.typing import (
    ActionName,
    BoolND,
    Float1D,
    FloatND,
    IntND,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)


def read_finite_nbegm_choices(
    *,
    query: NBEGMQuery,
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
    paired_marginal_read: bool,
    collect_continuation_read_counts: bool,
) -> FiniteNEGMQueryResult | tuple[FiniteNEGMQueryResult, ContinuationReadCounts]:
    """Validate each native O tuple and fold nodewise before passive interpolation."""
    from _lcm.egm.continuation import _blend_passive_axes  # noqa: PLC0415

    authority = query.model.authority
    outer, topology, role_queries = _validate_finite_nbegm_query_rows(
        query=query,
        grid=grid,
        value=value,
        marginal=marginal,
        row_queries=row_queries,
        action_names=action_names,
    )
    floor, nodes, bound_status = query.evaluate_floor(pool=pool)
    _weights, active, negative = _passive_weights(
        values=passive_values, grids=passive_grids
    )
    active_flat = active.reshape(-1)
    action_shape = tuple(values.size for values in action_values)
    passive_shape = tuple(axis.size for axis in passive_grids)
    n_rows = math.prod(passive_shape)
    keeper_query, adjuster_query = role_queries
    keeper_function = (
        None
        if outer.keeper_target is None
        else concatenate_regime_function(
            functions=query.model.functions, target=outer.keeper_target
        )
    )

    def at_action(
        action: ScalarInt,
    ) -> tuple[FiniteNEGMQueryResult, ContinuationReadCounts | None]:
        index = jnp.unravel_index(action, action_shape) if action_shape else ()
        action_positions = (slice(None),) * len(passive_grids) + index
        common_queries = read_at_codes(array=row_queries, positions=action_positions)
        common_gradients = read_at_codes(
            array=row_gradients, positions=action_positions
        )
        actual = {
            **pool,
            **{
                name: values[position]
                for name, values, position in zip(
                    action_names, action_values, index, strict=True
                )
            },
        }
        keeper = (
            actual[outer.outer_state]
            if keeper_function is None
            else _evaluate(function=keeper_function, pool=actual)
        )
        alternatives = []
        summaries = []
        any_eligible = jnp.asarray(a=False)
        for outer_index in range(authority.outer_capacity):
            reader = keeper_query if outer_index == 0 else adjuster_query
            identity = (
                keeper
                if outer_index == 0
                else jnp.asarray(outer.outer_nodes[outer_index - 1], dtype=grid.dtype)
            )
            chosen = {**actual, outer.outer_post_decision: identity}
            resources, eligible, domain = _finite_nbegm_publication_inputs(
                query=reader, pool=chosen, floor=floor
            )
            any_eligible |= eligible & domain
            positions = (*action_positions, outer_index)
            native_grid = read_at_codes(array=grid, positions=positions)
            native_value = read_at_codes(array=value, positions=positions)
            native_marginal = read_at_codes(array=marginal, positions=positions)
            native_data = select_query_data(
                data=data, positions=positions, floor_positions=()
            )
            row_data = jax.tree.map(
                lambda leaf: leaf.reshape(n_rows, *leaf.shape[len(passive_shape) :]),
                native_data,
            )
            raw = jax.vmap(
                functools.partial(
                    _read_finite_native_row,
                    owner_capacity=topology.owner_capacity,
                    side_is_row_side=True,
                )
            )(
                coordinate=common_queries.reshape(n_rows),
                grid=native_grid.reshape(n_rows, authority.row_capacity),
                value=native_value.reshape(n_rows, authority.row_capacity),
                marginal=native_marginal.reshape(n_rows, authority.row_capacity),
                data=row_data,
            )
            if collect_continuation_read_counts:
                counted = active_flat & raw.has_first
                summaries.append(
                    ContinuationReadCounts(
                        below_first_live=sum_counts(
                            counts=counted
                            & (common_queries.reshape(n_rows) < raw.first)
                        ),
                        eligible=sum_counts(counts=counted),
                    )
                )
            corners = _read_finite_nb_corners(
                query=reader,
                data=native_data,
                pool=chosen,
                passive_grids=passive_grids,
                resources=resources,
                floor=floor,
                eligible=eligible & domain,
                euler_gradient=euler_gradient,
            )
            alternatives.append(
                _validate_finite_outer_rows(
                    raw=raw,
                    queries=common_queries.reshape(n_rows),
                    gradients=common_gradients.reshape(n_rows),
                    floor=floor,
                    upper_saving=nodes[-1],
                    resources=resources,
                    eligible=eligible & domain,
                    corner_value=corners.value,
                    corner_marginal=corners.marginal,
                    supported_corner=corners.supported,
                    finite_corner=corners.finite,
                    invalid_corner=corners.invalid,
                    paired_marginal_read=paired_marginal_read,
                )
            )
        candidates = jax.tree.map(
            lambda *leaves: jnp.stack(leaves, axis=-1), *alternatives
        )
        selection = _select_finite_outer_profiles(profiles=candidates)
        maximum, right, left, right_alive = _collapse_finite_selection(
            selection=selection
        )
        maximum, marginal_arrays, coverage_missing = _prepare_finite_passive_blend(
            selection=selection,
            value=maximum,
            marginal_arrays=(right, left, right_alive),
            any_eligible=any_eligible,
            coverage_missing=jnp.any(candidates.coverage_missing, axis=-1),
            active=active_flat,
            negative=negative.reshape(-1),
        )
        blended_value, blended_marginal = _blend_passive_axes(
            value_at_child=maximum.reshape(passive_shape),
            marginal_arrays=tuple(
                leaf.reshape(passive_shape) for leaf in marginal_arrays
            ),
            child_passive_values=passive_values,
            child_passive_grids=passive_grids,
            n_outer_candidates=authority.outer_capacity,
        )
        result = FiniteNEGMQueryResult(
            value=jnp.where(any_eligible, blended_value, -jnp.inf),
            marginal=jnp.where(any_eligible, blended_marginal, 0.0),
            coverage_missing=coverage_missing,
            bound_status=bound_status,
        )
        if collect_continuation_read_counts:
            return result, ContinuationReadCounts(
                *(
                    sum_counts(
                        counts=jnp.stack([counts[channel] for counts in summaries])
                    )
                    for channel in range(2)
                )
            )
        return result, None

    result, summaries = jax.vmap(at_action)(
        jnp.arange(math.prod(action_shape), dtype=jnp.int32)
    )
    if collect_continuation_read_counts:
        if summaries is None:
            raise TypeError("Checked finite NB reads require original row counts.")
        return result, ContinuationReadCounts(
            *(sum_counts(counts=count) for count in summaries)
        )
    return result


def _validate_finite_nbegm_query_rows(
    *,
    query: NBEGMQuery,
    grid: FloatND,
    value: FloatND,
    marginal: FloatND,
    row_queries: FloatND,
    action_names: tuple[ActionName, ...],
) -> tuple[FiniteOuterQuerySpec, EGMRowTopologySpec, tuple[NBEGMQuery, NBEGMQuery]]:
    """Authenticate native finite rows against their actual role authorities."""
    authority = query.model.authority
    outer = authority.finite_outer
    topology = authority.row_topology
    if (
        outer is None
        or topology is None
        or topology.kind != "nb_budget_interval_run"
        or authority.coordinate_kind != "liquid"
        or authority.outer_mode != "finite_max"
        or authority.outer_capacity != len(outer.outer_nodes) + 1
        or query.finite_queries is None
        or grid.shape[:-1] != (*row_queries.shape, authority.outer_capacity)
        or value.shape != grid.shape
        or marginal.shape != grid.shape
        or action_names != authority.action_names
    ):
        raise TypeError("Finite NB rows differ from canonical native outer authority.")
    return outer, topology, query.finite_queries


def _finite_nbegm_publication_inputs(
    *, query: NBEGMQuery, pool: EconFunctionKwargs, floor: ScalarFloat
) -> tuple[ScalarFloat, ScalarBool, ScalarBool]:
    """Evaluate each actual candidate budget and declared predicates once."""
    resources = _evaluate(function=query.resources, pool=pool)
    eligible = jnp.asarray(a=True)
    for predicate in query.predicates:
        eligible &= _evaluate(function=predicate, pool=pool)
    domain = resources > floor
    return resources, eligible, domain


def _read_finite_nb_corners(
    *,
    query: NBEGMQuery,
    data: EGMQueryData,
    pool: EconFunctionKwargs,
    passive_grids: tuple[Float1D, ...],
    resources: ScalarFloat,
    floor: ScalarFloat,
    eligible: BoolND,
    euler_gradient: ScalarFloat,
) -> _FiniteCornerRead:
    """Keep the NB held continuation at each passive node until the outer fold."""
    held, supported, economic, invalid, invalid_preimages = (
        _read_nb_compact_corner_rows(
            recipe=query.model.intervals,
            data=data,
            pool=pool,
            passive_names=query.model.passive_state_names,
            passive_grids=passive_grids,
        )
    )
    continuation = jnp.where(economic, -jnp.inf, held)
    preferences = query.build_preferences(pool)
    beta = query.read_discount_factor(pool)
    consumption = jnp.where(eligible, resources - floor, 1.0)
    liquid_state = query.model.authority.bound.liquid_state

    def resources_at(liquid: ScalarFloat) -> ScalarFloat:
        return _evaluate(function=query.resources, pool={**pool, liquid_state: liquid})

    resource_slope = jax.grad(resources_at)(pool[liquid_state])
    if query.model.inverse_eis_name is None:
        value = preferences.utility(consumption) + zero_safe_weighted_term(
            weight=beta, value=continuation, subnormal_is_accounted_for=True
        )
        native_marginal = preferences.marginal_utility(consumption) * resource_slope
    else:
        elasticity = pool[query.model.inverse_eis_name]
        if not isinstance(elasticity, jax.Array | float):
            raise TypeError("NB inverse elasticity must be a numeric leaf.")
        inverse_eis = 1.0 / elasticity
        flow = preferences.utility(consumption)
        value = ez_period_value(
            flow=flow,
            nu=continuation,
            discount_factor=beta,
            inverse_eis=inverse_eis,
        )
        native_marginal = (
            ez_marginal_of_resource(
                log_flow_marginal=jnp.log(preferences.marginal_utility(consumption))
                - inverse_eis * jnp.log(flow),
                value=value,
                discount_factor=beta,
                inverse_eis=inverse_eis,
            )
            * resource_slope
        )
    economic_value = jnp.isneginf(value) & (query.model.inverse_eis_name is None)
    marginal = jnp.broadcast_to(
        jnp.where(economic_value, 0.0, native_marginal * euler_gradient), value.shape
    )
    invalid = (
        invalid
        | invalid_preimages
        | (
            eligible
            & supported
            & ((~jnp.isfinite(value) & ~economic_value) | ~jnp.isfinite(marginal))
        )
    )
    supported = eligible & supported & ~invalid
    return _FiniteCornerRead(
        value=value,
        marginal=marginal,
        savings=jnp.broadcast_to(floor, value.shape),
        supported=supported,
        finite=supported & jnp.isfinite(value) & jnp.isfinite(marginal),
        invalid=invalid,
    )
