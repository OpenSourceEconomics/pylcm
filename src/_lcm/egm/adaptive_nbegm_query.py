"""Read associated conditional NB tuples on a model-owned adaptive outer mesh."""

import functools
import math
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from _lcm.continuation import AdaptiveOuterQuerySpec
from _lcm.egm.branch_aggregation import aggregate_uniform_observed_fixed_cost
from _lcm.egm.carry import _read_owned_row_with_support
from _lcm.egm.dcegm_query import (
    _evaluate,
    _passive_weights,
    _read_owned_derivative_with_support,
    _weighted_sum,
)
from _lcm.egm.ez_kernel import ez_marginal_of_resource, ez_period_value
from _lcm.egm.nbegm_query import (
    AdaptiveNBEGMQueryResult,
    NBEGMQuery,
    _read_nb_compact_corner_rows,
)
from _lcm.egm.outer_interpolation import LocalCubicOuterInterpolant
from _lcm.egm.outer_refinement import (
    SafeguardedSearchResult,
    safeguarded_continuous_argmax,
)
from _lcm.egm.preferences import concatenate_regime_function
from _lcm.egm.query_data import (
    EGMAdaptiveOuterData,
    EGMQueryData,
    select_adaptive_query_data,
    select_query_data,
)
from _lcm.grids.discrete import read_at_codes
from _lcm.params.mapping_leaf import MappingLeaf
from _lcm.params.sequence_leaf import SequenceLeaf
from _lcm.solution.feasibility import BoundInputStatus, ContinuationReadCounts
from _lcm.typing import EconFunctionKwargs
from _lcm.utils.counts import sum_counts
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.typing import (
    ActionName,
    BoolND,
    Float1D,
    FloatND,
    FunctionName,
    IntND,
    ReferenceName,
    ScalarFloat,
    ScalarInt,
)

# Value, native liquid marginal, associated saving, mandatory coverage, known support.
type _TupleRead = tuple[FloatND, FloatND, FloatND, BoolND, BoolND]


def _numeric_operand(*, pool: EconFunctionKwargs, name: ReferenceName) -> FloatND:
    """Require a numeric leaf at a canonical state or preference operand."""
    value = pool[name]
    if isinstance(value, (MappingLeaf, SequenceLeaf)):
        raise TypeError("A canonical query operand must be numeric.")
    return jnp.asarray(value)


@dataclass(frozen=True, kw_only=True)
class _Profile:
    """One D's adjuster tuples at every actual passive node, before blending."""

    query: NBEGMQuery
    pool: EconFunctionKwargs
    outer_post_decision: FunctionName
    floor: ScalarFloat
    savings_top: ScalarFloat
    nodes: Float1D
    count: ScalarInt
    value: FloatND
    marginal: FloatND
    savings: FloatND
    supported: BoolND
    invalid: BoolND
    corner_value: FloatND
    corner_supported: BoolND
    corner_invalid: BoolND
    corner_economic: BoolND
    interval_sides: BoolND
    links: BoolND
    euler_gradient: ScalarFloat

    def read(self, targets: FloatND) -> _TupleRead:
        """Validate the full associated tuple and actual corner at each probe."""
        actual = {**self.pool, self.outer_post_decision: targets}
        value, valid_v = self.interpolate(values=self.value, targets=targets)
        marginal, valid_m = self.interpolate(values=self.marginal, targets=targets)
        saving, valid_a = self.interpolate(values=self.savings, targets=targets)
        held, held_support = self.interpolate(
            values=jnp.where(self.corner_economic, -jnp.inf, self.corner_value),
            targets=targets,
            corner=True,
        )
        arithmetic_invalid = (
            (valid_v & (jnp.isnan(value) | jnp.isposinf(value)))
            | (valid_m & ~jnp.isfinite(marginal))
            | (valid_a & ~jnp.isfinite(saving))
            | (held_support & (jnp.isnan(held) | jnp.isposinf(held)))
        )
        return _validate_tuple(
            query=self.query,
            pool=actual,
            floor=self.floor,
            savings_top=self.savings_top,
            value=value,
            marginal=marginal,
            savings=saving,
            supported=valid_v & valid_m & valid_a,
            held=held,
            held_supported=held_support,
            invalid=self.status_at(targets=targets) | arithmetic_invalid,
            euler_gradient=self.euler_gradient,
        )

    def interpolate(
        self, *, values: FloatND, targets: FloatND, corner: bool = False
    ) -> tuple[FloatND, BoolND]:
        """Use the accepted prefix cubic; exact saved nodes retain their records."""
        targets = jnp.broadcast_arrays(targets, values[0])[0]
        active_nodes = jnp.where(
            jnp.arange(self.nodes.size) < self.count, self.nodes, jnp.inf
        )
        left = jnp.clip(
            jnp.searchsorted(active_nodes, targets, side="right") - 1, 0, self.count - 2
        )
        exact_index = jnp.clip(
            jnp.searchsorted(active_nodes, targets, side="left"), 0, self.count - 1
        )
        exact = self.nodes[exact_index] == targets
        support = self.corner_supported if corner else self.supported
        links = self.links & support[:-1] & support[1:]
        links_at_query = _take(values=links, index=left)
        if corner:
            actual_sides = self.query.model.intervals.evaluate_sides(
                {
                    **self.pool,
                    self.outer_post_decision: targets,
                }
            )
            # The held interval is the same economic comparison on both endpoints.
            for index in range(len(self.query.model.intervals.source_identity)):
                same_side = (
                    self.interval_sides[:-1, index] == self.interval_sides[1:, index]
                )
                links &= same_side[:, None]
                actual = actual_sides[index]
                links_at_query &= _take(values=links, index=left)
                endpoint_side = _take(
                    values=jnp.broadcast_to(
                        self.interval_sides[:, index, None], self.value.shape
                    ),
                    index=left,
                )
                links_at_query &= actual == endpoint_side
        else:
            links_at_query = _take(values=links, index=left)
        interpolant = LocalCubicOuterInterpolant()
        numeric = interpolant.evaluate(
            nodes=self.nodes,
            values=values,
            query=targets,
            interval_valid=links,
            valid_count=self.count,
        )
        valid = (
            interpolant.evaluate_validity(
                nodes=self.nodes,
                values=values,
                query=targets,
                interval_valid=links,
                valid_count=self.count,
            )
            & links_at_query
        )
        economic = _take(values=jnp.isneginf(values), index=left) & _take(
            values=jnp.isneginf(values), index=left + 1
        )
        economic &= (
            links_at_query
            & (targets >= self.nodes[0])
            & (targets <= self.nodes[self.count - 1])
        )
        node = _take(values=values, index=exact_index)
        node_supported = _take(values=support, index=exact_index)
        return jnp.where(
            exact, node, jnp.where(economic, -jnp.inf, numeric)
        ), jnp.where(exact, node_supported, valid | economic)

    def status_at(self, *, targets: FloatND) -> BoolND:
        """Keep active arithmetic corruption separate from surrogate value masking."""
        targets = jnp.broadcast_arrays(targets, self.value[0])[0]
        active_nodes = jnp.where(
            jnp.arange(self.nodes.size) < self.count, self.nodes, jnp.inf
        )
        left = jnp.clip(
            jnp.searchsorted(active_nodes, targets, side="right") - 1, 0, self.count - 2
        )
        exact_index = jnp.clip(
            jnp.searchsorted(active_nodes, targets, side="left"), 0, self.count - 1
        )
        exact = self.nodes[exact_index] == targets
        invalid = self.invalid | self.corner_invalid
        return jnp.where(
            exact,
            _take(values=invalid, index=exact_index),
            _take(values=invalid, index=left) | _take(values=invalid, index=left + 1),
        )


def _take(*, values: FloatND | BoolND, index: IntND) -> FloatND | BoolND:
    """Gather one associated outer entry per passive node and probe."""
    return jnp.take_along_axis(
        values, index.reshape(-1, values.shape[-1]), axis=0
    ).reshape(index.shape)


def read_adaptive_choices(
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
) -> AdaptiveNBEGMQueryResult | tuple[AdaptiveNBEGMQueryResult, ContinuationReadCounts]:
    """Optimize and keeper-fold per node/D, signed-blend same D, then expose D rows."""
    authority = query.model.authority
    spec = authority.adaptive_outer
    mesh = data.adaptive_outer_data
    roles = query.model.adaptive_inner_models
    if (
        spec is None
        or mesh is None
        or roles is None
        or set(roles) != {"keeper", "adjuster"}
    ):
        raise TypeError(
            "An adaptive NB reader lacks canonical role and mesh authority."
        )
    if action_names != authority.action_names:
        raise TypeError("Adaptive conditional actions differ from canonical authority.")
    if query.adaptive_queries is None:
        raise TypeError("An adaptive NB reader lacks its target-period role bindings.")
    keeper, adjuster = query.adaptive_queries
    floor, nodes, bound_status = query.evaluate_floor(pool=pool)
    weights, active, negative = _passive_weights(
        values=passive_values, grids=passive_grids
    )
    weights = tuple(item.reshape(-1) for item in weights)
    active, negative = active.reshape(-1), negative.reshape(-1)
    action_shape = tuple(item.size for item in action_values)

    def read_one(
        index: ScalarInt,
    ) -> tuple[AdaptiveNBEGMQueryResult, ContinuationReadCounts | None]:
        codes = jnp.unravel_index(index, action_shape) if action_shape else ()
        positions = (slice(None),) * len(passive_grids) + codes
        actual = {
            **pool,
            **{
                name: values[code]
                for name, values, code in zip(
                    action_names, action_values, codes, strict=True
                )
            },
        }
        selected = select_adaptive_query_data(
            data=data, cell_positions=positions, floor_positions=()
        )
        return _read_one_choice(
            keeper=keeper,
            adjuster=adjuster,
            parent=query,
            data=selected,
            grid=read_at_codes(array=grid, positions=positions),
            value=read_at_codes(array=value, positions=positions),
            marginal=read_at_codes(array=marginal, positions=positions),
            row_queries=read_at_codes(array=row_queries, positions=positions),
            row_gradients=read_at_codes(array=row_gradients, positions=positions),
            passive_grids=passive_grids,
            actual=actual,
            floor=floor,
            savings_top=nodes[-1],
            bound_status=bound_status,
            weights=weights,
            active=active,
            negative=negative,
            euler_gradient=euler_gradient,
            paired_marginal_read=paired_marginal_read,
            collect_counts=collect_continuation_read_counts,
        )

    result, counts = jax.vmap(read_one)(
        jnp.arange(math.prod(action_shape), dtype=jnp.int32)
    )
    if collect_continuation_read_counts:
        if counts is None:
            raise TypeError("Adaptive checked choices lack their original-row summary.")
        return result, ContinuationReadCounts(
            *(sum_counts(counts=item) for item in counts)
        )
    return result


def _read_one_choice(
    *,
    keeper: NBEGMQuery,
    adjuster: NBEGMQuery,
    parent: NBEGMQuery,
    data: EGMQueryData,
    grid: FloatND,
    value: FloatND,
    marginal: FloatND,
    row_queries: FloatND,
    row_gradients: FloatND,
    passive_grids: tuple[Float1D, ...],
    actual: EconFunctionKwargs,
    floor: ScalarFloat,
    savings_top: ScalarFloat,
    bound_status: BoundInputStatus,
    weights: tuple[Float1D, ...],
    active: BoolND,
    negative: BoolND,
    euler_gradient: ScalarFloat,
    paired_marginal_read: bool,
    collect_counts: bool,
) -> tuple[AdaptiveNBEGMQueryResult, ContinuationReadCounts | None]:
    """Retain original owned reads, optimize same-D profiles, and signed-blend."""
    spec = parent.model.authority.adaptive_outer
    mesh = data.adaptive_outer_data
    topology = parent.model.authority.row_topology
    if (
        spec is None
        or mesh is None
        or topology is None
        or topology.kind != "nb_budget_interval_run"
    ):
        raise TypeError("An adaptive NB read lacks its native model topology.")
    n_rows = row_queries.size
    capacity = grid.shape[-1]
    n_outer = spec.adjuster_capacity + 1
    if grid.shape[-2] != n_outer:
        raise TypeError("Adaptive row bank differs from the canonical keeper capacity.")
    raw_data = replace(data, adaptive_outer_data=None)
    raw_data = replace(
        raw_data,
        floor_used=jnp.broadcast_to(raw_data.floor_used[..., None], grid.shape[:-1]),
    )
    rows = jax.tree.map(
        lambda leaf: leaf.reshape(
            n_rows * n_outer, *leaf.shape[row_queries.ndim + 1 :]
        ),
        raw_data,
    )
    queries = jnp.broadcast_to(row_queries[..., None], grid.shape[:-1])
    read = (
        _read_owned_derivative_with_support
        if paired_marginal_read
        else _read_owned_row_with_support
    )
    original, supported_node = jax.vmap(
        functools.partial(
            read,
            owner_capacity=topology.owner_capacity,
            side_is_row_side=True,
        )
    )(
        query=queries.reshape(-1),
        grid=grid.reshape(-1, capacity),
        value=value.reshape(-1, capacity),
        marginal=marginal.reshape(-1, capacity),
        data=rows,
    )
    original = jax.tree.map(lambda leaf: leaf.reshape(n_rows, n_outer).T, original)
    raw_marginal = original.marginal * row_gradients.reshape(1, n_rows)
    outer_active = jnp.concatenate(
        (
            jnp.ones((1,), dtype=jnp.bool_),
            jnp.arange(spec.adjuster_capacity) < mesh.outer_count,
        )
    )
    source_active = active[:, None] & outer_active[None, :]
    counts = None
    if collect_counts:
        supported_node = supported_node.reshape(n_rows, n_outer, capacity)
        has_first = jnp.any(supported_node, axis=-1)
        first_index = jnp.argmax(supported_node, axis=-1)
        first = jnp.take_along_axis(
            grid.reshape(n_rows, n_outer, capacity), first_index[..., None], axis=-1
        )[..., 0]
        eligible = source_active & has_first
        counts = ContinuationReadCounts(
            sum_counts(counts=eligible & (queries.reshape(n_rows, n_outer) < first)),
            sum_counts(counts=eligible),
        )

    def compact_at(
        *,
        reader: NBEGMQuery,
        slot: ScalarInt | int,
        target: ScalarFloat,
    ) -> tuple[FloatND, BoolND, BoolND, BoolND, BoolND]:
        selected = select_query_data(
            data=replace(data, adaptive_outer_data=None),
            positions=(slice(None),) * len(passive_grids) + (slot,),
            floor_positions=(),
        )
        return _read_nb_compact_corner_rows(
            recipe=reader.model.intervals,
            data=selected,
            pool={**actual, spec.outer_post_decision: target},
            passive_names=reader.model.passive_state_names,
            passive_grids=passive_grids,
        )

    keeper_target = (
        _numeric_operand(pool=actual, name=spec.outer_state)
        if spec.keeper_target is None
        else _evaluate(
            function=concatenate_regime_function(
                functions=parent.model.functions, target=spec.keeper_target
            ),
            pool=actual,
        )
    )
    (
        keeper_w,
        keeper_support,
        keeper_economic,
        keeper_invalid,
        keeper_preimage_invalid,
    ) = compact_at(reader=keeper, slot=0, target=keeper_target)
    kept = _validate_tuple(
        query=keeper,
        pool={**actual, spec.outer_post_decision: keeper_target},
        floor=floor,
        savings_top=savings_top,
        value=original.value[0],
        marginal=raw_marginal[0],
        savings=original.savings[0],
        supported=original.supported[0],
        held=jnp.where(keeper_economic, -jnp.inf, keeper_w),
        held_supported=keeper_support,
        invalid=original.invalid_numeric[0] | keeper_invalid | keeper_preimage_invalid,
        euler_gradient=euler_gradient,
    )

    targets = jnp.where(
        jnp.arange(spec.adjuster_capacity) < mesh.outer_count,
        mesh.outer_nodes,
        mesh.outer_nodes[0],
    )
    corner_w, corner_support, corner_economic, corner_invalid, preimage_invalid = (
        jax.vmap(
            lambda index, target: compact_at(
                reader=adjuster, slot=index + 1, target=target
            )
        )(jnp.arange(spec.adjuster_capacity, dtype=jnp.int32), targets)
    )
    side_vectors = jax.vmap(
        lambda target: adjuster.model.intervals.evaluate_sides(
            {**actual, spec.outer_post_decision: target}
        )
    )(targets)
    links = mesh.outer_link_valid.reshape(n_rows, spec.adjuster_capacity - 1).T
    profile = _Profile(
        query=adjuster,
        pool=actual,
        outer_post_decision=spec.outer_post_decision,
        floor=floor,
        savings_top=savings_top,
        nodes=mesh.outer_nodes,
        count=mesh.outer_count,
        value=original.value[1:],
        marginal=raw_marginal[1:],
        savings=original.savings[1:],
        supported=original.supported[1:],
        invalid=original.invalid_numeric[1:],
        corner_value=corner_w,
        corner_supported=corner_support,
        corner_invalid=corner_invalid | preimage_invalid[:, None],
        corner_economic=corner_economic,
        interval_sides=side_vectors,
        links=links,
        euler_gradient=euler_gradient,
    )
    # Every exact node is validated before it defines a conditional profile.
    at_nodes = profile.read(targets[:, None])
    profile = replace(
        profile,
        value=at_nodes[0],
        marginal=at_nodes[1],
        savings=at_nodes[2],
        supported=at_nodes[4],
    )

    def tuple_objective(target: FloatND) -> tuple[FloatND, tuple[FloatND, ...], BoolND]:
        reading = profile.read(target)
        return reading[0], (reading[1], reading[2]), reading[3]

    search = safeguarded_continuous_argmax(
        objective=lambda target: profile.read(target)[0],
        objective_with_status=tuple_objective,
        nodes=mesh.outer_nodes,
        node_values=at_nodes[0],
        node_companions=(at_nodes[1], at_nodes[2]),
        golden_iterations=spec.golden_iterations,
        valid_count=mesh.outer_count,
    )
    return _fold_adaptive_choice(
        parent=parent,
        spec=spec,
        mesh=mesh,
        actual=actual,
        search=search,
        kept=kept,
        at_nodes=at_nodes,
        active=active,
        negative=negative,
        weights=weights,
        bound_status=bound_status,
    ), counts


def _fold_adaptive_choice(
    *,
    parent: NBEGMQuery,
    spec: AdaptiveOuterQuerySpec,
    mesh: EGMAdaptiveOuterData,
    actual: EconFunctionKwargs,
    search: SafeguardedSearchResult,
    kept: _TupleRead,
    at_nodes: _TupleRead,
    active: BoolND,
    negative: BoolND,
    weights: tuple[Float1D, ...],
    bound_status: BoundInputStatus,
) -> AdaptiveNBEGMQueryResult:
    """Fold one action's keeper/adjuster tuples before signed passive blending."""
    if search.companions is None or search.probe_status is None:
        raise TypeError("An adaptive tuple search lost its associated channels/status.")
    adjusted_marginal = search.companions[0]
    node_status = jnp.any(
        (jnp.arange(spec.adjuster_capacity) < mesh.outer_count)[:, None] & at_nodes[3],
        axis=0,
    )
    status = kept[3] | node_status | search.probe_status
    if spec.branch_aggregation == "deterministic":
        choose_adjuster = search.value > kept[0]
        node_value = jnp.where(choose_adjuster, search.value, kept[0])
        node_marginal = jnp.where(choose_adjuster, adjusted_marginal, kept[1])
    else:
        if spec.fixed_cost_target is None or spec.fixed_cost_support is None:
            raise TypeError(
                "An adaptive expected fold lacks its canonical scale/support."
            )
        scale = _evaluate(
            function=concatenate_regime_function(
                functions=parent.model.functions, target=spec.fixed_cost_target
            ),
            pool=actual,
        )
        folded = aggregate_uniform_observed_fixed_cost(
            keeper_value=kept[0],
            adjuster_value=search.value,
            scale=scale,
            lower=spec.fixed_cost_support[0],
            upper=spec.fixed_cost_support[1],
        )
        probability = folded.adjustment_probability
        node_value = folded.expected_value
        node_marginal = zero_safe_weighted_term(
            weight=probability, value=adjusted_marginal, subnormal_is_accounted_for=True
        ) + zero_safe_weighted_term(
            weight=1.0 - probability, value=kept[1], subnormal_is_accounted_for=True
        )
        status |= ~jnp.isfinite(scale) | (scale < 0.0)
    economic = jnp.isneginf(node_value)
    negative_economic = jnp.any(active & negative & economic)
    positive_economic = jnp.any(active & ~negative & economic)
    blended = _weighted_sum(
        weights=weights, values=jnp.where(economic, 0.0, node_value)
    )
    blended_marginal = _weighted_sum(weights=weights, values=node_marginal)
    coverage = jnp.any(active & status) | negative_economic
    coverage |= ~positive_economic & ~jnp.isfinite(blended)
    coverage |= ~jnp.isfinite(blended_marginal)
    return AdaptiveNBEGMQueryResult(
        value=jnp.where(positive_economic, -jnp.inf, blended),
        marginal=jnp.where(positive_economic, 0.0, blended_marginal),
        coverage_missing=coverage,
        bound_status=bound_status,
    )


def _validate_tuple(
    *,
    query: NBEGMQuery,
    pool: EconFunctionKwargs,
    floor: ScalarFloat,
    savings_top: ScalarFloat,
    value: FloatND,
    marginal: FloatND,
    savings: FloatND,
    supported: BoolND,
    held: FloatND,
    held_supported: BoolND,
    invalid: BoolND,
    euler_gradient: ScalarFloat,
) -> _TupleRead:
    """Compare a whole represented tuple with its genuine actual-budget floor corner."""
    state = query.model.authority.bound.liquid_state
    liquid = _numeric_operand(pool=pool, name=state)
    resources, slope = jax.jvp(
        lambda liquid: _evaluate(
            function=query.resources, pool={**pool, state: liquid}
        ),
        (liquid,),
        (jnp.ones_like(liquid),),
    )
    preferences = query.build_preferences(pool)
    beta = query.read_discount_factor(pool)
    eligible = jnp.asarray(a=True)
    for predicate in query.predicates:
        eligible &= _evaluate(function=predicate, pool=pool)
    domain = resources > floor
    row_legal = (
        supported
        & ~invalid
        & (savings >= floor)
        & (savings <= savings_top)
        & (resources > savings)
    )
    consumption = jnp.where(eligible & domain, resources - floor, 1.0)
    if query.model.inverse_eis_name is None:
        corner = preferences.utility(consumption) + zero_safe_weighted_term(
            weight=beta,
            value=held,
            subnormal_is_accounted_for=True,
        )
        corner_native_marginal = preferences.marginal_utility(consumption) * slope
    else:
        inverse_eis = 1.0 / _numeric_operand(
            pool=pool, name=query.model.inverse_eis_name
        )
        flow = preferences.utility(consumption)
        corner = ez_period_value(
            flow=flow, nu=held, discount_factor=beta, inverse_eis=inverse_eis
        )
        corner_native_marginal = (
            ez_marginal_of_resource(
                log_flow_marginal=jnp.log(preferences.marginal_utility(consumption))
                - inverse_eis * jnp.log(flow),
                value=corner,
                discount_factor=beta,
                inverse_eis=inverse_eis,
            )
            * slope
        )
    economic = jnp.isneginf(corner) & (query.model.inverse_eis_name is None)
    corner_marginal = jnp.where(economic, 0.0, corner_native_marginal * euler_gradient)
    invalid |= ~jnp.isfinite(resources)
    invalid |= (
        eligible
        & domain
        & held_supported
        & (jnp.isnan(corner) | jnp.isposinf(corner) | ~jnp.isfinite(corner_marginal))
    )
    corner_legal = held_supported & ~invalid & (jnp.isfinite(corner) | economic)
    take_row = row_legal & (~corner_legal | (value >= corner))
    represented = row_legal | corner_legal
    return (
        jnp.where(
            eligible & domain & represented,
            jnp.where(take_row, value, corner),
            -jnp.inf,
        ),
        jnp.where(
            eligible & domain & represented,
            jnp.where(take_row, marginal, corner_marginal),
            0.0,
        ),
        jnp.where(take_row, savings, floor),
        invalid | (eligible & domain & ~represented),
        represented,
    )
