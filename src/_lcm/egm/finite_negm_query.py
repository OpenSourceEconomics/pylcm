"""Validate associated finite outer candidates before their nodewise maximum."""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from _lcm.continuation import EGMRowTopologySpec, FiniteOuterQuerySpec
from _lcm.egm.carry import (
    OwnedRowRead,
    _get_owned_local_piece,
    _read_owned_row_with_support,
)
from _lcm.egm.dcegm_query import (
    DCEGMQuery,
    FiniteNEGMQueryResult,
    _evaluate,
    _passive_weights,
    _read_compact_corner,
)
from _lcm.egm.interp import (
    _hermite_query_derivative,
    interp_left_germ_on_prepared_grid,
    interp_right_germ_on_prepared_grid,
    prepare_padded_grid,
)
from _lcm.egm.outer_envelope import right_germ_winner
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
    FunctionName,
    IntND,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
    StateName,
)

type _Germ = tuple[BoolND, FloatND, FloatND, FloatND]


class _FiniteRowRead(NamedTuple):
    """One native row's associated value and side-separated local payloads."""

    ordinary: OwnedRowRead
    left: OwnedRowRead
    right_germ: _Germ
    left_germ: _Germ
    right_derivative: FloatND
    left_derivative: FloatND
    first: FloatND
    last: FloatND
    has_first: BoolND
    germ_invalid: BoolND


class _FiniteOrdinaryRead(NamedTuple):
    """One independently legal ordinary side with its original row identity."""

    value: FloatND
    marginal: FloatND
    savings: FloatND
    supported: BoolND
    owner_id: IntND
    side_id: IntND
    dead: BoolND
    germ: _Germ


class _FiniteCornerRead(NamedTuple):
    """Associated actual-query corner channels with optional mapped axes."""

    value: FloatND
    marginal: FloatND
    savings: FloatND
    supported: BoolND
    finite: BoolND
    invalid: BoolND


def _single_row_finite_corner_read(
    *,
    value: ScalarFloat,
    marginal: ScalarFloat,
    savings: ScalarFloat,
    supported: BoolND,
    finite: BoolND,
    invalid: BoolND,
) -> _FiniteCornerRead:
    """Publish one passive node's checked scalar corner channels."""
    return _FiniteCornerRead(
        value=value,
        marginal=marginal,
        savings=savings,
        supported=supported,
        finite=finite,
        invalid=invalid,
    )


class _FiniteOuterProfile(NamedTuple):
    """Separate legal ordinary sides and the one actual corner for an O tuple."""

    right: _FiniteOrdinaryRead
    left: _FiniteOrdinaryRead
    corner: _FiniteCornerRead
    left_changed: BoolND
    coverage_missing: BoolND


class _FiniteSideSelection(NamedTuple):
    """Complete associated alternatives from one selected O and their kind."""

    ordinary: _FiniteOrdinaryRead
    corner: _FiniteCornerRead
    use_corner: BoolND


class _FiniteSelection(NamedTuple):
    """Original point maximum and complete selected right and left profiles."""

    value: FloatND
    outer_index: IntND
    right: _FiniteSideSelection
    left: _FiniteSideSelection


def read_finite_outer_choices(
    *,
    query: DCEGMQuery,
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
    """Read each real O tuple and corner before the existing nodewise O fold."""
    from _lcm.egm.continuation import (  # noqa: PLC0415
        _blend_passive_axes,
    )

    authority = query.model.authority
    outer, topology = _validate_finite_outer_query_rows(
        query=query,
        grid=grid,
        value=value,
        marginal=marginal,
        row_queries=row_queries,
        action_names=action_names,
    )
    # No candidate cost, preference or predicate is evaluated before this floor.
    floor, nodes, bound_status = query.evaluate_floor(pool=pool)
    weights, active, negative = _passive_weights(
        values=passive_values, grids=passive_grids
    )
    flat_weights = tuple(weight.reshape(-1) for weight in weights)
    active_flat = active.reshape(-1)
    action_shape = tuple(values.size for values in action_values)
    passive_shape = tuple(grid.size for grid in passive_grids)
    n_rows = math.prod(passive_shape)
    row_states = _finite_row_states(
        state_names=authority.state_names, passive_grids=passive_grids
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
        # Every O differs only in its index and identity, so the reads are one
        # program batched over the candidate axis rather than one per candidate.
        identities, row_identities = _finite_outer_identities(
            query=query,
            actual=actual,
            row_states=row_states,
            n_rows=n_rows,
            dtype=grid.dtype,
        )

        def at_outer(
            *,
            outer_index: ScalarInt,
            identity: ScalarFloat,
            row_identity: FloatND,
        ) -> tuple[
            _FiniteOuterProfile,
            ContinuationReadCounts | None,
            BoolND,
            _FiniteCornerRead,
        ]:
            chosen = {**actual, outer.outer_post_decision: identity}
            resources, eligible, domain, resource_gradient = (
                _finite_outer_publication_inputs(
                    query=query,
                    pool=chosen,
                    floor=floor,
                    euler_gradient=euler_gradient,
                )
            )
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
                    side_is_row_side=authority.corner_coordinate_kind != "liquid_row",
                )
            )(
                coordinate=common_queries.reshape(n_rows),
                grid=native_grid.reshape(n_rows, authority.row_capacity),
                value=native_value.reshape(n_rows, authority.row_capacity),
                marginal=native_marginal.reshape(n_rows, authority.row_capacity),
                data=row_data,
            )
            summary = None
            if collect_continuation_read_counts:
                counted = active_flat & raw.has_first
                summary = ContinuationReadCounts(
                    below_first_live=sum_counts(
                        counts=counted & (common_queries.reshape(n_rows) < raw.first)
                    ),
                    eligible=sum_counts(counts=counted),
                )
            row_resources = _finite_row_resources(
                query=query,
                actual=actual,
                row_states=row_states,
                outer_post_decision=outer.outer_post_decision,
                row_identity=row_identity,
            )
            corner = _read_finite_corner_row(
                grid=native_grid,
                data=native_data,
                row_queries=common_queries,
                weights=flat_weights,
                active=active_flat,
                negative=negative.reshape(-1),
                query=query,
                pool=chosen,
                floor=floor,
                resources=resources,
                eligible=eligible & domain,
                resource_gradient=resource_gradient,
                paired_marginal_read=paired_marginal_read,
            )
            # The corner is an actual-state object and competes with the blended
            # ordinary channel below, never inside this row's nodewise maximum.
            profile = _validate_finite_outer_rows(
                raw=raw,
                queries=common_queries.reshape(n_rows),
                gradients=common_gradients.reshape(n_rows),
                floor=floor,
                upper_saving=nodes[-1],
                resources=resources,
                eligible=eligible & domain & (row_resources > floor),
                corner_value=jnp.full((n_rows,), -jnp.inf, dtype=grid.dtype),
                corner_marginal=jnp.zeros((n_rows,), dtype=grid.dtype),
                supported_corner=jnp.zeros((n_rows,), dtype=jnp.bool_),
                finite_corner=jnp.zeros((n_rows,), dtype=jnp.bool_),
                invalid_corner=jnp.broadcast_to(corner.invalid, (n_rows,)),
                paired_marginal_read=paired_marginal_read,
                corner_in_row_maximum=False,
            )
            return profile, summary, eligible & domain, corner

        candidates, summaries, eligible_outer, corners = jax.vmap(
            at_outer, out_axes=(-1, 0, 0, 0)
        )(
            outer_index=jnp.arange(authority.outer_capacity, dtype=jnp.int32),
            identity=identities,
            row_identity=row_identities,
        )
        any_eligible = jnp.any(eligible_outer)
        selection = _select_finite_outer_profiles(profiles=candidates)
        maximum, right, left, right_alive = _collapse_finite_selection(
            selection=selection
        )
        maximum, marginal_arrays, ordinary_supported, dead_ordinary = (
            _prepare_finite_ordinary_blend(
                selection=selection,
                value=maximum,
                marginal_arrays=(right, left, right_alive),
                active=active_flat,
                negative=negative.reshape(-1),
            )
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
        value_at, marginal_at, represented = _choose_finite_actual_read(
            corners=corners,
            ordinary_value=jnp.where(dead_ordinary, -jnp.inf, blended_value),
            ordinary_marginal=jnp.where(dead_ordinary, 0.0, blended_marginal),
            ordinary_supported=ordinary_supported,
        )
        coverage_missing = (
            jnp.any(active_flat & jnp.any(candidates.coverage_missing, axis=-1))
            | jnp.any(corners.invalid)
            | (any_eligible & ~represented)
        )
        result = FiniteNEGMQueryResult(
            value=jnp.where(any_eligible, value_at, -jnp.inf),
            marginal=jnp.where(any_eligible, marginal_at, 0.0),
            coverage_missing=coverage_missing,
            bound_status=bound_status,
        )
        if summaries is not None:
            return result, ContinuationReadCounts(
                *(sum_counts(counts=summaries[channel]) for channel in range(2))
            )
        return result, None

    result, summaries = jax.vmap(at_action)(
        jnp.arange(math.prod(action_shape), dtype=jnp.int32)
    )
    if collect_continuation_read_counts:
        if summaries is None:
            raise TypeError(
                "Checked finite reads must retain actual paired-row counts."
            )
        return result, ContinuationReadCounts(
            *(sum_counts(counts=count) for count in summaries)
        )
    return result


def _prepare_finite_ordinary_blend(
    *,
    selection: _FiniteSelection,
    value: FloatND,
    marginal_arrays: tuple[FloatND, ...],
    active: BoolND,
    negative: BoolND,
) -> tuple[FloatND, tuple[FloatND, ...], ScalarBool, ScalarBool]:
    """Apply the canonical signed-stencil row rules to the ordinary channel.

    Mirrors the plain DC query's `_blend_owned_rows`. Per active row the
    ordinary channel is its nodewise O maximum over legal ordinary reads;
    across rows:

    - a row without any legal ordinary read leaves the channel unsupported;
    - a supported `-inf` row at a positive weight makes the channel `-inf`;
    - a supported `-inf` row at a negative weight leaves the channel
      unsupported, since extrapolating an infinity has no sign.

    Returns the finite row values to blend (infinite and unsupported rows
    replaced by zero), the marginal payloads with unsupported rows zeroed,
    whether the blended channel is supported, and whether it is `-inf`.
    """
    supported = selection.right.ordinary.supported[..., 0]
    economic = supported & jnp.isneginf(value)
    negative_economic = jnp.any(active & negative & economic)
    positive_economic = jnp.any(active & ~negative & economic)
    channel_supported = ~jnp.any(active & ~supported) & ~negative_economic
    return (
        jnp.where(supported & ~economic, value, 0.0),
        tuple(jnp.where(supported, leaf, 0.0) for leaf in marginal_arrays),
        channel_supported,
        channel_supported & positive_economic,
    )


def _choose_finite_actual_read(
    *,
    corners: _FiniteCornerRead,
    ordinary_value: ScalarFloat,
    ordinary_marginal: ScalarFloat,
    ordinary_supported: ScalarBool,
) -> tuple[ScalarFloat, ScalarFloat, ScalarBool]:
    """Let the best finite actual corner compete with the blended ordinary read.

    As in the plain DC query, a supported ordinary read wins ties; without a
    supported ordinary read the corner represents the query alone. Value and
    marginal come from the same winner, and across O the first maximal finite
    corner owns the marginal. Returns the pair and whether either channel
    represents the query.
    """
    corner_values = jnp.where(corners.finite, corners.value, -jnp.inf)
    best = jnp.argmax(corner_values)
    finite_corner = jnp.any(corners.finite)
    take_ordinary = ordinary_supported & (
        ~finite_corner | (ordinary_value >= corner_values[best])
    )
    return (
        jnp.where(take_ordinary, ordinary_value, corner_values[best]),
        jnp.where(take_ordinary, ordinary_marginal, corners.marginal[best]),
        ordinary_supported | finite_corner,
    )


def _finite_outer_identities(
    *,
    query: DCEGMQuery,
    actual: EconFunctionKwargs,
    row_states: tuple[tuple[StateName, FloatND], ...],
    n_rows: int,
    dtype: jnp.dtype,
) -> tuple[FloatND, FloatND]:
    """Return each O's outer post-decision at the actual query and at every row.

    The keeper's identity is its no-adjustment level, read at the actual query
    (for feasibility) and at each passive node (for that row's economics); an
    adjuster's identity is its outer node everywhere. Shapes `(O,)` and
    `(O, n_rows)`.
    """
    outer = query.model.authority.finite_outer
    if outer is None:
        raise TypeError("Finite outer identities need canonical outer authority.")

    def keeper_at(state_pool: EconFunctionKwargs) -> ScalarFloat:
        return jnp.asarray(
            state_pool[outer.outer_state]
            if query.finite_keeper is None
            else _evaluate(function=query.finite_keeper, pool=state_pool)
        )

    nodes = tuple(jnp.asarray(node, dtype=dtype) for node in outer.outer_nodes)
    row_keepers = tuple(
        keeper_at({**actual, **{name: values[row] for name, values in row_states}})
        for row in range(n_rows)
    )
    return (
        jnp.stack([keeper_at(actual), *nodes]),
        jnp.stack(
            [
                jnp.stack(row_keepers),
                *(jnp.broadcast_to(node, (n_rows,)) for node in nodes),
            ]
        ),
    )


def _finite_row_resources(
    *,
    query: DCEGMQuery,
    actual: EconFunctionKwargs,
    row_states: tuple[tuple[StateName, FloatND], ...],
    outer_post_decision: FunctionName,
    row_identity: FloatND,
) -> FloatND:
    """Evaluate each passive node's own budget for its ordinary row eligibility."""
    names = tuple(name for name, _ in row_states)

    def at_row(
        *, passive_row: tuple[ScalarFloat, ...], identity: ScalarFloat
    ) -> ScalarFloat:
        return _evaluate(
            function=query.resources,
            pool={
                **actual,
                **dict(zip(names, passive_row, strict=True)),
                outer_post_decision: identity,
            },
        )

    return jax.vmap(at_row)(
        passive_row=tuple(values for _, values in row_states), identity=row_identity
    )


def _finite_row_states(
    *, state_names: tuple[StateName, ...], passive_grids: tuple[Float1D, ...]
) -> tuple[tuple[StateName, FloatND], ...]:
    """Name each passive state and list its value at every flattened row.

    The authority orders the discrete states before the passive ones, so the
    passive names are the trailing `len(passive_grids)` state names. Rows run in
    the row-major order of the passive grids, the order of the row queries.
    """
    if not passive_grids:
        return ()
    names = state_names[len(state_names) - len(passive_grids) :]
    mesh = jnp.meshgrid(*passive_grids, indexing="ij")
    return tuple(
        (name, axis.reshape(-1)) for name, axis in zip(names, mesh, strict=True)
    )


def _validate_finite_outer_query_rows(
    *,
    query: DCEGMQuery,
    grid: FloatND,
    value: FloatND,
    marginal: FloatND,
    row_queries: FloatND,
    action_names: tuple[ActionName, ...],
) -> tuple[FiniteOuterQuerySpec, EGMRowTopologySpec]:
    """Authenticate finite row shapes against their canonical outer authority."""
    authority = query.model.authority
    outer = authority.finite_outer
    topology = authority.row_topology
    if (
        outer is None
        or topology is None
        or authority.outer_mode != "finite_max"
        or authority.outer_capacity != len(outer.outer_nodes) + 1
        or grid.shape[:-1] != (*row_queries.shape, authority.outer_capacity)
        or value.shape != grid.shape
        or marginal.shape != grid.shape
        or action_names != authority.action_names
    ):
        raise TypeError("Finite nested rows differ from canonical outer authority.")
    return outer, topology


def _prepare_finite_passive_blend(
    *,
    selection: _FiniteSelection,
    value: FloatND,
    marginal_arrays: tuple[FloatND, ...],
    any_eligible: ScalarBool,
    coverage_missing: BoolND,
    active: BoolND,
    negative: BoolND,
) -> tuple[FloatND, tuple[FloatND, ...], ScalarBool]:
    """Keep empty states and negative economic support out of passive arithmetic."""
    selected = selection.right
    supported = jnp.where(
        selected.use_corner,
        selected.corner.supported,
        selected.ordinary.supported,
    )[..., 0]
    negative_economic = active & negative & supported & jnp.isneginf(value)
    return (
        jnp.where(any_eligible & ~negative_economic, value, 0.0),
        tuple(jnp.where(any_eligible, leaf, 0.0) for leaf in marginal_arrays),
        jnp.any((active & coverage_missing) | negative_economic),
    )


def _finite_outer_publication_inputs(
    *,
    query: DCEGMQuery,
    pool: EconFunctionKwargs,
    floor: ScalarFloat,
    euler_gradient: ScalarFloat,
) -> tuple[ScalarFloat, BoolND, BoolND, ScalarFloat]:
    """Evaluate one outer choice's budget, predicates and native resource gradient."""
    resources = _evaluate(function=query.resources, pool=pool)
    eligible = jnp.asarray(a=True)
    for predicate in query.predicates:
        eligible &= _evaluate(function=predicate, pool=pool)
    domain = resources > floor
    resource_gradient = _read_finite_resource_gradient(
        query=query, pool=pool, euler_gradient=euler_gradient
    )
    return resources, eligible, domain, resource_gradient


def _read_finite_resource_gradient(
    *,
    query: DCEGMQuery,
    pool: EconFunctionKwargs,
    euler_gradient: ScalarFloat,
) -> ScalarFloat:
    """Differentiate this O tuple's actual budget in its canonical liquid state."""
    liquid_state = query.model.authority.bound.liquid_state

    def resources_at(state: ScalarFloat) -> ScalarFloat:
        return _evaluate(function=query.resources, pool={**pool, liquid_state: state})

    return jax.grad(resources_at)(pool[liquid_state]) * euler_gradient


def _read_finite_corner_row(
    *,
    grid: FloatND,
    data: EGMQueryData,
    row_queries: FloatND,
    weights: tuple[Float1D, ...],
    active: BoolND,
    negative: BoolND,
    query: DCEGMQuery,
    pool: EconFunctionKwargs,
    floor: ScalarFloat,
    resources: ScalarFloat,
    eligible: BoolND,
    resource_gradient: ScalarFloat,
    paired_marginal_read: bool,
) -> _FiniteCornerRead:
    """Price one O's floor corner at the actual query on the blended continuation.

    A row corner `u(c) + beta W_i(floor)` is affine in `W_i` at fixed actual
    economics, so blending row corners over the passive stencil equals one
    corner at the actual budget, preferences and floor on the passively
    blended continuation. That corner is an actual-state object: it competes
    with the blended ordinary maximum, never inside the nodewise one. Its
    support requires every active row's corner record to be known.
    """
    authority = query.model.authority
    if authority.corner_coordinate_kind == "liquid_row":
        corner = query._read_liquid_corner(  # noqa: SLF001
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
            interval_id=authority.interval_ids[0],
            side_id=authority.interval_side_ids[0],
        )
    preferences = query.build_preferences(pool)
    beta = query.read_discount_factor(pool)
    consumption = jnp.where(eligible, resources - floor, 1.0)
    value = preferences.utility(consumption) + zero_safe_weighted_term(
        weight=beta, value=corner.value, subnormal_is_accounted_for=True
    )
    marginal = (
        preferences.marginal_utility(consumption) + beta * corner.direct_marginal
    ) * resource_gradient
    invalid = corner.invalid_numeric | (
        eligible
        & corner.supported
        & ((~jnp.isfinite(value) & ~jnp.isneginf(value)) | ~jnp.isfinite(marginal))
    )
    supported = eligible & corner.supported & ~invalid
    return _single_row_finite_corner_read(
        value=jnp.where(supported, value, -jnp.inf),
        marginal=jnp.where(supported & ~jnp.isneginf(value), marginal, 0.0),
        savings=floor,
        supported=supported,
        finite=(supported & jnp.isfinite(value) & jnp.isfinite(marginal)),
        invalid=invalid,
    )


def _read_finite_native_row(
    *,
    coordinate: ScalarFloat,
    grid: Float1D,
    value: Float1D,
    marginal: Float1D,
    data: EGMQueryData,
    owner_capacity: int,
    side_is_row_side: bool,
) -> _FiniteRowRead:
    """Use the existing owned reader on both sides of the same original bracket."""
    read = functools.partial(
        _read_owned_row_with_support,
        grid=grid,
        value=value,
        marginal=marginal,
        data=data,
        owner_capacity=owner_capacity,
        side_is_row_side=side_is_row_side,
    )
    ordinary, supported = read(query=coordinate)
    left, _ = read(query=coordinate, _left_endpoint=True)

    read_side = functools.partial(
        _read_finite_owned_side,
        coordinate=coordinate,
        grid=grid,
        value=value,
        marginal=marginal,
        data=data,
        supported=supported,
        side_is_row_side=side_is_row_side,
    )
    right_germ, right_derivative = read_side(row=ordinary, left_endpoint=False)
    left_germ, left_derivative = read_side(row=left, left_endpoint=True)
    _, count = prepare_padded_grid(grid)
    first = jnp.min(jnp.where(supported, grid, jnp.inf))
    last = grid[jnp.maximum(count - 1, 0)]
    return _FiniteRowRead(
        ordinary=ordinary,
        left=left,
        right_germ=right_germ,
        left_germ=left_germ,
        right_derivative=right_derivative,
        left_derivative=left_derivative,
        first=first,
        last=last,
        has_first=jnp.any(supported),
        germ_invalid=(
            (right_germ[0] & ~jnp.all(jnp.isfinite(jnp.stack(right_germ[1:]))))
            | (left_germ[0] & ~jnp.all(jnp.isfinite(jnp.stack(left_germ[1:]))))
        ),
    )


def _read_finite_owned_side(
    *,
    coordinate: ScalarFloat,
    grid: Float1D,
    value: Float1D,
    marginal: Float1D,
    data: EGMQueryData,
    supported: BoolND,
    side_is_row_side: bool,
    row: OwnedRowRead,
    left_endpoint: bool,
) -> tuple[_Germ, ScalarFloat]:
    """Separate a directional owned value germ from its economic query derivative."""
    search, count = prepare_padded_grid(grid)
    last = jnp.maximum(count - 1, 0)
    insertion = jnp.searchsorted(search, coordinate, side="right")
    node_index = jnp.clip(insertion - 1, 0, last).astype(jnp.int32)
    if left_endpoint:
        incoming = jnp.clip(jnp.searchsorted(search, coordinate, side="left"), 0, last)
        node_index = jnp.where(grid[incoming] == coordinate, incoming, node_index)
    endpoint = (count > 0) & (
        (grid[node_index] == coordinate) | (coordinate >= grid[last])
    )
    node_index = jnp.where(coordinate > grid[last], last, node_index)
    upper = jnp.clip(insertion, 1, jnp.maximum(count - 1, 1)).astype(jnp.int32)
    piece = functools.partial(
        _get_owned_local_piece,
        grid=jnp.where(jnp.isfinite(grid), grid, 0.0),
        data=data,
        supported_node=supported,
        lower=upper - 1,
        upper=upper,
        endpoint=endpoint,
        node_index=node_index,
        last=last,
    )
    indices, length, has_piece = piece(use_left=jnp.asarray(a=left_endpoint))
    local_grid = grid[indices]
    germ_function = (
        interp_left_germ_on_prepared_grid
        if left_endpoint
        else interp_right_germ_on_prepared_grid
    )
    germ = germ_function(
        x_query=coordinate,
        search_grid=jnp.where(jnp.arange(2) < length, local_grid, jnp.inf),
        valid_length=length,
        xp=local_grid,
        fp=value[indices],
        fp_slopes=marginal[indices],
    )
    clamp_ray = (
        (coordinate > grid[last] if left_endpoint else coordinate >= grid[last])
        & (count > 0)
        & supported[last]
    )
    represented = row.supported & jnp.isfinite(row.value)
    finite = represented & (clamp_ray | (has_piece & germ[0]))
    result: _Germ = (
        finite,
        jnp.where(finite & ~clamp_ray, germ[1], 0.0),
        jnp.where(finite & ~clamp_ray, germ[2], 0.0),
        jnp.where(finite & ~clamp_ray, germ[3], 0.0),
    )
    use_left: ScalarBool = (
        left_endpoint
        | (node_index == last)
        | (side_is_row_side & (data.side_id[node_index] == -1))
    )
    economic_indices, economic_length, _ = piece(use_left=use_left)
    economic_grid = grid[economic_indices]
    derivative = _hermite_query_derivative(
        x_query=coordinate,
        search_grid=jnp.where(jnp.arange(2) < economic_length, economic_grid, jnp.inf),
        valid_length=economic_length,
        xp=economic_grid,
        fp=value[economic_indices],
        fp_slopes=marginal[economic_indices],
    )
    return result, jnp.where(represented, derivative, 0.0)


def _validate_finite_outer_rows(
    *,
    raw: _FiniteRowRead,
    queries: FloatND,
    gradients: FloatND,
    floor: ScalarFloat,
    upper_saving: ScalarFloat,
    resources: ScalarFloat,
    eligible: BoolND,
    corner_value: FloatND,
    corner_marginal: FloatND,
    supported_corner: BoolND,
    finite_corner: BoolND,
    invalid_corner: BoolND,
    paired_marginal_read: bool,
    corner_in_row_maximum: bool = True,
) -> _FiniteOuterProfile:
    """Validate each side's own action and retain the separate actual corner.

    With `corner_in_row_maximum` the corner competes in this row's nodewise
    maximum, so a row that neither channel represents is uncovered here.
    Without it the corner is an actual-state object read once per O, and
    representability is decided on the blended channel instead.
    """
    row = raw.ordinary
    left = raw.left
    right_legal = (
        row.supported
        & ~row.invalid_numeric
        & (row.savings >= floor)
        & (row.savings <= upper_saving)
        & (resources - row.savings > 0.0)
        & (queries >= raw.first)
    )
    left_legal = (
        left.supported
        & ~left.invalid_numeric
        & (left.savings >= floor)
        & (left.savings <= upper_saving)
        & (resources - left.savings > 0.0)
        & (queries >= raw.first)
    )
    above = queries > raw.last
    paired_invalid = paired_marginal_read & (
        (row.supported & ~jnp.isfinite(raw.right_derivative))
        | (left.supported & ~jnp.isfinite(raw.left_derivative))
    )
    right_supported = eligible & right_legal
    left_supported = (
        eligible & left_legal & (raw.left_germ[0] | jnp.isneginf(left.value))
    )
    return _FiniteOuterProfile(
        right=_FiniteOrdinaryRead(
            value=jnp.where(right_supported, row.value, -jnp.inf),
            marginal=jnp.where(
                right_supported & ~above & ~jnp.isneginf(row.value),
                (raw.right_derivative if paired_marginal_read else row.marginal)
                * gradients,
                0.0,
            ),
            savings=row.savings,
            supported=right_supported,
            owner_id=row.owner_id,
            side_id=row.side_id,
            dead=~right_supported | (queries >= raw.last) | (row.side_id == -1),
            germ=(
                right_supported & raw.right_germ[0],
                raw.right_germ[1],
                raw.right_germ[2],
                raw.right_germ[3],
            ),
        ),
        left=_FiniteOrdinaryRead(
            value=jnp.where(left_supported, left.value, -jnp.inf),
            marginal=jnp.where(
                left_supported & ~above & ~jnp.isneginf(left.value),
                (raw.left_derivative if paired_marginal_read else left.marginal)
                * gradients,
                0.0,
            ),
            savings=left.savings,
            supported=left_supported,
            owner_id=left.owner_id,
            side_id=left.side_id,
            dead=~left_supported | (queries <= raw.first) | (left.side_id == 1),
            germ=(
                left_supported & raw.left_germ[0],
                raw.left_germ[1],
                raw.left_germ[2],
                raw.left_germ[3],
            ),
        ),
        corner=_FiniteCornerRead(
            value=jnp.where(eligible & supported_corner, corner_value, -jnp.inf),
            marginal=jnp.where(eligible & finite_corner, corner_marginal, 0.0),
            savings=jnp.broadcast_to(floor, corner_value.shape),
            supported=eligible & supported_corner,
            finite=eligible & finite_corner,
            invalid=invalid_corner,
        ),
        left_changed=~left_legal | ~raw.left_germ[0],
        coverage_missing=(
            row.invalid_numeric
            | left.invalid_numeric
            | raw.germ_invalid
            | paired_invalid
            | invalid_corner
            | ~jnp.isfinite(resources)
            | (eligible & ~(right_legal | finite_corner) & corner_in_row_maximum)
        ),
    )


def _filter_finite_survivors(*, survivors: BoolND, key: FloatND) -> BoolND:
    """Apply one exact ordinary polynomial stage within its surviving tie set."""
    stage = jnp.where(survivors, key, -jnp.inf)
    return survivors & (stage >= jnp.max(stage, axis=-1, keepdims=True))


def _select_finite_outer_profiles(*, profiles: _FiniteOuterProfile) -> _FiniteSelection:
    """Select complete O/kind tuples under pointwise corner and ordinary side rules."""
    right, left, corner = profiles.right, profiles.left, profiles.corner
    point_values = jnp.stack((right.value, corner.value), axis=-1)
    flat_values = point_values.reshape(*right.value.shape[:-1], -1)
    maximum = jnp.max(flat_values, axis=-1, keepdims=True)
    point_supported = jnp.stack((right.supported, corner.supported), axis=-1).reshape(
        *right.value.shape[:-1], -1
    )
    point_ties = point_supported & (flat_values >= maximum)
    canonical_point = jnp.argmax(point_ties, axis=-1, keepdims=True).astype(jnp.int32)
    point_corner = jnp.any(
        corner.supported & (corner.value >= maximum), axis=-1, keepdims=True
    )

    survivors = right.supported & (right.value >= maximum)
    for key in (
        right.germ[0].astype(right.value.dtype),
        right.germ[1],
        right.germ[2],
        right.germ[3],
    ):
        survivors = _filter_finite_survivors(survivors=survivors, key=key)
    right_undecided = jnp.sum(survivors, axis=-1, keepdims=True) > 1
    right_winner = jnp.argmax(survivors, axis=-1, keepdims=True).astype(jnp.int32)

    # Preserve the shared primitive exactly when no discarded/recovered side or
    # competitive corner changes an ordinary-only right-indistinguishable tie.
    ordinary_winner, _ = right_germ_winner(
        value=right.value, right_germ=right.germ, left_germ=left.germ
    )
    left_values = jnp.stack((left.value, corner.value), axis=-1)
    allowed_left = jnp.stack((left.supported, corner.supported), axis=-1)
    allowed_left &= survivors[..., None]
    flat_left = jnp.where(allowed_left, left_values, -jnp.inf).reshape(
        *right.value.shape[:-1], -1
    )
    left_maximum = jnp.max(flat_left, axis=-1, keepdims=True)
    left_corner = jnp.any(
        survivors & corner.supported & (corner.value >= left_maximum),
        axis=-1,
        keepdims=True,
    )
    changed_left = (
        jnp.any(survivors & profiles.left_changed, axis=-1, keepdims=True) | left_corner
    )
    left_ties = allowed_left.reshape(*right.value.shape[:-1], -1) & (
        flat_left >= left_maximum
    )
    canonical_left = jnp.argmax(left_ties, axis=-1, keepdims=True).astype(jnp.int32)
    left_survivors = survivors & left.supported & (left.value >= left_maximum)
    for key in (
        left.germ[0].astype(left.value.dtype),
        -left.germ[1],
        left.germ[2],
        -left.germ[3],
    ):
        left_survivors = _filter_finite_survivors(survivors=left_survivors, key=key)
    left_winner = jnp.where(
        left_corner,
        canonical_left // 2,
        jnp.argmax(left_survivors, axis=-1, keepdims=True).astype(jnp.int32),
    )
    left_winner = jnp.where(
        jnp.any(left_ties, axis=-1, keepdims=True), left_winner, right_winner
    )
    winner = jnp.where(
        point_corner,
        canonical_point // 2,
        jnp.where(
            right_undecided,
            jnp.where(changed_left, left_winner, ordinary_winner),
            right_winner,
        ),
    )

    def gather(leaf: FloatND | IntND | BoolND) -> FloatND | IntND | BoolND:
        return jnp.take_along_axis(leaf, winner, axis=-1)

    selected_right = jax.tree.map(gather, right)
    selected_left = jax.tree.map(gather, left)
    selected_corner = jax.tree.map(gather, corner)
    right_corner = point_corner & (canonical_point % 2 == 1)
    # A side's own complete tuple is selected after the owning O is fixed.
    left_corner_selected = selected_corner.supported & (
        ~selected_left.supported | (selected_corner.value > selected_left.value)
    )
    return _FiniteSelection(
        value=maximum,
        outer_index=winner,
        right=_FiniteSideSelection(
            ordinary=selected_right, corner=selected_corner, use_corner=right_corner
        ),
        left=_FiniteSideSelection(
            ordinary=selected_left,
            corner=selected_corner,
            use_corner=left_corner_selected,
        ),
    )


def _collapse_finite_selection(
    *,
    selection: _FiniteSelection,
) -> tuple[FloatND, FloatND, FloatND, FloatND]:
    """Apply the existing side masks to the already associated finite selection."""
    right, left = selection.right, selection.left
    right_marginal = jnp.where(
        right.use_corner, right.corner.marginal, right.ordinary.marginal
    )[..., 0]
    left_marginal = jnp.where(
        left.use_corner, left.corner.marginal, left.ordinary.marginal
    )[..., 0]
    right_dead = jnp.where(
        right.use_corner, ~right.corner.supported, right.ordinary.dead
    )[..., 0]
    left_dead = jnp.where(left.use_corner, ~left.corner.supported, left.ordinary.dead)[
        ..., 0
    ]
    maximum = selection.value[..., 0]
    poisoned = jnp.isnan(maximum)
    return (
        maximum,
        jnp.where(poisoned, jnp.nan, jnp.where(right_dead, 0.0, right_marginal)),
        jnp.where(poisoned, jnp.nan, jnp.where(left_dead, 0.0, left_marginal)),
        jnp.where(right_dead, 0.0, 1.0),
    )
