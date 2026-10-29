"""Private derivative-aware scheduling over the original future lottery axes."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from _lcm.egm.continuation import (
    _block_node_indices,
    _ChildCarryReader,
    _ChildEulerState,
    _joint_node_weights,
    _node_index_mesh,
    _on_node_scale,
    _ReadAtNodes,
    _resolve_child_node_operands,
    _RowQueriesAndGradients,
    child_resources_params,
)
from _lcm.execution.reductions import (
    WEIGHTED_EXPECTATION_REDUCTION,
    BoundWeightedExpectationReduction,
    WeightedExpectationAccumulator,
)
from _lcm.probability import is_represented_zero, probabilities_form_distribution
from _lcm.solution.feasibility import (
    BoundedResult,
    ContinuationReadCounts,
    merge_continuation_read_counts,
    reduce_continuation_read_counts,
)
from _lcm.typing import EconFunctionKwargs
from _lcm.zero_safe import factored_weighted_term
from lcm.typing import (
    BoolND,
    FloatND,
    IntND,
    ReferenceName,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)


def read_future_source_direction(
    *,
    reader: _ChildCarryReader,
    direction: Mapping[ReferenceName, ScalarFloat],
    savings: ScalarFloat,
) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
    """Keep canonical primal reduction and explicit economic product derivatives."""
    source = _SourceNodeRead(reader=reader, direction=direction, savings=savings)
    names = tuple(direction)

    def weights(values: tuple[ScalarFloat, ...]) -> tuple[FloatND, ...]:
        pool = {**reader.combo_pool, **dict(zip(names, values, strict=True))}
        return _resolve_child_node_operands(read=reader.read, combo_pool=pool)[1]

    _, weight_directions = jax.jvp(
        weights,
        (tuple(reader.combo_pool[name] for name in names),),
        (tuple(direction[name] for name in names),),
    )
    indices = _node_index_mesh(stochastic_node_values=reader.stochastic_node_values)
    if len(reader.weight_vecs) != len(indices):
        raise TypeError(
            "Future node directions require the original factor per lottery."
        )
    n_nodes = indices[0].shape[0]
    width = (
        n_nodes
        if reader.stochastic_node_width is None
        else min(reader.stochastic_node_width, n_nodes)
    )
    blocks = _block_node_indices(flat_node_indices=indices, width=width)
    weights_at_nodes, shift = _joint_node_weights(
        weight_vecs=reader.weight_vecs, node_indices=indices
    )
    pad = blocks[0].size - n_nodes
    blocked_weights = jnp.pad(weights_at_nodes, (0, pad)).reshape(blocks[0].shape)
    padding_live = (jnp.arange(blocks[0].size) < n_nodes).reshape(blocks[0].shape)
    reduction = WEIGHTED_EXPECTATION_REDUCTION.bind(subnormal_is_accounted_for=True)
    zero = jnp.zeros_like(savings)
    valid = _future_law_valid(reader=reader, weight_directions=weight_directions)
    accumulated, _ = jax.lax.scan(
        _SourceNodeFold(
            source=source, reduction=reduction, weight_directions=weight_directions
        ),
        BoundedResult(
            output=(reduction.initialize(value_template=zero), zero, valid),
            bound_status=(),
            query_status=None,
            continuation_read_counts=ContinuationReadCounts(jnp.int32(0), jnp.int32(0))
            if reader.collect_continuation_read_counts
            else None,
        ),
        (blocks, blocked_weights, padding_live),
    )
    value_state, derivative, valid = accumulated.output
    value = _on_node_scale(
        values=reduction.finalize(accumulator=value_state).weighted_sum, shift=shift
    )
    valid &= jnp.isfinite(derivative) & (jnp.isfinite(value) | jnp.isneginf(value))
    return BoundedResult(
        output=(value, jnp.where(valid, derivative, jnp.nan)),
        bound_status=(),
        query_status=None,
        continuation_read_counts=accumulated.continuation_read_counts,
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _SourceNodeRead:
    """Compose live support motion with the original coordinate and row reader."""

    reader: _ChildCarryReader
    """Actual bound child, including its original support and prepared row bank."""
    direction: Mapping[ReferenceName, ScalarFloat]
    """Source liquid and continuous physical directions at fixed current bases."""
    savings: ScalarFloat
    """Original selected saving, fixed in the source economic envelope."""
    source_channel: bool = True
    """Whether the coordinate's source-direction JVP is traced and validated."""

    def __call__(
        self, indices: tuple[ScalarInt, ...]
    ) -> (
        tuple[ScalarFloat, ScalarFloat] | BoundedResult[tuple[ScalarFloat, ScalarFloat]]
    ):
        result = self.economic_directions(indices)
        value, _, source_direction = result.output
        return (
            BoundedResult(
                output=(value, source_direction),
                bound_status=result.bound_status,
                query_status=result.query_status,
                continuation_read_counts=result.continuation_read_counts,
            )
            if self.reader.collect_continuation_read_counts
            else (value, source_direction)
        )

    def economic_directions(
        self, indices: tuple[ScalarInt, ...]
    ) -> BoundedResult[tuple[ScalarFloat, ScalarFloat, ScalarFloat]]:
        """Preserve original finite-transform products and their operation order."""
        result = self.economic_components(indices)
        value, economic, savings_gradient, gradient, valid = result.output
        return BoundedResult(
            output=(
                value,
                jnp.where(valid, economic * savings_gradient, jnp.nan),
                jnp.where(valid, economic * gradient, jnp.nan),
            ),
            bound_status=result.bound_status,
            query_status=result.query_status,
            continuation_read_counts=result.continuation_read_counts,
        )

    def economic_components(
        self, indices: tuple[ScalarInt, ...]
    ) -> BoundedResult[
        tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat, ScalarBool]
    ]:
        """Keep economic marginal and coordinate JVPs separate until reduction."""
        reader = self.reader
        names = tuple(self.direction)

        def coordinate(values: tuple[ScalarFloat, ...]) -> ScalarFloat:
            pool = {**reader.combo_pool, **dict(zip(names, values, strict=True))}
            supports, _ = _resolve_child_node_operands(
                read=reader.read, combo_pool=pool
            )
            physical = tuple(
                support[index] for support, index in zip(supports, indices, strict=True)
            )
            return self.queries(pool=pool)(physical)[0]

        bases = tuple(reader.combo_pool[name] for name in names)
        if self.source_channel:
            query, gradient = jax.jvp(
                coordinate, (bases,), (tuple(self.direction[name] for name in names),)
            )
        else:
            query = coordinate(cast("tuple[ScalarFloat, ...]", bases))
            gradient = jnp.zeros_like(query)
        physical = tuple(
            support[index]
            for support, index in zip(
                reader.stochastic_node_values, indices, strict=True
            )
        )
        _, savings_gradient = self.queries(pool=reader.combo_pool)(physical)
        node_reader = _ReadAtNodes(
            read=reader.read,
            carry=reader.carry,
            prepared_search_grid=reader.prepared_search_grid,
            prepared_valid_length=reader.prepared_valid_length,
            stochastic_node_values=reader.stochastic_node_values,
            deterministic_index=(),
            carry_axis_stochastic_flags=reader.read.stochastic_flags,
            child_passive_values=(),
            queries_and_gradients=self.queries(pool=reader.combo_pool),
            shared_queries_and_gradients=None,
            risk_aversion=None,
            collect_continuation_read_counts=reader.collect_continuation_read_counts,
        )
        result = node_reader.read_coordinates(
            node_indices=indices,
            queries=query,
            gradients=jnp.ones_like(query),
        )
        value, economic = result.output if isinstance(result, BoundedResult) else result
        valid = jnp.isfinite(query) & jnp.isfinite(savings_gradient)
        if self.source_channel:
            valid &= jnp.isfinite(gradient)
        return BoundedResult(
            output=(value, economic, savings_gradient, gradient, valid),
            bound_status=(),
            query_status=result.query_status
            if isinstance(result, BoundedResult)
            else None,
            continuation_read_counts=result.continuation_read_counts
            if isinstance(result, BoundedResult)
            else None,
        )

    def queries(self, *, pool: EconFunctionKwargs) -> _RowQueriesAndGradients:
        read = self.reader.read
        return _RowQueriesAndGradients(
            read=read,
            child_euler_state=_ChildEulerState(
                next_state_func=read.euler_state_func,
                combo_pool=pool,
                post_decision_name=self.reader.post_decision_name,
                next_state_key=read.next_state_key,
            ),
            deterministic_resources_kwargs={},
            resources_param_kwargs=child_resources_params(read=read, combo_pool=pool),
            savings_value=self.savings,
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class _SourceNodeFold:
    """Fold original node reads without deleting derivative-needed null events."""

    source: _SourceNodeRead
    """Original node-row consumption with explicit coordinate directions."""
    reduction: BoundWeightedExpectationReduction
    """Canonical scaled primal accumulator, unchanged by the derivative channel."""
    weight_directions: tuple[FloatND, ...]
    """Original per-lottery weight derivatives before products or masks."""

    # keyword-only-exempt: library-callback=jax.lax.scan
    def __call__(
        self,
        state: BoundedResult[
            tuple[WeightedExpectationAccumulator, ScalarFloat, ScalarBool]
        ],
        block: tuple[tuple[IntND, ...], FloatND, BoolND],
    ) -> tuple[
        BoundedResult[tuple[WeightedExpectationAccumulator, ScalarFloat, ScalarBool]],
        None,
    ]:
        indices, weights, padding_live = block
        factors = jnp.stack(
            tuple(
                vec[index]
                for vec, index in zip(
                    self.source.reader.weight_vecs, indices, strict=True
                )
            )
        )
        tangents = jnp.stack(
            tuple(
                vec[index]
                for vec, index in zip(self.weight_directions, indices, strict=True)
            )
        )
        live = jnp.all(~is_represented_zero(factors), axis=0)
        derivative_needed = jnp.zeros_like(live)
        for column in range(factors.shape[0]):
            others = jnp.concatenate((factors[:column], factors[column + 1 :]), axis=0)
            derivative_needed |= ~is_represented_zero(tangents[column]) & jnp.all(
                ~is_represented_zero(others), axis=0
            )
        needed = padding_live & (live | derivative_needed)
        result = jax.vmap(self.source)(indices)
        values, directions = (
            result.output if isinstance(result, BoundedResult) else result
        )
        # Padding is an extra constant factor, never a declaration about the law.
        factors = jnp.concatenate(
            (factors, padding_live[None].astype(factors.dtype)), axis=0
        )
        terms = factored_weighted_term(factors=factors, values=directions)
        for column in range(tangents.shape[0]):
            differentiated = factors.at[column].set(tangents[column])
            terms += factored_weighted_term(factors=differentiated, values=values)
        value_state, total, valid = state.output
        valid &= jnp.all(~needed | jnp.isfinite(directions))
        valid &= jnp.all(~(padding_live & derivative_needed) | jnp.isfinite(values))
        counts = state.continuation_read_counts
        if isinstance(result, BoundedResult):
            if result.bound_status or result.query_status is not None:
                raise TypeError(
                    "Future economic node fold requires its original scalar row."
                )
            if result.continuation_read_counts is not None:
                counts = merge_continuation_read_counts(
                    left=counts,
                    right=reduce_continuation_read_counts(
                        counts=result.continuation_read_counts, active=needed
                    ),
                )
        return BoundedResult(
            output=(
                self.reduction.add(
                    accumulator=value_state, values=values, weights=weights
                ),
                total + jnp.sum(terms),
                valid,
            ),
            bound_status=(),
            query_status=None,
            continuation_read_counts=counts,
        ), None


def _future_law_valid(
    *, reader: _ChildCarryReader, weight_directions: tuple[FloatND, ...]
) -> ScalarBool:
    valid = jnp.ones((), dtype=jnp.bool_)
    for support in reader.stochastic_node_values:
        valid &= jnp.all(jnp.isfinite(support))
    for weights, tangent in zip(reader.weight_vecs, weight_directions, strict=True):
        valid &= probabilities_form_distribution(
            probabilities=tuple(weights), dtype=weights.dtype
        )
        valid &= jnp.all(jnp.isfinite(tangent))
    return valid
