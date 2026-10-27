"""Private scalar future-law directions using original economic child reads."""

from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from _lcm.certainty_equivalent import aggregates_nonlinearly
from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import (
    ContinuationPlan,
    _ChildCarryReader,
    _ChildEulerState,
    _ChildRead,
    _composed_simple_resources,
    _get_child_carry_reader,
    child_resources_params,
)
from _lcm.egm.current_iid_future import read_future_source_direction
from _lcm.egm.projected_child import ProjectedChildReader
from _lcm.egm.projected_continuation import ProjectedEGMCarry
from _lcm.probability import is_represented_zero, probabilities_form_distribution
from _lcm.solution.feasibility import (
    BoundedResult,
    ContinuationReadCounts,
    QueryInputStatus,
    merge_continuation_read_counts,
    reduce_continuation_read_counts,
    reduce_query_input_status,
)
from _lcm.typing import EconFunctionKwargs
from _lcm.zero_safe import factored_weighted_term
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import ReferenceName, RegimeName, ScalarFloat


def read_case_u_continuation(
    *,
    plan: ContinuationPlan,
    combo_pool: EconFunctionKwargs,
    direction: Mapping[ReferenceName, ScalarFloat],
    children: MappingProxyType[RegimeName, EGMCarry | ProjectedEGMCarry],
    savings: ScalarFloat,
    dtype: DTypeLike,
    collect_read_counts: bool = False,
    stochastic_node_width: int | None = None,
) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
    """Read linear continuation and its source direction at fixed savings.

    Only original coordinate/law callbacks are differentiated. The separately
    published child marginal supplies the value direction. Future node supports
    and weight factors use the original lottery axes. Passive coordinates and
    conditional queries require economic capabilities beyond this scalar read.
    """
    if plan.risk_aversion_param_name is not None or aggregates_nonlinearly(
        plan.certainty_equivalent
    ):
        raise UnsupportedOperationError(
            "Case-U nonlinear economic CE is not installed."
        )
    names = tuple(direction)
    primals = tuple(combo_pool[name] for name in names)
    tangents = tuple(direction[name] for name in names)

    def probabilities(
        values: tuple[ScalarFloat, ...],
    ) -> Mapping[RegimeName, ScalarFloat]:
        return plan.compute_regime_transition_probs(
            **{**combo_pool, **dict(zip(names, values, strict=True))}
        )

    probabilities_at_source, probability_direction = jax.jvp(
        probabilities, (primals,), (tangents,)
    )
    targets = (*plan.stateful_targets, *plan.scalar_targets)
    valid = probabilities_form_distribution(
        probabilities=(probabilities_at_source[target] for target in targets),
        dtype=dtype,
    )
    value = jnp.asarray(0.0, dtype=dtype)
    marginal = jnp.asarray(0.0, dtype=dtype)
    statuses: list[QueryInputStatus] = []
    counts: ContinuationReadCounts | None = None
    for target in targets:
        probability = probabilities_at_source[target]
        derivative = probability_direction[target]
        needed = ~is_represented_zero(probability) | ~is_represented_zero(derivative)
        if target in plan.stateful_targets:
            result = _read_source_child(
                plan=plan,
                target=target,
                combo_pool=combo_pool,
                direction=direction,
                carry=children[target],
                savings=savings,
                collect_read_counts=collect_read_counts,
                stochastic_node_width=stochastic_node_width,
            )
            _require_child_slots(result=result, target=target, plan=plan)
            if isinstance(result, BoundedResult):
                if result.bound_status:
                    raise TypeError(
                        "Economic child directions cannot own parent bound slots."
                    )
                child_value, child_direction = result.output
                if result.query_status:
                    statuses.extend(
                        reduce_query_input_status(
                            statuses=result.query_status, active=needed
                        )
                        or ()
                    )
                if result.continuation_read_counts is not None:
                    counts = merge_continuation_read_counts(
                        left=counts,
                        right=reduce_continuation_read_counts(
                            counts=result.continuation_read_counts, active=needed
                        ),
                    )
            else:
                child_value, child_direction = result
        else:
            child_value = children[target].value[0]
            child_direction = jnp.asarray(0.0, dtype=dtype)
        valid &= jnp.isfinite(derivative)
        valid &= ~needed | jnp.isfinite(child_direction)
        valid &= is_represented_zero(derivative) | jnp.isfinite(child_value)
        value += factored_weighted_term(
            factors=jnp.stack((probability,)), values=child_value
        )
        marginal += factored_weighted_term(
            factors=jnp.stack((probability,)), values=child_direction
        )
        marginal += factored_weighted_term(
            factors=jnp.stack((derivative,)), values=child_value
        )
    valid &= jnp.isfinite(marginal) & (jnp.isfinite(value) | jnp.isneginf(value))
    inactive = jnp.zeros((), dtype=jnp.bool_)
    return BoundedResult(
        output=(value, marginal),
        bound_status=(),
        query_status=(*statuses, QueryInputStatus(inactive, inactive, ~valid)),
        continuation_read_counts=counts,
    )


def _read_source_child(
    *,
    plan: ContinuationPlan,
    target: RegimeName,
    combo_pool: EconFunctionKwargs,
    direction: Mapping[ReferenceName, ScalarFloat],
    carry: EGMCarry | ProjectedEGMCarry,
    savings: ScalarFloat,
    collect_read_counts: bool,
    stochastic_node_width: int | None,
) -> tuple[ScalarFloat, ScalarFloat] | BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
    """Compose the actual scalar coordinate callback with the canonical row."""
    read = plan.child_reads[target]
    _require_scalar_child(read=read)
    reader = _get_child_carry_reader(
        read=read,
        carry=carry,
        combo_pool=combo_pool,
        post_decision_name=plan.post_decision_name,
        stochastic_node_width=stochastic_node_width,
        collect_continuation_read_counts=collect_read_counts,
    )
    if isinstance(reader, _ChildCarryReader) and (
        read.stochastic_state_names or read.local_draw_names
    ):
        return read_future_source_direction(
            reader=reader, direction=direction, savings=savings
        )
    names = tuple(direction)

    def coordinate(values: tuple[ScalarFloat, ...]) -> ScalarFloat:
        pool = {**combo_pool, **dict(zip(names, values, strict=True))}
        child_state = _ChildEulerState(
            next_state_func=read.euler_state_func,
            combo_pool=pool,
            post_decision_name=plan.post_decision_name,
            next_state_key=read.next_state_key,
        )
        if isinstance(carry, ProjectedEGMCarry):
            return child_state(savings)
        return _composed_simple_resources(
            savings,
            read=read,
            child_euler_state=child_state,
            resources_param_kwargs=child_resources_params(read=read, combo_pool=pool),
        )

    query, query_direction = jax.jvp(
        coordinate,
        (tuple(combo_pool[name] for name in names),),
        (tuple(direction[name] for name in names),),
    )
    if isinstance(reader, ProjectedChildReader):
        return reader.read_coordinate(query=query, jacobian=query_direction)
    if not isinstance(reader, _ChildCarryReader):
        raise TypeError("Case-U requires its actual canonical child reader.")
    return reader.read_scalar_coordinate(query=query, gradient=query_direction)


def _require_scalar_child(*, read: _ChildRead) -> None:
    """Refuse coordinates whose required economic capabilities are unrepresented."""
    if (
        read.query is not None
        or any(not flag for flag in read.stochastic_flags)
        or read.passive_state_names
        or read.row_arg_names
        or read.row_block_shape
        or read.n_outer_candidates
        or read.has_taste_shocks
        or read.co_map_state_names
        or any(name is not None for name in read.process_grid_names)
    ):
        raise UnsupportedOperationError(
            "Case-U economic future directions require scalar economic rows "
            "without passive, choice, query, deterministic-code or unresolved "
            "process-grid coordinates."
        )


def _require_child_slots(
    *,
    result: tuple[ScalarFloat, ScalarFloat]
    | BoundedResult[tuple[ScalarFloat, ScalarFloat]],
    target: RegimeName,
    plan: ContinuationPlan,
) -> None:
    """Keep the original query target as authority for child slot presence."""
    if target in plan.query_targets:
        if (
            not isinstance(result, BoundedResult)
            or result.query_status is None
            or len(result.query_status) != 1
        ):
            raise TypeError("Economic child direction lacks its installed slot.")
    elif isinstance(result, BoundedResult) and result.query_status is not None:
        raise TypeError("Economic child direction lacks a declared query target.")
