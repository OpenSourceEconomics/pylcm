"""Capacity-bounded refinement of an already owned pre-shock row domain."""

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp

from _lcm.egm.projected_continuation import (
    ProjectedEGMCarry,
    ProjectedRowRead,
    read_projected_row,
)
from _lcm.egm.query_data import QueryNodeKind
from _lcm.typing import PytreeChild
from lcm.typing import BoolND, FloatND, IntND, ScalarBool, ScalarFloat, ScalarInt

type _ProjectionChild = PytreeChild | ProjectedEGMCarry


class _ProjectionState(NamedTuple):
    """A bounded row and whether its represented smooth cells meet the budgets."""

    carry: ProjectedEGMCarry
    done: ScalarBool
    iteration: ScalarInt
    records_needed: ScalarInt
    slopes: FloatND | None

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[None, Iterable[_ProjectionChild]], _ProjectionState],
        tuple[None, tuple[_ProjectionChild, ...]],
    ]:
        """Retain the fixed keyed schema through original-buffer reconstruction."""
        return _unflatten_projection_state, (None, tuple(self))


def refine_projected_row(
    *,
    initial: ProjectedEGMCarry,
    reference: Callable[[ScalarFloat], ProjectedRowRead],
    accuracy: Literal["certified", "balanced"] = "certified",
    value_slope: Callable[[ScalarFloat], ScalarFloat] | None = None,
) -> tuple[ProjectedEGMCarry, ScalarInt, FloatND | None]:
    """Bisect failed smooth cells using the original selected-candidate reader.

    ``initial`` must contain all proven pullbacks and their owned jump sides.
    Refinement cannot prove that an undisclosed discontinuity does not exist.
    Missing topology, invalid references or exhausted capacity invalidate the
    mandatory status. Values and economic marginals have separate unit scales.
    With ``value_slope``, the derivative of the reference value, every finite
    record stores that slope and the row's level read is cubic Hermite.

    Returns:
        Tuple of the refined row, the record count its last bisection needed and
        the stored value slopes (None without ``value_slope``). The count exceeds
        the row capacity exactly when the capacity, and no other cause, refused
        that bisection.

    """
    if initial.grid.ndim != 1:
        raise ValueError("Projection refinement operates on one original passive row.")

    def step(current: _ProjectionState) -> _ProjectionState:
        hard_invalid, failed = _assess_cells(
            carry=current.carry,
            reference=reference,
            accuracy=accuracy,
            slopes=current.slopes,
        )
        any_failed = jnp.any(failed)
        room = (
            current.carry.valid_length + jnp.sum(failed, dtype=jnp.int32)
            <= current.carry.grid.size
        )
        needed = current.carry.valid_length + jnp.where(
            hard_invalid, 0, jnp.sum(failed, dtype=jnp.int32)
        )
        invalid = hard_invalid | (any_failed & ~room)
        carry = replace(current.carry, valid=current.carry.valid & ~invalid)

        def insert(
            row: tuple[ProjectedEGMCarry, FloatND | None],
        ) -> tuple[ProjectedEGMCarry, FloatND | None]:
            return _insert_projection_nodes(
                carry=row[0],
                failed=failed,
                reference=reference,
                slopes=row[1],
                value_slope=value_slope,
            )

        carry, slopes = jax.lax.cond(
            any_failed & room & ~hard_invalid,
            insert,
            lambda row: row,
            (carry, current.slopes),
        )
        return _ProjectionState(
            carry=carry,
            done=~any_failed | invalid | ~carry.valid,
            iteration=current.iteration + 1,
            records_needed=needed,
            slopes=slopes,
        )

    slopes = None
    if value_slope is not None:
        slopes = _record_slopes(
            value_slope=value_slope,
            grid=initial.grid,
            finite=(jnp.arange(initial.grid.size) < initial.valid_length)
            & (initial.node_kind == QueryNodeKind.FINITE),
        )

    state = jax.lax.while_loop(
        lambda current: ~current.done & (current.iteration < initial.grid.size),
        step,
        _ProjectionState(
            carry=initial,
            done=~initial.valid,
            iteration=jnp.asarray(0, dtype=jnp.int32),
            records_needed=initial.valid_length,
            slopes=slopes,
        ),
    )
    hard_invalid, failed = _assess_cells(
        carry=state.carry, reference=reference, accuracy=accuracy, slopes=state.slopes
    )
    return (
        replace(
            state.carry,
            valid=state.carry.valid & ~hard_invalid & ~jnp.any(failed),
        ),
        state.records_needed,
        state.slopes,
    )


def _record_slopes(
    *,
    value_slope: Callable[[ScalarFloat], ScalarFloat],
    grid: FloatND,
    finite: BoolND,
) -> FloatND:
    """Reference value slopes at finite records; other records store zero."""
    return jax.lax.map(
        lambda inputs: jax.lax.cond(inputs[1], value_slope, jnp.zeros_like, inputs[0]),
        (grid, finite),
    )


def _read_projected(
    *, carry: ProjectedEGMCarry, query: ScalarFloat, slopes: FloatND | None = None
) -> ProjectedRowRead:
    """The consumers' read; ``slopes`` stands in for an unpublished slope channel."""
    return read_projected_row(
        query=query,
        grid=carry.grid,
        value=carry.value,
        economic_marginal=carry.economic_marginal,
        owner_id=carry.owner_id,
        side_id=carry.side_id,
        node_kind=carry.node_kind,
        link_valid=carry.link_valid,
        valid_length=carry.valid_length,
        domain_lower=carry.domain_lower,
        domain_upper=carry.domain_upper,
        valid=carry.valid,
        value_slope=carry.value_slope_channel if slopes is None else slopes,
    )


def _assess_cells(
    *,
    carry: ProjectedEGMCarry,
    reference: Callable[[ScalarFloat], ProjectedRowRead],
    accuracy: Literal["certified", "balanced"],
    slopes: FloatND | None = None,
) -> tuple[ScalarBool, BoolND]:
    dtype = carry.grid.dtype
    eps = jnp.finfo(dtype).eps
    value_budget = jnp.maximum(1e-6 if accuracy == "balanced" else 1e-8, 64.0 * eps)
    marginal_budget = jnp.maximum(
        1e-4 if accuracy == "balanced" else 1e-6, 8.0 * jnp.sqrt(eps)
    )
    lower, upper = carry.grid[:-1], carry.grid[1:]
    active = jnp.arange(lower.size) < carry.valid_length - 1
    width = upper - lower
    smooth = active & (width > 0.0) & carry.link_valid
    missing = active & (width > 0.0) & ~carry.link_valid
    invalid_geometry = jnp.any(
        active & (~jnp.isfinite(lower) | ~jnp.isfinite(upper) | (width < 0.0))
    )
    # Endpoints are stored records; only interior probes inform the error.
    probes = jnp.stack(
        (lower + width / 4.0, lower + width / 2.0, lower + 3.0 * width / 4.0),
        axis=-1,
    )
    probes = jnp.where(smooth[:, None], probes, carry.domain_lower)

    def assess(query: ScalarFloat) -> tuple[ScalarBool, ScalarBool]:
        actual = reference(query)
        predicted = _read_projected(carry=carry, query=query, slopes=slopes)
        both_economic = jnp.isneginf(actual.value) & jnp.isneginf(predicted.value)
        finite = jnp.isfinite(actual.value) & jnp.isfinite(predicted.value)
        hard = (
            ~actual.valid
            | ~predicted.valid
            | (actual.owner_id != predicted.owner_id)
            | (actual.side_id != predicted.side_id)
            | ~(finite | both_economic)
        )
        safe_actual = jnp.where(finite, actual.value, 0.0)
        safe_predicted = jnp.where(finite, predicted.value, 0.0)
        value_error = jnp.abs(safe_predicted - safe_actual) / jnp.maximum(
            1.0, jnp.abs(safe_actual)
        )
        marginal_error = jnp.abs(
            predicted.economic_marginal - actual.economic_marginal
        ) / jnp.maximum(1.0, jnp.abs(actual.economic_marginal))
        return hard, (value_error > value_budget) | (marginal_error > marginal_budget)

    def assess_active(
        inputs: tuple[ScalarFloat, ScalarBool],
    ) -> tuple[ScalarBool, ScalarBool]:
        query, is_active = inputs
        return jax.lax.cond(
            is_active,
            assess,
            lambda _: (jnp.asarray(a=False), jnp.asarray(a=False)),
            query,
        )

    smooth_probes = jnp.broadcast_to(smooth[:, None], probes.shape).reshape(-1)
    hard, failed = jax.lax.map(assess_active, (probes.reshape(-1), smooth_probes))
    shape = probes.shape
    hard = hard.reshape(shape) & smooth[:, None]
    failed = failed.reshape(shape) & smooth[:, None]
    return (
        invalid_geometry
        | (jnp.any(missing) if accuracy == "certified" else False)
        | jnp.any(hard)
        | jnp.any(
            jnp.any(failed, axis=-1)
            & ((lower + width / 2.0 <= lower) | (lower + width / 2.0 >= upper))
        ),
        jnp.any(failed, axis=-1),
    )


def _insert_projection_nodes(
    *,
    carry: ProjectedEGMCarry,
    failed: BoolND,
    reference: Callable[[ScalarFloat], ProjectedRowRead],
    slopes: FloatND | None = None,
    value_slope: Callable[[ScalarFloat], ScalarFloat] | None = None,
) -> tuple[ProjectedEGMCarry, FloatND | None]:
    capacity = carry.grid.size
    middle = carry.grid[:-1] + (carry.grid[1:] - carry.grid[:-1]) / 2.0
    middle = jnp.where(failed, middle, carry.domain_lower)
    sample = jax.vmap(reference)(middle)
    insert = jnp.concatenate((failed, jnp.zeros((1,), dtype=bool)))
    active = jnp.arange(capacity) < carry.valid_length
    records = jnp.stack((active, insert), axis=-1).reshape(-1)
    positions = jnp.cumsum(records, dtype=jnp.int32) - 1
    slots = jnp.where(records, positions, capacity)

    def interleaved(
        *, values: FloatND | IntND | BoolND, inserted: FloatND | IntND | BoolND
    ) -> FloatND | IntND | BoolND:
        padding = jnp.zeros((1,), dtype=inserted.dtype)
        inserted = jnp.concatenate((inserted, padding))
        rows = jnp.stack((values, inserted), axis=-1).reshape(-1)
        return jnp.zeros_like(values).at[slots].set(rows, mode="drop")

    node_kind = jnp.where(
        jnp.isneginf(sample.value),
        QueryNodeKind.ECONOMIC_MINUS_INF,
        QueryNodeKind.FINITE,
    ).astype(jnp.int32)
    edge = jnp.concatenate((carry.link_valid, jnp.zeros((1,), dtype=bool)))
    edge_rows = jnp.repeat(edge, 2)
    links = jnp.zeros((capacity,), dtype=bool).at[slots].set(edge_rows, mode="drop")
    if slopes is not None and value_slope is not None:
        slopes = interleaved(
            values=slopes,
            inserted=_record_slopes(
                value_slope=value_slope,
                grid=middle,
                finite=failed & (node_kind == QueryNodeKind.FINITE),
            ),
        )
    return replace(
        carry,
        grid=interleaved(values=carry.grid, inserted=middle),
        value=interleaved(values=carry.value, inserted=sample.value),
        economic_marginal=interleaved(
            values=carry.economic_marginal, inserted=sample.economic_marginal
        ),
        owner_id=interleaved(values=carry.owner_id, inserted=sample.owner_id),
        side_id=interleaved(values=carry.side_id, inserted=sample.side_id),
        node_kind=interleaved(values=carry.node_kind, inserted=node_kind),
        link_valid=links[:-1],
        valid_length=carry.valid_length + jnp.sum(failed, dtype=jnp.int32),
        valid=carry.valid & jnp.all(~failed | sample.valid),
    ), slopes


def _flatten_projection_state(
    record: _ProjectionState,
) -> tuple[tuple[_ProjectionChild, ...], None]:
    return tuple(record), None


def _flatten_projection_state_with_keys(
    record: _ProjectionState,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, _ProjectionChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), leaf)
        for name, leaf in zip(_ProjectionState._fields, record, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_projection_state(
    metadata: None,
    children: Iterable[_ProjectionChild],
) -> _ProjectionState:
    """Restore native numeric, abstract or role leaves under the fixed schema."""
    del metadata
    leaves = tuple(children)
    if len(leaves) != len(_ProjectionState._fields):
        raise TypeError(
            "Native _ProjectionState reconstruction requires its exact fields."
        )
    return cast("_ProjectionState", tuple.__new__(_ProjectionState, leaves))


jax.tree_util.register_pytree_with_keys(
    _ProjectionState,
    _flatten_projection_state_with_keys,
    _unflatten_projection_state,
    _flatten_projection_state,
)
