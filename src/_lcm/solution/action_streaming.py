"""Blockwise action maximization for GridSearch solve kernels.

The streamed GridSearch route enumerates the canonical action product in fixed-width
blocks and combines them with a mergeable hard-max state. It preserves value,
feasibility, tie-breaking, and global action identity across block boundaries. The
collective route retains every stakeholder's value at one shared household winner;
the EV1 route hard-maxes continuous cells within each discrete prefix before logsum.
Compiler fusion, rematerialization, and allocation still determine measured runtime
and peak memory. Eligible folded routes stream actions before their unchanged full
quadrature reduction, and supported co-mapped routes preserve device-local continuation
reads. Co-map intersections with separate reference-value channels remain dense.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from _lcm.regime_building.collective import _weighted_sum
from _lcm.solution.action_reduction import (
    COLLECTIVE_HARD_MAX_REDUCTION,
    HARD_MAX_REDUCTION,
    LOGSUMEXP_REDUCTION,
    CollectiveHardMaxAccumulator,
    CollectiveHardMaxResult,
    HardMaxAccumulator,
    HardMaxResult,
    LogSumExpAccumulator,
    LogSumExpResult,
)
from _lcm.solution.logsumexp_action_reduction import (
    BoundLogSumExpReduction,
)

_INT32_MAX = 2_147_483_647
_COLLECTIVE_BLOCK_NDIM = 2
_Block = tuple[jax.Array, jax.Array, jax.Array]
_ScanCarry = tuple[HardMaxAccumulator, jax.Array]
_CollectiveBlock = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
_CollectiveScanCarry = tuple[CollectiveHardMaxAccumulator, jax.Array]


_EV1ScanCarry = tuple["_EV1ActionAccumulator", jax.Array]


class _EV1ActionAccumulator(NamedTuple):
    """Open discrete-branch group plus completed groups' exponential mass."""

    active_branch_group_id: jax.Array
    branch_group: HardMaxAccumulator
    completed_branch_groups: LogSumExpAccumulator


@dataclass(frozen=True)
class GridSearchEV1ActionReduction:
    """Composite reduction identity for streamed GridSearch EV1 values."""

    n_discrete_action_axes: int

    @property
    def semantic_key(self) -> tuple[object, ...]:
        """Identify the ordered branch-hard-max then log-sum-exp contract."""
        return (
            "grid-search-ev1-action-reduction",
            1,
            self.n_discrete_action_axes,
            HARD_MAX_REDUCTION.semantic_key,
            LOGSUMEXP_REDUCTION.semantic_key,
        )


def build_streaming_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[Any, Any]],
    action_names: tuple[str, ...],
    block_width: int,
) -> Callable[..., HardMaxResult]:
    """Build the fixed-state blockwise hard-max callable.

    ``action_names`` defines the canonical product order: the final action is the
    fastest-moving coordinate, exactly as C-order flattening of the corresponding
    product-map output.  ``block_width`` is an internal build decision standing in for
    a graph-wide planner choice.  It intentionally appears on neither a grid
    nor a solver's public configuration.

    The returned callable accepts one one-dimensional grid for each action name plus
    the scalar state, continuation, and parameter arguments consumed by ``Q_and_F``.
    It operates on one fixed state cell.  Callers may map it over state cells.

    At the source-program level, ``Q_and_F`` is vmapped over one block at a time and a
    padded final block is marked infeasible before reduction. The scan emits ``None``
    as its history. These source-level properties do not by themselves bound runtime
    or peak memory after compiler transformation. The reduction deliberately retains
    GridSearch's established feasible-NaN behavior:
    a NaN maximum publishes action identity zero, even when that action is infeasible.
    """
    _validate_streaming_configuration(
        action_names=action_names, block_width=block_width
    )
    return _StreamingHardMax(
        Q_and_F=Q_and_F,
        action_names=action_names,
        block_width=block_width,
    )


def build_streaming_ev1_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[Any, Any]],
    action_names: tuple[str, ...],
    n_discrete_action_axes: int,
    block_width: int,
    scale: Any,  # noqa: ANN401
) -> Callable[..., LogSumExpResult]:
    """Build the fixed-state EV1 expected-maximum callable.

    The leading discrete coordinates define contiguous branches in the canonical
    C-order product. Each branch is evaluated in fixed-width blocks over its trailing
    continuous coordinates, and padded cells never cross into the next branch.
    ``block_width`` is an upper bound: a shorter continuous branch uses its own extent.
    Exactly one finalized value per branch then enters a log-sum-exp reduction bound
    to ``scale`` for its complete lifetime.
    """
    _validate_streaming_configuration(
        action_names=action_names, block_width=block_width
    )
    if (
        not isinstance(n_discrete_action_axes, int)
        or isinstance(n_discrete_action_axes, bool)
        or not 1 <= n_discrete_action_axes <= len(action_names)
    ):
        raise ValueError(
            "n_discrete_action_axes must identify a non-empty leading action prefix"
        )
    return _StreamingEV1ExpectedMax(
        Q_and_F=Q_and_F,
        action_names=action_names,
        n_discrete_action_axes=n_discrete_action_axes,
        block_width=block_width,
        scale=scale,
    )


def build_streaming_collective_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[Any, Any]],
    action_names: tuple[str, ...],
    block_width: int,
    stakeholders: tuple[str, ...],
    weights: Mapping[str, Any],
) -> Callable[..., CollectiveHardMaxResult]:
    """Build the fixed-state collective hard-max callable.

    Each action cell is evaluated once through Q_and_F. Its trailing stakeholder
    vector is scalarized with the same zero-safe, canonical household objective as
    dense collective_readout. The collective reduction then retains every stakeholder
    value at one global C-order winner and keeps the empty feasible set explicit.
    """
    _validate_streaming_configuration(
        action_names=action_names, block_width=block_width
    )
    if not stakeholders:
        raise ValueError("stakeholders must not be empty")
    if len(set(stakeholders)) != len(stakeholders):
        raise ValueError("stakeholders must not contain duplicates")
    if set(stakeholders) != set(weights):
        raise ValueError("stakeholders and weights must have identical keys")
    return _StreamingCollectiveHardMax(
        Q_and_F=Q_and_F,
        action_names=action_names,
        block_width=block_width,
        stakeholders=stakeholders,
        weights=weights,
    )


def _validate_streaming_configuration(
    *, action_names: tuple[str, ...], block_width: int
) -> None:
    """Validate the common fixed-width action-product declaration."""
    if (
        not isinstance(block_width, int)
        or isinstance(block_width, bool)
        or block_width <= 0
    ):
        raise ValueError("block_width must be positive")
    if block_width > _INT32_MAX:
        raise ValueError("block_width exceeds the int32 identity range")
    if len(set(action_names)) != len(action_names):
        raise ValueError("action_names must not contain duplicates")


@dataclass(frozen=True)
class _StreamingHardMax:
    """Configured action-streaming callable."""

    Q_and_F: Callable[..., tuple[Any, Any]]
    action_names: tuple[str, ...]
    block_width: int

    def __call__(self, **kwargs: Any) -> HardMaxResult:  # noqa: ANN401
        if not self.action_names:
            return _reduce_no_action(Q_and_F=self.Q_and_F, kwargs=kwargs)

        action_grids, fixed_kwargs, action_sizes, n_actions = _prepare_action_call(
            action_names=self.action_names,
            kwargs=kwargs,
        )
        n_blocks = (n_actions + self.block_width - 1) // self.block_width
        evaluate_block = partial(
            _evaluate_block,
            Q_and_F=self.Q_and_F,
            action_names=self.action_names,
            action_grids=action_grids,
            action_sizes=action_sizes,
            fixed_kwargs=fixed_kwargs,
            n_actions=n_actions,
            block_width=self.block_width,
            block_offsets=jnp.arange(self.block_width, dtype=jnp.int32),
        )
        first_block = evaluate_block(block_index=jnp.asarray(0, dtype=jnp.int32))
        accumulator = _start_reduction(block=first_block)
        accumulator = _scan_remaining_blocks(
            accumulator=accumulator,
            evaluate_block=evaluate_block,
            n_remaining=n_blocks - 1,
        )
        return HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


@dataclass(frozen=True)
class _StreamingEV1ExpectedMax:
    """Configured discrete-branch hard-max followed by EV1 log-sum-exp."""

    Q_and_F: Callable[..., tuple[Any, Any]]
    action_names: tuple[str, ...]
    n_discrete_action_axes: int
    block_width: int
    scale: Any

    def __call__(self, **kwargs: Any) -> LogSumExpResult:  # noqa: ANN401
        action_grids, fixed_kwargs, action_sizes, n_actions = _prepare_action_call(
            action_names=self.action_names,
            kwargs=kwargs,
        )
        continuous_extent = math.prod(action_sizes[self.n_discrete_action_axes :])
        continuous_block_width = min(self.block_width, continuous_extent)
        n_discrete_branches = n_actions // continuous_extent
        branches_per_block = min(
            n_discrete_branches,
            max(1, self.block_width // continuous_block_width),
        )
        blocks_per_branch_group = (
            continuous_extent + continuous_block_width - 1
        ) // continuous_block_width
        n_branch_groups = (
            n_discrete_branches + branches_per_block - 1
        ) // branches_per_block
        n_blocks = n_branch_groups * blocks_per_branch_group
        reduction = LOGSUMEXP_REDUCTION.bind(scale=jnp.asarray(self.scale))
        evaluate_block = partial(
            _evaluate_ev1_branch_block,
            Q_and_F=self.Q_and_F,
            action_names=self.action_names,
            action_grids=action_grids,
            action_sizes=action_sizes,
            fixed_kwargs=fixed_kwargs,
            n_discrete_branches=n_discrete_branches,
            continuous_extent=continuous_extent,
            branches_per_block=branches_per_block,
            blocks_per_branch_group=blocks_per_branch_group,
            continuous_block_width=continuous_block_width,
            branch_offsets=jnp.arange(branches_per_block, dtype=jnp.int32),
            continuous_offsets=jnp.arange(
                continuous_block_width,
                dtype=jnp.int32,
            ),
        )
        first_block_index = jnp.asarray(0, dtype=jnp.int32)
        first_block = evaluate_block(block_index=first_block_index)
        accumulator = _initialize_ev1_reduction(
            branch_value_template=jnp.zeros_like(first_block[0][..., 0]),
            completed_value_template=jnp.zeros_like(first_block[0][0, 0]),
            reduction=reduction,
        )
        accumulator = _add_ev1_block(
            accumulator=accumulator,
            block=first_block,
            block_index=first_block_index,
            blocks_per_branch_group=blocks_per_branch_group,
            reduction=reduction,
        )
        accumulator = _scan_remaining_ev1_blocks(
            accumulator=accumulator,
            evaluate_block=evaluate_block,
            n_remaining=n_blocks - 1,
            blocks_per_branch_group=blocks_per_branch_group,
            reduction=reduction,
        )
        accumulator = _flush_ev1_branch_group(
            accumulator=accumulator,
            reduction=reduction,
        )
        return reduction.finalize(accumulator=accumulator.completed_branch_groups)


@dataclass(frozen=True)
class _StreamingCollectiveHardMax:
    """Configured collective action-streaming callable."""

    Q_and_F: Callable[..., tuple[Any, Any]]
    action_names: tuple[str, ...]
    block_width: int
    stakeholders: tuple[str, ...]
    weights: Mapping[str, Any]

    def __call__(self, **kwargs: Any) -> CollectiveHardMaxResult:  # noqa: ANN401
        if not self.action_names:
            return _reduce_collective_no_action(
                Q_and_F=self.Q_and_F,
                stakeholders=self.stakeholders,
                weights=self.weights,
                kwargs=kwargs,
            )

        action_grids, fixed_kwargs, action_sizes, n_actions = _prepare_action_call(
            action_names=self.action_names,
            kwargs=kwargs,
        )
        n_blocks = (n_actions + self.block_width - 1) // self.block_width
        evaluate_block = partial(
            _evaluate_collective_block,
            Q_and_F=self.Q_and_F,
            action_names=self.action_names,
            action_grids=action_grids,
            action_sizes=action_sizes,
            fixed_kwargs=fixed_kwargs,
            n_actions=n_actions,
            block_width=self.block_width,
            block_offsets=jnp.arange(self.block_width, dtype=jnp.int32),
            stakeholders=self.stakeholders,
            weights=self.weights,
        )
        first_block = evaluate_block(block_index=jnp.asarray(0, dtype=jnp.int32))
        accumulator = _start_collective_reduction(block=first_block)
        accumulator = _scan_remaining_collective_blocks(
            accumulator=accumulator,
            evaluate_block=evaluate_block,
            n_remaining=n_blocks - 1,
        )
        return COLLECTIVE_HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


def _prepare_action_call(
    *, action_names: tuple[str, ...], kwargs: dict[str, Any]
) -> tuple[tuple[jax.Array, ...], dict[str, Any], tuple[int, ...], int]:
    """Validate grids and split them from scalar Q arguments."""
    missing = tuple(name for name in action_names if name not in kwargs)
    if missing:
        raise TypeError(f"Missing action-grid arguments: {missing}")

    action_grids = tuple(jnp.asarray(kwargs[name]) for name in action_names)
    for name, grid in zip(action_names, action_grids, strict=True):
        if grid.ndim != 1:
            raise ValueError(f"Action grid '{name}' must be one-dimensional")
        if grid.shape[0] == 0:
            raise ValueError(f"Action grid '{name}' must not be empty")

    fixed_kwargs = {
        name: value for name, value in kwargs.items() if name not in action_names
    }
    action_sizes = tuple(grid.shape[0] for grid in action_grids)
    n_actions = math.prod(action_sizes)
    if n_actions > _INT32_MAX:
        raise ValueError(
            "The canonical action product exceeds the int32 identity range"
        )
    return action_grids, fixed_kwargs, action_sizes, n_actions


def _evaluate_block(
    *,
    block_index: jax.Array,
    Q_and_F: Callable[..., tuple[Any, Any]],
    action_names: tuple[str, ...],
    action_grids: tuple[jax.Array, ...],
    action_sizes: tuple[int, ...],
    fixed_kwargs: dict[str, Any],
    n_actions: int,
    block_width: int,
    block_offsets: jax.Array,
) -> _Block:
    """Evaluate one padded block, never the complete action product."""
    block_start = block_index * block_width
    remaining = n_actions - block_start
    valid = block_offsets < remaining
    safe_offsets = jnp.minimum(block_offsets, remaining - 1)
    global_ids = block_start + safe_offsets

    def evaluate_one(global_id: jax.Array) -> tuple[Any, Any]:
        action_kwargs = _decode_action(
            global_id=global_id,
            action_names=action_names,
            action_grids=action_grids,
            action_sizes=action_sizes,
        )
        return Q_and_F(**fixed_kwargs, **action_kwargs)

    values, feasible = jax.vmap(evaluate_one)(global_ids)
    values = jnp.asarray(values)
    feasible = jnp.asarray(feasible)
    _validate_block_Q_and_F(values=values, feasible=feasible)
    return values, feasible & valid, global_ids


def _evaluate_ev1_branch_block(
    *,
    block_index: jax.Array,
    Q_and_F: Callable[..., tuple[Any, Any]],
    action_names: tuple[str, ...],
    action_grids: tuple[jax.Array, ...],
    action_sizes: tuple[int, ...],
    fixed_kwargs: dict[str, Any],
    n_discrete_branches: int,
    continuous_extent: int,
    branches_per_block: int,
    blocks_per_branch_group: int,
    continuous_block_width: int,
    branch_offsets: jax.Array,
    continuous_offsets: jax.Array,
) -> _Block:
    """Evaluate one bounded block over complete or chunked EV1 branches."""
    branch_group_id = block_index // blocks_per_branch_group
    block_within_branch_group = block_index % blocks_per_branch_group

    branch_group_start = branch_group_id * branches_per_block
    remaining_branches = n_discrete_branches - branch_group_start
    valid_branches = branch_offsets < remaining_branches
    safe_branch_offsets = jnp.minimum(branch_offsets, remaining_branches - 1)
    safe_branch_ids = branch_group_start + safe_branch_offsets

    local_start = block_within_branch_group * continuous_block_width
    remaining = continuous_extent - local_start
    valid_continuous = continuous_offsets < remaining
    safe_continuous_offsets = jnp.minimum(continuous_offsets, remaining - 1)

    global_ids = (
        safe_branch_ids[:, jnp.newaxis] * continuous_extent
        + local_start
        + safe_continuous_offsets[jnp.newaxis, :]
    )
    valid = valid_branches[:, jnp.newaxis] & valid_continuous[jnp.newaxis, :]

    def evaluate_one(global_id: jax.Array) -> tuple[Any, Any]:
        action_kwargs = _decode_action(
            global_id=global_id,
            action_names=action_names,
            action_grids=action_grids,
            action_sizes=action_sizes,
        )
        return Q_and_F(**fixed_kwargs, **action_kwargs)

    values, feasible = jax.vmap(jax.vmap(evaluate_one))(global_ids)
    values = jnp.asarray(values)
    feasible = jnp.asarray(feasible)
    _validate_block_Q_and_F(
        values=values[0],
        feasible=feasible[0],
    )
    return values, feasible & valid, global_ids


def _evaluate_collective_block(
    *,
    block_index: jax.Array,
    Q_and_F: Callable[..., tuple[Any, Any]],
    action_names: tuple[str, ...],
    action_grids: tuple[jax.Array, ...],
    action_sizes: tuple[int, ...],
    fixed_kwargs: dict[str, Any],
    n_actions: int,
    block_width: int,
    block_offsets: jax.Array,
    stakeholders: tuple[str, ...],
    weights: Mapping[str, Any],
) -> _CollectiveBlock:
    """Evaluate and scalarize one padded collective action block."""
    block_start = block_index * block_width
    remaining = n_actions - block_start
    valid = block_offsets < remaining
    safe_offsets = jnp.minimum(block_offsets, remaining - 1)
    global_ids = block_start + safe_offsets

    def evaluate_one(global_id: jax.Array) -> tuple[Any, Any]:
        action_kwargs = _decode_action(
            global_id=global_id,
            action_names=action_names,
            action_grids=action_grids,
            action_sizes=action_sizes,
        )
        return Q_and_F(**fixed_kwargs, **action_kwargs)

    stakeholder_values, feasible = jax.vmap(evaluate_one)(global_ids)
    stakeholder_values = jnp.asarray(stakeholder_values)
    feasible = jnp.asarray(feasible)
    _validate_collective_block_Q_and_F(
        stakeholder_values=stakeholder_values,
        feasible=feasible,
        n_stakeholders=len(stakeholders),
    )
    objectives = _weighted_sum(
        stakeholder_Q={
            name: stakeholder_values[..., index]
            for index, name in enumerate(stakeholders)
        },
        weights=weights,
    )
    return objectives, stakeholder_values, feasible & valid, global_ids


def _start_reduction(*, block: _Block) -> HardMaxAccumulator:
    """Seed a hard-max reduction from the first evaluated block."""
    values, feasible, global_ids = block
    accumulator = HARD_MAX_REDUCTION.initialize(
        value_template=jnp.zeros_like(values[0])
    )
    return HARD_MAX_REDUCTION.add(
        accumulator=accumulator,
        values=values,
        feasible=feasible,
        action_ids=global_ids,
    )


def _scan_remaining_blocks(
    *,
    accumulator: HardMaxAccumulator,
    evaluate_block: Callable[..., _Block],
    n_remaining: int,
) -> HardMaxAccumulator:
    """Use a source-level scan whose returned history is the ``None`` pytree."""

    # keyword-only-exempt: library-callback=jax.lax.scan
    def scan_one_block(carry: _ScanCarry, _unused: None) -> tuple[_ScanCarry, None]:
        partial_accumulator, block_index = carry
        block = evaluate_block(block_index=block_index)
        partial_accumulator = _add_block(accumulator=partial_accumulator, block=block)
        return (partial_accumulator, block_index + 1), None

    (accumulator, _), _history = jax.lax.scan(
        scan_one_block,
        (accumulator, jnp.asarray(1, dtype=jnp.int32)),
        xs=None,
        length=n_remaining,
    )
    return accumulator


def _add_block(*, accumulator: HardMaxAccumulator, block: _Block) -> HardMaxAccumulator:
    """Merge one evaluated block into the hard-max state."""
    values, feasible, global_ids = block
    return HARD_MAX_REDUCTION.add(
        accumulator=accumulator,
        values=values,
        feasible=feasible,
        action_ids=global_ids,
    )


def _initialize_ev1_reduction(
    *,
    branch_value_template: jax.Array,
    completed_value_template: jax.Array,
    reduction: BoundLogSumExpReduction,
) -> _EV1ActionAccumulator:
    """Create an empty branch group and empty completed-group mass."""
    return _EV1ActionAccumulator(
        active_branch_group_id=jnp.asarray(-1, dtype=jnp.int32),
        branch_group=HARD_MAX_REDUCTION.initialize(
            value_template=branch_value_template
        ),
        completed_branch_groups=reduction.initialize(
            value_template=completed_value_template
        ),
    )


def _add_ev1_block(
    *,
    accumulator: _EV1ActionAccumulator,
    block: _Block,
    block_index: jax.Array,
    blocks_per_branch_group: int,
    reduction: BoundLogSumExpReduction,
) -> _EV1ActionAccumulator:
    """Merge one vector block, closing the preceding branch group."""
    values, feasible, global_ids = block
    branch_group_id = block_index // blocks_per_branch_group
    branch_group_changed = (accumulator.active_branch_group_id >= 0) & (
        accumulator.active_branch_group_id != branch_group_id
    )
    accumulator = jax.lax.cond(
        branch_group_changed,
        lambda current: _finalize_open_ev1_branch_group(
            accumulator=current,
            reduction=reduction,
        ),
        lambda current: current,
        accumulator,
    )
    branch_group = HARD_MAX_REDUCTION.add(
        accumulator=accumulator.branch_group,
        values=values,
        feasible=feasible,
        action_ids=global_ids,
    )
    return _EV1ActionAccumulator(
        active_branch_group_id=branch_group_id,
        branch_group=branch_group,
        completed_branch_groups=accumulator.completed_branch_groups,
    )


def _finalize_open_ev1_branch_group(
    *,
    accumulator: _EV1ActionAccumulator,
    reduction: BoundLogSumExpReduction,
) -> _EV1ActionAccumulator:
    """Move one vector of finalized branch values into log-sum-exp."""
    branch_group = HARD_MAX_REDUCTION.finalize(accumulator=accumulator.branch_group)
    completed_branch_groups = reduction.add(
        accumulator=accumulator.completed_branch_groups,
        values=branch_group.best_value,
    )
    return _EV1ActionAccumulator(
        active_branch_group_id=jnp.asarray(-1, dtype=jnp.int32),
        branch_group=HARD_MAX_REDUCTION.initialize(
            value_template=jnp.zeros_like(branch_group.best_value)
        ),
        completed_branch_groups=completed_branch_groups,
    )


def _scan_remaining_ev1_blocks(
    *,
    accumulator: _EV1ActionAccumulator,
    evaluate_block: Callable[..., _Block],
    n_remaining: int,
    blocks_per_branch_group: int,
    reduction: BoundLogSumExpReduction,
) -> _EV1ActionAccumulator:
    """Scan later vector blocks while keeping one branch group open."""

    # keyword-only-exempt: library-callback=jax.lax.scan
    def scan_one_block(
        carry: _EV1ScanCarry,
        _unused: None,
    ) -> tuple[_EV1ScanCarry, None]:
        partial, block_index = carry
        block = evaluate_block(block_index=block_index)
        partial = _add_ev1_block(
            accumulator=partial,
            block=block,
            block_index=block_index,
            blocks_per_branch_group=blocks_per_branch_group,
            reduction=reduction,
        )
        return (partial, block_index + 1), None

    (accumulator, _), _history = jax.lax.scan(
        scan_one_block,
        (accumulator, jnp.asarray(1, dtype=jnp.int32)),
        xs=None,
        length=n_remaining,
    )
    return accumulator


def _flush_ev1_branch_group(
    *,
    accumulator: _EV1ActionAccumulator,
    reduction: BoundLogSumExpReduction,
) -> _EV1ActionAccumulator:
    """Finalize the last non-padding branch group after the ordered scan."""
    return jax.lax.cond(
        accumulator.active_branch_group_id >= 0,
        lambda current: _finalize_open_ev1_branch_group(
            accumulator=current,
            reduction=reduction,
        ),
        lambda current: current,
        accumulator,
    )


def _reduce_no_action(
    *, Q_and_F: Callable[..., tuple[Any, Any]], kwargs: dict[str, Any]
) -> HardMaxResult:
    """Treat an empty action product as the one-cell identity product."""
    value, feasible = Q_and_F(**kwargs)
    value = jnp.asarray(value)
    feasible = jnp.asarray(feasible)
    _validate_scalar_Q_and_F(value=value, feasible=feasible)
    block = (
        value[jnp.newaxis],
        feasible[jnp.newaxis],
        jnp.array([0], dtype=jnp.int32),
    )
    accumulator = _start_reduction(block=block)
    return HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


def _start_collective_reduction(
    *, block: _CollectiveBlock
) -> CollectiveHardMaxAccumulator:
    """Seed a collective hard-max reduction from the first evaluated block."""
    objectives, stakeholder_values, feasible, global_ids = block
    accumulator = COLLECTIVE_HARD_MAX_REDUCTION.initialize(
        stakeholder_template=jnp.zeros_like(stakeholder_values[0])
    )
    return COLLECTIVE_HARD_MAX_REDUCTION.add(
        accumulator=accumulator,
        objectives=objectives,
        stakeholder_values=stakeholder_values,
        feasible=feasible,
        action_ids=global_ids,
    )


def _scan_remaining_collective_blocks(
    *,
    accumulator: CollectiveHardMaxAccumulator,
    evaluate_block: Callable[..., _CollectiveBlock],
    n_remaining: int,
) -> CollectiveHardMaxAccumulator:
    """Scan remaining collective blocks without retaining a block history."""

    # keyword-only-exempt: library-callback=jax.lax.scan
    def scan_one_block(
        carry: _CollectiveScanCarry, _unused: None
    ) -> tuple[_CollectiveScanCarry, None]:
        partial_accumulator, block_index = carry
        block = evaluate_block(block_index=block_index)
        partial_accumulator = _add_collective_block(
            accumulator=partial_accumulator,
            block=block,
        )
        return (partial_accumulator, block_index + 1), None

    (accumulator, _), _history = jax.lax.scan(
        scan_one_block,
        (accumulator, jnp.asarray(1, dtype=jnp.int32)),
        xs=None,
        length=n_remaining,
    )
    return accumulator


def _add_collective_block(
    *,
    accumulator: CollectiveHardMaxAccumulator,
    block: _CollectiveBlock,
) -> CollectiveHardMaxAccumulator:
    """Merge one collective block into the household hard-max state."""
    objectives, stakeholder_values, feasible, global_ids = block
    return COLLECTIVE_HARD_MAX_REDUCTION.add(
        accumulator=accumulator,
        objectives=objectives,
        stakeholder_values=stakeholder_values,
        feasible=feasible,
        action_ids=global_ids,
    )


def _reduce_collective_no_action(
    *,
    Q_and_F: Callable[..., tuple[Any, Any]],
    stakeholders: tuple[str, ...],
    weights: Mapping[str, Any],
    kwargs: dict[str, Any],
) -> CollectiveHardMaxResult:
    """Treat a collective empty action product as one shared identity cell."""
    stakeholder_values, feasible = Q_and_F(**kwargs)
    stakeholder_values = jnp.asarray(stakeholder_values)
    feasible = jnp.asarray(feasible)
    _validate_collective_scalar_Q_and_F(
        stakeholder_values=stakeholder_values,
        feasible=feasible,
        n_stakeholders=len(stakeholders),
    )
    objective = _weighted_sum(
        stakeholder_Q={
            name: stakeholder_values[index] for index, name in enumerate(stakeholders)
        },
        weights=weights,
    )
    block = (
        objective[jnp.newaxis],
        stakeholder_values[jnp.newaxis, :],
        feasible[jnp.newaxis],
        jnp.array([0], dtype=jnp.int32),
    )
    accumulator = _start_collective_reduction(block=block)
    return COLLECTIVE_HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


def _decode_action(
    *,
    global_id: jax.Array,
    action_names: tuple[str, ...],
    action_grids: tuple[jax.Array, ...],
    action_sizes: tuple[int, ...],
) -> dict[str, jax.Array]:
    """Decode a C-order global identity without materializing the product."""
    stride = math.prod(action_sizes)
    out: dict[str, jax.Array] = {}
    for name, grid, size in zip(action_names, action_grids, action_sizes, strict=True):
        stride //= size
        coordinate = (global_id // stride) % size
        out[name] = grid[coordinate]
    return out


def _validate_scalar_Q_and_F(*, value: jax.Array, feasible: jax.Array) -> None:
    """Validate the no-action identity against the streaming contract."""
    if value.ndim != 0 or feasible.ndim != 0:
        raise ValueError(
            "Ordinary-singleton action streaming requires scalar Q and "
            "feasibility outputs at each action cell"
        )
    if feasible.dtype != jnp.bool_:
        raise TypeError("Q_and_F feasibility output must have boolean dtype")


def _validate_block_Q_and_F(*, values: jax.Array, feasible: jax.Array) -> None:
    """Validate a vmapped block against the streaming contract."""
    if values.ndim != 1 or feasible.ndim != 1:
        raise ValueError(
            "Ordinary-singleton action streaming requires scalar Q and "
            "feasibility outputs at each action cell"
        )
    if feasible.dtype != jnp.bool_:
        raise TypeError("Q_and_F feasibility output must have boolean dtype")


def _validate_collective_scalar_Q_and_F(
    *,
    stakeholder_values: jax.Array,
    feasible: jax.Array,
    n_stakeholders: int,
) -> None:
    """Validate one collective action cell."""
    if (
        stakeholder_values.ndim != 1
        or stakeholder_values.shape[-1] != n_stakeholders
        or feasible.ndim != 0
    ):
        raise ValueError(
            "Collective action streaming requires one trailing stakeholder "
            "axis and scalar feasibility at each action cell"
        )
    if feasible.dtype != jnp.bool_:
        raise TypeError("Q_and_F feasibility output must have boolean dtype")


def _validate_collective_block_Q_and_F(
    *,
    stakeholder_values: jax.Array,
    feasible: jax.Array,
    n_stakeholders: int,
) -> None:
    """Validate a vmapped collective block against the streaming contract."""
    if (
        stakeholder_values.ndim != _COLLECTIVE_BLOCK_NDIM
        or stakeholder_values.shape[-1] != n_stakeholders
        or feasible.ndim != 1
        or stakeholder_values.shape[0] != feasible.shape[0]
    ):
        raise ValueError(
            "Collective action streaming requires one trailing stakeholder "
            "axis and scalar feasibility at each action cell"
        )
    if feasible.dtype != jnp.bool_:
        raise TypeError("Q_and_F feasibility output must have boolean dtype")
