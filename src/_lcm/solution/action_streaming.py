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
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax._src.interpreters import partial_eval

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

    @property
    def exactness(self) -> Literal["tolerance_equivalent"]:
        """Return `"tolerance_equivalent"`: the branch mass is a floating-point sum."""
        return "tolerance_equivalent"


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


@dataclass(frozen=True, kw_only=True)
class ActionPartitionLayout:
    """Contiguous runs of whole action blocks, one run per participant.

    The canonical action product is cut into `n_blocks` blocks of `block_width`
    identities, exactly as the unpartitioned stream cuts it. Partition `p` owns
    blocks `[p * n_blocks // P, (p + 1) * n_blocks // P)` for `P` partitions,
    so the partitions are ascending, disjoint intervals of global identities
    that together cover the product once, and their block counts differ by at
    most one. A partition is empty only when there are fewer blocks than
    participants.
    """

    n_actions: int
    """Number of identities in the canonical action product."""

    block_width: int
    """Identities per block, the planner-bound width of the action axis."""

    n_partitions: int
    """Number of participants sharing the product."""

    def __post_init__(self) -> None:
        """Require positive exact integers."""
        for label, value in (
            ("n_actions", self.n_actions),
            ("block_width", self.block_width),
            ("n_partitions", self.n_partitions),
        ):
            _fail_if_not_positive_int(label=label, value=value)
        if self.n_partitions * self.n_blocks > _INT32_MAX:
            raise ValueError(
                "n_partitions times the number of action blocks exceeds the int32 "
                "range the traced block ranges are computed in"
            )

    @property
    def n_blocks(self) -> int:
        """Return the number of blocks covering the product, the last padded."""
        return -(-self.n_actions // self.block_width)

    @property
    def blocks_per_partition(self) -> int:
        """Return how many blocks every participant evaluates, padding included.

        The longest run; a participant with a shorter run evaluates infeasible
        padding blocks for the rest.
        """
        return -(-self.n_blocks // self.n_partitions)

    def block_range(self, *, partition: int) -> tuple[int, int]:
        """Return the half-open range of block indices one participant owns."""
        return (
            partition * self.n_blocks // self.n_partitions,
            (partition + 1) * self.n_blocks // self.n_partitions,
        )

    def action_interval(self, *, partition: int) -> tuple[int, int]:
        """Return the half-open global identity interval one participant owns."""
        start_block, stop_block = self.block_range(partition=partition)
        return (
            min(start_block * self.block_width, self.n_actions),
            min(stop_block * self.block_width, self.n_actions),
        )


def build_partitioned_streaming_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[Any, Any]],
    action_names: tuple[str, ...],
    block_width: int,
    n_partitions: int,
    axis_name: str,
) -> _PartitionedStreamingHardMax:
    """Build the fixed-state hard max whose actions are shared by participants.

    The returned callable runs inside a mapped context binding `axis_name` to
    `n_partitions` members — a manual mesh axis of a `shard_map`, or a named
    `vmap`. Each member reads its partition from the axis index, reduces the
    blocks `ActionPartitionLayout` assigns it with the exact hard-max
    accumulator, gathers every member's accumulator over the axis and merges
    them in partition order. Every member therefore publishes the same
    `HardMaxResult`, equal to the unpartitioned stream at the same
    `block_width`: blocks are those of the unpartitioned stream, padded slots
    are infeasible, and the merge law is exact.

    Only one accumulator per member and state cell crosses the axis; the
    action values themselves never do.
    """
    _validate_streaming_configuration(
        action_names=action_names, block_width=block_width
    )
    _fail_if_not_positive_int(label="n_partitions", value=n_partitions)
    if not action_names:
        raise ValueError(
            "A partitioned action reduction requires a non-empty action product"
        )
    return _PartitionedStreamingHardMax(
        Q_and_F=Q_and_F,
        action_names=action_names,
        block_width=block_width,
        n_partitions=n_partitions,
        axis_name=axis_name,
    )


def merge_partition_accumulators(
    *, accumulators: HardMaxAccumulator, order: tuple[int, ...]
) -> HardMaxResult:
    """Merge per-partition accumulators along their leading axis, then finalize.

    `order` lists every partition once; the merge visits them in that order.
    The hard-max merge is exact, commutative and associative, so every order
    publishes the same bits. Production merges in ascending partition order so
    the program is one fixed sequence.
    """
    n_partitions = accumulators.best_value.shape[0]
    if sorted(order) != list(range(n_partitions)):
        msg = (
            f"order must list each of the {n_partitions} partitions once; "
            f"got {order!r}."
        )
        raise ValueError(msg)
    merged = jax.tree.map(lambda leaf: leaf[order[0]], accumulators)
    for partition in order[1:]:
        merged = HARD_MAX_REDUCTION.merge(
            left=merged,
            right=jax.tree.map(lambda leaf, index=partition: leaf[index], accumulators),
        )
    return HARD_MAX_REDUCTION.finalize(accumulator=merged)


def _fail_if_not_positive_int(*, label: str, value: object) -> None:
    """Require a positive exact integer, refusing bools and floats."""
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact int; got {value!r}")
    if value <= 0:
        raise ValueError(f"{label} must be positive; got {value!r}")


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
        accumulator = _scan_blocks(
            accumulator=_empty_reduction(evaluate_block=evaluate_block),
            evaluate_block=evaluate_block,
            n_blocks=n_blocks,
        )
        return HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


@dataclass(frozen=True)
class _PartitionedStreamingHardMax:
    """Configured action-partitioned hard-max callable."""

    Q_and_F: Callable[..., tuple[Any, Any]]
    action_names: tuple[str, ...]
    block_width: int
    n_partitions: int
    axis_name: str

    def __call__(self, **kwargs: Any) -> HardMaxResult:  # noqa: ANN401
        """Reduce this member's partition, exchange accumulators and merge them."""
        partition = jax.lax.axis_index(self.axis_name).astype(jnp.int32)
        local = self.local(partition=partition, **kwargs)
        gathered = jax.lax.all_gather(local, self.axis_name)
        return merge_partition_accumulators(
            accumulators=gathered, order=tuple(range(self.n_partitions))
        )

    def local(self, *, partition: jax.Array, **kwargs: Any) -> HardMaxAccumulator:  # noqa: ANN401
        """Reduce the blocks one partition owns into an unfinalized accumulator."""
        action_grids, fixed_kwargs, action_sizes, n_actions = _prepare_action_call(
            action_names=self.action_names,
            kwargs=kwargs,
        )
        layout = ActionPartitionLayout(
            n_actions=n_actions,
            block_width=self.block_width,
            n_partitions=self.n_partitions,
        )
        # The traced form of `ActionPartitionLayout.block_range`.
        n_blocks = jnp.int32(layout.n_blocks)
        first_block_index = partition * n_blocks // self.n_partitions
        stop_block_index = (partition + 1) * n_blocks // self.n_partitions
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
        # Every participant evaluates each block inside the scan body, exactly
        # as the unpartitioned stream does, so each action's value comes from
        # the same traced expression on either route. The scan starts at the
        # participant's first block and runs the longest run's length; a block
        # past its own run is padding whose every slot is infeasible, so an
        # empty partition keeps the empty accumulator.
        (accumulator, _), _history = jax.lax.scan(
            partial(
                _scan_one_block,
                evaluate_block=partial(
                    _evaluate_owned_block,
                    first_block_index=first_block_index,
                    stop_block_index=stop_block_index,
                    evaluate_block=evaluate_block,
                ),
            ),
            (_empty_reduction(evaluate_block=evaluate_block), first_block_index),
            xs=None,
            length=layout.blocks_per_partition,
        )
        return accumulator


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
        block = _trace_block(evaluate_block=evaluate_block)
        values = block.shapes[0]
        accumulator = _typed_ev1_reductions(
            accumulator=_initialize_ev1_reduction(
                branch_value_template=jnp.zeros(values.shape[:-1], dtype=values.dtype),
                completed_value_template=jnp.zeros(
                    values.shape[2:], dtype=values.dtype
                ),
                reduction=reduction,
            ),
            arrays=block.read_arrays,
        )
        accumulator = _scan_ev1_blocks(
            accumulator=accumulator,
            evaluate_block=evaluate_block,
            n_blocks=n_blocks,
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
        block = _trace_block(evaluate_block=evaluate_block)
        stakeholder_values = block.shapes[1]
        accumulator = _scan_collective_blocks(
            accumulator=_typed_like(
                accumulator=COLLECTIVE_HARD_MAX_REDUCTION.initialize(
                    stakeholder_template=jnp.zeros(
                        stakeholder_values.shape[1:], dtype=stakeholder_values.dtype
                    )
                ),
                arrays=block.read_arrays,
            ),
            evaluate_block=evaluate_block,
            n_blocks=n_blocks,
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

    evaluate_one = partial(
        _evaluate_one_action,
        Q_and_F=Q_and_F,
        action_names=action_names,
        action_grids=action_grids,
        action_sizes=action_sizes,
        fixed_kwargs=fixed_kwargs,
    )
    values, feasible = jax.vmap(evaluate_one)(global_ids)
    values = jnp.asarray(values)
    feasible = jnp.asarray(feasible)
    _validate_block_Q_and_F(values=values, feasible=feasible)
    return values, feasible & valid, global_ids


def _evaluate_owned_block(
    *,
    block_index: jax.Array,
    first_block_index: jax.Array,
    stop_block_index: jax.Array,
    evaluate_block: Callable[..., _Block],
) -> _Block:
    """Evaluate one block, marking it infeasible outside the partition's run."""
    values, feasible, global_ids = evaluate_block(block_index=block_index)
    owned = (first_block_index <= block_index) & (block_index < stop_block_index)
    return values, feasible & owned, global_ids


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

    evaluate_one = partial(
        _evaluate_one_action,
        Q_and_F=Q_and_F,
        action_names=action_names,
        action_grids=action_grids,
        action_sizes=action_sizes,
        fixed_kwargs=fixed_kwargs,
    )
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

    evaluate_one = partial(
        _evaluate_one_action,
        Q_and_F=Q_and_F,
        action_names=action_names,
        action_grids=action_grids,
        action_sizes=action_sizes,
        fixed_kwargs=fixed_kwargs,
    )
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


# keyword-only-exempt: library-callback=jax.vmap
def _evaluate_one_action(
    global_id: jax.Array,
    *,
    Q_and_F: Callable[..., tuple[Any, Any]],
    action_names: tuple[str, ...],
    action_grids: tuple[jax.Array, ...],
    action_sizes: tuple[int, ...],
    fixed_kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    """Evaluate ``Q_and_F`` at one global C-order action identity."""
    action_kwargs = _decode_action(
        global_id=global_id,
        action_names=action_names,
        action_grids=action_grids,
        action_sizes=action_sizes,
    )
    return Q_and_F(**fixed_kwargs, **action_kwargs)


class _BlockTrace(NamedTuple):
    """One block's output types and the outer arrays its outputs read."""

    shapes: tuple[jax.ShapeDtypeStruct, ...]
    read_arrays: list[jax.Array]


def _trace_block(
    *, evaluate_block: Callable[..., tuple[jax.Array, ...]]
) -> _BlockTrace:
    """Trace one block without staging it into the surrounding program.

    The empty accumulator is seeded from the shapes, so every block is
    evaluated in the scan body and `Q_and_F` appears once in the staged
    program. The read arrays are the surrounding program's values that reach
    the block's outputs, by the same dead-code elimination `jit` applies, so an
    array passed into a nested call that ignores it is not read.
    """
    closed = jax.make_jaxpr(partial(evaluate_block, block_index=jnp.int32(0)))()
    _, used = partial_eval.dce_jaxpr(
        partial_eval.convert_constvars_jaxpr(closed.jaxpr), used_outputs=True
    )
    return _BlockTrace(
        shapes=tuple(
            jax.ShapeDtypeStruct(aval.shape, aval.dtype) for aval in closed.out_avals
        ),
        read_arrays=[
            const
            for const, is_used in zip(
                closed.consts, used[: len(closed.consts)], strict=True
            )
            if is_used and isinstance(const, jax.Array)
        ],
    )


def _empty_reduction(*, evaluate_block: Callable[..., _Block]) -> HardMaxAccumulator:
    """Create the empty hard-max accumulator for the blocks `evaluate_block` makes."""
    block = _trace_block(evaluate_block=evaluate_block)
    values = block.shapes[0]
    return _typed_like(
        accumulator=HARD_MAX_REDUCTION.initialize(
            value_template=jnp.zeros(values.shape[1:], dtype=values.dtype)
        ),
        arrays=block.read_arrays,
    )


def _typed_like[Accumulator](
    *, accumulator: Accumulator, arrays: list[jax.Array]
) -> Accumulator:
    """Give every leaf of `accumulator` the batching and varying type of `arrays`.

    An empty accumulator is built from constants, while each block reduced into
    it reads `arrays`. Under `vmap` or `shard_map` the scan carry would then
    change type on its first step. Each leaf is selected through a predicate
    that reads every array and is always false, which keeps its value and gives
    it the arrays' type; the compiler folds the predicate and the selection
    away.
    """
    never = jnp.zeros((), dtype=bool)
    for array in arrays:
        never = never & jnp.any(jnp.not_equal(array, array))
    return jax.tree.map(lambda leaf: jnp.where(never, leaf, leaf), accumulator)


def _start_reduction(*, block: _Block) -> HardMaxAccumulator:
    """Seed a hard-max reduction from one evaluated block."""
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


def _scan_blocks(
    *,
    accumulator: HardMaxAccumulator,
    evaluate_block: Callable[..., _Block],
    n_blocks: int,
) -> HardMaxAccumulator:
    """Scan every block in order; the returned history is the ``None`` pytree."""
    (accumulator, _), _history = jax.lax.scan(
        partial(_scan_one_block, evaluate_block=evaluate_block),
        (accumulator, jnp.asarray(0, dtype=jnp.int32)),
        xs=None,
        length=n_blocks,
    )
    return accumulator


# keyword-only-exempt: library-callback=jax.lax.scan
def _scan_one_block(
    carry: _ScanCarry,
    _unused: None,
    *,
    evaluate_block: Callable[..., _Block],
) -> tuple[_ScanCarry, None]:
    """Evaluate the carried block index and merge it into the hard-max state."""
    partial_accumulator, block_index = carry
    block = evaluate_block(block_index=block_index)
    partial_accumulator = _add_block(accumulator=partial_accumulator, block=block)
    return (partial_accumulator, block_index + 1), None


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
        partial(_finalize_ev1_branch_group_operand, reduction=reduction),
        _keep_ev1_accumulator,
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


# keyword-only-exempt: library-callback=jax.lax.cond
def _finalize_ev1_branch_group_operand(
    accumulator: _EV1ActionAccumulator,
    *,
    reduction: BoundLogSumExpReduction,
) -> _EV1ActionAccumulator:
    """Close the open branch group of a ``lax.cond`` operand.

    The reopened empty branch group is built from constants, so it takes the
    operand's type to match the other branch, which returns the operand.
    """
    return _typed_ev1_reductions(
        accumulator=_finalize_open_ev1_branch_group(
            accumulator=accumulator,
            reduction=reduction,
        ),
        arrays=jax.tree.leaves(accumulator.branch_group),
    )


def _typed_ev1_reductions(
    *, accumulator: _EV1ActionAccumulator, arrays: list[jax.Array]
) -> _EV1ActionAccumulator:
    """Give both reductions of an EV1 accumulator the type of `arrays`.

    The open group's id keeps its own type: it follows the block index alone.
    """
    return _EV1ActionAccumulator(
        active_branch_group_id=accumulator.active_branch_group_id,
        branch_group=_typed_like(accumulator=accumulator.branch_group, arrays=arrays),
        completed_branch_groups=_typed_like(
            accumulator=accumulator.completed_branch_groups, arrays=arrays
        ),
    )


def _keep_ev1_accumulator(
    accumulator: _EV1ActionAccumulator,
) -> _EV1ActionAccumulator:
    """Return a ``lax.cond`` operand unchanged."""
    return accumulator


def _scan_ev1_blocks(
    *,
    accumulator: _EV1ActionAccumulator,
    evaluate_block: Callable[..., _Block],
    n_blocks: int,
    blocks_per_branch_group: int,
    reduction: BoundLogSumExpReduction,
) -> _EV1ActionAccumulator:
    """Scan every vector block in order while keeping one branch group open."""
    (accumulator, _), _history = jax.lax.scan(
        partial(
            _scan_one_ev1_block,
            evaluate_block=evaluate_block,
            blocks_per_branch_group=blocks_per_branch_group,
            reduction=reduction,
        ),
        (accumulator, jnp.asarray(0, dtype=jnp.int32)),
        xs=None,
        length=n_blocks,
    )
    return accumulator


# keyword-only-exempt: library-callback=jax.lax.scan
def _scan_one_ev1_block(
    carry: _EV1ScanCarry,
    _unused: None,
    *,
    evaluate_block: Callable[..., _Block],
    blocks_per_branch_group: int,
    reduction: BoundLogSumExpReduction,
) -> tuple[_EV1ScanCarry, None]:
    """Evaluate the carried block index and merge it into the open branch group."""
    partial_accumulator, block_index = carry
    block = evaluate_block(block_index=block_index)
    partial_accumulator = _add_ev1_block(
        accumulator=partial_accumulator,
        block=block,
        block_index=block_index,
        blocks_per_branch_group=blocks_per_branch_group,
        reduction=reduction,
    )
    return (partial_accumulator, block_index + 1), None


def _flush_ev1_branch_group(
    *,
    accumulator: _EV1ActionAccumulator,
    reduction: BoundLogSumExpReduction,
) -> _EV1ActionAccumulator:
    """Finalize the last non-padding branch group after the ordered scan."""
    return jax.lax.cond(
        accumulator.active_branch_group_id >= 0,
        partial(_finalize_ev1_branch_group_operand, reduction=reduction),
        _keep_ev1_accumulator,
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
    """Seed a collective hard-max reduction from one evaluated block."""
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


def _scan_collective_blocks(
    *,
    accumulator: CollectiveHardMaxAccumulator,
    evaluate_block: Callable[..., _CollectiveBlock],
    n_blocks: int,
) -> CollectiveHardMaxAccumulator:
    """Scan every collective block in order without retaining a block history."""
    (accumulator, _), _history = jax.lax.scan(
        partial(_scan_one_collective_block, evaluate_block=evaluate_block),
        (accumulator, jnp.asarray(0, dtype=jnp.int32)),
        xs=None,
        length=n_blocks,
    )
    return accumulator


# keyword-only-exempt: library-callback=jax.lax.scan
def _scan_one_collective_block(
    carry: _CollectiveScanCarry,
    _unused: None,
    *,
    evaluate_block: Callable[..., _CollectiveBlock],
) -> tuple[_CollectiveScanCarry, None]:
    """Evaluate the carried block index and merge it into the household state."""
    partial_accumulator, block_index = carry
    block = evaluate_block(block_index=block_index)
    partial_accumulator = _add_collective_block(
        accumulator=partial_accumulator,
        block=block,
    )
    return (partial_accumulator, block_index + 1), None


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
