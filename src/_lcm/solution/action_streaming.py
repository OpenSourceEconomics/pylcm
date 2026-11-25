"""Blockwise action maximization for GridSearch solve kernels.

The streamed GridSearch route enumerates the canonical action product in fixed-width
blocks and combines them with a mergeable hard-max state. It preserves value,
feasibility, tie-breaking, and global action identity across block boundaries, and
retains every member's value at one shared household winner. Compiler fusion,
rematerialization, and allocation still determine measured runtime and peak
memory. Supported co-mapped routes preserve device-local continuation
reads. Co-map intersections with separate reference-value channels remain dense.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple, Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
from jax._src.interpreters import partial_eval

from _lcm.regime_building.collective import _weighted_sum
from _lcm.regime_building.Q_and_F import BoundedQAndF, has_action_integrity
from _lcm.solution.action_reduction import (
    COLLECTIVE_HARD_MAX_REDUCTION,
    CollectiveHardMaxAccumulator,
    CollectiveHardMaxResult,
)
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    QueryInputStatus,
    merge_bound_input_status,
    merge_query_input_status,
    reduce_bound_input_status,
    reduce_query_input_status,
)
from _lcm.typing import PytreeValue, QAndFArg
from _lcm.utils.dispatchers import productmap, vary_carry_like_body
from lcm.typing import ActionName, BoolND, FloatND, ReferenceName

_INT32_MAX = 2_147_483_647
_COLLECTIVE_BLOCK_NDIM = 2
# A block is (values, feasible, ids); the household's puts its objectives before
# the member values.
type _Block = tuple[jax.Array, ...]
type _BoundedBlock = BoundedResult[_Block]
type _ActionOutput = tuple[FloatND, BoolND]
type _BoundedActionEvaluator = Callable[..., BoundedResult[_ActionOutput]]
type _ScanCarry = tuple[CollectiveHardMaxAccumulator, jax.Array]


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


def build_partitioned_streaming_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    action_names: tuple[ActionName, ...],
    block_width: int,
    n_partitions: int,
    axis_name: str,
    members: tuple[str, ...],
    weights: Mapping[str, FloatND | float],
) -> _PartitionedStreamingHardMax:
    """Build the fixed-state hard max whose actions are shared by participants.

    The returned callable runs inside a mapped context binding `axis_name` to
    `n_partitions` participants — a manual mesh axis of a `shard_map`, or a
    named `vmap`. Each participant reads its partition from the axis index,
    reduces the blocks `ActionPartitionLayout` assigns it with the exact
    hard-max accumulator, gathers every participant's accumulator over the axis
    and merges them in partition order. Every participant therefore publishes
    the same `CollectiveHardMaxResult`, equal to the unpartitioned stream of
    `build_streaming_collective_max_Q_over_a` at the same `block_width`: blocks
    are those of the unpartitioned stream, padded slots are infeasible, and the
    merge law is exact.

    Only one accumulator per participant and state cell crosses the axis; the
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
        members=members,
        weights=weights,
    )


def merge_partition_accumulators(
    *, accumulators: CollectiveHardMaxAccumulator, order: tuple[int, ...]
) -> CollectiveHardMaxResult:
    """Merge per-partition accumulators along their leading axis, then finalize.

    `order` lists every partition once; the merge visits them in that order.
    The household hard-max merge is exact, commutative and associative, so
    every order publishes the same bits. Production merges in ascending
    partition order so the program is one fixed sequence.
    """
    n_partitions = accumulators.best_global_action_id.shape[0]
    if sorted(order) != list(range(n_partitions)):
        msg = (
            f"order must list each of the {n_partitions} partitions once; "
            f"got {order!r}."
        )
        raise ValueError(msg)
    return _merge_in_order(
        reduction=COLLECTIVE_HARD_MAX_REDUCTION,
        accumulators=accumulators,
        order=order,
    )


@runtime_checkable
class _ExactMerge[Accumulator, Result](Protocol):
    """An exact reduction's merge of two accumulators and its finalization."""

    def merge(self, *, left: Accumulator, right: Accumulator) -> Accumulator: ...

    def finalize(self, *, accumulator: Accumulator) -> Result: ...


def _merge_in_order[Accumulator, Result](
    *,
    reduction: _ExactMerge[Accumulator, Result],
    accumulators: Accumulator,
    order: tuple[int, ...],
) -> Result:
    """Merge the stacked accumulators in `order`, then finalize."""
    merged = jax.tree.map(lambda leaf: leaf[order[0]], accumulators)
    for partition in order[1:]:
        merged = reduction.merge(
            left=merged,
            right=jax.tree.map(lambda leaf, index=partition: leaf[index], accumulators),
        )
    return reduction.finalize(accumulator=merged)


def _fail_if_not_positive_int(*, label: str, value: int) -> None:
    """Require a positive exact integer, refusing bools and floats."""
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact int; got {value!r}")
    if value <= 0:
        raise ValueError(f"{label} must be positive; got {value!r}")


def build_streaming_collective_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    action_names: tuple[ActionName, ...],
    block_width: int,
    members: tuple[str, ...],
    weights: Mapping[str, FloatND | float],
) -> Callable[..., CollectiveHardMaxResult | BoundedResult[CollectiveHardMaxResult]]:
    """Build the fixed-state collective hard-max callable.

    Each action cell is evaluated once through Q_and_F. Its trailing member
    vector is scalarized with the same zero-safe, canonical household objective as
    dense collective_readout. The collective reduction then retains every member
    value at one global C-order winner and keeps the empty feasible set explicit.
    A block covering the whole product maps the named grids and reduces that
    product as one block, as `build_streaming_max_Q_over_a` does.
    """
    _validate_streaming_configuration(
        action_names=action_names, block_width=block_width
    )
    if not members:
        raise ValueError("members must not be empty")
    if len(set(members)) != len(members):
        raise ValueError("members must not contain duplicates")
    if set(members) != set(weights):
        raise ValueError("members and weights must have identical keys")
    return _StreamingCollectiveHardMax(
        Q_and_F=Q_and_F,
        action_names=action_names,
        block_width=block_width,
        members=members,
        weights=weights,
    )


def _validate_streaming_configuration(
    *, action_names: tuple[ActionName, ...], block_width: int
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


def _bounded_hard_max(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    action_names: tuple[ActionName, ...],
    block_width: int,
    kwargs: dict[ReferenceName, QAndFArg],
    members: tuple[str, ...],
    weights: Mapping[str, FloatND | float],
) -> BoundedResult[CollectiveHardMaxResult]:
    """OR actual action-window validity beside the existing hard-max accumulator."""
    if not action_names:
        result = cast("_BoundedActionEvaluator", BoundedQAndF(function=Q_and_F))(
            **kwargs
        )
        values, feasible = result.output
        block_values = values[None, ...]
        return BoundedResult(
            output=COLLECTIVE_HARD_MAX_REDUCTION.finalize(
                accumulator=_start_collective_reduction(
                    block=(
                        _household_objectives(
                            member_values=block_values, members=members, weights=weights
                        ),
                        block_values,
                        feasible[None],
                        jnp.asarray([0], dtype=jnp.int32),
                    )
                )
            ),
            bound_status=result.bound_status,
            query_status=result.query_status,
        )
    grids, fixed, sizes, n_actions = _prepare_action_call(
        action_names=action_names,
        kwargs=kwargs,
    )
    width = min(block_width, n_actions)
    evaluate_block = partial(
        _evaluate_collective_block,
        evaluate_block=partial(
            _evaluate_block,
            Q_and_F=Q_and_F,
            action_names=action_names,
            action_grids=grids,
            action_sizes=sizes,
            fixed_kwargs=fixed,
            n_actions=n_actions,
            block_width=width,
            block_offsets=jnp.arange(width, dtype=jnp.int32),
            n_members=len(members),
        ),
        members=members,
        weights=weights,
    )
    carry = _scan_bounded_blocks(
        evaluate_block=cast("Callable[..., _BoundedBlock]", evaluate_block),
        first_block_index=jnp.int32(0),
        n_blocks=(n_actions + width - 1) // width,
    )
    return BoundedResult(
        output=COLLECTIVE_HARD_MAX_REDUCTION.finalize(accumulator=carry.output),
        bound_status=carry.bound_status,
        query_status=carry.query_status,
    )


def _scan_bounded_blocks(
    *,
    evaluate_block: Callable[..., _BoundedBlock],
    first_block_index: jax.Array,
    n_blocks: int,
) -> BoundedResult[CollectiveHardMaxAccumulator]:
    """Reduce `n_blocks` blocks from `first_block_index` on, ORing their records.

    Every block runs in the scan body, so `Q_and_F` is staged once; the carry
    starts from the empty accumulator and all-false bound records.
    """
    initial = BoundedResult(
        output=_empty_collective_reduction(evaluate_block=evaluate_block),
        bound_status=_empty_bound_status(evaluate_block=evaluate_block),
        query_status=_empty_query_status(evaluate_block=evaluate_block),
    )

    # keyword-only-exempt: library-callback=jax.lax.scan
    def add_window(
        carry: BoundedResult[CollectiveHardMaxAccumulator], offset: jax.Array
    ) -> tuple[BoundedResult[CollectiveHardMaxAccumulator], None]:
        block = evaluate_block(block_index=first_block_index + offset)
        return BoundedResult(
            output=_add_block(accumulator=carry.output, block=block.output),
            bound_status=tuple(
                merge_bound_input_status(left=left, right=right)
                for left, right in zip(
                    carry.bound_status, block.bound_status, strict=True
                )
            ),
            query_status=merge_query_input_status(
                left=carry.query_status, right=block.query_status
            ),
        ), None

    result, _history = jax.lax.scan(
        add_window,
        vary_carry_like_body(
            body=lambda state: add_window(state, jnp.int32(0))[0], carry=initial
        ),
        jnp.arange(n_blocks, dtype=jnp.int32),
    )
    return result


@dataclass(frozen=True)
class _PartitionedStreamingHardMax:
    """Configured action-partitioned hard-max callable."""

    Q_and_F: Callable[..., tuple[FloatND, BoolND]]
    action_names: tuple[ActionName, ...]
    block_width: int
    n_partitions: int
    axis_name: str
    members: tuple[str, ...]
    weights: Mapping[str, FloatND | float]

    def __call__(
        self, **kwargs: QAndFArg
    ) -> CollectiveHardMaxResult | BoundedResult[CollectiveHardMaxResult]:
        """Reduce this participant's partition, exchange accumulators, merge them."""
        partition = jax.lax.axis_index(self.axis_name).astype(jnp.int32)
        local = self.local(partition=partition, **kwargs)
        gathered = jax.lax.all_gather(local, self.axis_name)
        if isinstance(gathered, BoundedResult):
            return BoundedResult(
                output=merge_partition_accumulators(
                    accumulators=gathered.output,
                    order=tuple(range(self.n_partitions)),
                ),
                bound_status=reduce_bound_input_status(
                    statuses=gathered.bound_status,
                    active=jnp.ones((self.n_partitions,), dtype=bool),
                ),
                query_status=reduce_query_input_status(
                    statuses=gathered.query_status,
                    active=jnp.ones((self.n_partitions,), dtype=bool),
                ),
            )
        return merge_partition_accumulators(
            accumulators=gathered, order=tuple(range(self.n_partitions))
        )

    def local(
        self, *, partition: jax.Array, **kwargs: QAndFArg
    ) -> CollectiveHardMaxAccumulator | BoundedResult[CollectiveHardMaxAccumulator]:
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
        # The half-open range of block indices this participant owns.
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
            n_members=len(self.members),
        )
        # The household scalarizes each block and reduces it with its own law.
        evaluate_block = partial(
            _evaluate_collective_block,
            evaluate_block=evaluate_block,
            members=self.members,
            weights=self.weights,
        )
        if has_action_integrity(function=self.Q_and_F):
            return _scan_bounded_blocks(
                evaluate_block=cast(
                    "Callable[..., _BoundedBlock]",
                    partial(
                        _evaluate_owned_block,
                        first_block_index=first_block_index,
                        stop_block_index=stop_block_index,
                        evaluate_block=evaluate_block,
                    ),
                ),
                first_block_index=first_block_index,
                n_blocks=layout.blocks_per_partition,
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
                evaluate_block=cast(
                    "Callable[..., _Block]",
                    partial(
                        _evaluate_owned_block,
                        first_block_index=first_block_index,
                        stop_block_index=stop_block_index,
                        evaluate_block=evaluate_block,
                    ),
                ),
            ),
            (
                _empty_collective_reduction(evaluate_block=evaluate_block),
                first_block_index,
            ),
            xs=None,
            length=layout.blocks_per_partition,
        )
        return accumulator


@dataclass(frozen=True)
class _StreamingCollectiveHardMax:
    """Configured collective action-streaming callable."""

    Q_and_F: Callable[..., tuple[FloatND, BoolND]]
    action_names: tuple[ActionName, ...]
    block_width: int
    members: tuple[str, ...]
    weights: Mapping[str, FloatND | float]

    def __call__(
        self, **kwargs: QAndFArg
    ) -> CollectiveHardMaxResult | BoundedResult[CollectiveHardMaxResult]:
        if has_action_integrity(function=self.Q_and_F):
            return _bounded_hard_max(
                Q_and_F=self.Q_and_F,
                action_names=self.action_names,
                block_width=self.block_width,
                kwargs=kwargs,
                members=self.members,
                weights=self.weights,
            )
        if not self.action_names:
            return _reduce_collective_no_action(
                Q_and_F=self.Q_and_F,
                members=self.members,
                weights=self.weights,
                kwargs=kwargs,
            )

        action_grids, fixed_kwargs, action_sizes, n_actions = _prepare_action_call(
            action_names=self.action_names,
            kwargs=kwargs,
        )
        if self.block_width >= n_actions:
            return _reduce_collective_whole_product(
                Q_and_F=self.Q_and_F,
                action_names=self.action_names,
                n_actions=n_actions,
                members=self.members,
                weights=self.weights,
                kwargs=kwargs,
            )
        n_blocks = (n_actions + self.block_width - 1) // self.block_width
        evaluate_block = partial(
            _evaluate_collective_block,
            evaluate_block=partial(
                _evaluate_block,
                Q_and_F=self.Q_and_F,
                action_names=self.action_names,
                action_grids=action_grids,
                action_sizes=action_sizes,
                fixed_kwargs=fixed_kwargs,
                n_actions=n_actions,
                block_width=self.block_width,
                block_offsets=jnp.arange(self.block_width, dtype=jnp.int32),
                n_members=len(self.members),
            ),
            members=self.members,
            weights=self.weights,
        )
        accumulator = _scan_blocks(
            accumulator=_empty_collective_reduction(evaluate_block=evaluate_block),
            evaluate_block=cast("Callable[..., _Block]", evaluate_block),
            n_blocks=n_blocks,
        )
        return COLLECTIVE_HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


def _prepare_action_call(
    *,
    action_names: tuple[ActionName, ...],
    kwargs: dict[ReferenceName, QAndFArg],
) -> tuple[tuple[jax.Array, ...], dict[ReferenceName, QAndFArg], tuple[int, ...], int]:
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
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    action_names: tuple[ActionName, ...],
    action_grids: tuple[jax.Array, ...],
    action_sizes: tuple[int, ...],
    fixed_kwargs: dict[ReferenceName, QAndFArg],
    n_actions: int,
    block_width: int,
    block_offsets: jax.Array,
    n_members: int,
) -> _Block | _BoundedBlock:
    """Evaluate one padded block, never the complete action product.

    Each action's value carries a trailing axis of `n_members` members.
    """
    block_start = block_index * block_width
    remaining = n_actions - block_start
    valid = block_offsets < remaining
    safe_offsets = jnp.minimum(block_offsets, remaining - 1)
    global_ids = block_start + safe_offsets

    bounded = bool(has_action_integrity(function=Q_and_F))
    evaluate_one = partial(
        _evaluate_one_action,
        Q_and_F=cast("_BoundedActionEvaluator", BoundedQAndF(function=Q_and_F))
        if bounded
        else Q_and_F,
        action_names=action_names,
        action_grids=action_grids,
        action_sizes=action_sizes,
        fixed_kwargs=fixed_kwargs,
    )
    evaluated = jax.vmap(evaluate_one)(global_ids)
    if bounded:
        bounded_result = cast("BoundedResult[_ActionOutput]", evaluated)
        values, feasible = bounded_result.output
        statuses = bounded_result.bound_status
    else:
        values, feasible = cast("_ActionOutput", evaluated)
        statuses = ()
    values = jnp.asarray(values)
    feasible = jnp.asarray(feasible)
    _validate_collective_block_Q_and_F(
        member_values=values, feasible=feasible, n_members=n_members
    )
    output = (values, feasible & valid, global_ids)
    return (
        BoundedResult(
            output=output,
            bound_status=reduce_bound_input_status(
                statuses=statuses,
                active=valid,
            ),
            query_status=reduce_query_input_status(
                statuses=bounded_result.query_status, active=valid
            ),
        )
        if bounded
        else output
    )


def _evaluate_owned_block(
    *,
    block_index: jax.Array,
    first_block_index: jax.Array,
    stop_block_index: jax.Array,
    evaluate_block: Callable[..., _Block | _BoundedBlock],
) -> _Block | _BoundedBlock:
    """Evaluate one block, marking it infeasible outside the partition's run.

    The block's last two entries are its feasibility and its global identities;
    a household's block carries its objectives and member values before them.
    """
    evaluated = evaluate_block(block_index=block_index)
    *values, feasible, global_ids = (
        evaluated.output if isinstance(evaluated, BoundedResult) else evaluated
    )
    owned = (first_block_index <= block_index) & (block_index < stop_block_index)
    output = (*values, feasible & owned, global_ids)
    return (
        BoundedResult(
            output=output,
            bound_status=tuple(
                BoundInputStatus(
                    nonfinite_floor=owned & status.nonfinite_floor,
                    invalid_shifted_grid=owned & status.invalid_shifted_grid,
                )
                for status in evaluated.bound_status
            ),
            query_status=reduce_query_input_status(
                statuses=evaluated.query_status, active=owned
            ),
        )
        if isinstance(evaluated, BoundedResult)
        else output
    )


def _evaluate_collective_block(
    *,
    block_index: jax.Array,
    evaluate_block: Callable[..., _Block | _BoundedBlock],
    members: tuple[str, ...],
    weights: Mapping[str, FloatND | float],
) -> _Block | _BoundedBlock:
    """Evaluate one padded household block and put its objective first."""
    evaluated = evaluate_block(block_index=block_index)
    bounded = isinstance(evaluated, BoundedResult)
    member_values, feasible, global_ids = evaluated.output if bounded else evaluated
    output = (
        _household_objectives(
            member_values=member_values, members=members, weights=weights
        ),
        member_values,
        feasible,
        global_ids,
    )
    return replace(evaluated, output=output) if bounded else output


def _household_objectives(
    *,
    member_values: jax.Array,
    members: tuple[str, ...],
    weights: Mapping[str, FloatND | float],
) -> jax.Array:
    """Return the household objective of each action's trailing member values."""
    return _weighted_sum(
        member_Q={
            name: member_values[..., index] for index, name in enumerate(members)
        },
        weights=weights,
    )


# keyword-only-exempt: library-callback=jax.vmap
def _evaluate_one_action[Output: PytreeValue](
    global_id: jax.Array,
    *,
    Q_and_F: Callable[..., Output],
    action_names: tuple[ActionName, ...],
    action_grids: tuple[jax.Array, ...],
    action_sizes: tuple[int, ...],
    fixed_kwargs: dict[ReferenceName, QAndFArg],
) -> Output:
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
    *, evaluate_block: Callable[..., _Block | _BoundedBlock]
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


def _empty_collective_reduction(
    *, evaluate_block: Callable[..., _Block | _BoundedBlock]
) -> CollectiveHardMaxAccumulator:
    """Create the empty household accumulator for the blocks `evaluate_block` makes."""
    block = _trace_block(evaluate_block=evaluate_block)
    member_values = block.shapes[1]
    return _typed_like(
        accumulator=COLLECTIVE_HARD_MAX_REDUCTION.initialize(
            member_template=jnp.zeros(
                member_values.shape[1:], dtype=member_values.dtype
            )
        ),
        arrays=block.read_arrays,
    )


def _empty_bound_status(
    *, evaluate_block: Callable[..., _BoundedBlock]
) -> tuple[BoundInputStatus, ...]:
    """Create all-false bound records for the blocks `evaluate_block` makes.

    All false is a reduction over no active lane and the identity of
    `merge_bound_input_status`, so a scan seeded with it evaluates every block
    in its body and stages none outside it.
    """
    # Only the records leave the abstract evaluation: an output container may
    # refuse abstract leaves.
    template = jax.eval_shape(
        lambda: evaluate_block(block_index=jnp.int32(0)).bound_status
    )
    return _typed_like(
        accumulator=jax.tree.map(
            lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype), template
        ),
        arrays=_trace_block(evaluate_block=evaluate_block).read_arrays,
    )


def _empty_query_status(
    *, evaluate_block: Callable[..., _BoundedBlock]
) -> tuple[QueryInputStatus, ...] | None:
    """Use the original paired-read record shapes as the empty OR identity."""
    template = jax.eval_shape(
        lambda: evaluate_block(block_index=jnp.int32(0)).query_status
    )
    return _typed_like(
        accumulator=jax.tree.map(
            lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype), template
        ),
        arrays=_trace_block(evaluate_block=evaluate_block).read_arrays,
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


def _scan_blocks(
    *,
    accumulator: CollectiveHardMaxAccumulator,
    evaluate_block: Callable[..., _Block],
    n_blocks: int,
) -> CollectiveHardMaxAccumulator:
    """Scan every block in order; the returned history is the ``None`` pytree."""
    (reduced, _), _history = jax.lax.scan(
        partial(_scan_one_block, evaluate_block=evaluate_block),
        (accumulator, jnp.asarray(0, dtype=jnp.int32)),
        xs=None,
        length=n_blocks,
    )
    return reduced


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


def _add_block(
    *, accumulator: CollectiveHardMaxAccumulator, block: _Block
) -> CollectiveHardMaxAccumulator:
    """Merge one evaluated block into the household's hard-max state."""
    objectives, member_values, feasible, global_ids = block
    return COLLECTIVE_HARD_MAX_REDUCTION.add(
        accumulator=accumulator,
        objectives=objectives,
        member_values=member_values,
        feasible=feasible,
        action_ids=global_ids,
    )


def _reduce_collective_whole_product(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    action_names: tuple[ActionName, ...],
    n_actions: int,
    members: tuple[str, ...],
    weights: Mapping[str, FloatND | float],
    kwargs: dict[ReferenceName, QAndFArg],
) -> CollectiveHardMaxResult:
    """Reduce a collective block covering the action product, mapping the named grids.

    As `_reduce_whole_product` does for a single: the C-order flattening of the
    mapped product is the canonical identity order, and the flat block is
    validated and reduced exactly as one streamed collective block is.
    """
    member_values, feasible = productmap(
        func=Q_and_F,
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
    )(**kwargs)
    member_values, feasible = (
        jnp.reshape(array, (n_actions, *array.shape[len(action_names) :]))
        for array in (jnp.asarray(member_values), jnp.asarray(feasible))
    )
    _validate_collective_block_Q_and_F(
        member_values=member_values, feasible=feasible, n_members=len(members)
    )
    return COLLECTIVE_HARD_MAX_REDUCTION.finalize(
        accumulator=_start_collective_reduction(
            block=(
                _household_objectives(
                    member_values=member_values, members=members, weights=weights
                ),
                member_values,
                feasible,
                jnp.arange(n_actions, dtype=jnp.int32),
            )
        )
    )


def _start_collective_reduction(*, block: _Block) -> CollectiveHardMaxAccumulator:
    """Seed a collective hard-max reduction from one evaluated block."""
    objectives, member_values, feasible, global_ids = block
    accumulator = COLLECTIVE_HARD_MAX_REDUCTION.initialize(
        member_template=jnp.zeros_like(member_values[0])
    )
    return COLLECTIVE_HARD_MAX_REDUCTION.add(
        accumulator=accumulator,
        objectives=objectives,
        member_values=member_values,
        feasible=feasible,
        action_ids=global_ids,
    )


def _reduce_collective_no_action(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    members: tuple[str, ...],
    weights: Mapping[str, FloatND | float],
    kwargs: dict[ReferenceName, QAndFArg],
) -> CollectiveHardMaxResult:
    """Treat a collective empty action product as one shared identity cell."""
    member_values, feasible = Q_and_F(**kwargs)
    member_values = jnp.asarray(member_values)
    feasible = jnp.asarray(feasible)
    _validate_collective_scalar_Q_and_F(
        member_values=member_values,
        feasible=feasible,
        n_members=len(members),
    )
    objective = _weighted_sum(
        member_Q={name: member_values[index] for index, name in enumerate(members)},
        weights=weights,
    )
    block = (
        objective[jnp.newaxis],
        member_values[jnp.newaxis, :],
        feasible[jnp.newaxis],
        jnp.array([0], dtype=jnp.int32),
    )
    accumulator = _start_collective_reduction(block=block)
    return COLLECTIVE_HARD_MAX_REDUCTION.finalize(accumulator=accumulator)


def _decode_action(
    *,
    global_id: jax.Array,
    action_names: tuple[ActionName, ...],
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


def _validate_collective_scalar_Q_and_F(
    *,
    member_values: jax.Array,
    feasible: jax.Array,
    n_members: int,
) -> None:
    """Validate one collective action cell."""
    if (
        member_values.ndim != 1
        or member_values.shape[-1] != n_members
        or feasible.ndim != 0
    ):
        raise ValueError(
            "Collective action streaming requires one trailing member "
            "axis and scalar feasibility at each action cell"
        )
    if feasible.dtype != jnp.bool_:
        raise TypeError("Q_and_F feasibility output must have boolean dtype")


def _validate_collective_block_Q_and_F(
    *,
    member_values: jax.Array,
    feasible: jax.Array,
    n_members: int,
) -> None:
    """Validate a vmapped collective block against the streaming contract."""
    if (
        member_values.ndim != _COLLECTIVE_BLOCK_NDIM
        or member_values.shape[-1] != n_members
        or feasible.ndim != 1
        or member_values.shape[0] != feasible.shape[0]
    ):
        raise ValueError(
            "Collective action streaming requires one trailing member "
            "axis and scalar feasibility at each action cell"
        )
    if feasible.dtype != jnp.bool_:
        raise TypeError("Q_and_F feasibility output must have boolean dtype")
