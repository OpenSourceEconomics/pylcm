"""Fixed-graph, GPU-parallel fused continuation/envelope reference.

The scheduler has two deliberately separate notions of width:

`pair_ride_width x pair_branch_width`
    The static microtile evaluated by one batch-native economic callback.  This
    is the GPU-parallel width `W`.

`max_ride_block_size x max_branch_block_size`
    The largest storage rectangle admitted by one compiled executable.
    Runtime ride/branch partition sizes may be smaller, but they are masks and
    loop increments inside *the same executable*; they never change callback
    shapes, vector lanes, reduction shapes, or the compilation key.

That distinction is load-bearing.  Compiling one executable per outer
partition lets XLA reassociate floating expressions differently even when a
fixed microtile is used.  Separate compilations of the same fixed
microtile have been observed to disagree in the last few float64 digits.
Here every admitted partition is a runtime value in one fixed graph.
Combined with globally aligned microtiles, a global
`(ride, branch)` pair always has the same vector lane and every branch
comparison has the same static reduction shape.

The reference fuses continuation production with immediate envelope
consumption and cannot receive a full `R x B x I x S` continuation stack.
Production integration must supply genuinely batch-native callbacks.  A
scalar primitive whose batching rule is sequential does not satisfy the
parallelism contract even if wrapped in `vmap`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

type Array = jax.Array
type PyTree = Any


class ContinuationRow(NamedTuple):
    """Continuation value and marginal with shape `(W, I, S)`."""

    value: Array
    marginal: Array


class BranchCandidateRow(NamedTuple):
    """Candidate channels with shape `(W, Q)`."""

    value: Array
    policy: Array
    marginal: Array
    segment_owner: Array
    readable: Array


class FusedEnvelopeResult(NamedTuple):
    """Coupled final row after the global branch reduction."""

    value: Array
    policy: Array
    marginal: Array
    discrete_owner: Array
    segment_owner: Array
    readable: Array


class _CandidateBitsRow(NamedTuple):
    value_bits: Array
    policy_bits: Array
    marginal_bits: Array
    segment_owner: Array
    readable: Array


class _FusedBitsResult(NamedTuple):
    value_bits: Array
    policy_bits: Array
    marginal_bits: Array
    discrete_owner: Array
    segment_owner: Array
    readable: Array


@dataclass(frozen=True, slots=True)
class FusedStreamingConfig:
    """Static compilation envelope and invariant microtile dimensions.

    `max_*` and `pair_*` are static.  Runtime partitions are admitted when
    they are positive, no larger than their corresponding maximum, and exact
    multiples of the corresponding microtile dimension.
    """

    max_ride_block_size: int
    max_branch_block_size: int
    pair_ride_width: int
    pair_branch_width: int

    def __post_init__(self) -> None:
        values = (
            self.max_ride_block_size,
            self.max_branch_block_size,
            self.pair_ride_width,
            self.pair_branch_width,
        )
        if min(values) <= 0:
            raise ValueError("all maximum block and microtile widths must be positive")
        if self.max_ride_block_size % self.pair_ride_width:
            raise ValueError(
                "max_ride_block_size must be a multiple of pair_ride_width"
            )
        if self.max_branch_block_size % self.pair_branch_width:
            raise ValueError(
                "max_branch_block_size must be a multiple of pair_branch_width"
            )

    @property
    def pair_vector_width(self) -> int:
        return self.pair_ride_width * self.pair_branch_width

    def admits(self, *, ride_partition_size: int, branch_partition_size: int) -> bool:
        """Return whether host-side partition values satisfy the contract."""

        return (
            0 < ride_partition_size <= self.max_ride_block_size
            and 0 < branch_partition_size <= self.max_branch_block_size
            and ride_partition_size % self.pair_ride_width == 0
            and branch_partition_size % self.pair_branch_width == 0
        )

    def validate_partition(
        self, *, ride_partition_size: int, branch_partition_size: int
    ) -> None:
        if not self.admits(
            ride_partition_size=ride_partition_size,
            branch_partition_size=branch_partition_size,
        ):
            raise ValueError(
                "runtime partitions must be positive microtile multiples no larger "
                "than the static maxima"
            )


type ContinuationBatch = Callable[[PyTree, PyTree, PyTree], ContinuationRow]
type EnvelopeBatch = Callable[
    [PyTree, PyTree, Array, ContinuationRow, PyTree], BranchCandidateRow
]


def _leading_size(tree: PyTree, *, name: str) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError(f"{name} must contain at least one array leaf")
    size = int(leaves[0].shape[0])
    if size <= 0:
        raise ValueError(f"{name} must have a non-empty leading axis")
    for leaf in leaves:
        if leaf.ndim == 0 or int(leaf.shape[0]) != size:
            raise ValueError(f"all {name} leaves must share one non-empty leading axis")
    return size


def _pad_leading(tree: PyTree, *, pad: int) -> PyTree:
    """Append repeated, inactive rows so every dynamic gather is in bounds."""

    return jax.tree_util.tree_map(
        lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], repeats=pad, axis=0)], axis=0),
        tree,
    )


def _take_block(tree: PyTree, indices: Array) -> PyTree:
    return jax.tree_util.tree_map(lambda x: x[indices], tree)


def _bit_dtype(dtype: jnp.dtype) -> jnp.dtype:
    return jnp.uint32 if jnp.dtype(dtype).itemsize == 4 else jnp.uint64


def _to_bits(array: Array) -> Array:
    return jax.lax.bitcast_convert_type(array, _bit_dtype(array.dtype))


def _from_bits(array: Array, dtype: jnp.dtype) -> Array:
    return jax.lax.bitcast_convert_type(array, dtype)


def _empty_bits_result(
    *, n_rides: int, n_queries: int, dtype: jnp.dtype
) -> _FusedBitsResult:
    shape = (n_rides, n_queries)
    return _FusedBitsResult(
        value_bits=_to_bits(jnp.full(shape, -jnp.inf, dtype=dtype)),
        policy_bits=_to_bits(jnp.full(shape, jnp.nan, dtype=dtype)),
        marginal_bits=_to_bits(jnp.full(shape, jnp.nan, dtype=dtype)),
        discrete_owner=jnp.full(shape, -1, dtype=jnp.int32),
        segment_owner=jnp.full(shape, -1, dtype=jnp.int32),
        readable=jnp.zeros(shape, dtype=bool),
    )


def _take_branch_winner(array: Array, winner: Array) -> Array:
    return jnp.take_along_axis(array, winner[:, None, :], axis=1)[:, 0, :]


def _validate_batch_shapes(
    continuation: ContinuationRow,
    candidates: BranchCandidateRow,
    *,
    width: int,
    n_queries: int,
) -> None:
    if continuation.value.ndim != 3 or continuation.marginal.ndim != 3:
        raise ValueError("continuation batch arrays must have shape (W, I, S)")
    if continuation.value.shape != continuation.marginal.shape:
        raise ValueError("continuation value and marginal shapes must match")
    if int(continuation.value.shape[0]) != width:
        raise ValueError("continuation callback changed the invariant vector width")
    for name, array in zip(BranchCandidateRow._fields, candidates, strict=True):
        if array.ndim != 2 or array.shape != (width, n_queries):
            raise ValueError(
                f"candidate channel {name!r} must have shape ({width}, {n_queries})"
            )


def _evaluate_max_rectangle(
    *,
    ride_block: PyTree,
    branch_block: PyTree,
    query_grid: Array,
    params: PyTree,
    continuation_batch: ContinuationBatch,
    envelope_batch: EnvelopeBatch,
    config: FusedStreamingConfig,
) -> _CandidateBitsRow:
    """Evaluate one static maximum rectangle in globally aligned microtiles."""

    m_r = config.max_ride_block_size
    m_b = config.max_branch_block_size
    t_r = config.pair_ride_width
    t_b = config.pair_branch_width
    width = config.pair_vector_width
    n_tile_r = m_r // t_r
    n_tile_b = m_b // t_b
    n_tiles = n_tile_r * n_tile_b
    ride_lane = jnp.repeat(jnp.arange(t_r, dtype=jnp.int32), t_b)
    branch_lane = jnp.tile(jnp.arange(t_b, dtype=jnp.int32), t_r)

    def evaluate_tile(tile_number: Array) -> _CandidateBitsRow:
        tile_r = tile_number // n_tile_b
        tile_b = tile_number % n_tile_b
        ride_pos = tile_r * t_r + ride_lane
        branch_pos = tile_b * t_b + branch_lane
        ride_items = _take_block(ride_block, ride_pos)
        branch_items = _take_block(branch_block, branch_pos)
        continuation = continuation_batch(ride_items, branch_items, params)
        candidates = envelope_batch(
            ride_items, branch_items, query_grid, continuation, params
        )
        _validate_batch_shapes(
            continuation,
            candidates,
            width=width,
            n_queries=int(query_grid.shape[0]),
        )
        # The fixed tile boundary freezes every published floating payload before
        # the outer owner reduction.  Production may implement this boundary as
        # one batch-native custom call or another opaque fixed-shape kernel.
        return _CandidateBitsRow(
            value_bits=_to_bits(candidates.value),
            policy_bits=_to_bits(candidates.policy),
            marginal_bits=_to_bits(candidates.marginal),
            segment_owner=candidates.segment_owner.astype(jnp.int32),
            readable=jnp.asarray(candidates.readable, dtype=bool),
        )

    tiled = jax.lax.map(evaluate_tile, jnp.arange(n_tiles, dtype=jnp.int32))

    def tile_to_rectangle(array: Array) -> Array:
        trailing = array.shape[2:]
        shaped = array.reshape(n_tile_r, n_tile_b, t_r, t_b, *trailing)
        axes = (0, 2, 1, 3, *range(4, shaped.ndim))
        return jnp.transpose(shaped, axes).reshape(m_r, m_b, *trailing)

    return _CandidateBitsRow(*map(tile_to_rectangle, tiled))


def build_fused_continuation_envelope(
    *,
    continuation_batch: ContinuationBatch,
    envelope_batch: EnvelopeBatch,
    config: FusedStreamingConfig,
) -> Callable[[PyTree, PyTree, Array, PyTree, Array, Array], FusedEnvelopeResult]:
    """Build one fixed executable with runtime outer partition values.

    `ride_partition_size` and `branch_partition_size` must be scalar integer
    arrays satisfying `FusedStreamingConfig.validate_partition`.  They are
    intentionally runtime values.  The caller validates them before entering
    JIT, then may reuse the same compiled executable for every admitted pair.

    Exact tie semantics are global and partition-independent: within the static
    maximum branch rectangle `argmax` selects the earliest active branch; the
    dynamic branch loop visits rectangles in increasing global order and only a
    strict improvement replaces the carry.
    """

    m_r = config.max_ride_block_size
    m_b = config.max_branch_block_size
    t_r = config.pair_ride_width
    t_b = config.pair_branch_width

    def core(
        ride_payload: PyTree,
        branch_payload: PyTree,
        query_grid: Array,
        params: PyTree,
        ride_partition_size: Array,
        branch_partition_size: Array,
    ) -> FusedEnvelopeResult:
        n_rides = _leading_size(ride_payload, name="ride_payload")
        n_branches = _leading_size(branch_payload, name="branch_payload")
        if query_grid.ndim != 1 or int(query_grid.shape[0]) <= 0:
            raise ValueError("query_grid must be a non-empty one-dimensional array")
        n_queries = int(query_grid.shape[0])
        dtype = jnp.result_type(query_grid)
        ride_partition_size = jnp.asarray(ride_partition_size, dtype=jnp.int32)
        branch_partition_size = jnp.asarray(branch_partition_size, dtype=jnp.int32)

        # The host-side contract checks these.  Clamping here prevents an invalid
        # zero from creating a nonterminating while loop if a private caller
        # bypasses validation; it does not admit such a configuration.
        ride_step = jnp.maximum(ride_partition_size, jnp.int32(t_r))
        branch_step = jnp.maximum(branch_partition_size, jnp.int32(t_b))

        padded_ride = _pad_leading(ride_payload, pad=m_r)
        padded_branch = _pad_leading(branch_payload, pad=m_b)
        ride_offsets = jnp.arange(m_r, dtype=jnp.int32)
        branch_offsets = jnp.arange(m_b, dtype=jnp.int32)
        out = _empty_bits_result(
            n_rides=n_rides + m_r, n_queries=n_queries, dtype=dtype
        )

        def ride_cond(state: tuple[Array, _FusedBitsResult]) -> Array:
            ride_start, _ = state
            return ride_start < n_rides

        def ride_body(
            state: tuple[Array, _FusedBitsResult],
        ) -> tuple[Array, _FusedBitsResult]:
            ride_start, output = state
            ride_indices = ride_start + ride_offsets
            ride_active = (ride_offsets < ride_partition_size) & (
                ride_indices < n_rides
            )
            ride_block = _take_block(padded_ride, ride_indices)
            best0 = _empty_bits_result(n_rides=m_r, n_queries=n_queries, dtype=dtype)

            def branch_cond(state_b: tuple[Array, _FusedBitsResult]) -> Array:
                branch_start, _ = state_b
                return branch_start < n_branches

            def branch_body(
                state_b: tuple[Array, _FusedBitsResult],
            ) -> tuple[Array, _FusedBitsResult]:
                branch_start, best = state_b
                branch_indices = branch_start + branch_offsets
                branch_active = (branch_offsets < branch_partition_size) & (
                    branch_indices < n_branches
                )
                branch_block = _take_block(padded_branch, branch_indices)
                frozen = _evaluate_max_rectangle(
                    ride_block=ride_block,
                    branch_block=branch_block,
                    query_grid=query_grid,
                    params=params,
                    continuation_batch=continuation_batch,
                    envelope_batch=envelope_batch,
                    config=config,
                )
                candidate_value = _from_bits(frozen.value_bits, dtype)
                candidate_policy = _from_bits(frozen.policy_bits, dtype)
                candidate_marginal = _from_bits(frozen.marginal_bits, dtype)
                valid = (
                    ride_active[:, None, None]
                    & branch_active[None, :, None]
                    & frozen.readable
                    & jnp.isfinite(candidate_value)
                    & jnp.isfinite(candidate_policy)
                    & jnp.isfinite(candidate_marginal)
                )
                masked = jnp.where(valid, candidate_value, -jnp.inf)
                within = jnp.argmax(masked, axis=1).astype(jnp.int32)
                readable = jnp.any(valid, axis=1)
                block = _FusedBitsResult(
                    value_bits=_take_branch_winner(frozen.value_bits, within),
                    policy_bits=_take_branch_winner(frozen.policy_bits, within),
                    marginal_bits=_take_branch_winner(frozen.marginal_bits, within),
                    discrete_owner=_take_branch_winner(
                        jnp.broadcast_to(
                            branch_indices[None, :, None], masked.shape
                        ).astype(jnp.int32),
                        within,
                    ),
                    segment_owner=_take_branch_winner(frozen.segment_owner, within),
                    readable=readable,
                )
                block_value = _from_bits(block.value_bits, dtype)
                best_value = _from_bits(best.value_bits, dtype)
                improve = block.readable & (
                    (~best.readable) | (block_value > best_value)
                )
                best = _FusedBitsResult(
                    value_bits=jnp.where(improve, block.value_bits, best.value_bits),
                    policy_bits=jnp.where(improve, block.policy_bits, best.policy_bits),
                    marginal_bits=jnp.where(
                        improve, block.marginal_bits, best.marginal_bits
                    ),
                    discrete_owner=jnp.where(
                        improve, block.discrete_owner, best.discrete_owner
                    ),
                    segment_owner=jnp.where(
                        improve, block.segment_owner, best.segment_owner
                    ),
                    readable=best.readable | block.readable,
                )
                return branch_start + branch_step, best

            _, best = jax.lax.while_loop(
                branch_cond, branch_body, (jnp.int32(0), best0)
            )

            # Unique padded indices permit a fixed-shape scatter.  Inactive rows
            # leave the existing output unchanged.
            old = _FusedBitsResult(*(_take_block(output, ride_indices)))
            active = ride_active[:, None]
            selected = _FusedBitsResult(
                value_bits=jnp.where(active, best.value_bits, old.value_bits),
                policy_bits=jnp.where(active, best.policy_bits, old.policy_bits),
                marginal_bits=jnp.where(active, best.marginal_bits, old.marginal_bits),
                discrete_owner=jnp.where(
                    active, best.discrete_owner, old.discrete_owner
                ),
                segment_owner=jnp.where(active, best.segment_owner, old.segment_owner),
                readable=jnp.where(active, best.readable, old.readable),
            )
            output = _FusedBitsResult(
                *(
                    x.at[ride_indices].set(y)
                    for x, y in zip(output, selected, strict=True)
                )
            )
            return ride_start + ride_step, output

        _, output = jax.lax.while_loop(ride_cond, ride_body, (jnp.int32(0), out))
        return FusedEnvelopeResult(
            value=_from_bits(output.value_bits[:n_rides], dtype),
            policy=_from_bits(output.policy_bits[:n_rides], dtype),
            marginal=_from_bits(output.marginal_bits[:n_rides], dtype),
            discrete_owner=output.discrete_owner[:n_rides],
            segment_owner=output.segment_owner[:n_rides],
            readable=output.readable[:n_rides],
        )

    return core


@dataclass(frozen=True, slots=True)
class WorkspaceEstimate:
    continuation_tile_bytes: int
    candidate_rectangle_bytes: int
    branch_carry_bytes: int
    final_output_bytes: int

    @property
    def peak_explicit_block_bytes(self) -> int:
        return (
            self.continuation_tile_bytes
            + self.candidate_rectangle_bytes
            + self.branch_carry_bytes
        )


@dataclass(frozen=True, slots=True)
class LoweringSchedule:
    pair_vector_width: int
    tiles_per_max_rectangle: int
    traced_pair_callback_bodies: int
    static_max_ride_block: int
    static_max_branch_block: int
    partition_values_are_runtime: bool


def estimate_workspace_bytes(
    *,
    n_rides: int,
    n_queries: int,
    n_intervals: int,
    n_savings: int,
    config: FusedStreamingConfig,
    float_bytes: int = 8,
    int_bytes: int = 4,
    bool_bytes: int = 1,
) -> WorkspaceEstimate:
    """Explicit scheduler arrays; excludes callback/compiler-private scratch."""

    if min(n_rides, n_queries, n_intervals, n_savings) <= 0:
        raise ValueError("all logical dimensions must be positive")
    continuation = 2 * float_bytes * config.pair_vector_width * n_intervals * n_savings
    candidate = (
        (3 * float_bytes + int_bytes + bool_bytes)
        * config.max_ride_block_size
        * config.max_branch_block_size
        * n_queries
    )
    carry = (
        (3 * float_bytes + 2 * int_bytes + bool_bytes)
        * config.max_ride_block_size
        * n_queries
    )
    final = (3 * float_bytes + 2 * int_bytes + bool_bytes) * n_rides * n_queries
    return WorkspaceEstimate(continuation, candidate, carry, final)


def estimate_lowering_schedule(*, config: FusedStreamingConfig) -> LoweringSchedule:
    return LoweringSchedule(
        pair_vector_width=config.pair_vector_width,
        tiles_per_max_rectangle=(config.max_ride_block_size // config.pair_ride_width)
        * (config.max_branch_block_size // config.pair_branch_width),
        traced_pair_callback_bodies=1,
        static_max_ride_block=config.max_ride_block_size,
        static_max_branch_block=config.max_branch_block_size,
        partition_values_are_runtime=True,
    )
