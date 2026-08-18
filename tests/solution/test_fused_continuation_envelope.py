import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.solution._fused_continuation_envelope_oracle import (
    BranchCandidateRow,
    ContinuationRow,
    FusedStreamingConfig,
    build_fused_continuation_envelope,
    estimate_lowering_schedule,
    estimate_workspace_bytes,
)

pytestmark = pytest.mark.usefixtures("x64_enabled")


def callbacks(
    *, width: int, n_intervals: int = 3, n_savings: int = 5, dtype=jnp.float64
):
    i = jnp.arange(n_intervals, dtype=dtype)[None, :, None]
    s = jnp.arange(n_savings, dtype=dtype)[None, None, :]

    def continuation_batch(ride, branch, params):
        assert ride["location"].shape == (width,)
        assert branch["code"].shape == (width,)
        center = (
            0.3 * ride["location"][:, None, None] + 0.2 * branch["code"][:, None, None]
        )
        value = (
            ride["level"][:, None, None]
            + params["bonus"]
            + 0.37 * branch["code"][:, None, None]
            + 0.11 * i
            - 0.07 * (s - center) ** 2
        )
        marginal = (
            1.0
            + 0.03 * ride["location"][:, None, None]
            + 0.02 * branch["code"][:, None, None]
            + 0.01 * i
            + 0.005 * s
        )
        return ContinuationRow(value=value, marginal=marginal)

    def envelope_batch(ride, branch, query, continuation, params):
        assert ride["location"].shape == (width,)
        del params
        resources = (
            0.4 * ride["location"][:, None, None]
            + 0.15 * branch["code"][:, None, None]
            + 0.8 * i
            + 0.25 * s
        )
        score = (
            continuation.value[..., None]
            - 0.19 * (query[None, None, None, :] - resources[..., None]) ** 2
        )
        flat = score.reshape(width, n_intervals * n_savings, query.shape[0])
        winner = jnp.argmax(flat, axis=1)
        value = jnp.take_along_axis(flat, winner[:, None, :], axis=1)[:, 0, :]
        marginal_grid = jnp.broadcast_to(
            continuation.marginal[..., None], score.shape
        ).reshape(width, n_intervals * n_savings, query.shape[0])
        marginal = jnp.take_along_axis(marginal_grid, winner[:, None, :], axis=1)[
            :, 0, :
        ]
        policy = (
            0.4
            + 0.03 * branch["code"][:, None]
            + 0.01 * winner.astype(query.dtype)
            + 0.05 * query[None, :]
        )
        readable = (
            (query[None, :] >= -0.5)
            & (query[None, :] <= 4.5)
            & (branch["code"][:, None].astype(jnp.int32) != 6)
        )
        segment = branch["code"][:, None].astype(jnp.int32) * 1000 + winner.astype(
            jnp.int32
        )
        return BranchCandidateRow(value, policy, marginal, segment, readable)

    return continuation_batch, envelope_batch


def payloads(*, n_rides: int = 17, n_branches: int = 13, dtype=jnp.float64):
    ride = {
        "level": jnp.linspace(-0.4, 0.8, n_rides, dtype=dtype),
        "location": jnp.arange(n_rides, dtype=dtype) / dtype(3.0),
    }
    branch = {"code": jnp.arange(n_branches, dtype=dtype)}
    query = jnp.linspace(-0.25, 4.25, 11, dtype=dtype)
    params = {"bonus": dtype(0.125)}
    return ride, branch, query, params


def assert_result_equal(left, right):
    for a, b in zip(left, right, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


TILE = (4, 4)
CONFIG = FusedStreamingConfig(20, 20, *TILE)
PARTITIONS = [(4, 4), (8, 4), (4, 8), (8, 12), (16, 4), (20, 16), (20, 20)]


def build_test_core(*, dtype=jnp.float64):
    continuation_batch, envelope_batch = callbacks(
        width=CONFIG.pair_vector_width, dtype=dtype
    )
    return jax.jit(
        build_fused_continuation_envelope(
            continuation_batch=continuation_batch,
            envelope_batch=envelope_batch,
            config=CONFIG,
        )
    )


def call(core, args, partition):
    return core(
        *args,
        jnp.asarray(partition[0], dtype=jnp.int32),
        jnp.asarray(partition[1], dtype=jnp.int32),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_published_rows_are_bit_identical_across_runtime_partitions(dtype):
    core = build_test_core(dtype=dtype)
    args = payloads(dtype=dtype)
    baseline = call(core, args, PARTITIONS[0])
    for partition in PARTITIONS[1:]:
        assert_result_equal(call(core, args, partition), baseline)


@pytest.mark.parametrize("shape", [(11, 10), (17, 13), (31, 20)])
def test_edge_padding_and_remainders_are_partition_invariant(shape):
    core = build_test_core()
    args = payloads(n_rides=shape[0], n_branches=shape[1])
    baseline = call(core, args, (4, 4))
    for partition in PARTITIONS[1:]:
        assert_result_equal(call(core, args, partition), baseline)


def test_global_pair_lane_and_compilation_key_are_fixed():
    ride_index, branch_index = 5, 7
    expected_lane = (ride_index % TILE[0]) * TILE[1] + branch_index % TILE[1]
    assert expected_lane == 7

    core = build_test_core()
    args = payloads()
    # Scalars are dynamic operands, not Python-static compilation arguments.
    low_a = core.lower(*args, jnp.int32(4), jnp.int32(4)).as_text()
    low_b = core.lower(*args, jnp.int32(20), jnp.int32(20)).as_text()
    assert low_a == low_b


def test_exact_ties_keep_earliest_global_branch_for_every_partition():
    width = CONFIG.pair_vector_width

    def continuation_batch(ride, branch, params):
        del branch, params
        w = ride["x"].shape[0]
        return ContinuationRow(
            jnp.ones((w, 1, 1), dtype=jnp.float64),
            jnp.full((w, 1, 1), 2.0, dtype=jnp.float64),
        )

    def envelope_batch(ride, branch, query, continuation, params):
        del ride, continuation, params
        code = branch["code"].astype(jnp.int32)
        w = code.shape[0]
        assert w == width
        value = jnp.ones((w, query.size), dtype=query.dtype)
        return BranchCandidateRow(
            value,
            jnp.broadcast_to(10.0 + code[:, None], value.shape),
            jnp.broadcast_to(20.0 + code[:, None], value.shape),
            jnp.broadcast_to(100 + code[:, None], value.shape),
            jnp.ones(value.shape, dtype=bool),
        )

    core = jax.jit(
        build_fused_continuation_envelope(
            continuation_batch=continuation_batch,
            envelope_batch=envelope_batch,
            config=CONFIG,
        )
    )
    args = (
        {"x": jnp.arange(11, dtype=jnp.float64)},
        {"code": jnp.arange(13, dtype=jnp.int32)},
        jnp.array([0.25, 0.75], dtype=jnp.float64),
        {},
    )
    for partition in PARTITIONS:
        result = call(core, args, partition)
        np.testing.assert_array_equal(np.asarray(result.discrete_owner), 0)
        np.testing.assert_array_equal(np.asarray(result.policy), 10.0)
        np.testing.assert_array_equal(np.asarray(result.marginal), 20.0)
        np.testing.assert_array_equal(np.asarray(result.segment_owner), 100)


def test_invalid_static_and_runtime_partitions_fail_before_tracing():
    with pytest.raises(ValueError, match="multiple"):
        FusedStreamingConfig(18, 20, 4, 4)
    with pytest.raises(ValueError, match="multiple"):
        FusedStreamingConfig(20, 18, 4, 4)
    with pytest.raises(ValueError, match="positive"):
        FusedStreamingConfig(20, 20, 0, 4)
    with pytest.raises(ValueError, match="runtime partitions"):
        CONFIG.validate_partition(ride_partition_size=6, branch_partition_size=4)
    with pytest.raises(ValueError, match="runtime partitions"):
        CONFIG.validate_partition(ride_partition_size=4, branch_partition_size=24)


def test_workspace_has_no_full_stack_and_is_independent_of_runtime_partition():
    small = estimate_workspace_bytes(
        n_rides=1_000,
        n_queries=24,
        n_intervals=11,
        n_savings=200,
        config=CONFIG,
    )
    production = estimate_workspace_bytes(
        n_rides=102_600,
        n_queries=24,
        n_intervals=11,
        n_savings=200,
        config=CONFIG,
    )
    assert small.peak_explicit_block_bytes == production.peak_explicit_block_bytes
    assert production.continuation_tile_bytes == 2 * 8 * 16 * 11 * 200
    full_two_stack = 2 * 8 * 102_600 * 20 * 11 * 200
    assert production.peak_explicit_block_bytes < full_two_stack / 100
    assert production.final_output_bytes == (3 * 8 + 2 * 4 + 1) * 102_600 * 24


def test_lowering_schedule_has_one_fixed_pair_body_and_runtime_partitions():
    schedule = estimate_lowering_schedule(config=CONFIG)
    assert schedule.traced_pair_callback_bodies == 1
    assert schedule.pair_vector_width == 16
    assert schedule.tiles_per_max_rectangle == 25
    assert schedule.static_max_ride_block == 20
    assert schedule.static_max_branch_block == 20
    assert schedule.partition_values_are_runtime


def test_a100_candidate_width_128_is_partition_invariant():
    config = FusedStreamingConfig(64, 20, 32, 4)
    continuation_batch, envelope_batch = callbacks(
        width=config.pair_vector_width, n_intervals=11, n_savings=200
    )
    core = jax.jit(
        build_fused_continuation_envelope(
            continuation_batch=continuation_batch,
            envelope_batch=envelope_batch,
            config=config,
        )
    )
    ride, branch, _, params = payloads(n_rides=65, n_branches=20)
    query = jnp.linspace(-0.25, 4.25, 24, dtype=jnp.float64)
    args = (ride, branch, query, params)
    baseline = call(core, args, (32, 4))
    for partition in ((32, 8), (32, 20), (64, 4), (64, 20)):
        assert_result_equal(call(core, args, partition), baseline)


def test_lowering_body_count_does_not_scale_with_logical_rides_or_branches():
    continuation_batch, envelope_batch = callbacks(width=CONFIG.pair_vector_width)
    raw = build_fused_continuation_envelope(
        continuation_batch=continuation_batch,
        envelope_batch=envelope_batch,
        config=CONFIG,
    )
    small = payloads(n_rides=8, n_branches=8)
    large = payloads(n_rides=257, n_branches=20)
    small_text = str(jax.make_jaxpr(raw)(*small, jnp.int32(4), jnp.int32(4)))
    large_text = str(jax.make_jaxpr(raw)(*large, jnp.int32(20), jnp.int32(20)))
    assert small_text.count("while[") == large_text.count("while[") == 2
    assert small_text.count("scan[") == large_text.count("scan[") == 1
    # Only array-shape literals and final slice bounds may grow; the callback
    # body is not cloned with R, B, or the runtime partitions.
    assert abs(len(large_text) - len(small_text)) < 1_000
