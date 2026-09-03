import inspect
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from _lcm.logsum import logsum_and_softmax
from _lcm.solution.logsumexp_action_reduction import (
    LOGSUMEXP_REDUCTION,
    BoundLogSumExpReduction,
    LogSumExpAccumulator,
    LogSumExpResult,
)


def test_logsumexp_binds_one_scale_for_the_complete_reduction() -> None:
    """A reduction session cannot mix scales between partial operations."""
    reduction = LOGSUMEXP_REDUCTION.bind(scale=jnp.asarray(0.5))

    assert "scale" not in inspect.signature(reduction.add).parameters
    assert "scale" not in inspect.signature(reduction.merge).parameters
    assert "scale" not in inspect.signature(reduction.finalize).parameters

    empty = reduction.initialize(value_template=jnp.zeros(()))
    left = reduction.add(accumulator=empty, values=jnp.asarray([1.0, 3.0]))
    right = reduction.add(accumulator=empty, values=jnp.asarray([2.0]))
    accumulator = reduction.merge(left=left, right=right)
    result = reduction.finalize(accumulator=accumulator)

    assert_allclose(result.smoothed_value, 3.0714657, rtol=1e-6)


def _float64_oracle(*, values: np.ndarray, scale: float) -> np.ndarray:
    """Stable scalar log-sum-exp evaluated independently in NumPy float64."""
    values64 = np.asarray(values, dtype=np.float64)
    scale64 = np.float64(scale)
    out = np.empty(values64.shape[:-1], dtype=np.float64)

    for state_index in np.ndindex(values64.shape[:-1]):
        row = values64[state_index]
        if np.isnan(row).any():
            out[state_index] = np.nan
            continue
        anchor = np.max(row)
        if np.isneginf(anchor):
            out[state_index] = -np.inf
            continue
        shifted_mass = np.sum(
            np.exp((row - anchor) / scale64),
            dtype=np.float64,
        )
        out[state_index] = anchor + scale64 * np.log(shifted_mass)

    return out


def _empty_accumulator(
    *, values: jax.Array, reduction: BoundLogSumExpReduction
) -> LogSumExpAccumulator:
    return reduction.initialize(
        value_template=jnp.zeros(values.shape[:-1], dtype=values.dtype)
    )


def _partial(
    *,
    values: jax.Array,
    reduction: BoundLogSumExpReduction,
    block: tuple[int, ...],
) -> LogSumExpAccumulator:
    indices = jnp.asarray(block, dtype=jnp.int32)
    return reduction.add(
        accumulator=_empty_accumulator(values=values, reduction=reduction),
        values=jnp.take(values, indices, axis=-1),
    )


def _reduce_blocks(
    *,
    values: jax.Array,
    scale: jax.Array,
    blocks: Iterable[tuple[int, ...]],
) -> LogSumExpResult:
    reduction = LOGSUMEXP_REDUCTION.bind(scale=scale)
    accumulator = _empty_accumulator(values=values, reduction=reduction)
    for block in blocks:
        accumulator = reduction.merge(
            left=accumulator,
            right=_partial(values=values, reduction=reduction, block=block),
        )
    return reduction.finalize(accumulator=accumulator)


def _assert_matches_float64_oracle(
    *, result: LogSumExpResult, expected: np.ndarray
) -> None:
    actual = np.asarray(result.smoothed_value)
    if actual.dtype == np.dtype(np.float32):
        tolerance = float(32 * np.finfo(np.float32).eps)
    else:
        tolerance = float(64 * np.finfo(np.float64).eps)
    assert_allclose(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )


def test_logsumexp_matches_float64_oracle_for_arbitrary_blocks_and_order():
    values = np.array(
        [
            [
                [4.0, 9.0, -np.inf, 2.0, 7.0, -3.0, 8.0],
                [8.0, -2.0, 6.0, 8.0, 1.0, -np.inf, 4.0],
            ],
            [
                [-3.0, 0.0, 5.0, 5.0, 4.0, 2.0, -np.inf],
                [1.0, 3.0, 2.0, 4.0, 0.0, -8.0, 3.5],
            ],
        ],
        dtype=np.float64,
    )
    scale = 0.37
    expected = _float64_oracle(values=values, scale=scale)

    for blocks in (
        ((5, 1), (6,), (0, 4, 2), (3,)),
        ((3,), (0, 4, 2), (6,), (5, 1)),
    ):
        result = _reduce_blocks(
            values=jnp.asarray(values),
            scale=jnp.asarray(scale),
            blocks=blocks,
        )
        _assert_matches_float64_oracle(result=result, expected=expected)


def test_logsumexp_all_infeasible_is_negative_infinity_without_nan():
    result = _reduce_blocks(
        values=jnp.full((3, 5), -jnp.inf),
        scale=jnp.asarray(0.2),
        blocks=((1, 3), (4,), (0, 2)),
    )

    assert_array_equal(jnp.isneginf(result.smoothed_value), jnp.ones(3, dtype=bool))
    assert_array_equal(jnp.isnan(result.smoothed_value), jnp.zeros(3, dtype=bool))


def test_logsumexp_stays_finite_for_extreme_gaps_and_tiny_positive_scale():
    values = np.array(
        [
            [50.0, 49.999, -1e30, -np.inf],
            [-50.0, -50.001, -1e30, -np.inf],
        ],
        dtype=np.float64,
    )
    scale = 1e-30
    expected = _float64_oracle(values=values, scale=scale)
    result = _reduce_blocks(
        values=jnp.asarray(values),
        scale=jnp.asarray(scale),
        blocks=((1, 3), (0,), (2,)),
    )

    assert_array_equal(jnp.isfinite(result.smoothed_value), jnp.array([True, True]))
    _assert_matches_float64_oracle(result=result, expected=expected)


def test_logsumexp_propagates_nan_independently_of_block_order():
    values = np.array(
        [[3.0, np.nan, -np.inf, 7.0], [1.0, 2.0, 4.0, -np.inf]],
        dtype=np.float64,
    )
    scale = 0.5
    expected = _float64_oracle(values=values, scale=scale)

    for blocks in (((0, 3), (1,), (2,)), ((2,), (1,), (0, 3))):
        result = _reduce_blocks(
            values=jnp.asarray(values),
            scale=jnp.asarray(scale),
            blocks=blocks,
        )
        assert_array_equal(jnp.isnan(result.smoothed_value), jnp.array([True, False]))
        _assert_matches_float64_oracle(result=result, expected=expected)


def test_positive_infinity_matches_dense_logsum_value_semantics() -> None:
    """A positive-infinite branch makes both dense and blockwise values NaN."""
    values = jnp.asarray([[3.0, jnp.inf, -jnp.inf], [-jnp.inf, 4.0, jnp.inf]])
    scale = jnp.asarray(0.5)
    dense, _ = logsum_and_softmax(values=values, scale=scale, axes=(-1,))
    blockwise = _reduce_blocks(
        values=values,
        scale=scale,
        blocks=((2,), (0, 1)),
    )

    assert_array_equal(jnp.isnan(dense), jnp.ones(2, dtype=bool))
    assert_allclose(blockwise.smoothed_value, dense, equal_nan=True)


def test_logsumexp_supports_jit_and_vmap():
    # keyword-only-exempt: library-callback=jax.vmap
    @jax.jit
    def reduce_one(values: jax.Array, scale: jax.Array) -> LogSumExpResult:
        reduction = LOGSUMEXP_REDUCTION.bind(scale=scale)
        accumulator = reduction.initialize(
            value_template=jnp.zeros((), dtype=values.dtype)
        )
        accumulator = reduction.add(
            accumulator=accumulator,
            values=values,
        )
        return reduction.finalize(
            accumulator=accumulator,
        )

    scale = jnp.asarray(0.4)
    values = jnp.array([[1.0, 5.0, -jnp.inf], [7.0, 9.0, 8.0]])
    result = jax.vmap(reduce_one, in_axes=(0, None))(values, scale)
    expected = _float64_oracle(values=np.asarray(values), scale=float(scale))

    _assert_matches_float64_oracle(result=result, expected=expected)
