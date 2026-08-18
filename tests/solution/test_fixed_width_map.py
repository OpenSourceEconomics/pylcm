"""A fixed-width map publishes the same bits for every admitted block size.

`jax.lax.map(f, xs, batch_size=b)` reshapes `xs` by a static `b`, so each block
size compiles a differently vectorized body and the results can land on
representable neighbours. `map_fixed_width` takes the block size as a scalar
array instead: the static maximum and microtile width are the whole compilation
key, and the block size only changes the loop's stride and its commit mask.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.fixed_width_map import (
    admitted_block_size,
    map_fixed_width,
    validate_block_size,
)

pytestmark = pytest.mark.usefixtures("x64_enabled")

_MAX_BLOCK = 24
_MICROTILE = 4
_ADMITTED = (4, 8, 12, 16, 20, 24)


def _func(row: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """A per-row body whose reductions are long enough to be split differently."""
    scaled = row["level"] * row["weights"]
    weighted = scaled @ jnp.tanh(scaled)
    return jnp.stack([scaled.sum(), weighted, jnp.cumsum(scaled).sum()])


def _rows(*, n_rows: int) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(20260818)
    return {
        "level": jnp.asarray(rng.normal(size=n_rows)),
        "weights": jnp.asarray(rng.uniform(0.1, 2.0, size=(n_rows, 512))),
    }


def _mapped(*, rows, block_size: int) -> np.ndarray:
    out = jax.jit(
        lambda xs, size: map_fixed_width(
            func=_func,
            xs=xs,
            max_block_size=_MAX_BLOCK,
            microtile_width=_MICROTILE,
            block_size=size,
        )
    )(rows, jnp.int32(block_size))
    return np.asarray(jax.block_until_ready(out))


@pytest.mark.parametrize("n_rows", [7, 24, 31])
def test_every_admitted_block_size_publishes_identical_bits(n_rows: int) -> None:
    """All admitted block sizes agree bit for bit, including on a partial last block."""
    rows = _rows(n_rows=n_rows)
    baseline = _mapped(rows=rows, block_size=_ADMITTED[0])
    differing = sum(
        not np.array_equal(_mapped(rows=rows, block_size=size), baseline)
        for size in _ADMITTED[1:]
    )
    assert differing == 0


def test_result_covers_every_row() -> None:
    """The mapped output has one entry per input row, not per padded lane."""
    rows = _rows(n_rows=31)
    assert _mapped(rows=rows, block_size=8).shape == (31, 3)


@pytest.mark.parametrize("bad_size", [0, -4, 28, 6])
def test_inadmissible_block_sizes_are_rejected(bad_size: int) -> None:
    """Non-positive, oversized, and non-microtile-multiple block sizes raise."""
    with pytest.raises(ValueError, match="block_size must be"):
        validate_block_size(
            block_size=bad_size,
            max_block_size=_MAX_BLOCK,
            microtile_width=_MICROTILE,
        )


def test_static_block_size_map_is_partition_dependent() -> None:
    """Positive control: `lax.map`'s static `batch_size` does change published bits.

    Without this the zero disagreement count above would be consistent with a
    body too simple for XLA to reassociate at all, in which case the comparison
    would discriminate nothing.
    """
    rows = _rows(n_rows=31)

    def dense(*, batch_size: int) -> np.ndarray:
        out = jax.jit(
            lambda xs: jax.lax.map(_func, xs, batch_size=batch_size),
        )(rows)
        return np.asarray(jax.block_until_ready(out))

    baseline = dense(batch_size=_ADMITTED[0])
    differing = sum(
        not np.array_equal(dense(batch_size=size), baseline) for size in _ADMITTED[1:]
    )
    assert differing > 0


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(0, 24), (-1, 24), (1, 4), (4, 4), (5, 8), (23, 24), (24, 24), (99, 24)],
)
def test_requested_block_sizes_round_up_to_an_admitted_partition(
    requested: int, expected: int
) -> None:
    """A request is coarsened to the next microtile multiple, capped at the maximum."""
    assert (
        admitted_block_size(
            requested=requested,
            max_block_size=_MAX_BLOCK,
            microtile_width=_MICROTILE,
        )
        == expected
    )
