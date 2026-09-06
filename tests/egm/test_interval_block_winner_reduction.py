"""Standalone JAX reduction agrees with the exact stable-index oracle."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _interval_block_winner_oracle import Candidate, Ordering, one_shot_winner


def _dtype_or_skip(name: str) -> jnp.dtype:
    """Resolve a float name against the session's x64 policy."""
    if name == "float64" and not jax.config.jax_enable_x64:
        pytest.skip("float64 arrays need the session's x64 precision policy")
    return jnp.dtype(name)


def _argmax(*, fields: tuple[jax.Array, ...], stable: jax.Array) -> jax.Array:
    tied = jnp.ones_like(fields[0], dtype=bool)
    for field in fields:
        best = jnp.max(jnp.where(tied, field, -jnp.inf), axis=-1, keepdims=True)
        tied = tied & (field == best)
    earliest = jnp.min(
        jnp.where(tied, stable, jnp.iinfo(jnp.int32).max),
        axis=-1,
        keepdims=True,
    )
    tied = tied & (stable == earliest)
    return jnp.argmax(tied, axis=-1).astype(jnp.int32)


def _one_shot(
    *, fields: tuple[jax.Array, ...], stable: jax.Array, live: jax.Array
) -> jax.Array:
    offered = tuple(jnp.where(live, field, -jnp.inf) for field in fields)
    offered_stable = jnp.where(live, stable, jnp.iinfo(jnp.int32).max)
    position = _argmax(fields=offered, stable=offered_stable)
    return offered_stable[position]


def _blocked(
    *,
    fields: tuple[jax.Array, ...],
    stable: jax.Array,
    live: jax.Array,
    block_size: int,
) -> jax.Array:
    """Fold one winner per block, carrying its global identity through scan."""
    n = stable.shape[-1]
    n_blocks = -(-n // block_size)
    n_pad = n_blocks * block_size - n

    def pad_float(array: jax.Array) -> jax.Array:
        return jnp.pad(array, (0, n_pad), constant_values=-jnp.inf).reshape(
            n_blocks, block_size
        )

    field_blocks = tuple(pad_float(field) for field in fields)
    stable_blocks = jnp.pad(
        stable,
        (0, n_pad),
        constant_values=jnp.iinfo(jnp.int32).max,
    ).reshape(n_blocks, block_size)
    live_blocks = jnp.pad(live, (0, n_pad), constant_values=False).reshape(
        n_blocks, block_size
    )

    # keyword-only-exempt: library-callback=jax.lax.scan
    def step(
        carry: tuple[jax.Array, ...],
        block: tuple[jax.Array, ...],
    ) -> tuple[tuple[jax.Array, ...], None]:
        held_fields = carry[:-1]
        held_stable = carry[-1]
        block_fields = block[:-2]
        block_stable, block_live = block[-2:]
        offered_fields = tuple(
            jnp.where(block_live, field, -jnp.inf) for field in block_fields
        )
        offered_stable = jnp.where(block_live, block_stable, jnp.iinfo(jnp.int32).max)
        index = _argmax(fields=offered_fields, stable=offered_stable)
        candidate_fields = tuple(field[index] for field in offered_fields)
        candidate_stable = offered_stable[index]

        decided = jnp.zeros((), dtype=bool)
        take = jnp.zeros((), dtype=bool)
        for challenger, standing in zip(candidate_fields, held_fields, strict=True):
            take = take | (~decided & (challenger > standing))
            decided = decided | (challenger != standing)
        take = take | (~decided & (candidate_stable < held_stable))
        return (
            *(
                jnp.where(take, challenger, standing)
                for challenger, standing in zip(
                    candidate_fields, held_fields, strict=True
                )
            ),
            jnp.where(take, candidate_stable, held_stable),
        ), None

    initial = (
        *(jnp.asarray(-jnp.inf, dtype=field.dtype) for field in fields),
        jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32),
    )
    result, _ = jax.lax.scan(step, initial, (*field_blocks, stable_blocks, live_blocks))
    return result[-1]


def _case(
    *,
    dtype: jnp.dtype,
    ordering: Ordering,
    depth: int,
    permutation: tuple[int, ...],
) -> tuple[tuple[jax.Array, ...], jax.Array, jax.Array, int]:
    n = len(permutation)
    stable = jnp.asarray(permutation, dtype=jnp.int32)
    base = [
        np.full(n, 9.0),
        np.ones(n),
        np.full(n, 3.0),
        np.full(n, 1.0),
    ]
    winner = n // 2
    if depth == 1:
        base[1] = np.zeros(n)
        base[1][winner] = 1.0
    elif depth < len(base):
        base[depth][winner] += 1.0
    # Certified has no low slope word; ordinary has all four primary fields.
    count = 4 if ordering == "ordinary" else 3
    fields = tuple(
        jnp.asarray(field[list(permutation)], dtype=dtype) for field in base[:count]
    )
    live = jnp.ones(n, dtype=bool)
    candidates = [
        Candidate(
            stable_index=i,
            value=Fraction(int(base[0][i])),
            right_available=bool(base[1][i]),
            slope_high=Fraction(int(base[2][i])),
            slope_low=Fraction(int(base[3][i])),
        )
        for i in range(n)
    ]
    expected = one_shot_winner(candidates=candidates, ordering=ordering)
    assert expected is not None
    return fields, stable, live, expected.stable_index


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("block_size", [1, 2, 3, 7])
def test_jitted_ordinary_scan_matches_one_shot_and_exact_oracle(
    *, dtype_name: str, block_size: int
) -> None:
    """Every tie depth and representative permutation keeps global identity."""
    arrangements = (
        tuple(range(7)),
        tuple(reversed(range(7))),
        (3, 0, 6, 1, 5, 2, 4),
    )
    dtype = _dtype_or_skip(dtype_name)
    solve = jax.jit(
        lambda *args: _blocked(
            fields=args[:-2], stable=args[-2], live=args[-1], block_size=block_size
        )
    )
    for depth in range(5):
        for permutation in arrangements:
            fields, stable, live, expected = _case(
                dtype=dtype,
                ordering="ordinary",
                depth=depth,
                permutation=permutation,
            )
            one = int(_one_shot(fields=fields, stable=stable, live=live))
            blocked = int(solve(*fields, stable, live))
            assert one == expected
            assert blocked == expected


@pytest.mark.parametrize("block_size", [1, 2, 4, 7])
def test_certified_total_order_shape_is_partition_invariant(block_size: int) -> None:
    """The exact path's value/right/slope/index order has the same algebra."""
    arrangements = (
        tuple(range(4)),
        tuple(reversed(range(4))),
        (1, 3, 0, 2),
        (2, 0, 3, 1),
    )
    solve = jax.jit(
        lambda *args: _blocked(
            fields=args[:-2], stable=args[-2], live=args[-1], block_size=block_size
        )
    )
    for depth in range(4):
        for permutation in arrangements:
            fields, stable, live, expected = _case(
                dtype=jnp.float32,
                ordering="certified",
                depth=depth,
                permutation=tuple(permutation),
            )
            assert int(solve(*fields, stable, live)) == expected


def test_empty_and_excluded_blocks_do_not_seed_or_replace_a_winner() -> None:
    fields = (jnp.asarray([5.0, 99.0, 5.0, 99.0], dtype=jnp.float32),)
    stable = jnp.asarray([8, 0, 3, 1], dtype=jnp.int32)
    live = jnp.asarray([False, False, True, False])
    assert int(_blocked(fields=fields, stable=stable, live=live, block_size=2)) == 3


def test_vmapped_scan_preserves_each_members_global_identity() -> None:
    fields = jnp.asarray([[4.0, 4.0, 4.0], [2.0, 3.0, 3.0]], dtype=jnp.float32)
    stable = jnp.asarray([[9, 2, 7], [4, 8, 1]], dtype=jnp.int32)

    # keyword-only-exempt: library-callback=jax.vmap
    def solve(member_fields: jax.Array, member_stable: jax.Array) -> jax.Array:
        return _blocked(
            fields=(member_fields,),
            stable=member_stable,
            live=jnp.ones_like(member_stable, dtype=bool),
            block_size=2,
        )

    got = jax.jit(jax.vmap(solve))(fields, stable)
    np.testing.assert_array_equal(np.asarray(got), np.asarray([2, 1]))
