"""Versioned random addresses and half-open categorical sampling."""

import functools
import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.dtypes import ExtendedDType
from jaxtyping import Key, UInt32

from _lcm.dtypes import canonical_float_dtype
from lcm.typing import Float1D, ScalarFloat, ScalarInt

RANDOM_ADDRESS_VERSION = "pylcm.random.v1"

type RandomKind = Literal["shock", "state", "regime", "taste"]
type AddressWords = UInt32[jax.Array | np.ndarray, "8"]
type RowWords = UInt32[jax.Array | np.ndarray, "2"]
type Word = UInt32[jax.Array, ""]
type ScalarKey = Key[jax.Array, ""]
type KeyBank = Key[jax.Array, "..."]

_WORD_BASE = 2**32


@dataclass(frozen=True, kw_only=True)
class RandomAddress:
    """Identify one scalar innovation independently of its containing group."""

    kind: RandomKind
    """Disjoint domain for state, regime, shock and taste innovations."""
    name: str
    """Immutable scalar name or reserved stream name."""
    component: int = 0
    """Stable coordinate within a named vector law."""

    def __post_init__(self) -> None:
        if self.kind not in ("shock", "state", "regime", "taste"):
            raise ValueError(f"Unknown random address domain: {self.kind!r}.")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A random address requires a nonempty name.")
        if type(self.component) is not int or self.component < 0:
            raise ValueError("A random component must be a nonnegative integer.")

    def words(self) -> tuple[int, ...]:
        """Hash compact UTF-8 JSON into big-endian uint32 words."""
        payload = (RANDOM_ADDRESS_VERSION, self.kind, self.name, self.component)
        encoded = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        return struct.unpack(">8I", hashlib.sha256(encoded).digest())


def encode_subject_row(row: int) -> RowWords:
    """Encode an original row without narrowing its high bits in float32 mode."""
    if type(row) is not int or not 0 <= row < 2**64:
        raise ValueError(f"A random row must fit an unsigned 64-bit integer: {row!r}.")
    return np.asarray((row // _WORD_BASE, row % _WORD_BASE), dtype=np.uint32)


def draw_window_keys(
    *,
    key: ScalarKey,
    address_words: AddressWords,
    subject_start: RowWords,
    last_real_subject: RowWords,
    subject_count: int,
    period: int | Word | np.uint32,
) -> KeyBank:
    """Address one chunk, duplicating the last real row in its padding."""
    if type(subject_count) is not int or not 0 <= subject_count < 2**64:
        raise ValueError("A random chunk requires a nonnegative uint64 row count.")
    start, last = jnp.asarray(subject_start), jnp.asarray(last_real_subject)
    offset_high, offset_low = _row_offset_words(count=subject_count)
    low = start[1] + offset_low
    carry = (low < start[1]).astype(jnp.uint32)
    high_before_carry = start[0] + offset_high
    overflow = (high_before_carry < start[0]) | (
        (high_before_carry == np.uint32(_WORD_BASE - 1)) & (carry != 0)
    )
    high = high_before_carry + carry
    padded = overflow | (high > last[0]) | ((high == last[0]) & (low > last[1]))
    return draw_row_keys(
        key=key,
        address_words=address_words,
        row_high=jnp.where(padded, last[0], high),
        row_low=jnp.where(padded, last[1], low),
        period=period,
    )


def draw_row_keys(
    *,
    key: ScalarKey,
    address_words: AddressWords,
    row_high: UInt32[jax.Array | np.ndarray, " rows"],
    row_low: UInt32[jax.Array | np.ndarray, " rows"],
    period: int | Word | np.uint32,
) -> KeyBank:
    """Fold address words, original row high/low words, then period."""
    addressed_key = key
    for position in range(8):
        addressed_key = jax.random.fold_in(addressed_key, address_words[position])
    draw_axis = scalar_draw_axis(dtype=addressed_key.dtype)
    fold = (
        jax.custom_batching.sequential_vmap(_fold_subject_key)
        if draw_axis
        else _fold_subject_key
    )
    keys = jax.vmap(fold, in_axes=(None, 0, 0, None))(
        addressed_key,
        jnp.asarray(row_high),
        jnp.asarray(row_low),
        jnp.asarray(period, dtype=jnp.uint32),
    )
    return keys.reshape(*keys.shape, *draw_axis)


def scalar_draw_axis(*, dtype: np.dtype | ExtendedDType) -> tuple[int, ...]:
    """Mark roots whose batched draws require scalar dispatch per subject."""
    return () if dtype == _threefry_key_dtype() else (1,)


def inverse_cdf_index(*, probabilities: Float1D, uniform: ScalarFloat) -> ScalarInt:
    """Select a half-open cell from validated probabilities, preserving zero masses."""
    indices = jnp.arange(probabilities.shape[0])
    last_positive = jnp.max(jnp.where(probabilities > 0.0, indices, 0))
    cumulative = jnp.where(indices >= last_positive, 1.0, jnp.cumsum(probabilities))
    return jnp.sum(uniform >= cumulative, dtype=jnp.int32)


def draw_categorical_index(*, key: ScalarKey, probabilities: Float1D) -> ScalarInt:
    """Draw a half-open categorical cell in the active floating precision."""
    return inverse_cdf_index(
        probabilities=probabilities,
        uniform=jax.random.uniform(key=key, dtype=canonical_float_dtype()),
    )


@functools.cache
def _threefry_key_dtype() -> ExtendedDType:
    # An abstract seed keeps this a shape query; a literal one builds a key.
    return jax.eval_shape(
        lambda seed: jax.random.key(seed=seed, impl="threefry2x32"),
        jax.ShapeDtypeStruct((), jnp.uint32),
    ).dtype


def _row_offset_words(
    *, count: int
) -> tuple[UInt32[jax.Array, " rows"], UInt32[jax.Array, " rows"]]:
    """Represent chunk offsets without relying on an enabled uint64 dtype."""
    high_blocks, low_blocks = [], []
    for start in range(0, count, _WORD_BASE):
        size = min(_WORD_BASE, count - start)
        low = jnp.arange(size, dtype=jnp.uint32)
        high_blocks.append(jnp.full_like(low, start // _WORD_BASE))
        low_blocks.append(low)
    if not low_blocks:
        empty = jnp.empty((0,), dtype=jnp.uint32)
        return empty, empty
    if len(low_blocks) == 1:
        return high_blocks[0], low_blocks[0]
    return jnp.concatenate(high_blocks), jnp.concatenate(low_blocks)


# keyword-only-exempt: library-callback=jax.vmap
def _fold_subject_key(
    key: ScalarKey, high: Word, low: Word, period: int | Word | np.uint32
) -> ScalarKey:
    """Apply the versioned fixed-order original-row and period encoding."""
    return jax.random.fold_in(
        jax.random.fold_in(jax.random.fold_in(key, high), low), period
    )
