"""Address common taste-shock keys by exact age, action domain and subject row."""

import hashlib
import json
import struct
from collections.abc import Mapping
from fractions import Fraction
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Key, UInt32

from _lcm.simulation.memory import SimulationMemory, run_simulation_operation
from _lcm.simulation.random import _create_simulation_key, generate_simulation_keys
from _lcm.typing import PRNGKeyND
from lcm.ages import AgeGrid
from lcm.exceptions import ExecutionPlanningError
from lcm.grids import DiscreteGrid
from lcm.typing import ActionName, RegimeName, UserAge

type _DiscreteDomain = tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
type _AddressWords = UInt32[jax.Array | np.ndarray, "8"]
type _RowWords = UInt32[jax.Array | np.ndarray, "2"]
type _Word = UInt32[jax.Array, ""]
type _ScalarKey = Key[jax.Array, ""]

_WORD_BASE = 2**32


def create_taste_shock_key(
    *, seed: int | None, memory: SimulationMemory | None, live_inputs: object = ()
) -> PRNGKeyND | None:
    """Create an optional Threefry root, counting its caller's live inputs first."""
    if seed is None:
        return None
    if memory is not None:
        memory.hold(tree=live_inputs)
    return run_simulation_operation(
        memory=memory,
        function=_create_simulation_key,
        arguments={"seed": seed if memory is None else np.int64(seed)},
        static_arguments={
            "impl": "threefry2x32",
            "seed_offset": jax.config.jax_random_seed_offset,
        },
    )


def prepare_decision_taste_keys(
    *,
    key: PRNGKeyND,
    taste_key: PRNGKeyND | None,
    taste_address: tuple[int, ...] | None,
    n_subjects: int,
    subject_slice: slice,
    original_n_subjects: int | None,
    memory: SimulationMemory | None,
) -> tuple[PRNGKeyND, PRNGKeyND]:
    """Select ordinary or addressed taste keys while preserving ordinary advancement."""
    if taste_key is None:
        key, keys = generate_simulation_keys(
            key=key,
            names=["taste_shock"],
            n_initial_states=n_subjects,
            subject_slice=subject_slice,
            original_n_subjects=original_n_subjects,
            memory=memory,
        )
        return key, keys["key_taste_shock"]
    if taste_address is None:
        raise ExecutionPlanningError(
            "Independent taste stream lacks its domain address."
        )
    real_subjects = n_subjects if original_n_subjects is None else original_n_subjects
    next_key = advance_simulation_taste_key(
        key=key, original_n_subjects=real_subjects, memory=memory
    )
    taste_keys = generate_taste_shock_keys(
        key=taste_key,
        address_words=taste_address,
        subject_slice=subject_slice,
        original_n_subjects=real_subjects,
        memory=memory,
    )
    return next_key, taste_keys


def generate_taste_shock_keys(
    *,
    key: PRNGKeyND,
    address_words: tuple[int, ...],
    subject_slice: slice,
    original_n_subjects: int,
    memory: SimulationMemory | None,
) -> PRNGKeyND:
    """Admit only this chunk's addressed keys, with dynamic age and row words."""
    start = subject_slice.start
    return run_simulation_operation(
        memory=memory,
        function=draw_taste_shock_keys,
        arguments={
            "key": key,
            "address_words": np.asarray(address_words, dtype=np.uint32),
            "subject_start": _encode_subject_row(row=start),
            "last_real_subject": _encode_subject_row(row=original_n_subjects - 1),
        },
        static_arguments={"subject_count": subject_slice.stop - start},
        subject_outputs=True,
    )


def advance_simulation_taste_key(
    *, key: PRNGKeyND, original_n_subjects: int, memory: SimulationMemory | None
) -> PRNGKeyND:
    """Preserve the ordinary carry when an independent taste draw replaces its bank.

    The numerical split still has its original population size. Its compiler
    scratch is measured as-is; only the unused taste-key output bank is omitted.
    """
    return run_simulation_operation(
        memory=memory,
        function=_advance_simulation_taste_key,
        arguments={"key": key},
        static_arguments={
            "original_n_subjects": original_n_subjects,
            "partitionable": jax.config.jax_threefry_partitionable,
        },
    )


def build_taste_stream_addresses(
    *,
    ages: AgeGrid,
    discrete_actions_by_regime: Mapping[RegimeName, Mapping[ActionName, DiscreteGrid]],
) -> MappingProxyType[tuple[int, RegimeName], tuple[int, ...]]:
    """Describe matching ages and ordered choices without regime identity.

    The caller supplies the actual discrete product-axis order. Labels and codes
    describe each axis in that same order; continuous grids do not enter this
    address. Regime names only locate the returned metadata. Equal exact ages
    and domains share a stream even across different regimes and horizons.
    """
    domains: dict[str, _DiscreteDomain] = {
        regime: tuple(
            (name, tuple(zip(grid.categories, grid.codes, strict=True)))
            for name, grid in actions.items()
        )
        for regime, actions in discrete_actions_by_regime.items()
    }
    encoded: dict[tuple[UserAge, _DiscreteDomain], tuple[int, ...]] = {}
    addresses: dict[tuple[int, RegimeName], tuple[int, ...]] = {}
    for period, age in enumerate(ages.exact_values):
        for regime, domain in domains.items():
            identity = (age, domain)
            if identity not in encoded:
                encoded[identity] = _taste_address_words(age=age, domain=domain)
            addresses[(period, regime)] = encoded[identity]
    return MappingProxyType(addresses)


def _encode_subject_row(*, row: int) -> _RowWords:
    """Represent a valid global row without narrowing its high bits."""
    if type(row) is not int or not 0 <= row < 2**64:
        raise ValueError(
            f"A taste-key row must fit an unsigned 64-bit integer: {row!r}."
        )
    return np.asarray((row // _WORD_BASE, row % _WORD_BASE), dtype=np.uint32)


def _advance_simulation_taste_key(
    *, key: PRNGKeyND, original_n_subjects: int, partitionable: bool
) -> PRNGKeyND:
    """Compute exactly the carry from the existing population-wide split."""
    with jax.threefry_partitionable(partitionable):
        return jax.random.split(key=key, num=original_n_subjects + 1)[0]


def _taste_address_words(*, age: UserAge, domain: _DiscreteDomain) -> tuple[int, ...]:
    """Hash an unambiguous versioned UTF-8 representation into ordered words."""
    rational_age = Fraction(age)
    payload = (
        "pylcm.taste-stream.v1",
        (rational_age.numerator, rational_age.denominator),
        domain,
    )
    serialized = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return struct.unpack(">8I", hashlib.sha256(serialized).digest())


def draw_taste_shock_keys(
    *,
    key: _ScalarKey,
    address_words: _AddressWords,
    subject_start: _RowWords,
    last_real_subject: _RowWords,
    subject_count: int,
) -> PRNGKeyND:
    """Fold dynamic time/domain metadata and uint64 global rows into a key.

    Rows use two uint32 words, high first, including when JAX disables x64.
    Padded rows duplicate the final real subject. Only the current chunk is
    constructed. The caller supplies an explicitly Threefry root; no mutable
    transition stream, regime code or population-wide split enters these keys.
    """
    if type(subject_count) is not int or not 0 <= subject_count < 2**64:
        raise ValueError("A taste-key chunk requires a nonnegative uint64 row count.")
    address = jnp.asarray(address_words)
    start = jnp.asarray(subject_start)
    last = jnp.asarray(last_real_subject)
    addressed_key = key
    for position in range(8):
        addressed_key = jax.random.fold_in(addressed_key, address[position])

    offset_high, offset_low = _row_offset_words(count=subject_count)
    low = start[1] + offset_low
    carry = (low < start[1]).astype(jnp.uint32)
    high_before_carry = start[0] + offset_high
    overflow = (high_before_carry < start[0]) | (
        (high_before_carry == np.uint32(_WORD_BASE - 1)) & (carry != 0)
    )
    high = high_before_carry + carry
    padded = overflow | (high > last[0]) | ((high == last[0]) & (low > last[1]))
    row_high = jnp.where(padded, last[0], high)
    row_low = jnp.where(padded, last[1], low)
    return jax.vmap(_fold_subject_key, in_axes=(None, 0, 0))(
        addressed_key, row_high, row_low
    )


def _row_offset_words(*, count: int) -> tuple[jax.Array, jax.Array]:
    """Represent chunk offsets without relying on an enabled uint64 dtype."""
    high_blocks: list[jax.Array] = []
    low_blocks: list[jax.Array] = []
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
def _fold_subject_key(key: _ScalarKey, high: _Word, low: _Word) -> _ScalarKey:
    """Address one subject without shape-dependent random splitting."""
    return jax.random.fold_in(jax.random.fold_in(key, high), low)
