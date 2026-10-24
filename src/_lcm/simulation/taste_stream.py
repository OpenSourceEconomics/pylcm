"""Address destination taste draws by original subject row and period."""

from collections.abc import Mapping
from types import MappingProxyType

import jax
import numpy as np
from jaxtyping import Key, UInt32

from _lcm.simulation.memory import SimulationMemory, run_simulation_operation
from _lcm.simulation.random import _create_simulation_key
from _lcm.simulation.random_address import (
    RandomAddress,
    draw_window_keys,
)
from _lcm.time import TimeAxis
from _lcm.typing import ArrayTree, PRNGKeyND
from lcm.exceptions import ExecutionPlanningError
from lcm.grids import DiscreteGrid
from lcm.typing import ActionName, PRNGKey, RegimeName

type _AddressWords = UInt32[jax.Array | np.ndarray, "8"]
type _RowWords = UInt32[jax.Array | np.ndarray, "2"]
type _Word = UInt32[jax.Array, ""]
type _ScalarKey = Key[jax.Array, ""]

_WORD_BASE = 2**32
_INITIAL_PERIOD = 0


def create_taste_shock_key(
    *,
    seed: int | PRNGKey | None,
    memory: SimulationMemory | None,
    live_inputs: ArrayTree = (),
) -> PRNGKeyND | None:
    """Create an optional typed root, counting its caller's live inputs first."""
    if seed is None:
        return None
    if memory is not None:
        memory.hold(tree=live_inputs)
    if isinstance(seed, jax.Array):
        if memory is not None:
            memory.hold(tree=seed)
        return seed
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
    period: int = 0,
) -> tuple[PRNGKeyND, PRNGKeyND]:
    """Address destination taste draws without advancing the ordinary root."""
    if taste_address is None:
        raise ExecutionPlanningError(
            "Independent taste stream lacks its domain address."
        )
    real_subjects = n_subjects if original_n_subjects is None else original_n_subjects
    taste_keys = generate_taste_shock_keys(
        key=key if taste_key is None else taste_key,
        address_words=taste_address,
        subject_slice=subject_slice,
        original_n_subjects=real_subjects,
        memory=memory,
        period=period,
    )
    return key, taste_keys


def generate_taste_shock_keys(
    *,
    key: PRNGKeyND,
    address_words: tuple[int, ...],
    subject_slice: slice,
    original_n_subjects: int,
    memory: SimulationMemory | None,
    period: int = 0,
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
            "period": np.uint32(period),
        },
        static_arguments={"subject_count": subject_slice.stop - start},
        subject_outputs=True,
    )


def advance_simulation_taste_key(
    *, key: PRNGKeyND, original_n_subjects: int, memory: SimulationMemory | None
) -> PRNGKeyND:
    """Preserve the immutable ordinary root after an independent taste draw."""
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
    ages: TimeAxis,
    discrete_actions_by_regime: Mapping[RegimeName, Mapping[ActionName, DiscreteGrid]],
) -> MappingProxyType[tuple[int, RegimeName], tuple[int, ...]]:
    """Name each destination's taste stream independently of mutable action grids."""
    return MappingProxyType(
        {
            (period, regime): RandomAddress(kind="taste", name=regime).words()
            for period in range(ages.n_periods)
            for regime in discrete_actions_by_regime
        }
    )


def _encode_subject_row(*, row: int) -> _RowWords:
    """Represent a valid global row without narrowing its high bits."""
    if type(row) is not int or not 0 <= row < 2**64:
        raise ValueError(
            f"A taste-key row must fit an unsigned 64-bit integer: {row!r}."
        )
    return np.asarray((row // _WORD_BASE, row % _WORD_BASE), dtype=np.uint32)


def _advance_simulation_taste_key(
    *,
    key: PRNGKeyND,
    original_n_subjects: int,  # noqa: ARG001 - compatibility adapter signature
    partitionable: bool,  # noqa: ARG001 - compatibility adapter signature
) -> PRNGKeyND:
    """Return the immutable ordinary root after an independent taste draw."""
    return key


def draw_taste_shock_keys(
    *,
    key: _ScalarKey,
    address_words: _AddressWords,
    subject_start: _RowWords,
    last_real_subject: _RowWords,
    subject_count: int,
    period: int | _Word | np.uint32 = _INITIAL_PERIOD,
) -> PRNGKeyND:
    """Fold dynamic time/domain metadata and uint64 global rows into a key.

    Rows use two uint32 words, high first, including when JAX disables x64.
    Padded rows duplicate the final real subject. Only the current chunk is
    constructed. The caller supplies the caller's typed root; no mutable
    transition stream, regime code or population-wide split enters these keys.

    For a root whose batched folds or draws are not a map over its keys, every
    row's key is folded as a scalar and carries the trailing axis
    `scalar_draw_axis` names, which tells the decision to draw from it as a
    scalar too.
    """
    return draw_window_keys(
        key=key,
        address_words=address_words,
        subject_start=subject_start,
        last_real_subject=last_real_subject,
        subject_count=subject_count,
        period=period,
    )
