"""Address simulation innovations by typed name, original row and period."""

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

import jax
import numpy as np
from jax import Array
from jaxtyping import Int, Scalar, UInt32

from _lcm.simulation.memory import SimulationMemory, run_simulation_operation
from _lcm.simulation.random_address import (
    RandomAddress,
    RowWords,
    Word,
    draw_row_keys,
    draw_window_keys,
    encode_subject_row,
)
from _lcm.simulation.subject_groups import SubjectRows
from _lcm.typing import PRNGKeyND
from lcm.exceptions import ExecutionPlanningError, InvalidSimulationInputError
from lcm.typing import PRNGKey, UserSeed

_INITIAL_PERIOD = 0


def canonicalize_simulation_seed(
    *, seed: UserSeed | None, name: Literal["seed", "taste_shock_seed"]
) -> int | PRNGKey | None:
    """Refuse malformed random roots before processing inputs or solving.

    The ordinary-stream `seed` must be a Threefry key: draws inside a subject chunk
    are batched, and only Threefry gives every subject the same draws whatever the
    chunk width. The taste stream draws per subject key, so `taste_shock_seed`
    accepts any implementation.
    """
    if seed is None or (
        isinstance(seed, int) and (name == "seed" or type(seed) is int)
    ):
        return seed
    if isinstance(seed, jax.Array) and jax.dtypes.issubdtype(
        seed.dtype, jax.dtypes.prng_key
    ):
        if seed.shape != ():
            raise InvalidSimulationInputError(
                f"{name} requires a scalar jax.random.key, got shape {seed.shape}."
            )
        impl = str(jax.random.key_impl(seed))
        if name == "seed" and impl != "threefry2x32":
            raise InvalidSimulationInputError(
                f"seed requires a threefry2x32 key, got a {impl} key. Pass "
                "jax.random.key(..., impl='threefry2x32') or an integer seed."
            )
        return seed
    raise InvalidSimulationInputError(
        f"{name} must be an integer, a scalar typed key from jax.random.key, "
        f"or None; got {seed!r}."
    )


def create_simulation_key(
    *, seed: int | PRNGKey, memory: SimulationMemory | None = None
) -> PRNGKeyND:
    """Create the initial Threefry key under the optional forward workspace budget."""
    if isinstance(seed, jax.Array):
        if memory is not None:
            memory.hold(tree=seed)
        return seed
    return run_simulation_operation(
        memory=memory,
        function=_create_simulation_key,
        # JAX first converts Python seeds to host int64, including in fp32 mode,
        # before staging them. Preserve its wrapping of unsigned high-bit seeds.
        arguments={"seed": seed if memory is None else np.int64(seed)},
        static_arguments={
            "impl": "threefry2x32",
            "seed_offset": jax.config.jax_random_seed_offset,
        },
    )


def _create_simulation_key(
    *, seed: int | Int[Scalar, ""], impl: str, seed_offset: int
) -> PRNGKeyND:
    """Trace the declared PRNG and offset while keeping seed values dynamic.

    JAX exposes the seed offset through public configuration but has no public
    offset context manager. Verify it at eager/trace entry before the public key
    operation reads it. The operation cache binds this offset into a fresh
    callable, so an executable cannot be reused for another declared offset.
    """
    if seed_offset != jax.config.jax_random_seed_offset:
        raise ExecutionPlanningError("Simulation seed offset changed before tracing.")
    return jax.random.key(seed=seed, impl=impl)


def site_simulation_key(
    *,
    key: PRNGKeyND,
    period: int,
    regime_id: int,
    memory: SimulationMemory | None = None,
) -> PRNGKeyND:
    """Derive the key of one `(period, regime)` draw site from the call's root key.

    The site key depends on the root key, the period and the registered regime
    code only, so a subject's draws at a site do not move when the declared
    starts make other pairs visitable or unvisitable.
    """
    return run_simulation_operation(
        memory=memory,
        function=_site_simulation_key,
        arguments={
            "key": key,
            "period": np.uint32(period),
            "regime_id": np.uint32(regime_id),
        },
    )


def _site_simulation_key(
    *,
    key: PRNGKeyND,
    period: np.unsignedinteger | UInt32[Scalar, ""],
    regime_id: np.unsignedinteger | UInt32[Scalar, ""],
) -> PRNGKeyND:
    """Fold the period, then the regime code, into the root key."""
    return jax.random.fold_in(
        key=jax.random.fold_in(key=key, data=period), data=regime_id
    )


def split_simulation_key(
    *, key: PRNGKeyND, memory: SimulationMemory | None = None
) -> tuple[PRNGKeyND, PRNGKeyND, PRNGKeyND]:
    """Admit the period's three random streams before constructing their arrays."""
    return run_simulation_operation(
        memory=memory,
        function=_split_simulation_key,
        arguments={"key": key},
        static_arguments={"partitionable": jax.config.jax_threefry_partitionable},
    )


def _split_simulation_key(
    *, key: PRNGKeyND, partitionable: bool
) -> tuple[PRNGKeyND, PRNGKeyND, PRNGKeyND]:
    """Include the three row extractions in the compiled operation's workspace."""
    with jax.threefry_partitionable(partitionable):
        next_states_key, next_regime_key, next_key = jax.random.split(key=key, num=3)
    return next_states_key, next_regime_key, next_key


def generate_simulation_keys(
    *,
    key: PRNGKeyND,
    names: list[str],
    n_initial_states: int,
    subject_slice: slice | SubjectRows | None = None,
    original_n_subjects: int | None = None,
    memory: SimulationMemory | None = None,
    period: int = 0,
    addresses: Mapping[str, RandomAddress] = MappingProxyType({}),
) -> tuple[PRNGKeyND, dict[str, PRNGKeyND]]:
    """Address each named innovation by original row and computational period.

    Return the root unchanged. Binding names remain the consumer's DAG names;
    `addresses` supplies semantic identities independent of regime qualification.
    Only this chunk's keys are constructed, and padding repeats the last real row.
    """
    if type(period) is not int or not 0 <= period < 2**32:
        raise ExecutionPlanningError(
            "A random period must fit an unsigned 32-bit word."
        )
    real = n_initial_states if original_n_subjects is None else original_n_subjects
    if not 0 < real <= n_initial_states <= 2**64:
        raise ExecutionPlanningError("A random population requires valid uint64 rows.")
    encoded_addresses = tuple(
        addresses.get(name, RandomAddress(kind="state", name=name)).words()
        for name in names
    )
    static_arguments = {
        "names": tuple(names),
        "n_initial_states": n_initial_states,
        "original_n_subjects": real,
        "partitionable": jax.config.jax_threefry_partitionable,
        "addresses": encoded_addresses,
    }
    if isinstance(subject_slice, SubjectRows):
        original_rows = np.asarray(subject_slice.rows)
        if (
            original_rows.ndim != 1
            or original_rows.dtype.kind not in "iu"
            or np.any(original_rows < 0)
            or int(original_rows.max(initial=0)) >= n_initial_states
        ):
            raise ExecutionPlanningError(
                "Grouped random rows must belong to the population."
            )
        rows = np.asarray(original_rows, dtype=np.uint64)
        rows = np.minimum(rows, np.uint64(real - 1))
        words = np.stack((rows >> np.uint64(32), rows), axis=-1).astype(np.uint32)
        return key, run_simulation_operation(
            memory=memory,
            function=_generate_rowed_simulation_keys,
            arguments={"key": key, "rows": words, "period": np.uint32(period)},
            static_arguments={
                name: value
                for name, value in static_arguments.items()
                if name not in ("n_initial_states", "original_n_subjects")
            },
            subject_arg_names=("rows",),
            subject_outputs=True,
        )
    subject_slice = (
        slice(0, n_initial_states) if subject_slice is None else subject_slice
    )
    start, width = _validated_chunk_window(
        subject_slice=subject_slice, n_initial_states=n_initial_states
    )
    return key, run_simulation_operation(
        memory=memory,
        function=_generate_windowed_simulation_keys,
        arguments={
            "key": key,
            "start": encode_subject_row(start),
            "period": np.uint32(period),
        },
        static_arguments={**static_arguments, "width": width},
        subject_outputs=True,
    )


def _validated_chunk_window(
    *, subject_slice: slice, n_initial_states: int
) -> tuple[int, int]:
    """Refuse invalid chunk bounds instead of silently clamping or striding."""
    start = 0 if subject_slice.start is None else subject_slice.start
    stop = n_initial_states if subject_slice.stop is None else subject_slice.stop
    if (
        type(start) is not int
        or type(stop) is not int
        or subject_slice.step not in (None, 1)
        or not 0 <= start < stop <= n_initial_states <= 2**64
    ):
        raise ExecutionPlanningError(
            "A budgeted RNG chunk window needs in-range increasing bounds "
            "and unit step."
        )
    return start, stop - start


def _generate_windowed_simulation_keys(
    *,
    key: PRNGKeyND,
    start: RowWords,
    names: tuple[str, ...],
    n_initial_states: int,
    original_n_subjects: int | None,
    partitionable: bool,
    width: int,
    period: int | np.uint32 | Word = _INITIAL_PERIOD,
    addresses: tuple[tuple[int, ...], ...] | None = None,
) -> dict[str, PRNGKeyND]:
    """Construct addressed keys directly for a dynamic original-row window."""
    real = n_initial_states if original_n_subjects is None else original_n_subjects
    last = encode_subject_row(real - 1)
    with jax.threefry_partitionable(partitionable):
        return {
            f"key_{name}": draw_window_keys(
                key=key,
                address_words=np.asarray(
                    RandomAddress(kind="state", name=name).words()
                    if addresses is None
                    else addresses[index],
                    dtype=np.uint32,
                ),
                subject_start=start,
                last_real_subject=last,
                subject_count=width,
                period=period,
            )
            for index, name in enumerate(names)
        }


def _generate_rowed_simulation_keys(
    *,
    key: PRNGKeyND,
    rows: UInt32[Array | np.ndarray, "rows 2"],
    names: tuple[str, ...],
    partitionable: bool,
    period: int | np.uint32 | Word = _INITIAL_PERIOD,
    addresses: tuple[tuple[int, ...], ...] | None = None,
) -> dict[str, PRNGKeyND]:
    """Address grouped subjects by their encoded original rows."""
    with jax.threefry_partitionable(partitionable):
        return {
            f"key_{name}": draw_row_keys(
                key=key,
                address_words=np.asarray(
                    RandomAddress(kind="state", name=name).words()
                    if addresses is None
                    else addresses[index],
                    dtype=np.uint32,
                ),
                row_high=rows[:, 0],
                row_low=rows[:, 1],
                period=period,
            )
            for index, name in enumerate(names)
        }


def draw_random_seed() -> int:
    """Generate a random seed using the operating system's secure entropy pool."""
    return int.from_bytes(os.urandom(4), "little")
