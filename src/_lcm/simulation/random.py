"""Address simulation innovations by typed name, original row and period."""

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

import jax
import numpy as np
from jax import Array
from jaxtyping import Int, Scalar, UInt32

from _lcm.simulation.agent_groups import AgentRows
from _lcm.simulation.memory import SimulationMemory, run_simulation_operation
from _lcm.simulation.random_address import (
    RandomAddress,
    RowWords,
    Word,
    draw_row_keys,
    draw_window_keys,
    encode_agent_row,
)
from _lcm.typing import PRNGKeyND
from lcm.exceptions import ExecutionPlanningError, InvalidSimulationInputError
from lcm.typing import PRNGKey, UserSeed

_INITIAL_PERIOD = 0


def canonicalize_simulation_seed(
    *, seed: UserSeed | None, name: Literal["seed", "taste_shock_seed"]
) -> int | PRNGKey | None:
    """Refuse malformed random roots before processing inputs or solving.

    The ordinary-stream `seed` must be a Threefry key: draws inside an agent chunk
    are batched, and only Threefry gives every agent the same draws whatever the
    chunk width. The taste stream draws per agent key, so `taste_shock_seed`
    accepts any implementation.
    """
    if seed is None or (
        isinstance(seed, int)
        and not isinstance(seed, bool)
        and (name == "seed" or type(seed) is int)
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


def generate_simulation_keys(
    *,
    key: PRNGKeyND,
    names: list[str],
    n_initial_states: int,
    agent_slice: slice | AgentRows | None = None,
    original_n_agents: int | None = None,
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
    real = n_initial_states if original_n_agents is None else original_n_agents
    if not 0 < real <= n_initial_states <= 2**64:
        raise ExecutionPlanningError("A random population requires valid uint64 rows.")
    encoded_addresses = tuple(
        addresses.get(name, RandomAddress(kind="state", name=name)).words()
        for name in names
    )
    static_arguments = {
        "names": tuple(names),
        "n_initial_states": n_initial_states,
        "original_n_agents": real,
        "partitionable": jax.config.jax_threefry_partitionable,
        "addresses": encoded_addresses,
    }
    if isinstance(agent_slice, AgentRows):
        original_rows = np.asarray(agent_slice.rows)
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
                if name not in ("n_initial_states", "original_n_agents")
            },
            agent_arg_names=("rows",),
            agent_outputs=True,
        )
    agent_slice = slice(0, n_initial_states) if agent_slice is None else agent_slice
    start, width = _validated_chunk_window(
        agent_slice=agent_slice, n_initial_states=n_initial_states
    )
    return key, run_simulation_operation(
        memory=memory,
        function=_generate_windowed_simulation_keys,
        arguments={
            "key": key,
            "start": encode_agent_row(start),
            "period": np.uint32(period),
        },
        static_arguments={**static_arguments, "width": width},
        agent_outputs=True,
    )


def _validated_chunk_window(
    *, agent_slice: slice, n_initial_states: int
) -> tuple[int, int]:
    """Refuse invalid chunk bounds instead of silently clamping or striding."""
    start = 0 if agent_slice.start is None else agent_slice.start
    stop = n_initial_states if agent_slice.stop is None else agent_slice.stop
    if (
        type(start) is not int
        or type(stop) is not int
        or agent_slice.step not in (None, 1)
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
    original_n_agents: int | None,
    partitionable: bool,
    width: int,
    period: int | np.uint32 | Word = _INITIAL_PERIOD,
    addresses: tuple[tuple[int, ...], ...] | None = None,
) -> dict[str, PRNGKeyND]:
    """Construct addressed keys directly for a dynamic original-row window."""
    real = n_initial_states if original_n_agents is None else original_n_agents
    last = encode_agent_row(real - 1)
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
                agent_start=start,
                last_real_agent=last,
                agent_count=width,
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
    """Address grouped agents by their encoded original rows."""
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
