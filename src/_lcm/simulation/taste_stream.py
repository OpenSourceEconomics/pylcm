"""Address destination taste draws by original agent row and period."""

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
    encode_agent_row,
)
from _lcm.time import TimeAxis, step_count
from _lcm.typing import ArrayTree, PRNGKeyND
from lcm.exceptions import ExecutionPlanningError
from lcm.grids import DiscreteGrid
from lcm.typing import ActionName, PRNGKey, RegimeName

type _AddressWords = UInt32[jax.Array | np.ndarray, "8"]
type _RowWords = UInt32[jax.Array | np.ndarray, "2"]
type _Word = UInt32[jax.Array, ""]
type _ScalarKey = Key[jax.Array, ""]

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
    n_agents: int,
    agent_slice: slice,
    original_n_agents: int | None,
    memory: SimulationMemory | None,
    period: int = 0,
) -> tuple[PRNGKeyND, PRNGKeyND]:
    """Address destination taste draws without advancing the ordinary root."""
    if taste_address is None:
        raise ExecutionPlanningError(
            "Independent taste stream lacks its domain address."
        )
    real_agents = n_agents if original_n_agents is None else original_n_agents
    taste_keys = generate_taste_shock_keys(
        key=key if taste_key is None else taste_key,
        address_words=taste_address,
        agent_slice=agent_slice,
        original_n_agents=real_agents,
        memory=memory,
        period=period,
    )
    return key, taste_keys


def generate_taste_shock_keys(
    *,
    key: PRNGKeyND,
    address_words: tuple[int, ...],
    agent_slice: slice,
    original_n_agents: int,
    memory: SimulationMemory | None,
    period: int = 0,
) -> PRNGKeyND:
    """Admit only this chunk's addressed keys, with dynamic age and row words."""
    start = agent_slice.start
    return run_simulation_operation(
        memory=memory,
        function=draw_taste_shock_keys,
        arguments={
            "key": key,
            "address_words": np.asarray(address_words, dtype=np.uint32),
            "agent_start": encode_agent_row(row=start),
            "last_real_agent": encode_agent_row(row=original_n_agents - 1),
            "period": np.uint32(period),
        },
        static_arguments={"agent_count": agent_slice.stop - start},
        agent_outputs=True,
    )


def build_taste_stream_addresses(
    *,
    ages: TimeAxis,
    discrete_actions_by_regime: Mapping[RegimeName, Mapping[ActionName, DiscreteGrid]],
) -> MappingProxyType[tuple[int, RegimeName], tuple[int, ...]]:
    """Name each destination's taste stream independently of mutable action grids."""
    return MappingProxyType(
        {
            (step, regime): RandomAddress(kind="taste", name=regime).words()
            for step in range(step_count(ages))
            for regime in discrete_actions_by_regime
        }
    )


def draw_taste_shock_keys(
    *,
    key: _ScalarKey,
    address_words: _AddressWords,
    agent_start: _RowWords,
    last_real_agent: _RowWords,
    agent_count: int,
    period: int | _Word | np.uint32 = _INITIAL_PERIOD,
) -> PRNGKeyND:
    """Fold dynamic time/domain metadata and uint64 global rows into a key.

    Rows use two uint32 words, high first, including when JAX disables x64.
    Padded rows duplicate the final real agent. Only the current chunk is
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
        agent_start=agent_start,
        last_real_agent=last_real_agent,
        agent_count=agent_count,
        period=period,
    )
