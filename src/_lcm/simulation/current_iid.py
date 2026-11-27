"""Admit addressed scalar-innovation keys for one forward agent chunk."""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from _lcm.current_iid import CurrentIIDDrawPlan
from _lcm.simulation.memory import SimulationMemory, run_simulation_operation
from _lcm.simulation.random_address import (
    RandomAddress,
    draw_window_keys,
    encode_agent_row,
)
from _lcm.typing import PRNGKeyND


def prepare_current_iid_keys(
    *,
    plan: CurrentIIDDrawPlan,
    key: PRNGKeyND,
    period: int,
    agent_slice: slice,
    original_n_agents: int,
    memory: SimulationMemory | None,
) -> Mapping[str, PRNGKeyND]:
    """Retain row/period/name addresses without materializing physical draws."""
    keys = {}
    for name in plan.physical_names:
        keys[name] = run_simulation_operation(
            memory=memory,
            function=draw_window_keys,
            arguments={
                "key": key,
                "address_words": np.asarray(
                    RandomAddress(kind="shock", name=name).words(), dtype=np.uint32
                ),
                "agent_start": encode_agent_row(row=agent_slice.start),
                "last_real_agent": encode_agent_row(row=original_n_agents - 1),
                "period": np.uint32(period),
            },
            static_arguments={"agent_count": agent_slice.stop - agent_slice.start},
            agent_outputs=True,
        )
        if memory is not None:
            memory.hold(tree=keys)
    return MappingProxyType(keys)
