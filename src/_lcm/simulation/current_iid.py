"""Admit addressed scalar-innovation keys for one forward subject chunk."""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from _lcm.current_iid import CurrentIIDDrawPlan
from _lcm.simulation.memory import SimulationMemory, run_simulation_operation
from _lcm.simulation.random_address import (
    RandomAddress,
    draw_window_keys,
    encode_subject_row,
)
from _lcm.typing import PRNGKeyND


def prepare_current_iid_keys(
    *,
    plan: CurrentIIDDrawPlan,
    key: PRNGKeyND,
    period: int,
    subject_slice: slice,
    original_n_subjects: int,
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
                "subject_start": encode_subject_row(row=subject_slice.start),
                "last_real_subject": encode_subject_row(row=original_n_subjects - 1),
                "period": np.uint32(period),
            },
            static_arguments={
                "subject_count": subject_slice.stop - subject_slice.start
            },
            subject_outputs=True,
        )
        if memory is not None:
            memory.hold(tree=keys)
    return MappingProxyType(keys)
